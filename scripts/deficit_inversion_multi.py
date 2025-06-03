# %%
import pygmo as pg
import numpy as np
import pandas as pd
import os
from scipy.sparse import bsr_array
from time import time

"""
Script for running multiple subsets of full rupture catalgoue
"""
bn_dict = {1: [0.95, 16.5],
           2: [1.1, 21.5],
           3: [1.24, 27.9]}

start = time()
# %% Define Parameters
# Naming and inputs
procdir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerm\\output"
inversion_name = 'hk_lock_nrupt'  # Name of directory results will be stored in
deficit_file = "hk_lock.slip"  # Name of the file containing the target slip rate deficit (must be same patch geometry as the rupture sets)
rigidity_file = "wuatom.mu"
rupture_file = "hikkerm_lock_wuatom_NSHMarea_df_n50000.csv"  # Name of the file containing the rupture slips (must be same patch geometry as the slip deficits, assumes ruptures stored in random Mw order)
n_ruptures = 5000  # Number of ruptures to use in each island
patch_area = 5e3 ** 2 # Area of each patch (m^2) - used to calculate moment rate from slip rate

bn_combo = 2
b, N = bn_dict[bn_combo]  # B and N values to use for the GR relation (N value used only for linear GR rates and assigning upper limits)
tapered_gr = True # Use tapered GR relation of Rollins and Avouac (2019)
taper_max_Mw = 9.5  # Maximum magnitude to use in tapered GR relation
alpha_s = 1 # Portion of moment taken up by coseismic slip for tapered GR relation
max_patch = 6233 # ID of the patch at which to stop (exclusive) using when running the inversion (-1 for all patches. Assumes rupture patches start at 0, deficit patches start at 1)

# Weighting
rate_weight = 10  # Absolute misfit of slip deficit (int)
norm_weight = 1  # Relative misfit of slip deficit (int)
GR_weight = 500 # Mistfit of GR relation (int)
nrupt_weight = 1000  # Weighting for number of ruptures misfit (int)
nrupt_cuttoff = -16  # Cutoff for recurrence interval when counting number of ruptures (int, in log10 space, so -16 = 1e-16)

# Pygmo requirements
n_iterations = 500000  # Maximum number of iterations for each inversion
ftol = 0.0001  # Stopping Criteria
n_islands = 10  # Number of islands
pop_size = 20  # Number of populations per island
archipeligo = False  # True - Consider all islands as part of an archipeligo, using same rupture set. False: Run islands individually, with different rupture sets
topology_name = 'None'  # 'None', 'Ring', 'FullyConnected'
ring_plus = 1  # Number of connections to add to ring topology
define_population = False  # Predefine the starting population, either randomly or from a starting rate file (If defined, all Islands will have the same start)
starting_rate_file = None # Set to None for random initialisation

# %% No more user inputs below here
if 'uc03610' in os.getcwd():
    procdir = '/'.join(["/nesi/nobackup/uc03610/jack/fakequakes", procdir.split('\\')[-2], procdir.split('\\')[-1]])
elif 'mnt' in os.getcwd().split(os.sep)[:2]:
    root = procdir.split(':')[0].lower() + os.sep
    procdir = '/mnt/' + root + (os.sep).join(procdir.split('\\')[1:])
    
model_info_dir = f"{procdir}/../data\\model_info"
deficit_file = os.path.join(model_info_dir, deficit_file)
rigidity_file = os.path.abspath(os.path.join(model_info_dir, '..', '..', 'structure', rigidity_file))

outdir = os.path.abspath(os.path.join(procdir, inversion_name))
if not os.path.exists(outdir):
    os.mkdir(outdir)

# %% Pygmo Classes and Functions
class deficitInversion:
    def __init__(self, ruptures_df: pd.DataFrame, deficit: np.ndarray, b: float, N: float, rate_weight: float, norm_weight: float, GR_weight: float, nrupt_weight: float,
                 mu_A: np.ndarray, tapered_gr: bool = True, taper_max_Mw: float = 9.5, alpha_s: float = 1, min_Mw: float = 6, max_Mw: float = 10, max_patch: int = -1,
                 nrupt_cuttoff: int = -16):

        self.name = "Slip Deficit Inversion"
        self.deficit = deficit  # Slip deficit of ROI (on same grid as ruptures)
        self.n_patches = deficit.shape[0]  # Number of patches
        self.b = b  # b-value for GR-rate
        self.N5 = N  # N value for Mw 5 events
        self.a = np.log10(self.N5) + (b * 5)  # a-value for GR-rate calculated from b and the N-value for Mw 5 events
        self.rate_weight = rate_weight  # Weighting for rate misfit over GR-rate misfit
        self.norm_weight = norm_weight  # Weighting of the normalised GR-rate misfit
        self.GR_weight = GR_weight  # Weighting for GR-rate misfit
        self.nrupt_weight = nrupt_weight  # Weighting for number of ruptures misfit
        self.nrupt_cuttoff = nrupt_cuttoff  # Cutoff for recurrence interval when counting number of ruptures
        self.min_Mw = min_Mw  # Minimum magnitude to use for GR-rate
        self.max_Mw = max_Mw  # Maximum magnitude to use for GR-rate
        self.tapered_gr = tapered_gr  # Use tapered GR-rate relation
        self.mu_A = mu_A[:max_patch] # uA contribution of Mo = uAS - used to calculate tapered GR-rate
        self.alpha_s = alpha_s  # Portion of moment taken up by coseismic slip for tapered GR relation

        print("Reading rupture dataframe...")
        self.n_ruptures = ruptures_df.shape[0]
        self.id = ruptures_df['rupt_id'].values
        self.Mw = ruptures_df['mw'].values
        self.target = ruptures_df['target_mw'].values
        i0, i1 = ruptures_df.columns.get_loc('0'), ruptures_df.columns.get_loc(str(self.n_patches - 1)) + 1
        self.Mw_bins = np.unique(np.floor(np.array(self.Mw) * 100) / 100)  # Create bins for each 0.01 magnitude increment
        self.Mo_bins = self.mw2mo(self.Mw_bins)  # Convert Mw bins to Mo bins (in N/m^2)
        self.Mo_max = self.mw2mo(taper_max_Mw)  # Maximum moment for tapered GR-rate

        self.sparse_slip = bsr_array(ruptures_df.iloc[:, i0:i1].values.T * 1000)  # Slip in mm, convert from m to mm, place in sparse matrix
        self.max_patch = max_patch

        if max_patch != -1:
            self.extra_mu_A = mu_A[max_patch:] # uA contribution for patches outside the ROI - used to calculate tapered GR-rate
            i2 = ruptures_df.columns.get_loc(str(max_patch))
            self.extra_slip = bsr_array(ruptures_df.iloc[:, i2:].values.T * 1000)

        self.make_gr_matrix()
        self.GR_rate = self.mfd(tapered_gr, self.Mw_bins, self.deficit)  # Calculate target MFD

    # Properties to call matricies as full arrays
    @property
    def slip(self):
        return self.sparse_slip.toarray()

    @property
    def gr_matrix(self):
        return self.sparse_gr_matrix.toarray()

    def mfd(self, tapered_gr: bool, Mw: np.ndarray, slip: np.ndarray = None, extra_mu_A: np.ndarray = None, extra_slip: np.ndarray = None):
        if tapered_gr:
            Mo_rate = np.sum(self.mu_A * (slip * 1e-3))  # Calculate moment using slip rates (convert to m) and rigidity-area (N)
            if extra_mu_A is not None and extra_slip is not None:  # Include the moment contribution from the extra patches
                Mo_rate += np.sum(extra_mu_A * (extra_slip * 1e-3))
            return ((1 - ((2 * self.b) / 3)) / ((2 * self.b) / 3)) * ((self.alpha_s * Mo_rate) / self.Mo_max) * ((self.Mo_max / self.mw2mo(Mw)) ** ((2 * self.b) / 3) - 1)
        else:
            return 10 ** (self.a - (self.b * Mw))

    def get_name(self):
        return self.name

    def make_gr_matrix(self):
        print("Making GR Matrix...")
        gr_matrix = np.zeros((len(self.Mw_bins), self.n_ruptures)).astype('bool')  # Make an I-J matrix for calculating GR rates, where I is number of magnitude bins and J is number of ruptures
        for ix, bin in enumerate(self.Mw_bins):
            gr_matrix[ix, :] = (self.Mw >= bin)
        self.sparse_gr_matrix = bsr_array(gr_matrix.astype('int'))  # Convert GR matrix to sparse matrix

    def get_bounds(self):
        """
        Upper bound is going to be calculated using the same b-value, but assuming that 100 5Mw events occur per year
        Assuming that the ruptures are uniformly distributed across the Mw bins, a rupture rate for each Mw bin can be calculated
        Using linear interpolation, the rupture rate for each rupture is then assigned
        This method *should* be reasonable, as it is less impacted by variations in the actual input magnitude distribution

        Lower bound is 0
        """
        lower_bound = [-16] * self.n_ruptures  # Allow rupture to not occur (simulated as incredibly low rate, use -16 from NSHM)

        samples_Mw = np.linspace(min(self.Mw), max(self.Mw), self.n_ruptures)[::-1]  # Create a range of magnitudes to sample from, in decreasing magnitude order
        a = np.log10(100) + (b * 5)  # Upperbound from 100 5Mw events a year (Currently this will overshoot, as is the bin rate not indivdual rupture rate)
        GR_rate = 10 ** (a - (b * samples_Mw))  # Calculate N value for each rupture magnitude (number of events >= Mw per year)
        sample_rates = np.zeros_like(GR_rate)
        for ix in range(self.n_ruptures):
            sample_rates[ix] = GR_rate[ix] - np.sum(sample_rates[(ix + 1):])  # Calculate initial rate as (N-value - rate of higher magnitude)
        upper_bound = np.interp(self.Mw, samples_Mw[::-1], sample_rates[::-1])  # Interpolate the sample rates to the actual rupture magnitudes
        upper_bound = np.log10(upper_bound)  # Convert to log space

        return (lower_bound, upper_bound)

    def mw2mo(self, Mw):
        return 10 ** ((3 * Mw) / 2 + 9.1)

    def fitness(self, x: np.ndarray):

        rms, norm_rms, GR_rms, GR_lims_rms, n_rupt_rms = 0, 0, 0, 0, 0

        # Calculate slip-deficit component (Based on NSHM SRM constraint weights)
        if any([self.rate_weight > 0, self.norm_weight > 0]):
            total_slip = self.sparse_slip @ (10 ** x)  # Calculate slip by multiplying slip matrix by rupture rate vector (sparse matricies can't use matmul)
            rate_misfit = total_slip - self.deficit
            if self.rate_weight > 0:
                rms = np.sqrt(np.mean(rate_misfit ** 2))  # Calculate root mean square misfit of slip - Penalises large absolute differences
            if self.norm_weight > 0:
                norm_rms = np.sqrt(np.mean((rate_misfit / self.deficit) ** 2))  # Calculate root mean square misfit of normalised slip - penalises large relative misfits

        # Calculate GR-rate component
        if self.GR_weight > 0:
            GR_ix = np.where((self.Mw_bins >= self.min_Mw) & (self.Mw_bins <= self.max_Mw), True, False)  # Only use bins up to the maximum magnitude (so that few high Mw events aren't overweighted)
            GR_lims_ix = np.where((self.Mw_bins >= self.min_Mw) & (self.Mw_bins <= self.max_Mw), False, True)  # Use bins outside min-max magnitude (that that few high Mw events aren't totally unweighted)
            if self.tapered_gr and self.max_patch != -1:
                extra_slip = self.extra_slip @ (10 ** x)  # Calculate the slip on patches outside the ROI (in mm/yr) to calulate extra moment rate
                target_GR_rate = self.mfd(self.tapered_gr, self.Mw_bins, self.deficit, self.extra_mu_A, extra_slip) # Calculate Tapered GR-rate for each magnitude bin based on inverted slip rates and new moment
            else:
                target_GR_rate = self.GR_rate
            inv_GR = self.sparse_gr_matrix @ (10 ** x)  # Calculate GR-rate for each magnitude bin based on inverted slip rates
            GR_rms = np.sqrt(np.mean((np.log10(inv_GR[GR_ix]) - np.log10(target_GR_rate[GR_ix])) ** 2))  # Penalise for deviating from log(N)
            if any(GR_lims_ix):
                GR_lims_rms = np.sqrt(np.mean((np.log10(inv_GR[GR_lims_ix]) - np.log10(target_GR_rate[GR_lims_ix])) ** 2))  # Penalise for deviating from log(N)

        # Minimise number of ruptures
        if self.nrupt_weight > 0:
            n_rupt = np.sum(x > self.nrupt_cuttoff)
            n_rupt_rms = n_rupt / self.n_ruptures  # Normalise number of ruptures to the total number of ruptures

        cum_rms = (rms * self.rate_weight) + (norm_rms * self.norm_weight) + (GR_rms * self.GR_weight) + (GR_lims_rms) + (n_rupt_rms * self.nrupt_weight)  # Allow for variable weighting between slip deficit and GR-rate
        print(f"{cum_rms:.0f} {rms * self.rate_weight:.2f} {norm_rms * self.norm_weight:.2f} {GR_rms * self.GR_weight:.2f} {n_rupt_rms:.2f}:{n_rupt_rms * self.nrupt_weight:.2f} {self.nrupt_weight:.0f}")
        return np.array([cum_rms])  


def build_topology(topology_name: str = "UnConnected", n_islands: int = 1, ring_plus: int = 0):
        if topology_name == 'FullyConnected':
            print("\t\tTopo: Fully Connected")
            topo = pg.topology(pg.fully_connected(n_islands))
        elif topology_name == 'Ring':
            if ring_plus == 0:
                print("\t\tTopo: Ring")
                topo = pg.topology(pg.ring(n_islands))
            else:
                print(f"\t\tTopo: Ring+{ring_plus}")
                connections = ring_plus + 1
                topo = pg.free_form()
                for ii in range(n_islands):
                    topo.add_vertex()
                for ii in range(n_islands):
                    conns = ii - np.arange(-connections, connections + 1)
                    conns = np.delete(conns, connections)
                    conns = np.mod(conns, n_islands)
                    for conn in conns:
                        topo.add_edge(ii, conn)
                topo = pg.topology(topo)
        else:
            topo = pg.topology(pg.unconnected())
        return topo


def write_results(ix, archi, inversion, outtag, deficit_file, archipeligo_islands, tapered_gr):

    print(f'Writing results for archipeligo {ix}...')

    outtag += f"_archi{ix}"

    # Best slip distribution
    f_ix = np.array([champion[0] for champion in archi.get_champions_f()]).argsort()
    preferred_rate = 10 ** (np.array(archi.get_champions_x()).T[:, f_ix])

    # Reconstruct the slip deficit
    reconstructed_deficit = np.matmul(inversion.slip, preferred_rate[:, 0])

    # Output results
    outfile = os.path.join(outdir, f'{outtag}_inverted_ruptures.csv')
    out = np.zeros((inversion.n_ruptures, archipeligo_islands + 5))
    out[:, 0] = inversion.Mw
    out[:, 1] = initial_rates
    out[:, 2] = inversion.mfd(tapered_gr, inversion.Mw, inversion.deficit)  # Target GR-rate
    out[:, 3], out[:, 4] = 10 ** lower_lim, 10 ** upper_lim
    out[:, 5:] = preferred_rate

    columns = ["inverted_rate_" + str(n) for n in range(archipeligo_islands)]
    columns = ["Mw", "initial_rate", "target_rate", "lower", "upper"] + columns
    out_df = pd.DataFrame(out, columns=columns, index=inversion.id)
    out_df.to_csv(outfile, sep='\t', index=True)

    outfile = os.path.join(outdir, f'{outtag}_inverted_bins.csv')
    out = np.zeros((len(inversion.Mw_bins), 4))
    out[:, 0] = np.arange(len(inversion.Mw_bins))
    out[:, 1] = inversion.Mw_bins
    out[:, 2] = np.matmul(inversion.gr_matrix, initial_rates)
    out[:, 3] = np.matmul(inversion.gr_matrix, preferred_rate[:, 0])
    np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6e\t%.6e", header='No\tMw_bin\tinput_N\tinverted_N')

    # Output deficits
    header = '#No\tlon\tlat\tz(km)\ttarget-deficit(mm/yr)\tinverted-deficit(mm/yr)\tmisfit_rel\tmisfit_abs(mm/yr)'
    format = "%.0f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f"
    deficit = np.genfromtxt(deficit_file)
    deficit[:, 3] /= 1000  # Convert to km

    out = np.zeros((deficit.shape[0], 8))
    out[:, :4] = deficit[:, :4]
    out[:, 4] = deficit[:, 9] # Target deficit (mm/yr)
    out[:inversion.n_patches, 5] = reconstructed_deficit
    out[:inversion.n_patches, 6] = reconstructed_deficit / inversion.deficit  # Fractional misfit
    out[:inversion.n_patches, 7] = reconstructed_deficit - inversion.deficit  # Absolute misfit
    if tapered_gr:
        header += "\ttransition_deficit(mm/yr)\ttransition_rel(mm/yr)\ttransition_abs(mm/yr)"
        format += "\t%.6f\t%.6f\t%.6f"
        transitional_deficit = np.hstack([reconstructed_deficit, np.matmul(inversion.extra_slip.toarray(), preferred_rate[:, 0])]).T.reshape(-1, 1)
        transitional_rel = transitional_deficit / deficit[:, 9].reshape(-1, 1)
        transitional_abs = transitional_deficit - deficit[:, 9].reshape(-1, 1)
        out = np.hstack([out, transitional_deficit, transitional_rel, transitional_abs])

    outfile = os.path.join(outdir, f"{outtag}_inversion_results.inv")
    np.savetxt(outfile, out, fmt=format, header=header)


# %% Main function
if __name__ == "__main__":
    total_ruptures = int(rupture_file.split('_df_n')[1].split('.')[0])
    n_ruptures = int(n_ruptures)

    rupture_csv = os.path.join(procdir, rupture_file)

    taper_tag = f"_taper{taper_max_Mw}Mw_alphas{alpha_s:.1f}".replace('.', '-') if tapered_gr else ""
    gr_tag = f"b{str(b).replace('.','-')}" if tapered_gr else f"b{str(b).replace('.','-')}_N{str(N).replace('.','-')}" 
    outtag = f"n{n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_nr{int(nrupt_weight)}{int(nrupt_cuttoff)}{taper_tag}_{gr_tag}_pMax{max_patch}"

    # Check there is the correct number of ruptures available to invert
    if archipeligo and n_ruptures > total_ruptures:
        raise Exception(f"Not enough ruptures in {rupture_file}")
    elif not archipeligo:
        if (n_islands * n_ruptures) > total_ruptures:
            raise Exception(f"Only enough ruptures in {rupture_file} for {int(total_ruptures / n_ruptures)} islands with {n_ruptures} ruptures")

    # %% Load ruptures
    if archipeligo:
        print(f"Loading ruptures from {rupture_file}...  ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
        ruptures_df_list = [pd.read_csv(rupture_csv, nrows=n_ruptures)]
        archipeligo_islands = n_islands
    else:
        ruptures_df_list = []
        topology_name = "UnConnected"
        archipeligo_islands = 1
        print(f"Loading {n_ruptures * n_islands} ruptures from {rupture_file}...  ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
        full_df = pd.read_csv(rupture_csv, header=0, nrows=n_islands * n_ruptures)
        if n_islands * n_ruptures < full_df.shape[0]:
            raise Exception(f"Only {full_df.shape[0]} ruptures in {rupture_csv} - need {n_islands * n_ruptures}")
        for island in range(n_islands):
            print(f"Extracting {n_ruptures} ruptures for island {island + 1}/{n_islands}...", end="\r")
            ruptures_df_list.append(full_df.iloc[island * n_ruptures:(island + 1) * n_ruptures])

        print("")

    # Load target deficit
    deficit = np.genfromtxt(deficit_file)
    deficit = deficit[:, 9]  # d in d=Gm, keep in mm/yr

    if tapered_gr:
        rigidity = pd.read_csv(rigidity_file, sep=' ', index_col=0)
        rigidity_cols = [col for col in rigidity.columns if 'rigidity' in col.lower()]
        if len(rigidity_cols) == 1:
            rigidity = rigidity[rigidity_cols[0]].values
            Mo_rate = np.sum(rigidity * patch_area * (deficit * 1e-3))
            mu_A = rigidity * patch_area  # uA part of Mo = uAs, value per patch.
        elif len(rigidity_cols) > 1:
            raise Exception(f"Multiple rigidity columns found in {rigidity_file} - only 1 should have rigidity in the name")
        else:
            raise Exception(f"No rigidity columns found in {rigidity_file} - put rigidity in the name of the column")
    else:
        mu_A = 0

    if max_patch > -1:
        cols = [str(i) for i in range(max_patch)]
        all_slip_patches = cols.copy()
        mu_A_list = []
        rupture_ROI_thresh = 10  # Percentage threshold of rupture area that must be within the ROI to be included in the inversion
        for ii, ruptures_df in enumerate(ruptures_df_list):
            # Find total number of slip patches for each rupture
            total_patches = (ruptures_df.iloc[:, 3:] > 0).sum(axis=1)
            # Find number of patches in the ROI for each rupture
            roi_patches = (ruptures_df[cols] > 0).sum(axis=1)
            # Find the indicies of all ruptures that occur within the fault region with sufficient patches
            inversion_ruptures = ((roi_patches / total_patches) > (rupture_ROI_thresh / 100))
            inversion_ruptures = inversion_ruptures[inversion_ruptures].index
            # Reduce the dataframe to the ROI ruptures
            ruptures_df = ruptures_df.loc[inversion_ruptures]
            # Find the patches that are ruptured by the ROI ruptures
            slip_patches = ruptures_df[[str(i) for i in range(int(ruptures_df.columns[-1]) + 1)]].sum(axis=0).values
            slip_patches_id = [str(col) for col in np.where(slip_patches > 0)[0]]
            island_slip_patches = [str(patch) for patch in np.sort(np.array(list(set(cols + slip_patches_id))).astype('int'))]
            all_slip_patches += slip_patches_id
            ruptures_df_list[ii] = ruptures_df[['rupt_id', 'mw', 'target_mw'] + island_slip_patches]
            mu_A_list.append(mu_A[np.array(island_slip_patches).astype('int')])
        # all_slip_patches = np.sort(np.array(list(set(all_slip_patches))).astype('int'))  # Get the indexes of all patches that are ruptured
        deficit = deficit[:max_patch]  # Only keep the slip deficit for the ROI
        # mu_A = mu_A[all_slip_patches]  # Only keep the rigidity for the potential ruptured patches inside and out of ROI
    else:
        mu_A_list = [mu_A] * len(ruptures_df_list)  # If not using tapered GR, just use the same mu_A for all islands

    inversion_list = []
    for ruptures_df, mu_A in zip(ruptures_df_list, mu_A_list):
        inversion_list.append(deficitInversion(ruptures_df, deficit, b, N, rate_weight, norm_weight, GR_weight, nrupt_weight, tapered_gr=tapered_gr, taper_max_Mw=taper_max_Mw, mu_A=mu_A, max_patch=max_patch, nrupt_cuttoff=nrupt_cuttoff))

    # %% Write out starting conditions
    inversion = inversion_list[0]
    # Initially set recurrance rate to NSHM GR-rate for each rupture magnitude
    print(f'Calculating initial rupture rates...  ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})')
    lower_lim, upper_lim = inversion.get_bounds()
    lower_lim, upper_lim = np.array(lower_lim).astype(np.float64), np.array(upper_lim).astype(np.float64)

    n_start = 0
    if starting_rate_file:
        print(f"Loading initial rates from {starting_rate_file}")
        initial_rates = pd.read_csv(starting_rate_file, sep='\t', index_col=0)['inverted_rate_0'].values[:inversion.n_ruptures]
        if len(initial_rates) < inversion.n_ruptures:
            raise Exception(f"Initial rates file contains {len(initial_rates)} rates, expected {inversion.n_ruptures}")
        if define_population:
            # Add labeling, if using pre-defined population
            n_start = int(starting_rate_file.split('nIt')[-1].split('_')[0])
    else:
        initial_rates = 10 ** ((upper_lim - lower_lim.min()) * np.random.rand(inversion.n_ruptures) + lower_lim.min())  # Randomly initialise rates to values between lower and upper limit (for when working in log space)

    print(f"Preparing run {os.path.join(outdir, outtag)}")
    # Output the initial conditions
    outfile = os.path.join(outdir, f"{outtag}_input_ruptures.csv")
    out = np.zeros((inversion.n_ruptures, 7))
    out[:, 0] = np.arange(inversion.n_ruptures)
    out[:, 1] = inversion.Mw
    out[:, 2] = inversion.target
    out[:, 3] = initial_rates
    out[:, 4], out[:, 5] = lower_lim, upper_lim
    out[:, 6] = inversion.mfd(tapered_gr, inversion.Mw, inversion.deficit)
    np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.4f\t%.6e\t%.6e\t%.6e\t%.6e", header='No\tMw\ttarget\tinitial_rate\tlower\tupper\ttarget_rate')

    outfile = os.path.join(outdir, f"{outtag}_input_bins.csv")
    out = np.zeros((len(inversion.Mw_bins), 5))
    out[:, 0] = np.arange(len(inversion.Mw_bins))
    out[:, 1] = inversion.Mw_bins
    out[:, 2] = np.matmul(inversion.gr_matrix, initial_rates)
    out[:, 3], out[:, 4] = np.matmul(inversion.gr_matrix, 10 ** lower_lim), np.matmul(inversion.gr_matrix, 10 ** upper_lim)
    np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6e\t%.6e\t%.6e", header='No\tMw_bin\tinput_N\tlower\tupper')

    # %% Prepare pygmo algorithm
    print(f"Preparing pygmo...  ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
    # set up differential evolution algorithm
    print(f"\tSetting up algorithm...  ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
    algo = pg.algorithm(pg.de(gen=n_iterations, ftol=ftol))
    algo.set_verbosity(1000)
    print(algo)
  
    print(f"\tSetting up island topology...  ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
    topo = build_topology(topology_name, archipeligo_islands, ring_plus)
    print(topo)

    archi_list = []

    if define_population:
        print("\tAdding pre-defined starting populations...")
        for inversion in inversion_list:
            pop = pg.population(prob=inversion, size=pop_size)
            # Tell population object what starting values will be
            pop.push_back(np.log10(initial_rates))
            archi_list.append(pg.archipelago(n=archipeligo_islands, algo=algo, pop=pop, t=topo))
    else:
        for ix, inversion in enumerate(inversion_list):
            print(f"\tBuilding archipeligo {ix + 1} of {len(inversion_list)}...   ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
            prob = pg.problem(inversion)
            archi_list.append(pg.archipelago(n=archipeligo_islands, algo=algo, prob=prob, pop_size=pop_size, t=topo))

    # Incase using a defined input from previous run, reflect this in outtag
    if n_start != 0:
        n_iterations = f"{n_iterations + n_start}-init{int(n_start * 1e-3)}k"
    outtag += f"_nIt{n_iterations}"


    # %% Run inversions
    print(archi_list[0])
    print(f"Starting inversions...   ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
    for ix, archi in enumerate(archi_list):
        print(f"\tEvolving archipeligo {ix}...   ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})")
        archi.evolve()

    # Write out results when inversion has ended
    for ix, archi in enumerate(archi_list):
        archi.wait()
        write_results(ix, archi, inversion_list[ix], outtag, deficit_file, archipeligo_islands, tapered_gr)

    print(f"Final results written to {os.path.join(outdir, f'{outtag}')}")
    print(f'All complete :)   ({int((time() - start)/3600):0>2}:{int(((time() - start)/60)%60):0>2}:{int((time() - start)%60):0>2})')