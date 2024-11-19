import pygmo as pg
import numpy as np
from glob import glob
from time import time, sleep
import pandas as pd
import os
from scipy.optimize import lsq_linear
from scipy.sparse import bsr_array
import matplotlib.pyplot as plt

inversion_name = 'hires_rupt'

n_ruptures = 5000
iteration_list = [500000]
rate_weight = 1
norm_weight = 0
GR_weight = 10
ftol = 0.0001
n_islands = 30
pop_size = 20
archipeligo = False
topology_name = 'None'  # 'None', 'Ring', 'FullyConnected'
ring_plus = 1  # Number of connections to add to ring topology

b, N = 1.1, 21.5
max_Mw = 9.5  # Maximum magnitude to use to match GR-Rate

if 'rccuser' in os.getcwd():
    procdir = "/home/rccuser/MudPy/hires_ruptures"
    deficit_file = f"{procdir}/model_info/slip_deficit_trenchlock.slip"
    deficit_file = f"{procdir}/model_info/hk_hires.slip"
else:
    procdir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output"
    deficit_file = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\data\\model_info\\slip_deficit_trenchlock.slip"

starting_rate_file = os.path.abspath(os.path.join(procdir, "hires_deficit", "n5000_S1_N1_GR10_nIt50000_inverted_ruptures.csv"))  # Set to None for random initialisation
starting_rate_file = None

outdir = os.path.abspath(os.path.join(procdir, inversion_name))
if not os.path.exists(outdir):
    os.mkdir(outdir)

pygmo = True
define_population = True

class deficitInversion:
    def __init__(self, ruptures_df: pd.DataFrame, deficit: np.ndarray, b: float, N: float, rate_weight: float, norm_weight: float, GR_weight: float, max_Mw: float):

        self.name = "Slip Deficit Inversion"
        self.deficit = deficit  # Slip deficit (on same grid as ruptures)
        self.n_patches = deficit.shape[0]  # Number of patches
        self.b = b  # b-value for GR-rate
        self.N5 = N  # N value for Mw 5 events
        self.a = np.log10(self.N5) + (b * 5)  # a-value for GR-rate calculated from b and the N-value for Mw 5 events
        self.rate_weight = rate_weight  # Weighting for rate misfit over GR-rate misfit
        self.norm_weight = norm_weight  # Weighting of the normalised GR-rate misfit
        self.GR_weight = GR_weight  # Weighting for GR-rate misfit
        self.max_Mw = max_Mw  # Maximum magnitude to use for GR-rate

        print("Reading rupture dataframe...")
        self.n_ruptures = ruptures_df.shape[0]
        self.id = ruptures_df['rupt_id'].values
        self.Mw = ruptures_df['mw'].values
        self.target = ruptures_df['target_mw'].values
        i0, i1 = ruptures_df.columns.get_loc('0'), ruptures_df.columns.get_loc(str(self.n_patches - 1)) + 1
        self.Mw_bins = np.unique(np.floor(np.array(self.Mw) * 10) / 10)  # Create bins for each 0.1 magnitude increment

        self.sparse_slip = bsr_array(ruptures_df.iloc[:, i0:i1].values.T * 1000)  # Convert slip from m to mm, place in sparse matrix

        self.make_gr_matrix()
        self.GR_rate = 10 ** (self.a - (self.b * self.Mw_bins))  # N-value for each magnitude bin

    # Properties to call matricies as full arrays
    @property
    def slip(self):
        return self.sparse_slip.toarray()

    @property
    def gr_matrix(self):
        return self.sparse_gr_matrix.toarray()

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

    def fitness(self, x: np.ndarray):

        rms, norm_rms, GR_rms = 0, 0, 0

        # Calculate slip-deficit component (Based on NSHM SRM constraint weights)
        if any([self.rate_weight > 0, self.norm_weight > 0]):
            total_slip = self.sparse_slip @ (10 ** x)  # Calculate slip by multiplying slip ratrix by rupture rate vector (sparse matricies can't use matmul)
            rate_misfit = total_slip - self.deficit
            if self.rate_weight > 0:
                rms = np.sqrt(np.mean(rate_misfit ** 2))  # Calculate root mean square misfit of slip - Penalises large absolute differences
            if self.norm_weight > 0:
                norm_rms = np.sqrt(np.mean((rate_misfit / self.deficit) ** 2))  # Calculate root mean square misfit of normalised slip - penalises large relative misfits

        # Calculate GR-rate component
        if self.GR_weight > 0:
            inv_GR = self.sparse_gr_matrix @ (10 ** x)  # Calculate GR-rate for each magnitude bin based on inverted slip rates
            GR_ix = self.Mw_bins <= self.max_Mw  # Only use bins up to the maximum magnitude (so that few high Mw events aren't overweighted)
            # GR_rms = np.sqrt(np.mean((inv_GR - self.GR_rate) ** 2))  # Penalise for deviating from GR-rate
            GR_rms = np.sqrt(np.mean((np.log10(inv_GR[GR_ix]) - np.log10(self.GR_rate[GR_ix])) ** 2))  # Penalise for deviating from log(N)

        cum_rms = (rms * self.rate_weight) + (norm_rms * self.norm_weight) + (GR_rms * self.GR_weight)  # Allow for variable weighting between slip deficit and GR-rate

        return np.array([cum_rms])  

if __name__ == "__main__":
    if os.path.exists(os.path.abspath(os.path.join(procdir, f'rupture_df_n{n_ruptures}.csv'))):
        from_csv = True
        print(f"Loading ruptures from rupture_df_n{n_ruptures}.csv...")
        ruptures_df = pd.read_csv(os.path.abspath(os.path.join(procdir, f'rupture_df_n{n_ruptures}.csv')), nrows=n_ruptures)
    else:
        csv_list = glob(os.path.abspath(os.path.join(procdir, "rupture_df_n*.csv")))
        n_rupts = [int(csv.split('_n')[-1].split('.')[0]) for csv in csv_list]
        n_rupts.sort()
        n_rupts = [n for n in n_rupts if n > n_ruptures]
        if len(n_rupts) > 0:
            print(f"Loading {n_ruptures} ruptures from rupture_df_n{n_rupts[0]}.csv...")
            ruptures_df = pd.read_csv(os.path.abspath(os.path.join(procdir, f'rupture_df_n{n_rupts[0]}.csv')), nrows=n_ruptures)
        else:
            raise Exception(f"No csv files found with at least {n_ruptures} ruptures")

    deficit = np.genfromtxt(deficit_file)
    deficit = deficit[:, 9]  # d in d=Gm, keep in mm/yr

    inversion = deficitInversion(ruptures_df, deficit, b, N, rate_weight, norm_weight, GR_weight, max_Mw)

    # Initially set recurrance rate to NSHM GR-rate for each rupture magnitude
    print('Calculating initial rupture rates...')
    lower_lim, upper_lim = inversion.get_bounds()
    lower_lim, upper_lim = np.array(lower_lim).astype(np.float64), np.array(upper_lim).astype(np.float64)
    n_start = 0
    if starting_rate_file:
        print(f"Loading initial rates from {starting_rate_file}")
        initial_rates = pd.read_csv(starting_rate_file, sep='\t', index_col=0)['inverted_rate_0'].values[:n_ruptures]
        if len(initial_rates) < n_ruptures:
            raise Exception(f"Initial rates file contains {len(initial_rates)} rates, expected {n_ruptures}")
        if define_population:
            # Add labeling, if using pre-defined population
            n_start = int(starting_rate_file.split('nIt')[-1].split('_')[0])
    else:
        initial_rates = 10 ** ((upper_lim - lower_lim.min()) * np.random.rand(n_ruptures) + lower_lim.min())  # Randomly initialise rates to values between lower and upper limit (for when working in log space)

    # Output the initial conditions
    outfile = os.path.join(outdir, f'n{inversion.n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_input_ruptures.csv')
    out = np.zeros((inversion.n_ruptures, 7))
    out[:, 0] = np.arange(inversion.n_ruptures)
    out[:, 1] = inversion.Mw
    out[:, 2] = inversion.target
    out[:, 3] = initial_rates
    out[:, 4], out[:, 5] = lower_lim, upper_lim
    out[:, 6] = 10 ** (inversion.a - (inversion.b * inversion.Mw))
    np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.4f\t%.6e\t%.6e\t%.6e\t%.6e", header='No\tMw\ttarget\tinitial_rate\tlower\tupper\ttarget_rate')

    outfile = os.path.join(outdir, f'n{inversion.n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_input_bins.csv')
    out = np.zeros((len(inversion.Mw_bins), 5))
    out[:, 0] = np.arange(len(inversion.Mw_bins))
    out[:, 1] = inversion.Mw_bins
    out[:, 2] = np.matmul(inversion.gr_matrix, initial_rates)
    out[:, 3], out[:, 4] = np.matmul(inversion.gr_matrix, 10 ** lower_lim), np.matmul(inversion.gr_matrix, 10 ** upper_lim)
    np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6e\t%.6e\t%.6e", header='No\tMw_bin\tinput_N\tlower\tupper')

    initial_slip = np.matmul(inversion.slip, initial_rates)  # Calculate what the slip distribution would look like from the initial rates
    # Output the deficit to be resolved for, and the inital slip distribution based on events + input rates
    output = np.genfromtxt(deficit_file)
    output[:, 3] /= 1000  # Convert to km
    output = output[:, :6]
    output[:, 4] = deficit
    output[:, 5] = initial_slip
    outfile = os.path.join(outdir, f'n{inversion.n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_initial_deficit.inv')
    np.savetxt(outfile, output, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
            header='#No\tlon\tlat\tz(km)\tinital_deficit(mm/yr)\tinital_slip(mm/yr)')

    for n_iterations in iteration_list:
        if pygmo:
            print(f'Optimising {n_iterations} generations with pygmo...')

            # set up differential evolution algorithm
            algo = pg.algorithm(pg.de(gen=n_iterations, ftol=ftol))

            # set up self adaptive differential evolution algorithm
            # algo = pg.algorithm(pg.sade(gen=n_iterations, variant_adptv=2))

            # set up coronas simulated annealing
            # algo = pg.algorithm(pg.simulated_annealing(Ts=10., Tf=.1, n_T_adj=10, n_range_adj=10, bin_size=10, start_range=1.))

            # set up inversion class to run algorithm on
            if archipeligo:
                # Some output to check on progress
                algo.set_verbosity(1000)
                print(algo)
                print('Setting up islands...')
                if topology_name == 'FullyConnected':
                    topo = pg.topology(pg.fully_connected(n_islands))
                elif topology_name == 'Ring':
                    if ring_plus == 0:
                        topo = pg.topology(pg.ring(n_islands))
                    else:
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

                print(topo)

                if define_population:
                    print('Setting up archipeligo with defined starting population...')
                    pop = pg.population(prob=inversion, size=pop_size)
                    # Tell population object what starting values will be
                    pop.push_back(np.log10(initial_rates))
                    archi = pg.archipelago(n=n_islands, algo=algo, pop=pop, t=topo)
                else:
                    print('Setting up archipeligo...')
                    prob = pg.problem(inversion)
                    archi = pg.archipelago(n=n_islands, algo=algo, prob=prob, pop_size=pop_size, t=topo)

                print(archi)
                print('Evolving populations...')
                start = time()   
                archi.evolve()
                archi.wait()

                # Best slip distribution
                f_ix = np.array([champion[0] for champion in archi.get_champions_f()]).argsort()
                preferred_rate = 10 ** (np.array(archi.get_champions_x()).T[:, f_ix])
            else:
                # Lots of output to check on progress
                algo.set_verbosity(100)
                print(algo)
                n_islands = 1
                pop = pg.population(prob=inversion, size=pop_size)

                # Tell population object what starting values will be
                pop.push_back(np.log10(initial_rates))

                # Run algorithm
                print(f'Inverting {inversion.n_ruptures} ruptures...')
                start = time()
                pop = algo.evolve(pop)

                # Best slip distribution
                preferred_rate = 10 ** pop.champion_x.reshape(-1,1)
        else:
            n_islands = 1
            print('Prepping megamatrix...')
            mega_matrix = np.vstack([inversion.slip, inversion.gr_matrix * GR_weight])
            full_rates = np.hstack([inversion.deficit, inversion.GR_rate * GR_weight])
            print(f'Inverting {inversion.n_ruptures} ruptures with max_iter {n_iterations} on scipy...')
            start = time()
            results = lsq_linear(mega_matrix, full_rates, bounds=(lower_lim, upper_lim), verbose=2, method='bvls', max_iter=n_iterations)
            preferred_rate = results.x.reshape(-1,1)
            misfit = results.fun

        print(f'Inversion of {inversion.n_ruptures} ruptures complete in {time() - start:.2f}s...')

        # Reconstruct the slip deficit
        reconstructed_deficit = np.matmul(inversion.slip, preferred_rate[:, 0])

        if n_start != 0:
            n_iterations = f"{n_iterations + n_start}-init{int(n_start * 1e-3)}k"

        # Output results
        outfile = os.path.join(outdir, f'n{inversion.n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_nIt{n_iterations}_inverted_ruptures.csv')
        out = np.zeros((inversion.n_ruptures, n_islands + 5))
        out[:, 0] = inversion.Mw
        out[:, 1] = initial_rates
        out[:, 2] = 10 ** (inversion.a - (inversion.b * inversion.Mw))
        out[:, 3], out[:, 4] = 10 ** lower_lim, 10 ** upper_lim
        out[:, 5:] = preferred_rate
        # np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e", header='No\tMw\tinitial_rate\tinverted_rate\ttarget_rate\tlower\tupper')
        columns = ["inverted_rate_" + str(n) for n in range(n_islands)]
        columns = ["Mw", "initial_rate", "target_rate", "lower", "upper"] + columns
        out_df = pd.DataFrame(out, columns=columns, index=inversion.id)
        out_df.to_csv(outfile, sep='\t', index=True)

        outfile = os.path.join(outdir, f'n{inversion.n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_nIt{n_iterations}_inverted_bins.csv')
        out = np.zeros((len(inversion.Mw_bins), 4))
        out[:, 0] = np.arange(len(inversion.Mw_bins))
        out[:, 1] = inversion.Mw_bins
        out[:, 2] = np.matmul(inversion.gr_matrix, initial_rates)
        out[:, 3] = np.matmul(inversion.gr_matrix, preferred_rate[:, 0])
        np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6e\t%.6e", header='No\tMw_bin\tinput_N\tinverted_N')

        # Output deficits
        deficit = np.genfromtxt(deficit_file)
        deficit[:, 3] /= 1000  # Convert to km

        out = np.zeros_like(deficit[:, :6])
        out[:, :4] = deficit[:, :4]

        out[:, 4] = inversion.deficit
        out[:, 5] = reconstructed_deficit
        outfile = os.path.join(outdir, f'n{inversion.n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_nIt{n_iterations}_deficit.inv')
        np.savetxt(outfile, out, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
                header='#No\tlon\tlat\tz(km)\ttarget-deficit(mm/yr)\tinverted-deficit(mm/yr)')

        out[:, 4] = reconstructed_deficit / inversion.deficit  # Fractional misfit
        if pygmo:
            out[:, 5] = reconstructed_deficit - inversion.deficit  # Absolute misfit
        else:
            out[:, 5] = misfit[:inversion.n_patches]  # Absolute misfit
        outfile = os.path.join(outdir, f'n{inversion.n_ruptures}_S{int(rate_weight)}_N{int(norm_weight)}_GR{int(GR_weight)}_nIt{n_iterations}_misfit.inv')
        np.savetxt(outfile, out, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
                header='No\tlon\tlat\tz(km)\tmisfit_rel(mm/yr)\tmisfit_abs(mm/yr)')

    if not archipeligo:
        uda = algo.extract(pg.de)
        log = uda.get_log()
        plt.semilogy([entry[0] for entry in log], [entry[2] for entry in log], 'k--')
        plt.show()

    print('All Complete! :)')