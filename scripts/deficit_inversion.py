import pygmo as pg
import numpy as np
from glob import glob
from time import time
import pandas as pd
import os
from scipy.optimize import lsq_linear

rupture_step = 36000

rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures'
rupture_list = glob(f'{rupture_dir}\\hikkerk3D_locking_NZNSHMscaling*.rupt')[::rupture_step]

if os.path.exists(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{rupture_step}.csv'))):
   from_csv = True
   rupture_list = [os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{rupture_step}.csv'))]
else:
    from_csv = False
    rupture_list = glob(f'{rupture_dir}\\hikkerk3D_locking_NZNSHMscaling*.rupt')[::rupture_step]

deficit_file = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\data\\model_info\\slip_deficit_trenchlock.slip'

deficit = np.genfromtxt(deficit_file)
deficit = deficit[:, 9]  # d in d=Gm, keep in mm/yr

rate_weight = 1
GR_weight = 1

pygmo = False

if rate_weight == 0:
    create_G = False
else:
    create_G = True

b, N = 1.1, 21.5

class deficitInversion:
    def __init__(self, rupture_list: list, deficit: np.ndarray, b: float, N: float, rate_weight: float, GR_weight: float, from_csv: bool = False):

        self.name = "Slip Deficit Inversion"
        self.deficit = deficit  # Slip deficit (on same grid as ruptures)
        self.ruptures = rupture_list  # List of ruptures from fakequakes
        self.n_patches = deficit.shape[0]  # Number of patches
        self.b = b  # b-value for GR-rate
        self.N5 = N  # N value for Mw 5 events
        self.a = np.log10(self.N5) + (b * 5)  # a-value for GR-rate calculated from b and the N-value for Mw 5 events
        self.rate_weight = rate_weight  # Weighting for rate misfit over GR-rate misfit
        self.GR_weight = GR_weight  # Weighting for GR-rate misfit

        if from_csv:
            print(f"Reading rupture dataframe...")
            ruptures_df = pd.read_csv(rupture_list[0])
            self.n_ruptures = ruptures_df.shape[0]
            self.Mw = ruptures_df['mw'].values
            self.target = ruptures_df['target'].values
            self.Mw_bins = np.unique(np.floor(np.array(self.target) * 10) / 10)
            if self.rate_weight > 0:
                self.slip = ruptures_df.iloc[:, 2:].values.T * 1000  # Slip in mm, convert from m to mm
            else:
                self.slip = np.zeros((self.n_patches, self.n_ruptures))
        else:
            self.n_ruptures = len(rupture_list)  # Number of ruptures
            if self.rate_weight > 0:
                self.slip = self.create_g_matrix()  # I*J matrix of rupture slip, where I is the number of patches and J is the number of ruptures
            else:
                self.slip = np.zeros((self.n_patches, self.n_ruptures))
            self.find_magnitudes()  # Magnitude of each rupture and magnitude_bins
        
        self.make_gr_matrix()
        self.GR_rate = 10 ** (self.a - (self.b * self.Mw_bins))  # N-value for each magnitude bin

    def create_g_matrix(self):
        print(f"Creating rupture G matrix... ({0}/{self.n_ruptures})", end='\r')
        G = np.zeros((self.n_patches, self.n_ruptures))
        start = time()
        for ix, rupture_file in enumerate(self.ruptures):
            rupture = pd.read_csv(rupture_file, sep='\t')
            if 'total-slip(m)' in rupture.columns:
                G[:, ix] = rupture['total-slip(m)'][:self.n_patches] * 1000 # Save storage space fakequakes output, convert from m to mm
            else:
                G[:, ix] = (rupture['ss-slip(m)'] ** 2 + rupture['ds-slip(m)'] ** 2) ** 0.5 * 1000  # Default fakequakes output, convert from m to mm
            print(f"Creating rupture G matrix... ({ix + 1}/{self.n_ruptures}) {(time() - start)/(ix + 1):.5f}", end='\r')
        print('')
        return G

    def find_magnitudes(self):
        Mw = []  # Actual magnitude of each rupture
        target = []  # requested magnitude of the rupture when entered into fakeqaukes
        for ix, rupture in enumerate(self.ruptures):
            print(f"Finding rupture magnitudes... ({ix}/{self.n_ruptures})", end='\r')
            with open(rupture.replace('.rupt', '.log')) as fid:
                lines = fid.readlines()
                for line in lines:
                    if 'Target magnitude' in line:
                        target.append(float(line.strip('\n').split()[-1]))
                    if 'Actual magnitude' in line:
                        Mw.append(float(line.strip('\n').split()[-1]))
        print('')
        self.Mw = np.array(Mw)
        self.target = np.array(target)
        target_Mw = np.floor(np.array(self.target) * 10) / 10  # Get the fakequake rupture target magnitude, rounded to deal with floating point errors
        self.Mw_bins = np.unique(target_Mw)  # Use the unique target magnitudes as the bins for GR-rate

    def make_gr_matrix(self):
        print(f"Making GR Matrix...")
        gr_matrix = np.zeros((len(self.Mw_bins), self.n_ruptures)).astype('bool')  # Make an I-J matrix for calculating GR rates, where I is number of magnitude bins and J is number of ruptures
        for ix, bin in enumerate(self.Mw_bins):
            gr_matrix[ix, :] = (self.Mw >= bin)
        self.gr_matrix = gr_matrix.astype('int')

    def get_bounds(self):
        lower_bound = [0] * self.n_ruptures  # Allow rupture to not occur
        a = np.log10(100) + (self.b * 5)
        # upper_bound = 10 ** (a - (self.b * self.Mw))  # Upperbound from 100 5Mw events a year, and 90% b-value (Currently this will overshoot, as is the bin rate not indivdual rupture rate)
        unique_mag, sort_ix = np.unique(self.Mw, return_inverse=True)
        upper_bound = 10 ** (a - (self.b * unique_mag))  # Upperbound from 100 5Mw events a year, and 90% b-value (Currently this will overshoot, as is the bin rate not indivdual rupture rate)
        rates = upper_bound[:-1] - upper_bound[1:]  # Calculate the rate difference between each magnitude bin
        rates = np.append(rates, upper_bound[-1])  # Add the rate of the largest magnitude bin
        window = int(self.n_ruptures / 100)  # Window size for moving average
        rates[window-1:] = np.convolve(rates, np.ones(window), "valid") / (window)  # Calculate moving average
        # Calculate moving average of first n values that are excluded by convole
        rates[:window] = (np.convolve(rates[::-1], np.ones(window), "valid") / (window))[-window:][::-1]
        upper_bound = rates[sort_ix]
        #upper_bound = [100] * self.n_ruptures
        return (lower_bound, upper_bound)
    
    def fitness(self, x: np.ndarray):
        total_slip = np.matmul(self.slip, x)  # Calculate slip by multiplying slip ratrix by rupture rate vector
        rms = np.sqrt(np.mean((self.deficit - total_slip) ** 2))  # Calculate root mean square misfit of slip
        inv_GR = np.matmul(self.gr_matrix, x)  # Calculate GR-rate for each magnitude bin based on inverted slip rates
        GR_rms = np.sqrt(np.mean((inv_GR - self.GR_rate) ** 2))  # Penalise for deviating from GR-rate
        #GR_rms = np.sqrt(np.mean((np.log10(inv_GR) - np.log10(self.GR_rate)) ** 2))  # Penalise for deviating from GR-rate
        cum_rms = (rms * self.rate_weight) + (GR_rms * self.GR_weight)  # Allow for variable weighting between slip deficit and GR-rate

        mega_matrix = np.vstack([inversion.slip, inversion.gr_matrix])
        full_rates = np.hstack([inversion.deficit, inversion.GR_rate])
        total_results = np.matmul(mega_matrix, x)
        cum_rms = np.sqrt(np.mean((full_rates - total_results) ** 2))

        return np.array([cum_rms])
#    def gradient(self, x):
#        return pg.estimate_gradient(lambda x: self.fitness(x), x)

inversion = deficitInversion(rupture_list, deficit, b, N, rate_weight, GR_weight, from_csv)

# Initially set recurrance rate to NSHM GR-rate for each rupture magnitude
print('Calculating initial rupture rates...')
sorted_ix = np.argsort(inversion.Mw)[::-1]  # Sort based on largest magnitude first
Nvalue = 16.5  # N-value for Mw 5 events (set to different to the NSHM)
Bvalue = 0.95  # B-value for GR-rate (set to different to the NSHM)
a = np.log10(Nvalue) + (Bvalue * 5)  # Calculate a-value for GR-rate
GR_rate = 10 ** (a - (Bvalue * inversion.Mw))  # Calculate N value for each rupture magnitude (number of events >= Mw per year)
freq = np.zeros_like(GR_rate)
cum_freq = np.zeros_like(GR_rate)
initial_rates = np.zeros_like(GR_rate)
for ix in sorted_ix:
    mw = inversion.Mw[ix]  # Find magnitude of rupture
    freq[ix] = np.sum(inversion.Mw == mw)  # Find number ruptures of the exact same magnitude
    cum_freq[ix] = np.sum(inversion.Mw > mw)  # Find number of ruptures of a higher magnitude
    higher_rates = np.sum(initial_rates[inversion.Mw > mw])  # Calculate rupture rate for all higher magnitudes
    initial_rates[ix] = (GR_rate[ix] - higher_rates) / freq[ix]  # Calculate initial rate as (N-value - rate of higher magnitude) / number of ruptures of same magnitude

lower_lim, upper_lim = inversion.get_bounds()
if any(initial_rates > upper_lim):
    upper_lim = np.array(upper_lim)
    initial_rates[np.where(initial_rates > upper_lim)[0]] = upper_lim[np.where(initial_rates > upper_lim)[0]]
if any(initial_rates < lower_lim):
    lower_lim = np.array(lower_lim)
    initial_rates[np.where(initial_rates < lower_lim)[0]] = lower_lim[np.where(initial_rates < lower_lim)[0]]
#initial_rates = initial_rates * 0 + 1e-6
# Output the initial conditions
outfile = rupture_dir + f'\\..\\n{inversion.n_ruptures}_input_ruptures.txt'
out = np.zeros((inversion.n_ruptures, 7))
out[:, 0] = np.arange(inversion.n_ruptures)
out[:, 1] = inversion.Mw
out[:, 2] = inversion.target
out[:, 3] = initial_rates
out[:, 4], out[:, 5] = lower_lim, upper_lim
out[:, 6] = 10 ** (inversion.a - (inversion.b * inversion.Mw))
np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.4f\t%.6f\t%.6f\t%.6f\t%.6f", header='No\tMw\ttarget\tinitial_rate\tlower\tupper\ttarget_rate')

outfile = rupture_dir + f'\\..\\n{inversion.n_ruptures}_input_bins.txt'
out = np.zeros((len(inversion.Mw_bins), 5))
out[:, 0] = np.arange(len(inversion.Mw_bins))
out[:, 1] = inversion.Mw_bins
out[:, 2] = np.matmul(inversion.gr_matrix, initial_rates)
out[:, 3], out[:, 4] = np.matmul(inversion.gr_matrix, lower_lim), np.matmul(inversion.gr_matrix, upper_lim)
np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6f\t%.6f\t%.6f", header='No\tMw_bin\tinput_N\tlower\tupper')

initial_slip = np.matmul(inversion.slip, initial_rates)  # Calculate what the slip distribution would look like from the initial rates
# Output the deficit to be resolved for, and the inital slip distribution based on events + input rates
output = np.genfromtxt(deficit_file)
output[:, 3] /= 1000  # Convert to km
output[:, 8:10] = np.zeros_like(output[:, 8:10])
output[:, 8] = deficit
output[:, 9] = initial_slip
outfile = rupture_dir + f'\\..\\n{inversion.n_ruptures}_initial_deficit.inv'
np.savetxt(outfile, output, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f",
           header='#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel')

if pygmo:
    print('Optimising with pygmo...')
    # Choose optimisation algorithm
    nl = pg.scipy_optimize(method="L-BFGS-B")

    # Tolerance to make algorithm stop trying to improve result
    nl.ftol_abs = 0.0001

    # Set up basin-hopping metaalgorithm
    algo = pg.algorithm(uda=pg.mbh(nl, stop=5, perturb=1.))

    algo = pg.algorithm(pg.de(gen = 20000))

    # Lots of output to check on progress
    algo.set_verbosity(100)

    print(algo)

    # set up inversion class to run algorithm on
    print('Setting up inversion...')
    pop = pg.population(prob=inversion, size=20)
    # Tell population object what starting values will be
    pop.push_back(initial_rates)

    # Run algorithm
    print(f'Inverting {inversion.n_ruptures} ruptures...')
    start = time()
    pop = algo.evolve(pop)
    print(f'Inversion complete. Time taken: {time() - start:.2f}s')

    # Best slip distribution
    preferred_rate = pop.champion_x
else:
    print('Inverting with scipy...')
    mega_matrix = np.vstack([inversion.slip, inversion.gr_matrix * GR_weight])
    full_rates = np.hstack([inversion.deficit, inversion.GR_rate * GR_weight])
    preferred_rate = lsq_linear(mega_matrix, full_rates, bounds=(lower_lim, upper_lim), verbose=2, method='bvls').x

print(f'Inversion of {inversion.n_ruptures} ruptures complete...')

# Reconstruct the slip deficit
reconstructed_deficit = np.matmul(inversion.slip, preferred_rate)

# Output results
outfile = rupture_dir + f'\\..\\n{inversion.n_ruptures}_inverted_ruptures.txt'
out = np.zeros((inversion.n_ruptures, 7))
out[:, 0] = np.arange(inversion.n_ruptures)
out[:, 1] = inversion.Mw
out[:, 2] = initial_rates
out[:, 3] = preferred_rate
out[:, 4] = 10 ** (inversion.a - (inversion.b * inversion.Mw))
out[:, 5], out[:, 6] = lower_lim, upper_lim
np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f", header='No\tMw\tinitial_rate\tinverted_rate\ttarget_rate\tlower\tupper')

outfile = rupture_dir + f'\\..\\n{inversion.n_ruptures}_inverted_bins.txt'
out = np.zeros((len(inversion.Mw_bins), 4))
out[:, 0] = np.arange(len(inversion.Mw_bins))
out[:, 1] = inversion.Mw_bins
out[:, 2] = np.matmul(inversion.gr_matrix, initial_rates)
out[:, 3] = np.matmul(inversion.gr_matrix, preferred_rate)
np.savetxt(outfile, out, fmt="%.0f\t%.4f\t%.6f\t%.6f", header='No\tMw_bin\tinput_N\tinverted_N')

# Output deficits
deficit = np.genfromtxt(deficit_file)
deficit[:, 3] /= 1000  # Convert to km

deficit[:, 9] = reconstructed_deficit
outfile = rupture_dir + f'\\..\\n{inversion.n_ruptures}_deficit.inv'
np.savetxt(outfile, deficit, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.0f\t%.0f\t%.0f",
           header='#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel')

deficit[:, 8] = reconstructed_deficit / inversion.deficit  # Fractional misfit
deficit[:, 9] = reconstructed_deficit - inversion.deficit  # Absolute misfit
outfile = rupture_dir + f'\\..\\n{inversion.n_ruptures}_misfit.inv'
np.savetxt(outfile, deficit, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f",
           header='No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tmisfit_perc(mm/yr)\tmisfit_mag(mm/yr)\trupt_time\trigid\tvel')
