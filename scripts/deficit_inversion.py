import pygmo as pg
import numpy as np
from glob import glob
from time import time
import pandas as pd
import os

rupture_dir = 'C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\output\\ruptures'
rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures'
rupture_list = glob(f'{rupture_dir}\\hikkerk3D_locking_NZNSHMscaling*.rupt')[::3]

deficit_file = 'C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\data\\model_info\\slip_deficit_trenchlock.slip'

deficit = np.genfromtxt(deficit_file)
deficit = deficit[:, 9] / 1000  # d in d=Gm, convert from mm to m

lat_min = -90.0
min_mw = 8.5
# Filter rupture list
apply_filter = False
if apply_filter:
    n_ruptures = len(rupture_list)
    for ix, rupture in enumerate(rupture_list[::-1]):
        print(f"Filtering ruptures... ({ix+1}/{n_ruptures})", end='\r')
        with open(rupture.replace('.rupt', '.log')) as fid:
            lines = fid.readlines()
            for line in lines:
                if 'Hypocenter (lon,lat,z[km])' in line:
                    lon_tmp, lat_tmp, z_tmp = line.strip('\n').split()[-1].split(',')
                    if float(lat_tmp.strip('()')) < lat_min:
                        rupture_list.remove(rupture)
                        break
                if 'Actual magnitude' in line:
                    if float(line.strip('\n').split()[-1]) < min_mw:
                        rupture_list.remove(rupture)
                        break
    print('')
b, N = 1.1, 21.5

class deficitInversion:
    def __init__(self, rupture_list: list, deficit: np.ndarray, b: float, N: float):

        self.name = "Slip Deficit Inversion"
        self.deficit = deficit  # Slip deficit (on same grid as ruptures)
        self.ruptures = rupture_list  # List of ruptures from fakequakes
        self.n_patches = deficit.shape[0]  # Number of patches
        self.n_ruptures = len(rupture_list)  # Number of ruptures
        self.b = b  # b-value for GR-rate
        self.N = N  # N value for Mw 5 events
        self.a = np.log10(N) + (b * 5)  # a-value for GR-rate calculated from b, N
        self.rate_weight = 1  # Weighting for rate misfit over GR-rate misfit

        self.slip = self.create_g_matrix()  # I*J matrix of rupture slip, where I is the number of patches and J is the number of ruptures
        self.Mw, self.target_Mw, self.Mw_bins = self.find_magnitudes()  # Magnitude of each rupture, rupture target magnitudes, and magnitude_bins
        self.GR_rate = (self.a - (self.b * self.Mw_bins))  # log10(N) for each magnitude bin


    def create_g_matrix(self):
        print(f"Creating rupture G matrix... ({0}/{self.n_ruptures})", end='\r')
        G = np.zeros((self.n_patches, self.n_ruptures))
        start = time()
        for ix, rupture_file in enumerate(self.ruptures):
            rupture = pd.read_csv(rupture_file, sep='\t')
            if 'total-slip(m)' in rupture.columns:
                G[:, ix] = rupture['total-slip(m)']  # Save storage space fakequakes output
            else:
                G[:, ix] = (rupture['ss-slip(m)'] ** 2 + rupture['ds-slip(m)'] ** 2) ** 0.5  # Default fakequakes output
            print(f"Creating rupture G matrix... ({ix + 1}/{self.n_ruptures}) {(time() - start)/(ix + 1):.5f}", end='\r')
        print('')
        return G
    
    def find_magnitudes(self):
        Mw = []  # Actual magnitude of each rupture
        target = []  # requested magnitude of the rupture when entered into fakeqaukes
        for rupture in self.ruptures:
            with open(rupture.replace('.rupt', '.log')) as fid:
                lines = fid.readlines()
                for line in lines:
                    if 'Target magnitude' in line:
                        target.append(float(line.strip('\n').split()[-1]))
                    if 'Actual magnitude' in line:
                        Mw.append(float(line.strip('\n').split()[-1]))
        target_Mw = np.round(np.array(target), 4)
        binned_Mw = np.unique(target_Mw)
        min_Mw = binned_Mw[0]
        max_Mw = binned_Mw[-1]
        binned_Mw = np.linspace(min_Mw, max_Mw, len(binned_Mw))  # Requested magnitudes, to be used as bins for GR-rate (in case mag bin is missing)
        return np.array(Mw), target_Mw, binned_Mw

    def get_bounds(self):
        lower_bound = [0] * self.n_ruptures  # Allow rupture to not occur
        a = np.log10(100) + (self.b * 5)
        upper_bound = list(10 ** (a - (self.b * self.Mw)))  # Upperbound based on b-value, but 100 5Mw events a year
        return (lower_bound, upper_bound)

    def fitness(self, x: np.ndarray):
        total_slip = np.matmul(self.slip, x)  # Calculate slip by multiplying slip ratrix by rupture rate vector
        rms = np.sqrt(np.mean((self.deficit - total_slip) ** 2))  # Calculate root mean square misfit of slip
        inv_GR = []  # Calculate GR-rate for each magnitude bin based on inverted slip rates
        x = np.array([r if r > 0 else 0 for r in x])  # Sometimes rates become -1e8, so set negatives to 0
        for mw in self.Mw_bins:  # Use bins not actual to account for tailing off of Mw if magnitudes not forced in fakequakes
            cumfreq = np.sum(x[np.where(self.Mw >= mw)[0]])
            if cumfreq <= 0:
                inv_GR.append(0)  # Calculate log(N) for each magnitude bin (set to zero if no ruptures in or higher than bin)
            else:
                inv_GR.append(np.log10(cumfreq))  # Calculate log(N) for each magnitude bin
        GR_rms = np.sqrt(np.mean((self.GR_rate - inv_GR) ** 2))  # Penalise for deviating from GR-rate
        return np.array([(rms * self.rate_weight) + GR_rms])  # Allow for weighting of rate misfit over GR-rate misfit
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
inversion = deficitInversion(rupture_list, deficit, b, N)

# Choose optimisation algorithm
nl = pg.nlopt('slsqp')
# Tolerance to make algorithm stop trying to improve result
nl.ftol_abs = 1.

# Set up basin-hopping metaalgorithm
algo = pg.algorithm(uda=pg.mbh(nl, stop=1, perturb=.4))
# Lots of output to check on progress
algo.set_verbosity(1)

# set up inversion class to run algorithm on
print('Setting up inversion...')
pop = pg.population(prob=inversion)

# Initially set recurrance rate to NSHM GR-rate for each rupture magnitude
initial_array = 10 ** (inversion.a - (inversion.b * inversion.Mw))
# Tell population object what starting values will be
pop.push_back(initial_array)

# Run algorithm
print('Running inversion...')
start = time()
pop = algo.evolve(pop)
print(f'Inversion complete. Time taken: {time() - start:.2f}s')

# Best slip distribution
preferred_rate = pop.champion_x
#preferred_rate = initial_array
# Reconstruct the slip deficit
reconstructed_deficit = np.matmul(inversion.slip, preferred_rate)

# Output results
outfile = rupture_dir + f'\\preferred_rate_n{inversion.n_ruptures}.txt'
out = np.zeros((preferred_rate.shape[0], 3))
out[:, 0] = np.arange(preferred_rate.shape[0])
out[:, 1] = inversion.Mw
out[:, 2] = preferred_rate
np.savetxt(outfile, out, fmt="%.0f\t%.3f\t%.6f", header='#No\tMw\trate')


# Output deficits
deficit = np.genfromtxt(deficit_file)
deficit[:, 3] /= 1000  # Convert to km

deficit[:, 9] = reconstructed_deficit
outfile = rupture_dir + f'\\deficit_n{inversion.n_ruptures}.inv'
np.savetxt(outfile, deficit, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.0f\t%.0f\t%.0f",
           header='#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel')

deficit[:, 8] = reconstructed_deficit / inversion.deficit  # Absolute deficit
deficit[:, 9] = reconstructed_deficit - inversion.deficit  # Fractional deficit
outfile = rupture_dir + f'\\misfit_n{inversion.n_ruptures}.inv'
np.savetxt(outfile, deficit, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.0f\t%.0f\t%.0f",
           header='No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tmisfit_perc(mm/yr)\tmisfit_mag(mm/yr)\trupt_time\trigid\tvel')
