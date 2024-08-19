import pygmo as pg
import numpy as np
from time import time
b, N = 1.1, 21.5

rng = np.random.default_rng()
rupture_mws = []

rupture_bins = np.arange(7.5, 9.5, 0.1)
for i in np.arange(7.5, 9.5, 0.1):
    rupture_mws.append(np.ones((100,)) * i + rng.normal(0, 0.1/3, size=(100,)))
rupture_mws = np.concatenate(rupture_mws)
# rupture_mws = rupture_mws * rng.uniform(0, 1, rupture_mws.shape)

# Initially set recurrance rate to NSHM GR-rate for each rupture magnitude
print('Calculating initial rupture rates...')
sorted_ix = np.argsort(rupture_mws)[::-1]  # Sort based on largest magnitude first
Nvalue = 16.5  # N-value for Mw 5 events (set to different to the NSHM)
Bvalue = 0.95  # B-value for GR-rate (set to different to the NSHM)
a = np.log10(Nvalue) + (Bvalue * 5)  # Calculate a-value for GR-rate
GR_rate = 10 ** (a - (Bvalue * rupture_mws))  # Calculate N value for each rupture magnitude (number of events >= Mw per year)
freq = np.zeros_like(GR_rate)
cum_freq = np.zeros_like(GR_rate)
initial_rates = np.zeros_like(GR_rate)
for ix in sorted_ix:
    mw = rupture_mws[ix]  # Find magnitude of rupture
    freq[ix] = np.sum(rupture_mws == mw)  # Find number ruptures of the exact same magnitude
    cum_freq[ix] = np.sum(rupture_mws > mw)  # Find number of ruptures of a higher magnitude
    higher_rates = np.sum(initial_rates[rupture_mws > mw])  # Calculate rupture rate for all higher magnitudes
    initial_rates[ix] = (GR_rate[ix] - higher_rates) / freq[ix]  # Calculate initial rate as (N-value - rate of higher magnitude) / number of ruptures of same magnitude

# Bung a bit of noise on those rates
rng = np.random.default_rng()
noise = rng.normal(1, 0.4, size=len(initial_rates))
noise[np.where(noise == 0)] = 1  # Just in case a rate is set to 0
initial_rates_n = np.abs(initial_rates * noise)  # Keep those rates positive
initial_rates_n = rng.uniform(0, 10, initial_rates_n.shape)

class deficitInversion:
    def __init__(self, mws: np.array, b: float, N, rupture_bins: np.ndarray):

        self.name = "Slip Deficit Inversion"
        self.mws = mws
        self.rupture_bins = rupture_bins
        self.n_ruptures = len(mws)  # Number of ruptures
        self.gr_matrix = np.zeros((len(self.rupture_bins), self.n_ruptures))
        self.b = b  # b-value for GR-rate
        self.N = N  # N value for Mw 5 events
        self.a = np.log10(N) + (b * 5)  # a-value for GR-rate calculated from b, N
        self.rate_weight = 0  # Weighting for rate misfit over GR-rate misfit
        self.GR_weight = 1  # Weighting for GR-rate misfit
        self.find_magnitudes()  # Magnitude of each rupture, rupture target magnitudes, and magnitude_bins
        self.GR_rate = 10**(self.a - (self.b * self.rupture_bins))  # log10(N) for each magnitude bin

    def find_magnitudes(self):
        gr_matrix = np.zeros((len(self.rupture_bins), self.n_ruptures))
        for i, bin in enumerate(self.rupture_bins):
            for j, mw in enumerate(self.mws):
                if np.round(mw,1)==np.round(bin,1):
                    gr_matrix[i, j] = 1
        self.gr_matrix = gr_matrix
    def get_bounds(self):
        lower_bound = [0] * self.n_ruptures  # Allow rupture to not occur
        # a = np.log10(100) + (self.b * 5)
        # upper_bound = list(10 ** (a - (self.b * self.Mw)))  # Upperbound based on b-value, but 100 5Mw events a year
        upper_bound = [10] * self.n_ruptures
        return (lower_bound, upper_bound)

    def fitness(self, x: np.ndarray):

        # total_slip = np.matmul(self.slip, x)  # Calculate slip by multiplying slip ratrix by rupture rate vector
        # rms = np.sqrt(np.mean((self.deficit - total_slip) ** 2))  # Calculate root mean square misfit of slip
        rms = 0
        bin_rates = np.matmul(self.gr_matrix, x)
        GR_rms = np.sqrt(np.mean((bin_rates - self.GR_rate) ** 2)) # Penalise for deviating from GR-rate
        combined_rms = (rms * self.rate_weight) + (GR_rms * self.GR_weight)# Allow for variable weighting between slip deficit and GR-rate
        return np.array([combined_rms])

    # def gradient(self, x):
    #     return pg.estimate_gradient(lambda x: self.fitness(x), x)

inversion = deficitInversion(rupture_mws, b, N, rupture_bins)

# Choose optimisation algorithm
nl = pg.scipy_optimize(method="L-BFGS-B")
# Tolerance to make algorithm stop trying to improve result
nl.ftol_abs = 0.0001
# nl.ftol_rel = 1.e-4
#nl.xtol_rel = 1e-3

# Set up basin-hopping metaalgorithm
algo = pg.algorithm(uda=pg.mbh(nl, stop=15, perturb=0.99))
# algo = pg.algorithm(nl)
# Lots of output to check on progress
algo.set_verbosity(1)

pop = pg.population(prob=inversion)
# Tell population object what starting values will be
pop.push_back(initial_rates_n)
initial_fitness = inversion.fitness(initial_rates_n)
print(initial_fitness)

# Run algorithm
print(f'Inverting {inversion.n_ruptures} ruptures...')
start = time()
pop = algo.evolve(pop)
print(f'Inversion complete. Time taken: {time() - start:.2f}s')