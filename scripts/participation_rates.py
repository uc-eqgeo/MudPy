# %%  Allow script to run as jupyter notebook
"""
This script will calculate the rupture participation rates for each patch in the fault model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define directories
inversion_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk\\output\\'

velmod = 'prem'
locking = True
NZSHM = True
uniformSlip = False
force_Mw = False

if locking:
    tag = velmod + '_locking'
else:
    tag = velmod + '_nolocking'

if NZSHM:
    tag += '_NZNSHMScaling'
else:
    tag += '_noNZNSHMScaling'

if uniformSlip:
    tag += '_uniformSlip'
else:
    tag += ''

if force_Mw:
    tag += '_forceMw'

inversion_dir += 'FQ_' + tag.replace('uniformSlip', 'uniform').replace('_NZNSHMScaling', '').replace('_noNZNSHMScaling', '') + '_GR70-90'

# Define flags for results csv
n_ruptures = 5000
slip_weight = 10
norm_weight = 1
gr_weight = 500
n_its = 5e5
archi = '-merged'
islands = 10
b, N = 1.1, 21.5

# %% No user inputs below here
# Create filepaths
rupture_csv = os.path.join(inversion_dir, '..', f'hikkerk_{tag}_df_n50000.csv')  # CSV containing rupture slips
if norm_weight is not None:
    inv_file = f"n{int(n_ruptures)}_S{int(slip_weight)}_N{int(norm_weight)}_GR{int(gr_weight)}_b{str(b).replace('.','-')}_N{str(N).replace('.','-')}_nIt{int(n_its)}_archi{archi}_inverted_ruptures.csv"
else:
    inv_file = f"n{n_ruptures}_S{slip_weight}_GR{gr_weight}_nIt{n_its}_inverted_ruptures.csv"
inv_file = os.path.join(inversion_dir, inv_file)
patch_file = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk\\data\\model_info\\hk.fault'

inv_df = pd.read_csv(inv_file, sep='\t')

# Load data
n_ruptures = inv_df.shape[0]

rupture_df = pd.read_csv(rupture_csv, nrows=n_ruptures)

#%% Prepare for calculating patch participations
patch_xy = np.genfromtxt(os.path.abspath(patch_file))
patch_xy[:, 1] = np.mod(patch_xy[:, 1], 360)

patch_ids = rupture_df.columns.to_list()[3:]

# Create a grid to record the rupture rate of each rupture at each patch
participation_grid = np.array(rupture_df.iloc[:, 3:] > 0, dtype=np.float64)  # Identify which ruptures effect each patch
participation_grid *= np.array(inv_df['inverted_rate_0']).reshape(n_ruptures, 1)  # Get patch rupture rates

# Get Magnitude limits for GR and plotting
mw_ints = np.arange(np.floor(inv_df['Mw'].min()), np.ceil(inv_df['Mw'].max()))
mw_bins = np.round(np.arange(mw_ints[0], np.floor(inv_df['Mw'].max() * 10)/10, 0.1), 2)

# Create a dataframe to store N value at each Mw for each patch
gr_df = pd.DataFrame(columns=patch_ids, index=mw_bins)

# Create a matrix to calculate GR rates
gr_matrix = np.zeros((len(mw_bins), n_ruptures)).astype('bool')  # Make an I-J matrix for calculating GR rates, where I is number of magnitude bins and J is number of ruptures
for ix, bin in enumerate(mw_bins):
    gr_matrix[ix, :] = (rupture_df['mw'] >= bin)
gr_matrix = gr_matrix.astype('int')
# Calculate GR rates for each patch
for patch in patch_ids:
    print(f"Calculating GR rates for {patch}/{patch_ids[-1]}", end='\r')
    patch_participation = participation_grid[:, patch_ids.index(patch)]
    gr_df[patch] = gr_matrix @ patch_participation

# %% Plot results
max_mw = np.max(mw_bins)
mw_limits = []
for mw in mw_ints[:-1]:
    mw_limits.append((mw, max_mw))  # True N-value
    mw_limits.append((mw, mw + 1))  # Incremental N-value
mw_limits.append((mw_ints[-1], max_mw))

for min_mw, max_mw in mw_limits:
    plt.figure()
    plt.scatter(patch_xy[:, 1], patch_xy[:, 2], c=np.log10(gr_df.loc[min_mw] - gr_df.loc[max_mw]),
                s=0.1, marker='s', vmin=-5, vmax=0, cmap='tab20b')
    plt.title(f"GR Rate {min_mw} - {max_mw}")
    plt.colorbar()

# %%
