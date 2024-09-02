# %%  Allow script to run as jupyter notebook
"""
This script will calculate the rupture participation rates for each patch in the fault model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define directories
inversion_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\archi'

# Define flags for results csv
n_ruptures = 5000
slip_weight = 1
gr_weight = 10
n_its = 100000

# %% No user inputs below here
# Create filepaths
rupture_csv = os.path.join(inversion_dir, '..', 'rupture_df_n15000.csv')  # CSV containing rupture slips
inv_file = f"n{n_ruptures}_S{slip_weight}_GR{gr_weight}_nIt{n_its}_inverted_ruptures.csv"
inv_file = os.path.join(inversion_dir, inv_file)
patch_file = os.path.join(inversion_dir, '..', '..', 'data', 'model_info', 'hk.fault')

# Load data
rupture_df = pd.read_csv(rupture_csv)
rupture_df = rupture_df.iloc[:n_ruptures]

inv_df = pd.read_csv(inv_file, sep='\t')

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
