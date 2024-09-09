# %%  Allow script to run as jupyter notebook
"""
This script will compare inverted islands to search for global minimums
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# %% Load data
rupture_df = pd.read_csv(rupture_csv)
rupture_df = rupture_df.iloc[:n_ruptures]

inv_df = pd.read_csv(inv_file, sep='\t', index_col=0)
# %%
rupture_rates = inv_df.sort_values(by=['Mw']).drop(['initial_rate', 'target_rate', 'lower', 'upper'], axis=1)
rupture_rates['average'] = rupture_rates.iloc[:, 1:].apply(lambda x: np.log10(x)).median(axis=1)
rupture_rates['std'] = rupture_rates.iloc[:, 1:].apply(lambda x: np.log10(x)).std(axis=1)
# %%
plt.errorbar(rupture_rates['Mw'], rupture_rates['average'], yerr=rupture_rates['std'], markersize=1,
             marker='o', zorder=0)
plt.scatter(rupture_rates['Mw'], rupture_rates['average'], s=1, color='red', marker='o', label='Mean')
plt.scatter(rupture_rates['Mw'], rupture_rates['inverted_rate_0'].apply(lambda x: np.log10(x)), s=1,
            color='black', marker='o', label='Best')
plt.show()
# %%


