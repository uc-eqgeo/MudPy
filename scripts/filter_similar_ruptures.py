# %% Remove similar ruptures

import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.sparse import bsr_array

n_ruptures = 15000

rupture_dir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures"

if os.path.exists(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{n_ruptures}.csv'))):
    from_csv = True
    print(f"Loading ruptures from rupture_df_n{n_ruptures}.csv...")
    ruptures_df = pd.read_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{n_ruptures}.csv')), nrows=n_ruptures)
else:
    csv_list = glob(os.path.abspath(os.path.join(rupture_dir, "..", "rupture_df_n*.csv")))
    n_rupts = [int(csv.split('_n')[-1].split('.')[0]) for csv in csv_list]
    n_rupts.sort()
    n_rupts = [n for n in n_rupts if n > n_ruptures]
    if len(n_rupts) > 0:
        print(f"Loading {n_ruptures} ruptures from rupture_df_n{n_rupts[0]}.csv...")
        ruptures_df = pd.read_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{n_rupts[0]}.csv')), nrows=n_ruptures)
    else:
        raise Exception(f"No csv files found with at least {n_ruptures} ruptures")
# %%
ruptures_df = ruptures_df.sort_values(by="mw", ascending=False, ignore_index=True)

# %%

similar_dict = {}
dissimilar_dict = {}
thresh = 0.01  # 1cm
for ix, ref in ruptures_df.iterrows():
    similar_dict[ref['rupt_id']] = []
    dissimilar_dict[ref['rupt_id']] = []
    ref_array = np.array(ref)[3:]
    for ix2, comp in ruptures_df.iloc[ix + 1:].iterrows():
        print(ix, ix2, end='\r')
        if comp['mw'] >= (ref['mw'] - 0.1):  # Only compare similar magnitudes
            comp_array = np.array(comp)[3:]
            slip_patches = ((ref_array + comp_array) != 0)
            similarity = np.sqrt(np.mean((ref_array[slip_patches] - comp_array[slip_patches]) ** 2))
            if similarity < thresh:
                similar_dict[ref['rupt_id']].append((comp['rupt_id'], similarity))
            else:
                dissimilar_dict[ref['rupt_id']].append((comp['rupt_id'], similarity))
        else:
            break

# %%
