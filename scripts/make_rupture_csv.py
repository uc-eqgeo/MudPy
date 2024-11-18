import numpy as np
from glob import glob
import pandas as pd
import os

rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures'
rupture_list = glob(f'{rupture_dir}\\hikkerk3D_locking_NZNSHMscaling*.rupt')

n_ruptures = len(rupture_list)  # Number of ruptures
deficit = np.genfromtxt(rupture_list[0])
n_patches = deficit.shape[0]  # Number of patches
columns = ['mw', 'target'] + [patch for patch in range(n_patches)]
rupture_df = pd.DataFrame(columns=columns)

for i, ix in enumerate(np.random.permutation(n_ruptures)):
    rupture = pd.read_csv(rupture_list[ix], sep='\t')
    displacement = np.zeros(n_patches + 2)
    if 'total-slip(m)' in rupture.columns:
        displacement[2:] = rupture['total-slip(m)']
    else:
        displacement[1:] = (rupture['ss-slip(m)'] ** 2 + rupture['ds-slip(m)'] ** 2) ** 0.5
    with open(rupture_list[ix].replace('.rupt', '.log')) as fid:
        lines = fid.readlines()
        displacement[0] = float(lines[16].strip('\n').split()[-1])
        displacement[1] = float(lines[15].strip('\n').split()[-1])
    rupture_df.loc[i] = displacement
    if np.mod(i + 1, 1000) == 0:
        rupture_df.to_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i + 1}.csv')), index=False)
        if os.path.exists(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i - 999}.csv'))):
            os.remove(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i - 999}.csv')))
    print(f"Creating rupture dataframe... ({i + 1}/{n_ruptures})", end='\r')

rupture_df = rupture_df.sort_values('mw', ignore_ix=True)
rupture_df.to_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{n_ruptures}.csv')), index=False)