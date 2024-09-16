import numpy as np
from glob import glob
import pandas as pd
import os
import shutil

rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires\\output\\ruptures'
rupture_list = glob(f'{rupture_dir}\\hikkerk3D_locking_NZNSHMscaling*.rupt')

n_ruptures = len(rupture_list)  # Number of ruptures
deficit = np.genfromtxt(rupture_list[0])
n_patches = deficit.shape[0]  # Number of patches
columns = ['mw', 'target'] + [patch for patch in range(n_patches)]
rupture_df = pd.DataFrame(columns=columns)

for i, ix in enumerate(np.random.permutation(n_ruptures)):
    rupture = pd.read_csv(rupture_list[ix], sep='\t')
    displacement = np.zeros(n_patches + 2)
    index = rupture_list[ix].split('Mw')[1].split('.rupt')[0]
    if 'total-slip(m)' in rupture.columns:
        displacement[2:] = rupture['total-slip(m)']
    else:
        displacement[2:] = (rupture['ss-slip(m)'] ** 2 + rupture['ds-slip(m)'] ** 2) ** 0.5
    with open(rupture_list[ix].replace('.rupt', '.log')) as fid:
        lines = fid.readlines()
        displacement[1] = float(lines[16].strip('\n').split()[-1])
        displacement[0] = float(lines[15].strip('\n').split()[-1])
    rupture_df.loc[index] = displacement
    write_out = 1000
    if np.mod(i + 1, write_out) == 0:
        if os.path.exists(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i - write_out + 1}.csv'))):
            shutil.copy(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i - write_out + 1}.csv')), os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i + 1}.csv')))
            rupture_df.to_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i + 1}.csv')), mode='a', header=False)
            os.remove(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i - write_out + 1}.csv')))
        else:
            rupture_df.index.name = 'rupt_id'
            rupture_df.to_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i + 1}.csv')))
        rupture_df = pd.DataFrame(columns=columns)  # Reinitialise dataframe
    print(f"Creating rupture dataframe... ({i + 1}/{n_ruptures})", end='\r')

print(f"Completed {os.path.abspath(os.path.join(rupture_dir, "..", f'rupture_df_n{i + 1}.csv'))}! :)")