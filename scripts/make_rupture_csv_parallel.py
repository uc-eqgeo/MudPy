import numpy as np
from glob import glob
import pandas as pd
import os
from multiprocessing import Pool

def write_block(rupt_name, rupture_list, end, columns, rake=False):
    if rake:
        print(f'Writing block {end} with rake')
        rake_tag = '_rake'
    else:
        print(f'Writing block {end}')
        rake_tag = ''
    block_df = pd.DataFrame(columns=columns)
    for rupture_file in rupture_list:
        try:
            rupture = pd.read_csv(rupture_file, sep='\t')
            displacement = np.zeros(n_patches + 2)
            index = rupture_file.split('Mw')[1].split('.rupt')[0]
            if 'total-slip(m)' in rupture.columns:
                displacement[2:] = rupture['total-slip(m)']
            else:
                displacement[2:] = (rupture['ss-slip(m)'] ** 2 + rupture['ds-slip(m)'] ** 2) ** 0.5
            with open(rupture_file.replace('.rupt', '.log')) as fid:
                lines = fid.readlines()
                displacement[0] = float(lines[16].strip('\n').split()[-1])  # Actual Magnitude
                displacement[1] = float(lines[15].strip('\n').split()[-1])  # Target Magnitude
            block_df.loc[index] = displacement
            if rake and 'rake(deg)' in rupture.columns:
                rakes = np.zeros(n_patches + 2)
                rakes[1] = float(lines[16].strip('\n').split()[-1])
                rakes[0] = float(lines[15].strip('\n').split()[-1])
                rakes[2:] = rupture['rake(deg)']
                block_df.loc[index + '_rake'] = rakes
        except:
            print(f'Error in {rupture_file}')
            os.remove(rupture_file)
            os.remove(rupture_file.replace('.rupt', '.log'))

    block_df.index.name = 'rupt_id'
    block_df.to_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'{rupt_name}_df_n{end}{rake_tag}_block.csv')), header=True)

rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires\\output\\ruptures\\'
rupture_dir = '/nesi/nobackup/uc03610/jack/fakequakes/hikkerk/output/ruptures/'

run_name = 'hikkerk_3e10'
locking_model = False
NZNSHM_scaling = True
uniform_slip = False
rake = False

if locking_model:
    rupt_name = run_name + '_locking'
else:
    rupt_name = run_name + '_nolocking'

if NZNSHM_scaling:
    rupt_name += '_NZNSHMscaling'
else:
    rupt_name += '_noNZNSHMscaling'

if uniform_slip:
    rupt_name += '_uniformSlip'

print(f'Preparing {rupt_name}...')

rupture_list = glob(f'{rupture_dir}{rupt_name}.*.rupt')

if rake:
    rake_tag = '_rake'
else:
    rake_tag = ''

n_ruptures = len(rupture_list)  # Number of ruptures
rupture_list = [rupture_list[ix] for ix in np.random.permutation(n_ruptures)]
print(f"{n_ruptures} found")
deficit = np.genfromtxt(rupture_list[0])
n_patches = deficit.shape[0]  # Number of patches
columns = ['mw', 'target_mw'] + [patch for patch in range(n_patches)]
rupture_df = pd.DataFrame(columns=columns)

block_size = 1000
block_starts = np.arange(0, n_ruptures, block_size)
block_ends = block_starts + block_size
num_threads_plot = 15


if __name__ == '__main__':
    with Pool(processes=num_threads_plot) as block_pool:
        block_pool.starmap(write_block, [(rupt_name, rupture_list[start:end], end, columns, rake) for start, end in zip(block_starts, block_ends)])

    rupture_df.index.name = 'rupt_id'
    rupture_df.to_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'{rupt_name}_df_n{n_ruptures}{rake_tag}.csv')))

    total_ruptures = rupture_df.shape[0]
    for block in block_starts + block_size:
        print(f"Merging block {block}...", end='\r')
        rupture_df = pd.read_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'{rupt_name}_df_n{block}{rake_tag}_block.csv')))
        rupture_df.to_csv(os.path.abspath(os.path.join(rupture_dir, "..", f'{rupt_name}_df_n{n_ruptures}{rake_tag}.csv')), mode='a', header=False, index=False)
        total_ruptures += rupture_df.shape[0]
    print(f"\nCompleted {total_ruptures} ruptures! :) ", os.path.abspath(os.path.join(rupture_dir, "..", f'{rupt_name}_df_n{n_ruptures}{rake_tag}.csv')))
