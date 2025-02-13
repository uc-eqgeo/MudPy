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

rupture_dir = 'C:\\Users\\jdmcg\\Documents\\MudPy\\hikkerk\\output\\ruptures\\'
# Check to see if root is actually /mnt adjust accordingly
if not ':' in os.path.abspath(os.sep):
    root = rupture_dir.split(':')[0]
    rupture_dir = os.path.join(os.path.abspath(os.sep), 'mnt', root.lower(), rupture_dir.split(':')[1][1:])
    
run_name = 'plate70'
locking_model = True
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
else:
    rupt_name += '.'

rupt_name += '*.rupt'

print(f'Globbing {rupture_dir}{rupt_name} ...')

rupture_list = glob(f'{rupture_dir}{rupt_name}')

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
block_size = block_size if n_ruptures > block_size else n_ruptures
block_starts = np.arange(0, n_ruptures, block_size)
block_ends = block_starts + block_size
num_threads_plot = 10

n_blocks = block_starts.shape[0]
num_threads_plot = num_threads_plot if n_blocks > num_threads_plot else n_blocks

print(f"Using {num_threads_plot} threads for create rupture csv blocks...")
rupt_name = rupt_name.replace('*.rupt','').strip('.')
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
    print(f"\nCompleted {total_ruptures} ruptures! :) \n", os.path.abspath(os.path.join(rupture_dir, "..", f'{rupt_name}_df_n{n_ruptures}{rake_tag}.csv')))
