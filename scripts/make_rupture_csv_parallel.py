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

if '/mnt/' in os.getcwd():
    rupture_dir = '/mnt/z/McGrath/HikurangiFakeQuakes/hikkerk/output/ruptures/'
else:
    rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk\\output\\ruptures\\'
#rupture_dir = '/nesi/nobackup/uc03610/jack/fakequakes/hikkerk/output/ruptures/'

run_name = 'hikkerk_prem'
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
min_Mw = 6.5
max_Mw = 9.0

rupt_name = rupt_name.replace('*.rupt','').strip('.')
total_ruptures = len(rupture_list)  # Total number of prepared ruptures
rupture_list = [rupture_list[ix] for ix in np.random.permutation(total_ruptures) if float(rupture_list[ix].split('.Mw')[-1].split('_')[0].replace('-', '.')) < max_Mw and float(rupture_list[ix].split('.Mw')[-1].split('_')[0].replace('-', '.')) >= min_Mw]
n_ruptures = len(rupture_list)  # Number of ruptures
print(f"{total_ruptures} found, {n_ruptures} between {min_Mw}-{max_Mw}Mw")
if n_ruptures != 50000 and max_Mw < 9.5 and min_Mw > 6.5:
    raise Exception("Incorrect number of ruptures")

compiled_csv = os.path.abspath(os.path.join(rupture_dir, "..", f'{rupt_name}_df_n{n_ruptures}{rake_tag}.csv'))
check_prepared = False
if rake == False and os.path.exists(compiled_csv.replace('.csv', '_rake.csv')):
    print(f"Checking pre-prepared rupture order from {os.path.basename(compiled_csv.replace('.csv', '_rake.csv'))}")
    premade_df_file = compiled_csv.replace('.csv', '_rake.csv')
    check_prepared = True
elif rake == True and os.path.exists(compiled_csv.replace('_rake.csv', '.csv')):
    print(f"Checking pre-prepared rupture order from {os.path.basename(compiled_csv.replace('_rake.csv', '.csv'))}")
    premade_df_file = compiled_csv.replace('_rake.csv', '.csv')
    check_prepared = True

if check_prepared:
    print('\tLoading rupt_ids...')
    premade_df = pd.read_csv(premade_df_file, usecols=['rupt_id'])
    print('\tExtracting rupt_ids...')
    premade_rupture_list = [f"{rupture_dir}{rupt_name.replace('*.rupt','').strip('.')}.Mw{rupt_id}.rupt" for rupt_id in premade_df.rupt_id]
    print('\tCross checking rupt_ids...')
    premade_rupture_set = set(premade_rupture_list)
    check_rupts = [True if rupture in premade_rupture_set else False for rupture in rupture_list]
    del premade_df
    if all(check_rupts):
        print("All ruptures are in the pre-prepared df.")
        rupture_list = premade_rupture_list           
    else:
        for rupture in rupture_list:
            if rupture not in premade_rupture_set:
                print(f"Missing rupture: {rupture}")
        raise Exception("Some requested ruptures are not in the pre-prepared list.")

deficit = np.genfromtxt(rupture_list[0])
n_patches = deficit.shape[0]  # Number of patches
columns = ['mw', 'target_mw'] + [patch for patch in range(n_patches)]
rupture_df = pd.DataFrame(columns=columns)
rupture_df.index.name = 'rupt_id'

block_size = 1000
block_starts = np.arange(0, n_ruptures, block_size)
block_ends = block_starts + block_size
num_threads_plot = 10

if __name__ == '__main__':
    if not rake and check_prepared:
        print(f"Extracting from {os.path.basename(premade_df_file)}")
        try:
            print("\tLoading ruptures w/ rakes...")
            rake_df = pd.read_csv(premade_df_file)
            print(f"\tIdentifying rakes...")
            rake_ix = [ix for ix, rupt_id in enumerate(rake_df['rupt_id'].values) if 'rake' in rupt_id]
            print(f"\tDropping rakes...")
            rake_df.drop(rake_ix, inplace=True)
            total_ruptures = rake_df.shape[0]
            print(f"\tWriting {total_ruptures} ruptures...", end='\r')
            rake_df.to_csv(compiled_csv, index=False)
        except:
            print(f"\tRetry in blocks")
            rupture_df.to_csv(compiled_csv)
            total_ruptures = rupture_df.shape[0]
            for block_start in block_starts:
                print(f"Merging block {block_start}...", end='\r')
                rake_df = pd.read_csv(premade_df_file, skiprows=block_start * 2, nrows=block_size * 2, header=None)
                rake_df = rake_df.iloc[[ix for ix, row in rake_df.iterrows() if 'rake' not in row[0]]].reset_index(drop=True)
                rake_df.to_csv(compiled_csv, mode='a', header=False, index=False)
                total_ruptures += rake_df.shape[0]

    else:
        with Pool(processes=num_threads_plot) as block_pool:
            block_pool.starmap(write_block, [(rupt_name, rupture_list[start:end], end, columns, rake) for start, end in zip(block_starts, block_ends)])

        rupture_df.to_csv(compiled_csv)

        total_ruptures = rupture_df.shape[0]
        for block in block_starts + block_size:
            print(f"Merging block {block}...", end='\r')
            rupture_df = pd.read_csv(compiled_csv.replace(f"n{n_ruptures}", f"n{block}").replace(".csv", "_block.csv"))
            rupture_df.to_csv(compiled_csv, mode='a', header=False, index=False)
            total_ruptures += rupture_df.shape[0]

    print(f"\nCompleted {total_ruptures} ruptures! :) ", compiled_csv)
