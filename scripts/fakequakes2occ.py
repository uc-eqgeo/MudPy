#%% fakequakes2occ

import os
import numpy as np
import pandas as pd

bn_dict = {1: [0.95, 16.5],
           2: [1.1, 21.5],
           3: [1.24, 27.9]}

bn_combo = 2

b, N = bn_dict[bn_combo]

fault_name = "hikkerk"
velmod = "3e10"
locking = True
NZNSHMscaling = True
uniformSlip = False
GR_inv_min = 7.0
GR_inv_max = 9.5
file_suffix = '_max9'

max_Mw = None  # Trim maximum Mw
min_Mw = None

sz = "hk" if fault_name == "hikkerk" else ""
lock = "_locking" if locking else "_nolocking"
NZNSHM = "_NZNSHMscaling" if NZNSHMscaling else ""
uniform = "_uniformSlip" if uniformSlip else ""

n_ruptures = 5000
slip_weight = 10
gr_weight = 500
norm_weight = 1
n_iterations = 5e5
n_archipeligos = 10
island = 0
rupture_csv = f'{fault_name}_{velmod}{lock}{NZNSHM}{uniform}_df_n50000.csv'
zero_rate = 1e-6

prep_occ = False
remove_zero_rates = True  # Only removed in the occ data format, not in the merged results

procdir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk"
occ_home_dir = "Z:\\McGrath\\occ-coseismic"
if 'mnt' in os.getcwd():
    drive = procdir.split(':\\')[0]
    procdir = '/mnt/' + drive.lower() + '/' + '/'.join(procdir.split(':\\')[1].split('\\'))
    drive = occ_home_dir.split(':\\')[0]
    occ_home_dir = '/mnt/' + drive.lower() + '/' + '/'.join(occ_home_dir.split(':\\')[1].split('\\'))

rupture_dir = os.path.join(procdir, 'output', f"FQ_{velmod}{lock}{uniform.replace('Slip', '')}_GR{str(GR_inv_min).replace('.', '')}-{str(GR_inv_max).replace('.', '')}{file_suffix}")
model_dir = os.path.join(procdir, 'data', 'model_info')
deficit_file = os.path.join(procdir, 'data', 'model_info', 'hk_hires.slip')
rupture_csv = os.path.abspath(os.path.join(rupture_dir, "..", rupture_csv))

norm = f"_N{int(norm_weight)}" if norm_weight is not None else ""
max_Mw_tag = f"_maxMw{float(max_Mw)}".replace('.', '-') if max_Mw is not None else ""
max_Mw_tag += f"_minMw{float(min_Mw)}".replace('.', '-') if min_Mw is not None else ""

tag = f"n{int(n_ruptures)}_S{int(slip_weight)}{norm}_GR{int(gr_weight)}_b{str(b).replace('.','-')}_N{str(N).replace('.','-')}_nIt{int(n_iterations)}"

occ_proc_dir = os.path.join(occ_home_dir, 'data', 'sz_solutions', f"FakeQuakes_{sz}_{velmod}{lock}{uniform}_{tag}{file_suffix}{max_Mw_tag}_narchi{n_archipeligos}")

# Write rupture inversion file to occ
rates = np.array([])
isl = f"inverted_rate_{island}"

# Write patch information to occ
n_ruptures *= n_archipeligos
print(f'Loading ruptures from {os.path.basename(rupture_csv)}')
ruptures_df = pd.read_csv(rupture_csv, nrows=n_ruptures)

inversion_file = os.path.join(rupture_dir, f"{tag}_archi0_inverted_ruptures.csv")
inversion_df = pd.read_csv(inversion_file, sep='\t')
archi_df = inversion_df.copy()
rates = np.hstack([rates, inversion_df[isl].values])

print(f'Reading from {os.path.basename(rupture_dir)}/{tag}')
for a in range(1, n_archipeligos):
    print(f"Reading archipeligo {a}")
    inversion_file = os.path.join(rupture_dir, f"{tag}_archi{a}_inverted_ruptures.csv")
    inversion_df = pd.read_csv(inversion_file, sep='\t')
    rates = np.hstack([rates, inversion_df[isl].values])
    archi_df = pd.concat([archi_df, inversion_df], axis=0)

archi_df = archi_df.rename({'Unnamed: 0' : ''}, axis=1)
archi_df['inverted_rate_0'] = archi_df['inverted_rate_0'].apply(lambda x: (x / n_archipeligos))
if max_Mw is not None:
    rates[archi_df['Mw'] > max_Mw] = 0
    archi_df.loc[archi_df['Mw'] > max_Mw, 'inverted_rate_0'] = 0
if min_Mw is not None:
    rates[archi_df['Mw'] < min_Mw] = 0
    archi_df.loc[archi_df['Mw'] < min_Mw, 'inverted_rate_0'] = 0
archi_df.to_csv(os.path.join(rupture_dir, f"{tag}{max_Mw_tag}_archi-merged_inverted_ruptures.csv"), sep='\t', index=False)

deficit = np.genfromtxt(deficit_file)
deficit = deficit[:, 9]  # d in d=Gm, keep in mm/yr

n_patches = deficit.shape[0]
i0, i1 = ruptures_df.columns.get_loc('0'), ruptures_df.columns.get_loc(str(n_patches - 1)) + 1
slip_array = ruptures_df.iloc[:, i0:i1].values.T * 1000  # convert to mm
#archi_df['inverted_rate_0'] = np.where(archi_df['Mw'] > 9.2, 0, archi_df['inverted_rate_0']) # Test impact of removing largest ruptures
#archi_df['inverted_rate_0'] = np.where(archi_df['inverted_rate_0'] < 1e-5 0, archi_df['inverted_rate_0']) # Test impact of lowest rupture_rates

inverted_slip = np.matmul(slip_array, archi_df['inverted_rate_0'].values)

inv_results = pd.read_csv(os.path.join(rupture_dir, f"{tag}_archi0_inversion_results.inv"), sep='\t')
inv_results['inverted-deficit(mm/yr)'] = inverted_slip
inv_results['misfit_rel(mm/yr)'] = inv_results['inverted-deficit(mm/yr)'] / deficit
inv_results['misfit_abs(mm/yr)'] = inv_results['inverted-deficit(mm/yr)'] - deficit
inv_results.to_csv(os.path.join(rupture_dir, f"{tag}{max_Mw_tag}_archi-merged_inversion_results.inv"), sep='\t', index=False)

print(f"Written Merged Dataset {tag}{max_Mw_tag}_archi-merged_inversion_results.inv")

rates[rates < zero_rate] = 0
rates /= n_archipeligos

if remove_zero_rates:
    rupture_ix = np.where(rates > 0)[0]
    print('Not writing zero rates')
    rates = rates[rupture_ix]
    n_ruptures = len(rates)
    ruptures_df = ruptures_df.iloc[rupture_ix, :]
else:
    rupture_ix = np.arange(n_ruptures)

# %% Create average_slip.csv
if prep_occ:
    print(f'\nWriting OCC data to {os.path.basename(occ_proc_dir)}')
    for folder in ['ruptures', 'solution']:
        if not os.path.exists(os.path.join(occ_proc_dir, folder)):
            os.makedirs(os.path.join(occ_proc_dir, folder), exist_ok=True)
    del archi_df, inv_results
    print('Writing rates.csv')
    occ_dict = {"Rupture Index": rupture_ix, "Annual Rate": rates}
    occ_df = pd.DataFrame(occ_dict)
    occ_df.to_csv(os.path.join(occ_proc_dir, 'solution', 'rates.csv'), index=False)

# %% Create average_slip.csv
if prep_occ:
    print('Writing average_slips.csv')
    slip_df = ruptures_df.copy()
    for col in ['rupt_id', 'mw', 'target_mw']:
        del slip_df[col]

    av_slip = slip_df.copy()
    av_slip[av_slip == 0] = np.nan
    av_slip = np.nanmean(av_slip, axis=1)

    av_dict = {'Rupture Index': [ix for ix in slip_df.index],
                'Average Slip (m)': av_slip}
    av_df = pd.DataFrame(av_dict, columns=['Rupture Index', 'Average Slip (m)'])
    av_df.set_index('Rupture Index', inplace=True)
    av_df.to_csv(os.path.join(occ_proc_dir, 'ruptures', 'average_slips_trim.csv'), sep=',', index=True)

    slip_df.index.names = ['Rupture Index']
    slip_df = pd.concat([av_df, slip_df], axis=1)
    slip_df.to_csv(os.path.join(occ_proc_dir, 'ruptures', 'average_slips.csv'), sep=',', index=True)

# %% Create indicies.csv
if prep_occ:
    print('Writing indices.csv')
    patches = ruptures_df.copy()
    for col in ['rupt_id', 'mw', 'target_mw']:
        del patches[col]
    patches = (np.array(patches) > 0).astype(int)

    n_patches = patches.sum(axis=1)

    columns = ['Rupture Index', 'Num Sections'] + [f"# {ix}" for ix in range(1, n_patches.max() + 1)]

    with open(os.path.join(occ_proc_dir, 'ruptures', 'indices.csv'), 'w') as f:
        f.write(','.join(columns) + '\n')
        for n_rupt, rupt_ix in enumerate(rupture_ix):
            f.write(f"{rupt_ix},{n_patches[n_rupt]}," + ','.join([str(jj) for jj, kk in enumerate(patches[n_rupt, :]) if kk == 1]) + '\n')

