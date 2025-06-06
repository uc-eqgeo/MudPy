#%% fakequakes2occ

import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(__file__))
from helper_scripts import get_inv_results_tag, get_rupture_df_name, get_occ_directory

bn_dict = {1: [0.95, 16.5],
           2: [1.1, 21.5],
           3: [1.24, 27.9]}

bn_combo = 2

b, N = bn_dict[bn_combo]

fault_name = "hikkerm"
inversion_name = "hk_lock_nrupt_dynamic"  # Name of inversion folder
velmod = "wuatom"  # Velocity/Rigidity deficit model
deficit_model = "lock"  # Slip deficit model
locking = False  # Was locking used in the rupture generation
NZNSHMscaling = True # Was NZNSHM scaling used in the rupture generation
uniformSlip = False # Was uniform slip used in the rupture generation
tapered_gr = True  # Was tapered MFD used in the inversion (Rollins and Avouac 2019)
taper_max_Mw = 9.5  # Max Mw used for the tapered MFD
alpha_s = 1  # Tapered MFD alpha value
max_patch = 6233  # Patch number of max ROI (-1 for all patches)
file_suffix = ''  # Suffix for the FQ directory
occ_suffix = ''  # Suffix for the OCC directory
old_format = False  # Use old version of the rupture_df_file name

max_Mw = None  # Trim maximum Mw
min_Mw = None

sz = "hk" if fault_name == "hikkerm" else ""

n_ruptures = 5000
slip_weight = 10
norm_weight = 1
GR_weight = 500
nrupt_weight = 1
nrupt_cuttoff = -6
n_iterations = 500000
b, N = bn_dict[bn_combo]
n_archipeligos = 5
island = 0
zero_rate = 1e-6

prep_occ = True
remove_zero_rates = True  # Only removed in the occ data format, not in the merged results

procdir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerm"
occ_home_dir = "Z:\\McGrath\\occ-coseismic"
if 'mnt' in os.getcwd():
    drive = procdir.split(':\\')[0]
    procdir = '/mnt/' + drive.lower() + '/' + '/'.join(procdir.split(':\\')[1].split('\\'))
    drive = occ_home_dir.split(':\\')[0]
    occ_home_dir = '/mnt/' + drive.lower() + '/' + '/'.join(occ_home_dir.split(':\\')[1].split('\\'))

rupture_dir = os.path.join(procdir, 'output', inversion_name)
model_dir = os.path.join(procdir, 'data', 'model_info')
deficit_file = os.path.join(procdir, 'data', 'model_info', f'hk_{deficit_model}.slip')
rupture_csv = get_rupture_df_name(fault_id=fault_name, deficit_mod=deficit_model, velmod=velmod, rupt_lock=locking, NZNSHMscaling=NZNSHMscaling, uniformSlip=uniformSlip, old_format=old_format)
rupture_csv = os.path.abspath(os.path.join(rupture_dir, "..", rupture_csv))

max_Mw_tag = f"_maxMw{float(max_Mw)}".replace('.', '-') if max_Mw is not None else ""
max_Mw_tag += f"_minMw{float(min_Mw)}".replace('.', '-') if min_Mw is not None else ""

rupture_df_file = get_rupture_df_name(fault_id=fault_name, deficit_mod=deficit_model, velmod=velmod, rupt_lock=locking, NZNSHMscaling=NZNSHMscaling, uniformSlip=uniformSlip, old_format=old_format)
tag = get_inv_results_tag(n_ruptures=n_ruptures, slip_weight=slip_weight, GR_weight=GR_weight,
                          norm_weight=norm_weight, nrupt_weight=nrupt_weight, nrupt_cuttoff=nrupt_cuttoff,
                          taper_max_Mw=taper_max_Mw, alpha_s= alpha_s, b=b, N=N, pMax=max_patch,
                          max_iter=n_iterations)


occ_proc_dir = get_occ_directory(tag=tag, sz=sz, velmod=velmod, deficit=deficit_model, n_archipeligos=n_archipeligos, max_Mw_tag=max_Mw_tag, occ_suffix=occ_suffix)
occ_proc_dir = os.path.join(occ_home_dir, 'data', 'sz_solutions', occ_proc_dir)

# Write rupture inversion file to occ
rates = np.array([])
isl = f"inverted_rate_{island}"

# Write patch information to occ
n_ruptures *= n_archipeligos
print(f'Loading {n_ruptures} ruptures from {os.path.basename(rupture_csv)}')
ruptures_df = pd.read_csv(rupture_csv, nrows=n_ruptures, index_col=0)

inversion_file = os.path.join(rupture_dir, f"{tag}_archi0_inverted_ruptures.csv")
inversion_df = pd.read_csv(inversion_file, sep='\t', index_col=0)
archi_df = inversion_df.copy()
rates = np.hstack([rates, inversion_df[isl].values])

print(f'Reading from {os.path.basename(rupture_dir)}/{tag}')
for a in range(1, n_archipeligos):
    print(f"Reading archipeligo {a}")
    inversion_file = os.path.join(rupture_dir, f"{tag}_archi{a}_inverted_ruptures.csv")
    inversion_df = pd.read_csv(inversion_file, sep='\t', index_col=0)
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
ruptures_df = ruptures_df.loc[archi_df.index]
slip_array = ruptures_df.iloc[:, i0:i1].values.T * 1000  # convert to mm

inverted_slip = np.matmul(slip_array, archi_df['inverted_rate_0'].values)

inv_results = pd.read_csv(os.path.join(rupture_dir, f"{tag}_archi0_inversion_results.inv"), sep='\t')
inv_results['inverted-deficit(mm/yr)'] = inverted_slip
inv_results['misfit_rel'] = inv_results['inverted-deficit(mm/yr)'] / deficit
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
        if col in slip_df.columns:
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
        if col in slip_df.columns:
            del slip_df[col]
    patches = (np.array(patches) > 0).astype(int)

    n_patches = patches.sum(axis=1)

    columns = ['Rupture Index', 'Num Sections'] + [f"# {ix}" for ix in range(1, n_patches.max() + 1)]

    with open(os.path.join(occ_proc_dir, 'ruptures', 'indices.csv'), 'w') as f:
        f.write(','.join(columns) + '\n')
        for n_rupt, rupt_ix in enumerate(rupture_ix):
            f.write(f"{rupt_ix},{n_patches[n_rupt]}," + ','.join([str(jj) for jj, kk in enumerate(patches[n_rupt, :]) if kk == 1]) + '\n')
