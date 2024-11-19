#%% fakequakes2occ

import os
import numpy as np
import pandas as pd

b, N = 1.1, 21.5

n_ruptures = 5000
slip_weight = 1
gr_weight = 10
norm_weight = 0
n_iterations = 500000
n_archipeligos = 1
island = 0
rupture_csv = 'rupture_df_n17000.csv'
zero_rate = 1e-6

prep_occ = True

rupture_dir = "C:\\Users\\jmc753\Work\\MudPy\\cluster_processing\\output\\hires_rupt"
model_dir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\data\\model_info"
occ_home_dir = "C:\\Users\\jmc753\\Work\\occ-coseismic"
deficit_file = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\data\\model_info\\slip_deficit_trenchlock.slip"
rupture_csv = os.path.abspath(os.path.join(rupture_dir, "..", rupture_csv))

if norm_weight is not None:
    tag = f"n{int(n_ruptures)}_S{int(slip_weight)}_N{int(norm_weight)}_GR{int(gr_weight)}_nIt{int(n_iterations)}"
    inversion_file = os.path.join(rupture_dir, tag + "_inverted_ruptures.csv")
else:
    tag = f"n{int(n_ruptures)}_S{int(slip_weight)}_GR{int(gr_weight)}_nIt{int(n_iterations)}"

occ_proc_dir = os.path.join(occ_home_dir, 'data', 'sz_solutions', f"FakeQuakes_sz_b{str(b).replace('.', '')}_N{str(N).replace('.', '')}")
for folder in ['ruptures', 'solution']:
    if not os.path.exists(os.path.join(occ_proc_dir, folder)):
        os.makedirs(os.path.join(occ_proc_dir, folder), exist_ok=True)

# Write rupture inversion file to occ
rates = np.array([])
isl = f"inverted_rate_{island}"

# Write patch information to occ
n_ruptures *= n_archipeligos
ruptures_df = pd.read_csv(rupture_csv, nrows=n_ruptures)

inversion_file = os.path.join(rupture_dir, f"{tag}_inverted_ruptures.csv")
inversion_df = pd.read_csv(inversion_file, sep='\t')
archi_df = inversion_df.copy()
rates = np.hstack([rates, inversion_df[isl].values])

archi_df = archi_df.rename({'Unnamed: 0' : ''}, axis=1)
archi_df['inverted_rate_0'] = archi_df['inverted_rate_0'].apply(lambda x: (x / n_archipeligos))
archi_df.to_csv(os.path.join(rupture_dir, f"{tag}_archi-merged_inverted_ruptures.csv"), sep='\t', index=False)

deficit = np.genfromtxt(deficit_file)
deficit = deficit[:, 9]  # d in d=Gm, keep in mm/yr

n_patches = deficit.shape[0]
i0, i1 = ruptures_df.columns.get_loc('0'), ruptures_df.columns.get_loc(str(n_patches - 1)) + 1
slip_array = ruptures_df.iloc[:, i0:i1].values.T * 1000  # convert to mm

inverted_slip = np.matmul(slip_array, archi_df['inverted_rate_0'].values)

# inv_results = pd.read_csv(os.path.join(rupture_dir, f"{tag}_inversion_results.inv"), sep='\t')
# inv_results['inverted-deficit(mm/yr)'] = inverted_slip
# inv_results['misfit_rel(mm/yr)'] = inv_results['inverted-deficit(mm/yr)'] / deficit
# inv_results['misfit_abs(mm/yr)'] = inv_results['inverted-deficit(mm/yr)'] - deficit
# inv_results.to_csv(os.path.join(rupture_dir, f"{tag}_archi-merged_inversion_results.inv"), sep='\t', index=False)

rates[rates < zero_rate] = 0
rates /= n_archipeligos

if prep_occ:
    del archi_df

    occ_dict = {"Rupture Index": np.arange(n_ruptures), "Annual Rate": rates}
    occ_df = pd.DataFrame(occ_dict)
    occ_df.to_csv(os.path.join(occ_proc_dir, 'solution', 'rates.csv'), index=False)

# %% Create average_slip.csv
if prep_occ:
    slip_df = ruptures_df.copy()
    for col in ['rupt_id', 'mw', 'target']:
        del slip_df[col]

    av_slip = slip_df.copy()
    av_slip[av_slip == 0] = np.nan
    av_slip = np.nanmean(av_slip, axis=1)

    av_dict = {'Rupture Index': [ix for ix in slip_df.index],
                'Average Slip (m)': av_slip}
    av_df = pd.DataFrame(av_dict, columns=['Rupture Index', 'Average Slip (m)'])
    av_df.set_index('Rupture Index', inplace=True)

    slip_df.index.names = ['Rupture Index']
    slip_df = pd.concat([av_df, slip_df], axis=1)
    slip_df.to_csv(os.path.join(occ_proc_dir, 'ruptures', 'average_slips.csv'), sep=',', index=True)

# %% Create indicies.csv
if prep_occ:
    patches = ruptures_df.copy()
    for col in ['rupt_id', 'mw', 'target']:
        del patches[col]
    patches = (np.array(patches) > 0).astype(int)

    n_patches = patches.sum(axis=1)

    columns = ['Rupture Index', 'Num Sections'] + [f"# {ix}" for ix in range(1, n_patches.max() + 1)]

    with open(os.path.join(occ_proc_dir, 'ruptures', 'indices.csv'), 'w') as f:
        f.write(','.join(columns) + '\n')
        for n_rupt in np.arange(n_ruptures):
            f.write(f"{n_rupt},{n_patches[n_rupt]}," + ','.join([str(jj) for jj, kk in enumerate(patches[n_rupt, :]) if kk == 1]) + '\n')

