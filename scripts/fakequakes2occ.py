#%% fakequakes2occ

import os
import shutil
import numpy as np
import pandas as pd


b, N = 1.1, 21.5

n_ruptures = 5000
slip_weight = 1
gr_weight = 10
norm_weight = None
n_iterations = 1e6
island = 0
rupture_csv = 'rupture_df_n15000.csv'


rupture_dir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\archi_mini"
model_dir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\data\\model_info"
occ_home_dir = "C:\\Users\\jmc753\\occ-coseismic"
deficit_file = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\data\\model_info\\slip_deficit_trenchlock.slip"
rupture_csv = os.path.abspath(os.path.join(rupture_dir, "..", rupture_csv))



if norm_weight:
    inversion_file = os.path.join(rupture_dir, f"n{int(n_ruptures)}_S{int(slip_weight)}_N{int(norm_weight)}_GR{int(gr_weight)}_nIt{int(n_iterations)}_inverted_ruptures.csv")
else:
    inversion_file = os.path.join(rupture_dir, f"n{int(n_ruptures)}_S{int(slip_weight)}_GR{int(gr_weight)}_nIt{int(n_iterations)}_inverted_ruptures.csv")

occ_proc_dir = os.path.join(occ_home_dir, 'sz_solutions', f"FakeQuakes_sz_b{str(b).replace('.', '')}_N{str(N).replace('.', '')}")
for folder in ['ruptures', 'solution']:
    if not os.path.exists(os.path.join(occ_proc_dir, folder)):
        os.makedirs(os.path.join(occ_proc_dir, folder), exist_ok=True)

# Write rupture inversion file to occ
inversion_df = pd.read_csv(inversion_file, sep='\t')
isl = f"inverted_rate_{island}"

occ_dict = {"Rupture Index": np.arange(n_ruptures), "Annual Rate": inversion_df[isl].values}
occ_df = pd.DataFrame(occ_dict)
occ_df.to_csv(os.path.join(occ_proc_dir, 'ruptures', 'rates.csv'), sep=',', index=False)

# Write patch information to occ
ruptures_df = pd.read_csv(rupture_csv, nrows=n_ruptures)
