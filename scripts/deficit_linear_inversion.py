import numpy as np

from glob import glob

rupture_dir = 'C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\output\\ruptures'

rupture_list = glob(f'{rupture_dir}\\*.rupt')

deficit_file = 'C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\data\\model_info\\slip_deficit_trenchlock.slip'

deficit = np.genfromtxt(deficit_file)
deficit = deficit[:, 9]  # d in d=Gm

G = np.zeros((deficit.shape[0], len(rupture_list)))

for ix, rupture_file in enumerate(rupture_list):
    rupture = np.genfromtxt(rupture_file)
    G[:, ix] = (rupture[:, 8] ** 2 + rupture[:, 9] ** 2) ** 0.5

m = np.linalg.lstsq(G, deficit, rcond=None)[0]

recon_d = np.dot(G, m)

print(m)