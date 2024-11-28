import numpy as np
from glob import glob

rupture_dir = 'Z:/McGrath/HikurangiFakeQuakes/hikkerk3D_hires/output/ruptures'
rupt_name = 'hikkerk3D_locking_NZNSHMscaling'

rupt_list = glob(rupture_dir + '/' + rupt_name + '*.rupt')
rupt_list.sort()

min_mw = 6.5
max_mw = 9.5
mw_step = 0.01

n_task_arrays = (max_mw - min_mw) / mw_step + 1

# Key: Mw bin to start new properties at
# n_rupts: Number of ruptures to generate in these bins
# per task: Number of ruptures to be generated in each magnitude bin per task array
# array_step: change in magnitude that will be covered in a single task array
rupt_dict = {6.5: {'n_rupts': 150, 'per_task': 150, 'array_step': 1},
             7.0: {'n_rupts': 250, 'per_task': 250, 'array_step': 0.25},
             8.0: {'n_rupts': 125, 'per_task': 125, 'array_step': 0.1},
             9.0: {'n_rupts': 100, 'per_task': 100, 'array_step': 0.01},
             9.2: {'n_rupts': 100, 'per_task': 50, 'array_step': 0.01},
             9.3: {'n_rupts': 100, 'per_task': 25, 'array_step': 0.01},
             9.4: {'n_rupts': 100, 'per_task': 10, 'array_step': 0.01}}

rupt_dict = dict(sorted(rupt_dict.items()))
bins = np.array(list(rupt_dict.keys()))

array_bins = np.array([])
if min_mw < bins[0]:
    array_bins = np.hstack([array_bins, np.arange(min_mw, bins[0], rupt_dict[bins[0]]['array_step'])])

for ix, bin in enumerate(bins[:-1]):
    if max_mw > bin:
        array_bins = np.hstack([array_bins, np.arange(bin, bins[ix + 1], rupt_dict[bin]['array_step'])])

if max_mw > bins[-1]:
    array_bins = np.hstack([array_bins, np.arange(bins[-1], max_mw, rupt_dict[bins[-1]]['array_step'])])

array_bins = np.unique(np.round(np.append(array_bins, max_mw), 4))
array_bins = array_bins[np.where(array_bins >= min_mw)[0][0]:]

array_file = 'task_arrays.txt'
f=open(array_file,'w')

task_n = 0
for ix, mw in enumerate(array_bins[:-1]):
    rupt_bin = bins[np.where(mw >= bins)[0][-1]]
    n_rupt = rupt_dict[rupt_bin]['n_rupts']
    n_task = rupt_dict[rupt_bin]['per_task']
    n_task = min(n_task, n_rupt)
    rupt_num = np.arange(0, n_rupt + n_task, n_task)
    for ix2, n_start in enumerate(rupt_num[:-1]):
        n_end = rupt_num[ix2 + 1]
        check_list = []
        # Rounding nonsense due to numpy floating point errors
        for mwn in np.arange(round(mw / mw_step), round(array_bins[ix + 1] / mw_step), round(mw_step / mw_step)):
            mws = f"{mwn * mw_step:.2f}" 
            check_list += [f"{rupture_dir}\\{rupt_name}.Mw{mws.replace('.','-')}_{str(n).rjust(6,'0')}.rupt" for n in range(n_start, n_end)]
            total_rupts = len(set(check_list))
            yet_to_make = len(set(check_list) - set(rupt_list))
        if yet_to_make == 0:
            continue
        f.write(f"{task_n},{mw:.2f},{array_bins[ix + 1]:.2f},{mw_step},{n_start},{n_end},{yet_to_make}/{total_rupts}\n")
        task_n += 1

f.close()