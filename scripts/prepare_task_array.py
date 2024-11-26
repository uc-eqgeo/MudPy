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

n_rupts_bins = np.array([6.5, 7.0, 8.0, 9.0])
n_rupts_total = [150, 250, 125, 100]
rupts_per_task =[150, 250, 125, 10]


array_bins = np.array([])
if min_mw < 7:
    array_bins = np.hstack([array_bins, np.arange(min_mw, max_mw, 1)])

if max_mw > 7:
    array_bins = np.hstack([array_bins, np.arange(7, max_mw, 0.25)])

if max_mw > 8:
    array_bins = np.hstack([array_bins, np.arange(8, max_mw, 0.1)])

if max_mw > 9:
    array_bins = np.hstack([array_bins, np.arange(9, max_mw, 0.01)])

array_bins = np.unique(np.round(np.append(array_bins, max_mw), 4))
array_bins = array_bins[np.where(array_bins >= min_mw)[0][0]:]

array_file = 'task_arrays.txt'
f=open(array_file,'w')

task_n = 0
for ix, mw in enumerate(array_bins[:-1]):
    n_rupt = n_rupts_total[np.where(mw >= n_rupts_bins)[0][-1]]
    n_task = rupts_per_task[np.where(mw >= n_rupts_bins)[0][-1]]
    rupt_num = np.arange(0, n_rupt + n_task, n_task)
    for ix2, n_start in enumerate(rupt_num[:-1]):
        n_end = rupt_num[ix2 + 1]
        check_list = []
        for mwn in np.arange(mw, array_bins[ix + 1], mw_step):
            mwn = f"{mwn:.2f}"
            check_list += [f"{rupture_dir}\\{rupt_name}.Mw{mwn.replace('.','-')}_{str(n).rjust(6,'0')}.rupt" for n in range(n_start, n_end)]
            yet_to_make = len(set(check_list) - set(rupt_list))
        if yet_to_make == 0:
            continue
        f.write(f"{task_n},{mw:.2f},{array_bins[ix + 1]:.2f},{mw_step},{n_start},{n_end},{yet_to_make}\n")
        task_n += 1

f.close()