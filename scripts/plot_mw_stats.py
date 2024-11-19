# %%
"""
Created on Fri Aug  9 16:08:52 2024

@author: jmc753
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

locking = True
NZSHM = True
force_Mw = False

if locking:
    tag = 'locking'
else:
    tag = 'nolocking'

if NZSHM:
    tag += '_NZNSHM'
else:
    tag += '_noNZNSHM'

if force_Mw:
    tag += '_forceMw'

logs = glob(f"Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires\\output\\ruptures\\hikkerk3D_{tag}*.log")
logs.sort()
stats_csv = os.path.abspath(os.path.join(os.path.dirname(logs[0]), '..', 'rupture_stats.csv'))

remake_csv = False
min_mw = 6
max_mw = 10
if os.path.exists(stats_csv) and not remake_csv:
    mw_df = pd.read_csv(stats_csv)
else:
    length = []
    width = []
    target = []
    actual = []
    lon = []
    lat = []
    z = []
    clon = []
    clat = []
    cz = []
    requested = 100  # Original request per mag bin
    for ix, log in enumerate(logs):
        print(f"Processing {ix + 1}/{len(logs)}", end='\r')
        if float(log.split('Mw')[1].split('_')[0].replace('-','.')) < min_mw:
            continue
        elif float(log.split('Mw')[1].split('_')[0].replace('-','.')) > max_mw:
            continue
        else:
            target.append(float(log.split('Mw')[1].split('_')[0].replace('-','.')))
        with open(log) as fid:
            lines = fid.readlines()
            data_lines = [lines[ix] for ix in [11, 12, 16, 17, 19]]
            for line in data_lines:
                if 'length Lmax' in line:
                    length.append(float(line.split()[-2]))
                if 'width Wmax' in line:
                    width.append(float(line.split()[-2]))
                if 'Actual magnitude' in line:
                    actual.append(float(line.strip('\n').split()[-1]))
                if 'Hypocenter (lon,lat,z[km])' in line:
                    lon_tmp, lat_tmp, z_tmp = line.strip('\n').split()[-1].split(',')
                    if float(lon_tmp.strip('()')) < 0:
                        lon.append(float(lon_tmp.strip('()')) + 360)
                    else:
                        lon.append(float(lon_tmp.strip('()')))
                    lat.append(float(lat_tmp.strip('()')))
                    z.append(float(z_tmp.strip('()')))
                if 'Centroid (lon,lat,z[km])' in line:
                    lon_tmp, lat_tmp, z_tmp = line.strip('\n').split()[-1].split(',')
                    if float(lon_tmp.strip('()')) < 0:
                        clon.append(float(lon_tmp.strip('()')) + 360)
                    else:
                        clon.append(float(lon_tmp.strip('()')))
                    clat.append(float(lat_tmp.strip('()')))
                    cz.append(float(z_tmp.strip('()')))

    actual = np.array(actual)
    order = np.argsort(actual)
    length = np.array(length)
    width = np.array(width)
    lon = np.array(lon)
    lat = np.array(lat)
    z = np.array(z)
    clon = np.array(clon)
    clat = np.array(clat)
    cz = np.array(cz)

    target = np.array(target)

    mw_dict = {'actual': actual,
            'order': order,
            'length': length,
            'width': width,
            'lon': lon,
            'lat': lat,
            'z': z,
            'clon': clon,
            'clat': clat,
            'cz': cz,
            'target': target}

    mw_df = pd.DataFrame(mw_dict)
    mw_df.to_csv(stats_csv, index=False)


min_mw = np.floor(min(mw_df[['actual', 'target']].min()) * 20) / 20
max_mw = np.ceil(max(mw_df[['actual', 'target']].max()) * 20) / 20
step_mw = np.round(np.min(np.diff(np.unique(mw_df['target']))), 4)
sample_mw = np.round(np.arange(min_mw, max_mw, step_mw), 4)
# %%
rupt_ids = np.array([[np.round(float(log.split('Mw')[1].split('_')[0].replace('-','.')), 4), int(log.split('_')[-1].split('.')[0])] for log in logs])
requested = np.zeros_like(sample_mw)
for ix, mag in enumerate(sample_mw):
    if np.isin(rupt_ids[:, 0], mag).any():
        requested[ix] = int(np.max(rupt_ids[np.isin(rupt_ids[:, 0], mag), 1])) + 1
    else:
        requested[ix] = 0
# %%  Target vs Actual Scatter
# plt.scatter(target, actual)
sns.histplot(mw_df, x='target', y='actual', binwidth=0.05, binrange=[sample_mw[0], sample_mw[-1]])
plt.plot(sample_mw, sample_mw, color='red')
plt.xlabel('Target Mw')
plt.ylabel('Actual Mw')
plt.title(f"{tag} ({len(logs)} ruptures)")
plt.xlim([6, 9.5])
plt.ylim([6, 9.5])
plt.show()

# %% Histplot Actual vs Area
sns.histplot(mw_df, x='actual', y=np.log10(mw_df['length'] * mw_df['width']), binwidth=step_mw, cmap='viridis')
plt.plot(sample_mw, sample_mw - 4.0, color='red', linestyle=':', lw=1, label='NZ NSHM relation')
plt.plot(sample_mw, sample_mw - 3.6, color='red', linestyle=':', lw=0.5)
plt.plot(sample_mw, sample_mw - 4.1, color='red', linestyle=':', lw=0.5)
plt.xlabel('Mw')
plt.ylabel('log10(Area)')
# plt.legend()
plt.title(f"{tag} ({len(logs)} ruptures)")
plt.xlim([6, 9.5])
plt.ylim([1.5, 5.5])
plt.show()

# %% Scatter Actual vs Area
sns.scatterplot(mw_df, x='actual', y=np.log10(mw_df['length'] * mw_df['width']), hue='target', s=10, markers=False)
plt.plot(sample_mw, sample_mw - 4.0, color='red', linestyle=':', lw=1, label='NZ NSHM relation')
plt.plot(sample_mw, sample_mw - 3.6, color='red', linestyle=':', lw=0.5)
plt.plot(sample_mw, sample_mw - 4.1, color='red', linestyle=':', lw=0.5)
plt.xlabel('Mw')
plt.ylabel('log10(Area)')
# plt.legend()
plt.title(f"{tag} ({len(logs)} ruptures)")
plt.xlim([6, 9.5])
plt.ylim([1.5, 5.5])
plt.show()

# %% Histplot Target and Actual Mw
sns.histplot(mw_df['target'], binwidth=step_mw-1e-4, binrange=[sample_mw[0], sample_mw[-1]], label='Target')
sns.histplot(mw_df['actual'], binwidth=step_mw-1e-4, binrange=[sample_mw[0], sample_mw[-1]], label='Actual')
plt.plot(sample_mw, requested, color='red')
#plt.axhline(requested)
plt.axvline(mw_df['target'].min())
plt.axvline(mw_df['target'].max())
plt.legend()
plt.show()

# %% Hisplot Hypocenter
sns.histplot(mw_df, x='lon', y='lat', binwidth=0.1, cmap='flare_r')
plt.title('Hypocentre')
plt.show()

# %% Incremental Hypocenters
mw_bin_size = 1.
Mw_round = np.round(mw_df['actual'] * (1 / mw_bin_size), 0) / (1 / mw_bin_size)

for mag in np.unique(Mw_round):
    ix = np.where(Mw_round == mag)[0]
    sns.histplot(x=mw_df.loc[mw_df['actual'].round() == mag, 'lon'], y=mw_df.loc[mw_df['actual'].round() == mag, 'lat'], binwidth=0.1, cmap='flare_r')
    plt.title(f'Hypocentre {mag}Mw')
    plt.xlim([172, 186])
    plt.ylim([-43, -23])
    plt.show()

