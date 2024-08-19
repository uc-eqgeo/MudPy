# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:08:52 2024

@author: jmc753
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

# logs = glob(f"C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\output\\ruptures\\hikkerk3D_{tag}*.log")
logs = glob(f"Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures\\hikkerk3D_{tag}*.log")
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
    with open(log) as fid:
        lines = fid.readlines()
        for line in lines:
            if 'length Lmax' in line:
                length.append(float(line.split()[-2]))
            if 'width Wmax' in line:
                width.append(float(line.split()[-2]))
            if 'Target magnitude' in line:
                target.append(float(line.strip('\n').split()[-1]))
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

min_mw = np.floor(min(np.hstack([actual, target])) * 20) / 20
max_mw = np.ceil(max(np.hstack([actual, target])) * 20) / 20
sample_mw = np.arange(min_mw, max_mw, 0.01)
target_arr = np.array(target)
step_mw = np.min(np.diff(np.unique(target_arr)))

# %%
# plt.scatter(target, actual)
sns.histplot(x=target, y=actual, binwidth=0.05, binrange=[sample_mw[0], sample_mw[-1]])
plt.plot(sample_mw, sample_mw, color='red')
plt.xlabel('Target Mw')
plt.ylabel('Actual Mw')
plt.title(f"{tag} ({len(logs)} ruptures)")
plt.xlim([6, 9.5])
plt.ylim([6, 9.5])
plt.show()

# %%
# plt.scatter(actual, np.log10(length * width), label='Actual')
# plt.scatter(target, np.log10(length * width), s=25, label='Target')
sns.histplot(x=actual, y=np.log10(length * width), binwidth=step_mw, cmap='viridis')
# sns.histplot(x=target, y=np.log10(length * width), binwidth=step_mw, cmap='flare_r')
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

# %%
sns.scatterplot(x=actual, y=np.log10(length * width), hue=target, size=5, markers=False)
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


# %%
sns.histplot(actual, binwidth=step_mw, binrange=[sample_mw[0], sample_mw[-1]])
plt.axhline(requested)
plt.axvline(min(target))
plt.axvline(max(target))
plt.show()

# %%
sns.histplot(x=lon, y=lat, binwidth=0.1, cmap='flare_r')
plt.title('Hypocentre')
plt.show()

# %%
mw_bin_size = 1
Mw_round = np.round(actual * (1 / mw_bin_size), 0) / (1 / mw_bin_size)

for mag in np.unique(Mw_round):
    ix = np.where(Mw_round == mag)[0]
    sns.histplot(x=lon[ix], y=lat[ix], binwidth=0.1, cmap='flare_r')
    plt.title(f'Hypocentre {mag}Mw')
    plt.xlim([172, 186])
    plt.ylim([-43, -23])
    plt.show()
