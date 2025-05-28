import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import geopandas as gpd
from glob import glob
import os
import shapely
import matplotlib.pyplot as plt

# %% Prepare global variables
project_dir = 'Z:\\McGrath/HikurangiFakeQuakes/hikkerm/'
if 'uc03610' in os.getcwd():
    project_dir = '/nesi/nobackup/uc03610/jack/fakequakes/hikkerm'

run_name = 'hikkerm'
velmod = 'wuatom'
locking_model = 'hk_lock'
NZNSHM_scaling = True
uniform_slip = False

locking_model = locking_model.replace('hk_', '')
area = '_NSHMarea' if NZNSHM_scaling else '_noNSHMArea'
uniform = '_uniformSlip' if uniform_slip else ''

rupt_name = f"{run_name}_{locking_model}_{velmod}{area}{uniform}.*.rupt"

rupture_dir = os.path.join(project_dir, 'output', 'ruptures')
fault_name = 'hk.fault'
taper_length = 1e4  # Set to zero to revert to untapered length
taper_method = 'linear'

min_mw = 7.5
max_mw = 10

# if locking_model:
#     rupt_name = run_name + '_locking'
# else:
#     rupt_name = run_name + '_nolocking'

# if NZNSHM_scaling:
#     rupt_name += '_NZNSHMscaling'
# else:
#     rupt_name += '_noNZNSHMscaling'

# if uniform_slip:
#     rupt_name += '_uniformSlip'

# rupt_name += '.*.rupt'

print('Loading Fault:', fault_name)
fault = pd.read_csv(project_dir + '/data/model_info/' + fault_name, sep='\t').drop(0).astype(float)

print(f"Identifying ruptures: {rupt_name}")
rupture_list = glob(f'{rupture_dir}/{rupt_name}')
rupture_list.sort()

# %% Loop through ruptures
for ix, rupture_file in enumerate(rupture_list[::-1]):
    rupture_id = rupture_file.split('.')[-2]

    mw = float(rupture_id.split('_')[0].strip('Mw').replace('-','.'))
    if mw < min_mw or mw > max_mw:
        print(f'\t{ix}/{len(rupture_list)}:', end='\r')
        continue

    untaper = True if int(taper_length) == 0 else False

    rupture_log = rupture_file.replace('.rupt', '.log')
    # Check if already correctly tapered, or if it must first be untapered
    with open(rupture_log, 'r') as fid:
        log = fid.readlines()
        taper_info_line = [line for line in log if 'Taper length' in line]
    if len(taper_info_line) == 1:
        previous_taper = float(taper_info_line[0].split(' ')[-2])
    else:
        previous_taper = 0

    if previous_taper == float(taper_length):
        print(f'\t{ix}/{len(rupture_list)}:', end='\r')
        continue
    else:
        if previous_taper != 0:
            untaper = True

    taper_aim = 'untapered' if int(taper_length) == 0 else f'tapered from {previous_taper} m to {taper_length} m'
    # Load rupture data
    rupt = pd.read_csv(rupture_file, sep='\t', index_col='# No')
    rupt = rupt[['lon', 'lat', 'z(km)', 'rake(deg)', 'total-slip(m)']]
    rupt_gpd = gpd.GeoDataFrame(rupt, crs='EPSG:4326', geometry=gpd.points_from_xy(rupt['lon'], rupt['lat'])).to_crs(2193)

    # Identify slip and no slip patches
    no_slip_patches = rupt_gpd[rupt_gpd['total-slip(m)'] == 0]
    slip_patches = rupt_gpd[rupt_gpd['total-slip(m)'] > 0]

    # Identify boundary patches of the rupture
    slip_patches_center = shapely.MultiPoint([(data.geometry.x, data.geometry.y) for ix, data in slip_patches.iterrows()])
    ratio = 0.02
    # Use concave hull to accurately map the rupture boundary, iteratively checking that you're not overfitting
    slip_boundary = shapely.LineString(shapely.concave_hull(slip_patches_center, ratio=ratio).exterior)
    slip_boundary_check = shapely.LineString(shapely.concave_hull(slip_patches_center, ratio=ratio + 0.01).exterior)
    convex_hull = shapely.LineString(shapely.concave_hull(slip_patches_center, ratio=1).exterior)
    while slip_boundary.length > slip_boundary_check.length * 1.1 or slip_boundary.length > convex_hull.length * 1.5:
        ratio += 0.01
        slip_boundary = shapely.LineString(shapely.concave_hull(slip_patches_center, ratio=ratio).exterior)
        slip_boundary_check = shapely.LineString(shapely.concave_hull(slip_patches_center, ratio=ratio + 0.01).exterior)
    # Calculate distance to the boundary for each patch
    edge_distances = shapely.distance(slip_boundary, slip_patches.geometry) + fault.loc[slip_patches.index, 'width'] / 2

    # plt.scatter([point.x for point in slip_patches_center.geoms], [point.y for point in slip_patches_center.geoms], c=edge_distances.values, s=50, vmin=0, vmax=taper_length, cmap='tab20c')
    # plt.plot(convex_hull.xy[0], convex_hull.xy[1], label='Convex Hull', color='black', linestyle=':')
    # plt.plot(slip_boundary_check.xy[0], slip_boundary_check.xy[1], label='Boundary check', color='black', linestyle='--')
    # plt.plot(slip_boundary.xy[0], slip_boundary.xy[1], label='Boundary', color='red')
    # plt.legend()
    # plt.colorbar()
    # plt.title(f'{os.path.basename(rupture_file)}:\n{taper_aim}, ratio={ratio:.3f}')
    # plt.show()

    # First untaper if that is needed
    if untaper:
        slip_taper = np.where(edge_distances < previous_taper, previous_taper / edge_distances, 1)
        rupt.loc[slip_patches.index, 'total-slip(m)'] = rupt.loc[slip_patches.index, 'total-slip(m)'] * slip_taper

    # Calculate and apply taper
    if taper_length != 0:
        slip_taper = np.where(edge_distances < taper_length, edge_distances / taper_length, 1)
        rupt.loc[slip_patches.index, 'total-slip(m)'] = rupt.loc[slip_patches.index, 'total-slip(m)'] * slip_taper

    # Save tapered rupture
    np.savetxt(rupture_file,rupt.reset_index().to_numpy(),fmt='%d\t%10.6f\t%10.6f\t%8.4f\t%.2f\t%5.4f',header='No\tlon\tlat\tz(km)\trake(deg)\ttotal-slip(m)')

    with open(rupture_log, 'r') as fid:
        log = fid.readlines()

    with open(rupture_log, 'w') as fid:
        write_len, write_method = True, True
        for line in log:
            if 'Taper' in line:
                if 'length' in line:
                    fid.write(f'Taper length: {int(taper_length)} m\n')
                    write_len = False
                elif 'method' in line:
                    fid.write(f'Taper method: {taper_method}\n')
                    write_method = False
            else:
                fid.write(line.strip('\n') + '\n')
        if write_len:
            fid.write(f'Taper length: {int(taper_length)} m\n')
        if write_method:
            fid.write(f'Taper method: {taper_method}\n')
    
    print(f'\t{ix}/{len(rupture_list)}: {rupture_id} {taper_aim}', end='\r')

print('\nComplete :)')
