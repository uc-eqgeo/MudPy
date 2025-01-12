import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import geopandas as gpd
from glob import glob

# %% Prepare global variables
project_dir = 'Z:/McGrath/HikurangiFakeQuakes/hikkerk3D_hires/'
rupture_dir = project_dir + 'output/ruptures/'
fault_name = 'hk.fault'
taper_length = 1e4  # Set to zero to revert to untapered length
taper_method = 'linear'

min_mw = 8.5
max_mw = 9.5

print('Loading Fault')
fault = pd.read_csv(project_dir + '/data/model_info/' + fault_name, sep='\t').drop(0).astype(float)

print("Identifying ruptures")
rupture_list = glob(f'{rupture_dir}/*_??????.rupt')
rupture_list.sort()

# %% Loop through ruptures
for ix, rupture_file in enumerate(rupture_list[::-1]):
    rupture_id = rupture_file.split('.')[-2]

    mw = float(rupture_id.split('_')[0].strip('Mw').replace('-','.'))
    if mw < min_mw or mw >= max_mw:
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

    # Calculate distance of slip patches to no slip patches using KDTree
    zero_tree = KDTree(no_slip_patches.geometry.apply(lambda x: (x.x, x.y)).tolist())
    edge_distances = zero_tree.query(slip_patches.geometry.apply(lambda x: (x.x, x.y)).tolist())[0] - fault.loc[slip_patches.index, 'width'] / 2
    
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
