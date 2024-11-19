# %%
import numpy as np
import meshio
from pyproj import Transformer
from scipy.spatial import KDTree
from glob import glob
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

mesh_folder = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks'

mesh_name = 'hik_kerk3k_with_rake.vtk'

plot_ruptures = False
n_ruptures = 5000
inversion_name = 'Final_Jack'

file_keyword = 'archi-merged'

write_geojson = True

vtk = meshio.read(f'{mesh_folder}\\{mesh_name}')
vtk = meshio.read('C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\subduction_quads\\hk_tiles.vtk')
rupture_dir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires\\output\\ruptures"
output_dir = f"C:\\Users\\jmc753\\Work\\MudPy\\cluster_processing\\output\\{inversion_name}"

if plot_ruptures:
    rupture_list = glob(f'{rupture_dir}\\*{file_keyword}*.rupt')
else:
    rupture_list = glob(os.path.abspath(f'{output_dir}\\n{n_ruptures}*{file_keyword}*.inv'))

# Create interpolation object for mapping ruptures to the mesh
transformer = Transformer.from_crs("epsg:4326", "epsg:2193")
for rupture_file in rupture_list:
    rupture_file = os.path.abspath(rupture_file)
    rupture = pd.read_csv(rupture_file, sep='\t', index_col=0).reset_index(drop=True)
    patch_coords = np.zeros((rupture.shape[0], 4))
    patch_coords[:, 0] = rupture.index.to_numpy()
    if rupture['lon'].max() <= 180:
        patch_coords[:, 2], patch_coords[:, 1] = transformer.transform(rupture['lat'], rupture['lon'])
    else:
        patch_coords[:, 1], patch_coords[:, 2] = rupture['lat'], rupture['lon']
    if abs(rupture['z(km)'].max()) > 100:  # Convert to m, negative down
        patch_coords[:, 3] = rupture['z(km)'] * -1
    else:
        patch_coords[:, 3] = rupture['z(km)'] * -1000

    cells = vtk.cells[0].data
    n_cells = cells.shape[0]
    cell_centers = np.zeros((n_cells, 3))
    for ii in range(n_cells):
        if vtk.cells[0].data[ii, :].shape[0] == 3:
            p1, p2, p3 = vtk.cells[0].data[ii, :]
            cell_centers[ii, :] = np.mean(np.vstack([vtk.points[p1, :], vtk.points[p2, :], vtk.points[p3, :]]), axis=0)
            element = 'triangle'
            suffix = ''
        else:
            p1, p2, p3, p4 = vtk.cells[0].data[ii, :]
            cell_centers[ii, :] = np.mean(np.vstack([vtk.points[p1, :], vtk.points[p2, :], vtk.points[p3, :], vtk.points[p4, :]]), axis=0)
            element = 'polygon'
            suffix = '_rect'

    hikurangi_kd_tree = KDTree(patch_coords[:, 1:])
    _, nearest_indices = hikurangi_kd_tree.query(cell_centers)

    points = vtk.points

    ss, ds = False, False
    col_dict = {}

    for col in rupture.columns:
        if col in ['lat', 'lon', 'z(km)']:
            continue
        if len(np.unique(rupture[col])) > 1:  # Don't bother with constants
            col_dict[col] = [rupture.loc[nearest_indices, col].values]
            print('Adding', col)
            if 'ss' in col:
                ss, ss_col = True, col
            elif 'ds' in col:
                ds, ds_col = True, col

    if all([ds, ss]):
        col_dict['total'] = [np.sqrt(rupture[ss_col]**2 + rupture[ds_col]**2)]

    rupture_mesh = meshio.Mesh(points=points, cells=[(element, cells)], cell_data=col_dict)

    outfile = f"{rupture_file.replace('scaling.', 'scaling+').split('.')[0].replace('scaling+', 'scaling.')}{suffix}.vtk"
    rupture_mesh.write(outfile, file_format="vtk")
    print(f"Written {outfile}")

    if write_geojson:
        for key in col_dict.keys():
            col_dict[key] = col_dict[key][0]

        cell_poly = []
        for cell in rupture_mesh.cells[0].data:
            cell_poly.append(Polygon(rupture_mesh.points[cell]))

        col_dict["geometry"] = cell_poly
        col_dict["id"] = rupture.index.values

        gdf = gpd.GeoDataFrame(col_dict)
        gdf.set_crs(epsg=2193, inplace=True)
        gdf.to_file(outfile.replace('.vtk', '.geojson'), driver="GeoJSON")
        print(f"Written {outfile.replace('.vtk', '.geojson')}")

print('Complete :)')
