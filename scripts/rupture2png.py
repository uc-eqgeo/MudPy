import numpy as np
import meshio
from pyproj import Transformer
from scipy.spatial import KDTree
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
import pandas as pd


def plot_2d_surface(mesh, title, color_by='total'):
    # Extract points and cells from the mesh
    points = mesh.points[:, :2]  # Assuming 2D projection (only X and Y)
    cells = mesh.cells_dict[list(mesh.cells_dict.keys())[0]]
    colors = mesh.cell_data[color_by][0]  # Get the 'total' scalar values

    # Create polygons from cells and corresponding colors
    polygons = [points[cell] for cell in cells]

    # Plot using PolyCollection
    fig, ax = plt.subplots()
    collection = PolyCollection(polygons, array=colors, cmap='viridis', edgecolor=None)
    ax.add_collection(collection)
    ax.autoscale_view()

    # Add colorbar and labels
    plt.colorbar(collection, ax=ax, orientation='vertical', label=color_by)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

mesh_folder = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks'

mesh_name = 'hik_kerk3k_with_rake.vtk'

vtk = meshio.read(f'{mesh_folder}\\{mesh_name}')
vtk = meshio.read('C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\subduction_quads\\hk_tiles.vtk')
rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures\\'

rupture_list = glob(f'{rupture_dir}\\*Mw8*rupt')
rupture_list.sort()

# Create interpolation object for mapping ruptures to the mesh
transformer = Transformer.from_crs("epsg:4326", "epsg:2193")
for rupture_file in rupture_list[::-1]:
    rupture = pd.read_csv(rupture_file, sep='\t', index_col='# No')

    patch_coords = np.zeros((rupture.shape[0], 4))
    patch_coords[:, 0] = np.arange(rupture.shape[0])
    patch_coords[:, 2], patch_coords[:, 1] = transformer.transform(rupture['lat'], rupture['lon'])
    patch_coords[:, 3] = rupture['z(km)'] * -1

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
    if 'total-slip(m)' in rupture.columns:
        total = rupture['total-slip(m)']
        ss = np.zeros_like(total)
        ds = np.zeros_like(total)
    else:
        ss = rupture['ss-slip(m)']
        ds = rupture['ds-slip(m)']
        total = np.sqrt(ss**2 + ds**2)

    rupture_mesh = meshio.Mesh(points=points, cells=[(element, cells)], cell_data={'ss': [ss], 'ds': [ds], 'total': [total]})

    # Plot the mesh as a 2D surface
    plot_2d_surface(rupture_mesh, os.path.basename(rupture_file), color_by='total')
print('Complete :)')
