import numpy as np
import meshio
from pyproj import Transformer
from scipy.spatial import KDTree
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
import pandas as pd
import sys
sys.path.append('C:/Users/jmc753/Work/RSQSim/rsqsim-python-tools/src/rsqsim_api')
from rsqsim_api.visualisation.utilities import plot_background
import pickle


def plot_2d_surface(mesh, title, rupture_png_dir, hypo, max_slip=50, color_by='total'):
    # Load background map
    fig, ax = pickle.load(open(f"{rupture_png_dir}/temp.pkl", "rb"))
    # Extract points and cells from the mesh
    points = mesh.points[:, :2]  # Assuming 2D projection (only X and Y)
    cells = mesh.cells_dict[list(mesh.cells_dict.keys())[0]]
    colors = mesh.cell_data[color_by][0]  # Get the 'total' scalar values

    # Create polygons from cells and corresponding colors
    polygons = [points[cell] for cell in cells]

    # Plot using PolyCollection
    #fig, ax = plt.subplots()
    collection = PolyCollection(polygons, array=colors, cmap='magma', edgecolor=None)
    collection.set_clim(vmax=max_slip)
    alpha = np.where(colors == 0, 0.25, 1)
    collection.set_alpha(alpha)

    ax['main_figure'].add_collection(collection)
    ax['main_figure'].autoscale_view()

    # Add colorbar and labels
    plt.colorbar(collection, ax=ax['main_figure'], orientation='vertical', label=color_by)
    plt.plot(hypo[0], hypo[1], 'ro', markersize=10)
    plt.plot(hypo[2], hypo[3], 'b+', markersize=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.savefig(os.path.join(rupture_png_dir, f'{title}.png'))
    plt.close()

mesh_folder = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks'

mesh_name = 'hik_kerk3k_with_rake.vtk'
plot_every = 1  # Plot every nth rupture

vtk = meshio.read(f'{mesh_folder}\\{mesh_name}')
vtk = meshio.read('C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\subduction_quads\\hk_tiles.vtk')
rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires\\output\\ruptures\\'
rupture_png_dir = os.path.abspath(os.path.dirname(rupture_dir) + '/..\\rupture_pngs\\')
os.makedirs(rupture_png_dir, exist_ok=True)

rupture_list = glob(f'{rupture_dir}\\*Mw9-26_000047*rupt')
rupture_list.sort()

bounds = [int(bound) for bound in '1500000/5250000/3000000/7300000'.split('/')]

new_background = False
new_pngs = True

if new_background or not os.path.exists(os.path.join(rupture_png_dir,'temp.pkl')):
    print('Plotting new background')
    background = plot_background(plot_lakes=False, bounds=bounds,
                            plot_highways=False, plot_rivers=False, hillshading_intensity=0.3,
                            pickle_name=os.path.join(rupture_png_dir,'temp.pkl'), hillshade_fine=True,
                            hillshade_kermadec=True,
                            plot_edge_label=False, figsize=(10, 10))
    new_pngs = True

# Create interpolation object for mapping ruptures to the mesh
transformer = Transformer.from_crs("epsg:4326", "epsg:2193")
for rupture_file in rupture_list[::-plot_every]:
    if os.path.exists(os.path.join(rupture_png_dir, os.path.basename(rupture_file).replace('.rupt', '.png'))) and not new_pngs:
        continue
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

    with open(rupture_file.replace('.rupt', '.log'), 'r') as fid:
        lines = fid.readlines()
        lon, lat, z = [float(coord) for coord in lines[17].replace('\n', '').split(' ')[-1].strip('()').split(',')]
        lat, lon = transformer.transform(lat, lon)
        clon, clat, z = [float(coord) for coord in lines[19].replace('\n', '').split(' ')[-1].strip('()').split(',')]
        clat, clon = transformer.transform(clat, clon)

    # Plot the mesh as a 2D surface
    plot_2d_surface(rupture_mesh, os.path.basename(rupture_file), rupture_png_dir, [lon, lat, clon, clat], max_slip=50, color_by='total')
    print(os.path.basename(rupture_file))
print('Complete :)')
