import numpy as np
import meshio
from pyproj import Transformer
from scipy.spatial import KDTree
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
import os
import pandas as pd
import geopandas as gpd
import sys
sys.path.append('C:/Users/jmc753/Work/RSQSim/rsqsim-python-tools/src/rsqsim_api')
from rsqsim_api.visualisation.utilities import plot_background
import pickle


def plot_2d_surface(mesh, title, rupture_png_dir, hypo=[], min_slip=0.1, max_slip=50, log=False, color_by='total', cmap='magma'):
    # Load background map
    fig, ax = pickle.load(open(f"{rupture_png_dir}/temp.pkl", "rb"))
    # Extract points and cells from the mesh
    points = mesh.points[:, :2]  # Assuming 2D projection (only X and Y)
    cells = mesh.cells_dict[list(mesh.cells_dict.keys())[0]]
    colors = mesh.cell_data[color_by][0]  # Get the 'total' scalar values
    colors = np.where(colors == 0, 1e-6, colors)

    # Create polygons from cells and corresponding colors
    polygons = [points[cell] for cell in cells]

    # Plot using PolyCollection
    #fig, ax = plt.subplots()
    if log:
        collection = PolyCollection(polygons,
                            array=colors,
                            cmap=cmap,
                            edgecolor=None,
                            norm=LogNorm(vmin=max(min_slip, colors.min()), vmax=max_slip)
                            )
    else:
        collection = PolyCollection(polygons, array=colors, cmap=cmap, edgecolor=None)
        collection.set_clim(vmin=min_slip, vmax=max_slip)
    alpha = np.where(colors == 1e-6, 0.25, 1)
    collection.set_alpha(alpha)

    ax['main_figure'].add_collection(collection)
    ax['main_figure'].autoscale_view()

    # Add colorbar and labels
    cbar = plt.colorbar(collection, ax=ax['main_figure'], orientation='vertical', label=color_by)
    if log:
        ticks = np.arange(np.log10(min_slip), np.log10(max_slip), 1)
        ticks = list(10**ticks) + [max_slip]
        tick_labels = [f"{t:.0e}" for t in ticks]  # Format as scientific notation
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
    
    
    #plt.plot(hypo[0], hypo[1], 'b+', markersize=10)    # Hypocenter
    # plt.plot(hypo[2], hypo[3], 'ro', markersize=10)    # Centroid
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # Plot coastline ontop
    coastfile = "C:/Users/jmc753/Work/occ-coseismic/data/coastline/nz_coastline.geojson"
    coastline = gpd.read_file(coastfile)
    coastline.plot(ax=ax["main_figure"], color="k", linewidth=0.5)
    plt.title(title)
    plt.savefig(os.path.join(rupture_png_dir, f'{title}.png'))
    plt.savefig(os.path.join(rupture_png_dir, f'{title}.pdf'), dpi=300, format='pdf')
    plt.close()

mesh_folder = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks'

mesh_name = 'hik_kerk3k_with_rake.vtk'
plot_every = 1  # Plot every nth rupture (-ve to plot from largest first)
n_ruptures = 5000
inversion_name = 'Final_Jack'

file_keyword = 'archi-merged'


vtk = meshio.read(f'{mesh_folder}\\{mesh_name}')
vtk = meshio.read('C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\subduction_quads\\hk_tiles.vtk')
rupture_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires\\output\\ruptures\\'
rupture_png_dir = os.path.abspath(os.path.dirname(rupture_dir) + '/..\\rupture_deficit_pngs\\')
os.makedirs(rupture_png_dir, exist_ok=True)
output_dir = f"C:\\Users\\jmc753\\Work\\MudPy\\cluster_processing\\output\\{inversion_name}"

rupture_list = glob(os.path.abspath(f'{output_dir}\\n{n_ruptures}*{file_keyword}*.inv'))
rupture_list.sort()

# xmin, ymin, xmax, ymax
bounds = [int(bound) for bound in '1500000/5250000/3000000/7300000'.split('/')]
#bounds = [int(bound) for bound in '1500000/5250000/2200000/6200000'.split('/')]

new_background = False  # Recreate the background plot
new_pngs = True    # Overwrite any previously created pngs

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
rupture_list = rupture_list[::plot_every]
for ix, rupture_file in enumerate(rupture_list):
    if os.path.exists(os.path.join(rupture_png_dir, os.path.basename(rupture_file) + '.png')) and not new_pngs:
        continue
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

    # Plot the mesh as a 2D surface
    #plot_2d_surface(rupture_mesh, os.path.basename(rupture_file).replace('.inv', '_target'), rupture_png_dir, [], min_slip=0.1, max_slip=50, log=False, color_by='target-deficit(mm/yr)')
    #plot_2d_surface(rupture_mesh, os.path.basename(rupture_file).replace('.inv', '_inverted'), rupture_png_dir, [], min_slip=0.1, max_slip=50, log=False, color_by='inverted-deficit(mm/yr)')
    #plot_2d_surface(rupture_mesh, os.path.basename(rupture_file).replace('.inv', '_misfit_rel'), rupture_png_dir, [], min_slip=0, max_slip=2, log=False, color_by='misfit_rel(mm/yr)', cmap="RdYlBu_r")
    plot_2d_surface(rupture_mesh, os.path.basename(rupture_file).replace('.inv', '_misfit_abs'), rupture_png_dir, [], min_slip=-5, max_slip=5, log=False, color_by='misfit_abs(mm/yr)', cmap="RdYlBu_r")
    print(f"{os.path.basename(rupture_file)}\t{ix+1}/{len(rupture_list)}")
print('Complete :)')
