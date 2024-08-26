# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:55:43 2024

@author: jmc753
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import meshio
from pyproj import Transformer
from scipy.spatial import KDTree
from matplotlib.collections import PolyCollection

inversion_name = 'start_rand'
n_ruptures = 5000
slip_weight = 1
GR_weight = 10
max_iter = 30000
plot_ruptures = False
min_Mw, max_Mw = 4.5, 9.5

outdir = f"Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\{inversion_name}"

plot_results = True
if max_iter == 0:
    max_iter = '*'

rupture_list = glob(f'{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_nIt{max_iter}_inverted_ruptures.txt')
n_iter = [int(file.split('nIt')[1].split('_')[0]) for file in rupture_list]
order = np.array(n_iter).argsort()
if len(order) != 0:
    ruptures_list = []
    bins_list = []
    for ix in order:
        file = rupture_list[ix]
        ruptures_list.append(pd.read_csv(file, sep='\t', index_col=0).sort_values('Mw'))
        bins_df = pd.read_csv(file.replace('ruptures', 'bins'), sep='\t', index_col=0)
        bins_df['upper'] = pd.read_csv(f"{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_input_bins.txt", sep='\t', index_col=0)['upper']
        bins_list.append(bins_df)
else:
    plot_results = False
    ruptures_list = [pd.read_csv(f"{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_input_ruptures.txt", sep='\t', index_col=0).sort_values('Mw')]
    bins_list = [pd.read_csv(f"{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_input_bins.txt", sep='\t', index_col=0)]

n_runs = len(ruptures_list)
if n_runs > 1:
    min_Mw = 4.0

gr_matrix = np.zeros((bins_list[0].shape[0], ruptures_list[0].shape[0])).astype('bool')
rupture_matrix = np.zeros((ruptures_list[0].shape[0], ruptures_list[0].shape[0])).astype('bool')

for ix, mag in enumerate(bins_list[0]['Mw_bin']):
    gr_matrix[ix, :] = (np.round(ruptures_list[0]['Mw'], 1) >= mag)
gr_matrix = gr_matrix.astype('int')

for ix, mag in enumerate(ruptures_list[0]['Mw']):
    rupture_matrix[ix, :] = (ruptures_list[0]['Mw'] >= mag)
rupture_matrix = rupture_matrix.astype('int')

for run in range(n_runs):
    bins = bins_list[run]
    ruptures = ruptures_list[run]

    initial_rate = np.matmul(rupture_matrix, ruptures['initial_rate'])
    initial_bins = np.matmul(gr_matrix, ruptures['initial_rate'])
    lim_ix = np.where(bins['upper'] != 0)[0]
    if plot_results:
        inverted_rate = np.matmul(rupture_matrix, ruptures['inverted_rate'])
        inverted_bins = np.matmul(gr_matrix, ruptures['inverted_rate'])
        inverted_rate[inverted_rate == 0] = 1e-10
        inverted_bins[inverted_bins == 0] = 1e-10
        ruptures[ruptures['inverted_rate'] == 0] = 1e-10

    ruptures[ruptures['initial_rate'] == 0] = 1e-10
    ruptures[ruptures['upper'] == 0] = 1e-10

# %%
    if run == 0:
        # %%
        binwidth = (0.1, 0.5)
        plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='black', label='Target GR Relation', zorder=6)

        sns.scatterplot(x=ruptures['Mw'], y=np.log10(initial_rate), s=20, color='blue', label='Initial GR', edgecolors=None, zorder=4)
        sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['initial_rate']), s=1, label='Initial rate', color='blue', edgecolors=None, zorder=1)
        sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['upper']), s=5, color='green', label='Individual limit', edgecolors=None, zorder=0)
        sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['lower'] + 1e-30), s=5, color='green', edgecolors=None, zorder=0)

        if plot_results:
            sns.histplot(x=ruptures['Mw'], y=np.log10(ruptures['inverted_rate']), binwidth=binwidth, zorder=0)
            # sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['inverted_rate']), s=2, label='Inverted rate', color='orange', edgecolors=None, zorder=2)
            sns.scatterplot(x=ruptures['Mw'], y=np.log10(inverted_rate), s=10, color='red', label='Inverted GR', edgecolors=None, zorder=5)
        plt.ylabel('log10(N)')
        plt.xlim([min_Mw, max_Mw])
        plt.ylim([-11, 3])
        plt.legend(loc='lower left')
        plt.title(f"# Ruptures: {n_ruptures}")
        # %%
    # if plot_results:
    #     sns.scatterplot(x=bins['Mw_bin'], y=np.log10(inverted_bins + 1e-12), s=15, label=f'Inverted Bins {n_iter[order[run]]}', edgecolors=None)
plt.show()
# %%
if plot_results:
    binwidth = 0.5
    sns.histplot(x=np.log10(ruptures['initial_rate']), y=np.log10(ruptures['inverted_rate']), binwidth=binwidth, binrange=(-12 - binwidth / 2, 1 + binwidth / 2), zorder=0)
    plt.scatter(np.log10(ruptures['initial_rate']), np.log10(ruptures['inverted_rate']), c=ruptures['Mw'], s=3, zorder=2)
    plt.plot([-10, 1], [-10, 1], color='red', zorder=1)
    plt.xlim([-10, 1])
    plt.ylim([-10, 1])
    plt.xlabel('Log(Input Rate)')
    plt.ylabel('Log(Inverted Rate)')
    plt.title(f"# Ruptures: {n_ruptures}")
    plt.colorbar()
    plt.show()

    binwidth = 0.25
    brange = [np.floor(np.log10(ruptures['inverted_rate'].min())), 1]
    sns.histplot(x=np.log10(ruptures['initial_rate']), y=np.log10(ruptures['inverted_rate']), binwidth=binwidth, binrange=(brange[0] - binwidth / 2, brange[1] + binwidth / 2), zorder=0)
    plt.plot(brange, brange, color='red', zorder=1)
    plt.xlim(brange)
    plt.ylim(brange)
    plt.xlabel('Log(Input Rate)')
    plt.ylabel('Log(Inverted Rate)')
    plt.title(f"# Ruptures: {n_ruptures}")
    plt.show()
# %%
plot_extras = False
if plot_extras:
    plt.scatter(ruptures['Mw'], ruptures['inverted_rate'] / ruptures['initial_rate'])
    plt.xlabel('Magnitude')
    plt.ylabel('Change ratio (inv / inp)')
    plt.yscale('log')
    plt.show()


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


# %%
if plot_ruptures:
    groups = ruptures.groupby(ruptures['Mw'].apply(lambda x: np.floor(x)))
    rupture_dir = os.path.abspath(os.path.join(outdir, '..', 'ruptures'))

    vtk = meshio.read('C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\subduction_quads\\hk_tiles.vtk')

    # Create interpolation object for mapping ruptures to the mesh
    transformer = Transformer.from_crs("epsg:4326", "epsg:2193")

    n_plots = 3
    minMw_plot = 9
    for bad in [True, False]:
        for mw, group in groups:
            if mw >= minMw_plot:
                df = group.sort_values(['inverted_rate'], ascending=bad).iloc[:n_plots]
                for id, rupt in df.iterrows():
                    rupture_file = os.path.join(rupture_dir, f"hikkerk3D_locking_NZNSHMscaling.Mw{id}.rupt")
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
                    plot_2d_surface(rupture_mesh, f"{mw} Mw: {id} Bad = {bad} {1/rupt['inverted_rate']:.2f} yrs", color_by='total')
