# %%
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

bn_dict = {1: [0.95, 16.5],
           2: [1.1, 21.5],
           3: [1.24, 27.9]}


fault_name = "hikkerk"
velmod = "3e10"  # Velocity/Slip deficit model
locking = False  # Was locking used in the rupture generation
NZNSHMscaling = True # Was NZNSHM scaling used in the rupture generation
uniformSlip = False # Was uniform slip used in the rupture generation
GR_inv_min = 7.0  # Minimum GR value for the inversion weighting
GR_inv_max = 9.0  # Maximum GR value for the inversion weighting
tapered_gr = True  # Was tapered MFD used in the inversion (Rollins and Avouac 2019)
taper_max_Mw = 9.5  # Max Mw used for the tapered MFD
alpha_s = 1  # Tapered MFD alpha value
dir_suffix = '_GR70-90'  # Extra identifier for the directory name (e.g. '_test')

lock = "_locking" if locking else "_nolocking"
NZNSHM = "_NZNSHMscaling" if NZNSHMscaling else ""
uniform = "_uniformSlip" if uniformSlip else ""

if tapered_gr:
    inversion_name = f"FQ_{velmod}{lock}{uniform.replace('Slip', '')}{dir_suffix}"
else:
    inversion_name = f"FQ_{velmod}{lock}{uniform.replace('Slip', '')}_GR{str(GR_inv_min).replace('.', '')}-{str(GR_inv_max).replace('.', '')}{dir_suffix}"

n_ruptures = 5000
slip_weight = 10
norm_weight = 1
GR_weight = 500
max_iter = 5e4
bn_combo = 2
b, N = bn_dict[bn_combo]
plot_ruptures = False   # Plot sample ruptures
min_Mw, max_Mw = 6.5, 9.5
plot_all_islands = False
zero_rate = -6  # Rate at which a ruptures is considered not to have occurred
n_islands = 1
write_islands = False   # Write out islands with a zero'd rate
archi = '0'  #  '-merged' or number for which archipeligo to plot
if archi == '-merged':
    print('Adjusting zero rate for {} islands'.format(n_islands))
    zero_rate = np.log10((10 ** zero_rate) / n_islands)  # Take into account the 10x slip reduction when merging 10 archipeligos for best result
init = ''
plot_gr = True
plot_rates = True
plot_lines = True
plot_distributions = True
plot_hypocenters = False
out_formats = ['png']

drive = 'z'
if drive.lower() == 'c':
    procdir = 'C:\\Users\\jmc753\\Work\\MudPy\\nesi_outputs'
elif drive.lower() == 'z':
    procdir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires'

rupt_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D_hires\\output\\ruptures'

deficit_file = f"{procdir}\\data\\model_info\\hk_hires.slip"
outdir = f"{procdir}\\output\\{inversion_name}"

rupture_df_file = os.path.abspath(os.path.join(outdir, "..", f'{fault_name}_{velmod}{lock}{NZNSHM}{uniform}_df_n50000.csv'))
print(rupture_df_file)
print(inversion_name)

if max_iter == 0:
    max_iter = '*'

slip_weight, norm_weight, GR_weight, max_iter = [int(val) for val in [slip_weight, norm_weight, GR_weight, max_iter]]

if tapered_gr:
    taper_tag = f"_taper{taper_max_Mw}Mw_alphas{alpha_s:.1f}".replace('.', '-') if tapered_gr else ""

if norm_weight is not None:
    results_tag = f"n{n_ruptures}_S{slip_weight}_N{norm_weight}_GR{GR_weight}{taper_tag}_b{str(b).replace('.','-')}_N{str(N).replace('.','-')}_nIt{max_iter}"
else:
    results_tag = f"n{n_ruptures}_S{slip_weight}_GR{GR_weight}{taper_tag}_b{str(b).replace('.','-')}_N{str(N).replace('.','-')}_nIt{max_iter}"

if init:
    results_tag += f"-init{init}"

if archi is not None:
    results_tag += f"_archi{archi}"

print(results_tag)

input_tag = results_tag.replace(f'_nIt{max_iter}', '').replace(f"_archi{archi}", '')

# %% Load data
rupture_file_list = glob(f"{outdir}\\{results_tag}_inverted_ruptures.csv")
if init:
    order = [0]
else:
    n_iter = [int(file.split('nIt')[1].split('_')[0]) for file in rupture_file_list]
    order = np.array(n_iter).argsort()
if len(order) != 0:
    rupture_file_list = [rupture_file_list[ix] for ix in order]
    ruptures_list = []
    bins_list = []
    for file in rupture_file_list:
        ruptures_list.append(pd.read_csv(file, sep='\t', index_col=0).sort_values('Mw'))
        bins_df = pd.read_csv(file.replace('ruptures', 'bins').replace('-merged', '0'), sep='\t', index_col=0)
        bins_df['upper'] = pd.read_csv(f"{outdir}\\{input_tag}_input_bins.csv".replace(f"-init{init}", '').replace('-merged', '0'), sep='\t', index_col=0)['upper']
        bins_list.append(bins_df)
        islands = ruptures_list[0].columns.tolist()
        islands = [island for island in islands if island not in ['Mw', 'initial_rate', 'target_rate', 'lower', 'upper']]
        if not plot_all_islands:
            islands = [islands[0]]
else:
    plot_lines, plot_gr, plot_rates = False, False, False
    ruptures_list = [pd.read_csv(f"{outdir}\\{input_tag}_input_ruptures.csv", sep='\t', index_col=0).sort_values('Mw')]
    bins_list = [pd.read_csv(f"{outdir}\\{input_tag}_input_bins.csv", sep='\t', index_col=0)]
    islands = ['input_rate']
# %%  Plot Full GR-relations
n_runs = len(ruptures_list)
if n_runs > 1:
    min_Mw = 4.0

gr_matrix = np.zeros((bins_list[0].shape[0], ruptures_list[0].shape[0])).astype('bool')
rupture_matrix = np.zeros((ruptures_list[0].shape[0], ruptures_list[0].shape[0])).astype('bool')
rupture_bin_matrix = np.zeros((ruptures_list[0].shape[0], ruptures_list[0].shape[0])).astype('bool')

# Create a GR matrix for each magnitude bin
for ix, mag in enumerate(bins_list[0]['Mw_bin']):
    gr_matrix[ix, :] = (ruptures_list[0]['Mw'] >= mag)
gr_matrix = gr_matrix.astype('int')

# Create a GR matrix for each individual ruputure
for ix, mag in enumerate(ruptures_list[0]['Mw']):
    rupture_matrix[ix, :] = (ruptures_list[0]['Mw'] >= mag)
    rupture_bin_matrix[ix, :] = (ruptures_list[0]['Mw'] == mag)
rupture_matrix = rupture_matrix.astype('int')
rupture_bin_matrix = rupture_bin_matrix.astype('int')

for run in range(n_runs):
    inverted_bins_list = []
    for island in islands:
        bins = bins_list[run]
        ruptures = ruptures_list[run]

        initial_rate = np.matmul(rupture_matrix, ruptures['initial_rate'])
        initial_bins = np.matmul(gr_matrix, ruptures['initial_rate'])
        lim_ix = np.where(bins['upper'] != 0)[0]
        if plot_gr:
            inverted_rate = np.matmul(rupture_matrix, ruptures[island])
            inverted_rate_bins = np.matmul(rupture_bin_matrix, ruptures[island])
            inverted_bins = np.matmul(gr_matrix, ruptures[island])
            inverted_rate[inverted_rate == 0] = 1e-10
            inverted_rate_bins[inverted_rate_bins == 0] = 1e-10
            inverted_bins[inverted_bins == 0] = 1e-10
            ruptures[ruptures[island] == 0] = 1e-10
            inverted_bins_list.append(inverted_bins)

        ruptures[ruptures['initial_rate'] == 0] = 1e-10
        ruptures[ruptures['upper'] == 0] = 1e-10

        if run == 0:
            ## %%
            binwidth = (0.1, 0.5)
            plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='black', label='Target GR Relation', zorder=6)

            #sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['upper']), s=5, color='green', label='Individual limit', edgecolors=None, zorder=1)
            sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['lower'] + 1e-30), s=5, color='green', edgecolors=None, zorder=1)
            #sns.scatterplot(x=ruptures['Mw'], y=np.log10(initial_rate), s=20, color='blue', label='Initial GR', edgecolors=None, zorder=4)
            #sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['initial_rate']), s=1, label='Initial rate', color='blue', edgecolors=None, zorder=1)

            if plot_gr:
                #sns.histplot(x=ruptures['Mw'], y=np.log10(ruptures[island]), binwidth=binwidth, zorder=0)
                # sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures[island]), s=2, label='Inverted rate', color='orange', edgecolors=None, zorder=2)
                sns.scatterplot(x=ruptures['Mw'], y=np.log10(inverted_rate), s=10, color='red', label='Inverted GR', edgecolors=None, zorder=5)
                sns.scatterplot(x=ruptures['Mw'], y=np.log10(inverted_rate_bins), s=10, color='orange', label='Inverted Rate Bins', edgecolors=None, zorder=5)
                sns.scatterplot(x=bins_list[0].Mw_bin, y=np.log10(inverted_bins), s=7, color='blue', label='Inverted Bins', edgecolors=None, zorder=5)

            plt.ylabel('log10(N)')
            plt.xlim([min_Mw, max_Mw])
            plt.ylim([-11, 3])
            plt.legend(loc='lower left')
            plt.title(f" {input_tag} {max_iter} {island}")
        # if plot_results:
        #     sns.scatterplot(x=bins['Mw_bin'], y=np.log10(inverted_bins + 1e-12), s=15, label=f'Inverted Bins {n_iter[order[run]]}', edgecolors=None)
        for format in out_formats:
            print(f"{outdir}\\{input_tag}_GR_{island}_isl_{archi}.{format}")
            plt.savefig(f"{outdir}\\{input_tag}_GR_{island}_isl_{archi}.{format}", dpi=300, format=format)
        plt.show()
# %% Plot rate comparisons
if plot_rates:
    for island in islands[:1]:
        binwidth = 0.5
        sns.histplot(x=np.log10(ruptures['initial_rate']), y=np.log10(ruptures[island]), binwidth=binwidth, binrange=(-30 - binwidth / 2, 1 + binwidth / 2), zorder=0)
        plt.scatter(np.log10(ruptures['initial_rate']), np.log10(ruptures[island]), c=ruptures['Mw'], s=3, zorder=2)
        plt.plot([-100, 1], [-100, 1], color='red', zorder=1)
        plt.xlim([-16, 1])
        plt.ylim([-16, 1])
        plt.xlabel('Log(Input Rate)')
        plt.ylabel('Log(Inverted Rate)')
        plt.title(f"{input_tag} {max_iter} {island}")
        plt.colorbar()
        plt.show()

        g = sns.JointGrid(x=np.log10(ruptures['initial_rate']), y=np.log10(ruptures[island]), marginal_ticks=True, xlim=[-16,1], ylim=[-16,1])
        # Add the joint and marginal histogram plots
        g.plot_joint(sns.histplot, discrete=(True, False), pmax=.8, binwidth=binwidth)
        g.plot_marginals(sns.histplot, element="step", color="#03012d", kde=True)
        if zero_rate != 0:
            for ax in (g.ax_joint, g.ax_marg_y):
                ax.axhline(zero_rate, color='crimson', ls='--', lw=3)
            g.figure.suptitle(f"{input_tag} {island} Kept: {np.sum(np.log10(ruptures[island]) >= zero_rate)}/{ruptures.shape[0]}")
        else:
            g.figure.suptitle(f"{input_tag} {island}")
        plt.show()

        kept_ruptures = ruptures[ruptures[island] >= 10 ** zero_rate]

        plt.hist(kept_ruptures['Mw'], bins=np.arange(6.5, 9.5, 0.01), density=False, histtype="step", cumulative=-1)
        plt.title(f"Ruptures > Mw 8: {kept_ruptures[kept_ruptures['Mw'] > 8].shape[0]}")
        plt.show()
    
# %% Plot Island GR-relation comparisons
if plot_lines and plot_all_islands:
        plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='black', label='Target GR Relation', zorder=0)
        for ix, island in enumerate(islands[1:]):
            plt.plot(bins['Mw_bin'], np.log10(inverted_bins_list[ix + 1]))
        plt.plot(bins['Mw_bin'], np.log10(inverted_bins_list[0]), color='red')
        plt.ylabel('log10(N)')
        plt.xlabel('Mw')
        #plt.xlim([min_Mw, max_Mw])
        #plt.ylim([-11, 3])
        plt.title(f"{input_tag} {max_iter}")
        plt.show()

        for ix, island in enumerate(islands[1:]):
            plt.plot(bins['Mw_bin'], np.log10(inverted_bins_list[ix + 1]) / np.log10(inverted_bins_list[0]))
        plt.ylabel('Normalised log10(N)')
        plt.xlabel('Mw')
        #plt.xlim([min_Mw, max_Mw])
        #plt.ylim([-11, 3])
        plt.title(f"# Best Island Normalised Ruptures: {n_ruptures}")
        plt.show()

# %% Plot change ratios
plot_extras = False
if plot_extras:
    plt.scatter(ruptures['Mw'], ruptures[island] / ruptures['initial_rate'])
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


# %% Plot example ruptures
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
                df = group.sort_values([island], ascending=bad).iloc[:n_plots]
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
                    plot_2d_surface(rupture_mesh, f"{mw} Mw: {id} Bad = {bad} {1/rupt[island]:.2f} yrs", color_by='total')

# %% Writing the ruptures to a file with with zero'd rates
if plot_distributions and write_islands:
    ruptures_list = []
    for file in rupture_file_list:
        ruptures_list.append(pd.read_csv(file, sep='\t', index_col=0))

    ruptures_df = pd.read_csv(rupture_df_file, nrows=n_ruptures)
    i0, i1 = ruptures_df.columns.get_loc('0'), ruptures_df.columns.get_loc(ruptures_df.columns[-1]) + 1
    slip = ruptures_df.iloc[:, i0:i1].values.T * 1000  # Slip in mm, convert from m to mm

    # Output deficits
    deficit = np.genfromtxt(deficit_file)
    deficit[:, 3] /= 1000  # Convert to km

    if zero_rate < 0:
        zeroed_rate = 10 ** zero_rate
    else:
        zeroed_rate = zero_rate
        if zero_rate != 0:
            zero_rate = int(np.log10(zero_rate))

    for ix, ruptures in enumerate(ruptures_list):
        islands = ruptures.columns.tolist()
        islands = [island for island in islands if island not in ['Mw', 'initial_rate', 'target_rate', 'lower', 'upper']]
        if not plot_all_islands:
            islands = [islands[0]]
        for island in islands:
            print(island)
            rupture_rate = np.array(ruptures[island])
            island_ix = int(island.split('_')[-1])
            out_array = deficit.copy()
            out_array[:, 8] = deficit[:, 9]
            non_zeroed_deficit = np.matmul(slip, rupture_rate)

            if island == 'inverted_rate_0':
                best_non_zeroed = non_zeroed_deficit.copy()

            out_array[:, 9] = non_zeroed_deficit
            outfile = os.path.join(outdir, f'{results_tag}_zeroed{0}_isl{island_ix}_deficit.inv')
            np.savetxt(outfile, out_array, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.0f\t%.0f\t%.0f",
                    header='#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel')

            out_array[:, 8] = non_zeroed_deficit / deficit[:, 9]  # Fractional misfit
            out_array[:, 9] = non_zeroed_deficit - deficit[:, 9]  # Absolute misfit
            out_array[:, 10] = non_zeroed_deficit / best_non_zeroed  # Normalised misfit
            outfile = os.path.join(outdir, f'{results_tag}_zeroed{0}_isl{island_ix}_misfit.inv')
            np.savetxt(outfile, out_array, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f",
                    header='No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tmisfit_perc(mm/yr)\tmisfit_mag(mm/yr)\tmisfit_norm(mm/yr)\trigid\tvel')
            
            if zero_rate != 0:
                out_array = deficit.copy()
                rupture_rate = np.array(ruptures[island])
                rupture_rate[rupture_rate >= zeroed_rate] = 0
                zeros_deficit = np.matmul(slip, rupture_rate)
                out_array[:, 8] = zeros_deficit   # Zero Rate into the SS
                rupture_rate = np.array(ruptures[island])
                rupture_rate[rupture_rate < zeroed_rate] = 0
                reconstructed_deficit = np.matmul(slip, rupture_rate)
                out_array[:, 9] = reconstructed_deficit   # Zeroed Rate into the DS
                outfile = os.path.join(outdir, f'{results_tag}_zeroed{zero_rate}_isl{island_ix}_deficit.inv')
                np.savetxt(outfile, out_array, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f",
                        header='#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel')

                if island == 'inverted_rate_0':
                    best_zeroed = zeros_deficit.copy()

                out_array[:, 8] = reconstructed_deficit / deficit[:, 9]  # Fractional misfit
                out_array[:, 9] = reconstructed_deficit - deficit[:, 9]  # Absolute misfit
                out_array[:, 10] = reconstructed_deficit / best_zeroed  # Normalised misfit
                outfile = os.path.join(outdir, f'{results_tag}_zeroed{zero_rate}_isl{island_ix}_misfit.inv')
                np.savetxt(outfile, out_array, fmt="%.0f\t%.6f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f\t%.0f\t%.6f\t%.6f\t%.0f\t%.0f\t%.0f",
                        header='No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tmisfit_perc(mm/yr)\tmisfit_mag(mm/yr)\tmisfit_norm(mm/yr)\trigid\tvel')
# %%  Plot Island rate comparisons
if len(islands) > 1:
    df = ruptures_list[0].drop(['initial_rate', 'lower', 'upper', 'target_rate'], axis=1)
    df['Rupture number'] = np.arange(df.shape[0])
    df = df.melt(id_vars=['Rupture number', 'Mw'])
    df['log10(Rate)'] = np.log10(df['value'])
    sns.histplot(df, x='log10(Rate)', y='Rupture number', binwidth=(0.1, 1), cbar=True)
    plt.show()

    df.loc[df['log10(Rate)'] < zero_rate, 'log10(Rate)'] = zero_rate
    sns.histplot(df, x='log10(Rate)', y='Rupture number', binwidth=(0.1, 1), binrange=[(zero_rate, df['log10(Rate)'].max()), (-0.5, df['Rupture number'].max())], cbar=True)
# %% Load ruptures_df
if archi == '-merged':
    ruptures_df = pd.read_csv(rupture_df_file)
else:
    ruptures_df = pd.read_csv(rupture_df_file, nrows=n_ruptures)
# %% Plot patch specific GR relations
patch_numbers = np.arange(0, 14800, 100)
island_to_use = 0

if isinstance(island_to_use, (int, float)):
    island_to_use = f"inverted_rate_{int(island_to_use)}"

if not isinstance(patch_numbers, list):
    patch_numbers = list(patch_numbers)

i0, i1 = ruptures_df.columns.get_loc('0'), ruptures_df.columns.get_loc(ruptures_df.columns[-1]) + 1  # Get the column ids for each patch
patch_numbers = [patch for patch in patch_numbers if patch < int(ruptures_df.columns[-1])]  # Get the patch numbers for plotting GR (checking that they don't exceed the number of patches)
slip = ruptures_df.iloc[:, i0:i1].values.T * 1000  # Convert to mm the amount of slip per patch for each rupture
slip = (slip[patch_numbers, :] > 0).astype(int)  # Array consisting of whether or not a rupture slips a patch

rates = np.vstack([ruptures_list[0][island_to_use].values] * len(patch_numbers)) * slip
rates[rates < 10 ** zero_rate] = 0

patch_gr_df = pd.DataFrame(columns=['Mw'] + patch_numbers)
patch_gr_df['Mw'] = bins['Mw_bin']

mod_gr_df = pd.DataFrame(columns=['Mw'] + patch_numbers)
mod_gr_df['Mw'] = bins['Mw_bin']

patch_ba = pd.DataFrame(columns=['Lon', 'Lat', 'b', 'a', 'N'])

deficit = np.genfromtxt(deficit_file)

for ix, patch in enumerate(patch_numbers):
        n_value = np.matmul(gr_matrix, rates[ix, :])
        non_inf = n_value > 0
        sum_non_inf = np.sum(non_inf)
        if sum_non_inf < 2:
            continue
        patch_gr_df.loc[non_inf, patch] = np.log10(n_value[non_inf])

        b, a = np.linalg.lstsq(np.hstack([patch_gr_df['Mw'].values[non_inf].reshape(-1, 1) * -1, np.ones((sum_non_inf, 1))]),
                               patch_gr_df[patch].values[non_inf].astype(float).reshape(-1, 1), rcond = None)[0]
        patch_ba.loc[patch] = {'Lon': np.mod(deficit[patch, 1], 360), 'Lat': deficit[patch, 2], 'b': b[0], 'a': a[0], 'N': 10 ** (a[0] - b[0] * 5)}
        mod_gr_df[patch] = a - b * mod_gr_df['Mw']

patch_gr_df_long = patch_gr_df.melt(id_vars=['Mw'], var_name='Patch', value_name='log10(N)')
mod_gr_df_long = mod_gr_df.melt(id_vars=['Mw'], var_name='Patch', value_name='log10(N)')
# %%
if plot_distributions:
    sns.lineplot(data=patch_gr_df_long, x='Mw', y='log10(N)', hue='Patch', linewidth=0.25)
    plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='red', label='Target GR Relation', zorder=6)
    plt.ylim([zero_rate - 3, 0])
    plt.show()

    sns.histplot(patch_gr_df_long, x='Mw', y='log10(N)', binwidth=(0.1, 0.05), cbar=True)
    plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='red', label='Target GR Relation', zorder=6)
    plt.ylim([zero_rate - 3, 0])
    plt.show()
    # %%
    sns.lineplot(data=mod_gr_df_long, x='Mw', y='log10(N)', hue='Patch', linewidth=0.25)
    plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='red', label='Target GR Relation', zorder=6)
    plt.ylim([zero_rate - 3, 0])
    plt.show()
    # %%
    ax = sns.histplot(data=patch_ba, x='b', binwidth=0.02, binrange=(0.0, 1.5), stat='count')
    max_y_value = np.ceil(max([p.get_height() for p in ax.patches]) / 10) * 10
    plt.vlines(patch_ba['b'].mean(), 0, max_y_value, color='black', label=f"Mean: {patch_ba['b'].mean():.3f}")
    plt.vlines(patch_ba['b'].median(), 0, max_y_value, color='red', label=f"Median: {patch_ba['b'].median():.3f}")
    plt.legend()
    plt.show()

    plt.scatter(patch_ba['Lon'], patch_ba['Lat'], c=patch_ba['b'], s=1, vmin=0.75, vmax=1.25)
    plt.colorbar()
    plt.title('b-value distribution')
    plt.show()

    plt.scatter(patch_ba['Lon'], patch_ba['Lat'], c=patch_ba['N'], s=1, vmin=0)
    plt.colorbar()
    plt.title('N distribution')
    plt.show()

    ax = sns.histplot(data=patch_ba, x='N', binwidth=0.1, binrange=(0, 12), stat='count')
    max_y_value = np.ceil(max([p.get_height() for p in ax.patches]) / 10) * 10
    plt.vlines(patch_ba['N'].mean(), 0, max_y_value, color='black', label=f"Mean: {patch_ba['N'].mean():.3f}")
    plt.vlines(patch_ba['N'].median(), 0, max_y_value, color='red', label=f"Median: {patch_ba['N'].median():.3f}")
    plt.legend()
    plt.show()
# %% Plot hyopcentral locations
if plot_hypocenters:
    island_to_use = 0
    if isinstance(island_to_use, (int, float)):
        island_to_use = f"inverted_rate_{int(island_to_use)}"

    ruptures = ruptures_list[0]
    ruptures.loc[ruptures[island_to_use] > 10 ** zero_rate]

    ll_df = pd.DataFrame(columns=['lon', 'lat', 'depth', 'Mw', 'rate', 'log(rate)'])

    for ix, rupture in ruptures.iterrows():
        if rupture[island_to_use] < 10 ** zero_rate:
            continue
        rupt_log = f"{fault_name}_{velmod}{lock}{NZNSHM}{uniform}.Mw{rupture.name}.log"
        with open(os.path.join(rupt_dir, rupt_log), 'r') as f:
            lines = f.readlines()
            lon, lat, depth = [float(coord.strip('()')) for coord in lines[17].split()[2].split(',')]
            lon = np.mod(lon, 360)
            mw = float(lines[16].split()[3].strip())
            ll_df.loc[rupture.name] = {'lon': lon, 'lat': lat, 'depth': depth, 'Mw': mw, 'rate': rupture[island_to_use], 'log(rate)': np.log10(rupture[island_to_use])}

    ll_df = ll_df.sort_values('Mw', ascending=False)
    sns.scatterplot(data=ll_df, x='lon', y='lat', hue='log(rate)', size='Mw', sizes=(5, 50), palette='viridis')
    plt.show()
    sns.histplot(data=ll_df, x='lon', y='lat', binwidth=(0.25, 0.25), cbar=True)
    plt.show()
# %%
