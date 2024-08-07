import meshio
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import KDTree

# This script will read in the slip deficit rates for each subduction zone patch and use those to create a mean_slip file in the style of
# the .rupt files used by FakeQuakes, which can then be used to generate the correct mean slip distributions for each generated rupture

# CSV file containing slip deficit rates for each subduction zone patch (see Van Dissen et al 2022)
slip_deficit_csv = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\NSHM_geometries\\GNS SR2022-31 ESup_DFM_2_Hik-Kerm-locked.csv'

# GeoJSON file containing the centroids of each subduction zone patch (i.e. as created by occ-coseismic scripts (see Delano et al 2024))
patch_centroid_json = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\NSHM_geometries\\sz_all_rectangle_centroids.geojson'

# VTK file containing the subduction zone geometry
vtk_file = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\hik_kerk3k_with_rake.vtk'

# Mean slip file to be written
mean_slip_file = 'C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\data\\model_info\\slip_deficit_trenchlock.slip'

# Load Files
columns = ['Fault Name', 'Slip Rate (mm/yr)', 'Dip Angle', 'Upper Depth', 'Lower Depth', 'Lon1', 'Lat1', 'Z1', 'Lon2', 'Lat2', 'Z2']
slip_deficit = pd.read_csv(slip_deficit_csv, index_col=0, names=columns, skiprows=1)
centroids = gpd.read_file(patch_centroid_json)
vtk = meshio.read(vtk_file)

# Extract centroid coordinates from GeoDataFrame
centroid_coords = np.vstack([np.array(centroids.geometry.x), np.array(centroids.geometry.y), np.array(centroids.geometry.z)]).T

# Extract cell centers from VTK file
cells = vtk.cells[0].data
n_cells = cells.shape[0]
cell_centers = np.zeros((n_cells, 3))
for ii in range(n_cells):
    p1, p2, p3 = vtk.cells[0].data[ii, :]
    cell_centers[ii, :] = np.mean(np.vstack([vtk.points[p1, :], vtk.points[p2, :], vtk.points[p3, :]]), axis=0)

# Find nearest centroid to each cell center
centroid_kd_tree = KDTree(centroid_coords)
_, nearest_indices = centroid_kd_tree.query(cell_centers)

# Write out rupture file
transformer = Transformer.from_crs("epsg:2193", "epsg:4326")
strike, dip, rise, dura, ss_slip = 0, 0, 0, 0, 0
with open(mean_slip_file, 'w') as fid:
    # Write Headers. Only column that is actually used are slips (and even then, 1 i set to zero)
    fid.write('#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel\n')
    for ix, cell in enumerate(cell_centers):
        lat, lon = transformer.transform(cell[1], cell[0])
        z = cell[2]
        deficit = slip_deficit['Slip Rate (mm/yr)'].loc[nearest_indices[ix]]
        fid.write(f"{ix}\t{lon:.6f}\t{lat:.6f}\t{-z:.6f}\t{strike}\t{dip}\t{rise}\t{dura}\t{ss_slip}\t{deficit:.6f}\t0\t0\t0\n")