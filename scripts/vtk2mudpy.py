import numpy as np
from scipy.spatial import KDTree
import meshio
from pyproj import Transformer
import matplotlib.pyplot as plt
import math
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def vector_to_bearing(vector):
    # Calculate the angle in radians
    angle_radians = math.atan2(vector[1], vector[0])

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    # Convert to 360-degree bearing
    bearing = (90 - angle_degrees) % 360

    return bearing

data_folder = '../data/'
mesh_folder = data_folder
vtk_file = 'hik_kerm_adjusted_plate70.vtk'  # VTK file that contains the slip deficit model

sub_type = 'hk'
outname = 'plate70'

data = np.load(os.path.join(mesh_folder,f"{sub_type}_tile_outlines.npy"))  # .npy file that contains the coordinates of the subduction zone tiles

vtk = meshio.read(os.path.join(data_folder, vtk_file))

patch_centers = np.mean(data, axis=1)
dimensions = np.max(data, axis=1) - np.min(data, axis=1)

length = 5000
width = 5000

# Read in the mesh data, and find the center point of each cell
cells = vtk.cells[0].data
n_cells = cells.shape[0]
cell_centers = np.zeros((n_cells, 3))
for ii in range(n_cells):
    if 'triangle' in vtk.cells_dict.keys():
        p1, p2, p3 = vtk.cells[0].data[ii, :]
        cell_centers[ii, :] = np.mean(np.vstack([vtk.points[p1, :], vtk.points[p2, :], vtk.points[p3, :]]), axis=0)
    else:
        p1, p2, p3, p4 = vtk.cells[0].data[ii, :]
        cell_centers[ii, :] = np.mean(np.vstack([vtk.points[p1, :], vtk.points[p2, :], vtk.points[p3, :], vtk.points[p4, :]]), axis=0)

# Find the nearest cell to each patch center using KDTrees
mesh_kd_tree = KDTree(cell_centers)
_, nearest_indices = mesh_kd_tree.query(patch_centers)

rakes = vtk.cell_data['rake'][0][nearest_indices]
if 'dip' in vtk.cell_data.keys():
    dips = [math.degrees(dip) for dip in vtk.cell_data['dip'][0][nearest_indices]]
else:
    dips = np.zeros_like(rakes)
if 'strike_direction' in vtk.cell_data.keys():
    strikes = vtk.cell_data['strike_direction'][0][nearest_indices, :2]
else:
    strikes = np.zeros((rakes.shape[0], 3))

ll = np.zeros((patch_centers.shape[0], 2))

write_to_fault = False  # Write out the patch geometry to .fault file
write_to_slip = True  # Write out slip deficit to .slip file that matches .fault
write_vtk_to_slip = False  # Write out slip deficit to .slip file that matches the input VTK file

# Create a transformer object to convert between coordinate systems (NZTM to WGS84)
transformer = Transformer.from_crs("epsg:2193", "epsg:4326")

if write_to_fault:
    print("Write out fault geometry file")
    out_file = os.path.join(data_folder, f"{sub_type}_{outname}.fault")
    with open(out_file, 'w') as fid:
        fid.write('# No.\tlon\tlat\tz\tstrike\tdip\ttyp\trt\tlength\twidth\trake\n')
        fid.write('#\t(deg)\t(deg)\t(km)\t(deg)\t(deg)\t()\t()\t(m)\t(m)\t(deg)\n')
        for ix, coords in enumerate(patch_centers):
            ulon, ulat, z = coords
            lat, lon = transformer.transform(ulat, ulon)
            fid.write(f"{ix + 1}\t{lon:.6f}\t{lat:.6f}\t{abs(z) / 1000:.3f}\t{vector_to_bearing(strikes[ix]):.2f}\t{dips[ix]:.2f}\t0.5\t1.0\t{length}\t{width}\t{rakes[ix]:.2f}\n")
            if lon < 0:
                lon += 360
            ll[ix, :] = np.array([lon, lat])

if write_to_slip:
    out_file = os.path.join(data_folder, f"{sub_type}_{outname}.slip")
    print("Write out slip file")
    deficits = []
    with open(out_file, 'w') as fid:
        fid.write('#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel\n')
        for ix, cell in enumerate(patch_centers):
            lat, lon = transformer.transform(cell[1], cell[0])
            z = cell[2]
            deficit = vtk.cell_data['slip'][0][nearest_indices[ix]]
            if deficit == 0:
                deficit = 0.001
            deficits.append(deficit)
            fid.write(f"{ix+1}\t{lon:.6f}\t{lat:.6f}\t{-z:.6f}\t{vector_to_bearing(strikes[ix]):.2f}\t{dips[ix]}\t0\t0\t0\t{deficit:.6f}\t0\t0\t0\n")

            if lon < 0:
                lon += 360
            ll[ix, :] = np.array([lon, lat])

if write_vtk_to_slip:
    out_file = os.path.join(data_folder, vtk_file.replace('.vtk', '.slip'))
    print("Write out vtk to slip file (as a bonus)")
    with open(out_file, 'w') as fid:
        fid.write('#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel\n')
        for ix, cell in enumerate(cell_centers):
            lat, lon = transformer.transform(cell[1], cell[0])
            z = cell[2]
            deficit = vtk.cell_data['slip'][0][ix]
            if deficit == 0:
                deficit = 0.001
            fid.write(f"{ix+1}\t{lon:.6f}\t{lat:.6f}\t{-z:.6f}\t{vector_to_bearing(strikes[ix]):.2f}\t{dips[ix]}\t0\t0\t0\t{deficit:.6f}\t0\t0\t0\n")

for var in [rakes, deficits]:
    plt.scatter(patch_centers[:, 0], patch_centers[:, 1], c=var)
    plt.colorbar()
    plt.show()