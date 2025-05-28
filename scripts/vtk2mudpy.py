import numpy as np
from scipy.spatial import KDTree
import meshio
from pyproj import Transformer
import matplotlib.pyplot as plt
import math
import os

def vector_to_bearing(vector):
    # Calculate the angle in radians
    angle_radians = math.atan2(vector[1], vector[0])

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    # Convert to 360-degree bearing
    bearing = (90 - angle_degrees) % 360

    return bearing

mesh_folder = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks'
vtk_file = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\hik_kerm_rates\\fq_hik_kerm_adjusted_lw2025_final_slip_rates_coarse.vtk'

sub_type = 'hk'
outdir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerm\\data\\model_info'
outname = 'hk_lw2025'

data = np.load(f'{mesh_folder}\\subduction_quads\\{sub_type}_tile_outlines.npy')

vtk = meshio.read(vtk_file)


patch_centers = np.mean(data, axis=1)
dimensions = np.max(data, axis=1) - np.min(data, axis=1)

length = 5000
width = 5000

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

write_to_fault = False
write_to_slip = True

transformer = Transformer.from_crs("epsg:2193", "epsg:4326")

if write_to_fault:
    out_file = os.path.join(outdir, f"{outname}.fault")
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
    out_file = os.path.join(outdir, f"{outname}.slip")
    with open(out_file, 'w') as fid:
        fid.write('#No\tlon\tlat\tz(km)\tstrike\tdip\trise\tdura\tss-deficit(mm/yr)\tds-deficit(mm/yr)\trupt_time\trigid\tvel\n')
        for ix, cell in enumerate(cell_centers):
            lat, lon = transformer.transform(cell[1], cell[0])
            z = cell[2]
            deficit = vtk.cell_data['slip'][0][ix]
            if deficit == 0:
                deficit = 0.001
            fid.write(f"{ix+1}\t{lon:.6f}\t{lat:.6f}\t{-z:.6f}\t{vector_to_bearing(strikes[ix]):.2f}\t{dips[ix]}\t0\t0\t0\t{deficit:.6f}\t0\t0\t0\n")

            if lon < 0:
                lon += 360
            ll[ix, :] = np.array([lon, lat])

plt.scatter(ll[:,0], ll[:,1], c=dips)
plt.colorbar()
plt.show()

plt.scatter(patch_centers[:, 0], patch_centers[:, 1], c=dips)
plt.colorbar()
plt.show()