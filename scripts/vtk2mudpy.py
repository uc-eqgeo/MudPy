import numpy as np
from scipy.spatial import KDTree
import meshio
from pyproj import Transformer
import matplotlib.pyplot as plt
import math

def vector_to_bearing(vector):
    # Calculate the angle in radians
    angle_radians = math.atan2(vector[1], vector[0])

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    # Convert to 360-degree bearing
    bearing = (90 - angle_degrees) % 360

    return bearing

mesh_folder = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks'

sub_type = 'hk'

data = np.load(f'{mesh_folder}\\subduction_quads\\{sub_type}_tile_outlines.npy')
if sub_type == 'hk':
    vtk = meshio.read(f'{mesh_folder}\\hik_kerk3k_with_rake.vtk')


patch_centers = np.mean(data, axis=1)
dimensions = np.max(data, axis=1) - np.min(data, axis=1)

length = 5000
width = 5000

cells = vtk.cells[0].data
n_cells = cells.shape[0]
cell_centers = np.zeros((n_cells, 3))
for ii in range(n_cells):
    p1, p2, p3 = vtk.cells[0].data[ii, :]
    cell_centers[ii, :] = np.mean(np.vstack([vtk.points[p1, :], vtk.points[p2, :], vtk.points[p3, :]]), axis=0)

hikurangi_kd_tree = KDTree(cell_centers)
_, nearest_indices = hikurangi_kd_tree.query(patch_centers)

rakes = vtk.cell_data['rake'][0][nearest_indices]
dips = [math.degrees(dip) for dip in vtk.cell_data['dip'][0][nearest_indices]]
strikes = vtk.cell_data['strike_direction'][0][nearest_indices, :2]

ll = np.zeros((patch_centers.shape[0], 2))

write = True
if write:
    with open(f'{mesh_folder}\\subduction_quads\\{sub_type}.fault', 'w')as fid:
        fid.write('# No.  lon       lat          z    strike    dip    typ rt  length  width\n')
        fid.write('#     (deg)      (deg)       (km)  (deg)    (deg)             (m)    (m)\n')
        for ix, coords in enumerate(patch_centers):
            ulon, ulat, z = coords
            transformer = Transformer.from_crs("epsg:2193", "epsg:4326")
            lat, lon = transformer.transform(ulat, ulon)
            fid.write(f"{ix + 1}\t{lon:.6f}\t{lat:.6f}\t{abs(z) / 1000:.3f}\t{vector_to_bearing(strikes[ix]):.2f}\t{dips[ix]:.2f}\t0.5\t1.0\t{length}\t{width}\n")
            if lon < 0:
                lon += 360
            ll[ix, :] = np.array([lon, lat])

plt.scatter(ll[:,0], ll[:,1], c=dips)
plt.colorbar()
plt.show()

plt.scatter(patch_centers[:, 0], patch_centers[:, 1], c=dips)
plt.colorbar()
plt.show()