import numpy as np
import meshio
from pyproj import Transformer
from scipy.spatial import KDTree
from glob import glob


mesh_folder = 'C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks'

mesh_name = 'hik_kerk3k_with_rake.vtk'

vtk = meshio.read(f'{mesh_folder}\\{mesh_name}')
vtk = meshio.read('C:\\Users\\jmc753\\Work\\RSQSim\\Aotearoa\\fault_vtks\\subduction_quads\\hk_tiles.vtk')
rupture_dir = 'C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\output\\ruptures'

rupture_list = glob(f'{rupture_dir}\\*1184.inv')

# Create interpolation object for mapping ruptures to the mesh
transformer = Transformer.from_crs("epsg:4326", "epsg:2193")
for rupture_file in rupture_list[::-1]:
    rupture = np.loadtxt(rupture_file)

    patch_coords = np.zeros_like(rupture[:, :4])
    patch_coords[:, 0] = np.arange(rupture.shape[0])
    patch_coords[:, 2], patch_coords[:, 1] = transformer.transform(rupture[:, 2], rupture[:, 1])
    patch_coords[:, 3] = rupture[:, 3] * -1000

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
    ss = rupture[nearest_indices, 8]
    ds = rupture[nearest_indices, 9]
    total = np.sqrt(ss**2 + ds**2)

    rupture_mesh = meshio.Mesh(points=points, cells=[(element, cells)], cell_data={'ss': [ss], 'ds': [ds], 'total': [total]})

    outfile = f"{rupture_file.replace('scaling.', 'scaling+').split('.')[0].replace('scaling+', 'scaling.')}{suffix}.vtk"
    rupture_mesh.write(outfile, file_format="vtk")
    print(f"Written {outfile}")

print('Complete :)')
