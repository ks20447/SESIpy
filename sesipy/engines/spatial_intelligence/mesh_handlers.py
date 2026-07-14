import meshio
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from lyceanem.base_classes import (
    points,
    structures,
    antenna_structures,
)
from ...utils import NormalFactory

# from lyceanem.geometry.geometryfunctions import mesh_translate, mesh_rotate

# def translate(mesh, vec):
#     return mesh_translate(mesh, vec)


# def rotate(mesh, vec, center):
#     return mesh_rotate(mesh, vec, rotation_centre=center)


def translate(mesh: meshio.Mesh, translation: np.ndarray) -> None:
    translation = np.asarray(translation, dtype=float)
    mesh.points += translation
    return mesh


def rotate(
    mesh: meshio.Mesh,
    rotation: np.ndarray,
    center: np.ndarray | None = None,
) -> None:
    if center is None:
        center = mesh.points.mean(axis=0)
    else:
        center = np.asarray(center, dtype=float)

    R = Rotation.from_euler("xyz", rotation).as_matrix()

    mesh.points[:] = (mesh.points - center) @ R.T + center

    if "Normals" in mesh.point_data:
        mesh.point_data["Normals"][:] = (
            mesh.point_data["Normals"] @ R.T
        )

    if "Normals" in mesh.cell_data:
        mesh.cell_data["Normals"] = [
            normals @ R.T for normals in mesh.cell_data["Normals"]
        ]
        
    return mesh



class LyceanObject:

    def __init__(self, mesh):
        self.meshio_mesh = mesh
        self.translation = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def translate(self, vec):
        self.meshio_mesh = translate(self.meshio_mesh, vec)
        self.translation += vec

    def translate_origin(self):
        self.meshio_mesh = translate(self.meshio_mesh, -self.translation)
        self.translation[:] = 0

    def translate_to(self, coord):
        self.translate_origin()
        self.meshio_mesh = translate(self.meshio_mesh, coord)
        self.translation += coord

    def rotate(self, vec, center=np.zeros((1, 3))):
        self.meshio_mesh = rotate(self.meshio_mesh, vec, center)

    @property
    def points(self):
        return self.meshio_mesh.points

    @property
    def center(self):
        return np.mean(self.meshio_mesh.points, axis=0)

    def get_normals(self):
        return self.meshio_mesh.point_data["Normals"]

    def get_area(self):
        return self.meshio_mesh.point_data["Area"]

    def set_normals(self, normals):
        self.meshio_mesh.point_data["Normals"] = normals

    def set_area(self, area):
        self.meshio_mesh.point_data["Area"] = area

    def compute_normals(self):
        pv_mesh = pv.from_meshio(self.meshio_mesh).extract_surface()

        pv_mesh = pv_mesh.compute_normals(
            cell_normals=True,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=False,
            inplace=False,
        )

        self.meshio_mesh = pv.to_meshio(pv_mesh)

    def compute_areas(self):
        pv_mesh = pv.from_meshio(self.meshio_mesh).extract_surface()

        pv_mesh = pv_mesh.compute_cell_sizes(
            length=False,
            area=True,
            volume=False,
        )

        self.meshio_mesh = pv.to_meshio(pv_mesh)

    def initialise_mesh(self):
        self.compute_areas()
        self.compute_normals()


class AntennaArray:

    def __init__(self):
        self.points_mesh = None
        self.structure_mesh = None
        self._aperture = None
        self._normal_factory = None

    @property
    def aperture(self):
        return self._aperture

    def create_aperture(self):

        if self.points_mesh is None:
            raise ValueError("No antenna points have been assigned.")
        
        if "Normals" not in self.points_mesh.meshio_mesh.point_data:
            raise ValueError("Antenna point normals have not been assigned")
        
        if "Area" not in self.points_mesh.meshio_mesh.point_data:
            raise ValueError("Antenna point area have not been assigned")

        aperture_points = points([self.points_mesh.meshio_mesh])

        aperture_structure = structures(
            [self.structure_mesh.meshio_mesh] if self.structure_mesh is not None else []
        )

        self._aperture = antenna_structures(
            aperture_structure,
            aperture_points,
        )

    def _foreach_mesh(self, func, *args):
        func(self.points_mesh, *args)

        if self.structure_mesh is not None:
            func(self.structure_mesh, *args)

    def translate(self, vec):
        self._foreach_mesh(LyceanObject.translate, vec)

    def rotate(self, vec, center=np.zeros((1, 3))):
        self._foreach_mesh(LyceanObject.rotate, vec, center)

    def translate_origin(self):
        self._foreach_mesh(LyceanObject.translate_origin)

    def translate_to(self, coord):
        self._foreach_mesh(LyceanObject.translate_to, coord)

    def set_point_normals(self, normals):
        self.points_mesh.set_normals(normals)
        
    def set_point_data(self, key, data):
        self.points_mesh.meshio_mesh.point_data[key] = data


class AntennaWrapper(AntennaArray):

    def __init__(self):

        self._points = None
        self._structure = None

        self.points_mesh = None
        self.structure_mesh = None

    def _update_meshes(self):

        if self._points is None:
            self.points_mesh = None
        else:
            self.points_mesh = LyceanObject(self._points)
            self._normal_factory = NormalFactory(self.points_mesh.meshio_mesh)
            self.normal_factory.apply(None)

        if self._structure is None:
            self.structure_mesh = None
        else:
            self.structure_mesh = LyceanObject(self._structure)
            self.structure_mesh.initialise_mesh()

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points: np.ndarray):
        self._points = meshio.Mesh(points=points, cells=[])
        self._update_meshes()
        
    @property
    def normal_factory(self):
        return self._normal_factory

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure: meshio.Mesh):
        self._structure = structure
        self._update_meshes()

    @property
    def point_normals(self):
        return self.points_mesh.get_normals()

    @point_normals.setter
    def point_normals(self, normals):
        self.points_mesh.set_normals(normals)

    @property
    def point_area(self):
        return self.points_mesh.get_area()

    @point_area.setter
    def point_area(self, area):
        self.points_mesh.set_area(area)
        
    def _apply_normals(self, rot=None):
        if self.normal_factory.mode is not None:
            kwargs = self.normal_factory.get_args()
            self.normal_factory.apply(**kwargs, rotation=rot)
        
    def translate(self, vec):
        super().translate(vec)
        self._apply_normals()

    def rotate(self, vec, center=np.zeros((1, 3))):
        super().rotate(vec, center)
        self._apply_normals(rot=vec)

    def translate_origin(self):
        super().translate_origin()
        self._apply_normals()

    def translate_to(self, coord):
        super().translate_to(coord)
        self._apply_normals()
