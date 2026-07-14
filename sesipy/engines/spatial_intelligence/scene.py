import sys
import warnings
import numpy as np
import lyceanem.models.frequency_domain as fd
from tqdm import tqdm
from contextlib import nullcontext
from .mesh_handlers import LyceanObject
from ...utils import suppress_c_output
from lyceanem.base_classes import points, structures
from .utils import create_mesh_copies


class Scene:

    def __init__(self, scatter=False, cuda=True, **kwargs):

        self.scatter = int(scatter)
        self.cuda = cuda

        self._receiver = None
        self._transmitter = None

        self._scatterers = []
        self._blockers = []

        self.blocker_structure = None
        self.scatter_points = None

        self._suppress_output = kwargs.get("suppress_output", True)

        if self._suppress_output:
            warnings.filterwarnings("ignore")

    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, receiver):
        self._receiver = receiver

    @receiver.deleter
    def receiver(self):
        self._receiver = None

    @property
    def transmitter(self):
        return self._transmitter

    @transmitter.setter
    def transmitter(self, transmitter):
        self._transmitter = transmitter

    @transmitter.deleter
    def transmitter(self):
        self._transmitter = None

    @property
    def scatterers(self):
        return self._scatterers

    def add_scatterers(self, meshes):

        scatterers = []

        for mesh in meshes:
            mesh_object = LyceanObject(mesh)
            mesh_object.initialise_mesh()
            scatterers.append(mesh_object.meshio_mesh)

        self.scatterers.extend(scatterers)

    @property
    def blockers(self):
        return self._blockers

    def add_blockers(self, meshes):

        blockers = []

        for mesh in meshes:
            mesh_object = LyceanObject(mesh)
            mesh_object.initialise_mesh()
            blockers.append(mesh_object.meshio_mesh)

        self.blockers.extend(blockers)

    def set_scene(self):

        if self.transmitter:
            self.transmitter.create_aperture()
        if self.receiver:
            self.receiver.create_aperture()

        self.blocker_structure = structures(self.blockers)
        self.scatter_points = points(self.scatterers).export_points()
        self.scatter_points.points += 0.00 * self.scatter_points.point_data["Normals"]

    def calculate_receiver_scattering(self):

        self.set_scene()

        with suppress_c_output() if self._suppress_output else nullcontext():
            Ex, Ey, Ez = fd.calculate_scattering(
                aperture_coords=self.transmitter.aperture.export_all_points(),
                sink_coords=self.receiver.aperture.export_all_points(),
                antenna_solid=self.blocker_structure.export_combined_meshio(),
                desired_E_axis=self.transmitter.aperture.excitation_function(
                    self.transmitter.polarization,
                    wavelength=self.transmitter.wavelength,
                    transmit_power=self.transmitter.power,
                ),
                scatter_points=self.scatter_points,
                wavelength=self.transmitter.wavelength,
                scattering=self.scatter,
                project_vectors=False,
                elements=True,
                cuda=self.cuda,
            )

        return np.array([Ex[0][1::]]), np.array([Ey[0][1::]]), np.array([Ez[0][1::]])
    
    
    def calculate_scene_los(self, chunks=1):

        self.set_scene()

        with suppress_c_output() if self._suppress_output else nullcontext():
            Ex, Ey, Ez = fd.calculate_scattering(
                aperture_coords=self.transmitter.aperture.export_all_points(),
                sink_coords=self.scatter_points,
                antenna_solid=self.blocker_structure.export_combined_meshio(),
                desired_E_axis=self.transmitter.aperture.excitation_function(
                    self.transmitter.polarization,
                    wavelength=self.transmitter.wavelength,
                    transmit_power=self.transmitter.power,
                ),
                scatter_points=self.scatter_points,
                wavelength=self.transmitter.wavelength,
                scattering=0,
                project_vectors=False,
                elements=True,
                cuda=self.cuda,
                chunks=chunks,
            )

        return Ex, Ey, Ez
    
    def sample_receiver_scattering(self, locations, orientations):

        sample_scatter = []
        sample_meshes = create_mesh_copies(self.receiver.points_mesh.meshio_mesh, locations, orientations)

        for vec, rot in tqdm(
            zip(locations, orientations),
            desc="Processing Sample Points: Scatter",
            total=len(locations),
            file=sys.stderr,
            dynamic_ncols=True,
            disable=False,
        ):

            self.receiver.translate_to(vec)
            self.receiver.rotate(rot, center=self.receiver.points_mesh.center)

            scatter = self.calculate_receiver_scattering()
            sample_scatter.append(scatter)
            
            self.receiver.rotate(-rot, center=self.receiver.points_mesh.center)
            

        return sample_scatter, sample_meshes
    
    
    def sample_transmitter_los(self, locations, orientations):

        sample_los = []
        sample_meshes = create_mesh_copies(self.transmitter.points_mesh.meshio_mesh, locations, orientations)

        for vec, rot in tqdm(
            zip(locations, orientations),
            desc="Processing Sample Points: LOS",
            total=len(locations),
            file=sys.stderr,
            dynamic_ncols=True,
            disable=False,
        ):

            self.transmitter.translate_to(vec)
            self.transmitter.rotate(rot, center=self.transmitter.points_mesh.center)

            scatter = self.calculate_scene_los()
            sample_los.append(scatter)

            self.transmitter.rotate(-rot, center=self.transmitter.points_mesh.center)

        return sample_los, sample_meshes
