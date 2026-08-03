import meshio
import numpy as np
import pyvista as pv
from sesipy.utils import ArrayFactory
from sesipy.plotting import Plot2D, Plot3D
from sesipy.engines.mapping import Environment
from sesipy.simulation.worlds import Outdoor
from sesipy.engines.spatial_intelligence.utils import to_dBm, scattering_power
from sesipy.data_storage import DatabaseReader, DatabasePS, DatabaseAoA, decode
from sesipy.engines.spatial_intelligence.mesh_handlers import LyceanObject
from sesipy.engines.spatial_intelligence import (
    IsotropicReceiver,
    PointSource,
    TransmitterArray,
    Scene,
)
from sesipy.engines.spatial_intelligence.angle_of_arrival import (
    extract_aoa,
    aoa_projection_2D,
    AoA,
)
from sesipy.engines.spatial_intelligence.utils import (
    smooth_point_data,
    threshold_point_data,
)

FREQ = 2.4e9
POWER = 0.1


def initialise_receiver():

    receiver = IsotropicReceiver()

    receiver.target_freq = FREQ
    receiver.steering_points = ArrayFactory.circle(200, 0.5)
    receiver.beamform_array = ArrayFactory.circle(4, receiver.target_wavelength / 4)

    return receiver


def initialise_transmitter():

    transmitter = TransmitterArray(
        FREQ, POWER, polarization=np.array([0.0, 0.0, 1.0], dtype=np.complex64)
    )
    transmitter.points = ArrayFactory.rectangle(2, 2, 5.0, 5.0)
    transmitter.normal_factory.apply("z")
    transmitter.point_area = np.array([1.0] * len(transmitter.points_mesh.points))

    return transmitter


def initialise_scene(world, transmitter, receiver):

    scene = Scene(scatter=True, cuda=True)

    scene.receiver = receiver
    scene.transmitter = transmitter
    scene.add_blockers([world.blocker_mesh])
    scene.add_scatterers(world.scatterers)

    return scene


def main():

    world = Outdoor(scatter_resolution=0.2, seed=2)
    env = Environment(world.floor_plan, world.scatter_mesh)

    transmitter = initialise_transmitter()
    receiver = initialise_receiver()
    scene = initialise_scene(world, transmitter, receiver)

    los = scene.calculate_scene_los()
    los = to_dBm(scattering_power(los)).sum(axis=0)

    scatter_mesh = world.scatter_mesh
    scatter_mesh.point_data["target_los"] = los

    smooth_point_data(scatter_mesh, "target_los")
    threshold_point_data(scatter_mesh, "target_los", los.min())

    plotter = Plot3D()
    plotter.add_mesh(scatter_mesh, "target_los")
    plotter.plot_antenna_array(transmitter)
    plotter.plot_point_normals(transmitter.points.points, transmitter.point_normals)
    plotter.show()


if __name__ == "__main__":
    main()
