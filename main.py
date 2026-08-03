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

    transmitter = PointSource(FREQ, POWER)

    return transmitter


def initialise_scene(world, transmitter, receiver):

    scene = Scene(scatter=True, cuda=True)

    scene.receiver = receiver
    scene.transmitter = transmitter
    scene.add_blockers([world.blocker_mesh])
    scene.add_scatterers([world.scatterers[-1]])

    return scene


def main():

    world = Outdoor(scatter_resolution=0.2, seed=2)
    env = Environment(world.floor_plan, world.scatter_mesh)

    transmitter = initialise_transmitter()
    receiver = initialise_receiver()
    scene = initialise_scene(world, transmitter, receiver)

    np.random.seed(10)
    transmitter.translate_to(env.env2D.random_sample_2D(1, buffer=-0.5, z=0.5)[0])

    path, theta = env.env2D.linear_path_2D((-19, 14), (19, -14), 10, buffer=-1.0, z=0.5)

    idx = 3
    loc, rot = path[idx], theta[idx]

    array_rot = np.array([0.0, 0.0, rot])

    array_points = receiver.beamform_array + loc
    array_points = ArrayFactory.rotate(array_points, array_rot, loc)

    scatter, _ = scene.sample_receiver_scattering(
        array_points, [array_rot] * len(array_points)
    )
    mean_scatter = np.array([np.mean(scat, axis=2) for scat in scatter]).T[0]
    steering_mesh = receiver.wave_front_steering(array_points, mean_scatter)

    aoa = extract_aoa(steering_mesh, drop_dB=0.1)

    fov = aoa_projection_2D(loc[0:2], aoa, length=100)
    intersect_fov = env.env2D.polygon_intersect_2D(fov)

    fov_mask = env.env3D.footprint_sample_3D(
        intersect_fov, z_min=0.1, z_max=3.0, mask=True
    )

    transmitter.translate_to(loc)

    del scene.scatterers
    scene.add_scatterers(world.scatterers)

    los = scene.calculate_scene_los()
    los = to_dBm(scattering_power(los))[0]

    scatter_ob = LyceanObject(world.scatter_mesh)
    scatter_ob.initialise_mesh()
    scatter_mesh = scatter_ob.meshio_mesh

    scatter_mesh.point_data["target_los"] = los

    smooth_point_data(scatter_mesh, "target_los")
    threshold_point_data(scatter_mesh, "target_los", los.min())
    scatter_mesh.point_data["target_los"][~fov_mask] = 0.0
    point_mask = scatter_mesh.point_data["target_los"] > 0.0

    sample_points = scatter_mesh.points[point_mask][0::10]
    scatter_mesh.point_data["los_sampling"] = np.zeros(scatter_mesh.points.shape[0])
    
    los_samples, _ = scene.sample_transmitter_los(
        sample_points, np.array([[0.0, 0.0, 0.0]] * len(sample_points))
    )

    for sample in los_samples:
        
        power = to_dBm(scattering_power(sample))[0]
        scatter_mesh.point_data["los_sampling"] += power

    smooth_point_data(scatter_mesh, "los_sampling")
    threshold_point_data(scatter_mesh, "los_sampling", scatter_mesh.point_data["los_sampling"].min())

    plotter = Plot3D()
    plotter.add_mesh(scatter_mesh, "los_sampling")
    plotter.add_points(sample_points, color="red")
    plotter.show()


if __name__ == "__main__":
    main()
