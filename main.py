import meshio
import numpy as np
from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D
from sesipy.utils import ArrayFactory
from sesipy.engines.spatial_intelligence import extract_aoa, aoa_projection_2D
from sesipy.engines import (
    PointSource,
    IsotropicReceiver,
    Scene,
    scattering_power,
    to_dBm,
    Environment,
)


def main():

    world_indoor = Indoor(scatter_resolution=0.2)

    env = Environment(polygon=world_indoor.floor_plan, mesh=world_indoor.scatter_mesh)

    transmitter = PointSource(2.4e9, 1.0)
    transmitter.translate_to(np.array([0.0, 0.0, 0.5]))

    receiver = IsotropicReceiver()
    receiver.target_freq = transmitter.freq
    receiver.steering_points = ArrayFactory.circle(200, 0.5)
    receiver.beamform_array = ArrayFactory.circle(4, transmitter.wavelength / 4)

    array_loc = np.array([-5.0, -5.0, 0.5])
    array_rot = np.radians([0.0, 0.0, 0.0])

    plotter = Plot2D(1, 2)

    for val in [False, True]:
        scene = Scene(scatter=val, cuda=True)

        scene.receiver = receiver
        scene.transmitter = transmitter
        scene.add_blockers([world_indoor.blocker_mesh])
        scene.add_scatterers([world_indoor.scatter_mesh])

        array_points = receiver.beamform_array + array_loc
        array_points = ArrayFactory.rotate(
            array_points, rotation=array_rot, center=array_loc
        )

        scatter, _ = scene.sample_receiver_scattering(
            array_points, [array_rot] * len(array_points)
        )
        mean_scatter = np.array([np.mean(scat, axis=2) for scat in scatter]).T[0]

        steering_mesh = receiver.wave_front_steering(array_points, mean_scatter)

        aoa = extract_aoa(steering_mesh, 0.5)
        fov = aoa_projection_2D(array_loc[0:2], aoa, length=100)
        fov_intersect = env.polygon_intersect_2D(fov)

        plotter.set_ax(0, 0)
        plotter.plot_fov(array_loc[0:2], fov_intersect)

        plotter.set_ax(0, 1)
        plotter.plot_aoa(
            steering_mesh.point_data["Theta"], steering_mesh.point_data["Power"]
        )

    plotter.set_ax(0, 0)
    plotter.plot_polygon(world_indoor.floor_plan, fill_holes=True)

    plotter.show()


if __name__ == "__main__":
    main()
