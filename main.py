import meshio
import numpy as np
from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D
from sesipy.utils import ArrayFactory
from sesipy.engines import (
    PointSource,
    IsotropicReceiver,
    Scene,
    scattering_power,
    to_dBm,
)


def main():

    world_indoor = Indoor(scatter_resolution=0.25)

    transmitter = PointSource(2.4e9, 0.1)
    transmitter.translate_to(np.array([0.0, 0.0, 0.5]))

    receiver = IsotropicReceiver()
    receiver.target_freq = transmitter.freq
    receiver.steering_points = ArrayFactory.circle(200, 0.5)
    receiver.beamform_array = ArrayFactory.circle(4, transmitter.wavelength / 4)

    array_loc = np.array([5.0, 5.0, 0.5])
    array_rot = np.radians([0.0, 0.0, 0.0])

    scene = Scene(scatter=True, cuda=True)

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

    plotter = Plot3D()

    plotter.plot_blockers(scene.blockers)
    plotter.plot_antenna_array(transmitter)

    plotter.add_points(array_points)
    plotter.plot_indicator_line(array_loc, "red")
    plotter.add_mesh(steering_mesh, scalars="Power")

    plotter.show()


if __name__ == "__main__":
    main()
