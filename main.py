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

    world_indoor = Indoor(scatter_resolution=1.0)

    transmitter = PointSource(2.4e9, 0.1)
    transmitter.translate_to(np.array([5.5, 5.5, 0.5]))

    receiver = IsotropicReceiver()

    sample_locations = ArrayFactory.rectangle(10, 10, 0.5, height=0.5)
    sample_orientations = np.zeros_like(sample_locations)

    scene = Scene(scatter=True, cuda=True)
    scene.transmitter = transmitter
    scene.receiver = receiver

    scene.add_scatterers(world_indoor.scatterers)
    scene.add_blockers([world_indoor.blocker_mesh])

    scatters, meshes = scene.sample_receiver_scattering(
        sample_locations, sample_orientations
    )
    powers = [to_dBm(scattering_power(scatter)) for scatter in scatters]

    plotter = Plot3D()
    plotter.plot_antenna_array(transmitter)
    plotter.plot_blockers(scene.blockers)
    plotter.add_points(
        sample_locations,
        scalars=np.array([np.mean(power) for power in powers]),
        point_size=20,
    )

    plotter.show()


if __name__ == "__main__":
    main()
