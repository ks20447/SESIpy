import numpy as np
from sesipy.plotting import Plot3D
from sesipy.utils import ArrayFactory
from sesipy.simulation.worlds import Indoor
from sesipy.engines.spatial_intelligence import (
    IsotropicReceiver,
    PointSource,
    TransmitterArray,
    Scene,
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
    scene.add_scatterers(world.scatterers)

    return scene


def main():

    world = Indoor(scatter_resolution=0.2)

    transmitter = initialise_transmitter()
    receiver = initialise_receiver()

    transmitter.translate_to(np.array([0.0, 0.0, 0.5]))
    receiver.translate_to(np.array([1.0, 1.0, 0.5]))

    scene = initialise_scene(world, transmitter, receiver)

    plot = Plot3D(1, 2, backend="pyvista")
    plot.set_subplot_ratios(None, (1, 1))

    plot.set_plot(0, 0)
    plot.plot_scene(scene, scalars="Normals")

    plot.set_plot(0, 1)
    plot.plot_scatterers(scene.scatterers, normals=True)

    plot.show()


if __name__ == "__main__":
    main()
