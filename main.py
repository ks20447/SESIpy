import numpy as np
from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D
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
    transmitter.translate_to(np.array([0.0, 0.0, 0.5]))

    receiver = IsotropicReceiver()
    receiver.translate_to(np.array([1.0, 1.0, 0.5]))

    scene = Scene(scatter=True, cuda=True)
    scene.transmitter = transmitter

    scene.add_scatterers(world_indoor.scatterers)
    scene.add_blockers([world_indoor.blocker_mesh])

    plotter = Plot3D()
    plotter.plot_scene(scene, scalars=None, normals=False)

    scene.receiver = receiver
    scatter = scene.calculate_receiver_scattering()
    
    power = to_dBm(scattering_power(scatter))
    receiver.set_point_data("Power", power)

    plotter.plot_antenna_array(receiver, scalars="Power")

    plotter.show()


if __name__ == "__main__":
    main()
