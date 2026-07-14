import numpy as np
from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D
from sesipy.engines import PointSource, IsotropicReceiver


def main():

    world_indoor = Indoor(scatter_resolution=1.0)

    transmitter = PointSource(2.4e9, 0.1)
    transmitter.translate_to(np.array([0.0, 0.0, 0.5]))

    receiver = IsotropicReceiver()
    receiver.translate_to(np.array([1.0, 1.0, 0.5]))

    plotter = Plot3D()
    plotter.add_mesh(world_indoor.blocker_mesh)
    plotter.plot_antenna_array(transmitter, normals=True, color="red")
    plotter.plot_antenna_array(receiver, normals=True, color="blue")

    plotter.show()


if __name__ == "__main__":
    main()
