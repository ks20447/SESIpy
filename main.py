import numpy as np
from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D
from sesipy.engines import TransmitterArray, ReceiverArray


def main():

    world_indoor = Indoor(scatter_resolution=1.0)

    transmitter = TransmitterArray(
        2.4e9, 0.1, np.array([0.0, 0.0, 0.0], dtype=np.complex64)
    )
    transmitter.points = np.array([[0.0, 0.0, 0.0]])
    transmitter.structure = None
    transmitter.point_normals = np.array([[0.0, 0.0, 1.0]])
    transmitter.point_area = np.array([1.0])
    
    transmitter.translate_to(np.array([0.0, 0.0, 0.5]))

    receiver = ReceiverArray(gain=1.0)
    receiver.points = np.array([[0.0, 0.0, 0.0]])
    receiver.structure = None
    receiver.point_normals = np.array([[0.0, 0.0, 1.0]])
    receiver.point_area = np.array([1.0])
    
    receiver.translate_to(np.array([1.0, 1.0, 0.5]))
    
    plotter = Plot3D()
    plotter.add_mesh(world_indoor.blocker_mesh)
    plotter.plot_antenna_array(transmitter, color="red")
    plotter.plot_antenna_array(receiver, color="blue")
    
    plotter.show()


if __name__ == "__main__":
    main()
