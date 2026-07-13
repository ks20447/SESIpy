from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D


def main():

    world_indoor = Indoor(scatter_resolution=1.0)
    world_outdoor = Outdoor(scatter_resolution=1.0, seed=1)

    plotter = Plot2D(1, 2)

    plotter.set_ax(0, 0)
    plotter.plot_pgm(world_indoor.pgm)

    plotter.set_ax(0, 1)
    plotter.plot_pgm(world_outdoor.pgm)

    plotter.show()


if __name__ == "__main__":
    main()
