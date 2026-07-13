from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot3D


def main():

    world_indoor = Indoor(scatter_resolution=1.0)
    world_outdoor = Outdoor(scatter_resolution=1.0, seed=1)

    plotter = Plot3D(shape=(2, 2))
    plotter.set_subplot_ratios((1, 2), (1, 1))

    plotter.set_plot(ind=(0, 0))

    plotter.add_mesh(world_indoor.blocker_mesh, show_edges=False)

    plotter.set_plot(ind=(1, 0))
    plotter.add_mesh(world_indoor.scatter_mesh, show_edges=True)

    plotter.set_plot(ind=(0, 1))

    plotter.add_mesh(world_outdoor.blocker_mesh, show_edges=False)

    plotter.set_plot(ind=(1, 1))
    plotter.add_mesh(world_outdoor.scatter_mesh, show_edges=True)

    plotter.show()


if __name__ == "__main__":
    main()
