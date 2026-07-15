import meshio
import numpy as np
from sesipy.simulation.worlds.worlds import Indoor
from sesipy.plotting.plot2D import Plot2D
from sesipy.plotting.plot3D import Plot3D
from sesipy.engines.mapping.environment import Environment


def main():

    world_indoor = Indoor(scatter_resolution=0.25)

    env = Environment(polygon=world_indoor.floor_plan, mesh=world_indoor.scatter_mesh)

    grid = env.grid_sample_2D(1.0, buffer=-0.5)
    path, angles = env.linear_path_2D((-20, -20), (20, 20), 20, buffer=-1.0)

    box = env.box_sample_3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    path3D, theta = env.linear_path_3D((0, 0, 0), (1, 1, 1), 3)
    vectors = np.column_stack(
        (
            np.cos(np.radians(theta)),
            np.sin(np.radians(theta)),
            np.zeros_like(theta),
        )
    )

    plotter = Plot2D(1, 1)
    plotter.plot_polygon(env.env2D)
    plotter.plot_scatter(grid[:, 0:2])
    plotter.plot_waypoints(path[:, 0:2], angles)
    plotter.show()

    plotter = Plot3D(1, 1)
    plotter.plot_scatterers(world_indoor.scatterers)
    plotter.add_mesh(box, color="red")
    plotter.plot_path(path3D, vectors)
    plotter.show()


if __name__ == "__main__":
    main()
