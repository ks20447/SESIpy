import numpy as np
from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D
from sesipy.engines.mapping import Environment
from sesipy.engines import (
    PointSource,
    IsotropicReceiver,
    Scene,
    scattering_power,
    to_dBm,
    smooth_point_data,
    threshold_point_data,
)


def main():

    world_indoor = Indoor(scatter_resolution=0.25)
    env = Environment(world_indoor.floor_plan, world_indoor.blocker_mesh)

    grid = env.env2D.grid_sample_2D(1.0, buffer=-0.1, z=0.1)
    ori = np.zeros_like(grid)

    transmitter = PointSource(2.4e9, 0.1)

    scene = Scene(scatter=False, cuda=True)
    scene.transmitter = transmitter

    scene.add_scatterers(world_indoor.scatterers)
    scene.add_blockers([world_indoor.blocker_mesh])

    scatters, _ = scene.sample_transmitter_los(grid, ori)
    scatter_mesh = world_indoor.scatter_mesh

    scores = []

    for scatter in scatters:

        los = to_dBm(scattering_power(scatter)[0])

        scatter_mesh.point_data["los"] = los
        threshold_point_data(scatter_mesh, "los", np.min(los), 0.0, 1.0)
        smooth_point_data(scatter_mesh, "los")
        threshold_point_data(scatter_mesh, "los", 0.7, 0.0, 1.0)

        score = (
            np.sum(scatter_mesh.point_data["los"] > 0.0)
            / world_indoor.mesh_metadata["scatter"]["points"]
        )
        scores.append(score)

    plotter = Plot3D()

    plotter.add_mesh(world_indoor.blocker_mesh)
    plotter.add_points(grid, scalars=scores)
    plotter.show_bounds()

    plotter.show()


if __name__ == "__main__":
    main()
