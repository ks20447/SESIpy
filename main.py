import numpy as np
from sesipy.simulation import World, Indoor, Outdoor
from sesipy.plotting import Plot2D, Plot3D
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

    transmitter = PointSource(2.4e9, 0.1)
    transmitter.translate_to(np.array([15.0, -10.0, 0.5]))

    scene = Scene(scatter=False, cuda=True)
    scene.transmitter = transmitter

    scene.add_scatterers(world_indoor.scatterers)
    scene.add_blockers([world_indoor.blocker_mesh])

    scatter = scene.calculate_scene_los(chunks=1)
    los = to_dBm(scattering_power(scatter)[0])

    scatter_mesh = world_indoor.scatter_mesh
    scatter_mesh.point_data["los"] = los
    threshold_point_data(scatter_mesh, "los", np.min(los), 0.0, 1.0)
    smooth_point_data(scatter_mesh, "los")
    threshold_point_data(scatter_mesh, "los", 0.7, 0.0, 1.0)

    plotter = Plot3D()

    plotter.add_mesh(scatter_mesh, scalars="los")
    plotter.show_bounds()

    plotter.show()


if __name__ == "__main__":
    main()
