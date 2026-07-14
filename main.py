import meshio
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
    threshold_point_data,
    smooth_point_data,
)


def main():

    world_indoor = Indoor(scatter_resolution=1.0)

    transmitter = PointSource(2.4e9, 0.1)

    sample_locations = ArrayFactory.rectangle(5, 5, 2.5, height=0.5)
    sample_orientations = np.zeros_like(sample_locations)

    scene = Scene(scatter=True, cuda=True)
    scene.transmitter = transmitter

    scene.add_scatterers(world_indoor.scatterers)
    scene.add_blockers([world_indoor.blocker_mesh])

    los_samples, meshes = scene.sample_transmitter_los(
        sample_locations, sample_orientations
    )

    scatter_mesh = world_indoor.scatter_mesh

    los_meshes = [
        meshio.Mesh(
            points=scatter_mesh.points.copy(),
            cells=[(cell.type, cell.data.copy()) for cell in scatter_mesh.cells],
        )
        for _ in range(len(sample_locations))
    ]

    for los_mesh, los in zip(los_meshes, los_samples):
        power = to_dBm(scattering_power(los)[0])
        los_mesh.point_data["los"] = power
        threshold_point_data(los_mesh, "los", np.min(power), 0.0, 1.0)
        smooth_point_data(los_mesh, "los")
        threshold_point_data(los_mesh, "los", 0.5, 0.0, 1.0)

    combined_los = np.sum([los.point_data["los"] for los in los_meshes], axis=0)

    scatter_mesh.point_data["los"] = combined_los

    plotter = Plot3D()
    plotter.add_mesh(scatter_mesh, scalars="los")
    plotter.add_points(sample_locations)
    plotter.show()


if __name__ == "__main__":
    main()
