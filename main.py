import numpy as np
import pyvista as pv
import shapely as sp
from sesipy.engines.mapping import (
    map_yaml_to_polygon,
    simulate_lidar,
    clean_lidar,
    extract_lidar_metadata,
)
from sesipy.simulation.worlds import Indoor, WorldBuilder, WorldDescriptor
from sesipy.plotting import Plot2D, Plot3D


def main():

    indoor_world = Indoor()
    pv_meshes = [pv.from_meshio(mesh) for mesh in indoor_world.blockers]

    scan1 = simulate_lidar(
        pv_meshes, np.array([0.0, 0.0, 0.05]), v_fov=90, h_resolution=0.5
    )
    scan2 = simulate_lidar(
        pv_meshes, np.array([15.0, 9.0, 0.05]), v_fov=90, h_resolution=0.5
    )
    scan3 = simulate_lidar(
        pv_meshes, np.array([-15.0, -9.0, 0.05]), v_fov=90, h_resolution=0.5
    )
    scan4 = simulate_lidar(
        pv_meshes, np.array([15.0, -9.0, 0.05]), v_fov=90, h_resolution=0.5
    )
    scan5 = simulate_lidar(
        pv_meshes, np.array([-15.0, 9.0, 0.05]), v_fov=90, h_resolution=0.5
    )

    scan_points = np.vstack(
        [
            scan1["points"][scan1["valid"]],
            scan2["points"][scan2["valid"]],
            scan3["points"][scan3["valid"]],
            scan4["points"][scan4["valid"]],
            scan5["points"][scan5["valid"]],
        ]
    )

    lidar = clean_lidar(scan_points)
    meta = extract_lidar_metadata(lidar, eps=0.2)

    holes = [ob["footprint"].exterior.coords for ob in meta["objects"]]
    world_polygon = sp.Polygon(meta["boundary"].exterior.coords, holes=holes)

    recreate_world = WorldDescriptor(
        floor=True, roof=False, walls=False, boundary_z=meta["roof_height"]
    )
    recreate_world.build_from_polygon(
        world_polygon, [ob["height"] for ob in meta["objects"]]
    )

    world = WorldBuilder(recreate_world.get_data())

    plotter = Plot3D()
    plotter.plot_blockers(world.scatterers, color_b="r")

    for pts in [ob["points"] for ob in meta["objects"]]:
        plotter.add_points(pts)

    plotter.show()


if __name__ == "__main__":
    main()
