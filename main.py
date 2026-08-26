import numpy as np
import open3d as o3d
import shapely as sp
from sesipy.simulation.worlds import Indoor
from sesipy.plotting import Plot2D, Plot3D
from sesipy.engines.mapping import (
    Environment,
    extract_lidar_metadata,
    map_yaml_to_polygon,
)
from sesipy.engines.mapping.utils import (
    cluster_pointcloud,
    remove_small_holes,
    remove_boundary_points,
    mesh_error,
)
from sesipy.simulation.worlds import WorldDescriptor, WorldBuilder
from scipy.spatial.distance import cdist


def main():

    world = Indoor(scatter_resolution=0.2)
    env = Environment(world.floor_plan, world.scatter_mesh)

    lidar = o3d.io.read_point_cloud("pointcloud.pcd")

    lidar_downsampled = lidar.voxel_down_sample(voxel_size=0.1)
    o3d.visualization.draw_geometries([lidar_downsampled])

    lidar_meta = extract_lidar_metadata(
        lidar_downsampled.points, eps=0.5, min_samples=20, normal_radius=0.5
    )

    obj_footprints = [ob["footprint"].exterior.coords for ob in lidar_meta["objects"]]
    lidar_map = sp.Polygon(lidar_meta["boundary"].exterior.coords, holes=obj_footprints)

    yaml_map = map_yaml_to_polygon("indoor_map.yaml")
    yaml_map = yaml_map.simplify(0.1)
    yaml_map = remove_small_holes(yaml_map, 0.1)

    plot_map = Plot2D(1, 3)

    plot_map.set_ax(0, 0)
    plot_map.plot_polygon(env.env2D.polygon, fill_holes=True)

    plot_map.set_ax(0, 1)
    plot_map.plot_polygon(lidar_map, fill_holes=True)

    plot_map.set_ax(0, 2)
    plot_map.plot_polygon(yaml_map, fill_holes=True)

    plot_map.show()

    object_points = remove_boundary_points(
        lidar_downsampled.points,
        yaml_map,
        lidar_meta["floor_height"],
        lidar_meta["roof_height"],
        height_tolerance=0.1,
        boundary_tolerance=0.5,
    )

    object_clusters = cluster_pointcloud(object_points, eps=0.6)
    object_metas = [extract_lidar_metadata(o) for o in object_clusters]
    object_centers = np.array(
        [cluster[:, :2].mean(axis=0) for cluster in object_clusters]
    )
    object_heights = [m["roof_height"] for m in object_metas]

    interior_centers = np.array(
        [sp.Polygon(interior).centroid.coords[0] for interior in yaml_map.interiors]
    )

    distances = cdist(object_centers, interior_centers)
    matches = np.argmin(distances, axis=1)

    ordered_heights = np.empty(len(interior_centers))
    for ob_idx, interior_idx in enumerate(matches):
        ordered_heights[interior_idx] = object_heights[ob_idx]

    world_desc = WorldDescriptor(
        floor=True, roof=False, walls=False, boundary_z=lidar_meta["roof_height"]
    )
    world_desc.build_from_polygon(yaml_map, obstacle_heights=ordered_heights)

    indoor_recon = WorldBuilder(params=world_desc.get_data(), scatter_resolution=0.2)

    plotter = Plot3D(1, 2)

    plotter.set_plot(0, 0)
    plotter.plot_scatterers(world.scatterers)

    plotter.set_plot(0, 1)
    plotter.plot_scatterers(indoor_recon.scatterers)

    plotter.show()


if __name__ == "__main__":
    main()
