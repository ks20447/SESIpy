import os
import cv2
import vtk
import yaml
import numpy as np
import pyvista as pv
from shapely import Point, Polygon, MultiPoint, LineString
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from shapely.prepared import prep



def simulate_lidar(
    meshes,
    position,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    h_fov=360.0,
    v_fov=30.0,
    h_resolution=0.2,
    v_resolution=1.0,
    max_range=100.0,
):
    position = np.asarray(position, dtype=float)

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    scene = pv.merge(meshes)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(scene)
    locator.BuildLocator()

    h_angles = np.deg2rad(np.arange(-h_fov / 2, h_fov / 2 + h_resolution, h_resolution))
    v_angles = np.deg2rad(np.arange(-v_fov / 2, v_fov / 2 + v_resolution, v_resolution))

    ch, sh = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])

    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    R = Rz @ Ry @ Rx

    points = []
    ranges = []
    directions = []
    valid = []

    hit = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    weights = [0.0] * 8
    sub_id = vtk.reference(0)

    for v in v_angles:
        cv = np.cos(v)
        sv = np.sin(v)

        for h in h_angles:
            direction = np.array(
                [
                    cv * np.cos(h),
                    cv * np.sin(h),
                    sv,
                ]
            )

            direction = R @ direction
            direction /= np.linalg.norm(direction)

            end = position + direction * max_range

            t = vtk.mutable(0.0)

            intersect = locator.IntersectWithLine(
                position,
                end,
                1e-6,
                t,
                hit,
                pcoords,
                sub_id,
                vtk.reference(0),
            )

            directions.append(direction)

            if intersect:
                p = np.array(hit)
                r = np.linalg.norm(p - position)

                points.append(p)
                ranges.append(r)
                valid.append(True)
            else:
                points.append(position + direction * max_range)
                ranges.append(max_range)
                valid.append(False)

    return {
        "points": np.asarray(points),
        "ranges": np.asarray(ranges),
        "directions": np.asarray(directions),
        "valid": np.asarray(valid),
        "horizontal_angles": h_angles,
        "vertical_angles": v_angles,
    }


def clean_lidar(
    points,
    min_range=0.0,
    max_range=np.inf,
    sensor_origin=None,
    outlier_k=10,
    outlier_std=2.0,
    cluster_radius=None,
    min_cluster_size=20,
    voxel_size=None,
):
    points = np.asarray(points, dtype=float)

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    if sensor_origin is not None:
        sensor_origin = np.asarray(sensor_origin)
        ranges = np.linalg.norm(points - sensor_origin, axis=1)
        mask = (ranges >= min_range) & (ranges <= max_range)
        points = points[mask]

    if len(points) == 0:
        return points

    if outlier_k is not None:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=outlier_k + 1)
        mean_dist = dists[:, 1:].mean(axis=1)

        threshold = mean_dist.mean() + outlier_std * mean_dist.std()

        points = points[mean_dist < threshold]

    if len(points) == 0:
        return points

    if cluster_radius is not None:
        tree = cKDTree(points)

        neighbours = tree.query_ball_tree(tree, cluster_radius)

        rows = []
        cols = []

        for i, n in enumerate(neighbours):
            rows.extend([i] * len(n))
            cols.extend(n)

        graph = csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(points), len(points)),
        )

        _, labels = connected_components(graph)

        counts = np.bincount(labels)
        keep = counts[labels] >= min_cluster_size

        points = points[keep]

    if len(points) == 0:
        return points

    if voxel_size is not None:
        voxels = np.floor(points / voxel_size).astype(np.int64)

        _, inverse = np.unique(voxels, axis=0, return_inverse=True)

        output = np.empty((inverse.max() + 1, 3))

        for i in range(len(output)):
            output[i] = points[inverse == i].mean(axis=0)

        points = output

    return points


def extract_lidar_metadata(points, floor_z_tol=0.1, roof_z_tol=0.1, boundary_tol=0.2, eps=0.3, min_samples=10):
    points = np.asarray(points)
    
    floor_height = float(np.percentile(points[:, 2], 1))
    roof_height = float(np.percentile(points[:, 2], 99))
    
    floor_mask = points[:, 2] <= (floor_height + floor_z_tol)
    roof_mask = points[:, 2] >= (roof_height - roof_z_tol)
    
    middle_points = points[~(floor_mask | roof_mask)]
    
    if len(middle_points) < 3:
        return {
            "floor_height": floor_height,
            "roof_height": roof_height,
            "boundary": None,
            "objects": []
        }
        
    xy_points = middle_points[:, :2]
    hull = ConvexHull(xy_points)
    boundary = Polygon(xy_points[hull.vertices])
    
    shrunk_boundary = boundary.buffer(-boundary_tol)
    if shrunk_boundary.is_empty:
        shrunk_boundary = boundary
        
    prep_boundary = prep(shrunk_boundary)
    interior_mask = [prep_boundary.contains(Point(pt)) for pt in xy_points]
    interior_points = middle_points[interior_mask]
    
    objects = []
    if len(interior_points) > 0:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(interior_points[:, :2])
        labels = clustering.labels_
        
        for label in set(labels):
            if label == -1:
                continue
                
            pts = interior_points[labels == label]
            pts_2d = pts[:, :2]
            
            z_min = float(np.min(pts[:, 2]))
            z_max = float(np.max(pts[:, 2]))
            centre_xy = np.mean(pts_2d, axis=0)
            
            mpt = MultiPoint(pts_2d)
            footprint = mpt.minimum_rotated_rectangle
            
            if type(footprint) == LineString:
                continue
            
            if footprint.geom_type == 'Polygon':
                coords = list(footprint.exterior.coords)
                p1, p2, p3 = np.array(coords[0]), np.array(coords[1]), np.array(coords[2])
                
                l1 = np.linalg.norm(p2 - p1)
                l2 = np.linalg.norm(p3 - p2)
                
                width = float(min(l1, l2))
                length = float(max(l1, l2))
                
                edge = p2 - p1 if l1 > l2 else p3 - p2
                theta = float(np.arctan2(edge[1], edge[0]))
            else:
                width = 0.0
                length = float(footprint.length if footprint.geom_type == 'LineString' else 0.0)
                theta = 0.0

            objects.append(
                {
                    "id": int(label),
                    "center": np.array(
                        [
                            centre_xy[0],
                            centre_xy[1],
                            (z_min + z_max) / 2,
                        ]
                    ),
                    "width": width,
                    "length": length,
                    "height": float(z_max - z_min),
                    "theta": theta,
                    "footprint": footprint,
                    "points": pts,
                }
            )

    return {
        "floor_height": floor_height,
        "roof_height": roof_height,
        "boundary": boundary,
        "objects": objects,
    }


def map_yaml_to_polygon(yaml_file):

    with open(yaml_file, "r") as f:
        map_info = yaml.safe_load(f)

    image_path = map_info["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_file), image_path)

    resolution = map_info["resolution"]
    origin = map_info["origin"][:2]

    negate = map_info.get("negate", 0)
    occupied_thresh = map_info.get("occupied_thresh", 0.65)
    free_thresh = map_info.get("free_thresh", 0.196)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(image_path)

    if negate:
        occ = img.astype(np.float32) / 255.0
    else:
        occ = 1.0 - img.astype(np.float32) / 255.0

    free = (occ < free_thresh).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(free)

    if num_labels <= 1:
        raise ValueError("No connected free-space region found.")

    counts = np.bincount(labels.ravel())
    largest = np.argmax(counts[1:]) + 1

    mask = np.zeros_like(free, dtype=np.uint8)
    mask[labels == largest] = 255

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        raise ValueError("No contours found.")

    hierarchy = hierarchy[0]
    height = img.shape[0]

    def contour_to_world(contour):
        pts = contour[:, 0].astype(np.float64)

        x = pts[:, 0] * resolution + origin[0]
        y = (height - pts[:, 1]) * resolution + origin[1]

        return np.column_stack((x, y))

    polygons = []

    for i, (_, _, child, parent) in enumerate(hierarchy):
        if parent != -1:
            continue

        exterior = contour_to_world(contours[i])

        holes = []

        hole = child
        while hole != -1:
            holes.append(contour_to_world(contours[hole]))
            hole = hierarchy[hole][0]

        poly = Polygon(exterior, holes).buffer(0)

        if not poly.is_empty:
            polygons.append(poly)

    if not polygons:
        raise ValueError("No valid polygon found.")

    return max(polygons, key=lambda p: p.area)
