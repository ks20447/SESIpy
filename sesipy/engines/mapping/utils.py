import os
import cv2
import yaml
import numpy as np
from shapely.geometry import Polygon


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