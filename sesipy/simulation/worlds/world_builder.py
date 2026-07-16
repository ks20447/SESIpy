import cv2
import math
import yaml
import json
import shapely as sp
import pyvista as pv
from .utils import *
from shapely import wkb
from pathlib import Path
from dataclasses import dataclass


@dataclass(slots=True)
class Wall:
    x: float
    y: float
    width: float
    theta: float


@dataclass(slots=True)
class Rectangle:
    x: float
    y: float
    width: float
    length: float
    theta: float


class WorldDescriptor:

    def __init__(self, floor, roof, walls, boundary_z):

        self.floor = floor
        self.roof = roof
        self.walls = walls
        self.boundary_z = boundary_z

        self.generator = {
            "type": "fixed",
            "fixed_obstacles": [],
            "random_obstacles": {
                "small": 0,
                "medium": 0,
                "large": 0,
                "ranges": {
                    "small": [0, 0],
                    "medium": [0, 0],
                    "large": [0, 0],
                },
            },
        }

        self.meta_data = {
            "floor": self.floor,
            "roof": self.roof,
            "walls": self.walls,
            "boundary_z": self.boundary_z,
            "bound": {},
        }

    def _extract_walls(self, exterior):

        walls = []

        for p1, p2 in zip(exterior[:-1], exterior[1:]):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            walls.append(
                Wall(
                    x=(p1[0] + p2[0]) / 2,
                    y=(p1[1] + p2[1]) / 2,
                    width=math.hypot(dx, dy),
                    theta=math.atan2(dy, dx),
                )
            )

        return walls

    def _extract_rectangle_obstacles(self, interiors):

        rectangles = []

        for interior in interiors:
            hole = sp.Polygon(interior)
            rect = hole.minimum_rotated_rectangle

            pts = np.asarray(rect.exterior.coords[:-1])

            edges = np.roll(pts, -1, axis=0) - pts
            lengths = np.linalg.norm(edges, axis=1)

            unique = np.unique(np.round(lengths, 12))
            width = unique.min()
            length = unique.max()

            long_edge = edges[np.argmax(lengths)]
            theta = math.atan2(long_edge[1], long_edge[0])

            center = pts.mean(axis=0)

            rectangles.append(
                Rectangle(
                    x=center[0],
                    y=center[1],
                    width=width,
                    length=length,
                    theta=theta,
                )
            )

        return rectangles

    def build_from_polygon(self, polygon: sp.Polygon, obstacle_heights: list):

        minx, miny, maxx, maxy = polygon.bounds

        self.meta_data["bound"]["xmin"] = float(minx)
        self.meta_data["bound"]["xmax"] = float(maxx)
        self.meta_data["bound"]["ymin"] = float(miny)
        self.meta_data["bound"]["ymax"] = float(maxy)

        walls = (
            self._extract_walls(np.asarray(polygon.exterior.coords))
            if not self.walls
            else []
        )
        obstacles = self._extract_rectangle_obstacles(polygon.interiors)

        for wall in walls:

            wall_data = [
                wall.x,
                wall.y,
                self.boundary_z / 2,
                wall.width,
                0.0,
                self.boundary_z,
                np.rad2deg(wall.theta),
            ]
            self.generator["fixed_obstacles"].append(wall_data)

        for obstacle, height in zip(obstacles, obstacle_heights):

            obstacle_data = [
                obstacle.x,
                obstacle.y,
                height / 2,
                obstacle.width,
                obstacle.length,
                height,
                obstacle.theta,
            ]
            self.generator["fixed_obstacles"].append(obstacle_data)

    def write_to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.safe_dump(
                self._to_builtin(self.get_data()),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
            
    def write_to_json(self, filename, indent=4):
        with open(filename, "w") as f:
            json.dump(
                self._to_builtin(self.get_data()),
                f,
                indent=indent,
            )

    def get_data(self):
        self.meta_data["generator"] = self.generator
        return self.meta_data
    
    
    def _to_builtin(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_builtin(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._to_builtin(v) for v in obj]

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.generic):
            return obj.item()

        return obj



class World:

    def __init__(
        self,
        polygon: sp.Polygon,
        scatterers: list[meshio.Mesh],
        blockers: list[meshio.Mesh],
    ):

        self._floor_plan = polygon
        self._scatterers = scatterers
        self._blockers = blockers

        self._pgm = None
        self._pgm_params = {}

        self._combined_scatter_mesh = self.combine_meshes(self.scatterers)
        self._combined_blocker_mesh = self.combine_meshes(self.blockers)

    @property
    def floor_plan(self):
        return self._floor_plan

    @property
    def pgm(self):
        return self._pgm

    @property
    def bounds(self):
        return self.floor_plan.bounds

    @property
    def scatterers(self):
        return self._scatterers

    @property
    def blockers(self):
        return self._blockers

    @property
    def scatter_mesh(self):
        return self._combined_scatter_mesh

    @property
    def blocker_mesh(self):
        return self._combined_blocker_mesh

    @property
    def scatter_metadata(self):
        return self._mesh_list_metadata(self.scatterers)

    @property
    def blockers_metadata(self):
        return self._mesh_list_metadata(self.blockers)

    @property
    def mesh_metadata(self):
        return {
            "scatter": self._mesh_metadata(self.scatter_mesh),
            "blocker": self._mesh_metadata(self.blocker_mesh),
        }

    def _mesh_metadata(self, mesh):
        return {
            "points": len(mesh.points),
            "cells": sum(len(block.data) for block in mesh.cells),
            "point_data": list(mesh.point_data.keys()),
            "resolution": self.mesh_resolution(mesh),
        }

    def _mesh_list_metadata(self, meshes):
        return {
            "count": len(meshes),
            "points": [len(mesh.points) for mesh in meshes],
            "cells": [sum(len(block.data) for block in mesh.cells) for mesh in meshes],
            "point_data": [list(mesh.point_data.keys()) for mesh in meshes],
            "min_resolution": [self.mesh_resolution(mesh)["min"] for mesh in meshes],
            "max_resolution": [self.mesh_resolution(mesh)["max"] for mesh in meshes],
            "mean_resolution": [self.mesh_resolution(mesh)["mean"] for mesh in meshes],
        }

    def combine_meshes(self, meshes):
        combined = pv.MultiBlock([pv.from_meshio(m) for m in meshes]).combine()
        triangles = combined.cells_dict[pv.CellType.TRIANGLE]

        return meshio.Mesh(
            points=combined.points,
            point_data=combined.point_data,
            cells=[("triangle", triangles)],
        )

    def mesh_resolution(self, mesh):
        cached = getattr(mesh, "_resolution_cache", None)
        if cached is not None:
            return cached

        triangles = []
        for block in mesh.cells:
            if len(block.data) > 0:
                triangles.append(block.data)

        if not triangles:
            result = {"min": 0.0, "max": 0.0, "mean": 0.0}
            mesh._resolution_cache = result
            return result

        triangles = np.vstack(triangles)
        points = mesh.points[triangles]
        edge_lengths = np.concatenate(
            [
                np.linalg.norm(points[:, 1, :] - points[:, 0, :], axis=1),
                np.linalg.norm(points[:, 2, :] - points[:, 1, :], axis=1),
                np.linalg.norm(points[:, 0, :] - points[:, 2, :], axis=1),
            ]
        )

        result = {
            "min": float(edge_lengths.min()),
            "max": float(edge_lengths.max()),
            "mean": float(edge_lengths.mean()),
        }
        mesh._resolution_cache = result
        return result

    def save_floor_plan(self, filename):

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_bytes(wkb.dumps(self.floor_plan))

    def save_scatter_mesh(self, filename, file_format=None):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        meshio.write(path, self.scatter_mesh, file_format=file_format)

    def save_blocker_mesh(self, filename, file_format=None):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        meshio.write(path, self.blocker_mesh, file_format=file_format)

    def save_map(self, filename):

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(f"{path}.pgm", self.pgm)

        yaml_data = {
            "image": f"{path}.pgm",
            "mode": "trinary",
            "resolution": float(self._pgm_params["res"]),
            "origin": [
                float(self._pgm_params["minx"]),
                float(self._pgm_params["miny"]),
                0.0,
            ],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }

        with open(f"{path}.yaml", "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False)


class WorldBuilder(World):

    def __init__(self, params, scatter_resolution=1.0):

        self.params = params

        bound = self.params["bound"]
        self.xmin = float(bound["xmin"])
        self.xmax = float(bound["xmax"])
        self.ymin = float(bound["ymin"])
        self.ymax = float(bound["ymax"])
        self.x = float(self.xmax - self.xmin)
        self.y = float(self.ymax - self.ymin)
        self.cx = float((self.xmin + self.xmax) / 2.0)
        self.cy = float((self.ymin + self.ymax) / 2.0)

        self.floor = self.params["floor"]
        self.roof = self.params["roof"]
        self.walls = self.params["walls"]
        self.boundary_z = self.params["boundary_z"]

        generator = self.params["generator"]["type"]

        if generator == "fixed":
            self.obstacles = self.fixed_obstacles(
                self.params["generator"]["fixed_obstacles"]
            )
        elif generator == "random":
            self.obstacles = self.random_obstacles(
                self.params["generator"]["random_obstacles"]
            )
        else:
            self.obstacles = []

        self.res = scatter_resolution

        super().__init__(
            self.create_floor_plan(), self.create_scatterers(), self.create_blockers()
        )

    def _coerce_obstacle(self, obstacle):
        if isinstance(obstacle, Obstacle):
            return obstacle

        x, y, z, width, depth, height, theta = obstacle
        return Obstacle(
            x=x,
            y=y,
            z=z,
            width=width,
            depth=depth,
            height=height,
            theta=theta,
        )

    def _coerce_obstacles(self, obstacles):
        return [self._coerce_obstacle(obstacle) for obstacle in obstacles]

    def fixed_obstacles(self, obstacles):
        return self._coerce_obstacles(obstacles)

    def random_obstacles(self, params):
        obstacle_specs = [
            (params["ranges"]["small"], params["counts"]["small"]),
            (params["ranges"]["medium"], params["counts"]["medium"]),
            (params["ranges"]["large"], params["counts"]["large"]),
        ]

        obstacles = []
        for (s0, s1), count in obstacle_specs:
            obstacles.extend(self._random_obstacles_in_range(count, s0, s1))

        return obstacles

    def _random_obstacles_in_range(self, n, s0, s1):
        out = []
        for _ in range(n):
            width = np.random.uniform(s0, s1)
            depth = np.random.uniform(s0, s1)
            height = np.random.uniform(s0, s1)

            out.append(
                Obstacle(
                    x=np.random.uniform(self.xmin + width / 2, self.xmax - width / 2),
                    y=np.random.uniform(self.ymin + depth / 2, self.ymax - depth / 2),
                    z=height / 2,
                    width=width,
                    depth=depth,
                    height=height,
                    theta=np.random.uniform(0, 360),
                )
            )

        return out

    def _boundary_surface_specs(self):
        x, y = self.x, self.y
        z = self.boundary_z

        specs = []
        if self.floor:
            specs.append(
                {
                    "x": self.cx,
                    "y": self.cy,
                    "z": 0,
                    "length": x,
                    "width": y,
                    "plane": "xy",
                    "flip_normals": True,
                }
            )
        if self.roof:
            specs.append(
                {
                    "x": self.cx,
                    "y": self.cy,
                    "z": z,
                    "length": x,
                    "width": y,
                    "plane": "xy",
                    "flip_normals": False,
                }
            )
        if self.walls:
            specs.extend(
                [
                    {
                        "x": self.cx,
                        "y": self.ymax,
                        "z": z / 2,
                        "length": x,
                        "width": z,
                        "plane": "xz",
                        "flip_normals": True,
                    },
                    {
                        "x": self.cx,
                        "y": self.ymin,
                        "z": z / 2,
                        "length": x,
                        "width": z,
                        "plane": "xz",
                        "flip_normals": False,
                    },
                    {
                        "x": self.xmin,
                        "y": self.cy,
                        "z": z / 2,
                        "length": y,
                        "width": z,
                        "plane": "yz",
                        "flip_normals": True,
                    },
                    {
                        "x": self.xmax,
                        "y": self.cy,
                        "z": z / 2,
                        "length": y,
                        "width": z,
                        "plane": "yz",
                        "flip_normals": False,
                    },
                ]
            )

        return specs

    def _grid_count(self, size):
        return max(1, int(np.ceil(size / self.res)))

    def create_floor_plan(self):
        x, y = self.x, self.y

        shell = [
            (self.xmin, self.ymin),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
            (self.xmin, self.ymax),
        ]

        holes = []
        for obstacle in self.obstacles:
            box = sp.box(
                obstacle.x - obstacle.width / 2,
                obstacle.y - obstacle.depth / 2,
                obstacle.x + obstacle.width / 2,
                obstacle.y + obstacle.depth / 2,
            )
            box = sp.affinity.rotate(
                box,
                obstacle.theta,
                origin=(obstacle.x, obstacle.y),
                use_radians=False,
            )
            holes.append(box.exterior.coords[:-1])

        return sp.Polygon(shell=shell, holes=holes).simplify(1e-6).buffer(0.0)

    def create_blockers(self):
        blockers = []

        for spec in self._boundary_surface_specs():
            blockers.append(
                create_plane_mesh(
                    spec["x"],
                    spec["y"],
                    spec["z"],
                    spec["length"],
                    spec["width"],
                    plane=spec["plane"],
                    flip_normals=spec["flip_normals"],
                )
            )

        blockers.extend(
            [
                create_box_mesh(
                    obstacle.x,
                    obstacle.y,
                    obstacle.z,
                    obstacle.width,
                    obstacle.depth,
                    obstacle.height,
                    obstacle.theta,
                )
                for obstacle in self.obstacles
            ]
        )

        return blockers

    def create_scatterers(self):
        scatterers = []

        for spec in self._boundary_surface_specs():
            origin, u, v = plane_parameters(
                spec["x"],
                spec["y"],
                spec["z"],
                spec["length"],
                spec["width"],
                plane=spec["plane"],
            )
            scatterers.append(
                create_grid_plane_mesh(
                    origin,
                    u,
                    v,
                    self._grid_count(spec["length"]),
                    self._grid_count(spec["width"]),
                    flip_normals=spec["flip_normals"],
                )
            )

        scatterers.extend(
            [
                self.combine_meshes(
                    create_grid_box_mesh(
                        obstacle.x,
                        obstacle.y,
                        obstacle.z,
                        obstacle.width,
                        obstacle.depth,
                        obstacle.height,
                        obstacle.theta,
                        res=self.res,
                    )
                )
                for obstacle in self.obstacles
            ]
        )

        return scatterers

    def create_pgm(self, boundary=True):

        resolution = 0.05
        padding = 1.0

        minx, miny, maxx, maxy = self.floor_plan.bounds
        minx -= padding
        miny -= padding
        maxx += padding
        maxy += padding

        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

        img = np.full((height, width), 205, dtype=np.uint8)

        def coords_to_pixels(coords):
            pts = []
            for x, y in coords:
                px = int((x - minx) / resolution)
                py = height - int((y - miny) / resolution)
                pts.append([px, py])
            return np.array([pts], dtype=np.int32)

        ext_pts = coords_to_pixels(self.floor_plan.exterior.coords)
        cv2.fillPoly(img, ext_pts, 254)

        for interior in self.floor_plan.interiors:
            int_pts = coords_to_pixels(interior.coords)
            cv2.fillPoly(img, int_pts, 0)

        if boundary:
            cv2.polylines(img, ext_pts, isClosed=True, color=0, thickness=1)

        self._pgm = img
        self._pgm_params = {"res": resolution, "minx": minx, "miny": miny}
