import shapely as sp
import pyvista as pv
from .utils import *
from shapely import wkb

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

        self._combined_scatter_mesh = self.combine_meshes(self.scatterers)
        self._combined_blocker_mesh = self.combine_meshes(self.blockers)

    @property
    def floor_plan(self):
        return self._floor_plan

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



class WorldBuilder(World):

    def __init__(self, params, scatter_resolution=1.0):

        self.params = params

        self.x, self.y = self.params["bound"]["x"], self.params["bound"]["y"]
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
                    x=np.random.uniform(-self.x / 2 + width, self.x / 2 - width),
                    y=np.random.uniform(-self.y / 2 + depth, self.y / 2 - depth),
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
                    "x": 0,
                    "y": 0,
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
                    "x": 0,
                    "y": 0,
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
                        "x": 0,
                        "y": y / 2,
                        "z": z / 2,
                        "length": x,
                        "width": z,
                        "plane": "xz",
                        "flip_normals": True,
                    },
                    {
                        "x": 0,
                        "y": -y / 2,
                        "z": z / 2,
                        "length": x,
                        "width": z,
                        "plane": "xz",
                        "flip_normals": False,
                    },
                    {
                        "x": -x / 2,
                        "y": 0,
                        "z": z / 2,
                        "length": y,
                        "width": z,
                        "plane": "yz",
                        "flip_normals": True,
                    },
                    {
                        "x": x / 2,
                        "y": 0,
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

        shell = [(-x / 2, -y / 2), (x / 2, -y / 2), (x / 2, y / 2), (-x / 2, y / 2)]

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
