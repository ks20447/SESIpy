import meshio
import numpy as np
from dataclasses import dataclass


@dataclass(slots=True)
class Obstacle:
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    theta: float


def rotation_matrix_z(theta_deg):
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )


def create_plane_mesh(x, y, z, length, width, plane="xy", flip_normals=False):
    points = _plane_vertices(x, y, z, length, width, plane)

    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ]
    )

    if not flip_normals:
        triangles = triangles[:, ::-1]

    return meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
    )


def _plane_vertices(x, y, z, length, width, plane):
    l = length / 2
    w = width / 2

    plane_specs = {
        "xy": np.array(
            [
                [x - l, y - w, z],
                [x + l, y - w, z],
                [x + l, y + w, z],
                [x - l, y + w, z],
            ]
        ),
        "xz": np.array(
            [
                [x - l, y, z - w],
                [x + l, y, z - w],
                [x + l, y, z + w],
                [x - l, y, z + w],
            ]
        ),
        "yz": np.array(
            [
                [x, y - l, z - w],
                [x, y + l, z - w],
                [x, y + l, z + w],
                [x, y - l, z + w],
            ]
        ),
    }

    try:
        return plane_specs[plane]
    except KeyError as exc:
        raise ValueError("plane must be one of 'xy', 'xz', or 'yz'") from exc


def plane_parameters(x, y, z, length, width, plane):
    l = length / 2
    w = width / 2

    plane_specs = {
        "xy": (
            np.array([x - l, y - w, z]),
            np.array([length, 0, 0]),
            np.array([0, width, 0]),
        ),
        "xz": (
            np.array([x - l, y, z - w]),
            np.array([length, 0, 0]),
            np.array([0, 0, width]),
        ),
        "yz": (
            np.array([x, y - l, z - w]),
            np.array([0, length, 0]),
            np.array([0, 0, width]),
        ),
    }

    try:
        return plane_specs[plane]
    except KeyError as exc:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'") from exc


def create_grid_plane_mesh(origin, u, v, nu, nv, flip_normals=False):
    nu = int(nu)
    nv = int(nv)

    su = np.linspace(0, 1, nu + 1)
    sv = np.linspace(0, 1, nv + 1)

    U, V = np.meshgrid(su, sv, indexing="ij")
    points = origin + U[..., None] * u + V[..., None] * v
    points = points.reshape(-1, 3)

    triangles = []

    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = a + 1
            c = a + (nv + 1)
            d = c + 1

            triangles.extend(
                [
                    [a, b, d],
                    [a, d, c],
                ]
            )

    triangles = np.asarray(triangles)

    if flip_normals:
        triangles = triangles[:, ::-1]

    return meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
    )


def create_box_mesh(x, y, z, width, height, length, theta):
    dx = width / 2
    dy = height / 2
    dz = length / 2

    points = np.array(
        [
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz],
        ]
    )

    R = rotation_matrix_z(theta)
    points = points @ R.T
    points += [x, y, z]

    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ]
    )

    return meshio.Mesh(points, [("triangle", triangles)])


def create_grid_box_mesh(x, y, z, width, height, length, theta, res):
    R = rotation_matrix_z(theta)

    def rotate_mesh(mesh):
        points = mesh.points @ R.T
        points += np.array([x, y, z], dtype=float)
        return meshio.Mesh(
            points=points,
            point_data=mesh.point_data,
            cells=mesh.cells,
        )

    meshes = []

    faces = [
        # bottom/top (local XY)
        (
            np.array([-width / 2, -height / 2, -length / 2]),
            np.array([width, 0, 0]),
            np.array([0, height, 0]),
            False,
        ),
        (
            np.array([-width / 2, -height / 2, length / 2]),
            np.array([width, 0, 0]),
            np.array([0, height, 0]),
            True,
        ),
        # front/back (local XZ)
        (
            np.array([-width / 2, -height / 2, -length / 2]),
            np.array([width, 0, 0]),
            np.array([0, 0, length]),
            True,
        ),
        (
            np.array([-width / 2, height / 2, -length / 2]),
            np.array([width, 0, 0]),
            np.array([0, 0, length]),
            False,
        ),
        # left/right (local YZ)
        (
            np.array([-width / 2, -height / 2, -length / 2]),
            np.array([0, height, 0]),
            np.array([0, 0, length]),
            False,
        ),
        (
            np.array([width / 2, -height / 2, -length / 2]),
            np.array([0, height, 0]),
            np.array([0, 0, length]),
            True,
        ),
    ]

    for origin, u, v, flip in faces:
        nu = max(1, int(np.ceil(np.linalg.norm(u) / res)))
        nv = max(1, int(np.ceil(np.linalg.norm(v) / res)))

        mesh = create_grid_plane_mesh(
            origin,
            u,
            v,
            nu,
            nv,
            flip_normals=flip,
        )

        meshes.append(rotate_mesh(mesh))

    return meshes
