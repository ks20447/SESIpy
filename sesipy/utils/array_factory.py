import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class ArrayFactory:

    @staticmethod
    def rectangle(nx: int, ny: int, spacing: float, height: float = 0.0) -> np.ndarray:
        x = np.linspace(-spacing * nx, spacing * nx, nx)
        y = np.linspace(-spacing * ny, spacing * ny, ny)

        X, Y = np.meshgrid(x, y, indexing="xy")

        return np.column_stack(
            [
                X.ravel(),
                Y.ravel(),
                np.full(X.size, height),
            ]
        )

    @staticmethod
    def cubic(nx: int, ny: int, nz: int, spacing: float) -> np.ndarray:
        x = np.linspace(-spacing * nx, spacing * nx, nx)
        y = np.linspace(-spacing * ny, spacing * ny, ny)
        z = np.linspace(-spacing * nz, spacing * nz, nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

        return np.column_stack(
            [
                X.ravel(),
                Y.ravel(),
                Z.ravel(),
            ]
        )

    @staticmethod
    def circle(n_elements: int, radius: float, height: float = 0.0) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n_elements, endpoint=False)

        return np.column_stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.full(n_elements, height),
            ]
        )

    @staticmethod
    def spherical(
        n_azimuth: int,
        n_elevation: int,
        radius: float,
    ) -> np.ndarray:
        azimuth = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
        elevation = np.linspace(
            -np.pi / 2,
            np.pi / 2,
            n_elevation + 2,
        )[1:-1]

        az, el = np.meshgrid(azimuth, elevation, indexing="ij")

        x = radius * np.cos(el) * np.cos(az)
        y = radius * np.cos(el) * np.sin(az)
        z = radius * np.sin(el)

        points = np.column_stack((
            x.ravel(),
            y.ravel(),
            z.ravel(),
        ))

        return np.vstack((
            points,
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ))

    @staticmethod
    def offset(
        offset: float,
        separation: float,
        height: float = 0.0,
    ) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.15, height],
                [0.0, -0.15, height],
                [
                    -separation * np.cos(offset - np.pi / 2),
                    0.15 + separation * np.sin(offset - np.pi / 2),
                    height,
                ],
                [
                    -separation * np.cos(offset - np.pi / 2),
                    -(0.15 + separation * np.sin(offset - np.pi / 2)),
                    height,
                ],
            ]
        )

    @staticmethod
    def expand(points: np.ndarray, offset: float):

        center = points.mean(axis=0)
        directions = points - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)

        directions /= np.maximum(norms, 1e-12)

        return points + offset * directions

    @staticmethod
    def rotate(points, rotation, center=np.zeros((1, 3))):

        points = np.asarray(points, dtype=float)
        center = np.asarray(center, dtype=float)
        a, b, c = rotation

        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)

        rx = np.array(
            [
                [1, 0, 0],
                [0, ca, -sa],
                [0, sa, ca],
            ]
        )

        ry = np.array(
            [
                [cb, 0, sb],
                [0, 1, 0],
                [-sb, 0, cb],
            ]
        )

        rz = np.array(
            [
                [cc, -sc, 0],
                [sc, cc, 0],
                [0, 0, 1],
            ]
        )

        rotation = rz @ ry @ rx

        return (points - center) @ rotation.T + center

    @staticmethod
    def classify_dimensionality(points, tol=1e-3):
        pts = np.asarray(points)
        if len(pts) == 1:
            return False

        c = pts.mean(axis=0)
        x = pts - c

        cov = np.cov(x, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(eig)

        is_3d = (eig[0] / eig[-1]) > tol
        return is_3d