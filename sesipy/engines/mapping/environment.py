import meshio
import numpy as np
import shapely as sp


class Sampler2D:

    def __init__(self, polygon):

        self.polygon = polygon

        cent_x, cent_y = self.polygon.centroid.xy
        self.centroid = np.array([cent_x[0], cent_y[0], 0.0])

    def polygon_intersect_2D(self, polygon):
        return self.polygon.intersection(polygon).buffer(0.0)

    def grid_sample_2D(self, spacing, buffer=0.0, z=0.0):

        geom = self.polygon.buffer(buffer)

        minx, miny, maxx, maxy = geom.bounds

        xs = np.arange(minx, maxx + spacing, spacing)
        ys = np.arange(miny, maxy + spacing, spacing)

        X, Y = np.meshgrid(xs, ys)

        mask = sp.contains_xy(geom, X, Y)

        Z = np.full_like(X, z, dtype=float)

        return np.column_stack([X[mask], Y[mask], Z[mask]])

    def linear_path_2D(self, start, end, n_waypoints, buffer=0.0, z=0.0):

        geom = self.polygon.buffer(buffer)

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        points = np.linspace(start, end, n_waypoints)
        z = np.full(n_waypoints, z)

        mask = sp.contains_xy(geom, points[:, 0], points[:, 1])
        points = points[mask]
        z = z[mask]

        theta = np.full(
            points.shape[0],
            np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])),
        )

        return np.column_stack((points, z)), theta

    def circular_path_2D(
        self,
        start,
        end,
        radius,
        n,
        buffer=0.0,
        z=0.0,
        right=True,
    ):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        chord = end - start
        d = np.linalg.norm(chord)
        
        if radius == "min":
            radius = d / 2
        elif d > 2 * radius:
            raise ValueError("Radius too small for given start/end points.")

        midpoint = 0.5 * (start + end)
        h = np.sqrt(radius**2 - (d / 2) ** 2)

        perp = np.array([-chord[1], chord[0]]) / d
        center = midpoint + h * perp if right else midpoint - h * perp

        a0 = np.arctan2(start[1] - center[1], start[0] - center[0])
        a1 = np.arctan2(end[1] - center[1], end[0] - center[0])

        if right:
            if a1 < a0:
                a1 += 2 * np.pi
        else:
            if a1 > a0:
                a1 -= 2 * np.pi

        angles = np.linspace(a0, a1, n)

        xy = np.column_stack(
            (
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
            )
        )

        mask = sp.contains_xy(self.polygon.buffer(buffer), xy[:, 0], xy[:, 1])
        xy = xy[mask]

        theta = np.arctan2(
            np.gradient(xy[:, 1]),
            np.gradient(xy[:, 0]),
        )

        return (
            np.column_stack(
                (
                    xy,
                    np.full(len(xy), z),
                )
            ),
            theta,
        )

    def edge_sample_2D(self, n, buffer=0.0, z=0.0):
        boundary = self.polygon.buffer(buffer, join_style="mitre").exterior
        distances = np.linspace(0, boundary.length, n, endpoint=False)

        points = np.array([boundary.interpolate(d).coords[0] for d in distances])

        return np.column_stack((points, np.full(len(points), z)))

    def random_sample_2D(self, n, buffer=0.0, z=0.0):

        geom = self.polygon.buffer(buffer)
        minx, miny, maxx, maxy = geom.bounds

        sample_points = []
        while len(sample_points) < n:
            x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
            if geom.contains(sp.Point(x, y)):
                sample_points.append([x, y, z])

        return np.array(sample_points)

    @staticmethod
    def join_samples(*samples):
        return np.vstack(samples)


class Sampler3D:

    def __init__(self, mesh):

        self.mesh = mesh

    def box_sample_3D(
        self,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=1.0,
    ):

        points = self.mesh.points

        mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
        )

        sample_points = points[mask]

        return sample_points

    @staticmethod
    def points_mesh(points):
        return meshio.Mesh(
            points=sample_points,
            cells=[("vertex", np.arange(len(sample_points)).reshape(-1, 1))],
        )

    @staticmethod
    def join_samples(*samples):
        return np.vstack(samples)


class Environment:

    def __init__(self, polygon=None, mesh=None):

        self.env2D = Sampler2D(polygon)
        self.env3D = Sampler3D(mesh)

    def linear_path_3D(
        self,
        start,
        end,
        n_waypoints,
    ):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        points = np.linspace(start, end, n_waypoints)

        dx, dy = end[:2] - start[:2]
        theta = np.degrees(np.arctan2(dy, dx))

        mask = sp.contains_xy(self.env2D.polygon, points[:, 0], points[:, 1])
        points = points[mask]

        theta = np.full(points.shape[0], theta)

        orientations = np.column_stack(
            (
                np.cos(np.radians(theta)),
                np.sin(np.radians(theta)),
                np.zeros_like(theta),
            )
        )

        return points, orientations
