import meshio
import numpy as np
import shapely as sp


class Environment:

    def __init__(self, polygon=None, mesh=None):

        self.env2D = polygon
        self.env3D = mesh

    def grid_sample_2D(self, spacing, buffer=0.0, z=0.0):

        geom = self.env2D.buffer(buffer)

        minx, miny, maxx, maxy = geom.bounds

        xs = np.arange(minx, maxx + spacing, spacing)
        ys = np.arange(miny, maxy + spacing, spacing)

        X, Y = np.meshgrid(xs, ys)

        mask = sp.contains_xy(geom, X, Y)

        Z = np.full_like(X, z, dtype=float)

        return np.column_stack([X[mask], Y[mask], Z[mask]])

    def linear_path_2D(self, start, end, n_waypoints, buffer=0.0, z=0.0):
        
        geom = self.env2D.buffer(buffer)
        
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

        mask = sp.contains_xy(self.env2D, points[:, 0], points[:, 1])
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
    
    
    def box_sample_3D(
        self,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=1.0,
    ):

        points = self.env3D.points

        mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
        )

        sample_points = points[mask]

        return meshio.Mesh(
            points=sample_points,
            cells=[("vertex", np.arange(len(sample_points)).reshape(-1, 1))],
        )
