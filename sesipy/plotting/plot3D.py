import numpy as np
import pyvista as pv
from ..utility.formatting import Symbols

class Plot3D:

    def __init__(self, shape=(1, 1)):

        self.pl = pv.Plotter(shape=shape)
        self.sym = Symbols()
        
    def set_plot(self, ind=(0, 0)):
        self.pl.subplot(*ind)
        
    def set_subplot_ratios(self, row_ratios=None, col_ratios=None):
        nrows, ncols = self.pl.shape

        if row_ratios is None:
            row_ratios = [1] * nrows
        if col_ratios is None:
            col_ratios = [1] * ncols

        row_ratios = np.asarray(row_ratios, dtype=float)
        col_ratios = np.asarray(col_ratios, dtype=float)

        row_ratios /= row_ratios.sum()
        col_ratios /= col_ratios.sum()

        x_edges = np.concatenate(([0.0], np.cumsum(col_ratios)))
        y_edges = np.concatenate(([0.0], np.cumsum(row_ratios)))

        for r in range(nrows):
            for c in range(ncols):
                idx = r * ncols + c

                xmin = x_edges[c]
                xmax = x_edges[c + 1]

                ymin = 1.0 - y_edges[r + 1]
                ymax = 1.0 - y_edges[r]

                self.pl.renderers[idx].viewport = (
                    xmin, ymin, xmax, ymax
                )
                
    def add_mesh(self, mesh, scalars=None, **kwargs):
        self.pl.add_mesh(mesh, scalars=scalars, **kwargs)
        
    def add_points(self, points, scalars=None, point_size=10, **kwargs):
        self.pl.add_points(
            points,
            scalars=scalars,
            point_size=point_size,
            render_points_as_spheres=True,
            **kwargs,
        )

    def show_bounds(self):
        self.pl.show_bounds()

    def show_origin(self, labels=False):
        if not labels:
            self.pl.add_axes_at_origin(xlabel="", ylabel="", zlabel="")
        else:
            self.pl.add_axes_at_origin()

    @property
    def plotter(self):
        return self.pl

    def show(self, filename=None):
        if filename is not None:
            self.pl.show(auto_close=False)
            self.pl.save_graphic(filename)
            self.pl.close()
        else:
            self.pl.show()