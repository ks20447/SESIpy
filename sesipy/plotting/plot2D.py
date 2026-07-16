import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ..utils.formatting import Symbols
from shapely.geometry import Polygon, MultiPolygon


def mpl_use_latex(val : bool):
    rcParams.update({"text.usetex": val, "font.family": "Times"})
    
def mpl_font_size(size : int):
    rcParams.update({"font.size": size})
    
def mpl_use_seaborn():
    mpl.style.use("seaborn-v0_8")
    
def mpl_use_cmap(cmap : str):
    mpl.rcParams["image.cmap"] = cmap
    
    
class Plot2D:

    def __init__(self, n_rows=1, n_cols=1, figsize=(12, 6), set_style=True):
        
        self._equal_aspect = False
        self._grid = False
        
        if set_style:
            self._default_style()

        self.fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            squeeze=False,
        )

        self.axes = axs
        self._ax = self.axes[0, 0]
        self._sym = Symbols()

    def _default_style(self):
        mpl_use_latex(True)
        mpl_font_size(20)
        mpl_use_seaborn()
        mpl_use_cmap("viridis")
        
        self._equal_aspect = True
        self._grid = True
    
    @property
    def figure(self):
        return self.fig

    @property
    def ax(self):
        return self._ax

    @property
    def sym(self):
        return self._sym

    def get_ax(self, row=0, col=0):
        return self.axes[row, col]

    def set_ax(self, row=0, col=0):
        self._ax = self.get_ax(row, col)

    def use_ax(self, ax):
        if not any(ax is candidate for candidate in self.axes.flat):
            raise ValueError("Axis is not part of this figure.")
        self._ax = ax

    def _save_view(self, event, filename):
        event.canvas.figure.savefig(filename, bbox_inches="tight")

    def show(self, filename=None):
        
        for ax in self.axes.flatten():
            if self.equal_aspect:
                ax.set_aspect("equal")
            if self.grid:
                ax.grid(True)
        
        if filename is not None:
            self.fig.canvas.mpl_connect(
                "close_event", lambda event: self._save_view(event, filename)
            )
        plt.tight_layout()
        plt.show()
        
    @property
    def equal_aspect(self):
        return self._equal_aspect
    
    @equal_aspect.setter
    def equal_aspect(self, val : bool):
        self._equal_aspect = val
        
    @property
    def grid(self):
        return self._grid
    
    @grid.setter
    def grid(self, val : bool):
        self._grid = val
        

    def plot_polygon(self, polygon, c="k", fill_outline=False, fill_holes=False, **kwargs):

        alpha = kwargs.get("opacity", 1.0)

        if isinstance(polygon, MultiPolygon):
            for poly in polygon.geoms:
                self.plot_polygon(
                    poly,
                    c=c,
                    fill_outline=fill_outline,
                    fill_holes=fill_holes,
                    opacity=alpha,
                )
            return

        x, y = polygon.exterior.xy

        if fill_outline:
            self.ax.fill(x, y, c=c, alpha=alpha)
        else:
            self.ax.plot(x, y, c=c, alpha=alpha)

        for interior in polygon.interiors:
            xi, yi = interior.xy

            if fill_holes:
                self.ax.fill(xi, yi, c=c, alpha=alpha)
            else:
                self.ax.plot(xi, yi, c=c, alpha=alpha)
           
                
    def plot_fov(self, origin, polygon, c=None):

        if c is None:
            c = self.ax._get_lines.get_next_color()

        self.plot_polygon(
            polygon,
            c=c,
            fill_outline=True,
            fill_holes=False,
            opacity=0.7,
        )

        self.plot_scatter(
            np.array([origin]),
            separate=False,
            marker="o",
            c=c,
        )
                
            
    def plot_line(self, points, line_style="-"):
        
        self.ax.plot(points[:, 0], points[:, 1], line_style)
            
        
    def plot_scatter(self, points, separate=False, **kwargs):

        marker = kwargs.get("marker", "x")
        s = kwargs.get("s", 50)
        c = kwargs.get("c", None)

        if separate:
            for (x, y) in points:
                self.ax.scatter(x, y, s=s, marker=marker, c=c)
        else:
            self.ax.scatter(points[:, 0], points[:, 1], s=s, marker=marker, c=c)
            
        
    def plot_arrows(self, points, theta, separate=False, **kwargs):
    
        u = np.cos(np.radians(theta))
        v = np.sin(np.radians(theta))
        
        scale = kwargs.get("scale", 0.5)
        color = kwargs.get("arrow_c", "k")
        
        if separate:
            for (x, y), u, v in zip(points, u, v):
                self.ax.quiver(
                    x,
                    y,
                    u,
                    v,
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                    color=color,
                    width=0.005,
                )
        else:
            self.ax.quiver(
                points[:, 0],
                points[:, 1],
                u,
                v,
                angles="xy",
                scale_units="xy",
                scale=scale,
                color=color,
                width=0.005,
            )
            
    def plot_waypoints(self, points, orientation, **kwargs):
        
        self.plot_arrows(points, orientation, **kwargs)
        self.plot_scatter(points, marker="o", s=75, **kwargs)
        self.plot_line(points, line_style="--")
            
          
    def plot_pgm(self, img):
        
        cax = self.ax.imshow(img, cmap='gray', origin='upper')
        
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
            
            
    def plot_aoa(self, theta, r, aoa=None):
        
        row, col = [(i, j) for i in range(self.axes.shape[0]) for j in range(self.axes.shape[1]) if self.axes[i, j] is self.ax][0]
        
        if self.ax.name != "polar":
            spec = self.ax.get_subplotspec()
            self.ax.remove()
            self._ax = self.fig.add_subplot(spec, projection="polar")
            
        idx = np.argsort(theta)
        theta = theta[idx]
        theta = np.append(theta, theta[0])
        r = r[idx]
        r = np.append(r, r[0])
        
        self.ax.plot(theta, r)
        self.ax.set_ylim([r.min(), 0.0])
        
        if aoa:
            rmin, rmax = self.ax.get_ylim()
            self.ax.plot([aoa.peak, aoa.peak], [rmin, rmax], "r-")
            self.ax.plot([aoa.left, aoa.left], [rmin, rmax], "r--")
            self.ax.plot([aoa.right, aoa.right], [rmin, rmax], "r--")
        
        self.ax.set_xlabel(f"Theta")
        self.ax.set_ylabel(f"Power ({self.sym.DB})")
        
        if self.grid:
            self.ax.grid(True)
        
        self.axes[row, col] = self.ax