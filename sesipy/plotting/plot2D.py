import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def mpl_use_latex(val : bool):
    plt.rcParams.update({"text.usetex": val, "font.family": "Times"})
    
def mpl_font_size(size : int):
    plt.rcParams.update({"font.size": size})
    
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
        
    def plot_polygon(self, polygon, c="k", fill_holes=False):
        
        x, y = polygon.exterior.xy
        self.ax.plot(x, y, c=c)

        for interior in polygon.interiors:
            xi, yi = interior.xy
            
            if fill_holes:
                self.ax.fill(xi, yi, c=c)
            else:
                self.ax.plot(xi, yi, c=c)
        
        if self.equal_aspect:
            self.ax.set_aspect("equal")
            
        if self.grid:
            self.ax.grid(True)