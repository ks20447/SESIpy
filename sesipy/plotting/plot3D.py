import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..utils.formatting import Symbols
from abc import ABC, abstractmethod


def _mesh_arrays(mesh):
    if isinstance(mesh, pv.DataSet):
        points = np.asarray(mesh.points)
        faces = []
        if isinstance(mesh, pv.PolyData):
            values = np.asarray(mesh.faces)
            index = 0
            while index < len(values):
                count = int(values[index])
                cell = values[index + 1 : index + count + 1]
                faces.extend(
                    (cell[0], cell[face_index], cell[face_index + 1])
                    for face_index in range(1, count - 1)
                )
                index += count + 1
        return points, np.asarray(faces, dtype=int), mesh.point_data

    points = np.asarray(mesh.points)
    faces = []
    for cell_block in mesh.cells:
        if cell_block.type not in {"triangle", "quad", "polygon"}:
            continue
        for cell in np.asarray(cell_block.data):
            faces.extend(
                (cell[0], cell[face_index], cell[face_index + 1])
                for face_index in range(1, len(cell) - 1)
            )
    return points, np.asarray(faces, dtype=int), mesh.point_data


def _plotly_kwargs(kwargs):
    allowed = {"color", "opacity", "name", "showlegend", "visible", "hoverinfo"}
    return {key: value for key, value in kwargs.items() if key in allowed}


def _plotly_intensity(scalars, point_data):
    if isinstance(scalars, str):
        try:
            scalars = point_data[scalars]
        except KeyError as error:
            raise KeyError(f"Mesh point data does not contain '{scalars}'.") from error

    values = np.asarray(scalars)
    if values.ndim > 1:
        values = np.linalg.norm(values, axis=-1)
    return values


class Plot3DBackend(ABC):
    def __init__(self, n_rows=1, n_cols=1):
        self.shape = (n_rows, n_cols)
        self._subplot = (0, 0)

    @abstractmethod
    def subplot(self, row, col):
        pass

    @abstractmethod
    def set_subplot_ratios(self, row_ratios, col_ratios):
        pass

    @abstractmethod
    def add_points(self, points, **kwargs):
        pass

    @abstractmethod
    def add_mesh(self, mesh, faces=None, **kwargs):
        pass

    @abstractmethod
    def add_surface(self, x, y, z, **kwargs):
        pass

    @abstractmethod
    def add_vectors(self, positions, vectors, **kwargs):
        pass

    @abstractmethod
    def add_line(self, start, end, **kwargs):
        pass

    @abstractmethod
    def show(self):
        pass


class PyVistaBackend(Plot3DBackend):
    def __init__(self, n_rows=1, n_cols=1):
        super().__init__(n_rows, n_cols)
        shape = (n_rows, n_cols)
        self.plotter = pv.Plotter(shape=shape)

    def subplot(self, row, col):
        self._subplot = (row, col)
        self.plotter.subplot(row, col)

    def set_subplot_ratios(self, row_ratios, col_ratios):
        nrows, ncols = self.shape
        row_edges = np.concatenate(([0.0], np.cumsum(row_ratios)))
        col_edges = np.concatenate(([0.0], np.cumsum(col_ratios)))

        for row in range(nrows):
            for col in range(ncols):
                renderer = self.plotter.renderers[row * ncols + col]
                renderer.viewport = (
                    col_edges[col],
                    1.0 - row_edges[row + 1],
                    col_edges[col + 1],
                    1.0 - row_edges[row],
                )

    def add_points(self, points, **kwargs):
        return self.plotter.add_points(points, **kwargs)

    def add_mesh(self, mesh, faces=None, **kwargs):
        if not isinstance(mesh, pv.DataSet):
            mesh = pv.from_meshio(mesh)
        return self.plotter.add_mesh(mesh, **kwargs)

    def add_surface(self, x, y, z, **kwargs):
        return self.plotter.add_mesh(pv.StructuredGrid(x, y, z), **kwargs)

    def add_vectors(self, positions, vectors, **kwargs):
        key = kwargs.pop("key", "Vectors")
        for option in (
            "point_size",
            "render_points_as_spheres",
            "show_lines",
            "normals",
            "color_r",
            "color_t",
            "color_s",
            "color_b",
            "show_bounds",
            "show_origin",
        ):
            kwargs.pop(option, None)
        data = pv.PolyData(positions)
        data.point_data[key] = vectors
        arrows = data.glyph(
            orient=key,
            scale=False,
            factor=kwargs.pop("scale_factor", 0.5),
        )
        return self.plotter.add_mesh(arrows, **kwargs)

    def add_line(self, start, end, **kwargs):
        line = pv.Line(start, end)
        return self.plotter.add_mesh(line, **kwargs)

    def show_bounds(self, **kwargs):
        return self.plotter.show_bounds(**kwargs)

    def show_origin(self, labels=False):
        if labels:
            return self.plotter.add_axes_at_origin()
        return self.plotter.add_axes_at_origin(xlabel="", ylabel="", zlabel="")

    def show(self):
        return self.plotter.show()


class PlotlyBackend(Plot3DBackend):
    def __init__(self, n_rows=1, n_cols=1):
        super().__init__(n_rows, n_cols)
        specs = [[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)]
        self.figure = make_subplots(rows=n_rows, cols=n_cols, specs=specs)

    def subplot(self, row, col):
        self._subplot = (row, col)

    def set_subplot_ratios(self, row_ratios, col_ratios):
        nrows, ncols = self.shape
        row_edges = np.concatenate(([0.0], np.cumsum(row_ratios)))
        col_edges = np.concatenate(([0.0], np.cumsum(col_ratios)))

        for row in range(nrows):
            for col in range(ncols):
                scene_name = (
                    "scene"
                    if row * ncols + col == 0
                    else f"scene{row * ncols + col + 1}"
                )
                self.figure.update_layout(
                    **{
                        scene_name: {
                            "domain": {
                                "x": [col_edges[col], col_edges[col + 1]],
                                "y": [
                                    1.0 - row_edges[row + 1],
                                    1.0 - row_edges[row],
                                ],
                            }
                        }
                    }
                )

    def _add_trace(self, trace):
        return self.figure.add_trace(
            trace, row=self._subplot[0] + 1, col=self._subplot[1] + 1
        )

    def add_points(self, points, **kwargs):

        kwargs = dict(kwargs)
        scalars = kwargs.pop("scalars", None)
        kwargs.pop("render_points_as_spheres", None)
        marker = {"size": kwargs.pop("point_size", 10)}
        color = kwargs.pop("color", None)
        if scalars is not None:
            marker.update(
                {"color": scalars, "colorscale": kwargs.pop("cmap", "Viridis")}
            )
        elif color is not None:
            marker["color"] = color
        points = np.asarray(points)
        return self._add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=marker,
                **_plotly_kwargs(kwargs),
            )
        )

    def add_mesh(self, vertices, faces=None, **kwargs):

        if faces is None:
            points, mesh_faces, point_data = _mesh_arrays(vertices)
        else:
            points = np.asarray(vertices)
            mesh_faces = np.asarray(faces, dtype=int)
            point_data = {}
        scalars = kwargs.pop("scalars", None)
        trace_kwargs = _plotly_kwargs(kwargs)
        if kwargs.get("show_edges", False):
            trace_kwargs["contour"] = {"show": True}
        if scalars is not None:
            scalars = _plotly_intensity(scalars, point_data)
            trace_kwargs.update(
                {"intensity": scalars, "colorscale": kwargs.get("cmap", "Viridis")}
            )
        if len(mesh_faces) == 0:
            return self.add_points(points, scalars=scalars, **kwargs)
        return self._add_trace(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=mesh_faces[:, 0],
                j=mesh_faces[:, 1],
                k=mesh_faces[:, 2],
                **trace_kwargs,
            )
        )

    def add_surface(self, x, y, z, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("cmap", None)
        return self._add_trace(go.Surface(x=x, y=y, z=z, **_plotly_kwargs(kwargs)))

    def add_vectors(self, positions, vectors, **kwargs):
        positions = np.asarray(positions)
        vectors = np.asarray(vectors)
        kwargs = dict(kwargs)
        color = kwargs.pop("color", None)
        colorscale = kwargs.pop("cmap", None)
        if color is not None:
            colorscale = [[0, color], [1, color]]
        return self._add_trace(
            go.Cone(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                u=vectors[:, 0],
                v=vectors[:, 1],
                w=vectors[:, 2],
                sizemode="absolute",
                sizeref=kwargs.pop("scale_factor", 0.5),
                opacity=kwargs.pop("opacity", None),
                name=kwargs.pop("name", None),
                colorscale=colorscale,
            )
        )

    def add_line(self, start, end, **kwargs):
        start = np.asarray(start)
        end = np.asarray(end)
        line_kwargs = {}
        trace_kwargs = {}
        if "color" in kwargs:
            line_kwargs["color"] = kwargs["color"]
        if "opacity" in kwargs:
            trace_kwargs["opacity"] = kwargs["opacity"]
        if "line_width" in kwargs:
            line_kwargs["width"] = kwargs["line_width"]
        return self._add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=line_kwargs,
                **trace_kwargs,
            )
        )

    def show_bounds(self, **kwargs):
        axis_kwargs = {"showline": True, **kwargs}
        self.figure.update_scenes(
            xaxis=axis_kwargs,
            yaxis=axis_kwargs,
            zaxis=axis_kwargs,
        )

    def show_origin(self, labels=False):
        self.figure.update_scenes(
            xaxis_title="X" if labels else None,
            yaxis_title="Y" if labels else None,
            zaxis_title="Z" if labels else None,
        )

    def show(self):
        self.figure.update_layout(
            {
                f"scene{'' if i == 1 else i}": {"aspectmode": "data"}
                for i in range(1, (self.shape[0] * self.shape[1]) + 1)
            }
        )
        return self.figure.show()


def create_backend(backend_name, n_rows=1, n_cols=1):
    if backend_name == "pyvista":
        return PyVistaBackend(n_rows, n_cols)
    elif backend_name == "plotly":
        return PlotlyBackend(n_rows, n_cols)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


class Plot3D:

    def __init__(self, n_rows=1, n_cols=1, backend="pyvista"):

        self.pl = create_backend(backend, n_rows, n_cols)
        self._sym = Symbols()

    def set_plot(self, row, col):
        self.pl.subplot(row, col)

    def set_subplot_ratios(self, row_ratios=None, col_ratios=None):
        nrows, ncols = self.pl.shape

        if row_ratios is None:
            row_ratios = np.ones(nrows, dtype=float)
        elif np.isscalar(row_ratios):
            row_ratios = np.full(nrows, row_ratios, dtype=float)
        else:
            row_ratios = np.asarray(row_ratios, dtype=float)

        if col_ratios is None:
            col_ratios = np.ones(ncols, dtype=float)
        elif np.isscalar(col_ratios):
            col_ratios = np.full(ncols, col_ratios, dtype=float)
        else:
            col_ratios = np.asarray(col_ratios, dtype=float)

        if row_ratios.size != nrows or col_ratios.size != ncols:
            raise ValueError(
                "The number of subplot ratios must match the subplot shape."
            )
        if np.any(row_ratios <= 0) or np.any(col_ratios <= 0):
            raise ValueError("Subplot ratios must be positive.")

        row_ratios /= row_ratios.sum()
        col_ratios /= col_ratios.sum()

        x_edges = np.concatenate(([0.0], np.cumsum(col_ratios)))
        y_edges = np.concatenate(([0.0], np.cumsum(row_ratios)))
        self.pl.set_subplot_ratios(
            y_edges[1:] - y_edges[:-1], x_edges[1:] - x_edges[:-1]
        )

    def add_mesh(self, mesh, scalars=None, **kwargs):
        self.pl.add_mesh(mesh, scalars=scalars, **kwargs)

    def add_surface(self, x, y, z, **kwargs):
        self.pl.add_surface(x, y, z, **kwargs)

    def add_points(self, points, scalars=None, point_size=10, **kwargs):
        self.pl.add_points(
            points,
            scalars=scalars,
            point_size=point_size,
            render_points_as_spheres=True,
            **kwargs,
        )

    def add_arrows(self, points, orientations, key="Vectors", **kwargs):

        self.pl.add_vectors(points, orientations, key=key, **kwargs)

    def show_bounds(self):
        self.pl.show_bounds()

    def show_origin(self, labels=False):
        self.pl.show_origin(labels)

    @property
    def plotter(self):
        return self.pl

    @property
    def sym(self):
        return self._sym

    def show(self, filename=None):
        if filename is not None and isinstance(self.pl, PlotlyBackend):
            if str(filename).lower().endswith(".html"):
                return self.pl.figure.write_html(filename)
            return self.pl.figure.write_image(filename)
        if filename is not None:
            self.pl.show(auto_close=False)
            self.pl.save_graphic(filename)
            self.pl.close()
        else:
            self.pl.show()

    def plot_path(self, points, orientations, **kwargs):

        self.add_points(points, **kwargs)
        self.add_arrows(points, orientations, **kwargs)

    def plot_indicator_line(self, loc, line_color, **kwargs):
        loc = np.asarray(loc)
        self.pl.add_line(
            loc - np.array([0, 0, loc[-1]]),
            loc + np.array([0, 0, kwargs.get("line_height", 5)]),
            line_width=kwargs.get("line_width", 5),
            color=line_color,
            opacity=kwargs.get("line_opacity", 1.0),
        )

    def plot_point_normals(self, points, normals, **kwargs):
        self.add_arrows(points, normals, key="Normals", **kwargs)

    def plot_antenna_array(self, antenna, scalars=None, normals=False, **kwargs):

        loc = antenna.points_mesh.center
        color = kwargs.get("color", "blue")

        antenna_points = antenna.points_mesh.points

        self.add_points(
            antenna_points,
            color=color,
            point_size=kwargs.get("point_size", 10),
            scalars=antenna.points_mesh.meshio_mesh.point_data.get(scalars, None),
        )

        if antenna.structure_mesh is not None:
            mesh_kwargs = dict(kwargs)
            for key in (
                "color",
                "point_size",
                "render_points_as_spheres",
                "show_lines",
                "line_height",
                "line_width",
                "line_opacity",
                "normals",
                "color_r",
                "color_t",
                "show_bounds",
                "show_origin",
            ):
                mesh_kwargs.pop(key, None)
            self.add_mesh(antenna.structure_mesh, scalars=scalars, **mesh_kwargs)

        if kwargs.get("show_lines", True):
            self.plot_indicator_line(loc, color, **kwargs)

        if normals:
            self.plot_point_normals(
                antenna_points, antenna.points_mesh.get_normals(), **kwargs
            )

    def plot_scatterers(self, scatterers, scalars=None, normals=False, **kwargs):
        mesh_kwargs = dict(kwargs)
        for key in (
            "color_s",
            "scatterers_opacity",
            "show_edges",
            "cmap",
            "color_r",
            "color_t",
            "color_b",
            "blockers_opacity",
            "show_bounds",
            "show_origin",
            "show_lines",
            "normals",
        ):
            mesh_kwargs.pop(key, None)

        for scatter in scatterers:
            self.add_mesh(
                scatter,
                color=kwargs.get("color_s", "lightgrey"),
                scalars=scalars,
                opacity=kwargs.get("scatterers_opacity", 0.75),
                show_edges=kwargs.get("show_edges", True),
                cmap=kwargs.get("cmap", "viridis"),
                **mesh_kwargs,
            )

            if normals:
                self.plot_point_normals(scatter.points, scatter.point_data["Normals"])

    def plot_blockers(self, blockers, scalars=None, normals=False, **kwargs):
        mesh_kwargs = dict(kwargs)
        for key in (
            "color_b",
            "blockers_opacity",
            "show_edges",
            "cmap",
            "color_r",
            "color_t",
            "color_s",
            "scatterers_opacity",
            "show_bounds",
            "show_origin",
            "show_lines",
            "normals",
        ):
            mesh_kwargs.pop(key, None)

        for block in blockers:
            self.add_mesh(
                block,
                color=kwargs.get("color_b", "grey"),
                scalars=scalars,
                opacity=kwargs.get("blockers_opacity", 0.75),
                show_edges=kwargs.get("show_edges", False),
                cmap=kwargs.get("cmap", "viridis"),
                **mesh_kwargs,
            )

            if normals:
                self.plot_point_normals(block.points, block.point_data["Normals"])

    def plot_scene(self, scene, scalars=None, **kwargs):

        receiver = scene.receiver
        transmitter = scene.transmitter
        scatterers = scene.scatterers
        blockers = scene.blockers

        color_r = kwargs.get("color_r", "blue")
        color_t = kwargs.get("color_t", "red")
        antenna_kwargs = dict(kwargs)
        antenna_kwargs.pop("color_r", None)
        antenna_kwargs.pop("color_t", None)
        antenna_kwargs.pop("color", None)

        if receiver is not None:
            self.plot_antenna_array(
                receiver, scalars=scalars, color=color_r, **antenna_kwargs
            )

        if transmitter is not None:
            self.plot_antenna_array(
                transmitter, scalars=scalars, color=color_t, **antenna_kwargs
            )

        if len(scatterers) > 0:
            self.plot_scatterers(scatterers, scalars=scalars, **kwargs)

        if len(blockers) > 0:
            self.plot_blockers(blockers, scalars=scalars, **kwargs)

        if kwargs.get("show_bounds", True):
            self.show_bounds()
        if kwargs.get("show_origin", True):
            self.show_origin(labels=False)
