import meshio
import numpy as np

def scattering_power(scattering, weights=None, lower_bound=1e-15):

    Ex, Ey, Ez = scattering

    if weights is not None:
        Ex = np.sum(Ex * weights)
        Ey = np.sum(Ey * weights)
        Ez = np.sum(Ez * weights)

    power_density = (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2) / (
        2 * 120 * np.pi
    )

    if isinstance(power_density, np.ndarray):
        power_density[np.isnan(power_density)] = lower_bound

    return np.clip(power_density, a_min=lower_bound, a_max=None)


def to_dBm(array):
    return 10 * np.log10(array * 1000)


def smooth_point_data(mesh: meshio.Mesh, key: str, iterations: int = 1):
    values = mesh.point_data[key].copy()

    n_points = len(mesh.points)
    neighbours = [set() for _ in range(n_points)]

    for cell_block in mesh.cells:
        if cell_block.type != "triangle":
            continue

        for tri in cell_block.data:
            i, j, k = tri
            neighbours[i].update((j, k))
            neighbours[j].update((i, k))
            neighbours[k].update((i, j))

    for _ in range(iterations):
        new_values = values.copy()

        for i, nbrs in enumerate(neighbours):
            if nbrs:
                new_values[i] = values[[i, *nbrs]].mean(axis=0)

        values = new_values

    mesh.point_data[key] = values


def threshold_point_data(
    mesh: meshio.Mesh,
    key: str,
    threshold: float,
    low=0,
    high=1,
):
    values = mesh.point_data[key]
    mesh.point_data[key] = np.where(values > threshold, high, low)