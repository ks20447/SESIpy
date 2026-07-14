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