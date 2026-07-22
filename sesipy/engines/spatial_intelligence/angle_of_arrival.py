import numpy as np
import shapely as sp
from scipy.signal import find_peaks
from dataclasses import dataclass


def angle_diff(a, b):
    return (b - a + np.pi) % (2 * np.pi) - np.pi


@dataclass(slots=True)
class AoA:
    peak: float | None
    left: float | None
    right: float | None
    theta_peaks: np.ndarray
    power_peaks: np.ndarray

    @property
    def left_beam(self):
        if self.peak is None or self.left is None:
            return None
        return -angle_diff(self.left, self.peak)

    @property
    def right_beam(self):
        if self.peak is None or self.right is None:
            return None
        return angle_diff(self.peak, self.right)

    @property
    def beamwidth(self):
        if self.left is None or self.right is None:
            return None
        return angle_diff(self.left, self.right)
    
    @property
    def n_peaks(self):
        return len(self.theta_peaks)
    
    @property
    def beam_center(self):
        if self.left is None or self.right is None:
            return None

        return (self.left + 0.5 * angle_diff(self.left, self.right)) % (2 * np.pi)


def aoa_projection_2D(origin, aoa: AoA, length, arc_resolution=32):

    x, y = origin

    if aoa is None or aoa.peak is None:
        return sp.Point(origin).buffer(10.0)

    left = angle_diff(aoa.peak, aoa.left)
    right = angle_diff(aoa.peak, aoa.right)

    angles = aoa.peak + np.linspace(left, right, arc_resolution)

    arc = np.column_stack(
        (
            x + length * np.cos(angles),
            y + length * np.sin(angles),
        )
    )

    return sp.Polygon(
        np.vstack(
            (
                origin,
                arc,
                origin,
            )
        )
    )
    

def multi_aoa_projection_2D(origin, aoa : AoA, width, length, arc_resolution=32):
    
    aoa_list = [AoA(p, p - width, p + width, [], []) for p in aoa.theta_peaks]
    
    projections = [aoa_projection_2D(origin, a, length, arc_resolution) for a in aoa_list]
    
    return sp.MultiPolygon(projections)


def extract_aoa(steering_mesh, drop_dB=3.0):

    theta = np.asarray(steering_mesh.point_data["Theta"])
    power = np.asarray(steering_mesh.point_data["Power"])

    if len(theta) < 3:
        return None

    if np.allclose(power, power[0]):
        return None

    n = len(theta)

    peak_idx = np.argmax(power)
    peak = theta[peak_idx]

    threshold = power[peak_idx] - drop_dB

    i = peak_idx
    while True:
        j = (i - 1) % n

        if power[j] < threshold:
            left = theta[i]
            break

        if j == peak_idx:
            return None

        i = j

    i = peak_idx
    while True:
        j = (i + 1) % n

        if power[j] < threshold:
            right = theta[i]
            break

        if j == peak_idx:
            return None

        i = j
        
    peak_idxs, _ = find_peaks(
        power,
        height=-3.0,
        width=1,
    )
    
    if len(peak_idxs) == 0:
        peak_idxs = np.array([peak_idx])

    return AoA(
        peak=peak,
        left=left,
        right=right,
        theta_peaks=theta[peak_idxs],
        power_peaks=power[peak_idxs]
    )
    
    
