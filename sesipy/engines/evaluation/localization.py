import numpy as np
from shapely import contains_xy
from scipy.interpolate import griddata


def scoring_surface(sample_locs, scores, boundary=None, method="nearest"):
    
    x = sample_locs[:, 0]
    y = sample_locs[:, 1]

    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata(
        (x, y),
        scores,
        (X, Y),
        method=method
    )

    if boundary:
        x_min, x_max, y_min, y_max = boundary.bounds
        mask = contains_xy(boundary, X, Y)
        Z = np.where(mask, Z, np.nan)
        
    return X, Y, Z


def sample_surface(X, Y, Z, n_samples, bias=2):
    
    valid = np.isfinite(Z)

    x = X[valid]
    y = Y[valid]
    z = Z[valid]

    z = z - np.nanmin(z)

    bias = 2
    probability = z ** bias
    probability /= probability.sum()
    
    indices = np.random.choice(
        len(x),
        size=n_samples,
        replace=True,
        p=probability,
    )

    sample_points = np.column_stack([
        x[indices],
        y[indices],
    ])
    
    probability_surface = np.zeros_like(Z, dtype=float)
    probability_surface[valid] = probability
    probability_surface[~valid] = np.nan
    
    return sample_points, probability_surface