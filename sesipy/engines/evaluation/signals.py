import numpy as np

def normalize_metrics(results, metrics, lower_is_better):
    values = {}

    for metric in metrics:
        if metric == "mean_error":
            values[metric] = np.array(
                [abs(r.get(metric, np.nan)) for r in results],
                dtype=float,
            )
        else:
            values[metric] = np.array(
                [r.get(metric, np.nan) for r in results],
                dtype=float,
            )

    normalized = {}

    for metric, vals in values.items():
        norm = np.full_like(vals, np.nan)

        finite = np.isfinite(vals)

        if np.any(finite):
            vmin = vals[finite].min()
            vmax = vals[finite].max()

            if np.isclose(vmin, vmax):
                norm[finite] = 1.0
            else:
                norm[finite] = (vals[finite] - vmin) / (vmax - vmin)

                if metric in lower_is_better:
                    norm[finite] = 1.0 - norm[finite]

        normalized[metric] = norm

    return normalized


def compare_power_distributions(reference, target):
    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)

    if reference.shape != target.shape:
        raise ValueError("reference and target must have the same shape")

    ref = reference.ravel()
    pred = target.ravel()

    valid = np.isfinite(ref) & np.isfinite(pred)
    ref = ref[valid]
    pred = pred[valid]

    error = pred - ref
    abs_error = np.abs(error)
    sq_error = error**2

    rmse = np.sqrt(np.mean(sq_error))
    mae = np.mean(abs_error)
    mse = np.mean(sq_error)
    bias = np.mean(error)
    std_error = np.std(error)

    max_abs_error = np.max(abs_error)

    ref_range = np.max(ref) - np.min(ref)
    nrmse = rmse / ref_range if ref_range > 0 else np.nan

    ref_mean = np.mean(ref)
    if np.any(ref != 0):
        mape = np.mean(abs_error[np.abs(ref) > 1e-12] / np.abs(ref[np.abs(ref) > 1e-12])) * 100
    else:
        mape = np.nan

    correlation = np.corrcoef(ref, pred)[0, 1] if len(ref) > 1 else np.nan

    ss_res = np.sum(sq_error)
    ss_tot = np.sum((ref - ref_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "point_error": error.reshape(reference.shape),
        "point_abs_error": abs_error.reshape(reference.shape),
        "point_squared_error": sq_error.reshape(reference.shape),
        "mean_error": bias,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "rmse": rmse,
        "normalized_rmse": nrmse,
        "max_absolute_error": max_abs_error,
        "std_error": std_error,
        "mape": mape,
        "correlation": correlation,
        "r2": r2,
    }
    

def rank_power_distributions(results):
    metrics = [
        "mean_absolute_error",
        "mean_squared_error",
        "rmse",
        "normalized_rmse",
        "max_absolute_error",
        "std_error",
        "mape",
        "correlation",
        "r2",
        "mean_error",
    ]

    lower_is_better = {
        "mean_absolute_error",
        "mean_squared_error",
        "rmse",
        "normalized_rmse",
        "max_absolute_error",
        "std_error",
        "mape",
        "mean_error",
    }

    normalized = normalize_metrics(
        results,
        metrics,
        lower_is_better,
    )

    scores = np.array([
        np.nanmean([
            normalized[m][i]
            for m in metrics
        ])
        for i in range(len(results))
    ])

    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)

    return [
        {
            "normalized_metrics": {
                metric: normalized[metric][i]
                for metric in metrics
            },
            "overall_score": scores[i],
            "rank": ranks[i],
        }
        for i in range(len(results))
    ]