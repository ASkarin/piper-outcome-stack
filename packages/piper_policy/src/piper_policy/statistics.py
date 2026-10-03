"""Seven-dimensional numeric statistics, independent of any data provenance."""

import numpy as np


def numeric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("statistics require finite seven-dimensional samples")
    return {
        "mean": values.mean(0).astype(np.float32),
        "std": values.std(0).astype(np.float32),
        "min": values.min(0).astype(np.float32),
        "max": values.max(0).astype(np.float32),
        "count": np.array([len(values)], dtype=np.int64),
    }
