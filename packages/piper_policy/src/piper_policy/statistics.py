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


def apply_joint_state_std_floor(stats, joint_state_std_floor_rad):
    """Apply the angular floor to a processor-owned copy of training statistics."""
    if joint_state_std_floor_rad is not None:
        if type(joint_state_std_floor_rad) not in (int, float) or not (
            0 < joint_state_std_floor_rad < np.inf
        ):
            raise ValueError("joint state standard deviation floor must be positive finite radians")
        # The first six state components are angles; the final gripper opening is in metres.
        state_std = stats["observation.state"]["std"]
        state_std[:6] = np.maximum(state_std[:6], joint_state_std_floor_rad)
