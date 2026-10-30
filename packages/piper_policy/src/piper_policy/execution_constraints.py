"""Explicit execution projection; raw predictions and all other checks remain separate."""

import numpy as np


def project_gripper(actions, mode):
    source = np.asarray(actions)
    if source.ndim < 1 or source.shape[-1] != 7 or not np.isfinite(source).all():
        raise ValueError("execution targets require finite seven-dimensional actions")
    if mode not in ("strict", "zero"):
        raise ValueError("gripper lower-bound mode must be strict or zero")
    target = source.copy()
    if mode == "zero":
        target[..., 6] = np.maximum(0, target[..., 6])
    return target


def projection_stats(raw, targets):
    difference = np.asarray(targets)[..., 6] - np.asarray(raw)[..., 6]
    changed = difference > 0
    return {
        "count": int(changed.sum()),
        "total": int(difference.size),
        "max_correction_mm": float(difference.max(initial=0) * 1000),
        "sum_correction_mm": float(difference.sum() * 1000),
    }
