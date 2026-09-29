"""Tool-point/table geometry for the existing joint-command workspace checks.

This checks discrete targets and ideal points, not solid or swept-volume collision.
"""

from dataclasses import dataclass
import math


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} requires three finite values")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} requires three finite values")
    return result


@dataclass(frozen=True)
class WorkspaceGeometry:
    tip_offset_m: tuple[float, float, float]
    grasp_offset_m: tuple[float, float, float]
    opening_axis: tuple[float, float, float]
    table_normal: tuple[float, float, float]
    table_offset_m: float

    def __post_init__(self):
        for name in ("tip_offset_m", "grasp_offset_m", "opening_axis", "table_normal"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        for name in ("opening_axis", "table_normal"):
            if not math.isclose(
                math.sqrt(sum(v * v for v in getattr(self, name))), 1.0, abs_tol=1e-8
            ):
                raise ValueError(f"{name} must be a unit vector; it is not silently normalized")
        if self.table_normal[2] <= 0 or not math.isfinite(self.table_offset_m):
            raise ValueError("table needs a finite offset and upward unit normal")


def workspace_coordinates(pose, gripper, geometry):
    """First three values: tip X, Y, signed height. Others: sampled point heights."""
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise ValueError("workspace check requires a finite six-dimensional flange pose")
    if geometry is None:
        return tuple(float(v) for v in pose[:3])
    if not math.isfinite(gripper):
        raise ValueError("tool geometry requires finite gripper feedback")
    # Preserve signed measured gap. A small negative reading swaps ideal tip labels;
    # command bounds are still enforced before dispatch, with no zero clipping.
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("xyz", pose[3:])
    tip = np.asarray(geometry.tip_offset_m)
    half_opening = np.asarray(geometry.opening_axis) * (gripper / 2)
    local = np.asarray(
        [tip, tip - half_opening, tip + half_opening, geometry.grasp_offset_m, (0.0, 0.0, 0.0)]
    )
    points = rotation.apply(local) + np.asarray(pose[:3])
    heights = points @ np.asarray(geometry.table_normal) - geometry.table_offset_m
    return (float(points[0, 0]), float(points[0, 1]), *map(float, heights))


def workspace_step_allowed(current, target, lower, upper, *, allow_reentry=False):
    """Keep goals inside the box, or permit strictly inward supervised reentry.

    Reentry never increases violation on any axis or crosses the opposite face.
    A stationary out-of-box pose is handled as a no-dispatch waiting tick.
    """
    if not all(math.isfinite(v) for v in (*current, *target)):
        return False
    if all(lo <= v <= hi for v, lo, hi in zip(target, lower, upper, strict=True)):
        return True
    if not allow_reentry:
        return False
    improved = False
    for q, v, lo, hi in zip(current, target, lower, upper, strict=True):
        if q < lo:
            if not q <= v <= hi:
                return False
            improved |= v > q
        elif q > hi:
            if not lo <= v <= q:
                return False
            improved |= v < q
        elif not lo <= v <= hi:
            return False
    return improved


def workspace_pose_allowed(
    current_pose, target_pose, current_gripper, target_gripper, safety, *, allow_reentry=False
):
    current = workspace_coordinates(current_pose, current_gripper, safety.workspace_geometry)
    target = workspace_coordinates(target_pose, target_gripper, safety.workspace_geometry)
    extra = len(current) - 3
    lower = (*safety.workspace_lower, *((safety.workspace_lower[2],) * extra))
    # Only TCP has a maximum operating height; other tool points have a floor.
    upper = (*safety.workspace_upper, *((math.inf,) * extra))
    if not workspace_step_allowed(current, target, lower, upper, allow_reentry=allow_reentry):
        return False
    if safety.workspace_geometry is not None and target_gripper != current_gripper:
        # Joint SDK return is not arrival: gripper must also be legal at the current pose.
        return gripper_table_allowed(current_pose, current_gripper, target_gripper, safety)
    return True


def gripper_table_allowed(pose, current_gripper, target_gripper, safety):
    """Only the two tip heights change when opening at a fixed arm pose."""
    if safety.workspace_geometry is None:
        return True
    before = workspace_coordinates(pose, current_gripper, safety.workspace_geometry)[3:5]
    after = workspace_coordinates(pose, target_gripper, safety.workspace_geometry)[3:5]
    return workspace_step_allowed(
        before, after, [safety.workspace_lower[2]] * 2, [math.inf] * 2, allow_reentry=True
    )


def grasp_offset(geometry=None):
    return (0.0, 0.0, 0.105) if geometry is None else geometry.grasp_offset_m


def grasp_position(pose, geometry=None):
    import numpy as np
    from scipy.spatial.transform import Rotation

    return np.asarray(pose[:3]) + Rotation.from_euler("xyz", pose[3:]).apply(grasp_offset(geometry))
