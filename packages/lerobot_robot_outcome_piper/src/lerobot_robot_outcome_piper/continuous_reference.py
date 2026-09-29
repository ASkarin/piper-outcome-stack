"""Pure Cartesian reference generation; candidates commit only after SDK success."""

from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ReferenceSettings:
    period_s: float
    ramp_time_s: float
    lead_cycles: float

    def __post_init__(self):
        if not all(
            math.isfinite(v) and v > 0 for v in (self.period_s, self.ramp_time_s, self.lead_cycles)
        ):
            raise ValueError("reference period, ramp and lead must be finite and positive")


def reference_candidate(
    previous,
    position,
    rotation,
    locked_rotation,
    xyz_delta,
    rotation_delta,
    xyz_step,
    rotation_step,
    settings,
    now,
    *,
    lock_orientation=True,
):
    """No mutation, queuing, or integration of time lost to a missed control tick."""
    position = np.asarray(position, float)
    rotation = Rotation.from_matrix(rotation)
    if previous is None:
        origin = position.copy()
        orientation = Rotation.from_matrix(locked_rotation)
        velocity = np.zeros(6)
        dt = settings.period_s
    else:
        origin = np.asarray(previous["position"], float)
        orientation = Rotation.from_matrix(previous["rotation"])
        velocity = np.asarray(previous["velocity"], float)
        elapsed = now - previous["generated_s"]
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("reference clock must advance")
        dt = min(elapsed, settings.period_s)
    desired = np.asarray([*xyz_delta, *rotation_delta], float) / settings.period_s
    if not np.isfinite(desired).all():
        raise ValueError("non-finite reference input")
    linear_change = xyz_step / settings.period_s / settings.ramp_time_s * dt
    velocity[:3] += np.clip(desired[:3] - velocity[:3], -linear_change, linear_change)
    difference = desired[3:] - velocity[3:]
    limit = rotation_step / settings.period_s / settings.ramp_time_s * dt
    velocity[3:] += difference * min(1.0, limit / max(float(np.linalg.norm(difference)), 1e-15))
    if not any(xyz_delta):
        velocity[:3] = 0
    if not any(rotation_delta):
        velocity[3:] = 0

    def at(fraction):
        return origin + velocity[:3] * dt * fraction, Rotation.from_rotvec(
            velocity[3:] * dt * fraction
        ) * orientation

    def within(point, orient):
        return np.max(np.abs(point - position)) <= xyz_step * settings.lead_cycles and (
            not lock_orientation
            or (orient * rotation.inv()).magnitude() <= rotation_step * settings.lead_cycles
        )

    fraction = 1.0
    point, orient = at(1.0)
    if not within(point, orient):
        if not within(*at(0.0)):
            fraction = 0.0  # Retain the last accepted target; do not follow feedback drift.
        else:
            low, high = 0.0, 1.0
            for _ in range(28):
                middle = (low + high) / 2
                if within(*at(middle)):
                    low = middle
                else:
                    high = middle
            fraction = low
        point, orient = at(fraction)
        velocity *= fraction  # No stored velocity debt while the reference is blocked.
    return dict(
        position=point.tolist(),
        rotation=orient.as_matrix().tolist(),
        velocity=velocity.tolist(),
        generated_s=now,
        dt_s=dt,
        lead_limited=fraction < 1.0,
        progress_fraction=fraction,
    )
