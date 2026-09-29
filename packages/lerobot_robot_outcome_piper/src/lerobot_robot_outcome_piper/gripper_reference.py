"""Time-based Xbox gripper reference; no hardware or committed state here."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GripperReferenceSettings:
    period_s: float
    max_speed_m_s: float
    acceleration_m_s2: float
    max_lead_m: float

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.period_s, self.max_speed_m_s, self.acceleration_m_s2, self.max_lead_m)
        ):
            raise ValueError("gripper reference settings must be finite and positive")


def gripper_candidate(previous, target, feedback, command, settings, now, lower, upper):
    """Integrate a bounded-time velocity ramp from the last accepted command.

    Released triggers retain the target. Reversals discard old-direction velocity;
    elapsed stalls and blocked travel never accumulate motion debt.
    """
    if not all(math.isfinite(v) for v in (target, feedback, command, now)):
        raise ValueError("gripper reference inputs must be finite")
    if not -1 <= command <= 1:
        raise ValueError("gripper speed input must be normalized")
    dt = settings.period_s if previous is None else now - previous["time_s"]
    if dt < 0 or not math.isfinite(dt):
        raise ValueError("gripper reference time must be monotonic")
    dt = min(dt, settings.period_s)
    velocity = 0.0 if previous is None else previous["velocity_m_s"]
    desired = command * settings.max_speed_m_s
    if desired == 0 or velocity * desired < 0:
        velocity = 0.0
    if desired == 0:
        displacement, next_velocity = 0.0, 0.0
    else:
        difference = desired - velocity
        ramp = min(dt, abs(difference) / settings.acceleration_m_s2)
        acceleration = math.copysign(settings.acceleration_m_s2, difference)
        next_velocity = velocity + acceleration * ramp
        displacement = (velocity + next_velocity) * 0.5 * ramp + next_velocity * (dt - ramp)
    requested = target + displacement
    limited = False
    # Bound only new travel in the requested direction. Never pull an existing
    # holding target toward feedback when the object prevents closure.
    if displacement > 0:
        available = max(0.0, min(upper, feedback + settings.max_lead_m) - target)
        limited = displacement > available
        displacement = min(displacement, available)
    elif displacement < 0:
        available = max(0.0, target - max(lower, feedback - settings.max_lead_m))
        limited = -displacement > available
        displacement = -min(-displacement, available)
    if limited:
        next_velocity = 0.0
    return dict(
        target_m=target + displacement,
        velocity_m_s=next_velocity,
        time_s=now,
        dt_s=dt,
        desired_velocity_m_s=desired,
        requested_target_m=requested,
        travel_limited=limited,
    )
