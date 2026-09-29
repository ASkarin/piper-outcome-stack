"""Dense pose references with feedback-bounded progress and endpoint confirmation."""

import math
import logging
import time
from dataclasses import dataclass, asdict

from .errors import OutcomePiperIntentRejected, OutcomePiperStateError
from .safety import ACTION_KEYS, JOINT_KEYS, step_within_limit
from .teleop_control import JointHold
from .workspace import workspace_pose_allowed
from .stage_timing import measured


@dataclass(frozen=True)
class PoseTiming:
    """Explicit reference limits, not firmware settings or verified motor capability."""

    joint_velocity_rad_s: tuple[float, ...]
    joint_acceleration_rad_s2: tuple[float, ...]
    gripper_velocity_m_s: float
    gripper_acceleration_m_s2: float
    period_s: float

    def __post_init__(self):
        for name in ("joint_velocity_rad_s", "joint_acceleration_rad_s2"):
            value = tuple(getattr(self, name))
            if len(value) != 6 or not all(math.isfinite(v) and v > 0 for v in value):
                raise ValueError(name + " requires six positive finite limits")
            object.__setattr__(self, name, value)
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.gripper_velocity_m_s, self.gripper_acceleration_m_s2, self.period_s)
        ):
            raise ValueError("pose timing requires positive finite gripper limits and period")

    def minimum_duration(self, joints, gripper, goal, controls_gripper):
        deltas = [abs(b - a) for a, b in zip(joints, goal[:6], strict=True)]
        velocities = list(self.joint_velocity_rad_s)
        accelerations = list(self.joint_acceleration_rad_s2)
        if controls_gripper:
            deltas.append(abs(goal[6] - gripper))
            velocities.append(self.gripper_velocity_m_s)
            accelerations.append(self.gripper_acceleration_m_s2)
        return max(
            [self.period_s]
            + [
                max(1.875 * d / v, math.sqrt((10 / math.sqrt(3)) * d / a))
                for d, v, a in zip(deltas, velocities, accelerations, strict=True)
            ]
        )


class JointPoseSequence:
    @measured("pose_plan_initialize")
    def __init__(
        self, joints, gripper, safety, settings, goal, kind, *, timing=None, incremental=False
    ):
        from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

        if settings is None:
            raise OutcomePiperStateError("pose movement requires configured hold settings")
        self.controls_gripper = len(goal) == 7
        if len(goal) not in (6, 7) or not all(math.isfinite(v) for v in [*goal, gripper]):
            raise OutcomePiperIntentRejected(
                "pose target requires six joints and optional gripper width"
            )
        goal = list(goal) if self.controls_gripper else [*goal, gripper]
        if any(
            not lo <= g <= hi for g, lo, hi in zip(goal[:6], safety.joint_lower, safety.joint_upper)
        ):
            raise OutcomePiperIntentRejected("pose joint target is outside configured limits")
        if self.controls_gripper and not safety.gripper_lower <= goal[6] <= safety.gripper_upper:
            raise OutcomePiperIntentRejected("pose gripper target is outside configured limits")
        self.kind = kind
        self.timing = PoseTiming(**timing) if isinstance(timing, dict) else timing
        if self.timing is not None and not isinstance(self.timing, PoseTiming):
            raise ValueError("timing must be PoseTiming")
        # Five reference intervals within the admissible joint budget. With the
        # current 5-degree limit and 0.1-degree tolerance this is at most 0.96 deg.
        # This is reference sampling density, not a relaxed command/feedback limit.
        strides = [(step - 2 * settings.joint_tolerance_rad) / 5 for step in safety.max_joint_step]
        if min(strides) <= 0:
            raise OutcomePiperIntentRejected(
                "pose movement joint step must exceed twice the hold tolerance"
            )
        count = max(
            1,
            math.ceil(
                1.875
                * max(
                    [abs(g - q) / step for q, g, step in zip(joints, goal[:6], strides)]
                    + [abs(goal[6] - gripper) / (safety.max_gripper_step / 5)]
                )
            ),
        )
        if self.timing is not None:
            count = max(
                count,
                math.ceil(
                    self.timing.minimum_duration(joints, gripper, goal, self.controls_gripper)
                    / self.timing.period_s
                ),
            )
        self.advance_time_s = None
        self.next_reference_due_s = None
        self.nominal_duration_s = None if self.timing is None else count * self.timing.period_s
        # Quintic progress has zero first/second derivatives at both endpoints;
        # its maximum slope is 1.875, included in the sample-count bound above.
        # Waiting for feedback stretches time, so this does not promise a physical
        # acceleration/jerk bound or change the firmware acceleration setting.
        self.start_joints = tuple(joints)
        self.start_gripper = gripper
        self.goal = goal
        self.waypoint_count = count
        self.targets = []
        self.control_epoch = None  # Bound by the Xbox request; standalone callers leave unset.
        self.planning_wall_s = self.planning_cpu_s = self.planning_max_batch_s = 0.0
        self.planning_batches = 0
        self._fk = fk_from_mdh
        self._mdh = list(get_mdh("piper"))
        self._previous_pose = fk_from_mdh(self._mdh, joints)
        self._previous_gripper = gripper
        self.settings = settings
        self.safety = safety
        self.advance_ready = False
        self.last_received = None
        self.tracking_error = None
        self.tracking_deadline = None
        self.index = 0
        self.window = None
        self.confirmed = False
        if not incremental:
            while not self.planning_complete:
                self.plan_chunk()

    @property
    def planning_complete(self):
        return len(self.targets) == self.waypoint_count

    @measured("pose_plan_batch")
    def plan_chunk(self, *, max_waypoints=16, budget_s=0.003):
        """Bound CPU work between control ticks; never expose an unchecked route."""
        started, cpu = time.perf_counter(), time.thread_time()
        try:
            for _ in range(max_waypoints):
                if self.planning_complete:
                    break
                i = len(self.targets) + 1
                u = (i / self.waypoint_count) ** 3 * (
                    10 - 15 * i / self.waypoint_count + 6 * (i / self.waypoint_count) ** 2
                )
                values = [
                    q + (g - q) * u
                    for q, g in zip([*self.start_joints, self.start_gripper], self.goal)
                ]
                if i == self.waypoint_count:
                    values = list(self.goal)
                if any(
                    not lo <= q <= hi
                    for q, lo, hi in zip(
                        values[:6], self.safety.joint_lower, self.safety.joint_upper
                    )
                ):
                    raise OutcomePiperIntentRejected("pose waypoint exceeds joint limits")
                if (
                    self.controls_gripper
                    and not self.safety.gripper_lower <= values[6] <= self.safety.gripper_upper
                ):
                    raise OutcomePiperIntentRejected("pose waypoint exceeds gripper limits")
                pose = self._fk(self._mdh, values[:6])
                if not workspace_pose_allowed(
                    self._previous_pose,
                    pose,
                    self._previous_gripper,
                    values[6],
                    self.safety,
                    allow_reentry=True,
                ):
                    raise OutcomePiperIntentRejected("pose route leaves configured workspace")
                self.targets.append(values)
                self._previous_pose, self._previous_gripper = pose, values[6]
                if time.perf_counter() - started >= budget_s:
                    break
        finally:
            elapsed = time.perf_counter() - started
            self.planning_wall_s += elapsed
            self.planning_cpu_s += time.thread_time() - cpu
            self.planning_max_batch_s = max(self.planning_max_batch_s, elapsed)
            self.planning_batches += 1
        return self.planning_complete

    def validate_start(self, joints, gripper):
        if not self.planning_complete:
            raise OutcomePiperStateError("pose route has not finished validation")
        if any(
            abs(q - start) > self.settings.joint_tolerance_rad
            for q, start in zip(joints, self.start_joints, strict=True)
        ):
            raise OutcomePiperIntentRejected("pose start changed during planning; select A/Y again")
        if self.controls_gripper and not step_within_limit(
            self.start_gripper, gripper, self.safety.max_gripper_step
        ):
            raise OutcomePiperIntentRejected(
                "pose gripper start changed during planning; select A/Y again"
            )

    @property
    def values(self):
        if not self.planning_complete:
            raise OutcomePiperStateError("pose route has not finished validation")
        return dict(
            zip(ACTION_KEYS if self.controls_gripper else JOINT_KEYS, self.targets[self.index])
        )

    @property
    def complete(self):
        return self.confirmed and self.index == len(self.targets) - 1

    def next_waypoint(self):
        if not self.planning_complete:
            raise OutcomePiperStateError("pose route has not finished validation")
        if self.advance_ready and self.index < len(self.targets) - 1:
            self.index += 1
            self.tracking_error = None  # No feedback yet for this newly selected reference.
            self.window = None
            self.confirmed = False
            self.advance_ready = False

    def sent(self, now):
        if not self.planning_complete:
            raise OutcomePiperStateError("pose route has not finished validation")
        # Each admitted reference has one fixed deadline. Repeated sends and
        # feedback updates never renew it, matching the trial replay's wait rule.
        if self.window is None:
            if self.timing is not None:
                period = self.timing.period_s
                # Retain the nominal phase across normal SDK/loop jitter. Reset
                # after a full missed period, so a stall cannot create catch-up bursts.
                if (
                    self.next_reference_due_s is None
                    or self.advance_time_s is None
                    or now - self.advance_time_s >= period
                ):
                    self.next_reference_due_s = now + period
                elif self.advance_time_s >= self.next_reference_due_s + period:
                    # Feedback recovery happens on a control tick. Anchor its
                    # next tick there, not to the slightly later SDK return.
                    self.next_reference_due_s = self.advance_time_s + period
                else:
                    self.next_reference_due_s += period
            self.window = JointHold(self.targets[self.index][:6], self.settings, now)
            if self.index < len(self.targets) - 1:
                self.tracking_deadline = now + self.settings.timeout_s
                self.window.deadline = self.tracking_deadline
            logging.debug(
                "%s参考点 %s/%s 已下发：%s deg，夹爪目标 %.3f mm",
                self.kind,
                self.index + 1,
                len(self.targets),
                [round(math.degrees(q), 3) for q in self.targets[self.index][:6]],
                self.targets[self.index][6] * 1000,
            )

    def observe(self, joints, received, now, gripper):
        self.advance_ready = False
        if self.window is None:
            return
        self.tracking_error = [
            target - actual
            for target, actual in zip(self.targets[self.index][:6], joints, strict=True)
        ]
        if any(
            not step_within_limit(target, actual, limit)
            for target, actual, limit in zip(
                self.targets[self.index][:6], joints, self.safety.max_joint_step, strict=True
            )
        ):
            raise OutcomePiperStateError(
                f"pose tracking error exceeds joint step budget: {self.tracking_error}"
            )
        if self.index == len(self.targets) - 1:
            self.confirmed = self.window.observe(joints, received, now)
            return
        if now >= self.window.deadline:
            raise OutcomePiperStateError(
                "pose tracking timed out before the next reference became admissible"
            )
        if min(received) < self.window.after_s or (
            self.last_received is not None
            and any(
                current <= previous
                for current, previous in zip(received, self.last_received, strict=True)
            )
        ):
            return
        self.last_received = tuple(received)
        if self.timing is not None and now < self.next_reference_due_s - 1e-9:
            return
        following = self.targets[self.index + 1]
        # This is an admission test, not clipping: the requested route and endpoint
        # remain fixed. A lagging axis prevents the next target from being sent.
        self.advance_ready = all(
            step_within_limit(target, actual, limit - self.settings.joint_tolerance_rad)
            for target, actual, limit in zip(
                following[:6], joints, self.safety.max_joint_step, strict=True
            )
        ) and (
            not self.controls_gripper
            or step_within_limit(following[6], gripper, self.safety.max_gripper_step)
        )
        if self.advance_ready:
            self.advance_time_s = now

    def telemetry(self):
        return dict(
            kind=self.kind,
            reference_profile="quintic_timed_feedback_gated"
            if self.timing is not None
            else "quintic_dense_feedback_gated",
            nominal_duration_s=self.nominal_duration_s,
            next_reference_due_s=self.next_reference_due_s,
            reference_time_s=None
            if self.timing is None or not self.planning_complete
            else (self.index + 1) * self.timing.period_s,
            timing_limits=None if self.timing is None else asdict(self.timing),
            waypoint=self.index + 1 if self.planning_complete else 0,
            waypoint_count=self.waypoint_count,
            validated_waypoints=len(self.targets),
            planning_complete=self.planning_complete,
            planning_wall_s=self.planning_wall_s,
            planning_cpu_s=self.planning_cpu_s,
            planning_max_batch_s=self.planning_max_batch_s,
            planning_batches=self.planning_batches,
            target=self.values if self.planning_complete else None,
            joint_confirmed=self.confirmed,
            phase="planning"
            if not self.planning_complete
            else ("final_confirmation" if self.index == len(self.targets) - 1 else "tracking"),
            advance_ready=self.advance_ready,
            tracking_error_rad=self.tracking_error,
            tracking_deadline_s=self.tracking_deadline,
            gripper_completion="command_only" if self.controls_gripper else "not_commanded",
        )
