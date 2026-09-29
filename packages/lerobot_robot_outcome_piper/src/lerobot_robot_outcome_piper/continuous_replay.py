"""Offline derived PCHIP references and feedback-gated replay, no Dataset mutation."""

import math
import time
import numpy as np
from scipy.interpolate import PchipInterpolator
from .safety import ACTION_KEYS, JOINT_KEYS, step_within_limit
from .teleop_control import JointHold
from .execution_constraints import check_execution_target


def plan_replay(values, source_fps, control_hz, time_scale, safety):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("continuous replay needs at least two finite seven-value actions")
    if (
        not all(math.isfinite(v) and v > 0 for v in (source_fps, control_hz, time_scale))
        or time_scale < 1
    ):
        raise ValueError("replay rates must be positive and time_scale at least one")
    if np.any(values[:, 6] < 0):
        raise ValueError("source action gripper must be nonnegative")
    source_times = np.arange(len(values)) / source_fps
    curve = PchipInterpolator(source_times, values, axis=0, extrapolate=False)
    count = math.ceil(source_times[-1] * time_scale * control_hz)
    source_samples = np.linspace(0, source_times[-1], count + 1)
    references = curve(source_samples).astype(np.float32)
    references[0] = values[0]
    references[-1] = values[-1]
    # No component overshoot; complete pose/step checks still apply to generated references.
    knots = np.minimum((source_samples * source_fps).astype(int), len(values) - 2)
    lo = np.minimum(values[knots], values[knots + 1])
    hi = np.maximum(values[knots], values[knots + 1])
    tolerance = 4 * np.finfo(np.float32).eps * np.maximum(1, np.maximum(abs(lo), abs(hi)))
    if np.any(references < lo - tolerance) or np.any(references > hi + tolerance):
        raise ValueError("interpolated replay overshoots source interval")
    for i, target in enumerate(references):
        if target[6] < 0:
            raise ValueError("Dataset action gripper must already be nonnegative")
        try:
            check_execution_target(references[max(0, i - 1)], target, safety)
        except ValueError as exc:
            raise ValueError(f"derived reference {i}: {exc}") from exc
    return dict(
        actions=[dict(zip(ACTION_KEYS, map(float, v))) for v in references],
        source_time_s=source_samples.tolist(),
        source_rows=knots.tolist(),
        control_hz=control_hz,
        source_fps=source_fps,
        time_scale=time_scale,
        nominal_duration_s=count / control_hz,
        method="pchip_common_time_scale",
        source_action_count=len(values),
    )


def hold_current(robot, settings, clock, sleep):
    obs = robot.observe_control()
    target = [obs[k] for k in JOINT_KEYS]
    robot.send_joint_target(target)  # Retain the last gripper command/force.
    hold = JointHold(target, settings, clock())
    while True:
        start = clock()
        obs = robot.observe_control()
        if hold.observe(
            [obs[k] for k in JOINT_KEYS],
            robot.last_feedback_telemetry.received_monotonic_s[:3],
            clock(),
        ):
            return obs
        sleep(max(0.0, 0.05 - (clock() - start)))


def execute_replay(
    robot, plan, safety, settings, trace, terminal, clock=time.monotonic, sleep=time.sleep
):
    period = 1 / plan["control_hz"]
    last_received = None
    started = clock()
    for index, target in enumerate(plan["actions"]):
        deadline = clock() + settings.timeout_s
        while True:
            tick = clock()
            if terminal.poll() in ("stop", "quit"):
                final = hold_current(robot, settings, clock, sleep)
                return dict(status="cancelled_held", final_observation=final)
            obs = robot.observe_control()
            received = tuple(robot.last_feedback_telemetry.received_monotonic_s[:3])
            if clock() >= deadline:
                raise RuntimeError(f"derived reference {index} tracking timeout")
            fresh = last_received is None or all(a > b for a, b in zip(received, last_received))
            admissible = (
                fresh
                and all(
                    step_within_limit(target[k], obs[k], bound - settings.joint_tolerance_rad)
                    for k, bound in zip(JOINT_KEYS, safety.max_joint_step)
                )
                and step_within_limit(
                    target["gripper.pos"], obs["gripper.pos"], safety.max_gripper_step
                )
            )
            entry = dict(
                reference_index=index,
                source_time_s=plan["source_time_s"][index],
                source_row_lower=plan["source_rows"][index],
                source_row_upper=plan["source_rows"][index] + 1,
                elapsed_s=clock() - started,
                observed=obs,
                target=target,
                phase="advance" if admissible else "wait_feedback",
                sdk_call=None,
            )
            trace.append(entry)
            if admissible:
                entry["dispatch_started_s"] = clock()
                entry["dispatch_result"] = "failed"
                try:
                    robot.send_action(target)
                    entry["dispatch_result"] = "returned"
                finally:
                    entry.update(dispatch_ended_s=clock(), sdk_call=robot.last_action_telemetry)
            # The last command remains active while feedback is serviced; no fake resend timestamps.
            last_received = received
            sleep(max(0.0, period - (clock() - tick)))
            if admissible:
                break
    hold = JointHold([plan["actions"][-1][k] for k in JOINT_KEYS], settings, clock())
    while True:
        tick = clock()
        if terminal.poll() in ("stop", "quit"):
            return dict(
                status="cancelled_held",
                final_observation=hold_current(robot, settings, clock, sleep),
            )
        obs = robot.observe_control()
        if hold.observe(
            [obs[k] for k in JOINT_KEYS],
            robot.last_feedback_telemetry.received_monotonic_s[:3],
            clock(),
        ):
            return dict(
                status="targets_sent_final_joints_confirmed",
                final_observation=obs,
                gripper_completion="command_only",
            )
        sleep(max(0.0, period - (clock() - tick)))
