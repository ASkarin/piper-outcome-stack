"""One operator-controlled ACT segment using PiPER's existing hold/stop lifecycle."""

import math
import time
from copy import deepcopy

from lerobot_robot_outcome_piper.processor import OutcomePiperAction
from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from lerobot_robot_outcome_piper.teleop_control import TeleopState

from .policy_execution import predict_candidate


class PolicyInput:
    """LB permits policy; other motion inputs cancel, never become teleop commands."""

    def __init__(self, teleop, robot):
        self.teleop, self.robot = teleop, robot
        self.last_raw = None

    def poll(self):
        raw = self.teleop.get_action()
        self.last_raw = dict(raw)
        if raw["emergency_stop"]:
            self.robot.request_emergency_stop("Xbox B during policy trial")
            raise RuntimeError("Xbox B: policy trial stopped")
        neutral = raw["neutral"] and not any(
            raw[k] for k in ("home", "work", "mode_switch", "translation_switch")
        )
        # A conflicting input disarms. Keeping LB held cannot rearm the next tick.
        return {**raw, "hold": raw["hold"] and neutral, "neutral": neutral}


def hold_action(observation, control, clock):
    action = OutcomePiperAction(
        {k: float(observation[k]) for k in ACTION_KEYS},
        intent="hold",
        epoch=control.epoch,
    )
    action.generated_monotonic_s = clock()
    return action


def finish_hold(robot, control, inputs, observation, rows, clock, sleep):
    if control.state is TeleopState.RUNNING:
        control.request_hold()
    # Existing finalization phase services feedback/hold without waiting for camera frames.
    control.recording_phase = "finalizing"
    while True:
        start = clock()
        inputs.poll()  # B remains active; LB cannot restart this completed segment.
        robot.send_action(hold_action(observation, control, clock))
        rows.append(dict(phase="final_hold", sdk=deepcopy(robot.last_action_telemetry)))
        if control.hold_confirmed:
            return
        observation = robot.get_observation()
        sleep(max(0, 0.02 - (clock() - start)))


def run_policy_trial(robot, teleop, control, predictor, safety, rows, **budgets):
    from lerobot_robot_outcome_piper.robot import PiperState

    predictor.reset_execution()
    try:
        return _run_policy_trial(robot, teleop, control, predictor, safety, rows, **budgets)
    except (Exception, KeyboardInterrupt) as exc:
        if robot.state is PiperState.ACTIVE:
            robot.request_input_fault(exc)
        raise
    finally:
        predictor.reset_execution()


def _run_policy_trial(
    robot,
    teleop,
    control,
    predictor,
    safety,
    rows,
    *,
    max_actions=None,
    max_run_s=None,
    frame_recorder=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Robot must already be enabled by the operator-started entry. One segment only."""
    if (max_actions is not None and (type(max_actions) is not int or max_actions <= 0)) or (
        max_run_s is not None and (not math.isfinite(max_run_s) or max_run_s <= 0)
    ):
        raise ValueError("trial budgets must be positive")
    inputs = PolicyInput(teleop, robot)
    robot.camera_input_poll = inputs.poll
    sent, first_dispatch, ready_announced = 0, None, False
    observation = None
    reason = None
    while reason is None:
        start = clock()
        raw = inputs.poll()
        intent, epoch = control.observe(raw["hold"], raw["neutral"])
        if first_dispatch is not None:
            if not raw["hold"]:
                reason = "operator_released_or_cancelled"
            elif (max_actions is not None and sent >= max_actions) or (
                max_run_s is not None and clock() - first_dispatch >= max_run_s
            ):
                reason = "segment_budget_reached"
            if reason is not None:
                break
        observation = robot.get_observation()
        # Camera waits may have serviced a release and invalidated the old epoch.
        raw = inputs.poll()
        intent, epoch = control.observe(raw["hold"], raw["neutral"])
        row = dict(
            phase="waiting",
            input=dict(inputs.last_raw),
            observation={k: float(observation[k]) for k in ACTION_KEYS},
            input_telemetry=deepcopy(robot.last_observation_telemetry),
        )
        rows.append(row)
        if intent == "run":
            if frame_recorder is not None:
                telemetry = row["input_telemetry"]
                camera = telemetry.get("cameras", {}).get("d435", {})
                row["rgb_recording"] = frame_recorder.submit(
                    observation["d435"],
                    {
                        "row_index": len(rows) - 1,
                        "observation_sequence": telemetry["sequence"],
                        "observed_monotonic_s": telemetry["observed_monotonic_s"],
                        "camera_frame_number": camera.get("frame_number"),
                        "camera_timestamp_ms": camera.get("device_timestamp_ms"),
                    },
                    clock(),
                )
            generated = clock()
            try:
                target, details = predict_candidate(
                    predictor,
                    observation,
                    robot.last_observation_telemetry,
                    robot.config.capture_timing,
                    safety,
                    clock,
                )
            except ValueError as exc:
                row.update(phase="candidate_rejected", error=str(exc))
                if hasattr(exc, "policy_timing"):
                    row["prediction_failure_timing"] = exc.policy_timing
                raise
            row.update(phase="candidate", target=target, prediction=details)
            raw = inputs.poll()  # Mandatory post-inference check, before either SDK command.
            new_intent, new_epoch = control.observe(raw["hold"], raw["neutral"])
            expired = (
                max_run_s is not None
                and first_dispatch is not None
                and clock() - first_dispatch >= max_run_s
            )
            if new_intent != "run" or new_epoch != epoch or expired:
                reason = "segment_budget_reached" if expired else "input_changed_during_inference"
                row["discarded"] = reason
                break
            action = OutcomePiperAction(target, intent="run", epoch=epoch)
            action.generated_monotonic_s = generated
            if first_dispatch is None:
                first_dispatch = clock()
            robot.send_action(action)
            row["sdk"] = deepcopy(robot.last_action_telemetry)
            if row["sdk"]["result"] != "sdk_returned":
                reason = "dispatch_rejected"
                break
            sent += 1
            row.update(phase="policy_sent", policy_action_index=sent)
        else:
            robot.send_action(hold_action(observation, control, clock))
            row["sdk"] = deepcopy(robot.last_action_telemetry)
            if control.hold_confirmed and not ready_announced:
                print(
                    "[策略就绪] 当前保持已确认，模型尚未开始控制。\n保持摇杆和扳机回中，重新按住LB开始策略。\n运行中持续按住LB；松开LB将结束本轮并保持当前位置。\n推杆或其他动作键也会取消本轮；B急停。",
                    flush=True,
                )
                ready_announced = True
        row["cycle_work_s"] = clock() - start
        sleep(max(0, 0.02 - (clock() - start)))
    finish_hold(robot, control, inputs, observation, rows, clock, sleep)
    return dict(
        status="segment_ended_held",
        end_reason=reason,
        policy_actions_sent=sent,
        gripper_completion="command_only",
        hold_confirmed=control.hold_confirmed,
    )
