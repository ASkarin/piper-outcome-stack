"""Policy selection and the real dispatch transaction, using synthetic SDK devices."""

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_xbox_pause import session as session, settle, tick
from test_policy_motion import Input
from lerobot_robot_outcome_piper.execution_constraints import check_execution_target
from lerobot_robot_outcome_piper.gripper_reference import GripperReferenceSettings
from lerobot_robot_outcome_piper.robot import PiperState
from lerobot_robot_outcome_piper.timing import CaptureTiming
from lerobot_robot_outcome_piper.camera import CameraTelemetry
from piper_outcome_stack.policy_execution import plan_policy_gripper, predict_candidate
from piper_outcome_stack.policy_motion import run_policy_trial


SETTINGS = GripperReferenceSettings(0.02, 0.12, 1.2, 0.008)


class Predictor:
    n_action_steps = 25
    temporal_ensemble_coeff = temporal_ensembler = None
    last_contributors = 1

    def __init__(self, goal=0.10, callback=lambda: None):
        self.goal, self.callback = goal, callback
        self.execution_details = {"phase": "policy"}
        self.resets = 0

    def reset_execution(self):
        self.resets += 1

    def execution_chunk(self, image, state, sequence, **kwargs):
        self.generation_state, self.generation_sequence = list(state), sequence
        chunk = np.repeat(np.asarray(state)[None], 50, axis=0)
        chunk[:, 6] = self.goal
        self.callback()
        return chunk, 0, True

    def select_target(self, chunk, **kwargs):
        return chunk[0].copy()


def policy_session(session):
    robot, arm, control, clock = session
    settle(session)
    robot._safety = replace(robot._safety, gripper_upper=0.12, max_gripper_step=0.015)
    robot.config = replace(robot.config, capture_timing=CaptureTiming(0.2, 0.1, 0.05, 0.2))
    frame = 0

    def camera_read(timeout):
        nonlocal frame
        frame += 1
        return (
            {"color": np.zeros((2, 3, 3), np.uint8)},
            {"color": CameraTelemetry(frame, float(frame), "hardware_clock", clock.now, clock.now)},
        )

    robot.cameras = {
        "d435": NS(is_connected=True, disconnect=lambda: None, read_with_metadata=camera_read)
    }
    return robot, arm, control, clock


def test_retention_requires_exact_successful_target_and_preserves_other_limits(session):
    robot, _, control, clock = policy_session(session)
    target = [0.0] * 6 + [0.03125]
    state = [0.0] * 6 + [0.05]
    with pytest.raises(ValueError, match="gripper exceeds"):
        check_execution_target(state, target, robot._safety)
    check_execution_target(state, target, robot._safety, last_gripper_target=0.03125)
    with pytest.raises(ValueError, match="gripper exceeds"):
        check_execution_target(
            state, [*target[:6], 0.03125 + 1e-12], robot._safety, last_gripper_target=0.03125
        )
    with pytest.raises(ValueError, match="J1 exceeds"):
        check_execution_target(
            state, [0.2, *target[1:]], robot._safety, last_gripper_target=0.03125
        )
    with pytest.raises(ValueError, match="gripper outside"):
        check_execution_target(state, [*target[:6], 0.13], robot._safety, last_gripper_target=0.13)

    control.gripper_target = 0.01  # Explicit accepted contact target, distinct from feedback.
    before = control.gripper_reference_snapshot(control.epoch)
    plan = plan_policy_gripper(0.01, 0.03, control, control.epoch, SETTINGS, robot._safety, clock())
    assert plan["target_m"] == 0.01
    assert control.gripper_reference_snapshot(control.epoch) == before


def test_initial_inference_wait_retains_the_actual_grasp_target(session):
    from test_plugin import valid_action

    robot, arm, control, clock = policy_session(session)
    tick(session, True, True, valid_action(0.0, 0.03125))
    arm.gripper.width = 0.05
    predictor = Predictor(0.05)
    predictor.execution_details = {"phase": "initial_hold"}
    observation = robot.get_observation()
    before = control.gripper_reference_snapshot(control.epoch)
    target, details = predict_candidate(
        predictor,
        observation,
        robot.last_observation_telemetry,
        robot.config.capture_timing,
        robot._safety,
        clock,
        True,
        control=control,
        epoch=control.epoch,
        gripper_reference_settings=SETTINGS,
    )
    assert target["gripper.pos"] == 0.03125
    assert details["policy_goal_rad_m"][6] == 0.05
    assert details["gripper_plan"]["reference"]["raw_goal_m"] == 0.03125
    assert control.gripper_reference_snapshot(control.epoch) == before


@pytest.mark.parametrize(
    "outcome", ["success", "invalid_goal", "release", "B", "arm_failure", "gripper_failure"]
)
def test_policy_loop_commits_only_after_successful_dispatch(session, outcome):
    robot, arm, control, clock = policy_session(session)
    raw, rows = Input(), []
    # Fault holding needs independently advancing synthetic feedback timestamps.
    clock.auto = 0.001

    def callback():
        if outcome == "release":
            raw.hold = False
        elif outcome == "B":
            raw.b = True

    if outcome == "arm_failure":
        arm.move_j = lambda *a: (_ for _ in ()).throw(OSError("synthetic arm failure"))
    elif outcome == "gripper_failure":
        arm.gripper.move_gripper_m = lambda *a, **kw: (_ for _ in ()).throw(
            OSError("synthetic gripper failure")
        )
    predictor = Predictor(0.13 if outcome == "invalid_goal" else 0.10, callback)

    def run():
        return run_policy_trial(
            robot,
            raw,
            control,
            predictor,
            robot._safety,
            rows,
            max_actions=2,
            max_run_s=1,
            clock=clock,
            sleep=lambda dt: clock.advance(dt),
            record_action_chunks=True,
            gripper_reference_settings=SETTINGS,
        )

    if outcome == "invalid_goal":
        with pytest.raises(ValueError, match="outside the existing command range"):
            run()
        assert robot.state is PiperState.FAULT and robot.stop_outcome == "hold_confirmed"
        assert (
            next(r for r in rows if r["phase"] == "candidate_rejected")["prediction"][
                "policy_goal_rad_m"
            ][6]
            == 0.13
        )
    elif outcome in ("B", "arm_failure", "gripper_failure"):
        with pytest.raises(RuntimeError):
            run()
    else:
        result = run()
        assert result["policy_actions_sent"] == (2 if outcome == "success" else 0)

    sent = [r for r in rows if r["phase"] == "policy_sent"]
    if outcome == "success":
        assert len(sent) == 2
        for row in sent:
            plan = row["prediction"]["gripper_plan"]
            assert row["sdk"]["gripper_reference_committed"]
            assert row["sdk"]["gripper_plan"] == plan
            assert row["target"]["gripper.pos"] == plan["reference"]["target_m"]
            assert plan["reference"]["raw_goal_m"] == 0.10
            assert row["prediction"]["absolute_action_chunk_rad_m"][0][6] == 0.10
        widths = [r["target"]["gripper.pos"] for r in sent]
        assert widths[0] == pytest.approx(0.03024)
        assert 0 < widths[1] - widths[0] <= 0.0024
    else:
        assert not sent and control.gripper_reference_state is None
        assert not arm.gripper.commands
    assert predictor.resets == 2
