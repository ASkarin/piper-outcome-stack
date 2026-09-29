import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from lerobot.types import TransitionKey
from lerobot_robot_outcome_piper.processor import make_xbox_processor
from lerobot_robot_outcome_piper.action_audit import holding_reference
from lerobot_robot_outcome_piper.teleop_control import TeleopState
from test_plugin import valid_action
from test_xbox_pause import session as original_session, settle, tick  # noqa: F401
from test_processor import raw_action, require_sdk_kinematics


@pytest.fixture
def session(original_session):  # noqa: F811
    return original_session


def test_rearm_neutral_keeps_actual_hold_and_never_invokes_ik_or_sdk(session, monkeypatch):
    robot, arm, c, clock = session
    settle(session)
    tick(session, True, True, valid_action(0.05, 0.035))
    # The dispatched target can be ahead of the actual feedback when LB is released.
    c.commit_orientation(c.epoch, Rotation.from_rotvec([0.4, 0.2, 0.1]).as_matrix())
    arm.joints = [0.02] * 6
    tick(session, False, True)
    clock.advance()
    tick(session, False, True)
    assert c.state is TeleopState.PAUSED
    k = require_sdk_kinematics()
    held = Rotation.from_euler(
        "xyz", k.fk_from_mdh(list(k.get_mdh("piper")), robot._hold_window.target)[3:]
    ).as_matrix()
    assert np.asarray(c.orientation_target) == pytest.approx(held)
    pipeline = make_xbox_processor(
        robot._safety,
        max_xyz_step_m=0.0025,
        max_rotation_step_rad=0.0087,
        max_gripper_step_m=0.005,
        ik_max_nfev=100,
        ik_timeout_s=0.025,
        ik_residual_tolerance=1e-5,
        ik_min_singular_value=0.005,
    )
    p = pipeline.steps[0]
    p.control = c
    from lerobot_robot_outcome_piper.teleop_control import TranslationStrategy

    c.translation_strategy = TranslationStrategy.FIXED_ORIENTATION

    def no_ik(*args):
        pytest.fail("neutral rearm invoked IK")

    monkeypatch.setattr(p, "_solve", no_ik)
    before = list(arm.calls)
    grip = list(arm.gripper.commands)
    hold_id = robot._hold_id
    for _ in range(80):
        obs = robot.get_observation()
        p._current_transition = {TransitionKey.OBSERVATION: obs}
        a = p.action(raw_action())
        a.generated_monotonic_s = clock.now
        assert a.intent == "center"
        result = robot.send_action(a)
        assert result["gripper.pos"] == 0.035
        assert robot.last_action_telemetry["result"] == "holding"
        assert robot.last_action_telemetry["commands"] == []
        assert holding_reference(robot.last_action_telemetry)["values"]["gripper.pos"] == 0.035
        clock.advance()
        assert robot._watchdog_check_locked()
    assert arm.calls == before and arm.gripper.commands == grip
    assert robot._hold_id == hold_id
    assert np.asarray(c.orientation_target) == pytest.approx(held)
    calls = []
    monkeypatch.setattr(p, "_solve", lambda q, pose: calls.append((q, pose)) or q)
    p._current_transition = {TransitionKey.OBSERVATION: robot.get_observation()}
    a = p.action({**raw_action(), "stick_x": 0.1, "neutral": False})
    assert a.intent == "run" and calls[0][0] == arm.joints
    assert Rotation.from_euler("xyz", calls[0][1][3:]).as_matrix() == pytest.approx(held)
