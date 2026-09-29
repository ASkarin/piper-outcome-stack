from dataclasses import replace
import pytest
from test_processor import processor, raw_action, observation
from test_plugin import make_robot, valid_action
from lerobot.types import TransitionKey
from lerobot_robot_outcome_piper.safety import step_within_limit


@pytest.mark.parametrize(
    "current,limit", [(0.025, 0.0005), (-0.025, 0.0005), (1.0, 0.08726646259971647)]
)
def test_exact_step_roundoff_allowed_but_real_excess_rejected(current, limit):
    assert step_within_limit(current + limit, current, limit)
    assert step_within_limit(current - limit, current, limit)
    assert not step_within_limit(current + limit + 1e-12, current, limit)
    assert not step_within_limit(current - limit - 1e-12, current, limit)


def test_processor_full_gripper_tick_is_not_rejected(monkeypatch):
    p = processor(max_gripper_step_m=0.0005)
    p.safety = replace(p.safety, max_gripper_step=0.0005)
    p._current_transition = {TransitionKey.OBSERVATION: observation([0.1] * 6, gripper=0.025)}
    monkeypatch.setattr(p, "_solve", lambda q, target: q)
    result = p.action({**raw_action(), "right_trigger": 1.0, "neutral": False})
    assert result.intent == "center" and result["gripper.pos"] == 0.025 + 0.0005


def test_robot_full_gripper_tick_is_not_rejected(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    robot.connect()
    robot.enable()
    try:
        arm.gripper.width = 0.025
        robot._safety = replace(robot._safety, max_gripper_step=0.0005)
        action = valid_action()
        action["gripper.pos"] = 0.025 + 0.0005
        assert robot.send_action(action)["gripper.pos"] == action["gripper.pos"]
    finally:
        robot.disconnect()
