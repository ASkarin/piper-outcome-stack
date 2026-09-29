import copy
import numpy as np
import pytest
from action_evidence import dispatched
from lerobot_robot_outcome_piper.action_audit import validate_action
from lerobot_robot_outcome_piper.safety import ACTION_KEYS


@pytest.mark.parametrize("mutation", ["missing", "failed", "target", "timestamp", "gripper", "nan"])
def test_moving_action_requires_sdk_proof(mutation):
    action = dispatched(dict.fromkeys(ACTION_KEYS, 0.0))
    if mutation == "missing":
        action.pop("commands")
    if mutation == "failed":
        action["commands"][0]["result"] = "failed"
    if mutation == "target":
        action["commands"][0]["target"][0] = 1.0
    if mutation == "timestamp":
        action["commands"][0]["ended_monotonic_s"] = 0.0
    if mutation == "gripper":
        action["commands"].pop()
    if mutation == "nan":
        action["values"][ACTION_KEYS[0]] = float("nan")
    with pytest.raises(RuntimeError):
        validate_action(action)


def test_retained_gripper_and_float32_targets_need_no_fake_dispatch():
    action = dispatched(dict.fromkeys(ACTION_KEYS, 0.0123456789))
    action["retained_gripper_command"] = action["commands"].pop()
    action["values"] = {k: float(np.float32(v)) for k, v in action["values"].items()}
    before = copy.deepcopy(action)
    validate_action(action)
    assert action == before
