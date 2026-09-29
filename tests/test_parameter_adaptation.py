import math
import json
from pathlib import Path
import numpy as np
import pytest
from lerobot_robot_outcome_piper.continuous_reference import ReferenceSettings, reference_candidate
from lerobot_robot_outcome_piper.safety import load_motion_safety
from lerobot_robot_outcome_piper.teach_dataset import check_candidate


def test_doubled_input_preserves_ramp_acceleration_and_reference_lead():
    old = ReferenceSettings(0.05, 0.2, 2)
    new = ReferenceSettings(0.05, 0.4, 1)
    outputs = []
    for settings, xyz, rot in ((old, 0.0025, math.radians(0.5)), (new, 0.005, math.radians(1))):
        prev = None
        first = None
        for i in range(80):
            prev = reference_candidate(
                prev,
                [0, 0, 0],
                np.eye(3),
                np.eye(3),
                [xyz, 0, 0],
                [0, 0, rot],
                xyz,
                rot,
                settings,
                i * 0.05,
            )
            if first is None:
                first = prev["velocity"]
        outputs.append((first, prev))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-12)
    for _, last in outputs:
        assert last["position"][0] <= 0.005 + 1e-10
    np.testing.assert_allclose(outputs[0][1]["rotation"], outputs[1][1]["rotation"], atol=1e-8)


def test_requested_limits_accept_new_targets_and_keep_upper_bound(tmp_path):
    root = Path(__file__).parents[1]
    values = json.loads((root / "configs/scenes/new-table/safety.template.json").read_text())
    values["workspace_lower_m"] = [-2, -2, -2]
    values["workspace_upper_m"] = [2, 2, 2]
    p = tmp_path / "safety.json"
    p.write_text(json.dumps(values))
    safety = load_motion_safety(p)
    a = {"joint_rad": [0.1, 0.1, -0.1, 0.1, 0.1, 0.1], "gripper_m": 0.105}
    b = {"joint_rad": [0.1 + math.radians(9.9), 0.1, -0.1, 0.1, 0.1, 0.1], "gripper_m": 0.12}
    check_candidate(a, b, safety)
    with pytest.raises(ValueError, match="execution step"):
        check_candidate({**a, "gripper_m": 0.104}, b, safety)
    b["gripper_m"] = 0.12001
    with pytest.raises(ValueError, match="gripper outside"):
        check_candidate(a, b, safety)
    b["gripper_m"] = 0.12
    b["joint_rad"][0] = 0.1 + math.radians(10.1)
    with pytest.raises(ValueError, match="execution step"):
        check_candidate(a, b, safety)
