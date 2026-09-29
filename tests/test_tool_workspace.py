"""Tool workspace math and dispatch checks, without real devices."""

from dataclasses import replace
import json
import math
import pytest
from test_processor import safety as base_safety, processor
from lerobot_robot_outcome_piper.workspace import (
    WorkspaceGeometry,
    workspace_coordinates,
    workspace_pose_allowed,
)
from lerobot_robot_outcome_piper.safety import load_motion_safety
from lerobot_robot_outcome_piper.errors import OutcomePiperValidationError


def geometry(**kw):
    return WorkspaceGeometry(
        **dict(
            tip_offset_m=[0, 0, 0.135],
            grasp_offset_m=[0, 0, 0.105],
            opening_axis=[0, 1, 0],
            table_normal=[0, 0, 1],
            table_offset_m=0,
            **kw,
        )
    )


def limits():
    return replace(
        base_safety(),
        workspace_lower=(-1.0, -1.0, 0.01),
        workspace_upper=(1.0, 1.0, 0.6),
        gripper_upper=0.09,
        max_gripper_step=0.015,
        workspace_geometry=geometry(),
    )


def test_tool_offset_rotates_with_flange():
    values = workspace_coordinates([0.2, 0.3, 0.2, 0, math.pi / 2, 0], 0, geometry())
    assert values[:3] == pytest.approx([0.335, 0.3, 0.2])
    assert workspace_coordinates([0.2, 0.3, 0.2, 0, 0, 0], 0, None) == (0.2, 0.3, 0.2)


def test_flange_in_box_can_have_tip_outside_box_or_below_table():
    s = limits()
    pose = [0.9, 0, 0.2, 0, math.pi / 2, 0]
    assert workspace_pose_allowed(pose, pose, 0, 0, replace(s, workspace_geometry=None))
    assert not workspace_pose_allowed(pose, pose, 0, 0, s)
    pose = [0, 0, 0.13, math.pi, 0, 0]
    assert not workspace_pose_allowed(pose, pose, 0, 0, s)


def test_opening_changes_tip_clearance_even_when_tcp_does_not_move():
    s = limits()
    pose = [0, 0, 0.02, math.pi / 2, 0, 0]
    assert workspace_pose_allowed(pose, pose, 0, 0, s)
    assert not workspace_pose_allowed(pose, pose, 0, 0.04, s)
    closed = workspace_coordinates(pose, 0, s.workspace_geometry)
    opened = workspace_coordinates(pose, 0.04, s.workspace_geometry)
    assert closed[:3] == pytest.approx(opened[:3])
    assert min(opened[3:]) == pytest.approx(0.0)


def test_reentry_cannot_improve_tcp_while_worsening_one_finger():
    s = limits()
    a = [0, 0, 0.005, math.pi / 2, 0, 0]
    b = [0, 0, 0.006, math.pi / 2, 0, 0]
    assert workspace_pose_allowed(a, b, 0, 0, s, allow_reentry=True)
    assert not workspace_pose_allowed(a, b, 0, 0.04, s, allow_reentry=True)
    assert not workspace_pose_allowed(a, a, 0, 0, s, allow_reentry=True)


def test_table_normal_and_offset_determine_signed_height():
    g = WorkspaceGeometry([0, 0, 0.135], [0, 0, 0.105], [0, 1, 0], [0, 0.6, 0.8], 0.1)
    values = workspace_coordinates([0, 0.2, 0.3, 0, 0, 0], 0, g)
    assert values[2] == pytest.approx(0.6 * 0.2 + 0.8 * 0.435 - 0.1)


@pytest.mark.parametrize(
    "name,value",
    [
        ("tip_offset_m", [0, 1]),
        ("opening_axis", [0, 2, 0]),
        ("table_normal", [0, 0, -1]),
        ("table_offset_m", float("nan")),
    ],
)
def test_invalid_geometry_is_not_silently_repaired(name, value):
    v = dict(
        tip_offset_m=[0, 0, 0.135],
        grasp_offset_m=[0, 0, 0.105],
        opening_axis=[0, 1, 0],
        table_normal=[0, 0, 1],
        table_offset_m=0,
    )
    v[name] = value
    with pytest.raises(ValueError):
        WorkspaceGeometry(**v)


def test_processor_roundtrip_keeps_geometry():
    p = processor(safety=limits())
    restored = type(p)(**p.get_config())
    assert restored.safety.workspace_geometry == p.safety.workspace_geometry
    assert restored.get_config() == p.get_config()


def test_geometry_config_load_and_no_implicit_workspace(tmp_path):
    from dataclasses import asdict

    s = limits()
    v = dict(
        schema_version="outcome-piper-safety-v1",
        joint_lower_rad=list(s.joint_lower),
        joint_upper_rad=list(s.joint_upper),
        max_joint_step_rad=list(s.max_joint_step),
        gripper_lower_m=0,
        gripper_upper_m=0.09,
        max_gripper_step_m=0.015,
        workspace_lower_m=list(s.workspace_lower),
        workspace_upper_m=list(s.workspace_upper),
        feedback_timeout_s=s.feedback_timeout_s,
        watchdog_timeout_s=s.watchdog_timeout_s,
        motion_speed_percent=5,
        gripper_force_n=1,
        stop_strategy=s.stop_strategy,
        workspace_geometry=asdict(s.workspace_geometry),
    )
    p = tmp_path / "safety.json"
    p.write_text(json.dumps(v))
    assert load_motion_safety(p).workspace_geometry == s.workspace_geometry
    v["workspace_lower_m"] = None
    p.write_text(json.dumps(v))
    with pytest.raises(OutcomePiperValidationError):
        load_motion_safety(p)


def test_gripper_only_validation_checks_clearance(tmp_path, monkeypatch):
    from test_plugin import make_robot
    from lerobot_robot_outcome_piper.errors import OutcomePiperIntentRejected
    from pyAgxArm.utiles import mdh_kinematics

    (robot, arm, _) = make_robot(tmp_path)
    robot._safety = replace(limits(), max_gripper_step=0.05)
    monkeypatch.setattr(mdh_kinematics, "fk_from_mdh", lambda *a: [0, 0, 0.02, math.pi / 2, 0, 0])
    with pytest.raises(OutcomePiperIntentRejected, match="clearance"):
        robot._validate_gripper_target(0.04, 0, current_joints=[0] * 6)
    assert not arm.gripper.commands


def test_teach_conversion_uses_same_tool_workspace(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper.teach_dataset import check_candidate
    from pyAgxArm.utiles import mdh_kinematics

    a = {"joint_rad": [0] * 6, "gripper_m": 0.0}
    b = {"joint_rad": [0] * 6, "gripper_m": 0.01}
    monkeypatch.setattr(mdh_kinematics, "fk_from_mdh", lambda *a: [0, 0, 0.13, math.pi, 0, 0])
    with pytest.raises(ValueError, match="workspace/table"):
        check_candidate(a, b, limits())


def test_pose_sequence_does_not_bypass_tool_clearance(monkeypatch):
    from lerobot_robot_outcome_piper.joint_pose import JointPoseSequence
    from lerobot_robot_outcome_piper.teleop_control import HoldSettings
    from lerobot_robot_outcome_piper.errors import OutcomePiperIntentRejected
    from pyAgxArm.utiles import mdh_kinematics

    monkeypatch.setattr(mdh_kinematics, "fk_from_mdh", lambda *a: [0, 0, 0.13, math.pi, 0, 0])
    with pytest.raises(OutcomePiperIntentRejected, match="workspace"):
        JointPoseSequence(
            [0] * 6, 0.0, limits(), HoldSettings(0.001, 0.1, 1.0), [0.01] * 6 + [0.0], "work"
        )


def test_rejected_tool_goal_sends_no_motion_command(tmp_path, monkeypatch):
    from test_plugin import make_robot, connect_for_test, valid_action
    from lerobot_robot_outcome_piper.errors import OutcomePiperIntentRejected
    from pyAgxArm.utiles import mdh_kinematics

    robot, arm, _ = make_robot(tmp_path, mode="motion")
    connect_for_test(robot)
    robot._safety = limits()
    monkeypatch.setattr(mdh_kinematics, "fk_from_mdh", lambda *a: [0.9, 0, 0.2, 0, math.pi / 2, 0])
    before = list(arm.calls)
    try:
        with pytest.raises(OutcomePiperIntentRejected, match="workspace"):
            robot.send_action(valid_action())
        assert arm.calls == before
    finally:
        robot.disconnect()


def test_lifting_target_cannot_hide_unsafe_opening_at_current_pose():
    s = limits()
    current = [0, 0, 0.02, math.pi / 2, 0, 0]
    target = [0, 0, 0.1, math.pi / 2, 0, 0]
    assert not workspace_pose_allowed(current, target, 0, 0.04, s)


def test_signed_measured_gap_is_preserved_without_allowing_negative_commands(tmp_path):
    a = workspace_coordinates([0, 0, 0.2, math.pi / 2, 0, 0], -0.0009, geometry())
    b = workspace_coordinates([0, 0, 0.2, math.pi / 2, 0, 0], 0.0009, geometry())
    assert a[3] == pytest.approx(b[4]) and a[4] == pytest.approx(b[3])
    from test_plugin import make_robot
    from lerobot_robot_outcome_piper.errors import OutcomePiperIntentRejected

    robot, _, _ = make_robot(tmp_path)
    robot._safety = limits()
    with pytest.raises(OutcomePiperIntentRejected, match="outside frozen limits"):
        robot._validate_gripper_target(-0.0009, 0.0011, current_joints=[0] * 6)


def test_inward_arm_reentry_can_include_safe_gripper_change():
    s = limits()
    assert workspace_pose_allowed(
        [1.01, 0, 0.2, 0, 0, 0], [1.005, 0, 0.2, 0, 0, 0], 0, 0.01, s, allow_reentry=True
    )


def test_grasp_center_uses_configured_offset_in_fk_and_position_ik():
    from dataclasses import replace
    import numpy as np
    from scipy.spatial.transform import Rotation
    from lerobot_robot_outcome_piper.workspace import grasp_position, WorkspaceGeometry
    from lerobot_robot_outcome_piper.position_ik import solve_position
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    safety = limits()
    geometry = WorkspaceGeometry((0, 0, 0.135), (0.01, 0, 0.12), (0, 1, 0), (0, 0, 1), 0.0)
    safety = replace(
        safety,
        workspace_geometry=geometry,
        workspace_lower=(-2.0, -2.0, -2.0),
        workspace_upper=(2.0, 2.0, 2.0),
    )
    q = np.deg2rad([0, 40, -25, 10, 25, 0]).tolist()
    pose = fk_from_mdh(list(get_mdh("piper")), q)
    expected = np.asarray(pose[:3]) + Rotation.from_euler("xyz", pose[3:]).apply(
        geometry.grasp_offset_m
    )
    assert grasp_position(pose, geometry) == pytest.approx(expected)
    target = expected + np.array([0, 0, 0.0005])
    solved, detail = solve_position(q, q, target, safety, 0.02, 0.02, timeout=1.0, max_nfev=100)
    assert grasp_position(fk_from_mdh(list(get_mdh("piper")), solved), geometry) == pytest.approx(
        target, abs=1e-5
    )
    assert detail["level"] == "fixed_wrist"
    assert solved[3:] == q[3:]
