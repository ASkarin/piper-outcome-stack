"""Parse actual user JSON through the pinned LeRobot CLI without opening hardware."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

pytest.importorskip("lerobot")

sys.path.insert(0, str(Path(__file__).parents[1] / "packages/lerobot_robot_outcome_piper/src"))

import draccus  # noqa: E402
from draccus.utils import ParsingError  # noqa: E402
from lerobot.robots.config import RobotConfig  # noqa: E402
from lerobot_robot_outcome_piper import cli  # noqa: E402
from lerobot_robot_outcome_piper.config import OutcomePiperConfig  # noqa: E402
from test_plugin import xbox_config  # noqa: E402


def robot_values():
    return dict(type="outcome_piper", can_interface="can0", firmware="v189", feedback_timeout_s=0.2)


@pytest.mark.parametrize("firmware", ["default", "v183", "v188", "v189"])
def test_real_robot_config_parser_preserves_explicit_firmware(tmp_path, firmware):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps({**robot_values(), "firmware": firmware}))
    config = draccus.parse(RobotConfig, config_path=path, args=[])
    assert isinstance(config, OutcomePiperConfig)
    assert config.firmware == firmware
    assert config.execution_mode == "read_only"


@pytest.mark.parametrize(
    "values, message",
    [
        ({"firmware": "guess"}, "firmware"),
        ({"execution_mode": "force"}, "execution_mode"),
        ({"execution_mode": "motion"}, "safety_path"),
    ],
)
def test_real_parser_keeps_validation_and_motion_gate(tmp_path, values, message):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps({**robot_values(), **values}))
    with pytest.raises(ParsingError) as error:
        draccus.parse(RobotConfig, config_path=path, args=[])
    assert isinstance(error.value.__cause__, ValueError)
    assert message in str(error.value.__cause__)


@pytest.mark.parametrize("workflow", ["record", "teleoperate"])
def test_official_cli_parses_complete_config_before_workflow(tmp_path, monkeypatch, workflow):
    values = robot_values()
    values.update(
        execution_mode="motion",
        safety_path=str(tmp_path / "safety.json"),
        hardware_acceptance_path=str(tmp_path / "acceptance.json"),
        cameras={
            "d435": dict(
                type="intelrealsense",
                serial_number_or_name="123456789012",
                width=640,
                height=480,
                fps=30,
                use_rgb=True,
                use_depth=False,
            )
        },
        capture_timing=dict(
            camera_max_age_s=0.1,
            joint_max_skew_s=0.02,
            image_state_max_skew_s=0.1,
            observation_max_age_s=0.2,
        ),
    )
    # Explicitly synthetic values; parser test never constructs/opens a Robot.
    teleop = asdict(xbox_config(control_hz=30))
    teleop["type"] = "outcome_piper_xbox"
    for key in ("calibration_dir",):
        if teleop.get(key) is not None:
            teleop[key] = str(teleop[key])
    config = dict(robot=values, teleop=teleop, display_data=False)
    if workflow == "record":
        config["dataset"] = dict(
            repo_id="test/parser",
            single_task="parser only",
            fps=30,
            root=str(tmp_path / "data"),
            push_to_hub=False,
        )
    else:
        config.update(fps=30, teleop_time_s=1)
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(config))
    received = []
    monkeypatch.setattr(cli, workflow, received.append)
    getattr(cli, workflow + "_main")([f"--config_path={path}"])
    assert len(received) == 1
    parsed = received[0]
    assert parsed.robot.firmware == "v189"
    assert parsed.robot.execution_mode == "motion"
    assert parsed.robot.safety_path == tmp_path / "safety.json"
    assert set(parsed.robot.cameras) == {"d435"}
    assert parsed.robot.capture_timing.joint_max_skew_s == 0.02
    assert parsed.teleop.control_hz == 30
