import json
from pathlib import Path
import draccus
import pytest
from draccus.utils import ParsingError
from lerobot_robot_outcome_piper.cli import PiperRecordConfig as RecordConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot_robot_outcome_piper.config import OutcomePiperConfig


@pytest.mark.parametrize("name", ["measure", "teleoperate", "record"])
def test_scene_templates_require_site_values_then_parse_without_devices(tmp_path, name):
    template = Path(__file__).parents[1] / "configs/scenes/new-table" / f"{name}.template.json"
    values = json.loads(template.read_text())
    robot = values["robot"]
    robot["scene"] = dict(
        scene_id="synthetic-new-table",
        base_installation="synthetic fixed base",
        camera_view="synthetic view",
        work_area_notes="synthetic area",
        camera_to_base_calibration=None,
    )
    if name == "measure":
        robot.pop("type")
        cfg = draccus.decode(OutcomePiperConfig, robot)
        assert cfg.execution_mode == "read_only" and cfg.capture_timing is None
    else:
        with pytest.raises((ParsingError, ValueError, TypeError)):
            draccus.decode(RecordConfig if name == "record" else TeleoperateConfig, values)
        robot["safety_path"] = str(tmp_path / "synthetic-safety.json")
        values["teleop"]["translation_switch_button"] = 2  # Synthetic binding only.
        values["teleop"]["work_pose_button"] = 0  # Synthetic parser input, not hardware evidence.
        if name == "record":
            robot["capture_timing"] = dict(
                camera_max_age_s=0.1,
                joint_max_skew_s=0.01,
                image_state_max_skew_s=0.1,
                observation_max_age_s=0.2,
            )
            values["dataset"]["root"] = str(tmp_path / "synthetic-dataset")
        cfg = draccus.decode(RecordConfig if name == "record" else TeleoperateConfig, values)
        assert cfg.robot.scene.scene_id == "synthetic-new-table"
        assert cfg.teleop.work_pose_button == 0
        if name == "teleoperate":
            assert cfg.robot.cameras == {} and cfg.robot.capture_timing is None
