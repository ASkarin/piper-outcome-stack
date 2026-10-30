"""Fine control and protection filtering with fake devices and real Dataset I/O."""

from types import SimpleNamespace as NS
import json

import numpy as np
import pytest
from action_evidence import dispatched
from lerobot_robot_outcome_piper.demonstration_segments import (
    DemonstrationTracker,
    segments_from_telemetry,
)
from lerobot_robot_outcome_piper.record_control import EpisodeControls
from lerobot_robot_outcome_piper.teleop_control import TeleopControl, TeleopState
from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from test_record_control import Terminal


def neutral(**kwargs):
    return {"hold": False, "neutral": True, **kwargs}


def held(values, sequence, *, reason=None, gripper_command=False):
    normal = dispatched(values, sequence)
    joint, grip = normal["commands"]
    hold = {**joint, "name": "hold_move_j"}
    return {
        **normal,
        "result": "holding",
        "reason": reason,
        "rejection_reason": reason,
        "hold_id": sequence + 1,
        "hold_confirmed": True,
        "hold_command": hold,
        "retained_gripper_command": grip,
        "commands": [grip] if gripper_command else [hold],
        "control_state": "CENTERED" if gripper_command else "PAUSED",
    }


def test_precision_switch_is_paused_and_preserves_permission_and_gripper_reference():
    c = TeleopControl()
    c.confirm_hold()
    c.gripper_reference_state = {"target_m": 0.03, "velocity_m_s": 0.05}
    c.reference_state = {"velocity": [1] * 6}
    epoch = c.epoch
    c.select_precision("fine", **neutral())
    assert c.input_scale == 0.5 and c.reference_state is None
    assert c.gripper_reference_state["velocity_m_s"] == 0.05
    assert c.epoch == epoch and c.hold_confirmed
    for raw in [neutral(hold=True), neutral(neutral=False), neutral(conflicting=True)]:
        with pytest.raises(ValueError):
            c.select_precision("normal", **raw)
        assert c.precision == "fine"
    c.state = TeleopState.RUNNING
    with pytest.raises(ValueError):
        c.select_precision("normal", **neutral())


@pytest.mark.parametrize("precision,scale", [("normal", 1), ("fine", 0.5)])
def test_processor_scales_cartesian_target_but_keeps_gripper_increment(
    monkeypatch, precision, scale
):
    from test_processor import processor, raw_action, observation
    from lerobot.types import TransitionKey
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    p = processor()
    p.control.observe(False, True)
    p.control.confirm_hold()
    p.control.select_precision(precision, **neutral())
    p.control.observe(True, True)
    q = [0.2, 0.5, -0.4, 0.2, 0.5, -0.2]
    poses = []
    monkeypatch.setattr(p, "_solve", lambda seed, pose: poses.append(pose) or q)
    p._current_transition = {TransitionKey.OBSERVATION: observation(q)}
    raw = raw_action(dx=0.005)
    raw["right_trigger"] = 1
    result = p.action(raw)
    initial = fk_from_mdh(list(get_mdh("piper")), q)
    assert poses[0][0] - initial[0] == pytest.approx(0.005 * scale, abs=1e-9)
    assert result["gripper.pos"] == pytest.approx(0.034)


def test_lb_automatically_pauses_data_without_keyboard_and_keeps_task_holding():
    c = TeleopControl()
    c.recording_phase = "recording"
    c.confirm_hold()
    c.observe(False, True)
    t = DemonstrationTracker()
    values = dict.fromkeys(ACTION_KEYS, 0.03)

    def capture(index):
        action = held(values, index)
        action["demonstration_pause"] = c.demonstration_snapshot()
        return t.observe(action)

    assert not capture(0)["eligible"]  # Includes initial typing/menu delay.
    c.observe(True, True)
    c.arm_input_intent(c.epoch, False)
    assert c.state is TeleopState.CENTERED and capture(1)["eligible"]
    c.arm_input_intent(c.epoch, True)
    c.observe(False, True)
    assert not capture(2)["eligible"]
    c.confirm_hold()
    assert c.state is TeleopState.PAUSED
    assert all(not capture(i)["eligible"] for i in range(3, 103))  # Two seconds, no command.
    assert not c.permits(c.epoch)
    c.observe(True, True)
    c.arm_input_intent(c.epoch, False)
    assert capture(103)["eligible"]


def test_camera_wait_hold_revokes_data_before_next_regular_input():
    c = TeleopControl()
    c.recording_phase = "recording"
    c.confirm_hold()
    c.observe(False, True)
    c.observe(True, True)
    assert not c.demonstration_snapshot()["manual"]
    c.request_hold()  # Same path as shoulder release serviced during camera wait.
    assert c._previous_hold is True and c.demonstration_snapshot()["manual"]


def test_recording_has_no_keyboard_pause_resume_or_gain_switch(capsys):
    c = TeleopControl()
    c.confirm_hold()
    ui = EpisodeControls(c, {}, Terminal(["fine", "pause", "resume"]), lambda *a, **k: None)
    ui.enter("recording")
    for _ in range(3):
        ui.poll(neutral())
        assert ui.request is None and c.precision == "normal"
    assert "无需输入pause/resume" in capsys.readouterr().out


def test_status_reports_protection_recovery_changes_immediately(monkeypatch, capsys):
    import lerobot_robot_outcome_piper.record_control as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    c = TeleopControl()
    c.confirm_hold()
    c.request_protection_hold("工具目标越过桌面边界")
    ui = EpisodeControls(c, {}, Terminal([]), lambda *a, **k: None)
    ui.enter("recording")
    ui.show_status(neutral(hold=True), {})
    first = capsys.readouterr().out
    assert "工具目标越过桌面边界" in first and "请松开LB" in first
    now[0] += 20
    ui.show_status(neutral(hold=True), {})
    assert capsys.readouterr().out == ""
    ui.show_status(neutral(), {})
    assert "正在确认保持" in capsys.readouterr().out
    ui.show_status(neutral(neutral=False), {})
    assert "请让摇杆/扳机回中" in capsys.readouterr().out
    c.confirm_hold()
    ui.show_status(neutral(), {})
    assert "重新按LB" in capsys.readouterr().out
    c.state = TeleopState.FAULT
    ui.show_status(neutral(), {})
    assert "故障/急停会话已锁存" in capsys.readouterr().out


def test_status_does_not_refresh_or_compute_clearance_during_unchanged_motion(monkeypatch, capsys):
    import pyAgxArm.utiles.mdh_kinematics as kinematics
    import lerobot_robot_outcome_piper.workspace as workspace

    calls = []
    monkeypatch.setattr(kinematics, "fk_from_mdh", lambda mdh, q: calls.append(q) or [0] * 6)
    monkeypatch.setattr(workspace, "workspace_coordinates", lambda *args: [0, 0, 0.02, 0.03])
    safety = NS(workspace_geometry=object(), workspace_lower=(0, 0, 0.005))
    observation = dict.fromkeys(ACTION_KEYS, 0.03)
    c = TeleopControl()
    ui = EpisodeControls(c, {}, Terminal([]), lambda *a, **k: None)
    ui.enter("recording")
    c.confirm_hold()
    c.observe(False, True)
    c.observe(True, True)
    c.arm_input_intent(c.epoch, False)
    ui.show_status(neutral(hold=True), observation, safety)
    assert "示范记录有效" in capsys.readouterr().out
    assert len(calls) == 1
    c.arm_input_intent(c.epoch, True)
    for _ in range(400):
        observation["joint_1.pos"] += 0.0001
        ui.show_status(neutral(hold=True), observation, safety)
    assert capsys.readouterr().out == "" and len(calls) == 1
    c.observe(False, True)
    ui.show_status(neutral(), observation, safety)
    assert "示范暂停" in capsys.readouterr().out and len(calls) == 2
    c.confirm_hold()
    c.select_precision("fine", **neutral())
    ui.show_status(neutral(), observation, safety)
    assert "精细50%" in capsys.readouterr().out and len(calls) == 3
    ui.enter("preparing")
    capsys.readouterr()
    ui.show_status(neutral(), observation, safety)
    assert "准备中，尚未录制" in capsys.readouterr().out


def test_transient_protection_reason_persists_until_new_gripper_command():
    t = DemonstrationTracker()
    values = dict.fromkeys(ACTION_KEYS, 0.03)
    assert t.observe(dispatched(values))["eligible"]
    assert not t.observe(held(values, 2, reason="IK did not converge"))["eligible"]
    assert not t.observe(held(values, 3))["eligible"]
    resumed = t.observe(held({**values, "gripper.pos": 0.028}, 4, gripper_command=True))
    assert resumed["eligible"] and resumed["segment_id"] == 1
    assert t.observe(held(values, 5))["eligible"]  # Ordinary holding is useful data.


def test_pause_metadata_does_not_end_just_because_an_sdk_command_succeeds():
    t = DemonstrationTracker()
    action = dispatched(dict.fromkeys(ACTION_KEYS, 0.03))
    action["demonstration_pause"] = {"manual": True, "protection_reason": None}
    assert not t.observe(action)["eligible"]
    action["demonstration_pause"]["manual"] = False
    assert t.observe(action)["eligible"]


def make_capture(tmp_path, *, all_protected=False):
    from lerobot_robot_outcome_piper.recording import TelemetryDataset
    from lerobot_robot_outcome_piper.xbox_raw import RawFrames

    features = {
        key: {"dtype": "float32", "shape": (7,), "names": list(ACTION_KEYS)}
        for key in ("action", "observation.state")
    }
    features["observation.images.rgb"] = {
        "dtype": "image",
        "shape": (16, 16, 3),
        "names": ["height", "width", "channels"],
    }
    raw = RawFrames(
        tmp_path / "raw",
        fps=50,
        features=features,
        robot_type="outcome_piper",
        conversion_options={"use_videos": False},
    )
    robot = NS(config=NS(), last_depth_frames={})
    audit = TelemetryDataset(raw, robot)
    for ep, length in enumerate([6, 2] if not all_protected else [3]):
        audit.prepare_episode().result()
        for i in range(length):
            sequence = ep * 6 + i
            values = dict.fromkeys(ACTION_KEYS, sequence * 0.001)
            if all_protected or (ep == 0 and i in (2, 3)):
                action = held(
                    values, sequence, reason="workspace boundary" if i == 2 or i == 0 else None
                )
            elif ep == 0 and i == 4:
                action = held(values, sequence, gripper_command=True)
            else:
                action = dispatched(values, sequence)
            robot.last_action_telemetry = action
            robot.last_observation_telemetry = {
                "sequence": sequence,
                "quality": "checked",
                "observed_monotonic_s": 1 + sequence * 0.02,
                "cameras": {},
            }
            audit.add_frame(
                {
                    "action": np.array(list(values.values()), np.float32),
                    "observation.state": np.zeros(7, np.float32),
                    "task": "synthetic",
                    "observation.images.rgb": np.full((16, 16, 3), sequence, np.uint8),
                }
            )
        audit.save_episode()
        audit.emit(
            "episode_outcome",
            episode_index=ep,
            attempt=ep,
            task_outcome="success",
            data_valid=True,
            intended_split="train",
        )
    raw.finalize()
    audit.complete()
    audit.close()
    raw.finish_capture(True)
    return raw.root


def test_filtered_conversion_preserves_pixels_actions_and_chunk_boundaries(tmp_path):
    from lerobot_robot_outcome_piper.xbox_raw import convert
    from lerobot_robot_outcome_piper.action_audit import verify_telemetry
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from piper_outcome_stack.training_selection import prepare

    raw = make_capture(tmp_path)
    before = [p.read_bytes() for p in sorted(raw.glob("attempts/*/frames.jsonl"))]
    out = tmp_path / "dataset"
    result = convert(raw, out, "local/protection-filter")
    assert result["frames"] == 6 and result["episodes"] == 3
    assert result["source_episodes"] == 2 and result["excluded_wait_frames"] == 2
    assert before == [p.read_bytes() for p in sorted(raw.glob("attempts/*/frames.jsonl"))]
    ds = LeRobotDataset(
        "local/protection-filter",
        root=out,
        delta_timestamps={"action": [i / 50 for i in range(50)]},
    )
    assert verify_telemetry(out, ds) == {"episodes": 3, "frames": 6}
    assert np.allclose(ds[1]["action"][:, 0], 0.001)
    assert ds[1]["action_is_pad"].tolist() == [False] + [True] * 49
    assert float(ds[2]["action"][0, 0]) == pytest.approx(0.004)
    assert float(ds[2]["observation.images.rgb"].mean() * 255) == pytest.approx(4)
    mapping = json.loads((out / "demonstration-segments.json").read_text())
    assert [(s["first_frame"], s["end_frame_exclusive"]) for s in mapping["segments"]] == [
        (0, 2),
        (4, 6),
        (0, 2),
    ]
    config = tmp_path / "training.json"
    config.write_text(
        json.dumps(
            {"dataset": {"root": str(out), "repo_id": ds.repo_id}, "policy": {"type": "act"}}
        )
    )
    selected = prepare(config, tmp_path / "selected.json", [1])
    assert selected["validation_episodes"] == [0, 1] and selected["train_episodes"] == [2]
    # A map that includes a protected frame must fail the normal dataset audit.
    mapping["segments"][0]["end_frame_exclusive"] = 3
    (out / "demonstration-segments.json").write_text(json.dumps(mapping))
    with pytest.raises(RuntimeError, match="differs from source"):
        verify_telemetry(out, ds)


def test_entirely_protected_capture_is_not_a_training_dataset(tmp_path):
    from lerobot_robot_outcome_piper.xbox_raw import convert

    raw = make_capture(tmp_path, all_protected=True)
    assert segments_from_telemetry(raw)["retained_frames"] == 0
    out = tmp_path / "dataset"
    with pytest.raises(ValueError, match="no eligible"):
        convert(raw, out, "local/no-valid-segment")
    assert not out.exists() and json.loads((raw / "capture.json").read_text())["status"] == "closed"
