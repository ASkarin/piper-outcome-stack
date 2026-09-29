from action_evidence import dispatched
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lerobot_robot_outcome_piper.timing_report import TimingValidator
from lerobot_robot_outcome_piper.xbox_raw import RawFrames
from lerobot_robot_outcome_piper.recording import TelemetryDataset
from lerobot_robot_outcome_piper.scene import SceneContext
from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from test_timing_report import observation


def test_validator_keeps_bounded_history_and_rejects_duplicate_before_save():
    validator = TimingValidator()
    for i in range(2000):
        validator.check(observation(i))
    assert len(validator._sample.observation_times) == 1
    assert not validator._sample.metrics and not validator._sample.stream_times
    changed = observation(2000)
    changed["cameras"]["d435"]["frame_number"] = 3998
    with pytest.raises(ValueError, match="timing anomaly"):
        validator.check(changed)


@pytest.mark.parametrize(
    "change", ["domain", "device_time", "host_time", "observation_time", "missing_feedback"]
)
def test_validator_retains_clock_checks(change):
    validator = TimingValidator()
    validator.check(observation(1))
    row = observation(2)
    if change == "domain":
        row["cameras"]["d435"]["timestamp_domain"] = "changed"
    if change == "device_time":
        row["cameras"]["d435"]["device_timestamp_ms"] = 0
    if change == "host_time":
        row["cameras"]["d435"]["received_monotonic_s"] = 0
    if change == "observation_time":
        row = observation(1)
    if change == "missing_feedback":
        row.pop("feedback")
    with pytest.raises(ValueError):
        validator.check(row)


def test_seal_does_not_read_history_and_invalid_frame_never_seals(tmp_path, monkeypatch):
    sink = RawFrames(tmp_path / "raw", fps=20, features={}, robot_type="outcome_piper")
    robot = SimpleNamespace(
        config=SimpleNamespace(scene=SceneContext("synthetic", "base", "view", "area")),
        last_depth_frames={},
    )
    audit = TelemetryDataset(sink, robot)
    audit.prepare_episode().result()

    def append(i):
        robot.last_observation_telemetry = {**observation(i), "sequence": i, "quality": "checked"}
        robot.last_action_telemetry = dispatched(dict.fromkeys(ACTION_KEYS, 0.0), i, 1 + i * 0.05)
        audit.add_frame(
            {
                "action": np.zeros(7, np.float32),
                "observation.state": np.zeros(7, np.float32),
                "task": "test",
            }
        )

    append(0)
    append(1)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("seal reread historical log"))
        audit.save_episode()
    assert sink.num_episodes == 1
    audit.prepare_episode().result()
    append(0)
    robot.last_observation_telemetry = {**observation(1), "sequence": 1, "quality": "checked"}
    robot.last_observation_telemetry["cameras"]["d435"]["frame_number"] = 0
    robot.last_action_telemetry = dispatched(dict.fromkeys(ACTION_KEYS, 0.0), 1, 1.05)
    with pytest.raises(ValueError, match="timing anomaly"):
        audit.add_frame({"action": np.zeros(7, np.float32)})
    assert sink.num_episodes == 1 and sink.frames == 1
    sink.finalize()
    audit.close()
    sink.finish_capture(False)
    assert (
        len(
            [
                p
                for p in sink.root.glob("attempts/*/result.json")
                if json.loads(p.read_text())["status"] == "saved"
            ]
        )
        == 1
    )
