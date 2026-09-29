import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from action_evidence import dispatched
from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from lerobot_robot_outcome_piper.recording import TelemetryDataset
from lerobot_robot_outcome_piper.xbox_raw import RawFrames
from lerobot_robot_outcome_piper.xbox_recovery import recover_sealed, audit_raw


def failed_capture(tmp_path, outcome=True):
    raw = RawFrames(tmp_path / "raw", fps=20, features={}, robot_type="outcome_piper")
    robot = NS(config=NS(), last_depth_frames={})
    audit = TelemetryDataset(raw, robot)
    audit.prepare_episode().result()
    for n in range(3):
        robot.last_observation_telemetry = dict(
            sequence=n, quality="checked", observed_monotonic_s=1 + n * 0.05, cameras={}
        )
        robot.last_action_telemetry = dispatched(dict.fromkeys(ACTION_KEYS, 0.0), n)
        audit.add_frame(
            {
                "action": np.zeros(7, np.float32),
                "observation.state": np.zeros(7, np.float32),
                "task": "test",
            }
        )
    if outcome:
        audit.emit(
            "episode_save_requested",
            episode_index=0,
            attempt=0,
            position_id="P1",
            task_outcome="success",
        )
    audit.save_episode()
    # Interrupted next attempt must not be promoted.
    audit.prepare_episode().result()
    raw.add_frame({"action": np.zeros(7, np.float32), "observation.state": np.zeros(7, np.float32)})
    audit.emit("failed", error="post-seal fault")
    raw.finalize()
    raw.finish_capture(False)
    audit.close()
    return raw.root


@pytest.mark.parametrize("outcome", [True, False])
def test_recovery_preserves_failed_original_and_outcome_provenance(tmp_path, outcome):
    source = failed_capture(tmp_path, outcome)
    original = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    report = recover_sealed(source, tmp_path / "recovered", [0])
    assert report["audit"] == {"episodes": 1, "frames": 3}
    assert audit_raw(tmp_path / "recovered") == report["audit"]
    assert original == {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    events = [
        json.loads(line)
        for p in (tmp_path / "recovered/telemetry").glob("*/events.jsonl")
        for line in p.open()
    ]
    result = next(e for e in events if e["event"] == "episode_outcome")
    assert result["task_outcome"] == ("success" if outcome else "unknown")
    assert result["outcome_source"] == ("persisted_save_request" if outcome else "not_recorded")


@pytest.mark.parametrize("damage", ["command", "missing_frames", "unsealed", "active"])
def test_recovery_rejects_invalid_or_unfinished_data(tmp_path, damage):
    source = failed_capture(tmp_path)
    if damage == "command":
        log = next((source / "telemetry").glob("*/events.jsonl"))
        rows = [json.loads(line) for line in log.open()]
        next(e for e in rows if e["event"] == "frame_pending")["action"]["commands"] = []
        log.write_text("".join(json.dumps(e) + "\n" for e in rows))
    if damage == "missing_frames":
        path = next(
            p.parent
            for p in source.glob("attempts/*/result.json")
            if json.loads(p.read_text())["status"] == "saved"
        )
        (path / "frames.jsonl").write_text("")
    if damage == "active":
        (source / "capture.json").write_text('{"status":"running"}')
    with pytest.raises((RuntimeError, ValueError)):
        recover_sealed(source, tmp_path / "recovered", [1] if damage == "unsealed" else [0])
    if (tmp_path / "recovered/capture.json").exists():
        assert json.loads((tmp_path / "recovered/capture.json").read_text())["status"] == "failed"
