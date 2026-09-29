import json
from pathlib import Path
import pytest
from piper_outcome_stack.training_selection import require_successful_selection


def test_success_selection_is_explicit_and_whole_episode(tmp_path):
    log = tmp_path / "telemetry" / "session"
    log.mkdir(parents=True)
    events = []
    for i, outcome in enumerate(["success", "failure", "cancelled", "unknown"]):
        events.extend(
            [
                dict(event="episode_saved", episode_index=i, attempt=i),
                dict(
                    event="episode_outcome",
                    episode_index=i,
                    attempt=i,
                    task_outcome=outcome,
                    data_valid=True,
                ),
            ]
        )
    (log / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    require_successful_selection(tmp_path, [0])
    for values in [None, [], [0, 0], [1], [2], [3], [4]]:
        with pytest.raises(ValueError):
            require_successful_selection(tmp_path, values)


def test_raw_conversion_generates_reviewable_success_split(tmp_path):
    import numpy as np
    from types import SimpleNamespace as NS
    from action_evidence import dispatched
    from lerobot_robot_outcome_piper.safety import ACTION_KEYS
    from lerobot_robot_outcome_piper.recording import TelemetryDataset
    from lerobot_robot_outcome_piper.xbox_raw import RawFrames, convert
    from piper_outcome_stack.training_selection import prepare

    features = {
        key: dict(dtype="float32", shape=(7,), names=list(ACTION_KEYS))
        for key in ("action", "observation.state")
    }
    raw = RawFrames(
        tmp_path / "raw",
        fps=20,
        features=features,
        robot_type="outcome_piper",
        conversion_options={"use_videos": False},
    )
    robot = NS(config=NS(), last_depth_frames={})
    audit = TelemetryDataset(raw, robot)
    for episode, outcome in enumerate(["success", "success", "failure"]):
        audit.prepare_episode().result()
        for n in range(2):
            sequence = episode * 2 + n
            robot.last_observation_telemetry = dict(
                sequence=sequence,
                quality="checked",
                observed_monotonic_s=1 + sequence * 0.05,
                cameras={},
            )
            robot.last_action_telemetry = dispatched(dict.fromkeys(ACTION_KEYS, 0.0), sequence)
            audit.add_frame(
                {
                    "action": np.zeros(7, np.float32),
                    "observation.state": np.zeros(7, np.float32),
                    "task": "test",
                }
            )
        audit.save_episode()
        audit.emit(
            "episode_outcome",
            episode_index=episode,
            attempt=episode,
            position_id="P1",
            task_outcome=outcome,
            data_valid=True,
        )
    raw.finalize()
    audit.complete()
    audit.close()
    raw.finish_capture(True)
    dataset = tmp_path / "dataset"
    result = convert(raw.root, dataset, "local/selection-test")
    assert result["frames"] == 6
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "dataset": {"root": str(dataset), "repo_id": "local/selection-test"},
                "policy": {"type": "act"},
            }
        )
    )
    report = prepare(config, tmp_path / "selected.json", [1])
    assert report["train_episodes"] == [0]
    assert report["validation_episodes"] == [1]
    manifest = json.loads(Path(report["manifest"]).read_text())
    assert manifest["excluded_episodes"] == [2]
    selected = json.loads((tmp_path / "selected.json").read_text())
    assert selected["dataset"]["episodes"] == [0]
    with pytest.raises(FileExistsError):
        prepare(config, tmp_path / "selected.json", [1])
