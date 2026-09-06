"""Real official Dataset/record_loop/replay with synthetic devices only."""

from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import numpy as np
import pytest

pytest.importorskip("lerobot")
sys.path.insert(0, str(Path(__file__).parents[1] / "infra/acceptance"))
import lerobot_dataset_replay_smoke as smoke
from lerobot.scripts import lerobot_record as official
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.processor import make_default_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot_robot_outcome_piper.recording import record_with_telemetry, verify_telemetry


class Robot(smoke.FakeRobot):
    def __init__(self):
        super().__init__()
        self.config = NS(cameras={"d435": object()})
        self.last_observation_telemetry = None
        self.last_action_telemetry = None

    def get_observation(self):
        result = super().get_observation()
        self.last_observation_telemetry = {
            "sequence": self._observation_count,
            "quality": "checked",
            "observed_monotonic_s": self._observation_count * 0.04,
            "cameras": {"d435": {"frame_number": self._observation_count}},
        }
        return result

    def send_action(self, action):
        result = super().send_action(action)
        self.last_action_telemetry = {
            "observation_sequence": self._observation_count,
            "result": "sdk_returned",
            "values": dict(result),
        }
        return result


def configuration(root, *, resume=False, repo_id="local/piper-telemetry-smoke"):
    return NS(
        robot=NS(cameras={"d435": object()}),
        teleop=NS(),
        dataset=DatasetRecordConfig(
            repo_id=repo_id,
            root=root,
            single_task="synthetic timing acceptance",
            fps=30,
            episode_time_s=0.001,
            reset_time_s=0,
            num_episodes=1,
            video=False,
            push_to_hub=False,
        ),
        resume=resume,
        display_data=False,
        display_mode="rerun",
        display_compressed_images=False,
        play_sounds=False,
    )


def devices(monkeypatch):
    robot, teleop = Robot(), smoke.FakeTeleoperator()
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    listener = smoke.FakeKeyboardListener()
    monkeypatch.setattr(official, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(official, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(official, "init_keyboard_listener", lambda: (listener, events))
    return robot, teleop, events, listener


def record(cfg):
    processor, _, _ = make_default_processors()
    return record_with_telemetry(cfg, teleop_action_processor=processor)


def test_official_record_finalize_reload_replay_and_resume_with_sidecars(tmp_path, monkeypatch):
    robot, teleop, _, listener = devices(monkeypatch)
    cfg = configuration(tmp_path / "dataset")
    dataset = record(cfg)
    reloaded = smoke._reload(dataset, cfg.dataset.root)
    assert verify_telemetry(reloaded.root, reloaded) == {"episodes": 1, "frames": 1}
    features = dataset_to_policy_features(reloaded.features)
    assert set(features) == {"observation.state", "observation.images.d435", "action"}
    assert tuple(features["observation.state"].shape) == (7,)
    replayed = smoke._replay(reloaded, reloaded.root)
    np.testing.assert_array_equal(list(replayed.actions[0].values()), reloaded[0]["action"])
    assert not robot.is_connected and not teleop.is_connected and listener.stop_count == 1
    devices(monkeypatch)
    resumed = record(configuration(reloaded.root, resume=True, repo_id=dataset.repo_id))
    resumed = LeRobotDataset(resumed.repo_id, root=resumed.root)
    assert verify_telemetry(resumed.root, resumed) == {"episodes": 2, "frames": 2}


def test_rerecord_preserves_discarded_attempt_and_commits_only_replacement(tmp_path, monkeypatch):
    _, _, events, _ = devices(monkeypatch)
    events["rerecord_episode"] = True
    dataset = record(configuration(tmp_path / "dataset"))
    loaded = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert verify_telemetry(loaded.root, loaded)["frames"] == 1
    log = next((loaded.root / "telemetry").glob("*/events.jsonl"))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["attempt"] for r in rows if r["event"] == "frame_pending"] == [0, 1]
    assert len([r for r in rows if r["event"] == "episode_discarded"]) == 1


@pytest.mark.parametrize("failure", ["write", "finalize", "interrupt", "mismatch"])
def test_incomplete_recording_is_preserved_and_rejected(tmp_path, monkeypatch, failure):
    robot, teleop, _, listener = devices(monkeypatch)
    cfg = configuration(tmp_path / "dataset")
    if failure == "write":
        monkeypatch.setattr(
            LeRobotDataset,
            "add_frame",
            lambda *args: (_ for _ in ()).throw(OSError("write failed")),
        )
    elif failure == "finalize":
        original = LeRobotDataset.finalize

        def failed_finalize(self):
            original(self)
            raise OSError("finalize failed")

        monkeypatch.setattr(LeRobotDataset, "finalize", failed_finalize)
    elif failure == "interrupt":
        monkeypatch.setattr(
            robot, "send_action", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
    else:
        original = robot.send_action

        def mismatch(action):
            result = original(action)
            robot.last_action_telemetry["values"]["gripper.pos"] += 1
            return result

        monkeypatch.setattr(robot, "send_action", mismatch)
    with pytest.raises((OSError, RuntimeError, KeyboardInterrupt)):
        record(cfg)
    assert not robot.is_connected and not teleop.is_connected and listener.stop_count == 1
    assert list((cfg.dataset.root / "telemetry").glob("*/events.jsonl"))
    assert not list((cfg.dataset.root / "telemetry").glob("*/complete.json"))
    with pytest.raises(RuntimeError, match="incomplete telemetry"):
        verify_telemetry(cfg.dataset.root, NS(num_frames=0))
