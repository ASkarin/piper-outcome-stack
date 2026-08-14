"""Verify official LeRobot record/finalize/reload/replay with fake devices."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import Robot, RobotConfig
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

REPO_ID = "local/piper-dataset-replay-smoke"
ACTION_NAMES = (
    *[f"joint_{index}.pos" for index in range(1, 7)],
    "gripper.pos",
)


@dataclass(kw_only=True)
class FakeRobotConfig(RobotConfig):
    """Configuration token consumed by patched official factories."""


@dataclass(kw_only=True)
class FakeTeleoperatorConfig(TeleoperatorConfig):
    """Configuration token consumed by the patched official factory."""


class FakeRobot(Robot):
    """Minimal fake with the exact Outcome PiPER public schema."""

    config_class = FakeRobotConfig
    name = "outcome_piper"

    def __init__(self) -> None:
        self.cameras = {}
        self._connected = False
        self._observation_count = 0
        self.connect_count = 0
        self.disconnect_count = 0
        self.actions: list[dict[str, float]] = []

    @property
    def observation_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_NAMES, float)

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_NAMES, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected:
            raise RuntimeError("fake robot was connected twice")
        self._connected = True
        self.connect_count += 1

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("fake robot is disconnected")
        offset = self._observation_count * 0.001
        self._observation_count += 1
        return {
            name: float(value + offset)
            for name, value in zip(
                ACTION_NAMES,
                np.linspace(0.0, 0.06, 7, dtype=np.float32),
                strict=True,
            )
        }

    def send_action(self, action: dict[str, object]) -> dict[str, object]:
        if not self._connected:
            raise RuntimeError("fake robot is disconnected")
        if tuple(action) != ACTION_NAMES:
            raise RuntimeError(f"unexpected action fields: {tuple(action)}")
        self.actions.append({name: float(action[name]) for name in ACTION_NAMES})
        return action

    def disconnect(self) -> None:
        if not self._connected:
            raise RuntimeError("fake robot was disconnected twice")
        self._connected = False
        self.disconnect_count += 1


class FakeTeleoperator(Teleoperator):
    """Return one deterministic seven-field action to the official recorder."""

    config_class = FakeTeleoperatorConfig
    name = "outcome_piper_fake_teleoperator"

    def __init__(self) -> None:
        self._connected = False
        self.action_count = 0
        self.connect_count = 0
        self.disconnect_count = 0

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_NAMES, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected:
            raise RuntimeError("fake teleoperator was connected twice")
        self._connected = True
        self.connect_count += 1

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("fake teleoperator is disconnected")
        offset = self.action_count * 0.001 + 0.002
        self.action_count += 1
        return {
            name: float(value + offset)
            for name, value in zip(
                ACTION_NAMES,
                np.linspace(0.0, 0.06, 7, dtype=np.float32),
                strict=True,
            )
        }

    def send_feedback(self, feedback: dict[str, object]) -> None:
        del feedback

    def disconnect(self) -> None:
        if not self._connected:
            raise RuntimeError("fake teleoperator was disconnected twice")
        self._connected = False
        self.disconnect_count += 1


class FakeKeyboardListener:
    def __init__(self) -> None:
        self.stop_count = 0

    def stop(self) -> None:
        self.stop_count += 1


def _record(root: Path) -> tuple[LeRobotDataset, FakeRobot, FakeTeleoperator]:
    fake_robot = FakeRobot()
    fake_teleoperator = FakeTeleoperator()
    listener = FakeKeyboardListener()
    events = {
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
    }
    config = RecordConfig(
        robot=FakeRobotConfig(),
        teleop=FakeTeleoperatorConfig(),
        dataset=DatasetRecordConfig(
            repo_id=REPO_ID,
            single_task="synthetic PiPER Dataset replay smoke",
            root=root,
            fps=30,
            episode_time_s=0.001,
            reset_time_s=0,
            num_episodes=1,
            video=False,
            push_to_hub=False,
        ),
        play_sounds=False,
    )
    with (
        patch("lerobot.scripts.lerobot_record.make_robot_from_config", return_value=fake_robot),
        patch(
            "lerobot.scripts.lerobot_record.make_teleoperator_from_config",
            return_value=fake_teleoperator,
        ),
        patch(
            "lerobot.scripts.lerobot_record.init_keyboard_listener",
            return_value=(listener, events),
        ),
    ):
        dataset = record(config)

    if fake_robot.is_connected or fake_teleoperator.is_connected:
        raise RuntimeError("official record did not disconnect both fake devices")
    if (fake_robot.connect_count, fake_robot.disconnect_count) != (1, 1):
        raise RuntimeError("official record used an unexpected fake robot connection lifecycle")
    if (fake_teleoperator.connect_count, fake_teleoperator.disconnect_count) != (1, 1):
        raise RuntimeError(
            "official record used an unexpected fake teleoperator connection lifecycle"
        )
    if listener.stop_count != 1:
        raise RuntimeError("official record did not stop its keyboard listener exactly once")
    if dataset.num_episodes != 1 or dataset.num_frames != 1:
        raise RuntimeError(
            "official record did not finalize one complete one-frame episode: "
            f"episodes={dataset.num_episodes}, frames={dataset.num_frames}"
        )
    if len(fake_robot.actions) != dataset.num_frames:
        raise RuntimeError("recorded frame count does not match actions sent to the fake robot")
    return dataset, fake_robot, fake_teleoperator


def _reload(dataset: LeRobotDataset, root: Path) -> LeRobotDataset:
    reloaded = LeRobotDataset(repo_id=dataset.repo_id, root=root)
    if reloaded.num_episodes != 1 or reloaded.num_frames != 1:
        raise RuntimeError(
            "finalized Dataset did not reload as one complete episode: "
            f"episodes={reloaded.num_episodes}, frames={reloaded.num_frames}"
        )
    if reloaded.meta.robot_type != "outcome_piper":
        raise RuntimeError(f"unexpected reloaded robot type: {reloaded.meta.robot_type}")
    for feature in ("observation.state", "action"):
        schema = reloaded.features[feature]
        if tuple(schema["shape"]) != (7,) or tuple(schema["names"]) != ACTION_NAMES:
            raise RuntimeError(f"unexpected {feature} schema: {schema}")
    np.testing.assert_allclose(
        reloaded[0]["action"].numpy(),
        [dataset[0]["action"][index] for index in range(7)],
        rtol=0.0,
        atol=1e-7,
    )
    return reloaded


def _replay(dataset: LeRobotDataset, root: Path) -> FakeRobot:
    fake_robot = FakeRobot()
    config = ReplayConfig(
        robot=FakeRobotConfig(),
        dataset=DatasetReplayConfig(repo_id=dataset.repo_id, episode=0, root=root),
        play_sounds=False,
    )
    with (
        patch(
            "lerobot.scripts.lerobot_replay.make_robot_from_config",
            return_value=fake_robot,
        ),
        patch("lerobot.scripts.lerobot_replay.precise_sleep", return_value=None),
    ):
        replay(config)

    if fake_robot.is_connected:
        raise RuntimeError("official replay did not disconnect the fake robot")
    if (fake_robot.connect_count, fake_robot.disconnect_count) != (1, 1):
        raise RuntimeError("official replay used an unexpected connection lifecycle")
    if len(fake_robot.actions) != dataset.num_frames:
        raise RuntimeError(
            "official replay did not send every Dataset frame: "
            f"expected={dataset.num_frames}, actual={len(fake_robot.actions)}"
        )
    np.testing.assert_allclose(
        [fake_robot.actions[0][name] for name in ACTION_NAMES],
        dataset[0]["action"].numpy(),
        rtol=0.0,
        atol=1e-7,
    )
    return fake_robot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise FileExistsError(f"refusing to overwrite acceptance directory: {work_dir}")
    work_dir.mkdir(parents=True)

    dataset_root = work_dir / "dataset"
    recorded, record_robot, teleoperator = _record(dataset_root)
    reloaded = _reload(recorded, dataset_root)
    replay_robot = _replay(reloaded, dataset_root)

    summary = {
        "schema_version": "piper-lerobot-dataset-replay-smoke-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed",
        "scope": {
            "synthetic_data_only": True,
            "real_robot_hardware_verified": False,
            "official_lerobot_record": True,
            "official_lerobot_dataset_api": True,
            "official_lerobot_replay": True,
        },
        "dataset": {
            "repo_id": reloaded.repo_id,
            "root": str(dataset_root),
            "robot_type": reloaded.meta.robot_type,
            "episodes": reloaded.num_episodes,
            "frames": reloaded.num_frames,
            "observation_state_names": list(ACTION_NAMES),
            "action_names": list(ACTION_NAMES),
            "recorded_finalized_and_reloaded": True,
        },
        "record": {
            "actions_sent_to_fake_robot": len(record_robot.actions),
            "teleoperator_actions": teleoperator.action_count,
        },
        "replay": {
            "episode": 0,
            "actions_sent_to_fake_robot": len(replay_robot.actions),
        },
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
