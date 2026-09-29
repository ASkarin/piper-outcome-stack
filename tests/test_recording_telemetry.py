"""Real official Dataset/record_loop/replay with synthetic devices only."""

from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import numpy as np
import pytest
from action_evidence import dispatched

pytest.importorskip("lerobot")
sys.path.insert(0, str(Path(__file__).parents[1] / "infra/acceptance"))
import lerobot_dataset_replay_smoke as smoke
from lerobot.scripts import lerobot_record as official
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.processor import make_default_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot_robot_outcome_piper.recording import record_with_telemetry
from lerobot_robot_outcome_piper.action_audit import verify_telemetry


class Robot(smoke.FakeRobot):
    def __init__(self):
        super().__init__()
        self.config = NS(cameras={"d435": object()})
        self.last_observation_telemetry = None
        self.last_action_telemetry = None
        self.latched_cause = None

    def enable(self):
        assert self.is_connected

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
        self.last_action_telemetry = dispatched(result, self._observation_count)
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


def offline_fixture_record(cfg, *, teleop_action_processor):
    """Exercise official storage/audit with synthetic devices, not the public record entry."""
    from lerobot_robot_outcome_piper.recording import TelemetryDataset
    from lerobot_robot_outcome_piper.capture_gc import CaptureGC

    robot = official.make_robot_from_config(cfg.robot)
    teleop = official.make_teleoperator_from_config(cfg.teleop)
    _, ap, op = make_default_processors()
    from lerobot_robot_outcome_piper.recording import capture_features

    features = capture_features(cfg, robot, teleop_action_processor, op)
    ds = cfg.dataset
    if cfg.resume:
        verify_telemetry(ds.root, LeRobotDataset(ds.repo_id, root=ds.root))
        dataset = LeRobotDataset.resume(ds.repo_id, root=ds.root)
    else:
        dataset = LeRobotDataset.create(
            ds.repo_id,
            ds.fps,
            root=ds.root,
            robot_type=robot.name,
            features=features,
            use_videos=ds.video,
        )
    audit = TelemetryDataset(dataset, robot)
    listener, events = official.init_keyboard_listener()
    guard = CaptureGC()
    guard.prepare()
    guard.start()
    try:
        robot.connect()
        teleop.connect()
        robot.enable()
        for _ in range(ds.num_episodes):
            while True:
                official.record_loop(
                    robot=robot,
                    teleop=teleop,
                    events=events,
                    fps=ds.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=ap,
                    robot_observation_processor=op,
                    dataset=audit,
                    control_time_s=ds.episode_time_s,
                    single_task=ds.single_task,
                    display_data=False,
                )
                if not events["rerecord_episode"]:
                    audit.save_episode()
                    break
                audit.discard_episode()
                events["rerecord_episode"] = events["exit_early"] = False
        dataset.finalize()
        if robot.latched_cause is not None:
            raise RuntimeError(robot.latched_cause)
        audit.complete()
    except BaseException as exc:
        audit.emit("failed", error=str(exc))
        raise
    finally:
        robot.disconnect()
        teleop.disconnect()
        listener.stop()
        guard.stop()
        audit.close()
    return dataset


def record(cfg):
    processor, _, _ = make_default_processors()
    return offline_fixture_record(cfg, teleop_action_processor=processor)


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


@pytest.mark.parametrize("failure", ["write", "finalize", "interrupt", "mismatch", "late_fault"])
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
    elif failure == "late_fault":
        original = LeRobotDataset.finalize

        def faulted_finalize(self):
            original(self)
            robot.latched_cause = "watchdog fault"

        monkeypatch.setattr(LeRobotDataset, "finalize", faulted_finalize)
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


class PauseRobot(Robot):
    """Synthetic dispatch proof; the real official loop ignores returned actions."""

    def send_action(self, action):
        result = super().send_action(action)
        phase = (self._observation_count - 1) % 5
        if phase == 0:
            self.last_action_telemetry.update(result="waiting", values=None, commands=[])
        elif phase == 1:
            self.retained = dict(result)
            self.retained["gripper.pos"] = 0.025
            # This normal frame still uses the original action, checked strictly.
            self.gripper_command = dict(
                name="move_gripper_m",
                target=0.025,
                force=1.0,
                started_monotonic_s=1.0,
                ended_monotonic_s=1.01,
                result="sdk_returned",
            )
            self.hold_command = dict(
                name="hold_move_j",
                target=list(self.retained.values())[:6],
                started_monotonic_s=2.0,
                ended_monotonic_s=2.01,
                result="sdk_returned",
            )
        elif phase in (2, 3):
            self.last_action_telemetry.update(
                result="holding",
                values=self.retained,
                hold_id=1 + (self._observation_count - 1) // 5,
                hold_confirmed=phase == 3,
                hold_command=self.hold_command,
                retained_gripper_command=self.gripper_command,
                commands=[self.hold_command] if phase == 2 else [],
            )
            return self.retained
        return result


def pause_devices(monkeypatch):
    _, teleop, events, listener = devices(monkeypatch)
    robot = PauseRobot()
    monkeypatch.setattr(official, "make_robot_from_config", lambda _: robot)
    loop = official.record_loop

    def five_frames(**kwargs):
        for _ in range(5):
            loop(**kwargs)

    monkeypatch.setattr(official, "record_loop", five_frames)
    return robot, events


def test_pause_frames_finalize_reload_replay_and_resume(tmp_path, monkeypatch):
    robot, _ = pause_devices(monkeypatch)
    cfg = configuration(tmp_path / "dataset")
    dataset = record(cfg)
    loaded = LeRobotDataset(dataset.repo_id, root=cfg.dataset.root)
    assert verify_telemetry(loaded.root, loaded) == {"episodes": 1, "frames": 4}
    for i in (1, 2):
        np.testing.assert_array_equal(
            loaded[i]["action"], np.asarray(list(robot.retained.values()), dtype=np.float32)
        )
    replayed = smoke._replay(loaded, loaded.root)
    assert len(replayed.actions) == 4
    log = next((loaded.root / "telemetry").glob("*/events.jsonl"))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len([r for r in rows if r["event"] == "control_wait"]) == 1
    holds = [
        r["action"]
        for r in rows
        if r["event"] == "frame_pending" and r["action"]["result"] == "holding"
    ]
    assert len(holds[0]["commands"]) == 1 and holds[1]["commands"] == []
    assert holds[0]["hold_command"] == holds[1]["hold_command"]
    # A new recording session can reuse local hold_id=1 without conflating references.
    monkeypatch.setattr(official, "make_robot_from_config", lambda _: PauseRobot())
    resumed = record(configuration(loaded.root, resume=True, repo_id=dataset.repo_id))
    loaded = LeRobotDataset(resumed.repo_id, root=resumed.root)
    assert verify_telemetry(loaded.root, loaded) == {"episodes": 2, "frames": 8}
    # Corrupting one retained reference must fail the audit.
    for row in rows:
        if row["event"] == "hold_reference":
            row["reference"]["hold_command"]["target"][0] += 1
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="matching recorded reference"):
        verify_telemetry(loaded.root, loaded)


def test_pause_rerecord_retains_only_replacement_attempt(tmp_path, monkeypatch):
    _, events = pause_devices(monkeypatch)
    events["rerecord_episode"] = True
    dataset = record(configuration(tmp_path / "dataset"))
    loaded = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert verify_telemetry(loaded.root, loaded) == {"episodes": 1, "frames": 4}


def test_pause_failure_is_not_a_complete_training_episode(tmp_path, monkeypatch):
    robot, _ = pause_devices(monkeypatch)
    send = robot.send_action

    def fail_after_pause(action):
        if robot._observation_count == 5:
            raise RuntimeError("hold confirmation failed")
        return send(action)

    robot.send_action = fail_after_pause
    cfg = configuration(tmp_path / "dataset")
    with pytest.raises(RuntimeError, match="hold confirmation failed"):
        record(cfg)
    assert not list((cfg.dataset.root / "telemetry").glob("*/complete.json"))
    with pytest.raises(RuntimeError, match="incomplete telemetry session"):
        verify_telemetry(cfg.dataset.root, NS(num_frames=0))


@pytest.mark.parametrize("home", [False, True])
def test_mode_metadata_does_not_change_recorded_action_schema(tmp_path, monkeypatch, home):
    robot, _, _, _ = devices(monkeypatch)
    original = robot.send_action

    def send(action):
        result = original(action)
        robot.last_action_telemetry.update(
            teleop_mode="ORIENTATION",
            orientation_target=np.eye(3).tolist(),
            pose_plan={"waypoint": 1, "waypoint_count": 2} if home else None,
            pose_event="started" if home else None,
        )
        return result

    monkeypatch.setattr(robot, "send_action", send)
    ds = record(configuration(tmp_path / "dataset"))
    assert verify_telemetry(ds.root, ds) == {"episodes": 1, "frames": 1}
    log = next((ds.root / "telemetry").glob("*/events.jsonl"))
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    row = next(x for x in rows if x["event"] == "frame_pending")
    assert row["action"]["pose_event"] == ("started" if home else None)
    assert row["action"]["pose_plan"] == ({"waypoint": 1, "waypoint_count": 2} if home else None)
    assert row["action"]["teleop_mode"] == "ORIENTATION"
    assert row["action"]["orientation_target"] == np.eye(3).tolist()
    assert tuple(ds.features["action"]["shape"]) == (7,)


def test_centered_gripper_updates_hold_reference_without_faking_joint_writes(tmp_path, monkeypatch):
    robot, _ = pause_devices(monkeypatch)
    original = robot.send_action

    def send(action):
        result = original(action)
        if (robot._observation_count - 1) % 5 == 3:
            d = robot.last_action_telemetry
            d["values"] = {**d["values"], "gripper.pos": 0.03}
            d["hold_id"] += 1
            d["control_state"] = "CENTERED"
            grip = {
                **d["retained_gripper_command"],
                "target": 0.03,
                "started_monotonic_s": 3.0,
                "ended_monotonic_s": 3.01,
            }
            d["retained_gripper_command"] = grip
            d["commands"] = [grip]
            return d["values"]
        return result

    monkeypatch.setattr(robot, "send_action", send)
    ds = record(configuration(tmp_path / "dataset"))
    loaded = LeRobotDataset(ds.repo_id, root=ds.root)
    assert verify_telemetry(ds.root, loaded) == {"episodes": 1, "frames": 4}
    np.testing.assert_array_equal(loaded[1]["action"][:6], loaded[2]["action"][:6])
    assert float(loaded[1]["action"][-1]) == pytest.approx(0.025)
    assert float(loaded[2]["action"][-1]) == pytest.approx(0.03)
    rows = [
        json.loads(s)
        for s in next((ds.root / "telemetry").glob("*/events.jsonl")).read_text().splitlines()
    ]
    refs = [x for x in rows if x["event"] == "hold_reference"]
    assert len(refs) == 2
    frames = [
        x["action"]
        for x in rows
        if x["event"] == "frame_pending" and x["action"].get("control_state") == "CENTERED"
    ]
    assert [c["name"] for c in frames[0]["commands"]] == ["move_gripper_m"]


@pytest.mark.parametrize("redo", [False, True])
def test_interactive_scene_record_finalize_reload_and_scene_resume_rejection(
    tmp_path, monkeypatch, redo
):
    from lerobot_robot_outcome_piper.teleop_control import TeleopControl
    from lerobot_robot_outcome_piper.scene import SceneContext
    from lerobot_robot_outcome_piper import record_control
    from lerobot.processor import RobotActionProcessorStep
    from test_plugin import xbox_config
    from lerobot_robot_outcome_piper.safety import ACTION_KEYS

    robot, teleop, _, _ = devices(monkeypatch)
    scene = SceneContext(
        "synthetic-new-table",
        "synthetic fixed base",
        "synthetic RGB view",
        "synthetic four positions",
    )
    robot.config.scene = scene
    c = TeleopControl()
    c.gripper_target = 0.03  # Synthetic successful gripper command.
    robot.prepare_recording_gripper = lambda epoch: {"initialized": False}

    monkeypatch.setattr(
        robot, "configure_teleoperation", lambda control, settings: None, raising=False
    )
    monkeypatch.setattr(
        robot,
        "request_input_fault",
        lambda error: setattr(robot, "latched_cause", str(error)),
        raising=False,
    )
    original_obs = robot.get_observation
    seen_phases = []

    def obs():
        c.confirm_hold()
        seen_phases.append(c.recording_phase)
        result = original_obs()
        now = robot._observation_count * 0.04
        robot.last_observation_telemetry.update(
            feedback={"received_monotonic_s": [now - 0.003] * 5},
            cameras={
                "d435": dict(
                    frame_number=robot._observation_count,
                    device_timestamp_ms=now * 1000,
                    timestamp_domain="synthetic",
                    received_monotonic_s=now - 0.005,
                )
            },
        )
        return result

    monkeypatch.setattr(robot, "get_observation", obs)
    original_action = teleop.get_action
    monkeypatch.setattr(
        teleop,
        "get_action",
        lambda: {**original_action(), "emergency_stop": False, "hold": False, "neutral": True},
    )

    class StripControl(RobotActionProcessorStep):
        def action(self, action):
            return {key: action[key] for key in ACTION_KEYS}

        def transform_features(self, features):
            return features

    pipeline, _, _ = make_default_processors()
    pipeline.steps[0].control = c
    pipeline.steps.append(StripControl())

    class Commands:
        def __init__(self):
            self.sent = set()
            self.count = 0
            self.redid = False

        def poll(self):
            self.count += 1
            assert self.count < 500, "interactive phase stalled"
            phase = c.recording_phase
            if phase == "preparing" and "start" not in self.sent:
                return "start p1"  # Retry explicitly if the first request preceded preparation.
            if phase == "recording":
                self.sent.add("start")
                return "end"
            if phase == "review" and redo and not self.redid:
                self.redid = True
                self.sent.remove("start")
                return "redo"
            if phase == "review" and "save" not in self.sent:
                self.sent.add("save")
                return "save failure missed grasp"
            return None

    monkeypatch.setattr(record_control, "TerminalCommands", Commands)
    cfg = configuration(tmp_path / "interactive-data")
    cfg.robot.scene = scene
    cfg.robot.capture_timing = None
    cfg.raw_root = str(tmp_path / "raw")
    cfg.teleop = xbox_config(control_hz=30)
    raw = record_with_telemetry(cfg, teleop_action_processor=pipeline)
    from lerobot_robot_outcome_piper.xbox_raw import convert

    convert(raw.root, cfg.dataset.root, cfg.dataset.repo_id)
    ds = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
    loaded = ds
    assert verify_telemetry(ds.root, loaded) == {"episodes": 1, "frames": 1}
    assert {"preparing", "recording", "review", "saving", "finalizing"} <= set(seen_phases)
    log = next((ds.root / "telemetry").glob("*/events.jsonl"))
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["scene_id"] for e in events if e["event"] == "frame_pending"] == [scene.scene_id] * (
        2 if redo else 1
    )
    outcome = next(e for e in events if e["event"] == "episode_outcome")
    assert outcome["task_outcome"] == "failure" and outcome["data_valid"] is True
    summary = json.loads(log.with_name("timing-summary.json").read_text())
    assert summary["all_samples"]["samples"] == 1
    replayed = smoke._replay(loaded, loaded.root)
    np.testing.assert_array_equal(list(replayed.actions[0].values()), loaded[0]["action"])
    cfg.resume = True
    cfg.dataset.repo_id = ds.repo_id
    cfg.robot.scene = SceneContext("another-scene", "base", "view", "area")
    with pytest.raises(ValueError, match="resume schema or scene differs"):
        record_with_telemetry(cfg, teleop_action_processor=pipeline)

    cfg.robot.scene = scene
    robot.config.scene = cfg.robot.scene
    resumed = record_with_telemetry(cfg, teleop_action_processor=pipeline)
    next_output = tmp_path / "converted-resume"
    convert(resumed.root, next_output, cfg.dataset.repo_id)
    reloaded = LeRobotDataset(cfg.dataset.repo_id, root=next_output)
    assert verify_telemetry(next_output, reloaded) == {"episodes": 2, "frames": 2}


def test_motion_record_gc_is_deferred_until_device_cleanup(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper import capture_gc
    from test_capture_gc import Collector

    collector = Collector()
    guard = capture_gc.CaptureGC(collector)
    monkeypatch.setattr(capture_gc, "CaptureGC", lambda: guard)
    robot, _, _, _ = devices(monkeypatch)
    original_send = robot.send_action
    original_disconnect = robot.disconnect

    def send(action):
        assert not collector.enabled
        return original_send(action)

    def disconnect():
        assert not collector.enabled
        return original_disconnect()

    robot.send_action = send
    robot.disconnect = disconnect
    dataset = record(configuration(tmp_path / "dataset"))
    assert collector.enabled and guard.summary()["restored"]
    assert verify_telemetry(dataset.root, dataset)["frames"] == 1
