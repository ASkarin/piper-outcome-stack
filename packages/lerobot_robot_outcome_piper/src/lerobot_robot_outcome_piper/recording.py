"""Official record_loop and Dataset storage with row-associated telemetry sidecars."""

from __future__ import annotations
import copy
import math
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from .safety import ACTION_KEYS


def holding_reference(action):
    """Validate a retained command, not a fictitious dispatch for this frame."""
    command = action.get("hold_command")
    gripper = action.get("retained_gripper_command")
    values = action.get("values")
    if type(action.get("hold_id")) is not int or action["hold_id"] <= 0:
        raise RuntimeError("holding frame has no valid hold reference")
    if not isinstance(values, dict) or set(values) != set(ACTION_KEYS):
        raise RuntimeError("holding frame lacks seven effective action values")
    for item, name in ((command, "hold_move_j"), (gripper, "move_gripper_m")):
        if (
            not isinstance(item, dict)
            or item.get("name") != name
            or item.get("result") != "sdk_returned"
        ):
            raise RuntimeError("holding frame lacks a successful command reference")
        start, end = item.get("started_monotonic_s"), item.get("ended_monotonic_s")
        if (
            start is None
            or end is None
            or not all(math.isfinite(t) for t in (start, end))
            or end < start
        ):
            raise RuntimeError("holding command timestamps are invalid")
    expected = [values[k] for k in ACTION_KEYS]
    if command.get("target") != expected[:6] or gripper.get("target") != expected[6]:
        raise RuntimeError("holding values differ from retained command targets")
    if not all(math.isfinite(float(v)) for v in expected):
        raise RuntimeError("holding values are not finite")
    return {"hold_command": command, "retained_gripper_command": gripper, "values": values}


class TelemetryDataset:
    def __init__(self, dataset, robot):
        self.dataset, self.robot = dataset, robot
        self.session = uuid.uuid4().hex
        self.path = Path(dataset.root) / "telemetry" / self.session
        self.path.mkdir(parents=True, exist_ok=False)
        self.log = (self.path / "events.jsonl").open("x", encoding="utf-8")
        self.pending = []
        self.saved_frames = 0
        self.saved_episodes = set()
        self.attempt = 0
        self.last_sequence = None
        self.hold_references = {}
        self.emit(
            "session",
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            robot_config=repr(robot.config),
        )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def emit(self, event, **payload):
        self.log.write(
            json.dumps({"event": event, "session": self.session, **payload}, allow_nan=False) + "\n"
        )
        self.log.flush()

    def add_frame(self, frame):
        observation = copy.deepcopy(self.robot.last_observation_telemetry)
        action = copy.deepcopy(self.robot.last_action_telemetry)
        if observation is None or action is None:
            raise RuntimeError("recording requires complete observation and SDK dispatch telemetry")
        sequence = observation["sequence"]
        if sequence == self.last_sequence or action["observation_sequence"] != sequence:
            raise RuntimeError("telemetry does not belong to this observation/action pair")
        self.last_sequence = sequence
        if action["result"] in ("waiting", "discarded"):
            self.emit("control_wait", observation=observation, action=action)
            return
        if observation["quality"] != "checked":
            raise RuntimeError("measurement-only observation cannot enter a training episode")
        if action["result"] == "holding":
            reference = holding_reference(action)
            hold_id = action["hold_id"]
            if hold_id in self.hold_references and self.hold_references[hold_id] != reference:
                raise RuntimeError("hold reference changed while retaining the same identifier")
            if hold_id not in self.hold_references:
                self.hold_references[hold_id] = copy.deepcopy(reference)
                self.emit("hold_reference", hold_id=hold_id, reference=reference)
            frame = dict(frame)
            frame["action"] = np.asarray(
                [action["values"][k] for k in ACTION_KEYS], dtype=np.float32
            )
        elif action["result"] != "sdk_returned":
            raise RuntimeError("recording requires complete observation and SDK dispatch telemetry")
        expected = np.asarray([action["values"][k] for k in ACTION_KEYS], dtype=np.float32)
        if not np.array_equal(np.asarray(frame["action"]), expected):
            raise RuntimeError("Dataset action differs from SDK-dispatched action")
        row = {
            "episode_index": self.dataset.num_episodes,
            "frame_index": len(self.pending),
            "attempt": self.attempt,
            "observation": observation,
            "action": action,
        }
        self.emit("frame_pending", **row)
        self.dataset.add_frame(frame)
        self.pending.append(row["frame_index"])
        self.last_sequence = sequence

    def save_episode(self):
        if not self.pending:
            self.emit(
                "episode_empty", episode_index=self.dataset.num_episodes, attempt=self.attempt
            )
            self.dataset.clear_episode_buffer()
            self.attempt += 1
            return
        index = self.dataset.num_episodes
        self.dataset.save_episode()
        self.emit(
            "episode_saved", episode_index=index, attempt=self.attempt, frames=len(self.pending)
        )
        self.saved_frames += len(self.pending)
        self.saved_episodes.add(index)
        self.pending = []
        self.attempt += 1

    def discard_episode(self):
        self.emit(
            "episode_discarded",
            episode_index=self.dataset.num_episodes,
            attempt=self.attempt,
            frames=len(self.pending),
        )
        self.dataset.clear_episode_buffer()
        self.pending = []
        self.attempt += 1

    def complete(self):
        if self.pending:
            raise RuntimeError("uncommitted telemetry frames remain")
        self.emit("complete", frames=self.saved_frames)
        with (self.path / "complete.json").open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "session": self.session,
                    "status": "complete" if self.saved_frames else "empty",
                    "frames": self.saved_frames,
                    "episodes": sorted(self.saved_episodes),
                },
                stream,
            )

    def close(self):
        self.log.close()


def verify_telemetry(root, dataset):
    rows = {}
    for directory in sorted((Path(root) / "telemetry").iterdir()):
        if not (directory / "complete.json").exists():
            raise RuntimeError(f"incomplete telemetry session: {directory.name}")
        completion = json.loads((directory / "complete.json").read_text())
        if completion.get("status") not in ("complete", "empty"):
            raise RuntimeError(f"incomplete telemetry session: {directory.name}")
        events = [
            json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        if any(e["event"] == "failed" for e in events):
            raise RuntimeError(f"failed telemetry session: {directory.name}")
        references = {}
        for event in events:
            if event["event"] == "hold_reference":
                key = event["hold_id"]
                if key in references and references[key] != event["reference"]:
                    raise RuntimeError("inconsistent hold reference")
                references[key] = event["reference"]
        if completion.get("status") == "empty" and (
            completion.get("frames") != 0 or any(e["event"] == "episode_saved" for e in events)
        ):
            raise RuntimeError("empty session contains saved training frames")
        accepted = {
            (e["episode_index"], e["attempt"]) for e in events if e["event"] == "episode_saved"
        }
        for e in events:
            if e["event"] != "frame_pending" or (e["episode_index"], e["attempt"]) not in accepted:
                continue
            action = e["action"]
            if action.get("result") == "holding":
                reference = holding_reference(action)
                if references.get(action["hold_id"]) != reference:
                    raise RuntimeError("holding frame has no matching recorded reference")
            elif action.get("result") != "sdk_returned":
                raise RuntimeError("non-dispatched waiting/fault action entered the Dataset")
            key = (e["episode_index"], e["frame_index"])
            if key in rows:
                raise RuntimeError("duplicate committed telemetry row")
            rows[key] = e
    if len(rows) != dataset.num_frames:
        raise RuntimeError("telemetry and Dataset frame counts differ")
    actions = dataset.select_columns(["episode_index", "frame_index", "action"])
    for index in range(dataset.num_frames):
        frame = actions[index]
        key = (int(frame["episode_index"]), int(frame["frame_index"]))
        row = rows.get(key)
        if row is None:
            raise RuntimeError(f"missing telemetry row {key}")
        action = np.asarray([row["action"]["values"][k] for k in ACTION_KEYS], dtype=np.float32)
        if not np.array_equal(np.asarray(frame["action"]), action):
            raise RuntimeError(f"telemetry action mismatch at {key}")
    return {"episodes": dataset.num_episodes, "frames": len(rows)}


def validate_dataset_schema(dataset, robot, fps, features):
    # The pinned official resume helper still expects the retired robot_type
    # attribute. Use the current Robot.name contract without adding an alias.
    from lerobot.utils.constants import DEFAULT_FEATURES

    expected = {**features, **DEFAULT_FEATURES}
    if dataset.meta.robot_type != robot.name or dataset.fps != fps:
        raise ValueError("Dataset robot identity or fps mismatch")
    if set(dataset.features) != set(expected):
        raise ValueError("Dataset feature keys mismatch")
    for key, feature in expected.items():
        actual = dataset.features[key]
        if (
            actual["dtype"] != feature["dtype"]
            or tuple(actual["shape"]) != tuple(feature["shape"])
            or actual.get("names") != feature.get("names")
        ):
            raise ValueError(f"Dataset feature schema mismatch: {key}")


def record_with_telemetry(cfg, *, teleop_action_processor):
    from lerobot.scripts import lerobot_record as official
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_processors

    robot = official.make_robot_from_config(cfg.robot)
    teleop = official.make_teleoperator_from_config(cfg.teleop)
    _, action_processor, observation_processor = make_default_processors()
    features = official.combine_feature_dicts(
        *[
            official.aggregate_pipeline_dataset_features(
                pipeline=p,
                initial_features=official.create_initial_features(**f),
                use_videos=cfg.dataset.video,
            )
            for p, f in (
                (teleop_action_processor, {"action": robot.action_features}),
                (observation_processor, {"observation": robot.observation_features}),
            )
        ]
    )
    ds = cfg.dataset
    if ds.repo_id.split("/")[-1].startswith("eval_"):
        raise ValueError("eval_ datasets require the official rollout workflow")
    kwargs = dict(
        root=ds.root,
        batch_encoding_size=ds.video_encoding_batch_size,
        rgb_encoder=ds.rgb_encoder,
        depth_encoder=ds.depth_encoder,
        encoder_threads=ds.encoder_threads,
        streaming_encoding=ds.streaming_encoding,
        encoder_queue_maxsize=ds.encoder_queue_maxsize,
        image_writer_processes=ds.num_image_writer_processes,
        image_writer_threads=ds.num_image_writer_threads_per_camera * len(cfg.robot.cameras),
    )
    if cfg.resume:
        previous = LeRobotDataset(ds.repo_id, root=ds.root)
        verify_telemetry(previous.root, previous)
        dataset = LeRobotDataset.resume(ds.repo_id, **kwargs)
    else:
        ds.stamp_repo_id()
        dataset = LeRobotDataset.create(
            ds.repo_id,
            ds.fps,
            robot_type=robot.name,
            features=features,
            use_videos=ds.video,
            **kwargs,
        )
    audit, listener = None, None
    success = False
    try:
        validate_dataset_schema(dataset, robot, ds.fps, features)
        audit = TelemetryDataset(dataset, robot)
        if cfg.display_data:
            official.init_visualization(
                cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
            )
        from .config import OutcomePiperXboxConfig

        if isinstance(cfg.teleop, OutcomePiperXboxConfig):
            robot.configure_teleoperation(
                teleop_action_processor.steps[0].control, cfg.teleop.hold_settings()
            )
            audit.emit("teleoperation_config", config=repr(cfg.teleop))
        teleop.connect()
        if isinstance(cfg.teleop, OutcomePiperXboxConfig) and teleop.get_action()["emergency_stop"]:
            raise ValueError("release the emergency-stop button before starting a session")
        robot.connect()
        listener, events = official.init_keyboard_listener()
        loop_args = dict(
            robot=robot,
            teleop=teleop,
            events=events,
            fps=ds.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=action_processor,
            robot_observation_processor=observation_processor,
            display_data=cfg.display_data,
            display_mode=cfg.display_mode,
            display_compressed_images=cfg.display_compressed_images,
            single_task=ds.single_task,
        )
        with official.VideoEncodingManager(dataset):
            count = 0
            while count < ds.num_episodes and not events["stop_recording"]:
                official.log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                official.record_loop(**loop_args, dataset=audit, control_time_s=ds.episode_time_s)
                if events["rerecord_episode"]:
                    audit.discard_episode()
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                else:
                    audit.save_episode()
                    count += 1
                if count < ds.num_episodes and not events["stop_recording"] and ds.reset_time_s > 0:
                    official.record_loop(**loop_args, control_time_s=ds.reset_time_s)
        success = True
    except BaseException as exc:
        if audit is not None:
            try:
                audit.emit(
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                    observation=robot.last_observation_telemetry,
                    action=robot.last_action_telemetry,
                    stop_outcome=getattr(robot, "stop_outcome", None),
                )
            except Exception as log_error:
                exc.add_note(f"Telemetry failure record could not be written: {log_error}")
        raise
    finally:
        try:
            with ExitStack() as cleanup:
                cleanup.callback(dataset.finalize)
                if cfg.display_data:
                    cleanup.callback(official.shutdown_visualization, cfg.display_mode)
                if listener is not None:
                    cleanup.callback(listener.stop)
                if teleop.is_connected:
                    cleanup.callback(teleop.disconnect)
                if robot.is_connected:
                    cleanup.callback(robot.disconnect)
            if success and audit is not None:
                if robot.latched_cause is not None:
                    raise RuntimeError(f"recording session faulted: {robot.latched_cause}")
                audit.complete()
        finally:
            if audit is not None:
                audit.close()
    return dataset
