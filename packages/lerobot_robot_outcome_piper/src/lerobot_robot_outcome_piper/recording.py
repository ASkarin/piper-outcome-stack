"""Official record_loop and Dataset storage with row-associated telemetry sidecars."""

from __future__ import annotations
import copy
import json
import uuid
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from .console import operator_message
from .raw_io import read_jsonl
from .safety import ACTION_KEYS


from .action_audit import holding_reference, validate_action


class TelemetryDataset:
    def __init__(self, dataset, robot):
        self.dataset, self.robot = dataset, robot
        self.session = uuid.uuid4().hex
        self.path = Path(dataset.root) / "telemetry" / self.session
        self.path.mkdir(parents=True, exist_ok=False)
        self._log_lock = threading.RLock()
        self.scene = getattr(robot.config, "scene", None)
        self.log = (self.path / "events.jsonl").open("x", encoding="utf-8")
        self.pending = []
        self.saved_frames = 0
        self.saved_episodes = set()
        self.attempt = 0
        self.last_sequence = None
        self.hold_references = {}
        from .timing_report import TimingValidator

        self._raw = getattr(dataset, "is_raw", False)
        self._raw_timing = TimingValidator() if self._raw else None
        self.emit(
            "session",
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            robot_config=repr(robot.config),
            scene=None if self.scene is None else asdict(self.scene),
        )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def emit(self, event, **payload):
        if self._raw:
            return self.dataset.submit_io(self._write_event_payload, event, copy.deepcopy(payload))
        return self._write_event(event, **payload)

    def _write_event_payload(self, event, payload):
        self._write_event(event, **payload)

    def _write_event(self, event, **payload):
        with self._log_lock:
            self.log.write(
                json.dumps({"event": event, "session": self.session, **payload}, allow_nan=False)
                + "\n"
            )
            self.log.flush()

    def prepare_episode(self):
        attempt = self.attempt
        return self.dataset.prepare_episode(
            self.path / "raw_depth" / f"attempt-{attempt:06d}",
            completed=lambda timing: self._write_event(
                "attempt_prepared", attempt=attempt, timing=timing
            ),
        )

    def flush(self):
        if self._raw:
            self.dataset.flush_io()
        else:
            self.log.flush()

    def check_writer(self):
        if self._raw:
            self.dataset.check_writer()

    def add_frame(self, frame):
        ingest_started, ingest_cpu = time.monotonic(), time.thread_time()
        observation = copy.deepcopy(self.robot.last_observation_telemetry)
        action = copy.deepcopy(self.robot.last_action_telemetry)
        if observation is not None and observation.get("quality") == "control_only":
            raise RuntimeError("control-only observations cannot be recorded as demonstrations")
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
        validate_action(action)
        expected = np.asarray([action["values"][k] for k in ACTION_KEYS], dtype=np.float32)
        if not np.array_equal(np.asarray(frame["action"]), expected):
            raise RuntimeError("Dataset action differs from SDK-dispatched action")
        if self._raw_timing is not None and (self.scene is not None or "feedback" in observation):
            self._raw_timing.check(observation, action)
        row = {
            "scene_id": None if self.scene is None else self.scene.scene_id,
            "episode_index": self.dataset.num_episodes,
            "frame_index": len(self.pending),
            "attempt": self.attempt,
            "observation": observation,
            "action": action,
        }
        row["raw_depth"] = {}
        depth_files = {}
        for key, raw in getattr(self.robot, "last_depth_frames", {}).items():
            metadata = observation["cameras"][key]
            relative = (
                Path("raw_depth")
                / f"attempt-{self.attempt:06d}"
                / f"frame-{len(self.pending):06d}-{key}.npz"
            )
            path = self.path / relative
            if not self._raw:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    np.savez(stream, depth=raw)
            else:
                depth_files[str(path.relative_to(self.dataset.root))] = raw
            row["raw_depth"][key] = {
                "path": str(path.relative_to(self.dataset.root)),
                "depth_scale_m": metadata["depth_scale_m"],
                "shape": list(raw.shape),
            }
        if self._raw:
            from .stage_timing import snapshot

            row["stage_timing"] = snapshot()
            self.dataset.add_frame(
                frame, depth_files=depth_files, write_event=self._write_event, row=row
            )
        else:
            self.emit("frame_pending", **row)
            self.dataset.add_frame(frame)
        self.emit(
            "frame_ingest_returned",
            episode_index=row["episode_index"],
            frame_index=row["frame_index"],
            attempt=self.attempt,
            duration_s=time.monotonic() - ingest_started,
            thread_cpu_s=time.thread_time() - ingest_cpu,
            semantics="validation, snapshot and enqueue; not write completion"
            if self._raw
            else "offline Dataset ingest",
        )
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
        self.flush()
        self.saved_frames += len(self.pending)
        self.saved_episodes.add(index)
        self.pending = []
        self.attempt += 1
        if self._raw_timing is not None:
            from .timing_report import TimingValidator

            self._raw_timing = TimingValidator()

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
        if self._raw_timing is not None:
            from .timing_report import TimingValidator

            self._raw_timing = TimingValidator()

    def complete(self):
        if self.pending:
            raise RuntimeError("uncommitted telemetry frames remain")
        from .timing_report import summarize_events

        if self._raw:
            self.dataset.flush_io()
        self.log.flush()
        events = read_jsonl(self.path / "events.jsonl")
        timing = summarize_events(events)
        if self.scene is not None and timing["missing_timing_samples"]:
            raise RuntimeError("scene recording has missing timing evidence")
        (self.path / "timing-summary.json").write_text(
            json.dumps(timing, indent=2, allow_nan=False)
        )
        outcomes = [e for e in events if e["event"] == "episode_outcome"]
        (self.path / "episode-outcomes.json").write_text(
            json.dumps(outcomes, indent=2, ensure_ascii=False)
        )
        self.emit("complete", frames=self.saved_frames)
        if self._raw:
            self.dataset.flush_io()
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
        try:
            if self._raw:
                self.dataset.close()
        finally:
            self.log.close()


def capture_features(cfg, robot, teleop_action_processor, observation_processor):
    from lerobot.scripts import lerobot_record as official

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
    if getattr(cfg, "export_depth", False) and not any(
        f.get("info", {}).get("is_depth_map") for f in features.values()
    ):
        raise ValueError("no captured depth stream to export")
    if not getattr(cfg, "export_depth", False):
        features = {
            key: feature
            for key, feature in features.items()
            if not feature.get("info", {}).get("is_depth_map")
        }
    # Metric depth is optional; captured Z16 and its scale remain in the archive.
    for feature in features.values():
        if feature.get("info", {}).get("is_depth_map"):
            feature["dtype"] = "image"
            feature["info"]["depth_unit"] = "m"
    return features


def record_with_telemetry(cfg, *, teleop_action_processor):
    from lerobot.scripts import lerobot_record as official
    from lerobot.processor import make_default_processors

    from .config import OutcomePiperXboxConfig

    if not isinstance(cfg.teleop, OutcomePiperXboxConfig):
        raise ValueError("record requires Xbox raw capture configuration")
    robot = official.make_robot_from_config(cfg.robot)
    teleop = official.make_teleoperator_from_config(cfg.teleop)
    _, action_processor, observation_processor = make_default_processors()
    features = capture_features(cfg, robot, teleop_action_processor, observation_processor)
    ds = cfg.dataset
    if ds.repo_id.split("/")[-1].startswith("eval_"):
        raise ValueError("eval_ datasets require the official rollout workflow")
    from .xbox_raw import RawFrames

    if not getattr(cfg, "raw_root", None):
        raise ValueError("Xbox record requires raw_root; convert to Dataset offline")
    dataset = RawFrames(
        cfg.raw_root,
        fps=ds.fps,
        features=features,
        robot_type=robot.name,
        resume=cfg.resume,
        conversion_options=dict(
            use_videos=ds.video,
            rgb_encoder=asdict(ds.rgb_encoder) if ds.rgb_encoder else None,
            depth_encoder=asdict(ds.depth_encoder) if ds.depth_encoder else None,
            encoder_threads=ds.encoder_threads,
        ),
        capture_context=dict(
            scene=asdict(cfg.robot.scene) if cfg.robot.scene else None,
            timing=asdict(cfg.robot.capture_timing) if cfg.robot.capture_timing else None,
            safety=json.loads(Path(cfg.robot.safety_path).read_text())
            if getattr(cfg.robot, "safety_path", None) is not None
            else None,
        ),
    )
    from .capture_gc import CaptureGC

    capture_gc = CaptureGC()
    audit = None
    success = False
    dataset_finalized = False
    interactive = False
    try:
        audit = TelemetryDataset(dataset, robot)
        if cfg.display_data:
            official.init_visualization(
                cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
            )
        robot.configure_teleoperation(
            teleop_action_processor.steps[0].control, cfg.teleop.hold_settings()
        )
        audit.emit("teleoperation_config", config=repr(cfg.teleop))
        capture_gc.prepare()  # No hardware connection or motor control yet.
        capture_gc.start()
        teleop.connect()
        if teleop.get_action()["emergency_stop"]:
            raise ValueError("release the emergency-stop button before starting a session")
        interactive = True
        from .record_control import TerminalCommands

        terminal = TerminalCommands()  # Reject unusable stdin before connecting/enable.
        events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
        robot.camera_input_poll = teleop.get_action
        robot.emergency_stop_poll = lambda: teleop.poll_emergency_stop()
        robot.connect()
        robot.enable()
        from .console import print_controls

        print_controls(cfg.teleop)
        operator_message("[设备已连接] 请松开LB，让摇杆和扳机回中，等待保持确认。")
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
        from .record_control import run_interactive_episodes

        dataset_finalized = run_interactive_episodes(
            official,
            loop_args,
            audit,
            ds,
            teleop_action_processor.steps[0].control,
            terminal=terminal,
        )
        success = True
    except BaseException as exc:
        if interactive and robot.is_connected:
            robot.request_input_fault(exc)
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
                cleanup.callback(capture_gc.stop)  # Runs after device cleanup (LIFO).
                if not dataset_finalized:
                    cleanup.callback(dataset.finalize)
                if cfg.display_data:
                    cleanup.callback(official.shutdown_visualization, cfg.display_mode)
                if teleop.is_connected:
                    cleanup.callback(teleop.disconnect)
                if robot.is_connected:
                    cleanup.callback(robot.disconnect)
            if success and audit is not None:
                if robot.latched_cause is not None:
                    raise RuntimeError(f"recording session faulted: {robot.latched_cause}")
                audit.emit("capture_runtime", **capture_gc.summary())
                audit.complete()
                dataset.finish_capture(True)
                operator_message(
                    f"[已退出] 本次已保存 {len(audit.saved_episodes)} 回合、{audit.saved_frames} 帧。",
                )
                operator_message(f"原始数据目录：{dataset.root}")
                import shlex

                operator_message(
                    "离线转换：piper-outcome-stack xbox-convert --raw-root "
                    + shlex.quote(str(dataset.root))
                    + " --output "
                    + shlex.quote(str(ds.root))
                    + " --repo-id "
                    + shlex.quote(ds.repo_id),
                )
                operator_message("程序未自动回零或失能；请先完成停放再断电。")
        except BaseException:
            success = False
            raise
        finally:
            capture_gc.stop()
            try:
                if audit is not None:
                    audit.close()
                else:
                    dataset.close()
            finally:
                if not success or not (
                    audit is not None and (audit.path / "complete.json").exists()
                ):
                    dataset.finish_capture(False)
    return dataset
