"""Slow/failing storage must not occupy the control caller or publish false saves."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
import json
import os
import threading
import numpy as np
import pytest
from action_evidence import dispatched
from lerobot_robot_outcome_piper.recording import TelemetryDataset
from lerobot_robot_outcome_piper.xbox_raw import RawFrames, convert
from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from lerobot_robot_outcome_piper.record_control import EpisodeControls
from lerobot_robot_outcome_piper.teleop_control import TeleopControl
from test_record_control import Terminal


def setup(tmp_path):
    features = {
        k: dict(dtype="float32", shape=(7,), names=list(ACTION_KEYS))
        for k in ("action", "observation.state")
    }
    features["observation.images.rgb"] = dict(
        dtype="image", shape=(16, 16, 3), names=["h", "w", "c"]
    )
    raw = RawFrames(
        tmp_path / "raw",
        fps=50,
        features=features,
        robot_type="outcome_piper",
        conversion_options={"use_videos": False},
    )
    robot = NS(config=NS(), last_depth_frames={})
    audit = TelemetryDataset(raw, robot)
    return raw, robot, audit


def frame(robot, index):
    robot.last_observation_telemetry = dict(
        sequence=index,
        quality="checked",
        observed_monotonic_s=1 + index * 0.02,
        cameras={"depth": {"depth_scale_m": 0.001}},
    )
    robot.last_action_telemetry = dispatched(dict.fromkeys(ACTION_KEYS, index * 0.001), index)
    robot.last_depth_frames = {"depth": np.full((16, 16), index, np.uint16)}
    return {
        "action": np.full(7, index * 0.001, np.float32),
        "observation.state": np.zeros(7, np.float32),
        "observation.images.rgb": np.full((16, 16, 3), index, np.uint8),
        "task": "synthetic",
    }


def close(raw, audit, success=True):
    raw.finalize()
    if success:
        audit.complete()
    audit.close()
    raw.finish_capture(success)


def test_prepare_and_ingest_have_no_control_thread_file_io_and_freeze_pixels(tmp_path, monkeypatch):
    raw, robot, audit = setup(tmp_path)
    main = threading.get_ident()
    seen = []

    def guard(fn, name):
        def run(*args, **kwargs):
            assert threading.get_ident() != main, name + " ran on control caller"
            seen.append(name)
            return fn(*args, **kwargs)

        return run

    with monkeypatch.context() as patch:
        for owner, name in [
            (Path, "mkdir"),
            (Path, "open"),
            (os, "fsync"),
            (json, "dumps"),
            (np, "savez"),
            (np, "save"),
        ]:
            patch.setattr(owner, name, guard(getattr(owner, name), name))
        audit.prepare_episode().result()
        value = frame(robot, 1)
        audit.add_frame(value)
        value["observation.images.rgb"][:] = 99
        robot.last_depth_frames["depth"][:] = 99
        audit.flush()
    assert {"mkdir", "open", "fsync", "dumps", "savez", "save"} <= set(seen)
    audit.save_episode()
    close(raw, audit)
    image = next(raw.root.glob("attempts/*/*.npy"))
    assert np.all(np.load(image) == 1)
    depth = next(raw.root.glob("telemetry/*/raw_depth/*/*.npz"))
    with np.load(depth) as data:
        assert np.all(data["depth"] == 1)


def test_blocked_writer_rejects_ninth_frame_without_waiting_or_false_save(tmp_path):
    raw, robot, audit = setup(tmp_path)
    audit.prepare_episode().result()
    entered = threading.Event()
    release = threading.Event()

    def block():
        entered.set()
        assert release.wait(5)

    raw.submit_io(block)
    assert entered.wait(2)
    try:
        for _ in range(8):
            raw.add_frame({"action": np.zeros(7)})
        with pytest.raises(RuntimeError, match="queue full"):
            raw.add_frame({"action": np.zeros(7)})
        assert raw.frames == 8 and not release.is_set()
    finally:
        release.set()
    with pytest.raises(RuntimeError):
        raw.save_episode()
    with pytest.raises(RuntimeError):
        raw.finalize()
    with pytest.raises(RuntimeError):
        audit.close()
    raw.finish_capture(False)
    assert all(
        json.loads(p.read_text())["status"] != "saved"
        for p in raw.root.glob("attempts/*/result.json")
    )
    assert len(next(raw.root.glob("attempts/*/frames.jsonl")).read_text().splitlines()) == 8


def test_phase_events_are_bounded_too(tmp_path):
    raw, _, audit = setup(tmp_path)
    raw.flush_io()
    release = threading.Event()
    entered = threading.Event()

    def block():
        entered.set()
        assert release.wait(5)

    raw.submit_io(block)
    assert entered.wait(2)
    try:
        for _ in range(31):
            audit.emit("phase_probe")
        with pytest.raises(RuntimeError, match="queue full"):
            audit.emit("phase_probe")
    finally:
        release.set()
    with pytest.raises(RuntimeError):
        raw.finalize()
    with pytest.raises(RuntimeError):
        audit.close()
    raw.finish_capture(False)


def test_background_depth_failure_is_sticky_and_never_saved(tmp_path, monkeypatch):
    raw, robot, audit = setup(tmp_path)
    audit.prepare_episode().result()
    monkeypatch.setattr(
        np, "savez", lambda *a, **k: (_ for _ in ()).throw(OSError("depth disk full"))
    )
    audit.add_frame(frame(robot, 1))
    with pytest.raises(OSError, match="depth disk full"):
        raw.flush_io()
    with pytest.raises(OSError):
        audit.save_episode()
    with pytest.raises(OSError):
        raw.finalize()
    with pytest.raises(OSError):
        audit.close()
    raw.finish_capture(False)
    assert not list(raw.root.glob("telemetry/*/complete.json"))
    assert json.loads((raw.root / "capture.json").read_text())["status"] == "failed"


def test_two_episode_roundtrip_redo_and_empty_quit(tmp_path):
    raw, robot, audit = setup(tmp_path)
    audit.prepare_episode().result()
    audit.add_frame(frame(robot, 0))
    audit.discard_episode()
    for index in [1, 2]:
        audit.prepare_episode().result()
        assert audit.prepare_episode().done()  # Rejected start reuses the empty prepared attempt.
        audit.add_frame(frame(robot, index))
        audit.save_episode()
    audit.prepare_episode().result()  # quit before start: no extra saved episode.
    close(raw, audit)
    report = convert(raw.root, tmp_path / "dataset", "local/async-test")
    assert report["frames"] == report["episodes"] == 2
    events = [
        json.loads(line) for p in raw.root.glob("telemetry/*/events.jsonl") for line in p.open()
    ]
    written = [e for e in events if e["event"] == "frame_written"]
    assert len(written) == 3 and all(
        e["write_wall_s"] >= 0 and e["queue_wait_s"] >= 0 for e in written
    )
    assert (
        sum(
            json.loads(p.read_text())["status"] == "saved"
            for p in raw.root.glob("attempts/*/result.json")
        )
        == 2
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/async-test", root=tmp_path / "dataset")
    assert float(ds[1]["action"][0]) == pytest.approx(0.002)


def test_start_during_preparation_is_rejected_not_queued_and_quit_remains_available():
    c = TeleopControl()
    ui = EpisodeControls(c, {}, Terminal(["start P1", "quit"]), lambda *a, **k: None)
    ui.enter("preparing")
    c.confirm_hold()
    ui.preparation = Future()
    ui.poll()
    assert ui.request is None
    ui.poll()
    assert ui.request == ("quit",)


def test_save_waits_for_payload_before_saved_marker(tmp_path):
    raw, robot, audit = setup(tmp_path)
    audit.prepare_episode().result()
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()
    errors = []

    def block():
        entered.set()
        assert release.wait(5)

    raw.submit_io(block)
    assert entered.wait(2)
    audit.add_frame(frame(robot, 1))
    path = raw.path

    def save():
        try:
            audit.save_episode()
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=save)
    thread.start()
    try:
        assert not done.wait(0.05)
        assert json.loads((path / "result.json").read_text())["status"] == "recording"
    finally:
        release.set()
        thread.join(5)
    assert done.is_set() and not errors
    assert json.loads((path / "result.json").read_text())["status"] == "saved"
    close(raw, audit)


@pytest.mark.parametrize("failed", [False, True])
def test_b_preempts_pending_or_failed_preparation(failed):
    from lerobot_robot_outcome_piper.record_control import run_interactive_episodes

    future = Future()
    if failed:
        future.set_exception(OSError("prepare failed"))
    c = TeleopControl()

    def forbidden():
        pytest.fail("B must not wait for storage or terminal")

    def processor(value):
        assert value[0]["emergency_stop"]
        raise RuntimeError("B handled")

    def loop(**kwargs):
        kwargs["teleop_action_processor"](({"emergency_stop": True}, {}))

    audit = NS(prepare_episode=lambda: future, check_writer=forbidden, emit=lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="B handled"):
        run_interactive_episodes(
            NS(record_loop=loop),
            dict(events={}, teleop_action_processor=processor),
            audit,
            NS(num_episodes=1),
            c,
            NS(poll=forbidden),
        )


def test_stage_timing_is_bounded_and_preserves_wall_cpu_distinction():
    import time
    from lerobot_robot_outcome_piper.stage_timing import reset, span, snapshot

    reset()
    for _ in range(2):
        with span("controlled_wait"):
            time.sleep(0.01)
    result = snapshot()["controlled_wait"]
    assert result["calls"] == 2 and result["wall_s"] >= 0.015
    assert 0 <= result["thread_cpu_s"] < result["wall_s"]
    reset()
    assert snapshot() == {}


def test_prepared_attempt_and_accepted_unsealed_frames_stay_incomplete_on_interrupt(tmp_path):
    raw, robot, audit = setup(tmp_path)
    audit.prepare_episode().result()
    audit.add_frame(frame(robot, 1))
    close(raw, audit, False)
    assert json.loads((raw.root / "capture.json").read_text())["status"] == "failed"
    assert not list(raw.root.glob("telemetry/*/complete.json"))
    result = json.loads(next(raw.root.glob("attempts/*/result.json")).read_text())
    assert result["status"] == "interrupted" and result["frames"] == 1


def test_logging_after_enqueue_is_carried_into_next_frame_diagnostics():
    from lerobot_robot_outcome_piper.stage_timing import reset, span, snapshot

    reset()
    snapshot()
    with span("log_output"):
        pass
    reset()
    result = snapshot()
    assert result["previous_cycle_post_enqueue_log"]["calls"] == 1
    reset()
    assert snapshot() == {}
