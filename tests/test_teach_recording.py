"""No real hardware: raw teach lifecycle and official Dataset conversion."""

from lerobot_robot_outcome_piper.raw_io import write_json
from dataclasses import asdict
import json
from types import SimpleNamespace as NS
import time
import numpy as np
import pytest
from lerobot_robot_outcome_piper.camera import CameraTelemetry
from lerobot_robot_outcome_piper.teach_data import SCHEMA, read_json, load_config
from lerobot_robot_outcome_piper.teach_record import Attempt, run_session
from lerobot_robot_outcome_piper.teach_dataset import convert, audit_teach


def config():
    return dict(
        fps=20,
        episode_time_s=10,
        writer_queue_size=30,
        task="synthetic teach",
        robot=dict(
            can_interface="synthetic",
            firmware="v189",
            execution_mode="read_only",
            feedback_timeout_s=0.2,
            capture_timing=dict(
                camera_max_age_s=0.05,
                joint_max_skew_s=0.01,
                image_state_max_skew_s=0.05,
                observation_max_age_s=0.1,
            ),
            scene=dict(
                scene_id="synthetic",
                base_installation="fixture",
                camera_view="fixture",
                work_area_notes="fixture",
            ),
            cameras={
                "d435": dict(
                    type="intelrealsense",
                    serial_number_or_name="synthetic",
                    width=32,
                    height=24,
                    fps=30,
                    use_rgb=True,
                    use_depth=True,
                )
            },
        ),
    )


def sample(i, now=None):
    now = 10 + i * 0.05 if now is None else now
    meta = {
        key: asdict(
            CameraTelemetry(
                i + 1,
                100 + i * 50,
                "hardware_clock",
                now - 0.003,
                now - 0.002,
                stream="depth" if key.endswith(".depth") else "color",
                depth_scale_m=0.001 if key.endswith(".depth") else None,
            )
        )
        for key in ("d435", "d435.depth")
    }
    row = dict(
        frame_index=i,
        tick_index=i,
        feedback_complete=True,
        sampled_monotonic_s=now,
        joint_rad=[0.1 + i * 0.001] * 6,
        gripper_m=0.01 + i * 0.001,
        gripper_mode="width",
        received_monotonic_s=[now - 0.001] * 5,
        drivers=[dict(enabled=True, error=False, received_monotonic_s=now - 0.01)] * 6,
        ctrl_mode=2,
        teach_status=1,
        arm_status=0,
        err_code=0,
        camera=meta,
    )
    pixels = {
        "d435": np.full((24, 32, 3), i, np.uint8),
        "d435.depth": np.full((24, 32), 1000 + i, np.uint16),
    }
    return pixels, row


@pytest.fixture
def safety(tmp_path):
    s = dict(
        schema_version="outcome-piper-safety-v1",
        joint_lower_rad=[-3.14] * 6,
        joint_upper_rad=[3.14] * 6,
        max_joint_step_rad=[0.1] * 6,
        gripper_lower_m=0,
        gripper_upper_m=0.065,
        max_gripper_step_m=0.005,
        workspace_lower_m=[-2] * 3,
        workspace_upper_m=[2] * 3,
        feedback_timeout_s=0.2,
        watchdog_timeout_s=1,
        motion_speed_percent=5,
        gripper_force_n=1,
        stop_strategy="electronic_emergency_stop",
    )
    p = tmp_path / "safety.json"
    write_json(p, s)
    return p


def raw_session(tmp_path, count=4):
    root = tmp_path / "raw"
    root.mkdir()
    cfg = config()
    write_json(root / "session.json", dict(schema=SCHEMA, source="manual_teach", config=cfg))
    a = Attempt(root / "attempt-one", "debug", cfg)
    for i in range(count):
        a.append(*sample(i))
    a.finish("teach_stopped")
    a.save("success", "synthetic")
    return root, a


def test_config_requires_explicit_readonly_no_xbox_and_timing(tmp_path):
    p = tmp_path / "config.json"
    write_json(p, config())
    cfg, normalized = load_config(p)
    assert cfg.execution_mode == "read_only" and normalized["fps"] == 20
    for change in ("motion", "timing", "xbox"):
        c = config()
        if change == "motion":
            c["robot"]["execution_mode"] = "motion"
            c["robot"]["safety_path"] = "unused"
        if change == "timing":
            c["robot"]["capture_timing"] = None
        if change == "xbox":
            c["teleop"] = {}
        write_json(p, c)
        with pytest.raises(ValueError):
            load_config(p)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "device_time",
        "host_time",
        "stale",
        "missing_tick",
        "joint_stale",
        "skew",
        "mode",
        "driver",
        "nan",
    ],
)
def test_quality_errors_cannot_be_saved_as_valid(fault, tmp_path):
    a = Attempt(tmp_path / "attempt", "debug", config())
    a.append(*sample(0))
    pixels, r = sample(1)
    if fault == "duplicate":
        r["camera"]["d435"]["frame_number"] = 1
    if fault == "device_time":
        r["camera"]["d435"]["device_timestamp_ms"] = 0
    if fault == "host_time":
        r["sampled_monotonic_s"] = 9
    if fault == "stale":
        r["camera"]["d435"]["received_monotonic_s"] -= 1
    if fault == "missing_tick":
        r["tick_index"] = 3
    if fault == "joint_stale":
        r["received_monotonic_s"][0] -= 1
    if fault == "skew":
        r["received_monotonic_s"][0] -= 0.02
    if fault == "mode":
        r["ctrl_mode"] = 1
    if fault == "driver":
        r["drivers"][0]["error"] = True
    if fault == "nan":
        r["gripper_m"] = float("nan")
    with pytest.raises(RuntimeError) as e:
        a.append(pixels, r)
    a.fail(e.value)
    assert read_json(a.path / "result.json")["data_valid"] is False


def test_normal_early_stop_preserves_gripper_and_has_no_actions(tmp_path):
    root, a = raw_session(tmp_path)
    rows = [json.loads(s) for s in (a.path / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and rows[-1]["gripper_m"] == pytest.approx(0.013)
    assert all("action" not in r for r in rows)
    assert read_json(a.path / "result.json")["end_reason"] == "teach_stopped"
    with np.load(a.path / rows[-1]["files"]["d435.depth"]) as z:
        np.testing.assert_array_equal(z["depth"], sample(3)[0]["d435.depth"])


def test_compression_failure_keeps_source_and_invalidates_attempt(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper import episode_save

    a = Attempt(tmp_path / "attempt", "debug", config())
    for i in range(2):
        a.append(*sample(i))
    a.finish("operator_end")

    def fail(*a):
        raise OSError("disk full")

    monkeypatch.setattr(episode_save, "compress_depth_file", fail)
    with pytest.raises(OSError) as e:
        a.save("success", "")
    a.fail(e.value)
    assert read_json(a.path / "result.json")["status"] == "failed"
    assert len(list(a.path.glob("*.npz"))) == 2


def test_official_convert_reload_audit_and_tamper(tmp_path, safety, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_robot_outcome_piper.action_audit import verify_telemetry
    import lerobot_robot_outcome_piper.sdk as sdk

    monkeypatch.setattr(
        sdk, "create_piper", lambda *a: pytest.fail("offline conversion created hardware")
    )
    root, a = raw_session(tmp_path)
    out = tmp_path / "dataset"
    assert convert(root, out, "local/teach-test", safety, video=False) == dict(episodes=1, frames=3)
    ds = LeRobotDataset("local/teach-test", root=out, depth_output_unit="m")
    assert verify_telemetry(out, ds) == dict(episodes=1, frames=3)
    np.testing.assert_array_equal(ds[0]["action"], np.asarray([0.101] * 6 + [0.011], np.float32))
    np.testing.assert_array_equal(
        ds[0]["observation.state"], np.asarray([0.1] * 6 + [0.01], np.float32)
    )
    with pytest.raises(FileExistsError):
        convert(root, out, "local/teach-test", safety, video=False)
    p = out / "teach/index.json"
    index = read_json(p)
    index[0]["target_frame"] = 0
    write_json(p, index)
    with pytest.raises(RuntimeError, match="provenance"):
        audit_teach(out, ds)
    assert read_json(a.path / "result.json")["status"] == "saved"


def test_discarded_and_failed_attempts_do_not_enter_conversion(tmp_path, safety):
    root, a = raw_session(tmp_path)
    for name, status in [("attempt-discarded", "discarded"), ("attempt-failed", "failed")]:
        p = root / name
        p.mkdir()
        write_json(p / "result.json", dict(status=status, data_valid=False))
    assert (
        convert(root, tmp_path / "dataset", "local/teach-filter", safety, video=False)["frames"]
        == 3
    )


def test_excessive_target_is_rejected_without_clipping(tmp_path, safety):
    root, a = raw_session(tmp_path)
    s = read_json(safety)
    s["max_joint_step_rad"] = [0.0001] * 6
    write_json(safety, s)
    with pytest.raises(ValueError, match="execution step"):
        convert(root, tmp_path / "dataset", "local/teach-limit", safety, video=False)
    assert not (tmp_path / "dataset").exists()


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, dt):
        self.now += dt
        time.sleep(0.0001)


def test_terminal_stop_save_quit_without_hardware_commands(tmp_path, monkeypatch):
    from threading import Event

    finished = Event()
    original_save = Attempt.save

    def save_and_notify(self, outcome, note):
        result = original_save(self, outcome, note)
        finished.set()
        return result

    monkeypatch.setattr(Attempt, "save", save_and_notify)
    clock = Clock()
    cfg = config()
    n = [0]

    def feedback():
        r = sample(n[0], clock())[1]
        r["teach_status"] = 2 if n[0] >= 4 else 1
        return r

    def read():
        value = sample(n[0], clock())
        n[0] += 1
        return value

    source = NS(feedback=feedback, read=read)
    stage = [0]

    def poll():
        if stage[0] == 2:
            return "quit" if finished.is_set() else None
        results = list(tmp_path.glob("attempt-*/result.json"))
        status = read_json(results[0])["status"] if results else None
        if stage[0] == 0:
            stage[0] = 1
            return "start debug"
        if status == "review" and stage[0] == 1:
            stage[0] = 2
            return "save success"
        if status == "saved":
            return "quit"
        return None

    assert run_session(source, cfg, tmp_path, NS(poll=poll), clock=clock, sleep=clock.sleep) == 1
    result = read_json(next(tmp_path.glob("attempt-*/result.json")))
    assert result["status"] == "saved" and result["frames"] == 4


def test_receive_source_does_not_call_control_methods(monkeypatch):
    from lerobot_robot_outcome_piper.teach_source import TeachSource
    from lerobot_robot_outcome_piper import sdk, timing, camera

    calls = []
    comm = NS(send=lambda *a: pytest.fail("transmission"))
    arm = NS(
        get_context=lambda: NS(get_comm=lambda: comm),
        init_effector=lambda _: calls.append("effector") or object(),
        OPTIONS=NS(EFFECTOR=NS(AGX_GRIPPER="gripper")),
        connect=lambda: calls.append("connect"),
        disconnect=lambda: calls.append("disconnect"),
    )
    monkeypatch.setattr(sdk, "create_piper", lambda *a: arm)
    monkeypatch.setattr(timing, "FeedbackReceiver", lambda *a: NS(wait_ready=lambda _: True))
    monkeypatch.setattr(camera, "make_timed_cameras", lambda _: {})
    monkeypatch.setattr(TeachSource, "feedback", lambda self: sample(0)[1])
    s = TeachSource(NS(can_interface="synthetic", firmware="v189", cameras={}, capture_timing=None))
    assert not calls
    s.connect()
    with pytest.raises(RuntimeError, match="blocked"):
        comm.send(NS(arbitration_id=0x151))
    s.disconnect()
    assert calls == ["effector", "connect", "disconnect"]


def test_interrupt_during_save_never_leaves_complete_attempt(tmp_path, monkeypatch):
    from threading import Event

    clock = Clock()
    count = [0]
    saving = Event()
    release = Event()
    original = Attempt.save

    def slow_save(self, outcome, note):
        saving.set()
        release.wait(2)
        return original(self, outcome, note)

    monkeypatch.setattr(Attempt, "save", slow_save)

    def feedback():
        if saving.is_set():
            release.set()
            raise KeyboardInterrupt()
        r = sample(count[0], clock())[1]
        r["teach_status"] = 2 if count[0] >= 3 else 1
        return r

    def read():
        v = sample(count[0], clock())
        count[0] += 1
        return v

    started = [False]

    def poll():
        if not started[0]:
            started[0] = True
            return "start debug"
        p = list(tmp_path.glob("attempt-*/result.json"))
        if p and read_json(p[0])["status"] == "review":
            return "save success"
        return None

    with pytest.raises(KeyboardInterrupt):
        run_session(
            NS(feedback=feedback, read=read),
            config(),
            tmp_path,
            NS(poll=poll),
            clock=clock,
            sleep=clock.sleep,
        )
    result = read_json(next(tmp_path.glob("attempt-*/result.json")))
    assert result["status"] == "failed" and result["data_valid"] is False


def test_empty_wait_can_end_and_discard_without_dataset(tmp_path):
    clock = Clock()
    commands = iter(["start debug", "end", "redo", "quit"])

    def feedback():
        r = sample(0, clock())[1]
        r["teach_status"] = 2
        return r

    source = NS(feedback=feedback, read=lambda: pytest.fail("no recording yet"))
    assert (
        run_session(
            source,
            config(),
            tmp_path,
            NS(poll=lambda: next(commands)),
            clock=clock,
            sleep=clock.sleep,
        )
        == 0
    )
    assert read_json(next(tmp_path.glob("attempt-*/result.json")))["status"] == "discarded"


def test_multiple_attempts_do_not_pair_across_episode_boundaries(tmp_path, safety):
    root, a = raw_session(tmp_path)
    b = Attempt(root / "attempt-two", "second", config())
    for i in range(3):
        pixels, row = sample(i, 100 + i * 0.05)
        b.append(pixels, row)
    b.finish("operator_end")
    b.save("failure", "missed grasp")
    assert convert(root, tmp_path / "dataset", "local/teach-two", safety, video=False) == dict(
        episodes=2, frames=5
    )
    index = read_json(tmp_path / "dataset/teach/index.json")
    assert [(x["episode_index"], x["frame_index"]) for x in index] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
    ]


def test_conversion_write_failure_is_not_complete(tmp_path, safety, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, _ = raw_session(tmp_path)

    def fail(self, *a, **kw):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(LeRobotDataset, "save_episode", fail)
    out = tmp_path / "dataset"
    with pytest.raises(OSError, match="write failure"):
        convert(root, out, "local/teach-fail", safety, video=False)
    assert read_json(out / "teach-conversion.json")["status"] == "failed"
    assert read_json(root / "attempt-one/result.json")["data_valid"] is True


def test_missing_terminal_rgb_is_rejected(tmp_path, safety):
    root, a = raw_session(tmp_path)
    (a.path / a.rows[-1]["files"]["d435"]).unlink()
    with pytest.raises(FileNotFoundError):
        convert(root, tmp_path / "dataset", "local/teach-missing", safety, video=False)


def test_raw_resume_refuses_scene_change_before_connect(tmp_path, monkeypatch):
    from lerobot_robot_outcome_piper import teach_record, record_control
    from lerobot_robot_outcome_piper.teach_source import TeachSource

    path = tmp_path / "config.json"
    write_json(path, config())
    _, normalized = load_config(path)
    root = tmp_path / "raw"
    root.mkdir()
    write_json(root / "session.json", dict(schema=SCHEMA, source="manual_teach", config=normalized))
    changed = config()
    changed["robot"]["scene"]["scene_id"] = "another-table"
    write_json(path, changed)
    monkeypatch.setattr(record_control, "TerminalCommands", lambda: object())
    monkeypatch.setattr(TeachSource, "connect", lambda self: pytest.fail("must not connect"))
    with pytest.raises(ValueError, match="identical"):
        teach_record.main(["--config", str(path), "--output", str(root), "--resume"])


def test_default_video_roundtrip_on_linux(tmp_path, safety):
    import sys

    if sys.platform != "linux":
        pytest.skip("existing Mac FFmpeg/TorchCodec limitation; verified on controller Linux")
    root, _ = raw_session(tmp_path)
    assert (
        convert(root, tmp_path / "dataset", "local/teach-video", safety, video=True)["frames"] == 3
    )


def test_xbox_cannot_resume_teach_dataset(tmp_path):
    from lerobot_robot_outcome_piper.xbox_raw import RawFrames

    out = tmp_path / "dataset"
    out.mkdir()
    write_json(out / "teach-conversion.json", {"source": "manual_teach"})
    with pytest.raises(ValueError, match="Xbox cannot resume"):
        RawFrames(out, fps=20, features={}, robot_type="outcome_piper", resume=True)


def test_approved_65mm_endpoint_preserves_five_mm_step_and_force(safety):
    from lerobot_robot_outcome_piper.safety import load_motion_safety
    from lerobot_robot_outcome_piper.teach_dataset import check_candidate

    limits = load_motion_safety(safety)
    a, b = sample(0)[1], sample(1)[1]
    a["gripper_m"], b["gripper_m"] = 0.060, 0.065
    check_candidate(a, b, limits)
    assert limits.max_gripper_step == 0.005 and limits.gripper_force_n == 1
    b["gripper_m"] = 0.0651
    with pytest.raises(ValueError, match="gripper outside bounds"):
        check_candidate(a, b, limits)


def test_serialized_float32_target_is_checked_at_joint_boundary(safety):
    from lerobot_robot_outcome_piper.safety import load_motion_safety
    from lerobot_robot_outcome_piper.teach_dataset import check_candidate

    values = read_json(safety)
    values["joint_upper_rad"][0] = 0.3
    write_json(safety, values)
    a, b = sample(0)[1], sample(1)[1]
    a["joint_rad"][0], b["joint_rad"][0] = 0.299, 0.3
    with pytest.raises(ValueError, match="outside joint bounds"):
        check_candidate(a, b, load_motion_safety(safety))


def test_camera_logical_name_does_not_determine_stream_kind(tmp_path, safety):
    cfg = config()
    cfg["robot"]["cameras"]["view.depth"] = cfg["robot"]["cameras"].pop("d435")
    root = tmp_path / "raw"
    root.mkdir()
    write_json(root / "session.json", dict(schema=SCHEMA, source="manual_teach", config=cfg))
    a = Attempt(root / "attempt-camera", "debug", cfg)
    for i in range(3):
        pixels, row = sample(i)
        pixels = {k.replace("d435", "view.depth"): v for k, v in pixels.items()}
        row["camera"] = {k.replace("d435", "view.depth"): v for k, v in row["camera"].items()}
        a.append(pixels, row)
    a.finish("teach_stopped")
    a.save("success", "")
    assert (
        convert(root, tmp_path / "dataset", "local/teach-camera-name", safety, video=False)[
            "frames"
        ]
        == 2
    )


def test_rgb_only_teach_config_and_conversion(tmp_path, safety):
    cfg = config()
    cfg["robot"]["cameras"]["d435"]["use_depth"] = False
    p = tmp_path / "config.json"
    write_json(p, cfg)
    (parsed, _) = load_config(p)
    assert not parsed.cameras["d435"].use_depth
    root = tmp_path / "raw"
    root.mkdir()
    write_json(root / "session.json", dict(schema=SCHEMA, source="manual_teach", config=cfg))
    a = Attempt(root / "attempt-rgb", "debug", cfg)
    for i in range(3):
        (pixels, row) = sample(i)
        pixels.pop("d435.depth")
        row["camera"].pop("d435.depth")
        a.append(pixels, row)
    a.finish("teach_stopped")
    a.save("success", "")
    assert not list(a.path.glob("*.npz"))
    assert convert(root, tmp_path / "rgb", "local/teach-rgb", safety, video=False)["frames"] == 2
    with pytest.raises(ValueError, match="no captured depth"):
        convert(
            root,
            tmp_path / "depth",
            "local/teach-no-depth",
            safety,
            video=False,
            include_depth=True,
        )


def test_default_rgb_export_keeps_one_depth_archive_and_audits_without_it(tmp_path, safety):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    (root, a) = raw_session(tmp_path)
    out = tmp_path / "rgb"
    convert(root, out, "local/teach-rgb-default", safety, video=False)
    assert not list(out.rglob("*.npz"))
    ds = LeRobotDataset("local/teach-rgb-default", root=out)
    assert not ds.meta.depth_keys
    root.rename(tmp_path / "archive-moved")
    assert audit_teach(out, ds)["frames"] == 3


def test_explicit_depth_export_uses_original_z16_and_supports_relocation(tmp_path, safety):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    (root, _) = raw_session(tmp_path)
    out = tmp_path / "rgbd"
    convert(root, out, "local/teach-depth-export", safety, video=False, include_depth=True)
    assert not list(out.rglob("*.npz"))
    ds = LeRobotDataset("local/teach-depth-export", root=out, depth_output_unit="m")
    assert ds.meta.depth_keys == ["observation.images.d435.depth"]
    assert audit_teach(out, ds)["frames"] == 3
    moved = tmp_path / "moved"
    root.rename(moved)
    with pytest.raises(FileNotFoundError):
        audit_teach(out, ds)
    assert audit_teach(out, ds, raw_source=moved)["frames"] == 3
    first = next((moved / "attempt-one").glob("*.npz"))
    with np.load(first) as z:
        values = z["depth"].copy()
    np.savez(first, depth=values + 1)
    with pytest.raises(RuntimeError, match="depth differs"):
        audit_teach(out, ds, raw_source=moved)


def test_negative_gripper_maps_action_only_roundtrip(tmp_path, safety):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_robot_outcome_piper.teach_dataset import ACTION_RULE

    root = tmp_path / "raw"
    root.mkdir()
    cfg = config()
    write_json(root / "session.json", dict(schema=SCHEMA, source="manual_teach", config=cfg))
    attempt = Attempt(root / "attempt-negative", "P3", cfg)
    values = [-0.0004, -0.0035, 0.0012, -0.0003]
    for i, value in enumerate(values):
        pixels, row = sample(i)
        row["gripper_m"] = value
        attempt.append(pixels, row)
    attempt.finish("operator_end")
    attempt.save("success", "synthetic negative readings")
    original = (attempt.path / "samples.jsonl").read_bytes()
    out = tmp_path / "dataset"
    assert convert(root, out, "local/negative-grip", safety, video=False) == dict(
        episodes=1, frames=3
    )
    ds = LeRobotDataset("local/negative-grip", root=out, depth_output_unit="m")
    for i in range(3):
        assert float(ds[i]["observation.state"][6]) == float(np.float32(values[i]))
        assert float(ds[i]["action"][6]) == float(np.float32(max(0, values[i + 1])))
    assert original == (attempt.path / "samples.jsonl").read_bytes()
    assert audit_teach(out, ds)["frames"] == 3
    info = read_json(out / "teach-conversion.json")
    assert info["rule"] == ACTION_RULE and info["gripper_mapping"]["mapped_frames"] == 2
    assert info["gripper_mapping"]["raw_target_min_m"] == -0.0035
    index = read_json(out / "teach/index.json")
    index[0]["raw_next_gripper_m"] = 0
    write_json(out / "teach/index.json", index)
    with pytest.raises(RuntimeError, match="mapping provenance"):
        audit_teach(out, ds)


def test_gripper_mapping_does_not_bypass_step_or_geometry(safety, monkeypatch):
    from lerobot_robot_outcome_piper import teach_dataset as module
    from lerobot_robot_outcome_piper.safety import load_motion_safety

    limits = load_motion_safety(safety)
    a, b = sample(0)[1], sample(1)[1]
    a["gripper_m"] = 0.01
    b["gripper_m"] = -0.0003
    with pytest.raises(ValueError, match="execution step"):
        module.check_candidate(a, b, limits)
    a["gripper_m"] = -0.0003
    from lerobot_robot_outcome_piper import execution_constraints

    monkeypatch.setattr(execution_constraints, "workspace_pose_allowed", lambda *args: False)
    with pytest.raises(ValueError, match="workspace"):
        module.check_candidate(a, b, limits)


def test_original_action_rule_remains_strict():
    from lerobot_robot_outcome_piper.teach_dataset import action_state, RULE

    row = sample(0)[1]
    row["gripper_m"] = -0.0004
    assert action_state(row, RULE)[6] == np.float32(-0.0004)
    assert action_state(row)[6] == 0


def test_stale_frame_error_preserves_age_limit_and_read_stage():
    from lerobot_robot_outcome_piper.teach_data import validate_sample

    pixels, row = sample(0)
    row["frame_index"] = 0
    row["tick_index"] = 0
    row["camera"]["d435"]["received_monotonic_s"] = row["sampled_monotonic_s"] - 0.051
    row["sampling_read_timing"] = {"feedback_started_s": row["sampled_monotonic_s"] - 0.03}
    with pytest.raises(
        RuntimeError,
        match=r"camera frame stale: stream=d435.*age_ms=51.000.*limit_ms=50.000.*feedback_started_s",
    ):
        validate_sample(row, None, config())


def test_explicit_position_selection_keeps_rejected_attempt_unchanged(tmp_path, safety):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "raw"
    root.mkdir()
    cfg = config()
    write_json(root / "session.json", dict(schema=SCHEMA, source="manual_teach", config=cfg))
    for position in ("P1", "P2"):
        attempt = Attempt(root / ("attempt-" + position), position, cfg)
        for i in range(3):
            pixels, row = sample(i)
            if position == "P1" and i:
                row["joint_rad"][0] = 1.0
            attempt.append(pixels, row)
        attempt.finish("operator_end")
        attempt.save("success", "synthetic")
    original = (root / "attempt-P1/samples.jsonl").read_bytes()
    with pytest.raises(ValueError, match="step"):
        convert(root, tmp_path / "all", "local/all", safety, video=False)
    out = tmp_path / "selected"
    assert convert(root, out, "local/selected", safety, video=False, positions=["P2"]) == {
        "episodes": 1,
        "frames": 2,
    }
    ds = LeRobotDataset("local/selected", root=out)
    assert audit_teach(out, ds)["frames"] == 2
    assert (root / "attempt-P1/samples.jsonl").read_bytes() == original
    assert read_json(out / "teach-conversion.json")["selected_positions"] == ["P2"]
    with pytest.raises(ValueError, match="requested positions"):
        convert(root, tmp_path / "missing", "local/missing", safety, video=False, positions=["P3"])
