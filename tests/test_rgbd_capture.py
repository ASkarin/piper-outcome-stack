"""Configured RGB/depth streams and actual upstream Dataset TIFF round trips."""

from types import SimpleNamespace as NS
import json
import threading
import numpy as np
import pytest

pytest.importorskip("lerobot")
from lerobot_robot_outcome_piper.realsense import TimedRealSenseCamera
from lerobot_robot_outcome_piper.camera import CameraTelemetry
from lerobot_robot_outcome_piper.action_audit import verify_telemetry
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from piper_outcome_stack.training import select_policy_inputs
from test_recording_telemetry import Robot, devices, configuration, record


def fake_camera(use_depth=True, use_rgb=True, scale=0.00025):
    camera = TimedRealSenseCamera.__new__(TimedRealSenseCamera)
    camera.stop_event = threading.Event()
    camera.new_frame_event = threading.Event()
    camera.frame_lock = threading.Lock()
    camera.metadata_condition = threading.Condition(camera.frame_lock)
    camera.latest_metadata = {}
    camera.latest_frames = {}
    camera.capture_error = None
    camera._warming_up = False
    camera.use_rgb = use_rgb
    camera.use_depth = use_depth
    camera.serial_number = "test-device"
    camera.rotation = None
    from lerobot.cameras.configs import ColorMode

    camera.color_mode = ColorMode.RGB
    camera._postprocess_image = lambda image, depth_frame=False: image
    intr = NS(width=8, height=6, fx=10.0, fy=10.0, ppx=4.0, ppy=3.0, model="test", coeffs=[0.0] * 5)
    profile = NS(
        as_video_stream_profile=lambda: NS(get_intrinsics=lambda: intr),
        get_extrinsics_to=lambda _: NS(
            rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], translation=[0.01, 0, 0]
        ),
    )

    def raw(array):
        return NS(
            profile=profile,
            get_data=lambda: array,
            get_frame_number=lambda: 1,
            get_timestamp=lambda: 10.0,
            get_frame_timestamp_domain=lambda: "hardware_clock",
            get_units=lambda: scale,
        )

    rgb = raw(np.full((6, 8, 3), 42, dtype=np.uint8))
    depth = raw(np.arange(48, dtype=np.uint16).reshape(6, 8))
    calls = 0

    def read():
        nonlocal calls
        calls += 1
        if calls == 2:
            camera.stop_event.set()
        return NS(get_color_frame=lambda: rgb, get_depth_frame=lambda: depth)

    camera._read_from_hardware = read
    return camera


def test_worker_publishes_atomic_rgbd_native_values_and_geometry():
    camera = fake_camera()
    camera._read_loop()
    assert set(camera.latest_frames) == {"color", "depth"}
    np.testing.assert_array_equal(
        camera.latest_frames["depth"], np.arange(48, dtype=np.uint16).reshape(6, 8)
    )
    meta = camera.latest_metadata["depth"]
    assert meta.depth_scale_m == 0.00025
    assert meta.intrinsics["fx"] == 10
    assert meta.depth_to_color["translation_m"] == [0.01, 0, 0]
    assert meta.alignment == "native"
    assert camera.capture_error is None


def test_depth_only_and_missing_or_invalid_depth():
    camera = fake_camera(use_rgb=False)
    camera._read_loop()
    assert set(camera.latest_frames) == {"depth"}
    bad = fake_camera(scale=float("nan"))
    bad._read_loop()
    assert "scale" in str(bad.capture_error)
    assert not bad.latest_frames
    missing = fake_camera()
    missing._read_from_hardware = lambda: NS(
        get_color_frame=lambda: True, get_depth_frame=lambda: None
    )
    missing._read_loop()
    assert "missing" in str(missing.capture_error)


class RGBDRobot(Robot):
    raw = np.arange(48, dtype=np.uint16).reshape(6, 8)
    scale = 0.00025

    @property
    def observation_features(self):
        return {**super().observation_features, "front.depth": (6, 8, 1)}

    def get_observation(self):
        result = super().get_observation()
        self.last_depth_frames = {"front.depth": self.raw.copy()}
        result["front.depth"] = (self.raw.astype(np.float32) * self.scale)[..., None]
        self.last_observation_telemetry["cameras"]["front.depth"] = {"depth_scale_m": self.scale}
        return result


def rgbd_devices(monkeypatch):
    _, teleop, events, listener = devices(monkeypatch)
    from lerobot.scripts import lerobot_record as official

    robot = RGBDRobot()
    monkeypatch.setattr(official, "make_robot_from_config", lambda _: robot)
    return robot, teleop, events, listener


@pytest.mark.parametrize("video,streaming", [(False, False), (True, False), (True, True)])
def test_rgbd_official_record_save_reload_resume_and_policy_selection(
    tmp_path, monkeypatch, video, streaming
):
    rgbd_devices(monkeypatch)
    cfg = configuration(tmp_path / "dataset")
    cfg.export_depth = True
    cfg.dataset.video = video
    cfg.dataset.streaming_encoding = streaming
    dataset = record(cfg)
    loaded = LeRobotDataset(dataset.repo_id, root=dataset.root, depth_output_unit="m")
    key = "observation.images.front.depth"
    assert loaded.features[key]["dtype"] == "image"
    assert loaded.features[key]["info"]["depth_unit"] == "m"
    np.testing.assert_array_equal(
        loaded[0][key].numpy().squeeze(0), RGBDRobot.raw.astype(np.float32) * RGBDRobot.scale
    )
    assert verify_telemetry(loaded.root, loaded) == {"episodes": 1, "frames": 1}
    raw_path = next((loaded.root / "telemetry").glob("*/raw_depth/*/*.npz"))
    with np.load(raw_path) as values:
        np.testing.assert_array_equal(values["depth"], RGBDRobot.raw)
    inputs = select_policy_inputs(loaded.features)
    assert set(inputs) == {"observation.state", "observation.images.d435"}
    assert tuple(inputs["observation.state"].shape) == (7,)
    explicit = select_policy_inputs(loaded.features, [key])
    assert tuple(explicit[key].shape) == (1, 6, 8)
    rgbd_devices(monkeypatch)
    resume = configuration(loaded.root, resume=True, repo_id=loaded.repo_id)
    resume.export_depth = True
    resume.dataset.video = video
    resume.dataset.streaming_encoding = streaming
    record(resume)
    # Default upstream loader returns depth in mm; audit handles its documented unit.
    loaded = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert verify_telemetry(loaded.root, loaded)["frames"] == 2
    raw_path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_telemetry(loaded.root, loaded)


def test_rgbd_rerecord_keeps_raw_attempts_and_only_accepts_saved_rows(tmp_path, monkeypatch):
    _, _, events, _ = rgbd_devices(monkeypatch)
    events["rerecord_episode"] = True
    ds = record(configuration(tmp_path / "dataset"))
    assert len(list((ds.root / "telemetry").glob("*/raw_depth/*/*.npz"))) == 2
    loaded = LeRobotDataset(ds.repo_id, root=ds.root)
    assert verify_telemetry(ds.root, loaded) == {"episodes": 1, "frames": 1}


def test_rgbd_robot_separates_raw_counts_from_metric_observation(tmp_path):
    from test_plugin import make_robot

    robot, _, _ = make_robot(tmp_path)
    connect_for_test(robot)
    raw = np.array([[0, 1000]], dtype=np.uint16)
    meta = CameraTelemetry(
        1, 1.0, "hardware_clock", 100.0, 100.0, stream="depth", depth_scale_m=0.00025
    )
    robot.cameras = {
        "range": NS(
            is_connected=True,
            disconnect=lambda: None,
            read_with_metadata=lambda _: ({"depth": raw}, {"depth": meta}),
        )
    }
    try:
        obs = robot.get_observation()
        assert obs["range.depth"].shape == (1, 2, 1)
        assert obs["range.depth"][0, 1, 0] == pytest.approx(0.25)
        np.testing.assert_array_equal(robot.last_depth_frames["range.depth"], raw)
        assert len([k for k in obs if k.endswith(".pos")]) == 7
    finally:
        robot.disconnect()


def test_backend_color_order_does_not_corrupt_depth_or_dataset_rgb():
    from lerobot.cameras.configs import ColorMode

    camera = TimedRealSenseCamera.__new__(TimedRealSenseCamera)
    camera.color_mode = ColorMode.BGR
    camera.rotation = None
    camera.capture_width = 2
    camera.capture_height = 1
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    depth = np.array([[0, 32000]], dtype=np.uint16)
    np.testing.assert_array_equal(camera._postprocess_image(rgb), rgb[..., ::-1])
    np.testing.assert_array_equal(camera._postprocess_image(depth, depth_frame=True), depth)


@pytest.mark.parametrize("explicit", [False, True])
def test_train_entry_parses_real_config_and_preserves_explicit_inputs(
    tmp_path, monkeypatch, explicit
):
    from lerobot.scripts import lerobot_train as official_train
    from lerobot.datasets import lerobot_dataset
    from piper_outcome_stack.training import train_main

    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": list("abcdefg")},
        "action": {"dtype": "float32", "shape": (7,), "names": list("abcdefg")},
        "observation.images.front": {
            "dtype": "image",
            "shape": (6, 8, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        "observation.images.side": {
            "dtype": "image",
            "shape": (6, 8, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        "observation.images.front.depth": {
            "dtype": "image",
            "shape": (6, 8, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
    }
    queries = []

    def metadata(*args, **kwargs):
        queries.append((args, kwargs))
        return NS(features=features)

    monkeypatch.setattr(lerobot_dataset, "LeRobotDatasetMetadata", metadata)
    calls = []
    monkeypatch.setattr(official_train, "train", lambda cfg: calls.append(cfg))
    config = {
        "dataset": {"repo_id": "local/parser", "root": str(tmp_path / "dataset")},
        "policy": {"type": "act", "device": "cpu", "push_to_hub": False},
        "output_dir": str(tmp_path / "training"),
    }
    if explicit:
        config["policy"]["input_features"] = {
            "observation.state": {"type": "STATE", "shape": [7]},
            "observation.images.side": {"type": "VISUAL", "shape": [3, 6, 8]},
        }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config))
    train_main([f"--config_path={path}"])
    assert len(calls) == 1
    assert set(calls[0].policy.input_features) == (
        {"observation.state", "observation.images.side"}
        if explicit
        else {"observation.state", "observation.images.front", "observation.images.side"}
    )
    assert len(queries) == (0 if explicit else 1)


def test_robot_normalizes_backend_bgr_without_changing_camera_cache(tmp_path):
    from test_plugin import make_robot

    robot, _, _ = make_robot(tmp_path)
    connect_for_test(robot)
    bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
    meta = CameraTelemetry(1, 1.0, "hardware_clock", 100.0, 100.0, pixel_format="bgr8")
    robot.cameras = {
        "front": NS(
            is_connected=True,
            disconnect=lambda: None,
            read_with_metadata=lambda _: ({"color": bgr}, {"color": meta}),
        )
    }
    try:
        obs = robot.get_observation()
        np.testing.assert_array_equal(obs["front"], [[[255, 0, 0]]])
        np.testing.assert_array_equal(bgr, [[[0, 0, 255]]])
        assert robot.last_observation_telemetry["cameras"]["front"]["observation_format"] == "rgb8"
    finally:
        robot.disconnect()


def test_raw_depth_write_failure_keeps_diagnostics_without_completion(tmp_path, monkeypatch):
    rgbd_devices(monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("raw depth write failed")

    monkeypatch.setattr(np, "savez", fail)
    root = tmp_path / "dataset"
    with pytest.raises(OSError, match="raw depth write failed"):
        record(configuration(root))
    assert not list((root / "telemetry").glob("*/complete.json"))
    events = [
        json.loads(line)
        for p in (root / "telemetry").glob("*/events.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert any(e["event"] == "failed" and "raw depth write failed" in e["error"] for e in events)


from test_plugin import connect_for_test  # noqa: E402


@pytest.mark.parametrize("warming_up", [True, False])
def test_startup_depth_counter_restart_is_not_runtime_restart(warming_up):
    camera = fake_camera()
    camera._warming_up = warming_up
    read = camera._read_from_hardware
    count = 0

    def sequence():
        nonlocal count
        packet = read()
        number = 5 if count == 0 else 0
        for frame in (packet.get_color_frame(), packet.get_depth_frame()):
            frame.get_frame_number = lambda: number
            frame.get_timestamp = lambda: float(number + 1)
        count += 1
        return packet

    camera._read_from_hardware = sequence
    camera._read_loop()
    if warming_up:
        assert camera.capture_error is None
        assert camera.latest_metadata["depth"].frame_number == 0
    else:
        assert "backwards" in str(camera.capture_error)
        assert camera.latest_metadata["depth"].frame_number == 5


def test_rgbd_capture_defaults_to_rgb_dataset_and_single_z16_archive(tmp_path, monkeypatch):
    rgbd_devices(monkeypatch)
    cfg = configuration(tmp_path / "dataset")
    dataset = record(cfg)
    loaded = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert not loaded.meta.depth_keys
    assert verify_telemetry(loaded.root, loaded)["frames"] == 1
    files = list((loaded.root / "telemetry").glob("*/raw_depth/*/*.npz"))
    assert len(files) == 1
    files[0].unlink()
    with pytest.raises(FileNotFoundError):
        verify_telemetry(loaded.root, loaded)


def test_record_cli_exposes_optional_metric_depth_export():
    from lerobot_robot_outcome_piper.cli import PiperRecordConfig
    from dataclasses import fields

    assert next((f for f in fields(PiperRecordConfig) if f.name == "export_depth")).default is False


def test_metric_export_rejects_rgb_only_source(tmp_path, monkeypatch):
    devices(monkeypatch)
    cfg = configuration(tmp_path / "dataset")
    cfg.export_depth = True
    with pytest.raises(ValueError, match="no captured depth"):
        record(cfg)
