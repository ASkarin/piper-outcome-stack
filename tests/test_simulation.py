"""Geometry, replay and isolation checks; no real device or simulation-trained policy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
import numpy as np

from piper_outcome_stack.sim.model import ASSETS, ACTION_KEYS, build_xml, limits
from piper_outcome_stack.sim.runtime import Simulation, load_config
from piper_outcome_stack.sim.analysis import alignment_report, PlaybackClock, replay
from piper_outcome_stack.sim2real.recording import read_record

ROOT = Path(__file__).parents[1]


@pytest.fixture
def sim():
    value = Simulation(load_config())
    yield value
    value.close()


def test_original_model_conversion_reproduces_checked_in_asset():
    assert build_xml() == (ASSETS / "piper.xml").read_text()


def test_geometry_matches_independent_original_urdf(sim):
    result = alignment_report(sim)
    assert result["conversion_passed"]
    assert result["sample_count"] == 34
    assert len(result["sdk_comparison"]["samples"]) == 18
    assert result["hardware_validated"] is False


def test_seven_value_mapping_flange_and_gripper(sim):
    q = np.r_[np.deg2rad([-42.488, 132.772, -106.826, -4.781, 43.002, 66.067]), 0.065]
    sim.set_measured(dict(zip(ACTION_KEYS, q, strict=True)))
    np.testing.assert_allclose(sim.action(), q, atol=1e-12)
    assert sim.data.qpos[sim.qids[6]] == pytest.approx(0.0325)
    assert sim.data.qpos[sim.qids[7]] == pytest.approx(-0.0325)
    assert sim.model.nq == 8 and sim.model.nu == 7
    assert sim.model.opt.timestep == 0.002
    assert sim.servo["identified"] is False


@pytest.mark.parametrize(
    "q",
    [
        [0] * 6,
        [0] * 8,
        [float("nan")] * 7,
        [float("inf")] * 7,
        [0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, -0.01],
        {k: 0 for k in ACTION_KEYS[:-1]},
    ],
)
def test_invalid_actions_fail_before_state_changes(sim, q):
    before = sim.action()
    with pytest.raises((ValueError, TypeError)):
        sim.set_measured(q)
    np.testing.assert_array_equal(sim.action(), before)


def test_j6_uses_narrower_model_intersection():
    lo, hi = limits()
    assert hi[5] < np.pi and lo[5] > -np.pi
    assert hi[5] == pytest.approx(2.0943951)


def record_file(tmp_path):
    # Synthetic no-hardware fixture; real source reports are tested separately on local.
    doc = {
        "status": "test_only",
        "samples": [
            {"monotonic_s": 10, "joint_rad": [0] * 6, "gripper_width_m": 0},
            {"monotonic_s": 10.02, "joint_rad": [0.01, 0, 0, 0, 0, 0], "gripper_width_m": 0.01},
            {"monotonic_s": 10.04, "joint_rad": [0] * 6, "gripper_width_m": 0},
        ],
        "commands": [
            {
                "name": "joint_waypoint",
                "args": [[0.01, 0, 0, 0, 0, 0]],
                "started_monotonic_s": 10.01,
            },
            {"name": "set_gripper_width", "args": [0.01, 1], "started_monotonic_s": 10.015},
            {"name": "joint_waypoint", "args": [[0] * 6], "started_monotonic_s": 10.03},
            {"name": "set_gripper_width", "args": [0, 1], "started_monotonic_s": 10.035},
        ],
    }
    for command in doc["commands"]:
        command["returned_monotonic_s"] = command["started_monotonic_s"] + 0.0001
    path = tmp_path / "record.json"
    path.write_text(json.dumps(doc))
    return path


def test_record_preserves_times_and_interpolates_only_for_display(tmp_path):
    rec = read_record(record_file(tmp_path))
    assert rec.duration == pytest.approx(0.04)
    assert rec.time_origin == 10
    assert rec.commands[0][0] == pytest.approx(0.01)
    assert rec.measured(0.01)[0] == pytest.approx(0.005)
    assert len(rec.samples) == 3


@pytest.mark.parametrize(
    "defect", ["missing_width", "nonfinite", "reversed", "unknown_command", "no_joint_commands"]
)
def test_malformed_record_rejected(tmp_path, defect):
    path = record_file(tmp_path)
    doc = json.loads(path.read_text())
    if defect == "missing_width":
        del doc["samples"][0]["gripper_width_m"]
    if defect == "nonfinite":
        doc["samples"][0]["monotonic_s"] = float("nan")
    if defect == "reversed":
        doc["samples"].reverse()
    if defect == "unknown_command":
        doc["commands"][0]["name"] = "unknown"
    if defect == "no_joint_commands":
        doc["commands"] = [c for c in doc["commands"] if c["name"] == "set_gripper_width"]
    path.write_text(json.dumps(doc))
    with pytest.raises((ValueError, KeyError)):
        read_record(path)


def test_pause_and_quit_are_explicit():
    clock = PlaybackClock()
    clock.key(32)
    assert clock.paused
    clock.key(32)
    assert not clock.paused
    clock.key(ord("Q"))
    assert clock.stop


@pytest.mark.parametrize("mode", ["measured", "commanded"])
def test_replay_modes_and_evidence(sim, tmp_path, mode, monkeypatch):
    monkeypatch.setattr(sim, "render", lambda: np.zeros((32, 32, 3), dtype=np.uint8))
    summary = replay(sim, read_record(record_file(tmp_path)), mode, tmp_path)
    assert summary["completed"]
    assert summary["original_samples"] == 3
    assert summary["rows"] > 3
    assert (summary["rmse_rad_m"] is None) == (mode == "measured")
    assert (tmp_path / f"{mode}.csv").exists()
    if mode == "measured":
        np.testing.assert_allclose(sim.action(), np.zeros(7), atol=1e-10)


def test_dynamic_model_uses_original_inertia_and_no_gravity_compensation(sim):
    assert sim.servo["kp"][0] > 0 and sim.servo["kv"][0] > 0
    initial = sim.action()
    for _ in range(10):
        sim.step()
    assert sim.data.time == pytest.approx(0.02)
    assert np.isfinite(sim.action()).all()
    assert not np.allclose(sim.action(), initial)


def test_window_without_display_fails_with_instruction(tmp_path):
    env = {**os.environ}
    env.pop("DISPLAY", None)
    result = subprocess.run(
        [sys.executable, "-m", "piper_outcome_stack.ops", "sim", "view"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "No DISPLAY" in result.stderr


def test_render_and_hardware_import_isolation(tmp_path):
    # A fresh process selects EGL before importing MuJoCo, unlike the geometry tests.
    code = """
import sys
from piper_outcome_stack.sim.cli import main
status=main(['doctor','--headless','--output',sys.argv[1]])
assert not any(n == 'pyAgxArm' or n.startswith('lerobot_robot_outcome_piper') or n == 'pyrealsense2' or n == 'can' for n in sys.modules)
raise SystemExit(status)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "doctor")],
        env={**os.environ, "MUJOCO_GL": "egl"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "doctor/summary.json").read_text())
    assert report["render_shape"] == [480, 640, 3]
    assert report["visible_robot_pixels"] > 100
    assert report["hardware_io"] is False
    assert (tmp_path / "doctor/camera.png").is_file()


def test_real_feedback_excursion_is_preserved_but_not_accepted_as_target(
    sim, tmp_path, monkeypatch
):
    path = record_file(tmp_path)
    doc = json.loads(path.read_text())
    doc["samples"][1]["joint_rad"][2] = 0.0005410520681182421
    path.write_text(json.dumps(doc))
    record = read_record(path)
    assert record.samples[1, 2] > 0
    with pytest.raises(ValueError):
        sim.set_target(record.samples[1])
    monkeypatch.setattr(sim, "render", lambda: np.zeros((32, 32, 3), dtype=np.uint8))
    summary = replay(sim, record, "measured", tmp_path)
    assert summary["measured_limit_excursion_samples"] == 1
    assert summary["measured_max_limit_excursion_rad_m"][2] == pytest.approx(0.0005410520681182421)
    assert summary["rmse_rad_m"] is None


def test_command_without_dispatch_return_is_not_invented(tmp_path):
    path = record_file(tmp_path)
    doc = json.loads(path.read_text())
    del doc["commands"][0]["returned_monotonic_s"]
    path.write_text(json.dumps(doc))
    with pytest.raises(KeyError):
        read_record(path)
