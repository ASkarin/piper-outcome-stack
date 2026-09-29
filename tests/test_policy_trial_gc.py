"""Exercise the operator entry's GC lifetime without connecting real devices."""

import gc
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest


@pytest.mark.parametrize("failure", [None, "model", "interrupt", "recorder"])
def test_trial_gc_restores_after_device_cleanup(tmp_path, monkeypatch, failure):
    import draccus
    import lerobot_robot_outcome_piper as plugin
    import lerobot_robot_outcome_piper.config as configs
    import lerobot_robot_outcome_piper.safety as safety
    import piper_outcome_stack.policy_startup as startup
    import piper_outcome_stack.policy_rgb_recording as recording
    from lerobot_robot_outcome_piper.robot import PiperState

    path = Path(__file__).parents[1] / "infra/acceptance/piper_policy_trial.py"
    spec = importlib.util.spec_from_file_location("trial_gc_test_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []

    class RobotCfg:
        execution_mode = "motion"
        capture_timing = NS()
        safety_path = "unused"

    class TeleopCfg:
        control_hz = 50

        def hold_settings(self):
            return None

    monkeypatch.setattr(configs, "OutcomePiperConfig", RobotCfg)
    monkeypatch.setattr(configs, "OutcomePiperXboxConfig", TeleopCfg)
    monkeypatch.setattr(
        draccus, "decode", lambda cls, d: RobotCfg() if d["kind"] == "robot" else TeleopCfg()
    )
    monkeypatch.setattr(safety, "load_motion_safety", lambda p: NS(motion_speed_percent=75))
    # The entry uses dataclasses.asdict only to serialize safety.
    monkeypatch.setattr(module, "asdict", lambda x: {"motion_speed_percent": 75})

    class Robot:
        state = PiperState.CONNECTED
        stop_outcome = "hold_confirmed"
        stop_error = None
        last_action_telemetry = {}

        def __init__(self, cfg):
            pass

        def configure_teleoperation(self, *a):
            pass

        def connect(self):
            assert not gc.isenabled()
            events.append("robot_connect")

        def enable(self):
            self.state = PiperState.ACTIVE

        def request_input_fault(self, e):
            events.append("hold")
            self.state = PiperState.FAULT

        def disconnect(self):
            assert not gc.isenabled()
            events.append("robot_disconnect")

    class Teleop:
        def __init__(self, cfg):
            pass

        def connect(self):
            assert not gc.isenabled()
            events.append("teleop_connect")

        def poll_emergency_stop(self):
            return False

        def disconnect(self):
            assert not gc.isenabled()
            events.append("teleop_disconnect")

    class Recorder:
        def __init__(self, *a):
            assert not gc.isenabled()

        def close(self):
            assert not gc.isenabled()
            events.append("recorder_close")
            if failure == "recorder":
                raise RuntimeError("recorder failure")
            return {"status": "completed"}

    monkeypatch.setattr(plugin, "OutcomePiper", Robot)
    monkeypatch.setattr(plugin, "OutcomePiperXbox", Teleop)
    monkeypatch.setattr(recording, "PolicyRGBRecorder", Recorder)
    monkeypatch.setattr(module, "PolicyInput", lambda *a: NS(poll=lambda: {}))
    monkeypatch.setattr(startup, "confirm_work_pose", lambda cfg: True)
    monkeypatch.setattr(startup, "prepare_work_pose", lambda *a: {"status": "arrived_held"})
    chunk = np.zeros((50, 7), np.float32)
    predictor = NS(
        predict=lambda *a: chunk,
        configure_temporal_ensemble=lambda *a: None,
        configure_action_steps=lambda *a: None,
        execution_chunk=lambda *a: (chunk, 0, True),
        select_target=lambda *a, **k: chunk[0],
        ensemble_updates=1,
        reset_execution=lambda: None,
    )
    monkeypatch.setattr(module.ACTChunkPredictor, "from_checkpoint", lambda *a, **k: predictor)
    monkeypatch.setattr(module, "recorded_target_check", lambda *a: {"status": "passed"})

    def run(*a, **k):
        assert not gc.isenabled()
        if failure == "model":
            raise ValueError("model failure")
        if failure == "interrupt":
            raise KeyboardInterrupt()
        return {"status": "segment_ended_held"}

    monkeypatch.setattr(module, "run_policy_trial", run)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"robot": {"kind": "robot"}, "teleop": {"kind": "teleop"}}))
    reference = tmp_path / "reference.npz"
    np.savez(
        reference,
        images=np.zeros((1, 2, 2, 3), np.uint8),
        states=np.zeros((1, 7)),
        expected_absolute_chunks=chunk[None],
    )
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trial",
            "run",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path),
            "--reference-inputs",
            str(reference),
            "--output",
            str(output),
            "--record-rgb",
        ],
    )
    before = gc.isenabled()
    gc.enable()
    try:
        if failure:
            error = {"model": ValueError, "interrupt": KeyboardInterrupt, "recorder": RuntimeError}[
                failure
            ]
            with pytest.raises(error):
                module.main()
        else:
            module.main()
        assert gc.isenabled()
        assert (
            events.index("robot_disconnect")
            < events.index("teleop_disconnect")
            < events.index("recorder_close")
        )
        if failure != "recorder":
            runtime = json.loads(output.read_text())["policy_runtime"]
            assert runtime["automatic_cyclic_gc_deferred"] and runtime["restored"]
    finally:
        if not before:
            gc.disable()
