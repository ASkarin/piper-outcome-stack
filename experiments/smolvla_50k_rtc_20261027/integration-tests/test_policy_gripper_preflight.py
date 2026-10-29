"""Run the check-only entry with synthetic model outputs and no device construction."""

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_plugin import safety as synthetic_safety


def test_check_entry_shapes_gripper_and_commits_only_valid_virtual_targets(tmp_path, monkeypatch):
    import draccus
    import lerobot_robot_outcome_piper as plugin
    import lerobot_robot_outcome_piper.config as configs
    import lerobot_robot_outcome_piper.safety as limits
    import piper_policy.smolvla as smolvla
    import piper_policy.smolvla_execution as execution
    from lerobot_robot_outcome_piper.gripper_reference import GripperReferenceSettings
    from lerobot_robot_outcome_piper.timing import CaptureTiming

    path = Path(__file__).parents[1] / "piper_policy_trial.py"
    spec = importlib.util.spec_from_file_location("gripper_preflight_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    class RobotCfg:
        execution_mode = "motion"
        capture_timing = CaptureTiming(0.2, 0.1, 0.05, 0.2)
        safety_path = "unused"

    class TeleopCfg:
        control_hz = 50
        gripper_reference = asdict(GripperReferenceSettings(0.02, 0.12, 1.2, 0.008))

    monkeypatch.setattr(configs, "OutcomePiperConfig", RobotCfg)
    monkeypatch.setattr(configs, "OutcomePiperXboxConfig", TeleopCfg)
    monkeypatch.setattr(
        draccus, "decode", lambda cls, d: RobotCfg() if d["kind"] == "robot" else TeleopCfg()
    )
    monkeypatch.setattr(
        limits,
        "load_motion_safety",
        lambda p: replace(
            synthetic_safety(),
            gripper_upper=0.12,
            max_gripper_step=0.015,
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("check mode must not construct hardware")

    monkeypatch.setattr(plugin, "OutcomePiper", forbidden)
    monkeypatch.setattr(plugin, "OutcomePiperXbox", forbidden)
    raw = NS(
        model=NS(
            config=NS(
                rtc_config=NS(
                    max_guidance_weight=10,
                    prefix_attention_schedule=NS(value="LINEAR"),
                )
            )
        )
    )
    monkeypatch.setattr(smolvla.SmolVLAChunkPredictor, "from_checkpoint", lambda *a, **kw: raw)
    state = np.asarray([[0.0] * 6 + [0.03]])
    monkeypatch.setattr(
        entry,
        "record_candidate_references",
        lambda *a, **kw: (np.zeros((1, 2, 3, 3), np.uint8), state, []),
    )
    now = [100.0]
    monkeypatch.setattr(entry.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(entry.time, "sleep", lambda dt: now.__setitem__(0, now[0] + dt))

    class Executor:
        rtc_guidance, rtc_num_steps, rtc_initial_delay = "native_joint6", 10, 20
        ensemble_updates = 0

        def __init__(self, *args, **kwargs):
            self.step = -1

        def execution_chunk(self, image, observed, sequence, **kwargs):
            self.step += 1
            chunk = np.repeat(np.asarray(observed)[None], 50, axis=0)
            chunk[:, 6] = 0.10
            if self.step == 2:
                chunk[:, 0] = 0.3  # Existing joint bound rejects this virtual candidate.
            self.execution_details = {"rtc_request": {"prefix_model_actions": [[0.0] * 7]}}
            return chunk, 0, True

        def select_target(self, chunk, **kwargs):
            return chunk[0].copy()

        def reset_execution(self):
            pass

    monkeypatch.setattr(execution, "AsyncSmolVLAExecutor", Executor)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"robot": {"kind": "robot"}, "teleop": {"kind": "teleop"}}))
    reference, output = tmp_path / "reference.npz", tmp_path / "check.json"
    np.savez(reference, tasks=np.array(["test"]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trial",
            "check",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path),
            "--reference-inputs",
            str(reference),
            "--output",
            str(output),
            "--policy-type",
            "smolvla",
            "--task",
            "test",
            "--vlm-assets",
            str(tmp_path),
            "--joint-representation",
            "absolute",
            "--n-action-steps",
            "25",
        ],
    )
    entry.main()
    result = json.loads(output.read_text())
    assert result["status"] == "software_check_completed" and result["hardware_io"] is False
    checks = result["execution_preflight"]["recorded_target_checks"]
    assert len(checks) == 51
    assert checks[0]["gripper_plan"]["target_m"] == pytest.approx(0.03024)
    assert checks[2]["status"] == "rejected"
    assert not checks[2]["virtual_reference_committed"]
    assert checks[3]["gripper_plan"]["base_revision"] == checks[2]["gripper_plan"]["base_revision"]
    assert all(c["virtual_reference_committed"] for i, c in enumerate(checks) if i != 2)
    assert max(c["gripper_plan"]["target_m"] for c in checks) <= 0.038
