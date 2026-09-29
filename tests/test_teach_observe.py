"""Fake receive-only teach diagnostics, no actual devices."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import json
import numpy as np
import pytest
from lerobot_robot_outcome_piper.camera import CameraTelemetry

spec = importlib.util.spec_from_file_location(
    "teach_observer", Path(__file__).parents[1] / "infra/acceptance/piper_teach_observe.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_transmit_is_blocked_before_underlying_send():
    attempts = []
    comm = NS(send=lambda *a: pytest.fail("must never transmit"))
    module.deny_transmission(comm, attempts)
    with pytest.raises(RuntimeError, match="blocked"):
        comm.send(NS(arbitration_id=0x151))
    assert len(attempts) == 1


def test_teaching_requires_explicit_fresh_teach_status():
    assert module.teaching(dict(feedback_complete=True, status_age_s=0.01, teach_status=1), 0.2)
    assert not module.teaching(dict(feedback_complete=True, status_age_s=0.3, teach_status=1), 0.2)
    assert not module.teaching(dict(feedback_complete=True, status_age_s=0.01, teach_status=3), 0.2)


def test_short_capture_preserves_states_and_rgbd_without_actions(tmp_path, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    arm = NS(has_comm_error=lambda: False)
    status = NS(ctrl_mode=2, arm_status=11, teach_status=1, mode_feedback=1, err_code=0)
    rx = NS(
        condition=threading.RLock(),
        received={0x2A1: 1.0},
        snapshot=lambda: NS(
            status=NS(msg=status),
            joints=NS(msg=[0.1] * 6),
            gripper=NS(msg=NS(value=0.012, mode="width")),
            received_s=(clock[0],) * 5,
        ),
        driver_states=lambda: (
            [
                (
                    NS(
                        msg=NS(foc_status=NS(driver_enable_status=False, driver_error_status=False))
                    ),
                    clock[0],
                )
            ]
            * 6
        ),
    )

    def read(timeout):
        metadata = {
            k: CameraTelemetry(
                int(clock[0] * 100),
                clock[0] * 1000,
                "device",
                clock[0],
                clock[0],
                depth_scale_m=0.001 if k == "depth" else None,
            )
            for k in ["color", "depth"]
        }
        return {
            "color": np.zeros((4, 6, 3), np.uint8),
            "depth": np.ones((4, 6), np.uint16),
        }, metadata

    result = module.collect(arm, rx, NS(read_with_metadata=read), tmp_path, [], seconds=0.25, hz=10)
    rows = [json.loads(s) for s in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert result["frames"] == 3
    assert all("action" not in row for row in rows)
    assert rows[0]["joint_rad"] == [0.1] * 6 and rows[0]["gripper_m"] == 0.012
    assert len(list(tmp_path.glob("*.png"))) == len(list(tmp_path.glob("*.npz"))) == 3
    with np.load(tmp_path / "frame-0000.npz") as data:
        np.testing.assert_array_equal(data["depth"], np.ones((4, 6), np.uint16))
    rx.snapshot = lambda: NS(
        status=NS(msg=status),
        joints=NS(msg=[0.1] * 6),
        gripper=NS(msg=NS(value=25.0, mode="angle")),
        received_s=(clock[0],) * 5,
    )
    row = module.feedback_row(arm, rx)
    assert row["gripper_m"] is None and row["gripper_value"] == 25.0
