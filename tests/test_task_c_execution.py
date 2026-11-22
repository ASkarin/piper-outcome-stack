import json
from types import SimpleNamespace as NS

import pytest
import torch

from piper_outcome_stack import task_c_execution as execution
from test_task_c import c_config as c_config  # noqa: F401


def prepare_stub(monkeypatch):
    resets = []
    predictor = NS(reset_execution=lambda: resets.append(True))
    monkeypatch.setattr(
        execution,
        "prepare_policy",
        lambda *a: (
            NS(),
            NS(hold_settings=lambda: NS()),
            NS(),
            NS(),
            predictor,
            {"reference_errors": []},
        ),
    )
    return resets


def test_check_never_constructs_devices_or_runs(c_config, tmp_path, monkeypatch):
    import lerobot_robot_outcome_piper as plugin

    resets = prepare_stub(monkeypatch)

    def forbidden(*a, **kw):
        pytest.fail("C check attempted device construction/run")

    monkeypatch.setattr(plugin, "OutcomePiper", forbidden)
    monkeypatch.setattr(plugin, "OutcomePiperXbox", forbidden)
    monkeypatch.setattr(execution, "run_segment", forbidden)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(c_config))
    result = execution.execute(path, "check", tmp_path / "check.json")
    assert result["status"] == "software_check_completed" and not result["hardware_io"]
    assert resets and result["recipe"]["joint_lowpass"] is False
    assert result["recipe"]["inference_precision"] == "fp32"
    assert "task_outcome" not in result


@pytest.mark.parametrize("failure", [None, "motion", "robot_cleanup", "teleop_cleanup"])
def test_current_pose_entry_and_cleanup_keep_existing_motion_contract(
    c_config, tmp_path, monkeypatch, failure
):
    from contextlib import nullcontext
    import lerobot_robot_outcome_piper as plugin
    from lerobot_robot_outcome_piper.robot import PiperState
    from test_policy_motion import Input

    resets = prepare_stub(monkeypatch)
    events = []

    class Robot:
        state = PiperState.DISCONNECTED
        stop_outcome = "hold_confirmed"
        stop_error = None
        last_action_telemetry = {"result": "synthetic"}

        def configure_teleoperation(self, *args):
            events.append("configured")

        def connect(self):
            self.state = PiperState.ACTIVE
            events.append("connect")

        def enable(self):
            events.append("enable")

        def request_input_fault(self, error):
            self.state = PiperState.FAULT
            events.append("fault")

        def disconnect(self):
            events.append("robot_cleanup")
            if failure == "robot_cleanup":
                raise RuntimeError("robot cleanup failure")

    class Teleop(Input):
        def connect(self):
            events.append("teleop_connect")

        def poll_emergency_stop(self):
            return False

        def disconnect(self):
            events.append("teleop_cleanup")
            if failure == "teleop_cleanup":
                raise RuntimeError("teleop cleanup failure")

    def run(*args, **kwargs):
        events.append("run")
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert kwargs["max_run_s"] == 2 and kwargs["max_actions"] == 100
        assert kwargs["record_action_chunks"] and kwargs["gripper_reference_settings"] is not None
        if failure == "motion":
            raise RuntimeError("motion failure")
        return dict(status="segment_ended_held", hold_confirmed=True, policy_actions_sent=1)

    monkeypatch.setattr(plugin, "OutcomePiper", lambda cfg: Robot())
    monkeypatch.setattr(plugin, "OutcomePiperXbox", lambda cfg: Teleop())
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.input_safety.motion_input_safety_scope", nullcontext
    )
    monkeypatch.setattr("piper_outcome_stack.policy_motion.run_policy_trial", run)
    monkeypatch.setattr(
        "piper_outcome_stack.policy_startup.prepare_work_pose",
        lambda *a, **kw: pytest.fail("C must not go to A"),
    )
    path, output = tmp_path / "c.json", tmp_path / "trial.json"
    path.write_text(json.dumps(c_config))
    if failure:
        with pytest.raises(RuntimeError, match="failure"):
            execution.execute(path, "run", output)
    else:
        execution.execute(path, "run", output)
    report = json.loads(output.read_text())
    assert report["status"] == ("failed" if failure else "segment_ended_held")
    assert report["last_action_telemetry"] == Robot.last_action_telemetry
    assert events[-2:] == ["robot_cleanup", "teleop_cleanup"]
    assert ("fault" in events) == (failure == "motion")
    assert resets and report["rgb_recording"]["status"] == "completed"
    assert "task_outcome" not in report


def test_cancel_before_connection_has_no_device_io(c_config, tmp_path, monkeypatch):
    import lerobot_robot_outcome_piper as plugin

    prepare_stub(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "cancel")
    monkeypatch.setattr(plugin, "OutcomePiper", lambda *a: pytest.fail("cancel constructed robot"))
    monkeypatch.setattr(
        plugin, "OutcomePiperXbox", lambda *a: pytest.fail("cancel constructed teleop")
    )
    path = tmp_path / "c.json"
    path.write_text(json.dumps(c_config))
    result = execution.execute(path, "run", tmp_path / "cancel.json")
    assert result["status"] == "cancelled_before_connection" and not result["hardware_io"]


def test_prepare_uses_reference_task_and_exact_closeout_recipe(c_config, tmp_path, monkeypatch):
    from dataclasses import asdict
    import draccus
    import numpy as np
    from lerobot.robots.config import RobotConfig
    from lerobot_robot_outcome_piper.timing import CaptureTiming
    from piper_policy.smolvla import SmolVLAChunkPredictor
    from piper_outcome_stack.task_c import select_task
    from test_plugin import config, xbox_config
    from test_policy_gripper_reference import SETTINGS

    robot = config(tmp_path, mode="motion")
    robot.capture_timing = CaptureTiming(0.2, 0.1, 0.05, 0.2)
    teleop = xbox_config(control_hz=50, gripper_reference=asdict(SETTINGS))
    monkeypatch.setattr(
        draccus, "decode", lambda kind, raw: robot if kind is RobotConfig else teleop
    )
    checkpoint, vlm, reference = tmp_path / "checkpoint", tmp_path / "vlm", tmp_path / "refs.npz"
    checkpoint.mkdir()
    vlm.mkdir()
    selected = select_task(c_config)
    from piper_outcome_stack import task_c_reference

    first = np.zeros((50, 7), dtype=np.float32)
    committed = np.arange(350, dtype=np.float32).reshape(50, 7)
    second = np.full((50, 7), 0.01, dtype=np.float32)
    projected = second.copy()
    projected[:, 6] = 0
    fixture = dict(
        tasks=np.array([selected["task"]]),
        inference_precision=np.array("fp32"),
        images=np.zeros((1, 480, 640, 3), np.uint8),
        states=np.zeros((1, 7)),
        seeds=np.array([7]),
        rtc_seeds=np.array([9]),
        expected_absolute_chunks=first[None],
        expected_rtc_chunks=second[None],
        expected_rtc_execution_chunks=projected[None],
    )
    np.savez(reference, **fixture)
    c_config["policy"] = {
        "checkpoint": str(checkpoint),
        "vlm_assets": str(vlm),
        "reference_inputs": {"push_to_G": str(reference)},
    }
    calls, seeds = [], []
    model = torch.nn.Linear(1, 1).double()
    model.config = NS(use_amp=True)
    model.reset = lambda: None

    def predict(image, state, **kw):
        assert next(model.parameters()).dtype == torch.float32 and not model.config.use_amp
        assert kw["gripper_lower_bound"] == "zero"
        seeds.append(torch.initial_seed())
        if "previous_model_actions" not in kw:
            return NS(actions=first, committed_model_actions=committed)
        np.testing.assert_array_equal(kw["previous_model_actions"], committed[25:])
        assert kw["inference_delay"] == 20 and kw["overlap_steps"] == 25
        return NS(actions=second, execution_actions=projected)

    predictor = NS(
        model=model, predict_chunk=predict, configure_rtc=lambda *a, **kw: calls.append(kw)
    )

    def load(path, *, task, vlm_assets):
        assert path == checkpoint and vlm_assets == vlm and task == selected["task"]
        return predictor

    monkeypatch.setattr(SmolVLAChunkPredictor, "from_checkpoint", load)
    monkeypatch.setattr(task_c_reference, "recorded_target_check", lambda *a: {"status": "passed"})
    with task_c_reference.fp32_runtime(), torch.random.fork_rng(devices=[]):
        *_, executor, report = execution.prepare_policy(c_config, selected, tmp_path)
        assert torch.initial_seed() == 42
    assert executor.joint_lowpass is False and executor.n_action_steps == 25
    assert calls[-1] == {"num_steps": 10, "guidance": "native_joint6"}
    assert seeds == [7, 9]
    row = report["reference_errors"][0]
    assert row["rtc_prefix_steps"] == 25 and row["rtc_commands_checked"] == 50
    assert row["rtc_error"]["joint_rad"] == 0 and row["rtc_execution_error"]["gripper_m"] == 0
    assert row["rtc_command_rejections"] == []

    # Failure in the second (prefixed) prediction must stop before the operator's motion entry.
    np.savez(reference, **(fixture | {"expected_rtc_chunks": second[None] + 0.01}))
    path = tmp_path / "c.json"
    path.write_text(json.dumps(c_config))
    monkeypatch.setattr(
        execution, "run_segment", lambda *a: pytest.fail("preflight allowed motion")
    )
    with pytest.raises(ValueError, match="reference mismatch: sample=0 RTC"):
        execution.execute(path, "run", tmp_path / "failed.json")
    assert not json.loads((tmp_path / "failed.json").read_text())["hardware_io"]

    np.savez(reference, tasks=np.array(["wrong task"]))
    with pytest.raises(ValueError, match="reference instructions differ"):
        execution.prepare_policy(c_config, selected, tmp_path)
    np.savez(reference, tasks=fixture["tasks"])
    with pytest.raises(ValueError, match="FP32 RTC reference"):
        execution.prepare_policy(c_config, selected, tmp_path)


def test_c_fp32_backend_settings_restore_after_failure():
    from piper_outcome_stack.task_c_reference import fp32_runtime

    def settings():
        return (
            torch.get_float32_matmul_precision(),
            torch.get_num_threads(),
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        )

    before = settings()
    with pytest.raises(RuntimeError, match="interrupted"), fp32_runtime():
        assert settings() == ("highest", 4, False, False, True)
        assert not torch.backends.cuda.matmul.allow_tf32
        raise RuntimeError("interrupted")
    assert settings() == before
