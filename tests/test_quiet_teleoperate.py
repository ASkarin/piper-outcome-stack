"""Compare official loop dispatch with fake devices; no hardware or stdout filtering."""

from types import SimpleNamespace as NS
import logging
import json
import sys
from pathlib import Path

import pytest
from lerobot.scripts import lerobot_record, lerobot_teleoperate
from lerobot_robot_outcome_piper import workflows

sys.path.insert(0, str(Path(__file__).parents[1] / "infra/acceptance"))
from lerobot_dataset_replay_smoke import FakeRobot, FakeTeleoperator, ACTION_NAMES  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def execute(monkeypatch):
    from lerobot.utils import utils, visualization_utils

    monkeypatch.setattr(utils, "init_logging", lambda: None)
    monkeypatch.setattr(visualization_utils, "init_visualization", lambda *a, **kw: None)
    monkeypatch.setattr(visualization_utils, "shutdown_visualization", lambda *a: None)
    original_record_loop = lerobot_record.record_loop

    def run(
        *,
        legacy=False,
        duration=0.11,
        display=False,
        fault=None,
        process_time=0.005,
        trace_path=None,
    ):
        clock = Clock()
        trace, visual, invocations = [], [], []
        monkeypatch.setattr(lerobot_record.time, "perf_counter", clock.time)
        monkeypatch.setattr(lerobot_record, "precise_sleep", clock.sleep)
        monkeypatch.setattr(lerobot_teleoperate, "precise_sleep", clock.sleep)
        for module in (lerobot_record, lerobot_teleoperate):
            monkeypatch.setattr(
                module, "log_visualization_data", lambda *a, **kw: visual.append(kw)
            )

        class Robot(FakeRobot):
            stop_outcome = None
            stop_error = None

            def disconnect(self):
                if trace_path is not None:
                    assert Path(trace_path).stat().st_size == 0
                return super().disconnect()

            def configure_teleoperation(self, *args):
                pass

            def enable(self):
                trace.append("enable")

            def get_observation(self):
                trace.append("observation")
                if self._observation_count == 3 and duration is None:
                    raise KeyboardInterrupt
                return super().get_observation()

            def send_action(self, action):
                trace.append("send")
                clock.now += process_time
                return super().send_action(action)

        class Teleop(FakeTeleoperator):
            def get_action(self):
                trace.append("input")
                return {**super().get_action(), "emergency_stop": False}

        class Processor:
            steps = [NS(control=object())]

            def __call__(self, pair):
                trace.append("teleop_processor")
                print("操作提示：松开LB保持")
                if fault:
                    logging.warning("输入拒绝：%s", fault)
                    raise RuntimeError(fault)
                return {key: float(pair[0][key]) for key in ACTION_NAMES}

        def action_processor(pair):
            trace.append("action_processor")
            return pair[0]

        def observation_processor(obs):
            return obs

        robot, teleop, processor = Robot(), Teleop(), Processor()
        robot_config = NS()
        teleop_config = NS(control_hz=20, hold_settings=lambda: object())
        cfg = NS(
            robot=robot_config,
            teleop=teleop_config,
            fps=20,
            display_data=display,
            display_mode="rerun",
            display_ip=None,
            display_port=None,
            display_compressed_images=False,
            teleop_time_s=duration,
            control_trace_path=trace_path,
        )
        monkeypatch.setattr(
            workflows, "_validate_workflow_configs", lambda *a: (robot_config, teleop_config)
        )
        monkeypatch.setattr(workflows, "_processor", lambda *a: processor)
        monkeypatch.setattr(workflows, "make_robot_from_config", lambda *a: robot)
        monkeypatch.setattr(workflows, "make_teleoperator_from_config", lambda *a: teleop)
        monkeypatch.setattr(
            workflows,
            "make_default_processors",
            lambda: (None, action_processor, observation_processor),
        )

        def loop(**kwargs):
            invocations.append(kwargs)
            assert kwargs["dataset"] is None and kwargs["fps"] == 20
            if legacy:
                return lerobot_teleoperate.teleop_loop(
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ("dataset", "events", "control_time_s")
                    },
                    duration=duration,
                )
            return original_record_loop(**kwargs)

        monkeypatch.setattr(lerobot_record, "record_loop", loop)
        try:
            workflows.teleoperate(cfg)
        finally:
            assert not robot.is_connected and not teleop.is_connected
        return robot.actions, trace, clock.sleeps, visual, invocations

    return run


def test_trace_save_is_after_disconnect_without_changing_loop(execute, tmp_path):
    baseline = execute()
    path = tmp_path / "control.jsonl"
    traced = execute(trace_path=str(path))
    assert baseline[:3] == traced[:3]
    summary = json.loads(path.read_text().splitlines()[-1])
    assert summary["event"] == "summary" and summary["trace_complete"]


@pytest.mark.parametrize("duration", [0.11, 0.0, -1.0, None])
@pytest.mark.parametrize("display", [False, True])
def test_quiet_loop_preserves_dispatch_cadence_duration_and_visuals(
    execute, capsys, duration, display
):
    old = execute(legacy=True, duration=duration, display=display)
    capsys.readouterr()
    new = execute(duration=duration, display=display)
    output = capsys.readouterr().out
    assert new[:3] == old[:3]  # Targets, core call order and scheduled sleep durations.
    assert len(new[3]) == len(old[3])
    assert len(new[0]) == (3 if duration in (0.11, None) else 1)
    assert "操作提示：松开LB保持" in output
    assert "Teleop loop time" not in output and "\x1b[" not in output
    assert "NAME" not in output and "NORM" not in output


def test_errors_propagate_and_console_warnings_remain(execute, caplog, capsys):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="synthetic failure"):
            execute(fault="synthetic failure")
    assert "输入拒绝：synthetic failure" in caplog.text
    assert "操作提示" in capsys.readouterr().out


def test_actual_slow_loop_warning_is_not_suppressed(execute, caplog):
    with caplog.at_level(logging.WARNING):
        execute(process_time=0.08)
    assert "running slower" in caplog.text
