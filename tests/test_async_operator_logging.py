import io
import logging
import threading
import pytest
from test_xbox_pause import session as session
from types import SimpleNamespace as NS
from lerobot_robot_outcome_piper.console import AsyncOperatorHandler, _active_output
from lerobot_robot_outcome_piper.errors import OutcomePiperLogError
from lerobot_robot_outcome_piper.control_trace import PreparationTimingTrace


def record(message="test", level=logging.INFO):
    return logging.LogRecord("root", level, __file__, 1, message, (), None)


def test_slow_output_does_not_block_producer_and_preserves_warning_order():
    entered = threading.Event()
    release = threading.Event()
    values = []

    class Slow(logging.Handler):
        def emit(self, value):
            entered.set()
            assert release.wait(3)
            values.append(value.getMessage())

    sink = Slow()
    queue = AsyncOperatorHandler([sink], capacity=4)
    try:
        queue.handle(record("first"))
        assert entered.wait(1)
        queue.handle(record("warning", logging.WARNING))
        assert not release.is_set() and queue.records.qsize() == 1
    finally:
        release.set()
        summary = queue.finish()
    assert values == ["first", "warning"] and not summary["error"]
    assert summary["sink_timings"]["console_0"]["max_wall_s"] > 0
    assert not queue.worker.is_alive()


def test_bounded_overflow_is_explicit_and_never_raises_inside_stop_logging():
    entered = threading.Event()
    release = threading.Event()

    class Slow(logging.Handler):
        def emit(self, value):
            entered.set()
            assert release.wait(3)

    queue = AsyncOperatorHandler([Slow()], capacity=1)
    try:
        queue.handle(record())
        assert entered.wait(1)
        queue.handle(record())
        queue.handle(record("B requested", logging.ERROR))
        with pytest.raises(OutcomePiperLogError, match="queue"):
            queue.check_health()
        assert queue.records.qsize() == 1
    finally:
        release.set()
        summary = queue.finish()
    assert summary["rejected"] == 1 and summary["processed"] == 2


def test_builtin_stream_error_is_observable_and_other_sink_still_receives_record():
    class Broken:
        def write(self, value):
            raise BrokenPipeError("terminal disconnected")

        def flush(self):
            pass

    good = io.StringIO()
    broken = logging.StreamHandler(Broken())
    queue = AsyncOperatorHandler([broken, logging.StreamHandler(good)])
    original = queue.old_errors[0][1]
    queue.handle(record("preserved warning", logging.WARNING))
    summary = queue.finish()
    assert "terminal disconnected" in summary["error"]
    assert "preserved warning" in good.getvalue()
    assert broken.handleError == original


def test_mutable_message_arguments_are_frozen_before_enqueue():
    entered = threading.Event()
    release = threading.Event()
    stream = io.StringIO()

    class Slow(logging.StreamHandler):
        def emit(self, value):
            entered.set()
            assert release.wait(3)
            super().emit(value)

    queue = AsyncOperatorHandler([Slow(stream)])
    value = [1]
    event = record("values=%s")
    event.args = (value,)
    try:
        queue.handle(event)
        assert entered.wait(1)
        value[0] = 99
    finally:
        release.set()
        queue.finish()
    assert "[1]" in stream.getvalue() and "99" not in stream.getvalue()


def test_output_failure_uses_hold_path_but_b_retains_priority(monkeypatch):
    from test_processor import processor, raw_action
    from lerobot_robot_outcome_piper import processor as module
    from lerobot_robot_outcome_piper.teleop_control import TeleopState

    queue = NS(check_health=lambda: (_ for _ in ()).throw(OutcomePiperLogError("output blocked")))
    token = _active_output.set(queue)
    held = []
    stopped = []
    monkeypatch.setattr(module, "request_input_fault_hold", lambda exc: held.append(str(exc)))
    monkeypatch.setattr(
        module, "request_input_emergency_stop", lambda exc: stopped.append(str(exc))
    )
    try:
        p = processor()
        with pytest.raises(OutcomePiperLogError):
            p.action(raw_action())
        assert held and not stopped
        with pytest.raises(Exception, match="emergency stop"):
            p.action({**raw_action(), "emergency_stop": True})
        assert stopped and p.control.state is TeleopState.E_STOP
    finally:
        _active_output.reset(token)


def test_preparation_trace_retains_pose_and_timing_without_dataset_rows():
    emitted = []
    trace = PreparationTimingTrace(lambda event, **value: emitted.append((event, value)))
    observation = dict(
        event="observation", sequence=1, monotonic_s=1.0, values={"joint_1.pos": 0.1}
    )
    trace.append(observation)
    observation["values"]["joint_1.pos"] = 9
    trace.append(
        dict(
            event="action",
            recording_phase="preparing",
            intent="pose",
            stage_timing={"sdk_move_j": {"wall_s": 0.001}},
            telemetry={"pose_plan": {"waypoint": 1}},
        )
    )
    assert emitted[0][0] == "preparation_tick"
    assert emitted[0][1]["observation"]["values"]["joint_1.pos"] == 0.1
    assert emitted[0][1]["stage_timing"] and emitted[0][1]["telemetry"]["pose_plan"]


def test_real_robot_trace_covers_preparation_actions_and_restores_without_motion_changes(session):
    from test_xbox_pause import settle, tick

    robot, arm, control, clock = session
    events = []
    robot.control_trace = PreparationTimingTrace(lambda name, **value: events.append((name, value)))
    control.recording_phase = "preparing"
    settle(session)
    tick(session)
    assert events and all(name == "preparation_tick" for name, _ in events)
    row = events[-1][1]
    assert row["recording_phase"] == "preparing" and row["observation"]["sequence"]
    assert row["stage_timing"]["feedback"] and row["telemetry"]["hold_command"]
    robot.control_trace = None


def test_log_after_shutdown_is_reported_not_silently_accepted():
    queue = AsyncOperatorHandler([logging.StreamHandler(io.StringIO())])
    queue.finish()
    queue.handle(record())
    with pytest.raises(OutcomePiperLogError, match="shutdown"):
        queue.check_health()
