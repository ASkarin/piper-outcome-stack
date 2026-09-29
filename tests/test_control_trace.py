import json
import pytest
from lerobot_robot_outcome_piper.control_trace import ControlTrace
from lerobot_robot_outcome_piper.safety import ACTION_KEYS
from test_xbox_pause import session as session, tick, settle, moves  # noqa: F401


def test_buffer_does_no_file_writes_during_capture_and_copies_values(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = ControlTrace(path)
    event = {"event": "action", "target": [1.0, 2.0]}
    trace.append(event)
    event["target"][0] = 99.0
    assert path.stat().st_size == 0
    trace.save(stop_outcome=None)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["target"] == [1.0, 2.0]
    assert rows[-1]["trace_complete"] and rows[-1]["events"] == 1


def test_overflow_is_bounded_and_reported_without_changing_control(tmp_path):
    trace = ControlTrace(tmp_path / "trace.jsonl", max_events=2)
    for i in range(5):
        trace.append({"i": i})
    assert len(trace.events) == 2 and trace.dropped == 3
    trace.save()
    summary = json.loads(trace.path.read_text().splitlines()[-1])
    assert not summary["trace_complete"] and summary["dropped_events"] == 3


def test_existing_trace_is_not_overwritten(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        ControlTrace(path)
    assert path.read_text() == "existing"


def test_observation_send_and_hold_frames_retain_real_telemetry(session, tmp_path):
    robot, arm, control, clock = session
    trace = ControlTrace(tmp_path / "trace.jsonl")
    robot.control_trace = trace
    settle(session)
    tick(session, hold=True)
    clock.advance()
    action, returned = tick(session, hold=True)
    assert returned == dict(action)
    event = trace.events[-1]
    assert event["event"] == "action" and event["call_result"] == "returned"
    assert event["telemetry"]["values"] == returned
    assert event["telemetry"]["commands"][0]["result"] == "sdk_returned"
    obs = trace.events[-2]
    assert obs["event"] == "observation" and set(obs["values"]) == set(ACTION_KEYS)
    assert len(obs["feedback"]["received_monotonic_s"]) == 5
    tick(session, hold=False)
    clock.advance()
    tick(session)
    clock.advance()
    tick(session)
    before = len(moves(arm))
    clock.advance()
    tick(session)
    assert len(moves(arm)) == before  # Logging a paused frame does not send a new hold.
    assert trace.events[-1]["telemetry"]["commands"] == []
    robot.control_trace = None
    robot.disconnect()
    trace.save(stop_outcome=robot.stop_outcome)


def test_failed_dispatch_keeps_attempt_and_original_exception(session, tmp_path, monkeypatch):
    robot, arm, control, clock = session
    settle(session)
    tick(session, hold=True)
    trace = ControlTrace(tmp_path / "trace.jsonl")
    robot.control_trace = trace

    def fail(q):
        raise RuntimeError("injected motor write failure")

    monkeypatch.setattr(arm, "move_j", fail)
    with pytest.raises(Exception, match="injected motor write failure"):
        clock.advance()
        tick(session, hold=True)
    assert trace.events[-1]["call_result"] == "raised"
    assert trace.events[-1]["telemetry"]["commands"][0]["result"] == "failed"
    robot.control_trace = None
    robot.disconnect()
    trace.save(stop_outcome=robot.stop_outcome)
