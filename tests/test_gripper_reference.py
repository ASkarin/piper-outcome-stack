from dataclasses import asdict, replace
import io
import logging
import pytest

from lerobot_robot_outcome_piper.gripper_reference import (
    GripperReferenceSettings,
    gripper_candidate,
)
from lerobot_robot_outcome_piper.processor import OutcomePiperAction
from test_xbox_pause import session as session, settle
from test_plugin import valid_action, xbox_config

S = GripperReferenceSettings(0.02, 0.12, 1.2, 0.008)


def candidate(previous=None, target=0.03, feedback=0.03, command=1, now=0, settings=S):
    return gripper_candidate(previous, target, feedback, command, settings, now, 0, 0.12)


def test_equal_wall_time_at_20_and_50_hz():
    results = []
    for hz in (20, 50):
        s = replace(S, period_s=1 / hz)
        previous, target = None, 0.01
        for i in range(round(0.5 * hz)):
            previous = candidate(previous, target, target, now=i / hz, settings=s)
            target = previous["target_m"]
        results.append(target)
    assert results == pytest.approx([0.064, 0.064])


def test_lag_boundary_release_reversal_and_no_catchup():
    prev, target = None, 0.03
    for i in range(100):
        prev = candidate(prev, target, now=i * 0.02)
        target = prev["target_m"]
    assert target == pytest.approx(0.038)
    assert prev["velocity_m_s"] == 0 and prev["travel_limited"]
    released = candidate(prev, target, command=0, now=10)
    assert released["target_m"] == target and released["velocity_m_s"] == 0
    resumed = candidate(released, target, feedback=target, now=20)
    assert resumed["dt_s"] == 0.02
    assert resumed["target_m"] - target == pytest.approx(0.00024)
    reversed_ = candidate(resumed, resumed["target_m"], feedback=target, command=-1, now=20.02)
    assert reversed_["target_m"] < resumed["target_m"]
    assert candidate(target=0.1199, feedback=0.1199)["target_m"] == 0.12
    assert candidate(target=0.0001, feedback=0.0001, command=-1)["target_m"] == 0
    # A gripping target is retained even if the object prevents reaching it.
    assert candidate(target=0.01, feedback=0.03, command=0)["target_m"] == 0.01
    with pytest.raises(ValueError, match="monotonic"):
        candidate(resumed, now=19)


@pytest.mark.parametrize(
    "outcome", ["success", "failure", "gripper_failure", "release", "stale", "B"]
)
@pytest.mark.parametrize("center", [False, True])
def test_only_commit_dispatched_current_generation(session, outcome, center):
    robot, arm, c, clock = session
    settle(session)
    _, epoch = c.observe(True, True)
    if center:
        c.arm_input_intent(epoch, True)
        intent, epoch = c.arm_input_intent(epoch, False)
    else:
        intent = "run"
    a = OutcomePiperAction(valid_action(0.01, 0.035), intent=intent, epoch=epoch)
    a.generated_monotonic_s = clock.now
    a.gripper_input = True
    a.gripper_plan = {
        "base_revision": c.gripper_reference_revision,
        "reference": dict(target_m=0.035, velocity_m_s=0.02, time_s=clock.now),
    }
    if outcome == "failure":
        # Both center hold and arm motion use move_j.
        arm.move_j = lambda *a: (_ for _ in ()).throw(OSError("SDK failed"))
    if outcome == "gripper_failure":
        arm.gripper.move_gripper_m = lambda *a, **kw: (_ for _ in ()).throw(OSError("SDK failed"))
    if outcome == "release":
        original = arm.gripper.move_gripper_m

        def release(*args, **kwargs):
            original(*args, **kwargs)
            c.request_hold()

        arm.gripper.move_gripper_m = release
    if outcome == "stale":
        c.request_hold()
    if outcome == "B":
        robot.request_emergency_stop("B")
    if outcome in ("failure", "gripper_failure", "B"):
        with pytest.raises(RuntimeError):
            robot.send_action(a)
    else:
        robot.send_action(a)
    assert (c.gripper_reference_state is not None) == (outcome == "success")


def test_config_roundtrip_processor_seed_and_reset():
    from test_processor import processor
    from lerobot_robot_outcome_piper.processor import OutcomePiperXboxProcessor

    cfg = replace(xbox_config(), gripper_reference=asdict(replace(S, period_s=0.05)))
    with pytest.raises(ValueError, match="control_hz"):
        replace(cfg, gripper_reference=asdict(S))
    p = processor(gripper_reference=asdict(S))
    restored = OutcomePiperXboxProcessor(**p.get_config())
    assert restored.gripper_reference == asdict(S)
    p.control.gripper_target = 0.034
    target, plan = p._plan_gripper(0.03, p.max_gripper_step_m)
    assert target == pytest.approx(0.03424)
    assert p.control.gripper_reference_state is None
    assert p.control.commit_gripper_reference(p.control.epoch, plan)
    assert not p.control.gripper_reference_valid(p.control.epoch, plan)
    p.reset()
    assert p.control.gripper_reference_state is None
    assert not p.control.gripper_reference_valid(p.control.epoch, plan)


def test_preparation_pose_warning_console_only_and_resets(tmp_path):
    from lerobot_robot_outcome_piper.console import AsyncOperatorHandler, OperatorInfoFilter
    from lerobot_robot_outcome_piper.stage_timing import reset, mark_preparation_pose

    text = io.StringIO()
    console = logging.StreamHandler(text)
    console.addFilter(OperatorInfoFilter())
    file = logging.FileHandler(tmp_path / "operator.log")
    output = AsyncOperatorHandler([console, file])

    def emit(message, path="lerobot_record.py"):
        output.handle(logging.LogRecord("root", logging.WARNING, path, 1, message, (), None))

    try:
        reset()
        mark_preparation_pose(True)
        emit("Record loop is running slower (14.2 Hz) pose")
        emit("feedback timeout")
        emit("Record loop is running slower (other source)", "other.py")
        reset()
        emit("Record loop is running slower (recording)")
    finally:
        output.finish()
        file.close()
        reset()
    assert "14.2 Hz" not in text.getvalue()
    assert "feedback timeout" in text.getvalue() and "other source" in text.getvalue()
    assert "recording" in text.getvalue()
    assert "14.2 Hz" in (tmp_path / "operator.log").read_text()
