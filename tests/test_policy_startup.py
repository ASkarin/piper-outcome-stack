import pytest
from test_xbox_pause import session as session, moves  # noqa: F401
from test_plugin import xbox_config
from lerobot_robot_outcome_piper.robot import PiperState
from lerobot_robot_outcome_piper.teleop_control import TeleopState
from piper_outcome_stack.policy_startup import confirm_work_pose, prepare_work_pose


def config():
    return xbox_config(
        work_pose_button=4, work_joint_rad=(0.02, 0.0, 0.0, 0.0, 0.0, 0.0), work_gripper_m=0.015
    )


class Operator:
    def __init__(self, control, cancel=None):
        self.control, self.cancel, self.calls = control, cancel, 0

    def get_action(self):
        self.calls += 1
        assert self.calls < 3000, "A preparation did not terminate"
        moving = self.control.state is TeleopState.POSE_MOVING
        return dict(
            stick_x=0.0,
            stick_y=0.0,
            stick_z=0.0,
            stick_yaw=0.0,
            left_trigger=0.0,
            right_trigger=0.0,
            neutral=not (moving and self.cancel == "stick"),
            hold=moving and self.cancel == "LB",
            home=False,
            work=False,
            mode_switch=False,
            translation_switch=False,
            emergency_stop=moving and self.cancel == "B",
        )


def run(session, cancel=None):
    robot, arm, control, clock = session
    rows = []

    def sleep(dt):
        commands = moves(arm)
        if commands:
            arm.joints = list(commands[-1][1])
        if arm.gripper.commands:
            arm.gripper.width = arm.gripper.commands[-1][0]
        clock.advance(dt)

    result = prepare_work_pose(
        robot, Operator(control, cancel), control, config(), rows, clock=clock, sleep=sleep
    )
    return result, rows


def test_enter_confirmation_and_cancel():
    assert confirm_work_pose(config(), input_fn=lambda _: "")
    assert not confirm_work_pose(config(), input_fn=lambda _: "q")
    with pytest.raises(EOFError):
        confirm_work_pose(config(), input_fn=lambda _: (_ for _ in ()).throw(EOFError()))


def test_A_arrives_and_holds_without_starting_policy(session, monkeypatch):
    # Real RGB acquisition services this callback while waiting for a frame.
    robot = session[0]
    original = robot.get_observation

    def observation_with_camera_wait():
        robot._service_camera_wait()
        return original()

    monkeypatch.setattr(robot, "get_observation", observation_with_camera_wait)
    result, rows = run(session)
    robot, arm, control, _ = session
    assert result["status"] == "arrived_held" and result["hold_confirmed"]
    assert arm.joints == pytest.approx(config().work_joint_rad)
    assert arm.gripper.width == pytest.approx(config().work_gripper_m)
    assert robot.state is PiperState.ACTIVE
    assert all(r["phase"] in ("startup_A", "final_hold") for r in rows)
    # Automatic A's synthetic hold cannot authorize a following policy action.
    assert control.state is not TeleopState.RUNNING
    assert control.observe(True, True)[0] == "run"


@pytest.mark.parametrize("button", ["LB", "stick"])
def test_operator_can_cancel_auto_A_and_confirm_hold(session, button):
    result, rows = run(session, button)
    assert result["status"] == "cancelled_held" and result["hold_confirmed"]
    assert not any(r["phase"] == "policy_sent" for r in rows)


def test_B_during_A_uses_existing_stop(session):
    with pytest.raises(RuntimeError, match="Xbox B"):
        run(session, "B")
    assert session[0].state is PiperState.E_STOP
    assert "electronic_emergency_stop" in session[1].calls


@pytest.mark.parametrize("button", ["LB", "B"])
def test_camera_wait_keeps_real_cancel_and_B_active(session, monkeypatch, button):
    from piper_outcome_stack.policy_startup import WorkPoseInput

    robot, arm, control, clock = session
    # Verify the real camera wait path, not only the outer preparation loop.
    from test_xbox_pause import settle

    settle(session)
    control.state = TeleopState.POSE_MOVING
    inputs = WorkPoseInput(Operator(control, button), robot, control)
    robot.camera_input_poll = inputs.camera_poll
    if button == "B":
        with pytest.raises(RuntimeError, match="Xbox B"):
            robot._service_camera_wait()
        assert robot.state is PiperState.E_STOP
    else:
        robot._service_camera_wait()
        assert inputs.cancelled and control.state is TeleopState.HOLD_REQUESTED
