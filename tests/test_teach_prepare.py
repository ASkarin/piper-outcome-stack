from lerobot_robot_outcome_piper.raw_io import write_json
from types import SimpleNamespace as NS
import math
import time
import pytest

from test_plugin import make_robot, connect_for_test
from test_xbox_pause import Clock, Receiver
from test_teach_recording import config, sample, safety as safety  # noqa: F401
from lerobot_robot_outcome_piper.teach_prepare import TeachPreparation, normalize_preparation
from lerobot_robot_outcome_piper.teach_record import Attempt, run_session
from lerobot_robot_outcome_piper.teach_source import check_feedback
from lerobot_robot_outcome_piper.teach_data import SCHEMA, read_json, validate_sample
from lerobot_robot_outcome_piper.teach_dataset import convert
from lerobot_robot_outcome_piper.joint_pose import JointPoseSequence
from lerobot_robot_outcome_piper.safety import JOINT_KEYS


def motion_fixture(tmp_path):
    robot, arm, _ = make_robot(tmp_path, mode="motion")
    clock = Clock()
    robot._monotonic = clock
    robot._receiver_factory = Receiver
    robot._start_watchdog = lambda: None
    settings = {"joint_tolerance_rad": math.radians(0.1), "stable_time_s": 0.3, "timeout_s": 10.0}
    normalized = normalize_preparation(
        {
            "work_joint_rad": [0.1, 0.2, -0.2, 0.1, 0.1, 0.1],
            "safety_path": str(robot.config.safety_path),
            "hold_settings": settings,
        }
    )
    old_status = arm.get_arm_status

    def status():
        result = old_status()
        result.msg.teach_status = 2
        return result

    arm.get_arm_status = status
    order = []

    def factory(cfg):
        order.append("motion_created")
        robot.config = cfg
        return robot

    def sleep(dt):
        clock.advance(dt)
        commands = [x[1] for x in arm.calls if isinstance(x, tuple) and x[0] == "move_j"]
        if commands:
            arm.joints = list(commands[-1])

    preparation = TeachPreparation(
        robot.config, normalized, tmp_path, robot_factory=factory, clock=clock, sleep=sleep
    )

    def reconnect(**kwargs):
        assert not robot.is_connected
        order.append("receive_connected")

    source = NS(
        feedback=lambda **kw: dict(teach_status=2, arm_status=0),
        disconnect=lambda: order.append("receive_disconnected"),
        connect=reconnect,
    )
    return preparation, robot, arm, clock, source, order


def test_preparation_handover_and_no_gripper_commands(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)
    arm.gripper.width = 0.04
    out = p.perform(source, NS(poll=lambda: None))
    assert out["status"] == "arrived"
    assert order == ["receive_disconnected", "motion_created", "receive_connected"]
    assert not arm.gripper.commands
    assert robot._last_gripper_target is None and robot._last_gripper_command is None
    assert not any(x in ("disable", "reset", "home") for x in arm.calls if isinstance(x, str))
    assert list(tmp_path.glob("preparation-*/result.json"))


def test_teaching_active_rejects_without_handover_or_control(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)
    source.feedback = lambda **kw: dict(teach_status=1, arm_status=0)
    assert p.perform(source, NS(poll=lambda: None))["status"] == "rejected"
    assert not order and not arm.calls


def test_cancel_motion_holds_without_gripper_or_estop(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)
    out = p.perform(source, NS(poll=lambda: "stop"))
    assert out["status"] == "cancelled" and "held" in out
    assert robot.stop_outcome is None and not arm.gripper.commands


def test_joint_only_api_keeps_existing_gripper_command(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)
    connect_for_test(robot)
    robot.get_observation()
    robot._last_gripper_target = 0.025
    command = {"target": 0.025, "force": 1.0}
    robot._last_gripper_command = dict(command)
    arm.gripper.width = 0.05
    goal = list(arm.joints)
    goal[0] += 0.001
    result = robot.send_joint_target(goal)
    assert set(result) == set(JOINT_KEYS)
    assert not arm.gripper.commands
    assert robot._last_gripper_target == 0.025 and robot._last_gripper_command == command
    assert robot.last_action_telemetry["gripper_commanded"] is False
    robot.disconnect()


def test_teaching_restarted_during_handover_blocks_enable(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)
    old = arm.get_arm_status

    def status():
        value = old()
        value.msg.teach_status = 1
        return value

    arm.get_arm_status = status
    result = p.perform(source, NS(poll=lambda: None))
    assert result["status"] == "rejected" and not p.control_commands_attempted
    assert not any(isinstance(x, tuple) and x[0] == "move_j" for x in arm.calls)


def test_interrupt_in_preparation_stops_and_does_not_restart_receiver(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)

    def interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        p.perform(source, NS(poll=interrupt))
    assert robot.stop_outcome is not None and "receive_connected" not in order
    assert not arm.gripper.commands


def test_missing_status_after_handover_is_fault_not_mode_retry(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)
    original = robot.get_observation
    reads = [0]

    def observe():
        value = original()
        reads[0] += 1
        if reads[0] == 2:
            arm.get_arm_status = lambda: None
        return value

    robot.get_observation = observe
    with pytest.raises(RuntimeError, match="status unavailable"):
        p.perform(source, NS(poll=lambda: None))
    assert not p.control_commands_attempted and "receive_connected" not in order


def test_joint_only_reference_ignores_gripper_motion_intent(tmp_path):
    p, robot, arm, clock, source, order = motion_fixture(tmp_path)
    seq = JointPoseSequence([0.0] * 6, 0.04, p.safety, p.settings, [0.02] * 6, "prepare")
    assert set(seq.values) == set(JOINT_KEYS)
    assert seq.telemetry()["gripper_completion"] == "not_commanded"
    seq.sent(1.0)
    seq.observe([0.0] * 6, (1.1,) * 3, 1.1, 0.09)
    assert seq.advance_ready  # Observed opening is not a target to be commanded.


def test_idle_feedback_allowed_only_outside_raw_sampling():
    _, row = sample(0)
    row.update(ctrl_mode=1, teach_status=0)
    check_feedback(row, 0.2, require_teach=False)
    with pytest.raises(RuntimeError, match="unexpected teach mode"):
        check_feedback(row, 0.2)
    with pytest.raises(RuntimeError):
        validate_sample(row, None, config())


@pytest.mark.parametrize("direct", [False, True])
def test_save_then_reconfirm_and_second_attempt_convert(
    tmp_path, monkeypatch, safety, direct, capsys
):
    cfg = config()
    root = tmp_path / "raw"
    root.mkdir()
    write_json(root / "session.json", dict(schema=SCHEMA, source="manual_teach", config=cfg))
    clock = Clock()
    state = {
        "mode": 1,
        "teach": 0,
        "n": 0,
        "saved": 0,
        "phase": "cancel",
        "preparations": 0,
        "prompts": 0,
    }
    from lerobot_robot_outcome_piper import teach_prepare

    monkeypatch.setattr(
        teach_prepare, "prompt_preparation", lambda goal: state.update(prompts=state["prompts"] + 1)
    )
    original_save = Attempt.save

    def save(self, *args):
        original_save(self, *args)
        state["saved"] += 1

    monkeypatch.setattr(Attempt, "save", save)

    def feedback(**kw):
        row = sample(state["n"], clock())[1]
        row.update(ctrl_mode=state["mode"], teach_status=state["teach"])
        if kw.get("require_teach"):
            check_feedback(row, 0.2)
        return row

    def read():
        assert state["preparations"] >= 1
        value = sample(state["n"], clock())
        state["n"] += 1
        return value

    source = NS(feedback=feedback, read=read)

    def prepare(*args):
        state["preparations"] += 1
        state["mode"] = 1
        state["teach"] = 0
        return {"status": "arrived"}

    preparation = NS(goal=[0.0] * 6, perform=prepare)

    def poll():
        phase = state["phase"]
        if phase == "cancel":
            state["phase"] = "early_start"
            return "cancel"
        if phase == "early_start":
            state["phase"] = "prepare"
            return "start ignored"
        if phase == "prepare":
            state["phase"] = "start"
            return "prepare"
        if phase == "start":
            state.update(mode=2, teach=1, n=0, phase="record")
            return "start P" + str(state["saved"] + 1)
        if phase == "record" and state["n"] >= 3:
            state["phase"] = "save"
            return "end"
        if phase == "save":
            state["phase"] = "saving"
            return "save success"
        if phase == "saving" and "[下一回合]" in capsys.readouterr().out:
            state["phase"] = "after_saved"
            return None  # Let the main loop consume job.done().
        if phase == "after_saved":
            if state["saved"] == 2:
                return "quit"
            if direct:
                state["phase"] = "start"
                return None
            state["phase"] = "stop_teach"
            return "prepare"  # Still teaching: must not move.
        if phase == "stop_teach":
            state.update(mode=2, teach=2, phase="prepare")
            return None
        return None

    def sleep(dt):
        clock.advance(dt)
        time.sleep(0.001)

    assert (
        run_session(
            source, cfg, root, NS(poll=poll), clock=clock, sleep=sleep, preparation=preparation
        )
        == 2
    )
    attempts = list(root.glob("attempt-*"))
    assert len(attempts) == 2 and state["preparations"] == (1 if direct else 2)
    assert sorted(read_json(p / "result.json")["position_id"] for p in attempts) == ["P1", "P2"]
    assert all(read_json(p / "result.json")["frames"] == 3 for p in attempts)
    assert convert(root, tmp_path / "dataset", "local/prepared-teach", safety, video=False) == {
        "episodes": 2,
        "frames": 4,
    }
