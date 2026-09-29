from concurrent.futures import Future
from types import SimpleNamespace as NS
import time
import pytest
from lerobot_robot_outcome_piper.teleop_control import TeleopControl
from lerobot_robot_outcome_piper.record_control import EpisodeControls, run_interactive_episodes


def ready_future():
    future = Future()
    future.set_result(None)
    return future


class Terminal:
    def __init__(self, commands):
        self.commands = list(commands)

    def poll(self):
        return self.commands.pop(0) if self.commands else None


def test_episode_commands_wait_for_hold_and_do_not_allow_pose_during_recording():
    c = TeleopControl()
    c.gripper_target = 0.03  # Synthetic successful gripper command.
    events = {"exit_early": False}
    log = []
    t = Terminal(["start p1", "start p1", "end", "save failure slipped"])
    ui = EpisodeControls(c, events, t, lambda *a, **k: log.append((a, k)))
    ui.enter("preparing")
    ui.poll()
    assert ui.request is None
    c.confirm_hold()
    ui.poll()
    assert ui.request == ("start",) and ui.position == "p1"
    ui.enter("recording")
    ui.poll()
    assert ui.request == ("end",)
    c.observe(False, True)
    c.observe(False, True, home=True)
    assert c.pose_event == "request_rejected"
    ui.enter("review")
    c.confirm_hold()
    ui.poll()
    assert ui.request == ("save", "failure", "slipped")


def test_save_worker_keeps_loop_alive_and_prep_does_not_write_frames():
    c = TeleopControl()
    assert c.gripper_target is None
    events = {"exit_early": False}
    log = []
    phases = []

    class T:
        sent = set()

        def poll(self):
            phase = c.recording_phase
            if not c.hold_confirmed:
                return None
            if phase == "preparing" and "start" not in self.sent:
                self.sent.add("start")
                return "start p1"
            if phase == "recording" and "end" not in self.sent:
                self.sent.add("end")
                return "end"
            if phase == "review" and "save" not in self.sent:
                self.sent.add("save")
                return "save success"
            return None

    class Audit:
        attempt = 0
        pending = []

        def __init__(self):
            self.dataset = NS(num_episodes=0, finalize=lambda: None)

        def emit(self, *a, **k):
            log.append((a, k))

        def prepare_episode(self):
            return ready_future()

        def check_writer(self):
            pass

        def flush(self):
            pass

        def save_episode(self):
            time.sleep(0.03)
            self.dataset.num_episodes += 1
            self.pending = []
            self.attempt += 1

        def discard_episode(self):
            pytest.fail("unexpected discard")

    audit = Audit()

    def processor(pair):
        return pair[0]

    def loop(**kw):
        for _ in range(5000):
            if kw["events"]["exit_early"]:
                kw["events"]["exit_early"] = False
                return
            c.confirm_hold()
            phases.append(c.recording_phase)
            kw["teleop_action_processor"](({"hold": False, "neutral": True}, {}))
            if kw["dataset"] is not None:
                audit.pending.append(1)
                assert c.recording_phase == "recording"
            time.sleep(0.0001)
        pytest.fail("loop did not finish")

    class Manager:
        def __init__(self, ds):
            self.ds = ds

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.ds.finalize()

    def acquire(epoch):
        assert c.recording_phase == "preparing" and epoch == c.epoch
        assert c.gripper_target is None
        c.gripper_target = 0.0
        return {"initialized": True}

    official = NS(record_loop=loop, VideoEncodingManager=Manager)
    ds = NS(num_episodes=1, episode_time_s=60)
    run_interactive_episodes(
        official,
        dict(
            events=events,
            teleop_action_processor=processor,
            robot=NS(prepare_recording_gripper=acquire),
        ),
        audit,
        ds,
        c,
        T(),
    )
    assert phases.count("saving") > 3
    assert audit.dataset.num_episodes == 1
    assert any(k.get("task_outcome") == "success" for _, k in log)
    assert c.recording_phase is None


def test_terminal_requires_interactive_stream():
    # B precedence is exercised in the real processor tests as well; the terminal
    # may fail, but must not be consulted before a supplied raw B event.
    from lerobot_robot_outcome_piper.record_control import TerminalCommands

    with pytest.raises(ValueError, match="operator terminal"):
        TerminalCommands(NS(isatty=lambda: False))


def test_raw_b_preempts_terminal_failure():
    c = TeleopControl()
    c.gripper_target = 0.03  # Synthetic successful gripper command.

    class TerminalFailure:
        def poll(self):
            pytest.fail("B must be forwarded before consulting terminal")

    def processor(value):
        assert value[0]["emergency_stop"]
        raise RuntimeError("B handled by normal processor")

    def loop(**kwargs):
        kwargs["teleop_action_processor"](({"emergency_stop": True}, {}))

    audit = NS(emit=lambda *a, **k: None, prepare_episode=ready_future, check_writer=lambda: None)
    official = NS(record_loop=loop)
    with pytest.raises(RuntimeError, match="B handled"):
        run_interactive_episodes(
            official,
            dict(events={}, teleop_action_processor=processor),
            audit,
            NS(num_episodes=1),
            c,
            TerminalFailure(),
        )
    assert c.recording_phase is None


def test_pending_pose_cannot_start_recording_but_can_quit():
    from lerobot_robot_outcome_piper.teleop_control import TeleopState

    c = TeleopControl()
    c.gripper_target = 0.03  # Synthetic successful gripper command.
    ui = EpisodeControls(
        c, {"exit_early": False}, Terminal(["start p1", "quit"]), lambda *a, **k: None
    )
    ui.enter("preparing")
    c.state = TeleopState.POSE_READY
    c.hold_confirmed = True
    ui.poll()
    assert ui.request is None
    ui.poll()
    assert ui.request == ("quit",)


def test_operator_prompts_match_bound_keys_duration_and_phase(capsys):
    for available in (False, True):
        c = TeleopControl()
        c.gripper_target = 0.03  # Synthetic successful gripper command.
        ui = EpisodeControls(
            c, {}, Terminal([]), lambda *a, **k: None, work_available=available, duration_s=10
        )
        ui.enter("preparing")
        text = capsys.readouterr().out
        assert ("A进入工作姿态" in text) is available
        assert "Y回零" in text
        ui.position = "debug1"
        ui.enter("recording")
        text = capsys.readouterr().out
        assert "10 秒" in text and "暂停期间仍计时" in text
        ui.enter("saving")
        text = capsys.readouterr().out
        assert "不接受遥操作" in text and "B急停仍有效" in text
        assert "按LB" not in text


def test_rejection_explains_next_operator_step(capsys):
    c = TeleopControl()
    c.gripper_target = 0.03  # Synthetic successful gripper command.
    terminal = Terminal(["save success", "start", "save success"])
    ui = EpisodeControls(c, {}, terminal, lambda *a, **k: None)
    ui.enter("recording")
    capsys.readouterr()
    ui.poll()
    assert "请先输入 end" in capsys.readouterr().out
    ui.enter("preparing")
    c.confirm_hold()
    capsys.readouterr()
    ui.poll()
    assert "格式：start" in capsys.readouterr().out
    ui.enter("review")
    c.hold_confirmed = False
    capsys.readouterr()
    ui.poll()
    assert "等待‘保持已确认’" in capsys.readouterr().out


def test_start_accepts_initial_gripper_reference_request(capsys):
    c = TeleopControl()
    events = {"exit_early": False}
    ui = EpisodeControls(
        c, events, Terminal(["start save-test", "start save-test"]), lambda *a, **k: None
    )
    ui.enter("preparing")
    c.confirm_hold()
    ui.poll()
    assert ui.request == ("start",) and events["exit_early"]
    assert "首次 start" in capsys.readouterr().out
    assert c.gripper_target is None  # Terminal parsing does not invent a command.


@pytest.mark.parametrize(
    "raw",
    [
        {"hold": True, "neutral": True},
        {"hold": False, "neutral": False},
        {"hold": False, "neutral": True, "mode_switch": True},
        {"hold": False, "neutral": True, "home": True},
    ],
)
def test_start_never_queues_with_active_inputs(raw):
    c = TeleopControl()
    events = {}

    def loop(**kwargs):
        c.confirm_hold()
        kwargs["teleop_action_processor"]((raw, {}))
        assert not events["exit_early"]
        raise RuntimeError("probe complete")

    robot = NS(prepare_recording_gripper=lambda epoch: pytest.fail("rejected start must not send"))
    with pytest.raises(RuntimeError, match="probe complete"):
        run_interactive_episodes(
            NS(record_loop=loop),
            dict(events=events, teleop_action_processor=lambda value: value[0], robot=robot),
            NS(emit=lambda *a, **k: None, prepare_episode=ready_future, check_writer=lambda: None),
            NS(num_episodes=1),
            c,
            Terminal(["start P1"]),
        )
