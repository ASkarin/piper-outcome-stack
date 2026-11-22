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


@pytest.mark.parametrize("tasks", [None, {"1": "green left", "2": "blue right"}])
def test_save_worker_keeps_loop_alive_and_prep_does_not_write_frames(tasks):
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
            if tasks and phase == "preparing" and audit.dataset.num_episodes < 2:
                return str(audit.dataset.num_episodes + 1)
            if tasks and phase == "recording":
                return ""
            if tasks and phase == "review":
                return "1"
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
                if tasks:
                    assert kw["single_task"] == tasks[str(audit.dataset.num_episodes + 1)]
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
        if not tasks or audit.dataset.num_episodes == 0:
            assert c.gripper_target is None
        c.gripper_target = 0.0
        return {"initialized": True}

    official = NS(record_loop=loop, VideoEncodingManager=Manager)
    ds = NS(num_episodes=2 if tasks else 1, episode_time_s=60)
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
        tasks=tasks,
        position="B-layout1" if tasks else None,
    )
    assert phases.count("saving") > 3
    assert audit.dataset.num_episodes == (2 if tasks else 1)
    assert any(k.get("task_outcome") == "success" for _, k in log)
    if tasks:
        for event in ("episode_started", "episode_outcome"):
            rows = [k for a, k in log if a == (event,)]
            assert [r["task"] for r in rows] == ["green left", "blue right"]
            assert [r["task_id"] for r in rows] == ["1", "2"]
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


def test_task_menu_shortcuts_keep_hold_checks_and_explicit_labels():
    c = TeleopControl()
    ui = EpisodeControls(
        c,
        {},
        Terminal(["1", "1", "", "", "2 slipped"]),
        lambda *a, **k: None,
        tasks={"1": "green left"},
        position="B-layout1",
    )
    ui.enter("preparing")
    ui.poll()
    assert ui.request is None
    c.confirm_hold()
    ui.poll()
    assert ui.request == ("start",) and ui.task == "green left"
    assert ui.position == "B-layout1" and ui.task_id == "1"
    ui.enter("recording")
    ui.poll()
    assert ui.request == ("end",)
    ui.enter("review")
    c.confirm_hold()
    ui.poll()
    assert ui.request is None  # Blank cannot silently label a success.
    ui.poll()
    assert ui.request == ("save", "failure", "slipped")


def test_task_menu_rejects_old_start_and_unknown_choice():
    c = TeleopControl()
    ui = EpisodeControls(
        c,
        {},
        Terminal(["start P1", "5", "q"]),
        lambda *a, **k: None,
        tasks={"1": "green left"},
        position="B-layout1",
    )
    ui.enter("preparing")
    c.confirm_hold()
    ui.poll()
    assert ui.request is None
    ui.poll()
    assert ui.request is None
    ui.poll()
    assert ui.request == ("quit",)


def test_task_progress_counts_outcomes_separately_and_shows_camera_layout(capsys):
    c = TeleopControl()
    ui = EpisodeControls(
        c,
        {},
        Terminal([]),
        lambda *a, **k: None,
        tasks={"1": "green left", "2": "green right"},
        position="B-layout2",
        layout_description="六边形在上，绿色在下",
        split="train",
        task_goal=5,
    )
    ui.task_id = "1"
    for outcome in ("success", "failure", "cancelled"):
        ui.note_saved(outcome)
    ui.task_id = "2"
    ui.note_saved("success")
    ui.enter("preparing")
    text = capsys.readouterr().out
    assert "B-layout2" in text and "六边形在上，绿色在下" in text
    assert "保存3/5｜成功1 失败1 取消1" in text
    assert "保存1/5｜成功1 失败0 取消0" in text
    assert "本会话" in text and "相机画面" in text


@pytest.mark.parametrize(("goal", "successes"), [(50, 50), (6, 6), (50, 2)])
def test_fine_batch_counts_eligible_successes_and_keeps_early_quit(goal, successes, capsys):
    from unittest.mock import Mock
    from lerobot_robot_outcome_piper.demonstration_segments import DemonstrationTracker

    c = TeleopControl(initial_precision="fine")
    attempts = [
        ("fine", "2 missed", True),
        ("fine", "3 cancelled", True),
        ("fine", "r", True),
        ("fine", "1", False),
        ("fine", "1", True),  # Only normal was active; the final gain switch was paused.
        ("normal", "1", True),  # Normal approach, fine push, normal withdrawal.
        *[("fine", "1", True)] * (successes - 1),
    ]
    audit = Mock(attempt=0, pending=[], demonstration=DemonstrationTracker())
    audit.dataset.num_episodes = 0
    audit.prepare_episode.side_effect = ready_future

    def finish_attempt(saved):
        audit.attempt += 1
        audit.dataset.num_episodes += int(saved)
        audit.pending.clear()
        audit.demonstration = DemonstrationTracker()

    audit.save_episode.side_effect = lambda: finish_attempt(True)
    audit.discard_episode.side_effect = lambda: finish_attempt(False)

    def command():
        if c.recording_phase == "preparing":
            if audit.attempt == len(attempts):
                assert successes < goal, "a full batch should finalize without another prompt"
                return "q"
            precision = attempts[audit.attempt][0]
            return "3" if c.precision == precision else precision
        if c.recording_phase == "recording":
            return "end"
        if c.recording_phase == "review":
            return attempts[audit.attempt][1]
        return None

    def loop(**kw):
        for _ in range(5000):
            if kw["events"]["exit_early"]:
                kw["events"]["exit_early"] = False
                return
            c.confirm_hold()
            kw["teleop_action_processor"](({"hold": False, "neutral": True}, {}))
            if kw["dataset"] is not None:
                assert c.precision == attempts[audit.attempt][0]
                frames = [(c.precision, attempts[audit.attempt][2])]
                if audit.attempt == 4:
                    frames = [("normal", True), ("fine", False)]
                elif audit.attempt == 5:
                    frames = [("normal", True), ("fine", True), ("normal", True)]
                for gain, eligible in frames:
                    audit.pending.append(len(audit.pending))
                    audit.demonstration.observe(
                        {
                            "precision": gain,
                            "demonstration_pause": {
                                "manual": not eligible,
                                "protection_reason": None,
                            },
                        }
                    )
            time.sleep(0.0002)
        pytest.fail(f"batch stalled during {c.recording_phase}, attempt {audit.attempt}")

    run_interactive_episodes(
        NS(record_loop=loop),
        dict(
            events={},
            teleop_action_processor=lambda value: value[0],
            robot=NS(prepare_recording_gripper=lambda epoch: {}),
        ),
        audit,
        NS(num_episodes=goal, episode_time_s=180),
        c,
        NS(poll=command),
        tasks={"3": "Push-T"},
        position="start-03",
        target_precision="fine",
    )
    outcomes = [
        call.kwargs for call in audit.emit.call_args_list if call.args == ("episode_outcome",)
    ]
    assert [o["task_outcome"] for o in outcomes[:4]] == [
        "failure",
        "cancelled",
        "success",
        "success",
    ]
    assert outcomes[2]["eligible_frames"] == 0 and not outcomes[2]["data_valid"]
    assert outcomes[3]["precision"] == "normal" and outcomes[3]["data_valid"]
    assert len(outcomes) == 4 + successes
    assert outcomes[4]["precision"] == "mixed" and outcomes[4]["data_valid"]
    assert set(outcomes[4]["eligible_frames_by_precision"]) == {"normal", "fine"}
    assert outcomes[3]["eligible_frames_by_precision"] == {"normal": 1}
    assert all(
        o["task_outcome"] == "success" and o["precision"] == "fine" and o["data_valid"]
        for o in outcomes[5:]
    )
    audit.dataset.finalize.assert_called_once()
    assert f"fine 有效成功示范 {successes}/{goal}" in capsys.readouterr().out
