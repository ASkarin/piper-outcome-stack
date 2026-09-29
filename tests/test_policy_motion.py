import pytest
from types import SimpleNamespace

from test_xbox_pause import session as session, settle, moves  # noqa: F401
from lerobot_robot_outcome_piper.robot import PiperState
from piper_outcome_stack.policy_motion import PolicyInput, run_policy_trial


class Input:
    def __init__(self):
        self.calls = 0
        self.hold = True
        self.neutral = True
        self.b = False
        self.work = False

    def get_action(self):
        self.calls += 1
        assert self.calls < 1000, "trial must terminate"
        return dict(
            hold=self.hold and self.calls > 2,
            neutral=self.neutral,
            emergency_stop=self.b,
            home=False,
            work=self.work,
            mode_switch=False,
            translation_switch=False,
        )


def candidate(monkeypatch, change=None):
    def predict(predictor, obs, *args):
        target = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        target["joint_1.pos"] += 0.005
        if change:
            change()
        return target, {}

    monkeypatch.setattr("piper_outcome_stack.policy_motion.predict_candidate", predict)


def trial(session, raw, **kwargs):
    robot, _, control, clock = session
    rows = []
    budgets = dict(max_actions=2, max_run_s=1.0)
    budgets.update(kwargs)
    resets = []
    predictor = SimpleNamespace(reset_execution=lambda: resets.append(True))
    try:
        result = run_policy_trial(
            robot,
            raw,
            control,
            predictor,
            robot._safety,
            rows,
            clock=clock,
            sleep=lambda dt: clock.advance(dt),
            **budgets,
        )
    finally:
        assert len(resets) == 2
    return result, rows


@pytest.mark.parametrize("count,seconds", [(2, 1.0), (150, 3.0)])
def test_action_budget_holds_without_disabling(session, monkeypatch, count, seconds):
    settle(session)
    candidate(monkeypatch)
    result, rows = trial(session, Input(), max_actions=count, max_run_s=seconds)
    assert result["policy_actions_sent"] == count
    assert result["end_reason"] == "segment_budget_reached"
    assert result["hold_confirmed"]
    assert len([r for r in rows if r["phase"] == "policy_sent"]) == count
    assert "electronic_emergency_stop" not in session[1].calls
    assert session[0].state is PiperState.ACTIVE


@pytest.mark.parametrize("cancel", ["release", "stick", "work"])
def test_post_inference_cancel_never_sends_predicted_target(session, monkeypatch, cancel):
    settle(session)
    raw = Input()
    field, value = {
        "release": ("hold", False),
        "stick": ("neutral", False),
        "work": ("work", True),
    }[cancel]
    candidate(monkeypatch, lambda: setattr(raw, field, value))
    result, rows = trial(session, raw)
    assert result["policy_actions_sent"] == 0
    assert result["end_reason"] == "input_changed_during_inference"
    assert not session[1].gripper.commands
    assert all(m[1][0] != pytest.approx(0.005) for m in moves(session[1]))


def test_b_during_inference_stops_before_sdk_target(session, monkeypatch):
    settle(session)
    raw = Input()
    candidate(monkeypatch, lambda: setattr(raw, "b", True))
    with pytest.raises(RuntimeError, match="Xbox B"):
        trial(session, raw)
    assert session[0].state is PiperState.E_STOP
    assert "electronic_emergency_stop" in session[1].calls
    assert not session[1].gripper.commands


@pytest.mark.parametrize("error", [ValueError("invalid candidate"), KeyboardInterrupt()])
def test_model_error_or_interrupt_holds_then_latches(session, monkeypatch, error):
    settle(session)
    # Fake receive timestamps advance during the existing synchronous fault hold.
    session[3].auto = 0.001

    def fail(*a):
        raise error

    monkeypatch.setattr("piper_outcome_stack.policy_motion.predict_candidate", fail)
    with pytest.raises(type(error)):
        trial(session, Input())
    assert session[0].state is PiperState.FAULT
    assert session[0].stop_outcome == "hold_confirmed"
    assert not session[1].gripper.commands


def test_deadline_crossed_during_prediction_discards_next_target(session, monkeypatch):
    settle(session)
    calls = []

    def step():
        calls.append(1)
        if len(calls) == 2:
            session[3].advance(1.0)

    candidate(monkeypatch, step)
    result, rows = trial(session, Input())
    assert result["policy_actions_sent"] == 1
    assert result["end_reason"] == "segment_budget_reached"
    assert rows[-1]["phase"] == "final_hold"


def test_conflict_requires_physical_release_before_rearming(session):
    settle(session)
    robot, _, control, _ = session
    raw = Input()
    gate = PolicyInput(raw, robot)
    for _ in range(3):
        v = gate.poll()
        intent, _ = control.observe(v["hold"], v["neutral"])
    assert intent == "run"
    raw.work = True
    v = gate.poll()
    assert control.observe(v["hold"], v["neutral"])[0] == "hold"
    control.confirm_hold()
    raw.work = False
    v = gate.poll()
    assert control.observe(v["hold"], v["neutral"])[0] == "hold"
    raw.hold = False
    v = gate.poll()
    control.observe(v["hold"], v["neutral"])
    raw.hold = True
    v = gate.poll()
    assert control.observe(v["hold"], v["neutral"])[0] == "run"


def test_unlimited_runs_past_previous_budgets_until_release(session, monkeypatch):
    settle(session)
    raw = Input()
    calls = []

    def release_after_200():
        calls.append(1)
        if len(calls) == 201:
            raw.hold = False

    candidate(monkeypatch, release_after_200)
    started = session[3].now
    result, rows = trial(session, raw, max_actions=None, max_run_s=None)
    assert result["policy_actions_sent"] == 200
    assert session[3].now - started > 3.0
    assert result["end_reason"] == "input_changed_during_inference"
    assert result["hold_confirmed"]
    assert len([r for r in rows if r["phase"] == "policy_sent"]) == 200


@pytest.mark.parametrize("cancel", ["release", "stick", "work"])
def test_rgb_recording_preserves_post_inference_cancellation(session, monkeypatch, cancel):
    import numpy as np

    settle(session)
    robot = session[0]
    original = robot.get_observation
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    def observation():
        return {**original(), "d435": image}

    monkeypatch.setattr(robot, "get_observation", observation)
    recorded = []

    def submit(rgb, meta, now):
        assert rgb is image
        recorded.append(meta)
        return {"status": "queued", "frame_id": 0}

    raw = Input()
    field, value = {
        "release": ("hold", False),
        "stick": ("neutral", False),
        "work": ("work", True),
    }[cancel]
    candidate(monkeypatch, lambda: setattr(raw, field, value))
    result, rows = trial(session, raw, frame_recorder=SimpleNamespace(submit=submit))
    assert result["policy_actions_sent"] == 0 and result["hold_confirmed"]
    assert len(recorded) == 1
    row = rows[recorded[0]["row_index"]]
    assert row["phase"] == "candidate" and row["discarded"] == "input_changed_during_inference"
    assert recorded[0]["observation_sequence"] == row["input_telemetry"]["sequence"]
