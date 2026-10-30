from types import SimpleNamespace as NS

import pytest

from test_xbox_pause import session as session, settle, moves  # noqa: F401
from test_policy_motion import Input
from piper_outcome_stack.policy_motion import run_policy_trial
from lerobot_robot_outcome_piper.safety import ACTION_KEYS


@pytest.mark.parametrize("cancel", [None, "release", "b"])
def test_state_candidate_callback_keeps_shared_budget_hold_and_post_inference_b(
    session, monkeypatch, cancel
):
    settle(session)
    robot, arm, control, clock = session
    raw = Input()
    rows = []
    calls = []
    monkeypatch.setattr(
        "piper_outcome_stack.policy_motion.predict_candidate",
        lambda *a, **kw: pytest.fail("RGB candidate must not be used"),
    )

    def state_candidate(predictor, observation, *args, **kwargs):
        calls.append(1)
        if cancel == "release":
            raw.hold = False
        if cancel == "b":
            raw.b = True
        target = {key: float(observation[key]) for key in ACTION_KEYS}
        target["joint_1.pos"] += 0.005
        return target, {"method": "grid"}

    predictor = NS(reset_execution=lambda: None)
    before = len(moves(arm))
    kwargs = dict(
        candidate_fn=state_candidate,
        max_actions=2,
        max_run_s=1,
        clock=clock,
        sleep=lambda seconds: clock.advance(seconds),
    )
    if cancel == "b":
        with pytest.raises(RuntimeError, match="Xbox B"):
            run_policy_trial(robot, raw, control, predictor, robot._safety, rows, **kwargs)
        assert len(moves(arm)) == before
    else:
        result = run_policy_trial(robot, raw, control, predictor, robot._safety, rows, **kwargs)
        assert result["policy_actions_sent"] == (0 if cancel else 2)
        assert result["hold_confirmed"]
    assert calls
