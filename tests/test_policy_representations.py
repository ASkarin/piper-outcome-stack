from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from lerobot.processor.relative_action_processor import (
    RelativeActionsProcessorStep,
    AbsoluteActionsProcessorStep,
)
from piper_outcome_stack.policy_execution import validate_action_processors, predict_candidate

NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]


def processors(kind):
    relative = RelativeActionsProcessorStep(
        enabled=True,
        action_names=NAMES,
        exclude_joints=NAMES[:6] if kind == "absolute" else [],
    )
    absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)
    return NS(steps=[relative]), NS(steps=[absolute])


@pytest.mark.parametrize("kind", ["relative", "absolute"])
def test_expected_representation_matches_checkpoint_mask(kind):
    pre, post = processors(kind)
    validate_action_processors(pre, post, kind)
    other = "absolute" if kind == "relative" else "relative"
    with pytest.raises(ValueError, match="do not match"):
        validate_action_processors(pre, post, other)


def test_wrong_order_and_unpaired_processors_fail():
    pre, post = processors("absolute")
    pre.steps[0].action_names = list(reversed(NAMES))
    with pytest.raises(ValueError, match="do not match"):
        validate_action_processors(pre, post, "absolute")
    pre, post = processors("absolute")
    post.steps[0].relative_step = processors("absolute")[0].steps[0]
    with pytest.raises(ValueError, match="do not match"):
        validate_action_processors(pre, post, "absolute")


def test_recorded_diagnostic_does_not_disable_live_target_rejection(monkeypatch):
    path = Path(__file__).parents[1] / "infra/acceptance/piper_policy_trial.py"
    spec = importlib.util.spec_from_file_location("trial_representation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def reject(*args):
        raise ValueError("candidate J1 exceeds execution step")

    monkeypatch.setattr(
        "lerobot_robot_outcome_piper.execution_constraints.check_execution_target", reject
    )
    state = [0.0] * 7
    assert module.recorded_target_check(state, [1.0] * 7, None)["status"] == "rejected"
    predictor = NS(
        execution_chunk=lambda *args: (np.ones((50, 7), np.float32), 0, True),
        select_target=lambda x, **kwargs: x[0],
    )
    obs = {**dict(zip(NAMES, state)), "d435": None}
    with pytest.raises(ValueError, match="exceeds execution step"):
        predict_candidate(
            predictor,
            obs,
            dict(quality="checked", oldest_received_monotonic_s=1.0, sequence=1),
            NS(observation_max_age_s=0.1),
            None,
            lambda: 1.0,
        )
