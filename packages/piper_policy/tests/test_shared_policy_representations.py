from types import SimpleNamespace as NS

import pytest
from lerobot.processor.relative_action_processor import (
    RelativeActionsProcessorStep,
    AbsoluteActionsProcessorStep,
)
from piper_policy.policy_execution import validate_action_processors

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
