"""LeRobot integration shared by real-data and simulation training callers."""

from contextlib import contextmanager
from copy import deepcopy
import numpy as np
from .positive_act import PositiveACTPolicy
from .policy_execution import validate_action_processors


@contextmanager
def register_act_factory():
    """Add the existing positive ACT type without changing official policy types."""
    import lerobot.policies.factory as factory

    original = factory.get_policy_class

    def select(name):
        return PositiveACTPolicy if name == "act_positive_gripper" else original(name)

    factory.get_policy_class = select
    try:
        yield
    finally:
        factory.get_policy_class = original


def act_processors(config, *, checkpoint=None, dataset_stats=None, relative_stats=None):
    """Preserve saved relative normalizers on resume; construct them once for fresh ACT."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor.relative_action_processor import (
        RelativeActionsProcessorStep,
        AbsoluteActionsProcessorStep,
    )
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    if config.type != "act_positive_gripper":
        raise ValueError("this processor recipe requires positive ACT")
    if checkpoint is not None:
        pre, post = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": config.device}},
        )
    else:
        if dataset_stats is None or relative_stats is None:
            raise ValueError("fresh ACT requires explicit train-only statistics")
        stats = deepcopy(dataset_stats)
        stats["action"] = {key: np.asarray(value) for key, value in relative_stats.items()}
        pre, post = make_pre_post_processors(config, dataset_stats=stats)
        relative = RelativeActionsProcessorStep(
            enabled=True,
            exclude_joints=[],
            action_names=[f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"],
        )
        index = next(
            i for i, step in enumerate(pre.steps) if isinstance(step, NormalizerProcessorStep)
        )
        pre.steps.insert(index, relative)
        post.steps.insert(1, AbsoluteActionsProcessorStep(enabled=True, relative_step=relative))
    validate_action_processors(pre, post, "relative")
    return pre, post
