"""Explicit train-time sqrt(m) gripper representation, serialized with processors."""

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import TransitionKey

from .statistics import numeric_stats


@ProcessorStepRegistry.register("piper_sqrt_gripper")
@dataclass
class SqrtGripperStep(ProcessorStep):
    decode: bool = False

    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        if action is None and not self.decode:
            return transition.copy()  # Inference observations have no action labels.
        if not isinstance(action, torch.Tensor) or action.shape[-1] != 7:
            raise ValueError("sqrt gripper requires a seven-dimensional action tensor")
        if not torch.isfinite(action).all():
            raise ValueError("sqrt gripper requires finite actions")
        result = action.clone()
        if self.decode:
            result[..., 6] = result[..., 6].square()
        else:
            if torch.any(result[..., 6] < 0):
                raise ValueError("negative source gripper action cannot be encoded")
            result[..., 6] = result[..., 6].sqrt()
        if not torch.isfinite(result).all():
            raise ValueError("non-finite sqrt gripper result")
        output = transition.copy()
        output[TransitionKey.ACTION] = result
        return output

    def transform_features(self, features):
        return features

    def get_config(self):
        return {"decode": self.decode}


def make_processors(config, training, encoding):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    stats = deepcopy(training.meta.stats)
    if encoding == "sqrt_gripper":
        encoded = training.actions.copy()
        if not np.isfinite(encoded).all() or np.any(encoded[:, 6] < 0):
            raise ValueError("sqrt gripper requires finite, nonnegative training openings")
        encoded[:, 6] = np.sqrt(encoded[:, 6])
        stats["action"] = numeric_stats(encoded)
    elif encoding != "absolute":
        raise ValueError(f"unsupported action encoding: {encoding}")
    pre, post = make_smolvla_pre_post_processors(config, dataset_stats=stats)
    if encoding == "sqrt_gripper":
        pre.steps.insert(0, SqrtGripperStep())
        post.steps.append(SqrtGripperStep(decode=True))
    return pre, post
