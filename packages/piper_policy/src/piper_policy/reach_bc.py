"""State/goal-only reach BC using the official LeRobot policy lifecycle."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamConfig
from lerobot.policies.pretrained import PreTrainedPolicy

FEEDBACK_GAIN = 0.2


def feedback_endpoint_target(action, state):
    """Recover the fixed teacher's endpoint from the actual physical labels."""
    return (action[:, :6] - (1 - FEEDBACK_GAIN) * state[:, :6]) / FEEDBACK_GAIN


@PreTrainedConfig.register_subclass("piper_reach_bc")
@dataclass
class ReachBCConfig(PreTrainedConfig):
    hidden_dims: tuple[int, int] = (128, 128)
    optimizer_lr: float = 1e-3
    fixed_gripper_m: float = 0.001
    joint_representation: str = "absolute"
    input_mean: list[float] = field(default_factory=list)
    input_scale: list[float] = field(default_factory=list)
    joint_mean: list[float] = field(default_factory=list)
    joint_scale: list[float] = field(default_factory=list)

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        if self.joint_representation not in ("absolute", "delta", "feedback_endpoint"):
            raise ValueError("unsupported reach joint representation")
        expected = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        }
        if self.input_features != expected or self.output_features != {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        }:
            raise ValueError("reach BC requires physical state7, goal XYZ3 and absolute action7")
        if self.n_obs_steps != 1 or tuple(self.hidden_dims) != (128, 128):
            raise ValueError("this reach BC recipe uses one observation and two 128-wide layers")
        for name, length in [
            ("input_mean", 10),
            ("input_scale", 10),
            ("joint_mean", 6),
            ("joint_scale", 6),
        ]:
            values = getattr(self, name)
            if len(values) != length or not all(math.isfinite(value) for value in values):
                raise ValueError(f"invalid train-only {name}")
            if name.endswith("scale") and min(values) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.fixed_gripper_m) or self.fixed_gripper_m < 0:
            raise ValueError("fixed gripper must be a finite nonnegative physical command")


class ReachBCPolicy(PreTrainedPolicy):
    config_class = ReachBCConfig
    name = "piper_reach_bc"

    def __init__(self, config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        input_width = 3 if config.joint_representation == "feedback_endpoint" else 10
        self.model = nn.Sequential(
            nn.Linear(input_width, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )
        for name in ("input_mean", "input_scale", "joint_mean", "joint_scale"):
            self.register_buffer(name, torch.tensor(getattr(config, name), dtype=torch.float32))

    def reset(self):
        return None

    def get_optim_params(self):
        return self.model.parameters()

    def _joint_prediction(self, batch):
        state = batch["observation.state"]
        goal = batch["observation.environment_state"]
        if state.ndim != 2 or state.shape[1] != 7 or goal.shape != (state.shape[0], 3):
            raise ValueError("reach inputs require batched state7 and goal XYZ3")
        values = torch.cat([state, goal], dim=-1)
        if not torch.isfinite(values).all():
            raise ValueError("reach input is not finite")
        normalized = (values - self.input_mean) / self.input_scale
        if self.config.joint_representation == "feedback_endpoint":
            normalized = normalized[:, 7:]
        return self.model(normalized)

    def forward(self, batch, reduction="mean"):
        predicted = self._joint_prediction(batch)
        target = batch["action"]
        if target.shape != (predicted.shape[0], 7) or not torch.isfinite(target).all():
            raise ValueError("reach labels require finite physical action7")
        if not torch.allclose(
            target[:, 6],
            torch.full_like(target[:, 6], self.config.fixed_gripper_m),
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError("reach label changes the fixed gripper")
        joint_target = target[:, :6]
        if self.config.joint_representation == "delta":
            joint_target = joint_target - batch["observation.state"][:, :6]
        elif self.config.joint_representation == "feedback_endpoint":
            joint_target = feedback_endpoint_target(target, batch["observation.state"])
        truth = (joint_target - self.joint_mean) / self.joint_scale
        per_sample = (predicted - truth).square().mean(dim=-1)
        if reduction not in ("mean", "none"):
            raise ValueError("unsupported loss reduction")
        loss = per_sample.mean() if reduction == "mean" else per_sample
        return loss, {"joint_normalized_mse": per_sample.mean().detach().item()}

    @torch.no_grad()
    def select_action(self, batch):
        joints = self._joint_prediction(batch) * self.joint_scale + self.joint_mean
        if self.config.joint_representation == "delta":
            joints = joints + batch["observation.state"][:, :6]
        elif self.config.joint_representation == "feedback_endpoint":
            state = batch["observation.state"][:, :6]
            joints = state + FEEDBACK_GAIN * (joints - state)
        gripper = joints.new_full((len(joints), 1), self.config.fixed_gripper_m)
        actions = torch.cat([joints, gripper], dim=-1)
        if not torch.isfinite(actions).all():
            raise ValueError("reach prediction is not finite")
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        return self.select_action(batch).unsqueeze(1)


def reach_processors(policy_cfg, **kwargs):
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
    )
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    policy_cfg.validate_features()
    return (
        PolicyProcessorPipeline(
            steps=[
                RenameObservationsProcessorStep(rename_map={}),
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=policy_cfg.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@contextmanager
def register_reach_factory():
    import lerobot.policies.factory as factory

    original = factory.get_policy_class

    def select(name):
        return ReachBCPolicy if name == "piper_reach_bc" else original(name)

    factory.get_policy_class = select
    try:
        yield
    finally:
        factory.get_policy_class = original
