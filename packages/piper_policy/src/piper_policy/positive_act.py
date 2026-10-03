"""Experimental ACT output parameterization; no Dataset or controller changes."""

from dataclasses import dataclass
import math
from torch.nn import functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy


@PreTrainedConfig.register_subclass("act_positive_gripper")
@dataclass
class PositiveACTConfig(ACTConfig):
    gripper_state_mean: float | None = None
    gripper_state_std: float | None = None
    gripper_action_mean: float | None = None
    gripper_action_std: float | None = None
    positive_scale_m: float = 0.001
    positive_floor_m: float = 0.000001


def parameterize(actions, state, cfg):
    anchor = state[:, 6] * cfg.gripper_state_std + cfg.gripper_state_mean
    if actions.ndim == 3:
        anchor = anchor[:, None]
    unconstrained = actions[..., 6] * cfg.gripper_action_std + cfg.gripper_action_mean + anchor
    width = cfg.positive_floor_m + cfg.positive_scale_m * F.softplus(
        unconstrained / cfg.positive_scale_m
    )
    out = actions.clone()
    out[..., 6] = (width - anchor - cfg.gripper_action_mean) / cfg.gripper_action_std
    return out


class PositiveACT(ACT):
    def forward(self, batch):
        actions, latent = super().forward(batch)
        return parameterize(actions, batch["observation.state"], self.config), latent


class PositiveACTPolicy(ACTPolicy):
    config_class = PositiveACTConfig
    name = "act_positive_gripper"

    def __init__(self, config, **kwargs):
        assert isinstance(config, PositiveACTConfig)
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        assert config.temporal_ensemble_coeff is None, "this experiment uses full chunk inference"
        values = [
            config.gripper_state_mean,
            config.gripper_state_std,
            config.gripper_action_mean,
            config.gripper_action_std,
            config.positive_scale_m,
            config.positive_floor_m,
        ]
        assert all(x is not None and math.isfinite(x) for x in values)
        assert (
            config.gripper_state_std > 0
            and config.gripper_action_std > 0
            and config.positive_scale_m > 0
            and config.positive_floor_m > 0
        )
        self.config = config
        self.model = PositiveACT(config)
        self.reset()
