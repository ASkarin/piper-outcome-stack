"""Refit six ACT joint output rows while preserving the gripper and frozen features."""

import math

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from ._head_refit import FrozenACTHead, ridge_solution


class JointHeadRefiner(FrozenACTHead):
    def __init__(self, predictor, checkpoint):
        super().__init__(predictor, checkpoint)
        paths = list(self.checkpoint.glob("policy_postprocessor*unnormalizer*.safetensors"))
        if len(paths) != 1:
            raise ValueError("requires the checkpoint's single action unnormalizer")
        stats = load_file(str(paths[0]))
        self.mean = stats["action.mean"][:6].float()
        self.std = stats["action.std"][:6].float()
        if self.mean.shape != (6,) or self.std.shape != (6,) or not torch.all(self.std > 0):
            raise ValueError("six positive checkpoint joint scales required")
        self.weight = torch.nn.Parameter(self.head.weight[:6].detach().cpu().clone())
        self.bias = torch.nn.Parameter(self.head.bias[:6].detach().cpu().clone())

    def predict_joints(self, features, states):
        return (
            F.linear(features, self.weight, self.bias) * self.std + self.mean + states[:, None, :6]
        )

    @torch.no_grad()
    def fit_ridge(
        self,
        features,
        states,
        absolute_targets,
        valid,
        train_indices,
        *,
        relative_regularization=0.01,
    ):
        desired = (
            absolute_targets.double() - states[:, None, :6].double() - self.mean.double()
        ) / self.std.double()
        prior = torch.cat([self.weight.detach().double().T, self.bias.detach().double()[None]])
        fitted, result = ridge_solution(
            features, desired, valid, train_indices, prior, relative_regularization
        )
        self.weight.copy_(fitted[:-1].T.float())
        self.bias.copy_(fitted[-1].float())
        return result

    def replacement(self):
        return (
            slice(0, 6),
            self.weight,
            self.bias,
            "joint_refit.json",
            ["model.action_head.weight[:6]", "model.action_head.bias[:6]"],
        )

    @staticmethod
    @torch.no_grad()
    def temporal_design(features, observed_states, coefficient):
        """Time-align frozen decoder features and generation anchors for joint TE.

        Rows are consecutive observations from one episode, with a fresh TE state.
        The affine joint decoder commutes with this averaging; gripper softplus
        is deliberately excluded. No target actions enter this transformation.
        """
        if not math.isfinite(coefficient):
            raise ValueError("temporal ensemble coefficient must be finite")
        n, horizon, dimension = features.shape
        if observed_states.shape != (n, 7):
            raise ValueError("consecutive N,7 observed states required")
        design = torch.empty(n, dimension, dtype=torch.float64)
        anchors = torch.empty(n, 6, dtype=torch.float64)
        for tick in range(n):
            ages = torch.arange(min(tick + 1, horizon))
            weights = torch.softmax(float(coefficient) * ages.double(), dim=0)
            design[tick] = (features[tick - ages, ages].double() * weights[:, None]).sum(0)
            anchors[tick] = (observed_states[tick - ages, :6].double() * weights[:, None]).sum(0)
        return design, anchors

    def predict_temporal_joints(self, design, anchors):
        return (
            F.linear(design.double(), self.weight.double(), self.bias.double()) * self.std.double()
            + self.mean.double()
            + anchors.double()
        )

    @torch.no_grad()
    def fit_temporal_ridge(
        self, design, anchors, absolute_targets, train_indices, *, relative_regularization=0.01
    ):
        desired = (
            absolute_targets.double() - anchors.double() - self.mean.double()
        ) / self.std.double()
        prior = torch.cat([self.weight.detach().double().T, self.bias.detach().double()[None]])
        valid = torch.ones(len(design), 1, dtype=torch.bool)
        fitted, result = ridge_solution(
            design[:, None], desired[:, None], valid, train_indices, prior, relative_regularization
        )
        self.weight.copy_(fitted[:-1].T.float())
        self.bias.copy_(fitted[-1].float())
        return result
