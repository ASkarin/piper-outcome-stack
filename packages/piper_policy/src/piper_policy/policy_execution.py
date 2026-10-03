"""Synchronous ACT chunks decoded at their generation state, with absolute targets."""

import math

import numpy as np
import torch

# Checkpoint contract: trained at 50Hz control/Dataset with one RGB feature named d435.
CONTROL_HZ = 50
CONTROL_PERIOD_S = 1 / CONTROL_HZ
POLICY_CAMERA = "d435"
IMAGE_FEATURE = f"observation.images.{POLICY_CAMERA}"


def validate_action_processors(pre, post, joint_representation):
    from lerobot.processor.relative_action_processor import (
        RelativeActionsProcessorStep,
        AbsoluteActionsProcessorStep,
    )

    masks = {"relative": [True] * 7, "absolute": [False] * 6 + [True]}
    if joint_representation not in masks:
        raise ValueError("joint representation must be relative or absolute")
    relative = [s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep)]
    absolute = [s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep)]
    names = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    if (
        len(relative) != 1
        or len(absolute) != 1
        or not relative[0].enabled
        or not absolute[0].enabled
        or absolute[0].relative_step is not relative[0]
        or relative[0].action_names != names
        or relative[0]._build_mask(7) != masks[joint_representation]
    ):
        raise ValueError(
            f"checkpoint processors do not match {joint_representation} joints and relative gripper"
        )


class ACTChunkPredictor:
    """Single-threaded batch-one inference; never queue normalized relative actions."""

    def __init__(self, model, preprocessor, postprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.temporal_ensembler = None
        self.temporal_ensemble_coeff = None
        self.ensemble_updates = 0
        self.n_action_steps = 1
        self._chunk = None
        self._next_chunk_index = 0
        self.generation_state = None
        self.generation_sequence = None
        self._control_tick = 0
        self._sparse_history = []
        self.last_contributors = 1

    def configure_action_steps(self, steps):
        if type(steps) is not int or not 1 <= steps <= self.model.config.chunk_size:
            raise ValueError("action steps must be an integer within the chunk horizon")
        self.n_action_steps = steps
        self.reset_execution()

    def execution_chunk(self, image, state, sequence):
        """Refresh only at the selected cadence; cached targets keep their old anchor."""
        inferred = self._chunk is None or self._next_chunk_index >= self.n_action_steps
        if inferred:
            self._chunk = self.predict(image, state)
            self._next_chunk_index = 0
            self.generation_state = list(state)
            self.generation_sequence = sequence
        index = self._next_chunk_index
        self._next_chunk_index += 1
        return self._chunk, index, inferred

    def configure_temporal_ensemble(self, coefficient):
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        if coefficient is not None and not math.isfinite(coefficient):
            raise ValueError("temporal ensemble coefficient must be finite")
        self.temporal_ensemble_coeff = coefficient
        self.temporal_ensembler = (
            None
            if coefficient is None
            else ACTTemporalEnsembler(
                coefficient,
                self.model.config.chunk_size,
            )
        )
        self.reset_execution()

    def reset_execution(self):
        self.ensemble_updates = 0
        self._chunk = None
        self._next_chunk_index = 0
        self.generation_state = self.generation_sequence = None
        self._control_tick = 0
        self._sparse_history = []
        self.last_contributors = 1
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    def select_target(self, absolute_chunk, *, chunk_index=0, new_chunk=True):
        """Ensemble only after each entire chunk has been decoded at its own anchor."""
        if self.temporal_ensembler is None:
            return absolute_chunk[chunk_index].copy()
        if self.n_action_steps > 1:
            tick = self._control_tick
            if new_chunk:
                self._sparse_history.append((tick, absolute_chunk.copy()))
                self.ensemble_updates += 1
            self._sparse_history = [
                (born, chunk) for born, chunk in self._sparse_history if tick - born < len(chunk)
            ]
            if not self._sparse_history:
                raise ValueError("no time-aligned prediction for execution tick")
            # Keep coefficient units in 20ms control steps, not inference calls.
            oldest = self._sparse_history[0][0]
            weights = np.exp(
                -self.temporal_ensemble_coeff
                * np.array([born - oldest for born, _ in self._sparse_history], dtype=np.float64)
            )
            candidates = np.stack([chunk[tick - born] for born, chunk in self._sparse_history])
            target = np.average(candidates, axis=0, weights=weights).astype(absolute_chunk.dtype)
            self.last_contributors = len(candidates)
            self._control_tick += 1
            return target
        with torch.inference_mode():
            target = self.temporal_ensembler.update(torch.from_numpy(absolute_chunk)[None])[0]
        self.ensemble_updates += 1
        self.last_contributors = min(self.ensemble_updates, len(absolute_chunk))
        return target.numpy().copy()

    @classmethod
    def from_checkpoint(cls, checkpoint, *, joint_representation="relative"):
        from .positive_act import PositiveACTConfig, PositiveACTPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        if not torch.cuda.is_available():
            raise RuntimeError("controller policy requires CUDA")
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(cfg, PositiveACTConfig) or cfg.temporal_ensemble_coeff is not None:
            raise ValueError("requires the positive-gripper full-chunk checkpoint")
        cfg.device = "cuda"
        cfg.pretrained_backbone_weights = None  # All saved weights are loaded strictly below.
        model = PositiveACTPolicy.from_pretrained(checkpoint, config=cfg, strict=True).eval()
        pre, post = make_pre_post_processors(
            cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        validate_action_processors(pre, post, joint_representation)
        return cls(model, pre, post)

    def predict(self, image, state):
        state = np.asarray(state, dtype=np.float32)
        shape = self.model.config.input_features[IMAGE_FEATURE].shape
        if image.dtype != np.uint8 or image.shape != (shape[1], shape[2], shape[0]):
            raise ValueError("policy requires checkpoint-shaped HWC RGB uint8 input")
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("policy requires seven finite observed rad/m values")
        batch = {
            IMAGE_FEATURE: torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0,
            "observation.state": torch.from_numpy(state.copy()),
        }
        with torch.inference_mode():
            relative = self.model.predict_action_chunk(self.preprocessor(batch))
            # Consume the cached generation anchor NOW, before another preprocessor call.
            absolute = self.postprocessor(relative)
        chunk = absolute[0].detach().cpu().numpy().copy()
        if chunk.shape != (self.model.config.chunk_size, 7) or not np.isfinite(chunk).all():
            raise ValueError("invalid absolute ACT action chunk")
        return chunk

    def predict_batch(self, images, states):
        """Decode a batch at its own anchors; no target actions enter inference.

        Images are float tensors NCHW in [0, 1], states are N×7 in rad/m.
        The returned independent NumPy array is N×horizon×7 absolute targets.
        """
        shape = self.model.config.input_features[IMAGE_FEATURE].shape
        if not isinstance(images, torch.Tensor) or not isinstance(states, torch.Tensor):
            raise TypeError("batch images and states must be tensors")
        if images.ndim != 4 or tuple(images.shape[1:]) != tuple(shape):
            raise ValueError("batch RGB shape does not match checkpoint")
        if states.shape != (len(images), 7) or not len(images):
            raise ValueError("batch states must be nonempty N by 7")
        if not torch.isfinite(images).all() or not torch.isfinite(states).all():
            raise ValueError("batch observations must be finite")
        if not images.is_floating_point() or images.min() < 0 or images.max() > 1:
            raise ValueError("batch RGB must be float in [0, 1]")
        batch = {IMAGE_FEATURE: images, "observation.state": states.to(torch.float32)}
        with torch.inference_mode():
            relative = self.model.predict_action_chunk(self.preprocessor(batch))
            absolute = self.postprocessor(relative)
        result = absolute.detach().cpu().numpy().copy()
        if (
            result.shape != (len(images), self.model.config.chunk_size, 7)
            or not np.isfinite(result).all()
        ):
            raise ValueError("invalid absolute ACT batch")
        return result
