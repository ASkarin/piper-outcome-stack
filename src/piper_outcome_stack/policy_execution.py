"""Synchronous ACT chunks decoded at their generation state, with absolute targets."""

import math
import time

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


def verify_reference_inputs(predictor, path, safety, *, warmup=30):
    """Reproduce recorded reference chunks, check first targets, then warm up.

    Returns (images, states, rows); each row keeps the target check as a report entry.
    """
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    with np.load(path) as data:
        images, states = data["images"], data["states"]
        expected = data["expected_absolute_chunks"]
    if len(states) == 0 or len(images) != len(states) or len(expected) != len(states):
        raise ValueError("reference inputs need matching, non-empty images/states/chunks")
    rows = []
    for i in range(len(states)):
        chunk = predictor.predict(images[i], states[i])
        delta = np.abs(chunk - expected[i])
        joint, grip = float(delta[:, :6].max()), float(delta[:, 6].max())
        if joint >= 1e-4 or grip >= 1e-5:
            raise ValueError(f"reference mismatch: sample={i}, joint={joint}, gripper={grip}")
        try:
            check_execution_target(states[i], chunk[0], safety)
            bounds = dict(status="passed")
        except ValueError as exc:
            bounds = dict(status="rejected", reason=str(exc))
        rows.append(dict(sample=i, joint_rad=joint, gripper_m=grip, recorded_target_check=bounds))
    for i in range(warmup):
        predictor.predict(images[i % len(states)], states[i % len(states)])
    return images, states, rows


def check_policy_observation(telemetry, timing, now):
    """Use the same input contract as get_observation and the pre-command age check."""
    if telemetry["quality"] != "checked":
        raise ValueError("policy needs a timing-checked observation")
    age = now - telemetry["oldest_received_monotonic_s"]
    if not math.isfinite(age) or age < 0 or age > timing.observation_max_age_s:
        raise ValueError(
            f"policy observation expired before candidate was ready: "
            f"age_s={age}, limit_s={timing.observation_max_age_s}"
        )
    return age


def predict_candidate(predictor, observation, telemetry, timing, safety, clock=time.monotonic):
    """One observation → absolute chunk → optional TE → checked target; sends nothing."""
    from lerobot_robot_outcome_piper.safety import ACTION_KEYS
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    started = clock()
    check_policy_observation(telemetry, timing, started)
    state = [float(observation[k]) for k in ACTION_KEYS]
    chunk, chunk_index, inferred = predictor.execution_chunk(
        observation[POLICY_CAMERA],
        state,
        telemetry["sequence"],
    )
    predicted = clock()
    target = predictor.select_target(chunk, chunk_index=chunk_index, new_chunk=inferred)
    try:
        check_execution_target(state, target, safety)
    except ValueError as exc:
        raise ValueError(
            f"{exc}; observed_rad_m={state}; candidate_rad_m={target.tolist()}"
        ) from exc
    checked = clock()
    try:
        age = check_policy_observation(telemetry, timing, checked)
    except ValueError as exc:
        exc.policy_timing = {
            "input_age_before_prediction_s": started - telemetry["oldest_received_monotonic_s"],
            "inference_s": predicted - started,
            "selection_and_validation_s": checked - predicted,
            "age_at_rejection_s": checked - telemetry["oldest_received_monotonic_s"],
            "inference_performed": inferred,
        }
        raise
    return dict(zip(ACTION_KEYS, map(float, target))), {
        "observation_sequence": telemetry["sequence"],
        "anchor": predictor.generation_state,
        "generation_observation_sequence": predictor.generation_sequence,
        "inference_performed": inferred,
        "n_action_steps": predictor.n_action_steps,
        "inference_s": predicted - started,
        "validation_s": checked - predicted,
        "observation_age_at_candidate_s": age,
        "executed_chunk_steps": chunk_index,
        "candidate_chunk_index": chunk_index,
        "temporal_ensemble_coeff": predictor.temporal_ensemble_coeff,
        "temporal_ensemble_contributors": predictor.last_contributors
        if predictor.temporal_ensembler is not None
        else 1,
    }


def run_shadow(robot, predictor, safety, cycles, fps, rows, clock=time.monotonic, sleep=time.sleep):
    """Connected read-only robot only. No enable, mode change or action dispatch."""
    if robot.config.execution_mode != "read_only" or robot.config.capture_timing is None:
        raise ValueError("shadow requires explicit read_only and measured capture_timing")
    if cycles <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("shadow requires positive cycles and fps")
    for i in range(cycles):
        started = clock()
        observation = robot.get_observation()
        observed = clock()
        target, entry = predict_candidate(
            predictor,
            observation,
            robot.last_observation_telemetry,
            robot.config.capture_timing,
            safety,
            clock,
        )
        ended = clock()
        entry.update(
            cycle=i,
            target=target,
            acquisition_s=observed - started,
            cycle_work_s=ended - started,
            input_telemetry=robot.last_observation_telemetry,
        )
        rows.append(entry)
        sleep(max(0, 1 / fps - (clock() - started)))
        entry["cycle_elapsed_s"] = clock() - started
        if (i + 1) % 50 == 0:
            print(f"shadow {i + 1}/{cycles}: work={1000 * entry['cycle_work_s']:.2f}ms", flush=True)
