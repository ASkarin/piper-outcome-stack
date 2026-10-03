"""PiPER's explicit seven-dimensional, single-RGB SmolVLA contract."""

from pathlib import Path

import numpy as np
import torch

from .async_chunks import ActionChunk

IMAGE_FEATURE = "observation.images.d435"
TASK_A = "Grasp the green block, lift, place and release."
ACTION_NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]


def check_contract(config):
    if config.type != "smolvla":
        raise ValueError("expected an official SmolVLA checkpoint")
    if set(config.input_features) != {"observation.state", IMAGE_FEATURE}:
        raise ValueError("PiPER SmolVLA requires exactly state and d435 RGB")
    if tuple(config.input_features["observation.state"].shape) != (7,):
        raise ValueError("PiPER state must contain seven rad/m values")
    if tuple(config.input_features[IMAGE_FEATURE].shape) != (3, 480, 640):
        raise ValueError("PiPER RGB must be 640x480")
    if set(config.output_features) != {"action"} or tuple(config.action_feature.shape) != (7,):
        raise ValueError("PiPER action must contain seven rad/m values")
    if config.adapt_to_pi_aloha or config.use_delta_joint_actions_aloha or config.empty_cameras:
        raise ValueError("PiPER does not use Aloha conversions or empty camera inputs")
    if config.n_obs_steps != 1 or config.chunk_size != 50:
        raise ValueError("this SmolVLA integration requires one observation and a 50-action chunk")


def finetune_config(source_checkpoint, vlm_assets):
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401

    config = PreTrainedConfig.from_pretrained(source_checkpoint)
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        IMAGE_FEATURE: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    config.pretrained_path = Path(source_checkpoint)
    config.vlm_model_name = str(Path(vlm_assets).resolve())
    # The complete robot checkpoint supplies both VLM and action-expert weights.
    # Avoid a second VLM-only weight initialization/download before that load.
    config.load_vlm_weights = False
    config.device = "cuda"
    config.push_to_hub = False
    config.empty_cameras = 0
    config.adapt_to_pi_aloha = False
    config.use_delta_joint_actions_aloha = False
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    config.n_action_steps = 10
    config.use_amp = True
    check_contract(config)
    return config


class SmolVLAChunkPredictor:
    """Official absolute-action inference with explicit task and chunk cadence."""

    def __init__(self, model, preprocessor, postprocessor, task):
        check_contract(model.config)
        if not isinstance(task, str) or not task.strip():
            raise ValueError("SmolVLA requires a nonempty task instruction")
        self.model, self.preprocessor, self.postprocessor = model, preprocessor, postprocessor
        self.task = task
        self.configure_action_steps(model.config.n_action_steps)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, task, vlm_assets=None):
        # Register the action representation stored in checkpoint processors.
        from . import smolvla_gripper  # noqa: F401
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(checkpoint)
        check_contract(config)
        config.device = "cuda"
        overrides = {"device_processor": {"device": "cuda"}}
        if vlm_assets is not None:
            assets = str(Path(vlm_assets).resolve(strict=True))
            config.vlm_model_name = assets
            overrides["tokenizer_processor"] = {"tokenizer_name": assets}
        config.load_vlm_weights = False
        model = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True).eval()
        model.requires_grad_(False)
        pre, post = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides=overrides,
        )
        return cls(model, pre, post, task)

    def configure_action_steps(self, steps):
        if type(steps) is not int or not 1 <= steps <= self.model.config.chunk_size:
            raise ValueError("action steps must be within the predicted chunk")
        self.n_action_steps = steps
        self._chunk = None
        self._next_index = 0
        self.model.reset()

    def configure_temporal_ensemble(self, coefficient):
        if coefficient is not None:
            raise ValueError("SmolVLA currently executes queued chunks without temporal ensemble")

    def configure_rtc(self, enabled=True, *, num_steps=10, guidance="full-vjp"):
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.configs import RTCAttentionSchedule
        from .rtc import PaperRTCProcessor

        if num_steps not in (5, 10):
            raise ValueError("this RTC validation supports five or ten denoising steps")
        self.model.config.num_steps = num_steps
        self.model.config.rtc_config = (
            RTCConfig(
                enabled=True,
                max_guidance_weight=5,
                execution_horizon=25,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            )
            if enabled
            else None
        )
        processor = PaperRTCProcessor(guidance=guidance) if enabled else None
        self.model.rtc_processor = self.model.model.rtc_processor = processor

    def encode_execution_actions(self, physical):
        from lerobot.processor import NormalizerProcessorStep

        if any(
            getattr(step, "_registry_name", None) == "piper_sqrt_gripper"
            for step in self.preprocessor.steps
        ):
            raise ValueError(
                "execution projection currently requires the original absolute-action checkpoint"
            )
        normalizers = [
            step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)
        ]
        if len(normalizers) != 1:
            raise ValueError("expected one saved action normalizer")
        values = torch.as_tensor(physical, dtype=torch.float32, device=self.model.config.device)
        with torch.no_grad():
            encoded = normalizers[0]({"action": values})["action"]
        return encoded.detach().float().cpu().numpy().copy()

    def predict(self, image, state):
        return self.predict_chunk(image, state).actions

    def predict_chunk(
        self,
        image,
        state,
        *,
        noise=None,
        previous_model_actions=None,
        inference_delay=0,
        overlap_steps=None,
        gripper_lower_bound="strict",
    ):
        if image.dtype != np.uint8 or image.shape != (480, 640, 3):
            raise ValueError("SmolVLA requires 640x480 RGB uint8")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("SmolVLA requires seven finite observed rad/m values")
        batch = {
            IMAGE_FEATURE: torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0,
            "observation.state": torch.from_numpy(state.copy()),
            "task": self.task,
        }
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.model.config.use_amp
            ),
        ):
            options = {}
            if noise is not None:
                noise = torch.as_tensor(noise, dtype=torch.float32, device=self.model.config.device)
                if noise.shape != (1, 50, self.model.config.max_action_dim):
                    raise ValueError("invalid explicit SmolVLA noise shape")
                options["noise"] = noise
            if previous_model_actions is not None:
                if self.model.config.rtc_config is None or not self.model.config.rtc_config.enabled:
                    raise ValueError("a previous chunk requires RTC to be enabled")
                options.update(
                    prev_chunk_left_over=previous_model_actions,
                    inference_delay=inference_delay,
                    execution_horizon=overlap_steps,
                )
            normalized = self.model.predict_action_chunk(self.preprocessor(batch), **options)
            model_actions = normalized[0].detach().float().cpu().numpy().copy()
            absolute = self.postprocessor(normalized.clone())
        result = absolute[0].detach().float().cpu().numpy().copy()
        if result.shape != (50, 7) or not np.isfinite(result).all():
            raise ValueError("invalid decoded SmolVLA action chunk")
        from .execution_constraints import project_gripper

        execution = project_gripper(result, gripper_lower_bound)
        committed = (
            self.encode_execution_actions(execution)
            if gripper_lower_bound == "zero"
            else model_actions.copy()
        )
        return ActionChunk(
            result,
            model_actions,
            None if noise is None else noise.detach().cpu().numpy().copy(),
            execution,
            committed,
        )

    def execution_chunk(self, image, state, sequence):
        inferred = self._chunk is None or self._next_index >= self.n_action_steps
        if inferred:
            self._chunk = self.predict(image, state)
            self._next_index = 0
        index = self._next_index
        self._next_index += 1
        return self._chunk, index, inferred

    def select_target(self, absolute_chunk, *, chunk_index=0, new_chunk=True):
        return absolute_chunk[chunk_index].copy()
