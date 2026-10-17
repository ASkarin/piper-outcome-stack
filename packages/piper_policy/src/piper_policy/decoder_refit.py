"""Bounded ACT decoder tuning on cached inference encoder outputs, without device I/O."""

import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file

from .policy_execution import ACTChunkPredictor
from .positive_act import inverse_positive_width, parameterize


class ACTDecoderRefiner:
    """Keep the inference zero latent and eval-mode dropout while enabling decoder gradients."""

    def __init__(self, predictor, checkpoint):
        self.predictor = predictor
        self.checkpoint = Path(checkpoint)
        self.policy = predictor.model
        self.model = self.policy.model
        self.config = self.policy.config
        if len(self.model.decoder.layers) != 1:
            raise ValueError("this refit supports the checkpoint's single decoder layer")
        self.policy.eval().requires_grad_(False)
        self.model.decoder.requires_grad_(True)
        self.model.action_head.requires_grad_(True)
        self.device = self.model.action_head.weight.device
        paths = list(self.checkpoint.glob("policy_postprocessor*unnormalizer*.safetensors"))
        if len(paths) != 1:
            raise ValueError("one checkpoint action unnormalizer required")
        stats = load_file(str(paths[0]))
        self.mean = stats["action.mean"].float().to(self.device)
        self.std = stats["action.std"].float().to(self.device)
        if self.mean.shape != (7,) or self.std.shape != (7,) or not (self.std > 0).all():
            raise ValueError("seven positive normalization scales required")

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls(ACTChunkPredictor.from_checkpoint(checkpoint), checkpoint)

    def trainable_names(self):
        return [name for name, value in self.policy.named_parameters() if value.requires_grad]

    def extract_context(self, images, states):
        captured = []

        def capture(module, args, kwargs):
            captured.append((args[1].detach().cpu(), kwargs["encoder_pos_embed"].detach().cpu()))

        hook = self.model.decoder.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            prediction = self.predictor.predict_batch(images, states)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("expected one decoder call")
        # Clone outside inference_mode: cached tensors must support backward in the decoder.
        memory, position = captured[0]
        # ACT broadcasts the common positional encoding across the batch.
        position = position.expand(-1, memory.shape[1], -1)
        return memory.transpose(0, 1).clone(), position.transpose(0, 1).clone(), prediction

    def _raw_prediction(self, memory, position, states):
        memory = memory.to(self.device)
        position = position.to(self.device)
        states = states.to(self.device)
        if memory.ndim != 3 or position.shape != memory.shape or states.shape != (len(memory), 7):
            raise ValueError("expected N,L,D memory/positions and N,7 states")
        query = torch.zeros(
            self.config.chunk_size,
            len(memory),
            self.config.dim_model,
            dtype=memory.dtype,
            device=self.device,
        )
        decoded = self.model.decoder(
            query,
            memory.transpose(0, 1),
            encoder_pos_embed=position.transpose(0, 1),
            decoder_pos_embed=self.model.decoder_pos_embed.weight.unsqueeze(1),
        )
        return self.model.action_head(decoded.transpose(0, 1))

    def _parameterize(self, raw, states):
        states = states.to(self.device)
        normalized_state = torch.zeros_like(states)
        normalized_state[:, 6] = (
            states[:, 6] - self.config.gripper_state_mean
        ) / self.config.gripper_state_std
        return parameterize(raw, normalized_state, self.config)

    def normalized_prediction(self, memory, position, states):
        return self._parameterize(self._raw_prediction(memory, position, states), states)

    def predict(self, memory, position, states):
        return (
            self.normalized_prediction(memory, position, states) * self.std
            + self.mean
            + states.to(self.device)[:, None]
        )

    def loss(self, memory, position, states, truth, valid, parent, old, consistency_weight=0.5):
        prediction = self.normalized_prediction(memory, position, states)
        valid = valid.to(self.device)
        if valid.dtype != torch.bool or valid.shape != prediction.shape[:2] or not valid.any():
            raise ValueError("nonempty boolean valid-action mask required")
        anchor = states.to(self.device)[:, None]
        target = (truth.to(self.device) - anchor - self.mean) / self.std
        if target.shape != prediction.shape or parent.shape != prediction.shape:
            raise ValueError("teacher and parent chunks must match decoder output")
        teacher_loss = (prediction - target).abs()[valid].mean()
        old_mask = valid & old.to(self.device)[:, None]
        consistency = prediction.new_zeros(())
        if old_mask.any():
            reference = (parent.to(self.device).detach() - anchor - self.mean) / self.std
            consistency = (prediction - reference).abs()[old_mask].mean()
        return teacher_loss + consistency_weight * consistency, teacher_loss, consistency

    def execution_loss(
        self,
        memory,
        position,
        states,
        truth,
        valid,
        parent,
        old,
        *,
        front_steps=5,
        consistency_weight=0.5,
    ):
        """Mean of full, front and first-action L1; gripper labels use inverse positivity.

        Teacher and old-sample consistency use the same three equally weighted
        windows. Consistency stays in normalized physical-action space.
        """
        raw = self._raw_prediction(memory, position, states)
        valid = valid.to(self.device)
        if (
            valid.dtype != torch.bool
            or valid.shape != raw.shape[:2]
            or not valid[:, 0].all()
            or not 1 <= front_steps <= raw.shape[1]
        ):
            raise ValueError("valid first actions and a front window within the chunk required")
        anchor = states.to(self.device)[:, None]
        truth = truth.to(self.device)
        if truth.shape != raw.shape or parent.shape != raw.shape:
            raise ValueError("teacher and parent chunks must match decoder output")
        target = (truth - anchor - self.mean) / self.std
        grip_target = torch.zeros_like(raw[..., 6])
        unconstrained = inverse_positive_width(truth[..., 6][valid].double(), self.config)
        grip_target[valid] = (
            (
                unconstrained
                - anchor[..., 6].expand_as(valid)[valid]
                - self.config.gripper_action_mean
            )
            / self.config.gripper_action_std
        ).to(raw.dtype)
        target[..., 6] = grip_target
        error = (raw - target).abs()
        full = error[valid].mean()
        front = error[:, :front_steps][valid[:, :front_steps]].mean()
        first = error[:, 0].mean()
        teacher = (full + front + first) / 3
        prediction = self._parameterize(raw, states)
        old_mask = valid & old.to(self.device)[:, None]
        consistency = raw.new_zeros(())
        parent_full = parent_front = parent_first = consistency
        old_teacher = consistency
        if old_mask.any():
            old_teacher = (
                error[old_mask].mean()
                + error[:, :front_steps][old_mask[:, :front_steps]].mean()
                + error[:, 0][old_mask[:, 0]].mean()
            ) / 3
            reference = (parent.to(self.device).detach() - anchor - self.mean) / self.std
            parent_error = (prediction - reference).abs()
            parent_full = parent_error[old_mask].mean()
            parent_front = parent_error[:, :front_steps][old_mask[:, :front_steps]].mean()
            parent_first = parent_error[:, 0][old_mask[:, 0]].mean()
            consistency = (parent_full + parent_front + parent_first) / 3
        return teacher + consistency_weight * consistency, {
            "teacher_full": full,
            "teacher_front": front,
            "teacher_first": first,
            "teacher_combined": teacher,
            "parent_consistency": consistency,
            "parent_full": parent_full,
            "parent_front": parent_front,
            "parent_first": parent_first,
            "old_teacher": old_teacher,
        }

    def physical_joint_losses(self, memory, position, states, truth, valid):
        """Separate joint full-chunk and first-action MAE in radians, no gripper averaging."""
        prediction = self.predict(memory, position, states)
        valid = valid.to(self.device)
        truth = truth.to(self.device)
        if (
            truth.shape != prediction.shape
            or valid.shape != prediction.shape[:2]
            or valid.dtype != torch.bool
            or not valid[:, 0].all()
        ):
            raise ValueError("matching chunks and valid first actions required")
        error = (prediction[..., :6] - truth[..., :6]).abs()
        return {"joint_full": error[valid].mean(), "joint_first": error[:, 0].mean()}

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.policy.save_pretrained(output)
        for source in self.checkpoint.iterdir():
            if (
                source.name.startswith(("policy_preprocessor", "policy_postprocessor"))
                or source.name == "representation.json"
            ):
                shutil.copy2(source, output / source.name)
        (output / "decoder_refit.json").write_text(
            json.dumps(
                dict(
                    source_checkpoint=str(self.checkpoint),
                    trainable_names=self.trainable_names(),
                    scope="inference checkpoint; new local optimizer, not upstream optimizer resume",
                    feature_path="RGB/state only, zero VAE latent, eval mode including decoder dropout",
                ),
                indent=2,
            )
        )
