"""Evaluate a fixed C checkpoint at deployment precision and export independent controller targets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from piper_outcome_stack.smolvla_evaluate import evaluate
from piper_outcome_stack.task_c_reference import act_reference_outputs, fp32_runtime
from piper_policy.policy_execution import ACTChunkPredictor
from piper_policy.smolvla import SmolVLAChunkPredictor
from piper_policy.smolvla_execution import AsyncSmolVLAExecutor


def file_digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(spec, variant, recorded_inputs, output):
    run = Path(spec["run_root"])
    training = json.loads((run / "training-status.json").read_text())
    if training["status"] != "complete" or training["updates"] != spec["budget"]["max_steps"]:
        raise ValueError("candidate is not the completed fixed-budget checkpoint")
    checkpoint = Path(training["final_checkpoint"])
    family = "act" if variant == "act" else "smolvla"
    evaluate(spec, checkpoint, output / "development-fp32", family, deployment_fp32=True)
    if variant == "original":
        return

    with np.load(recorded_inputs) as recorded:
        ref = {k: recorded[k] for k in ("images", "states", "tasks", "seeds", "rtc_seeds")}
    tasks = set(ref["tasks"].tolist())
    if len(tasks) != 1:
        raise ValueError("C reference inputs require one fixed instruction")
    raw, second, projected = [], [], []
    with fp32_runtime():
        if family == "act":
            policy = ACTChunkPredictor.from_checkpoint(checkpoint)
            policy.configure_action_steps(1)
            policy.configure_temporal_ensemble(0.01)
        else:
            policy = SmolVLAChunkPredictor.from_checkpoint(
                checkpoint, task=next(iter(tasks)), vlm_assets=spec["vlm_assets"]
            )
        policy.model.float()
        policy.model.config.use_amp = False
        if family == "smolvla":
            rtc = AsyncSmolVLAExecutor(policy, request_stride=25, joint_lowpass=False)
        for i, (image, state) in enumerate(zip(ref["images"], ref["states"], strict=True)):
            if family == "act":
                chunk, goals = act_reference_outputs(policy, image, state)
                raw.append(chunk)
                second.append(goals)
            else:
                torch.manual_seed(int(ref["seeds"][i]))
                first = policy.predict_chunk(image, state, gripper_lower_bound="zero")
                prefix = first.committed_model_actions[rtc.n_action_steps :].copy()
                torch.manual_seed(int(ref["rtc_seeds"][i]))
                next_chunk = policy.predict_chunk(
                    image,
                    state,
                    previous_model_actions=prefix,
                    inference_delay=rtc.rtc_initial_delay,
                    overlap_steps=len(prefix),
                    gripper_lower_bound="zero",
                )
                raw.append(first.actions)
                second.append(next_chunk.actions)
                projected.append(next_chunk.execution_actions)
        if family == "smolvla":
            rtc.reset_execution()
        else:
            policy.reset_execution()

    ref.update(
        policy_type=np.asarray(family),
        inference_precision=np.asarray("fp32"),
        expected_absolute_chunks=np.stack(raw),
    )
    if family == "act":
        del ref["seeds"], ref["rtc_seeds"]
        ref["expected_ensemble_targets"] = np.stack(second)
    else:
        ref.update(
            expected_rtc_chunks=np.stack(second), expected_rtc_execution_chunks=np.stack(projected)
        )
    reference = output / "push_to_G-reference.npz"
    with reference.open("xb") as f:
        np.savez_compressed(f, **ref)
    handoff = dict(
        run_id=spec["run_id"],
        variant=variant,
        policy_type=family,
        step=training["updates"],
        checkpoint=str(checkpoint),
        model_files={
            p.relative_to(checkpoint).as_posix(): file_digest(p)
            for p in sorted(checkpoint.rglob("*"))
            if p.is_file()
        },
        reference=str(reference),
        reference_sha256=file_digest(reference),
        recorded_inputs=str(recorded_inputs),
        dataset_manifest_sha256=spec["data_manifest_sha256"],
        source_device=torch.cuda.get_device_name(),
    )
    if family == "smolvla":
        vlm = Path(spec["vlm_assets"])
        handoff.update(
            vlm_assets=str(vlm),
            vlm_files={p.name: file_digest(p) for p in sorted(vlm.iterdir()) if p.is_file()},
        )
    path = output / "model-return.json"
    with path.open("x") as f:
        json.dump(handoff, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--variant", choices=("original", "flat", "act"), required=True)
    parser.add_argument("--recorded-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.spec.read_text()), args.variant, args.recorded_inputs, args.output)
