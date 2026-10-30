"""Real3060 native-RTC/padding checks on recorded inputs; no robot connection."""

from pathlib import Path
import json
import time
import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import numpy as np
import torch
from piper_policy.smolvla import SmolVLAChunkPredictor

BASE = Path(__file__).resolve().parent


def one(arm, operator):
    with np.load(BASE / "reference-inputs" / (arm + "-green-left.npz"), allow_pickle=False) as data:
        image, state = data["images"][0], data["states"][0]
    predictor = SmolVLAChunkPredictor.from_checkpoint(
        operator["models"][arm]["checkpoint"],
        task=operator["policies"]["green-left"]["instruction"],
        vlm_assets=operator["vlm_assets"],
    )
    for _ in range(3):
        predictor.predict(image, state)
    predictor.configure_rtc(True, num_steps=10, guidance="native_joint6")
    assert tuple(predictor.model.config.action_feature.shape) == (7,)
    assert predictor.model.config.max_action_dim == predictor.model.config.max_state_dim == 32
    processor = type(predictor.model.model.rtc_processor).__name__
    assert processor == "NativeJointRTCProcessor"
    flow = predictor.model.model
    original = flow.denoise_step
    observed = []
    active_noise = None

    def check(prefix_pad_masks, past_key_values, x_t, timestep):
        expected = timestep[:, None, None] * active_noise[..., 7:]
        error = float((x_t[..., 7:] - expected).abs().max())
        if error != 0:
            raise RuntimeError(f"RTC structural padding left t*epsilon path: {error}")
        observed.append(error)
        return original(prefix_pad_masks, past_key_values, x_t, timestep)

    flow.denoise_step = check
    rng = np.random.default_rng(42)
    seconds = []
    first = second = None
    try:
        epsilon = rng.standard_normal((1, 50, 32)).astype(np.float32)
        active_noise = torch.as_tensor(epsilon, device=predictor.model.config.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        first = predictor.predict_chunk(image, state, noise=epsilon, gripper_lower_bound="zero")
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        epsilon = rng.standard_normal((1, 50, 32)).astype(np.float32)
        active_noise = torch.as_tensor(epsilon, device=predictor.model.config.device)
        prefix = first.committed_model_actions[25:].copy()
        torch.cuda.synchronize()
        started = time.perf_counter()
        second = predictor.predict_chunk(
            image,
            state,
            noise=epsilon,
            previous_model_actions=prefix,
            inference_delay=20,
            overlap_steps=25,
            gripper_lower_bound="zero",
        )
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
    finally:
        flow.denoise_step = original
    assert len(observed) == 20
    assert second.actions.shape == (50, 7) and np.isfinite(second.actions).all()
    np.testing.assert_array_equal(second.execution_actions[:, :6], second.actions[:, :6])
    assert np.all(second.execution_actions[:, 6] >= 0)
    np.savez_compressed(
        BASE / (arm + "-rtc-padding.npz"),
        first_raw=first.actions,
        second_raw=second.actions,
        second_execution=second.execution_actions,
        prefix_model_actions=prefix,
    )
    return dict(
        model=arm,
        status="passed",
        processor=processor,
        denoising_calls_checked=len(observed),
        padding_tail_max_error=max(observed),
        guided_action_indices=list(range(6)),
        joint_targets_preserved=True,
        gripper_projection="zero",
        lowpass=False,
        inference_seconds_without_then_with_prefix=seconds,
        hardware_io=False,
    )


def main():
    torch.set_num_threads(4)
    operator = json.loads((BASE / "operator-config.json").read_text())
    result = dict(
        status="passed",
        models=[one(arm, operator) for arm in ["original", "flat"]],
        hardware_io=False,
        scope="recorded single input,explicit prior25-step prefix; not real-robot or full closed-loop validation",
    )
    (BASE / "rtc-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
