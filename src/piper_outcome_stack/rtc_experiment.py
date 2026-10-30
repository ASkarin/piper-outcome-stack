"""Fixed-checkpoint RTC feasibility, paired observations and six simulation diagnostics."""

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from piper_policy.smolvla import SmolVLAChunkPredictor, TASK_A, IMAGE_FEATURE
from .smolvla_training import make_views, write_json
from .smolvla_evaluate import action_errors, command_envelope_counts


def observation(dataset, index):
    item = dataset[index]
    return item, item[IMAGE_FEATURE].permute(1, 2, 0).numpy(), item["observation.state"].numpy()


def noise(rng):
    return rng.standard_normal((1, 50, 32)).astype(np.float32)


def timed(predictor, image, state, **options):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = predictor.predict_chunk(image, state, **options)
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def run(spec_path):
    spec = json.loads(spec_path.read_text())
    root = Path(spec["output"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + spec["budget_seconds"]
    status = {"status": "running", "hardware_io": False, "training": False, "spec": spec}
    write_json(root / "status.json", status)

    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed RTC experiment budget exhausted")

    try:
        torch.set_num_threads(4)
        data_spec = json.loads(Path(spec["data_spec"]).read_text())
        _, dev = make_views(data_spec)
        predictor = SmolVLAChunkPredictor.from_checkpoint(spec["checkpoint"], task=TASK_A)
        versions = [p._version for p in predictor.model.parameters()]
        rng = np.random.default_rng(spec["seed"])
        item, image, state = observation(dev, dev.starts[0])
        _, next_image, next_state = observation(dev, dev.starts[0] + 25)
        profiles = []
        selected = None
        for steps in (10, 5):
            budget()
            predictor.configure_rtc(False, num_steps=steps)
            fixed_noise = noise(rng)
            old = predictor.predict_chunk(image, state, noise=fixed_noise)
            predictor.configure_rtc(True, num_steps=steps)
            identical = predictor.predict_chunk(image, state, noise=fixed_noise)
            difference = float(np.max(np.abs(old.actions - identical.actions)))
            if difference > 1e-6:
                raise ValueError("unguided RTC differs from ordinary same-noise inference")
            values = []
            for i in range(25):
                budget()
                _, elapsed = timed(
                    predictor,
                    next_image,
                    next_state,
                    noise=noise(rng),
                    previous_model_actions=old.model_actions[25:],
                    inference_delay=12,
                    overlap_steps=25,
                )
                if i >= 5:
                    values.append(elapsed)
            entry = {
                "num_steps": steps,
                "warmups": 5,
                "seconds": values,
                "unguided_identity_max_difference": difference,
                "initial_delay": math.ceil(max(values) / 0.02),
            }
            profiles.append(entry)
            write_json(root / "latency-profile.json", profiles)
            if max(values) <= 0.5:
                selected = entry
                break
        if selected is None:
            status.update(
                status="infeasible", stop_reason="RTC latency exceeds half of prediction horizon"
            )
            return
        steps = selected["num_steps"]
        delay = selected["initial_delay"]
        status.update(num_steps=steps, initial_delay=delay, stage="paired")
        write_json(root / "status.json", status)
        pairdir = root / "pairs"
        pairdir.mkdir()
        rows = []
        old_predictions = []
        naive_predictions = []
        rtc_predictions = []
        targets = []
        states = []
        naive_times = []
        rtc_times = []
        for episode, (start, end) in enumerate(zip(dev.starts, dev.ends)):
            if end - start < 75:
                raise ValueError("development episode is too short for 25+50 pair")
            indices = np.linspace(start, end - 75, 8, dtype=np.int64)
            if len(set(indices.tolist())) != 8:
                raise ValueError("development pair anchors are not unique")
            for anchor in indices:
                budget()
                i = len(rows)
                first, img1, state1 = observation(dev, int(anchor))
                second, img2, state2 = observation(dev, int(anchor) + 25)
                pair_rng = np.random.default_rng(spec["seed"] + i)
                old_noise, new_noise = noise(pair_rng), noise(pair_rng)
                predictor.configure_rtc(False, num_steps=steps)
                old = predictor.predict_chunk(img1, state1, noise=old_noise)
                naive, nt = timed(predictor, img2, state2, noise=new_noise)
                predictor.configure_rtc(True, num_steps=steps)
                rtc, rt = timed(
                    predictor,
                    img2,
                    state2,
                    noise=new_noise,
                    previous_model_actions=old.model_actions[25:],
                    inference_delay=delay,
                    overlap_steps=25,
                )
                gt = second["action"].numpy()
                if second["action_is_pad"].any():
                    raise ValueError("paired evaluation unexpectedly padded")
                np.savez_compressed(
                    pairdir / f"{i:03d}.npz",
                    old_noise=old_noise,
                    new_noise=new_noise,
                    first_image=img1,
                    second_image=img2,
                    first_state=state1,
                    second_state=state2,
                    old_model_actions=old.model_actions,
                    old_physical=old.actions,
                    naive_model_actions=naive.model_actions,
                    naive_physical=naive.actions,
                    rtc_model_actions=rtc.model_actions,
                    rtc_physical=rtc.actions,
                    targets=gt,
                    anchor=int(anchor),
                    second_index=int(anchor) + 25,
                    naive_seconds=nt,
                    rtc_seconds=rt,
                    delay=delay,
                )
                rows.append(
                    {
                        "pair": i,
                        "episode": episode,
                        "first_retained_index": int(anchor),
                        "second_retained_index": int(anchor) + 25,
                        "lineage": dev.lineage[episode],
                    }
                )
                old_predictions.append(old.actions)
                naive_predictions.append(naive.actions)
                rtc_predictions.append(rtc.actions)
                targets.append(gt)
                states.append(state2)
                naive_times.append(nt)
                rtc_times.append(rt)
        old, naive, rtc, gt, states = map(
            np.asarray, [old_predictions, naive_predictions, rtc_predictions, targets, states]
        )
        valid = np.ones(gt.shape[:2], dtype=bool)
        comparison = {
            "pairs": len(rows),
            "num_steps": steps,
            "delay": delay,
            "continuity_reference": "old model predictions: proxy, not ground truth",
            "ground_truth": "unchanged retained development dataset physical actions",
        }
        for name, pred, times in [("naive", naive, naive_times), ("rtc", rtc, rtc_times)]:
            jump = pred[:, delay] - old[:, 25 + delay - 1]
            comparison[name] = {
                "ground_truth_errors": action_errors(pred, gt, valid),
                "overlap_proxy_errors": action_errors(pred[:, :25], old[:, 25:], valid[:, :25]),
                "boundary_joint_max_deg": np.rad2deg(abs(jump[:, :6]).max(1)).tolist(),
                "boundary_gripper_mm": (abs(jump[:, 6]) * 1000).tolist(),
                "first_command_envelope": command_envelope_counts(
                    states, pred, data_spec["evaluation"]["command_limits"]
                ),
                "inference_ms_p50_p95_p99": (np.percentile(times, [50, 95, 99]) * 1000).tolist(),
            }
        np.savez_compressed(
            root / "paired-predictions.npz",
            old=old,
            naive=naive,
            rtc=rtc,
            targets=gt,
            states=states,
            valid=valid,
            naive_seconds=naive_times,
            rtc_seconds=rtc_times,
        )
        write_json(root / "pairs.json", rows)
        write_json(root / "paired-summary.json", comparison)
        if any(p.requires_grad or p.grad is not None for p in predictor.model.parameters()):
            raise ValueError("inference acquired parameter gradients")
        if versions != [p._version for p in predictor.model.parameters()]:
            raise ValueError("inference modified parameter versions")
        status["parameter_versions_unchanged"] = True
        del predictor, dev
        gc.collect()
        torch.cuda.empty_cache()
        status["stage"] = "simulation"
        status["simulations"] = []
        write_json(root / "status.json", status)
        for seed in spec["simulation_seeds"]:
            for method in ["async-measured", "rtc-measured"]:
                budget()
                out = root / f"{method}-seed{seed}"
                cmd = [
                    spec["simulation_python"],
                    "-m",
                    "piper_sim.cli",
                    "policy",
                    "--policy-type",
                    "smolvla",
                    "--task",
                    TASK_A,
                    "--action-steps",
                    "25",
                    "--execution",
                    method,
                    "--denoising-steps",
                    str(steps),
                    "--rtc-initial-delay",
                    str(delay),
                    "--temporal-ensemble-coeff",
                    "none",
                    "--config",
                    spec["simulation_config"],
                    "--checkpoint",
                    spec["checkpoint"],
                    "--layout",
                    "1",
                    "--object",
                    "green",
                    "--region",
                    "placement",
                    "--duration",
                    "40",
                    "--seed",
                    str(seed),
                    "--output",
                    str(out),
                ]
                with (root / f"{method}-seed{seed}.log").open("x") as log:
                    result = subprocess.run(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=max(1, deadline - time.monotonic()),
                    )
                entry = {
                    "method": method,
                    "seed": seed,
                    "exit_code": result.returncode,
                    "command": cmd,
                }
                status["simulations"].append(entry)
                write_json(root / "status.json", status)
                if result.returncode:
                    raise RuntimeError(f"simulation process failed: {method} seed{seed}")
        status.update(status="complete", stop_reason="declared_diagnostics_completed")
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        write_json(root / "status.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    run(parser.parse_args().spec)


if __name__ == "__main__":
    main()
