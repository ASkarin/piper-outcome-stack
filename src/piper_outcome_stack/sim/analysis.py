"""Geometric cross-checks and honest recorded-feedback / servo comparisons."""

from __future__ import annotations

import csv
import json
import time
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .model import ASSETS, limits, urdf_fk


def alignment_report(sim):
    lo, hi = limits(sim.asset_dir)
    rng = np.random.default_rng(20260908)
    qs = [
        np.zeros(7),
        np.r_[np.deg2rad([-42.488, 132.772, -106.826, -4.781, 43.002, 66.067]), 0.065],
    ]
    qs.extend(rng.uniform(lo, hi, size=(32, 7)))
    errors = []
    for q in qs:
        sim.set_measured(q)
        ref = urdf_fk(q, sim.asset_dir)
        for name in (
            "base_link",
            *[f"link{i}" for i in range(1, 7)],
            "flange_link",
            "gripper_base",
            "gripper_link1",
            "gripper_link2",
        ):
            body = sim.data.body(name)
            actual = np.eye(4)
            actual[:3, :3] = body.xmat.reshape(3, 3)
            actual[:3, 3] = body.xpos - sim.config["base_xyz_m"]
            errors.append(
                (
                    float(np.linalg.norm(actual[:3, 3] - ref[name][:3, 3])),
                    float(Rotation.from_matrix(actual[:3, :3] @ ref[name][:3, :3].T).magnitude()),
                )
            )
    e = np.array(errors)
    sdk = []
    golden = json.loads((ASSETS / "sdk_fk_reference.json").read_text())
    for row in golden["samples"]:
        sim.set_measured([*row["joint_rad"], 0.0])
        actual = sim.flange_matrix()
        rot = Rotation.from_euler("xyz", row["flange_pose_m_rad"][3:]).as_matrix()
        sdk.append(
            {
                "joint_rad": row["joint_rad"],
                "position_difference_m": float(
                    np.linalg.norm(actual[:3, 3] - row["flange_pose_m_rad"][:3])
                ),
                "orientation_difference_rad": float(
                    Rotation.from_matrix(actual[:3, :3] @ rot.T).magnitude()
                ),
            }
        )
    sim.set_measured(np.zeros(7))
    return {
        "sample_count": len(qs),
        "max_position_error_m": float(e[:, 0].max()),
        "max_orientation_error_rad": float(e[:, 1].max()),
        "conversion_passed": bool(e[:, 0].max() <= 1e-6 and e[:, 1].max() <= 1e-6),
        "sdk_comparison": {
            "sdk_commit": golden["sdk_commit"],
            "role": "reported_difference_not_physical_accuracy",
            "samples": sdk,
        },
        "joint6_model_limit_rad": [float(lo[5]), float(hi[5])],
        "hardware_validated": False,
    }


class PlaybackClock:
    def __init__(self):
        self.paused = False
        self.stop = False

    def key(self, key):
        if key == 32:
            self.paused = not self.paused
        if key in (ord("Q"), ord("q"), 256):
            self.stop = True


def replay(sim, record, mode, out, *, window=False):
    if mode not in ("measured", "commanded"):
        raise ValueError("unknown replay mode")
    clock = PlaybackClock()
    viewer_ctx = nullcontext(None)
    if window:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=clock.key)
    target = record.initial_target()
    sim.set_measured(target, recorded=True)
    if mode == "commanded":
        sim.set_target(target)
    index = 0
    start = time.perf_counter()
    render_s = 0.0
    frames = 0
    rows = 0
    next_frame = 0.0
    finished = False
    t = 0.0
    dt = sim.model.opt.timestep
    error_sum = np.zeros(7)
    error_max = np.zeros(7)
    lo, hi = limits(sim.asset_dir)
    excursions = np.maximum(np.maximum(lo - record.samples, record.samples - hi), 0)
    milestones = {0: False, 1: False, 2: False}
    with (out / f"{mode}.csv").open("x", newline="") as stream, viewer_ctx as viewer:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sim_time_s",
                *[f"reference_{i}" for i in range(7)],
                *[f"simulated_{i}" for i in range(7)],
            ]
        )
        if viewer is not None:
            viewer.cam.lookat[:] = [0.1, 0, 0.2]
            viewer.cam.distance = 1.4
        try:
            while t <= record.duration + 1e-12:
                if viewer is not None and (not viewer.is_running() or clock.stop):
                    break
                if clock.paused:
                    viewer.sync()
                    time.sleep(1 / 30)
                    continue
                iteration = time.perf_counter()
                while index < len(record.commands) and record.commands[index][0] <= t:
                    _, kind, value = record.commands[index]
                    if kind == "joints":
                        target[:6] = value
                    else:
                        target[6] = value
                    index += 1
                reference = record.measured(min(t, record.duration))
                if mode == "measured":
                    sim.set_measured(reference, recorded=True)
                    sim.data.time = t
                    sim.record_contacts()
                else:
                    sim.set_target(target)
                actual = sim.action()
                error = actual - reference
                error_sum += error**2
                error_max = np.maximum(error_max, np.abs(error))
                rows += 1
                writer.writerow([t, *reference, *actual])
                if t + 1e-12 >= next_frame:
                    if viewer is not None:
                        viewer.set_texts(
                            (
                                None,
                                None,
                                f"{mode} t={t:.2f}s | Space pause | Q quit\n" + sim.status_text(),
                                "",
                            )
                        )
                        stamp = time.perf_counter()
                        viewer.sync()
                        render_s += time.perf_counter() - stamp
                        frames += 1
                    next_frame += 1 / 30
                # Representative timestamps, not separately certified task endpoints.
                for n, when in enumerate(
                    (0.0, record.duration * 0.6, max(0, record.duration - dt))
                ):
                    if not milestones[n] and t >= when:
                        stamp = time.perf_counter()
                        Image.fromarray(sim.render()).save(out / f"{mode}-{n}.png")
                        render_s += time.perf_counter() - stamp
                        frames += 1
                        milestones[n] = True
                if t + dt > record.duration and t < record.duration:
                    if mode == "commanded":
                        # Keep physics step fixed; endpoint comparisons use the last full step.
                        finished = True
                        break
                    t = record.duration
                else:
                    if mode == "commanded":
                        sim.step()
                    t += dt
                if viewer is not None:
                    time.sleep(max(0, dt - (time.perf_counter() - iteration)))
            else:
                finished = True
        finally:
            stream.flush()
            from .cli import save_json

            summary = {
                "mode": mode,
                "completed": finished,
                "rows": rows,
                "original_samples": len(record.samples),
                "measured_limit_excursion_samples": int(np.any(excursions > 0, axis=1).sum()),
                "measured_max_limit_excursion_rad_m": excursions.max(axis=0).tolist(),
                "simulated_duration_s": min(t, record.duration),
                "wall_duration_s": time.perf_counter() - start,
                "render_wall_s": render_s,
                "rendered_frames": frames,
                "contacts": dict(sim.contact_counts),
                "max_penetration_m": sim.max_penetration_m,
                "final_rad_m": sim.action().tolist(),
                "comparison": "interpolated_feedback_reproduction_not_tracking_accuracy"
                if mode == "measured"
                else "unidentified_servo_vs_recorded_feedback",
                "rmse_rad_m": None
                if mode == "measured"
                else np.sqrt(error_sum / max(rows, 1)).tolist(),
                "max_error_rad_m": None if mode == "measured" else error_max.tolist(),
            }
            summary["real_time_factor"] = summary["simulated_duration_s"] / max(
                summary["wall_duration_s"], 1e-9
            )
            summary["render_fps"] = frames / max(render_s, 1e-9)
            save_json(out / f"{mode}-summary.json", summary)
    if not finished:
        raise KeyboardInterrupt("playback stopped before the recorded endpoint")
    return summary


def plot_tracking(csv_path, path):
    with csv_path.open() as f:
        a = np.loadtxt(f, delimiter=",", skiprows=1)
    image = Image.new("RGB", (1000, 1120), "white")
    draw = ImageDraw.Draw(image)
    for j in range(7):
        top = 20 + j * 155
        draw.text(
            (10, top), f"J{j + 1} (rad)" if j < 6 else "Gripper total width (m)", fill="black"
        )
        ref, sim = a[:, 1 + j], a[:, 8 + j]
        low = min(ref.min(), sim.min())
        high = max(ref.max(), sim.max())
        span = max(high - low, 1e-6)
        draw.rectangle((60, top + 20, 970, top + 130), outline="gray")
        for y, color in ((ref, "blue"), (sim, "red")):
            stride = max(1, len(a) // 2000)
            points = [
                (60 + 910 * t / max(a[-1, 0], 1e-9), top + 125 - 100 * (v - low) / span)
                for t, v in zip(a[::stride, 0], y[::stride])
            ]
            draw.line(points, fill=color, width=2)
        draw.text((650, top), "blue: recorded   red: unidentified simulation", fill="black")
    image.save(path)
