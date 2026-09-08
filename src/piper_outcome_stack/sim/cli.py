"""Simulation-only command line. Hardware plugins are never imported."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog="piper-outcome-stack sim")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "view", "target", "replay", "compare"):
        q = sub.add_parser(name)
        q.add_argument("--config", type=Path)
        q.add_argument("--output", type=Path, required=name in ("target", "replay", "compare"))
        q.add_argument("--headless", action="store_true")
        if name in ("view", "target"):
            q.add_argument("--duration", type=float, default=0 if name == "view" else 2)
        if name == "target":
            q.add_argument("--joint-deg", nargs=6, type=float, required=True)
            q.add_argument("--gripper-mm", type=float, required=True)
        if name in ("replay", "compare"):
            q.add_argument("--report", type=Path, required=True)
        if name == "replay":
            q.add_argument("--mode", choices=("measured", "commanded"), required=True)
    return p


def save_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv=None):
    args = parser().parse_args(argv)
    headless = args.headless or args.command in ("doctor", "compare")
    if not headless and not os.environ.get("DISPLAY"):
        print("No DISPLAY: run in the local desktop terminal, or pass --headless.", file=sys.stderr)
        return 2
    os.environ["MUJOCO_GL"] = "egl" if headless and sys.platform == "linux" else "glfw"
    out = args.output
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "domain": "sim", "hardware_io": False, "command": args.command}
    sim = None
    try:
        import numpy as np
        from PIL import Image
        from .model import ROOT, SOURCE_COMMIT
        from .runtime import DEFAULT_CONFIG, Simulation, load_config
        from .analysis import alignment_report, plot_tracking, replay
        from piper_outcome_stack.sim2real.recording import read_record

        cfg = load_config(args.config or DEFAULT_CONFIG)
        report.update(
            config=cfg,
            source_commit=subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            source_dirty=bool(
                subprocess.check_output(
                    ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
                )
            ),
            model_source_commit=SOURCE_COMMIT,
            mujoco_version=importlib.metadata.version("mujoco"),
            python=sys.version,
            render_backend=os.environ["MUJOCO_GL"],
            scene_calibrated=False,
        )
        sim = Simulation(cfg)
        report["servo"] = sim.servo
        if args.command == "doctor":
            start = time.perf_counter()
            image = sim.render()
            report.update(
                render_shape=list(image.shape),
                visible_robot_pixels=sim.visible_robot_pixels(),
                render_wall_s=time.perf_counter() - start,
                display_available=bool(os.environ.get("DISPLAY")),
                alignment=alignment_report(sim),
            )
            if out is not None:
                Image.fromarray(image).save(out / "camera.png")
            if report["visible_robot_pixels"] == 0:
                raise ValueError(
                    "virtual camera does not see the robot; inspect camera intrinsics/pose"
                )
        elif args.command in ("view", "target"):
            if args.duration < 0 or not np.isfinite(args.duration):
                raise ValueError("duration must be finite and nonnegative")
            if headless and args.duration == 0:
                raise ValueError("headless target/view requires a finite positive duration")
            target = (
                np.r_[np.deg2rad(args.joint_deg), args.gripper_mm / 1000]
                if args.command == "target"
                else np.zeros(7)
            )
            sim.set_measured(target)
            sim.set_target(target)
            sim.record_contacts()
            report.update(
                mode="kinematic_pose_preview",
                target_rad_m=target.tolist(),
                flange_matrix=sim.flange_matrix().tolist(),
            )
            if out is not None:
                Image.fromarray(sim.render()).save(out / "camera.png")
            if not headless:
                show_pose(sim, args.duration)
        else:
            record = read_record(args.report)
            report.update(
                source_report=record.source,
                source_status=record.source_status,
                original_samples=len(record.samples),
                recorded_duration_s=record.duration,
                alignment=alignment_report(sim),
            )
            modes = ("measured", "commanded") if args.command == "compare" else (args.mode,)
            report["replays"] = {}
            for mode in modes:
                sim.close()
                sim = Simulation(cfg)
                report["replays"][mode] = replay(sim, record, mode, out, window=not headless)
            if args.command == "compare":
                plot_tracking(out / "commanded.csv", out / "tracking.png")
        report["status"] = "passed"
        if report.get("alignment", {}).get("conversion_passed") is False:
            report["status"] = "failed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if sim is not None:
            report["contacts"] = {
                "counts": sim.contact_counts,
                "max_penetration_m": sim.max_penetration_m,
            }
            sim.close()
        if out is not None:
            save_json(out / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


def show_pose(sim, duration):
    import mujoco.viewer

    start = time.monotonic()
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [0.1, 0, 0.2]
        viewer.cam.distance = 1.4
        while viewer.is_running() and (duration == 0 or time.monotonic() - start < duration):
            viewer.set_texts((None, None, sim.status_text(), ""))
            viewer.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    raise SystemExit(main())
