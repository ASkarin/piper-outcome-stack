"""Five normal-plugin read-only sessions, with explicit development/acceptance provenance."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from piper_read_only_probe import snapshot, validate_snapshot


def runtime_provenance(root: Path, plugin_path: Path, run_kind: str) -> dict:
    if not plugin_path.is_relative_to(root):
        raise RuntimeError(f"plugin was imported outside the selected environment: {plugin_path}")
    result = {"run_kind": run_kind, "plugin_source": str(plugin_path)}
    if run_kind == "acceptance":
        if not (root / ".piper-release-accepted").is_file():
            raise RuntimeError("formal acceptance requires an accepted immutable release")
        result["runtime_release"] = str(root)
    elif run_kind == "development":

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], text=True)

        result["development_source"] = {
            "root": str(root),
            "commit": git("rev-parse", "HEAD").strip(),
            "diff": git("diff", "HEAD", "--"),
            # Include the actual plugin sources, including any untracked .py files.
            "plugin_files": {
                str(path.relative_to(root)): path.read_text(encoding="utf-8")
                for path in sorted(plugin_path.parent.rglob("*.py"))
            },
        }
    else:
        raise ValueError("unknown read-only run kind")
    return result


def run_cycles(robot_factory, expected_identity, report, sample_count=30, interval_s=0.1):
    cycles = report["cycles"] = []
    for index in range(1, 6):
        cycle = {
            "cycle": index,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "stage": "construct",
            "status": "started",
            "samples": [],
        }
        cycles.append(cycle)
        robot = robot_factory()
        threads = None
        try:
            cycle["stage"] = "connect"
            started = time.monotonic()
            robot.connect()
            cycle["connect_s"] = time.monotonic() - started
            cycle["firmware_identity"] = robot.firmware_identity
            if robot.firmware_identity != expected_identity:
                raise RuntimeError("live firmware identity changed from the successful probe")
            context = robot._arm.get_context()
            threads = (context._read_th, context._monitor_th, context.fps.thread)
            if any(thread is None for thread in threads):
                raise RuntimeError("an expected SDK receive/monitor/FPS thread is absent")
            cycle["stage"] = "read"
            for _ in range(sample_count):
                observation = robot.get_observation()
                raw = snapshot(robot._arm, robot._gripper)
                cycle["samples"].append(
                    {
                        "observation": observation,
                        "telemetry": asdict(robot.last_feedback_telemetry),
                        "raw_feedback": raw,
                    }
                )
                validate_snapshot(raw, raw["host_wall_s"] - robot.config.feedback_timeout_s)
                time.sleep(interval_s)
            cycle["stage"] = "disconnect"
        except (Exception, KeyboardInterrupt) as exc:
            cycle["status"] = "failed"
            cycle["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            robot.disconnect()
            cycle["connected_after_disconnect"] = robot.is_connected
            cycle["state_after_disconnect"] = robot.state.value
            cycle["sdk_threads_alive_after_disconnect"] = (
                None
                if threads is None
                else [thread.name for thread in threads if thread.is_alive()]
            )
            cycle["finished_at_utc"] = datetime.now(UTC).isoformat()
            if cycle["connected_after_disconnect"] or cycle["sdk_threads_alive_after_disconnect"]:
                cycle["status"] = "failed"
                raise RuntimeError("plugin retained a connection or SDK thread after disconnect")
        cycle["stage"] = "complete"
        cycle["status"] = "passed"
        print(
            f"Read-only cycle {index}/5 passed; disconnected and SDK threads stopped.", flush=True
        )
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("acceptance", "development"), default="acceptance")
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("use piper-socketcan exec to run under the administrator's ordinary UID")
    if not os.path.samefile("/proc/self/ns/net", "/run/netns/piper-can"):
        parser.error("the process must run inside piper-can")
    prior = json.loads(args.identity_report.read_text())
    if prior["status"] != "feedback_received_motors_disabled":
        parser.error("identity report must be the successful initial read-only probe")
    report = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "started",
        "run_kind": args.run_kind,
        "interface": args.interface,
        "expected_firmware_identity": prior["firmware_identity"],
        "firmware_driver": prior["selected_driver"],
        "diagnostic_feedback_timeout_s": 1.0,
        "diagnostic_timeout_scope": "read-only connection/feedback inspection, not frozen motion limits",
        "samples_per_cycle": 30,
        "sample_interval_s": 0.1,
        "python": sys.executable,
        "script": str(Path(__file__).resolve()),
        "motor_enable_executed": False,
        "motion_executed": False,
    }
    with args.output.open("x", encoding="utf-8") as output:
        try:
            from lerobot_robot_outcome_piper.config import OutcomePiperConfig
            from lerobot_robot_outcome_piper import robot as plugin_module

            runtime_root = Path(sys.prefix).resolve().parent
            plugin_path = Path(plugin_module.__file__).resolve()
            report.update(runtime_provenance(runtime_root, plugin_path, args.run_kind))
            report["dependencies"] = {}
            expected = {
                "lerobot": "30da8e687a6dfc617fcd94afc367ac7071c376ce",
                "pyAgxArm": "799b8412fbe8b9156bc9892d3dbeb2df7e98be71",
            }
            for name, commit in expected.items():
                dist = importlib.metadata.distribution(name)
                origin = json.loads(dist.read_text("direct_url.json"))
                actual = origin["vcs_info"]["commit_id"]
                report["dependencies"][name] = {"version": dist.version, "commit": actual}
                if actual != commit:
                    raise RuntimeError(f"{name} differs from the fixed dependency")

            def robot_factory():
                config = OutcomePiperConfig(
                    can_interface=args.interface,
                    firmware=prior["selected_driver"],
                    feedback_timeout_s=1.0,
                    execution_mode="read_only",
                    calibration_dir=args.output.parent / "plugin-calibration",
                    cameras={},
                )
                return plugin_module.OutcomePiper(config)

            run_cycles(robot_factory, prior["firmware_identity"], report)
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_cycles": sum(
                    cycle["status"] == "passed" for cycle in report.get("cycles", [])
                ),
                "error": report.get("error"),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
