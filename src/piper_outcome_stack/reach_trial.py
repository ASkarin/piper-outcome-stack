"""S2 state-only software checks and an explicitly operator-started single leg."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def state_config(config_path, output):
    config = json.loads(Path(config_path).read_text())
    config["robot"]["cameras"] = {}
    with Path(output).open("x") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return dict(
        status="state_only_configuration_written",
        output=str(output),
        hardware_io=False,
        source_config=str(config_path),
        numeric_limits_changed=False,
    )


def load_configuration(path):
    import draccus
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot_robot_outcome_piper.config import OutcomePiperConfig, OutcomePiperXboxConfig
    from lerobot_robot_outcome_piper.safety import load_motion_safety
    from .reach_execution import check_reach_geometry

    raw = json.loads(Path(path).read_text())
    robot = draccus.decode(RobotConfig, raw["robot"])
    teleop = draccus.decode(TeleoperatorConfig, raw["teleop"])
    if (
        not isinstance(robot, OutcomePiperConfig)
        or robot.execution_mode != "motion"
        or robot.cameras
    ):
        raise ValueError("S2 requires explicit motion configuration with no camera devices")
    if not isinstance(teleop, OutcomePiperXboxConfig) or teleop.control_hz != 50:
        raise ValueError("S2 requires the measured Xbox mapping at50Hz")
    if robot.capture_timing is None:
        raise ValueError("S2 requires measured robot observation timing")
    safety = load_motion_safety(robot.safety_path)
    check_reach_geometry(safety)
    return raw, robot, teleop, safety


def check_recorded_inputs(bundle, reference, safety, output):
    import importlib.metadata
    import sys
    import numpy as np
    from piper_policy.reach_grid import ReachEndpointGrid
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target
    from . import policy_motion
    from .reach_execution import ReachPredictor, tip_fk_base_m

    started = time.monotonic()
    data = json.loads(Path(reference).read_text())
    targets = json.loads((Path(bundle) / "targets.json").read_text())
    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
    rows = data["rows"]
    anchors = [
        row for row in rows if row["source_row_index"] == manifest["anchor_source_row_index"]
    ]
    if len(anchors) != 1:
        raise ValueError("original source row index must identify exactly one recorded anchor")
    anchor = anchors[0]["state"]
    np.testing.assert_array_equal(anchor, targets["anchor_state_rad_m"])
    current_tip = tip_fk_base_m(anchor)
    anchor_checks, counts, rejected = [], {}, 0
    details_path = Path(output).with_suffix(".candidates.jsonl")

    def evaluate(predictor, row):
        begin = time.perf_counter()
        target = predictor.predict(row["state"])
        predicted = time.perf_counter()
        result = dict(
            source_row_index=row["source_row_index"],
            case=predictor.case["id"],
            direction=predictor.direction,
            method=predictor.method,
            target_rad_m=target.tolist(),
        )
        try:
            check_execution_target(row["state"], target, safety)
            result["status"] = "allowed"
        except ValueError as exc:
            result.update(status="rejected", reason=str(exc))
        result.update(
            candidate_s=predicted - begin, candidate_and_validation_s=time.perf_counter() - begin
        )
        return result

    for case in targets["cases"]:
        for method in ("grid", "ik_reference"):
            for direction in ("out", "back"):
                predictor = ReachPredictor(bundle, case["id"], direction, method)
                anchor_checks.append(evaluate(predictor, anchors[0]))
    # The full archived sequence is diagnostic coverage, not a startup-state selection.
    pilot = targets["cases"][0]["id"]
    with details_path.open("x") as stream:
        for method in ("grid", "ik_reference"):
            for direction in ("out", "back"):
                predictor = ReachPredictor(bundle, pilot, direction, method)
                cell = dict(allowed=0, rejected=0)
                for row in rows:
                    result = evaluate(predictor, row)
                    cell[result["status"]] += 1
                    rejected += result["status"] == "rejected"
                    stream.write(json.dumps(result) + "\n")
                counts[method + "/" + direction] = cell
    return dict(
        status="offline_check_completed",
        hardware_io=False,
        source_reference=data["source"],
        recorded_rows=len(rows),
        anchor_source_row_index=manifest["anchor_source_row_index"],
        anchor_state_rad_m=anchor,
        sdk_tip_fk_base_m=current_tip.tolist(),
        source_anchor_tip_base_m=targets["anchor_tip_base_m"],
        anchor_fk_mapping_delta_m=float(np.linalg.norm(current_tip - targets["anchor_tip_base_m"])),
        anchor_target_checks=anchor_checks,
        full_sequence_pilot_case=pilot,
        historical_target_checks=counts,
        rejected_candidates_preserved=int(rejected),
        candidate_records=str(details_path),
        reference_safety=data["safety"],
        safety_used=asdict(safety),
        historical_timing_scope="not live freshness verification",
        runtime_policy_version=importlib.metadata.version("piper-policy"),
        runtime_sources=dict(
            entry=__file__,
            predictor=sys.modules[ReachPredictor.__module__].__file__,
            grid=sys.modules[ReachEndpointGrid.__module__].__file__,
            target_validation=sys.modules[check_execution_target.__module__].__file__,
            control_loop=policy_motion.__file__,
        ),
        analysis_wall_s=time.monotonic() - started,
    )


def run_leg(bundle, config_path, case, direction, method, duration, output):
    # No device class is constructed on either configure/check path.
    from lerobot_robot_outcome_piper import OutcomePiper, OutcomePiperXbox
    from lerobot_robot_outcome_piper.input_safety import motion_input_safety_scope
    from lerobot_robot_outcome_piper.robot import PiperState
    from lerobot_robot_outcome_piper.teleop_control import TeleopControl
    from .policy_motion import PolicyInput, run_policy_trial
    from .reach_execution import ReachPredictor, reach_candidate

    raw, robot_cfg, teleop_cfg, safety = load_configuration(config_path)
    predictor = ReachPredictor(bundle, case, direction, method)
    report = dict(
        status="prepared",
        hardware_io=False,
        config=raw,
        safety=asdict(safety),
        bundle=str(bundle),
        source=predictor.manifest,
        case=case,
        direction=direction,
        method=method,
        goal_base_m=predictor.goal.tolist(),
        max_run_s=duration,
        rows=[],
        outcome_scope="program/hold status only; physical outcome requires independent measurement",
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    robot = teleop = None
    with Path(output).open("x") as stream:
        try:
            print(
                "S2去程：请先由常用遥操作入口完成空载就位并退出。"
                if direction == "out"
                else "S2回程：从去程末状态开始，保留位置，不重新就位或复位。",
                flush=True,
            )
            print("保持LB松开、摇杆回中；本命令建立保持后，再重新按住LB运行。", flush=True)
            print(
                f"方法={method}，目标={case}/{direction}，目标XYZ(m)={predictor.goal.tolist()}。",
                flush=True,
            )
            print(
                f"当前配置SDK速度={safety.motion_speed_percent}%，最长策略运行={duration}s。不自动回A或回零。",
                flush=True,
            )
            if input("回车：连接并使能、建立当前位置保持；其他输入：取消。\n> ") != "":
                report["status"] = "cancelled_before_connection"
                return
            control = TeleopControl()
            robot, teleop = OutcomePiper(robot_cfg), OutcomePiperXbox(teleop_cfg)
            robot.configure_teleoperation(control, teleop_cfg.hold_settings())
            with motion_input_safety_scope():
                report["hardware_io"] = True
                teleop.connect()
                inputs = PolicyInput(teleop, robot)
                inputs.poll()
                robot.emergency_stop_poll = teleop.poll_emergency_stop
                robot.connect()
                robot.enable()
                report.update(
                    run_policy_trial(
                        robot,
                        teleop,
                        control,
                        predictor,
                        safety,
                        report["rows"],
                        candidate_fn=reach_candidate,
                        max_actions=math.ceil(50 * duration),
                        max_run_s=duration,
                    )
                )
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if robot is not None and robot.state is PiperState.ACTIVE:
                robot.request_input_fault(exc)
            raise
        finally:
            try:
                if robot is not None:
                    try:
                        robot.disconnect()
                    finally:
                        report.update(
                            stop_outcome=robot.stop_outcome,
                            stop_error=robot.stop_error,
                            last_action_telemetry=robot.last_action_telemetry,
                        )
            except Exception as exc:
                report.update(status="failed", disconnect_error=str(exc))
                raise
            finally:
                try:
                    if teleop is not None:
                        teleop.disconnect()
                except Exception as exc:
                    report.update(status="failed", teleop_disconnect_error=str(exc))
                    raise
                finally:
                    json.dump(report, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["configure", "check", "run"])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--direction", choices=["out", "back"])
    parser.add_argument("--method", choices=["grid", "ik_reference"])
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "configure":
        print(json.dumps(state_config(args.config, args.output)))
        return 0
    if args.bundle is None:
        parser.error("bundle is required for check/run")
    if args.mode == "check":
        if args.reference is None:
            parser.error("check requires an explicit historical reference")
        raw, _, _, safety = load_configuration(args.config)
        result = check_recorded_inputs(args.bundle, args.reference, safety, args.output)
        result["config_snapshot"] = raw
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "status",
                        "recorded_rows",
                        "rejected_candidates_preserved",
                        "hardware_io",
                    ]
                }
            )
        )
        return 0
    if (
        args.case is None
        or args.direction is None
        or args.method is None
        or args.duration_s is None
        or not math.isfinite(args.duration_s)
        or args.duration_s <= 0
    ):
        parser.error("run requires explicit case, direction, method and positive finite duration")
    run_leg(
        args.bundle,
        args.config,
        args.case,
        args.direction,
        args.method,
        args.duration_s,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
