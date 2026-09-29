"""Operator-started one-segment ACT motion trial; check mode never opens devices."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from piper_outcome_stack.policy_execution import (
    CONTROL_HZ,
    ACTChunkPredictor,
    verify_reference_inputs,
)
from piper_outcome_stack.policy_motion import PolicyInput, run_policy_trial


def recorded_target_check(state, target, safety):
    """Diagnostic on recorded inputs, distinct from mandatory live target validation."""
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    try:
        check_execution_target(state, target, safety)
    except ValueError as exc:
        return dict(status="rejected", reason=str(exc))
    return dict(status="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--joint-representation", choices=("relative", "absolute"), default="relative"
    )
    parser.add_argument("--reference-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record-rgb",
        action="store_true",
        help="record existing policy RGB inputs at up to10Hz in a bounded background writer",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=1,
        help="number of50Hz actions between model inferences",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        help="TE on decoded absolute chunks; standard ACT coefficient is 0.01",
    )
    parser.add_argument(
        "--position",
        help="operator start-position label (e.g. P1); outcome is labeled after the run",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        help="optional duration budget; omitted means run until operator cancels",
    )
    args = parser.parse_args()
    if args.duration_s is not None and (not math.isfinite(args.duration_s) or args.duration_s <= 0):
        parser.error("duration must be positive and finite when specified")
    if args.temporal_ensemble_coeff is not None and not math.isfinite(args.temporal_ensemble_coeff):
        parser.error("temporal ensemble coefficient must be finite")
    import draccus
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot_robot_outcome_piper import OutcomePiper, OutcomePiperXbox
    from lerobot_robot_outcome_piper.config import OutcomePiperConfig, OutcomePiperXboxConfig
    from lerobot_robot_outcome_piper.input_safety import motion_input_safety_scope
    from lerobot_robot_outcome_piper.robot import PiperState
    from lerobot_robot_outcome_piper.teleop_control import TeleopControl
    from lerobot_robot_outcome_piper.safety import load_motion_safety

    config = json.loads(args.config.read_text())
    robot_cfg = draccus.decode(RobotConfig, config["robot"])
    teleop_cfg = draccus.decode(TeleoperatorConfig, config["teleop"])
    if not isinstance(robot_cfg, OutcomePiperConfig) or robot_cfg.execution_mode != "motion":
        raise ValueError("trial requires explicit PiPER motion configuration")
    if not isinstance(teleop_cfg, OutcomePiperXboxConfig) or teleop_cfg.control_hz != CONTROL_HZ:
        raise ValueError("trial requires the measured Xbox configuration at 50Hz")
    if robot_cfg.capture_timing is None:
        raise ValueError("trial requires measured capture timing")
    safety = load_motion_safety(robot_cfg.safety_path)
    with args.output.open("x") as output:
        report = dict(
            status="started",
            mode=args.mode,
            config=config,
            safety=asdict(safety),
            checkpoint=str(args.checkpoint),
            reference_inputs=str(args.reference_inputs),
            position=args.position,
            max_policy_actions=None
            if args.duration_s is None
            else math.ceil(CONTROL_HZ * args.duration_s),
            max_run_s=args.duration_s,
            rows=[],
        )
        report["execution"] = dict(
            n_action_steps=args.n_action_steps,
            joint_representation=args.joint_representation,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            ensemble_space="absolute rad/m after generation-anchor decoding",
            checkpoint_temporal_ensemble="disabled; execution layer owns TE",
        )
        robot = teleop = recorder = capture_gc = None
        try:
            torch.set_num_threads(4)
            predictor = ACTChunkPredictor.from_checkpoint(
                args.checkpoint,
                joint_representation=args.joint_representation,
            )
            images, states, report["reference_errors"] = verify_reference_inputs(
                predictor, args.reference_inputs, safety
            )
            predictor.configure_temporal_ensemble(args.temporal_ensemble_coeff)
            predictor.configure_action_steps(args.n_action_steps)
            if args.temporal_ensemble_coeff is not None:
                print(
                    f"TE已开启：系数 {args.temporal_ensemble_coeff}，在绝对动作上融合。", flush=True
                )
            # Exercise the actual target-selection path before any device connection.
            bounds_checks = []
            preflight_inferences = 0
            for i in range(max(3, args.n_action_steps + 1)):
                chunk, index, inferred = predictor.execution_chunk(images[0], states[0], i)
                target = predictor.select_target(chunk, chunk_index=index, new_chunk=inferred)
                preflight_inferences += inferred
                if target.shape != (7,) or not np.isfinite(target).all():
                    raise ValueError("invalid execution target during preflight")
                bounds_checks.append(recorded_target_check(states[0], target, safety))
            report["execution_preflight"] = dict(
                status="passed",
                samples=len(bounds_checks),
                inferences=preflight_inferences,
                ensemble_updates=predictor.ensemble_updates,
                recorded_target_checks=bounds_checks,
            )
            rejected = sum(
                x["recorded_target_check"]["status"] == "rejected"
                for x in report["reference_errors"]
            )
            rejected += sum(x["status"] == "rejected" for x in bounds_checks)
            if rejected:
                print(
                    f"录制输入中有{rejected}个候选未通过动作约束；未下发。实机每步仍执行相同检查。",
                    flush=True,
                )
            predictor.reset_execution()
            if args.mode == "check":
                report.update(status="software_check_completed", hardware_io=False)
                return
            from piper_outcome_stack.policy_startup import confirm_work_pose, prepare_work_pose

            if not confirm_work_pose(teleop_cfg):
                report.update(status="cancelled_before_start", policy_actions_sent=0)
                return
            from lerobot_robot_outcome_piper.capture_gc import CaptureGC

            capture_gc = CaptureGC()
            capture_gc.prepare()  # Collect before any hardware connection or control.
            capture_gc.start()
            if args.record_rgb:
                from piper_outcome_stack.policy_rgb_recording import PolicyRGBRecorder

                recorder = PolicyRGBRecorder(args.output.with_name(args.output.stem + "-rgb"))
            control = TeleopControl()
            robot, teleop = OutcomePiper(robot_cfg), OutcomePiperXbox(teleop_cfg)
            robot.configure_teleoperation(control, teleop_cfg.hold_settings())
            with motion_input_safety_scope():
                try:
                    teleop.connect()
                    inputs = PolicyInput(teleop, robot)
                    inputs.poll()  # Startup B is rejected before connecting/enabling.
                    robot.camera_input_poll = inputs.poll
                    robot.emergency_stop_poll = teleop.poll_emergency_stop
                    print(
                        f"关节表示：{args.joint_representation}；SDK速度上限：{safety.motion_speed_percent}%。",
                        flush=True,
                    )
                    print(f"[策略] 每{args.n_action_steps}步推理一次，控制频率50Hz。", flush=True)
                    print(
                        "[连接] 即将连接机械臂、使能并建立当前位置保持。\n此时请松开LB，保持摇杆和扳机回中。\n随后自动回A；此阶段尚未执行模型动作。",
                        flush=True,
                    )
                    robot.connect()
                    robot.enable()
                    report["startup_A"] = prepare_work_pose(
                        robot,
                        teleop,
                        control,
                        teleop_cfg,
                        report["rows"],
                    )
                    if report["startup_A"]["status"] != "arrived_held":
                        report.update(status="startup_cancelled_held", policy_actions_sent=0)
                        return
                    report.update(
                        run_policy_trial(
                            robot,
                            teleop,
                            control,
                            predictor,
                            safety,
                            report["rows"],
                            max_actions=report["max_policy_actions"],
                            max_run_s=report["max_run_s"],
                            frame_recorder=recorder,
                        )
                    )
                except (Exception, KeyboardInterrupt) as exc:
                    if robot.state is PiperState.ACTIVE:
                        robot.request_input_fault(exc)  # Existing bounded hold and terminal latch.
                    raise
        except (Exception, KeyboardInterrupt) as exc:
            report.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            try:
                if robot is not None:
                    try:
                        robot.disconnect()
                    finally:
                        # disconnect() can still latch a requested electronic stop.
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
                finally:
                    try:
                        if recorder is not None:
                            report["rgb_recording"] = recorder.close()
                    finally:
                        if capture_gc is not None:
                            capture_gc.stop()  # Restore only after device/recorder cleanup.
                            report["policy_runtime"] = capture_gc.summary()
                    output.write(json.dumps(report, indent=2))
                    output.flush()
                    print(
                        json.dumps(
                            {
                                k: report[k]
                                for k in (
                                    "status",
                                    "end_reason",
                                    "policy_actions_sent",
                                    "error",
                                    "stop_outcome",
                                )
                                if k in report
                            },
                            indent=2,
                        )
                    )
                    print(f"记录：{args.output}", flush=True)


if __name__ == "__main__":
    main()
