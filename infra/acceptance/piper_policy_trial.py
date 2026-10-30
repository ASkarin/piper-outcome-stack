"""Operator-started policy motion trial; check mode never opens devices."""

import argparse
from dataclasses import asdict
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from piper_policy.policy_execution import CONTROL_HZ, ACTChunkPredictor
from piper_outcome_stack.policy_execution import plan_policy_gripper, verify_reference_inputs
from piper_outcome_stack.policy_motion import PolicyInput, run_policy_trial


def recorded_target_check(state, target, safety, *, last_gripper_target=None):
    """Diagnostic on recorded inputs, distinct from mandatory live target validation."""
    from lerobot_robot_outcome_piper.execution_constraints import check_execution_target

    try:
        check_execution_target(state, target, safety, last_gripper_target=last_gripper_target)
    except ValueError as exc:
        return dict(status="rejected", reason=str(exc))
    return dict(status="passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-type", choices=("act", "smolvla"), default="act")
    parser.add_argument("--task", help="explicit SmolVLA instruction")
    parser.add_argument("--vlm-assets", type=Path)
    parser.add_argument("--seed", type=int, default=42, help="SmolVLA inference seed")
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
        "--record-action-chunks",
        action="store_true",
        help="save each newly generated absolute chunk with its anchor and execution index",
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
    parser.add_argument(
        "--completion-config", type=Path, help="fixed-scene placement detector configuration"
    )
    args = parser.parse_args()
    if args.policy_type == "smolvla" and (
        not args.task
        or args.vlm_assets is None
        or args.joint_representation != "absolute"
        or args.n_action_steps != 25
        or args.temporal_ensemble_coeff is not None
    ):
        parser.error(
            "SmolVLA requires task, VLM assets, absolute joints, request stride25 and no ACT temporal ensemble"
        )
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
    gripper_reference_settings = None
    if args.policy_type == "smolvla":
        from lerobot_robot_outcome_piper.gripper_reference import GripperReferenceSettings

        if teleop_cfg.gripper_reference is None:
            raise ValueError("SmolVLA requires the measured gripper reference configuration")
        gripper_reference_settings = GripperReferenceSettings(**teleop_cfg.gripper_reference)
    completion_detector = None
    completion_config = None
    if args.completion_config is not None:
        from piper_outcome_stack.policy_completion import PlacementCompletion

        completion_config = json.loads(args.completion_config.read_text())
        if completion_config["scene_id"] != config["robot"]["scene"]["scene_id"]:
            raise ValueError("completion region belongs to a different scene")
        completion_detector = PlacementCompletion(completion_config)
    with args.output.open("x") as output:
        report = dict(
            status="started",
            mode=args.mode,
            record_action_chunks=args.record_action_chunks,
            completion_config=completion_config,
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
        robot = teleop = recorder = capture_gc = predictor = None
        try:
            torch.set_num_threads(4)
            report["policy_type"] = args.policy_type
            if args.policy_type == "smolvla":
                from piper_policy.smolvla import SmolVLAChunkPredictor
                from piper_policy.smolvla_execution import AsyncSmolVLAExecutor

                with np.load(args.reference_inputs) as reference:
                    if set(reference["tasks"].tolist()) != {args.task}:
                        raise ValueError("reference instructions differ from selected SmolVLA task")
                raw_predictor = SmolVLAChunkPredictor.from_checkpoint(
                    args.checkpoint, task=args.task, vlm_assets=args.vlm_assets
                )
                images, states, report["reference_errors"] = verify_reference_inputs(
                    raw_predictor, args.reference_inputs, safety, warmup=5
                )
                predictor = AsyncSmolVLAExecutor(raw_predictor, request_stride=args.n_action_steps)
                torch.manual_seed(args.seed)
                report["execution"].update(
                    joint_representation="absolute",
                    task=args.task,
                    asynchronous=True,
                    initial_wait="explicit_initial_target_hold",
                    gripper_projection="zero",
                    ensemble_space=None,
                    checkpoint_temporal_ensemble=None,
                    seed=args.seed,
                    rtc=True,
                    rtc_guidance=predictor.rtc_guidance,
                    rtc_num_steps=predictor.rtc_num_steps,
                    action_padding="initial_noise_times_flow_t",
                    rtc_initial_delay=predictor.rtc_initial_delay,
                    rtc_prefix="normalized joint execution plan; gripper prefix is not guided",
                    target_transform="butterworth2_5hz_joints_and_continuous_gripper_reference",
                    gripper_reference=asdict(gripper_reference_settings),
                    gripper_reference_commit="after successful SDK dispatch in current control epoch",
                    joint_filter_cutoff_hz=5.0,
                    rtc_processor="piper_policy.native_joint_rtc.NativeJointRTCProcessor",
                    rtc_max_guidance_weight=raw_predictor.model.config.rtc_config.max_guidance_weight,
                    rtc_prefix_attention_schedule=raw_predictor.model.config.rtc_config.prefix_attention_schedule.value,
                )
                report["reference_target_check_space"] = "raw_model_output"
                print(
                    "[SmolVLA] 六关节RTC与二阶5Hz低通；夹爪负开度归零后生成连续参考，发送成功后提交。原始预测与参考分别记录。",
                    flush=True,
                )
            else:
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
            asynchronous_chunks = []
            # No SDK/device access: model preflight uses explicitly virtual accepted state.
            preflight_control = None
            if gripper_reference_settings is not None:
                preflight_control = TeleopControl()
                preflight_control.confirm_hold()
                preflight_control.observe(False, True)
                _, preflight_epoch = preflight_control.observe(True, True)
            cycles = 51 if args.policy_type == "smolvla" else max(3, args.n_action_steps + 1)
            for i in range(cycles):
                kwargs = (
                    {"observed_monotonic_s": time.monotonic()}
                    if args.policy_type == "smolvla"
                    else {}
                )
                chunk, index, inferred = predictor.execution_chunk(
                    images[0], states[0], i, **kwargs
                )
                target = predictor.select_target(chunk, chunk_index=index, new_chunk=inferred)
                gripper_plan = None
                if gripper_reference_settings is not None:
                    goal = float(target[6])
                    if (
                        predictor.execution_details.get("phase") == "initial_hold"
                        and preflight_control.gripper_target is not None
                    ):
                        goal = preflight_control.gripper_target
                    gripper_plan = plan_policy_gripper(
                        goal,
                        float(states[0, 6]),
                        preflight_control,
                        preflight_epoch,
                        gripper_reference_settings,
                        safety,
                        time.monotonic(),
                    )
                    target = np.asarray(target, dtype=np.float64).copy()
                    target[6] = gripper_plan["target_m"]
                preflight_inferences += inferred
                if inferred and args.policy_type == "smolvla":
                    asynchronous_chunks.append(predictor.execution_details)
                if target.shape != (7,) or not np.isfinite(target).all():
                    raise ValueError("invalid execution target during preflight")
                check = (
                    recorded_target_check(states[0], target, safety)
                    if preflight_control is None
                    else recorded_target_check(
                        states[0],
                        target,
                        safety,
                        last_gripper_target=preflight_control.gripper_target,
                    )
                )
                if gripper_plan is not None:
                    check["gripper_plan"] = gripper_plan
                    check["virtual_reference_committed"] = False
                    if check["status"] == "passed":
                        preflight_control.gripper_target = float(target[6])
                        check["virtual_reference_committed"] = (
                            preflight_control.commit_gripper_reference(
                                preflight_epoch, gripper_plan
                            )
                        )
                bounds_checks.append(check)
                if args.policy_type == "smolvla":
                    time.sleep(1 / CONTROL_HZ)
            if args.policy_type == "smolvla" and not preflight_inferences:
                raise RuntimeError("asynchronous preflight produced no completed prediction")
            if args.policy_type == "smolvla" and not any(
                event.get("rtc_request", {}).get("prefix_model_actions") is not None
                for event in asynchronous_chunks
            ):
                raise RuntimeError("RTC preflight did not exercise a remaining-plan prefix")
            report["execution_preflight"] = dict(
                status="passed",
                samples=len(bounds_checks),
                inferences=preflight_inferences,
                ensemble_updates=predictor.ensemble_updates,
                recorded_target_checks=bounds_checks,
                asynchronous_chunks=asynchronous_chunks,
                gripper_reference_scope=None
                if preflight_control is None
                else "virtual accepted targets, no SDK dispatch",
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
            if args.policy_type == "smolvla":
                torch.manual_seed(args.seed)
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
                        f"关节表示：{report['execution']['joint_representation']}；SDK速度上限：{safety.motion_speed_percent}%。",
                        flush=True,
                    )
                    print(f"[策略] 每{args.n_action_steps}步请求推理，控制频率50Hz。", flush=True)
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
                            record_action_chunks=args.record_action_chunks,
                            completion_detector=completion_detector,
                            gripper_reference_settings=gripper_reference_settings,
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
                        if predictor is not None and args.policy_type == "smolvla":
                            predictor.reset_execution()
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
