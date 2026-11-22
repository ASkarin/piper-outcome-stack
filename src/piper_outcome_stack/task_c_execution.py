"""C single-skill binding to the existing RTC and PiPER motion lifecycle."""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .task_c import EXECUTION_RECIPE, select_task, text_field


def prepare_policy(config, selected, directory):
    """Load configuration/model and check recorded inputs; no device construction."""
    import draccus
    import torch
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot_robot_outcome_piper.config import OutcomePiperConfig, OutcomePiperXboxConfig
    from lerobot_robot_outcome_piper.gripper_reference import GripperReferenceSettings
    from lerobot_robot_outcome_piper.safety import load_motion_safety
    from piper_policy.smolvla import SmolVLAChunkPredictor
    from piper_policy.smolvla_execution import AsyncSmolVLAExecutor
    from .task_c_reference import read_reference, verify_rtc_reference

    if not isinstance(config.get("robot"), dict) or not isinstance(config.get("teleop"), dict):
        raise ValueError("C requires actual robot and teleop configurations")
    robot_cfg = draccus.decode(RobotConfig, config["robot"])
    teleop_cfg = draccus.decode(TeleoperatorConfig, config["teleop"])
    if not isinstance(robot_cfg, OutcomePiperConfig) or robot_cfg.execution_mode != "motion":
        raise ValueError("C requires the explicit PiPER motion configuration")
    if not isinstance(teleop_cfg, OutcomePiperXboxConfig) or teleop_cfg.control_hz != 50:
        raise ValueError("C uses the existing measured Xbox configuration at50Hz")
    if robot_cfg.capture_timing is None or teleop_cfg.gripper_reference is None:
        raise ValueError("C requires capture timing and measured continuous gripper settings")
    if teleop_cfg.work_joint_rad is None or teleop_cfg.work_gripper_m is None:
        raise ValueError("C startup requires the configured A work pose and gripper opening")
    safety = load_motion_safety(robot_cfg.safety_path)
    gripper = GripperReferenceSettings(**teleop_cfg.gripper_reference)
    policy = config.get("policy") or {}
    key = selected["context"]["skill_id"]

    def asset(value, name):
        path = Path(text_field(value, name))
        path = path if path.is_absolute() else directory / path
        if not path.exists():
            raise FileNotFoundError(f"missing C {name}: {path}")
        return path

    checkpoint = asset(policy.get("checkpoint"), "policy.checkpoint")
    vlm = asset(policy.get("vlm_assets"), "policy.vlm_assets")
    reference = asset((policy.get("reference_inputs") or {}).get(key), f"reference_inputs.{key}")
    recorded = read_reference(reference, selected["task"])
    print("正在加载 Push-T FP32 模型并核对录制参考。", flush=True)
    predictor = SmolVLAChunkPredictor.from_checkpoint(
        checkpoint, task=selected["task"], vlm_assets=vlm
    )
    predictor.model.float()
    predictor.model.config.use_amp = False
    executor = AsyncSmolVLAExecutor(predictor, request_stride=25, joint_lowpass=False)
    checks = verify_rtc_reference(executor, recorded, safety)
    torch.manual_seed(42)
    evidence = dict(
        checkpoint=str(checkpoint),
        startup_work_pose=dict(
            name="A",
            target_joints_rad=list(teleop_cfg.work_joint_rad),
            target_gripper_m=teleop_cfg.work_gripper_m,
        ),
        reference_inputs=str(reference),
        reference_errors=checks,
        reference_scope="FP32 first chunks, locally rebuilt RTC prefixes and projected targets on recorded inputs",
        inference_settings=dict(
            use_amp=predictor.model.config.use_amp,
            matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        ),
        safety=asdict(safety),
        gripper_reference=asdict(gripper),
    )
    return robot_cfg, teleop_cfg, safety, gripper, executor, evidence


def run_segment(robot_cfg, teleop_cfg, safety, gripper, predictor, selected, report, output):
    from lerobot_robot_outcome_piper import OutcomePiper, OutcomePiperXbox
    from lerobot_robot_outcome_piper.capture_gc import CaptureGC
    from lerobot_robot_outcome_piper.input_safety import motion_input_safety_scope
    from lerobot_robot_outcome_piper.robot import PiperState
    from lerobot_robot_outcome_piper.teleop_control import TeleopControl
    from .policy_motion import PolicyInput, run_policy_trial
    from .policy_rgb_recording import PolicyRGBRecorder

    from .policy_startup import confirm_work_pose, prepare_work_pose

    print(selected["context"]["start_condition"])
    print(f"Push-T｜{selected['position']}｜{selected['task']}  上限 {selected['duration_s']} 秒")
    if not confirm_work_pose(teleop_cfg, input_fn=input):
        report.update(status="cancelled_before_connection", policy_actions_sent=0)
        return
    with ExitStack() as resources:
        guard = CaptureGC()
        guard.prepare()
        guard.start()
        resources.callback(guard.stop)
        recorder = PolicyRGBRecorder(output.with_name(output.stem + "-rgb"))

        def close_recorder():
            report["rgb_recording"] = recorder.close()

        resources.callback(close_recorder)
        control = TeleopControl()
        robot, teleop = OutcomePiper(robot_cfg), OutcomePiperXbox(teleop_cfg)

        def close_teleop():
            try:
                teleop.disconnect()
            except Exception as exc:
                report["teleop_disconnect_error"] = str(exc)
                raise

        def close_robot():
            try:
                robot.disconnect()
            except Exception as exc:
                report["disconnect_error"] = str(exc)
                raise
            finally:
                report.update(
                    stop_outcome=robot.stop_outcome,
                    stop_error=robot.stop_error,
                    last_action_telemetry=robot.last_action_telemetry,
                )

        resources.callback(close_teleop)
        resources.callback(close_robot)
        robot.configure_teleoperation(control, teleop_cfg.hold_settings())
        with motion_input_safety_scope():
            try:
                report["hardware_io"] = True
                teleop.connect()
                inputs = PolicyInput(teleop, robot)
                inputs.poll()
                robot.camera_input_poll = inputs.poll
                robot.emergency_stop_poll = teleop.poll_emergency_stop
                robot.connect()
                robot.enable()
                startup = prepare_work_pose(robot, teleop, control, teleop_cfg, report["rows"])
                report["startup_A"] = startup
                if not startup["hold_confirmed"]:
                    raise RuntimeError("A work pose preparation ended without a confirmed hold")
                if startup["status"] == "cancelled_held":
                    report.update(
                        status="startup_cancelled_held", hold_confirmed=True, policy_actions_sent=0
                    )
                    return
                if startup["status"] != "arrived_held":
                    raise RuntimeError(f"unexpected A work pose result: {startup['status']}")
                report.update(
                    run_policy_trial(
                        robot,
                        teleop,
                        control,
                        predictor,
                        safety,
                        report["rows"],
                        max_actions=math.ceil(50 * selected["duration_s"]),
                        max_run_s=selected["duration_s"],
                        frame_recorder=recorder,
                        record_action_chunks=True,
                        gripper_reference_settings=gripper,
                    )
                )
            except (Exception, KeyboardInterrupt) as exc:
                if robot.state is PiperState.ACTIVE:
                    robot.request_input_fault(exc)
                raise


def execute(config_path, mode, output):
    if mode not in ("check", "run"):
        raise ValueError("C execution requires check or run")
    config_path, output = Path(config_path), Path(output)
    report = dict(
        status="preparing",
        task_family="C",
        mode=mode,
        hardware_io=False,
        recipe={**EXECUTION_RECIPE, "seed": 42},
        rows=[],
        created_utc=datetime.now(timezone.utc).isoformat(),
        outcome_scope="single skill; program/hold status is not a physical task outcome",
        source=str(Path(__file__).resolve()),
    )
    from .task_c_reference import fp32_runtime

    with output.open("x") as stream, ExitStack() as runtime:
        predictor = None
        try:
            runtime.enter_context(fp32_runtime())
            config = json.loads(config_path.read_text())
            selected = select_task(config)
            report.update(
                task=selected["task"],
                task_context=selected["context"],
                position=selected["position"],
                max_run_s=selected["duration_s"],
                config=config,
            )
            robot_cfg, teleop_cfg, safety, gripper, predictor, evidence = prepare_policy(
                config, selected, config_path.resolve().parent
            )
            report.update(evidence)
            if mode == "check":
                report.update(status="software_check_completed")
            else:
                run_segment(
                    robot_cfg, teleop_cfg, safety, gripper, predictor, selected, report, output
                )
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                if predictor is not None:
                    predictor.reset_execution()
            except Exception as exc:
                report.update(status="failed", executor_cleanup_error=str(exc))
                raise
            finally:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
    return {key: value for key, value in report.items() if key not in ("rows", "config")}
