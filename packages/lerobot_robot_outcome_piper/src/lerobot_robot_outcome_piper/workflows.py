"""Thin official-LeRobot wrappers that inject the canonical Xbox processor."""

from __future__ import annotations

import math
from typing import Any

from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config

from .console import operator_message
from .config import OutcomePiperConfig, OutcomePiperXboxConfig
from .input_safety import motion_input_safety_scope
from .processor import make_xbox_processor
from .safety import load_motion_safety


def _validate_workflow_configs(
    robot: Any, teleop: Any
) -> tuple[OutcomePiperConfig, OutcomePiperXboxConfig]:
    if not isinstance(robot, OutcomePiperConfig):
        raise ValueError("robot must use type=outcome_piper")
    if not isinstance(teleop, OutcomePiperXboxConfig):
        raise ValueError("teleop must use type=outcome_piper_xbox")
    if robot.execution_mode != "motion":
        raise ValueError("Xbox workflows require robot.execution_mode=motion")
    assert robot.safety_path is not None
    return robot, teleop


def _processor(robot: OutcomePiperConfig, teleop: OutcomePiperXboxConfig):
    safety = load_motion_safety(robot.safety_path)
    return make_xbox_processor(
        safety,
        work_joint_rad=teleop.work_joint_rad,
        work_gripper_m=teleop.work_gripper_m,
        pose_timing=teleop.pose_timing,
        streaming_reference=teleop.streaming_reference,
        gripper_reference=teleop.gripper_reference,
        max_xyz_step_m=teleop.xyz_step_m,
        max_rotation_step_rad=teleop.rotation_step_rad,
        max_gripper_step_m=teleop.gripper_step_m,
        ik_max_nfev=teleop.ik_max_nfev,
        ik_timeout_s=teleop.ik_timeout_s,
        ik_residual_tolerance=teleop.ik_residual_tolerance,
        ik_min_singular_value=teleop.ik_min_singular_value,
    )


def teleoperate(cfg: Any) -> None:
    """Use the official no-dataset loop without periodic console redraws."""

    from lerobot.scripts import lerobot_record as official
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    robot_config, teleop_config = _validate_workflow_configs(cfg.robot, cfg.teleop)
    if cfg.fps != teleop_config.control_hz:
        raise ValueError("workflow fps must match the measured Xbox control_hz")
    import logging

    if not logging.getLogger().handlers:
        init_logging()
    if cfg.display_data:
        init_visualization(
            cfg.display_mode,
            session_name="teleoperation",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )
    display_compressed_images = (
        True
        if cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None
        else cfg.display_compressed_images
    )
    teleop = make_teleoperator_from_config(teleop_config)
    robot = make_robot_from_config(robot_config)
    teleop_action_processor = _processor(robot_config, teleop_config)
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    robot.configure_teleoperation(
        teleop_action_processor.steps[0].control, teleop_config.hold_settings()
    )
    trace = None
    if getattr(cfg, "control_trace_path", None) is not None:
        from .control_trace import ControlTrace

        trace = ControlTrace(cfg.control_trace_path)
        robot.control_trace = trace
    try:
        teleop.connect()
        if teleop.get_action()["emergency_stop"]:
            raise ValueError("release the emergency-stop button before starting a session")
        with motion_input_safety_scope():
            robot.camera_input_poll = teleop.get_action
            robot.emergency_stop_poll = lambda: teleop.poll_emergency_stop()
            robot.connect()
            try:
                robot.enable()
                from .console import print_controls

                print_controls(teleop_config)
                operator_message("[设备已连接] 请松开LB并回中，等待保持确认。")
                operator_message("结束遥操作：松开LB、等待保持确认，再按Ctrl+C。")
                # The prior teleop loop executes one tick even for a non-positive
                # duration. Keep that behavior while using record_loop without storage.
                duration = math.inf if cfg.teleop_time_s is None else cfg.teleop_time_s
                if duration <= 0:
                    duration = math.ulp(0.0)
                official.record_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    dataset=None,
                    events={
                        "exit_early": False,
                        "stop_recording": False,
                        "rerecord_episode": False,
                    },
                    control_time_s=duration,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    display_compressed_images=display_compressed_images,
                )
            except KeyboardInterrupt:
                pass
            finally:
                robot.disconnect()
    finally:
        try:
            teleop.disconnect()
            if cfg.display_data:
                shutdown_visualization(cfg.display_mode)
        finally:
            if trace is not None:
                robot.control_trace = None
                trace.save(stop_outcome=robot.stop_outcome, stop_error=robot.stop_error)
                operator_message(f"[诊断记录已保存] {trace.path}")

    operator_message("[已退出] 程序未自动回零或失能；请先完成停放再断电。")


def record(cfg: Any) -> Any:
    """Call the official recorder with the canonical Xbox processor."""

    from .recording import record_with_telemetry

    robot_config, teleop_config = _validate_workflow_configs(cfg.robot, cfg.teleop)
    if cfg.dataset.fps != teleop_config.control_hz:
        raise ValueError("dataset fps must match the measured Xbox control_hz")
    if cfg.dataset.push_to_hub:
        raise ValueError(
            "record inside piper-can requires dataset.push_to_hub=false; "
            "publish after the hardware session"
        )
    # The session decides which cameras/streams to record. Rates need not be identical;
    # per-frame freshness/skew checks determine whether a sample is usable.
    if any(camera.fps < cfg.dataset.fps for camera in robot_config.cameras.values()):
        raise ValueError("dataset fps exceeds a configured camera frame rate")
    if robot_config.scene is None:
        raise ValueError("record requires an explicit current scene context")
    if robot_config.capture_timing is None:
        raise ValueError("record requires measured capture_timing")
    with motion_input_safety_scope():
        return record_with_telemetry(
            cfg,
            teleop_action_processor=_processor(robot_config, teleop_config),
        )


def replay(cfg: Any) -> None:
    """Adapt the pinned LeRobot replay sequence to explicit PiPER enable.

    Dataset selection and action processing retain lerobot_replay semantics;
    the official entry has no lifecycle hook between connect and its loop.
    """
    import time
    from lerobot.scripts import lerobot_replay as official
    from .safety import ACTION_KEYS

    if not isinstance(cfg.robot, OutcomePiperConfig) or cfg.robot.execution_mode != "motion":
        raise ValueError("replay requires an outcome_piper motion configuration")
    import logging

    if not logging.getLogger().handlers:
        official.init_logging()
    dataset = official.LeRobotDataset(
        cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode]
    )
    names = dataset.features[official.ACTION]["names"]
    if tuple(names) != ACTION_KEYS:
        raise ValueError("replay requires the canonical seven joint/gripper action fields")
    actions = dataset.select_columns(official.ACTION)
    if getattr(cfg, "trajectory", None) is not None:
        return _continuous_replay(cfg, dataset, actions, names)
    process_action = official.make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    official.log_say("Replaying episode", cfg.play_sounds, blocking=True)
    robot.connect()
    try:
        robot.enable()
        for index in range(dataset.num_frames):
            started = time.perf_counter()
            row = actions[index][official.ACTION]
            action = {name: row[i] for i, name in enumerate(names)}
            robot.send_action(process_action((action, robot.get_observation())))
            official.precise_sleep(max(1 / dataset.fps - (time.perf_counter() - started), 0.0))
    finally:
        robot.disconnect()


def _continuous_replay(cfg, dataset, actions, names):
    import time
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    from .continuous_replay import plan_replay, execute_replay
    from .record_control import TerminalCommands
    from .teleop_control import HoldSettings
    from .safety import JOINT_KEYS, load_motion_safety
    from .robot import OutcomePiper

    values = dict(cfg.trajectory)
    if set(values) != {"time_scale", "control_hz", "hold_settings"}:
        raise ValueError("trajectory requires time_scale, control_hz and hold_settings")
    if cfg.robot.cameras:
        raise ValueError("continuous replay trial does not use cameras")
    if not cfg.trajectory_report_path:
        raise ValueError("continuous replay needs an explicit report path")
    output = Path(cfg.trajectory_report_path)
    if output.exists():
        raise FileExistsError(output)
    settings = HoldSettings(**values["hold_settings"])
    safety = load_motion_safety(cfg.robot.safety_path)
    rows = [[float(v) for v in actions[i]["action"]] for i in range(dataset.num_frames)]
    plan = plan_replay(rows, dataset.fps, values["control_hz"], values["time_scale"], safety)
    operator_message(
        f"[连续回放] {len(rows)}个来源动作→{len(plan['actions'])}个执行参考；名义{plan['nominal_duration_s']:.2f}秒。",
    )
    operator_message(
        "保持在轨迹起点、手离开后输入 start；其他输入取消。回放中stop取消并保持；Ctrl+C电子急停可能下沉。Xbox不参与。",
    )
    terminal = TerminalCommands()
    line = None
    while line is None:
        line = terminal.poll()
        if line is None:
            time.sleep(0.05)
    if line != "start":
        return
    robot = OutcomePiper(cfg.robot)
    from dataclasses import asdict

    report = dict(
        status="started",
        source_dataset=str(cfg.dataset.root),
        episode=cfg.dataset.episode,
        safety=asdict(safety),
        source_fps=dataset.fps,
        settings=values,
        method=plan["method"],
        source_action_count=len(rows),
        reference_count=len(plan["actions"]),
        nominal_duration_s=plan["nominal_duration_s"],
        original_dataset_modified=False,
        trace=[],
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    enabled = False
    try:
        robot.connect()
        obs = robot.get_observation()
        if any(
            abs(obs[k] - plan["actions"][0][k]) > settings.joint_tolerance_rad for k in JOINT_KEYS
        ):
            raise ValueError("not at confirmed replay start; no automatic approach")
        robot._validate_action(plan["actions"][0])
        enabled = True
        robot.enable()
        report.update(execute_replay(robot, plan, safety, settings, report["trace"], terminal))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if enabled:
            try:
                robot.request_emergency_stop(exc)
            except Exception as stop:
                report["stop_error"] = str(stop)
        raise
    finally:
        try:
            robot.disconnect()
        finally:
            report.update(
                stop_outcome=robot.stop_outcome,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2))
            operator_message("[回放记录] " + str(output))
