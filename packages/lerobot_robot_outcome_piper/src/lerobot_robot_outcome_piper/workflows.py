"""Thin official-LeRobot wrappers that inject the canonical Xbox processor."""

from __future__ import annotations

from typing import Any

from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config

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
    assert robot.hardware_acceptance_path is not None
    return robot, teleop


def _processor(robot: OutcomePiperConfig, teleop: OutcomePiperXboxConfig):
    safety = load_motion_safety(
        robot.safety_path,
        robot.hardware_acceptance_path,
        can_interface=robot.can_interface,
        firmware=robot.firmware,
    )
    return make_xbox_processor(
        safety,
        max_xyz_step_m=teleop.xyz_step_m,
        max_yaw_step_rad=teleop.yaw_step_rad,
        max_gripper_step_m=teleop.gripper_step_m,
        ik_max_nfev=teleop.ik_max_nfev,
        ik_timeout_s=teleop.ik_timeout_s,
        ik_residual_tolerance=teleop.ik_residual_tolerance,
        ik_min_singular_value=teleop.ik_min_singular_value,
    )


def teleoperate(cfg: Any) -> None:
    """Run the official teleop loop with the canonical Xbox processor."""

    from lerobot.scripts import lerobot_teleoperate as official
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

    robot_config, teleop_config = _validate_workflow_configs(cfg.robot, cfg.teleop)
    if cfg.fps != teleop_config.control_hz:
        raise ValueError("workflow fps must match the measured Xbox control_hz")
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
    teleop.connect()
    try:
        with motion_input_safety_scope():
            robot.connect()
            try:
                official.teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    duration=cfg.teleop_time_s,
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
        teleop.disconnect()
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)


def record(cfg: Any) -> Any:
    """Call the official recorder with the canonical Xbox processor."""

    from lerobot.scripts import lerobot_record as official

    robot_config, teleop_config = _validate_workflow_configs(cfg.robot, cfg.teleop)
    if cfg.dataset.fps != teleop_config.control_hz:
        raise ValueError("dataset fps must match the measured Xbox control_hz")
    if cfg.dataset.push_to_hub:
        raise ValueError(
            "record inside piper-can requires dataset.push_to_hub=false; "
            "publish after the hardware session"
        )
    with motion_input_safety_scope():
        return official.record(
            cfg,
            teleop_action_processor=_processor(robot_config, teleop_config),
        )
