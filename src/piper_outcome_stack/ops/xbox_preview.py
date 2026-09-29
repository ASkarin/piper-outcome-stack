"""Input-only preview of the shared Xbox mode mapping; no Robot or IK instance."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time


def preview_config(config_path, buttons_report):
    from lerobot_robot_outcome_piper.config import OutcomePiperXboxConfig

    cfg = json.loads(Path(config_path).read_text())["teleop"]
    cfg.pop("type", None)
    report = json.loads(Path(buttons_report).read_text())
    if report["status"] != "input_measured" or report["device"]["guid"] != cfg["device_guid"]:
        raise ValueError("measured button report must match the configured Xbox GUID")
    buttons = report["buttons"]
    for key in ("hold_button", "emergency_stop_button"):
        if cfg[key] != buttons[key]:
            raise ValueError(f"measured {key} differs from configuration")
    if "mode_switch_button" in cfg and cfg["mode_switch_button"] != buttons["mode_switch_button"]:
        raise ValueError("measured RB differs from configuration")
    cfg["mode_switch_button"] = buttons["mode_switch_button"]
    if "translation_switch_button" in buttons:
        cfg["translation_switch_button"] = buttons["translation_switch_button"]
    return OutcomePiperXboxConfig(**cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--buttons-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    from lerobot_robot_outcome_piper.teleoperator import OutcomePiperXbox
    from lerobot_robot_outcome_piper.teleop_control import TeleopControl, TeleopState
    from lerobot_robot_outcome_piper.processor import input_deltas

    cfg = preview_config(args.config, args.buttons_report)
    xbox = OutcomePiperXbox(cfg)
    control = TeleopControl()
    control.confirm_hold()  # Input preview only; never evidence of physical holding.
    args.output.mkdir(parents=True, exist_ok=False)
    print(
        "只读手柄预览；保持确认是模拟值，不连接机械臂/相机，不运行IK。B或Ctrl+C退出。", flush=True
    )
    try:
        xbox.connect()
        with (args.output / "events.jsonl").open("x") as log:
            while True:
                raw = xbox.get_action()
                if raw["emergency_stop"]:
                    control.stop(True)
                    log.write(
                        json.dumps(dict(event="B", simulated_hold=True, mode=control.mode.value))
                        + "\n"
                    )
                    print("B事件已收到；未发送急停命令。", flush=True)
                    break
                previous_state = control.state
                intent, epoch = control.observe(
                    raw["hold"],
                    raw["neutral"],
                    raw["mode_switch"],
                    raw["home"],
                    raw["work"],
                    raw["translation_switch"],
                )
                xyz, rotation, gripper = input_deltas(
                    raw,
                    control.mode,
                    cfg.xyz_step_m,
                    cfg.rotation_step_rad,
                    cfg.gripper_step_m,
                    control.translation_strategy,
                )
                if intent in ("run", "center"):
                    intent, epoch = control.arm_input_intent(epoch, any((*xyz, *rotation)))
                if intent != "run":
                    xyz, rotation = [0.0] * 3, [0.0] * 3
                if intent not in ("run", "center"):
                    gripper = 0.0
                event = dict(
                    monotonic_s=time.monotonic(),
                    raw=raw,
                    mode=control.mode.value,
                    translation_strategy=control.translation_strategy.value,
                    state=control.state.value,
                    intent=intent,
                    epoch=epoch,
                    mode_event=control.mode_event,
                    pose_event=control.pose_event,
                    xyz_delta_m=xyz,
                    rotation_vector_rad=rotation,
                    gripper_delta_m=gripper,
                    simulated_hold=True,
                )
                log.write(json.dumps(event) + "\n")
                log.flush()
                if (
                    control.state is not previous_state
                    or control.mode_event is not None
                    or raw["home"]
                    or raw["work"]
                    or any((*xyz, *rotation, gripper))
                ):
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                if control.state in (
                    TeleopState.HOLD_REQUESTED,
                    TeleopState.CENTERING,
                    TeleopState.POSE_READY,
                ):
                    control.confirm_hold()
                time.sleep(1 / cfg.control_hz)
    except KeyboardInterrupt:
        pass
    finally:
        xbox.disconnect()
    return 0
