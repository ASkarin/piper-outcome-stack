"""Guided SDL input measurement; imports no robot, camera or CAN implementation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

STAGES = (
    ("neutral", "松开所有按键、摇杆和扳机，保持不动"),
    ("hold_button", "按下并松开要用作遥操作控制的肩键，只操作这一个键"),
    ("emergency_stop_button", "按下并松开 B 键，只操作 B 键"),
    ("left_stick_right", "左摇杆向右推到底，再回中"),
    ("left_stick_up", "左摇杆向上推到底，再回中"),
    ("right_stick_right", "右摇杆向右推到底，再回中"),
    ("right_stick_up", "右摇杆向上推到底，再回中"),
    ("left_trigger", "左扳机按到底，再完全释放"),
    ("right_trigger", "右扳机按到底，再完全释放"),
)


def identity(joystick):
    return dict(
        name=joystick.get_name(),
        guid=joystick.get_guid(),
        instance_id=joystick.get_instance_id(),
        axes=joystick.get_numaxes(),
        buttons=joystick.get_numbuttons(),
        hats=joystick.get_numhats(),
    )


def sample_stage(pygame, joystick, duration, hz, emit, *, clock=time.monotonic, sleep=time.sleep):
    started = clock()
    samples = []
    pressed_events = set()
    while clock() - started < duration:
        pygame.event.pump()
        for event in pygame.event.get():
            if (
                event.type == pygame.JOYDEVICEREMOVED
                and event.instance_id == joystick.get_instance_id()
            ):
                raise RuntimeError(
                    "selected Xbox disconnected; measurement ended without reconnecting"
                )
            if (
                event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP)
                and event.instance_id == joystick.get_instance_id()
            ):
                if event.type == pygame.JOYBUTTONDOWN:
                    pressed_events.add(event.button)
                emit(
                    "button",
                    button=event.button,
                    pressed=event.type == pygame.JOYBUTTONDOWN,
                    received_monotonic_s=clock(),
                )
        if not joystick.get_init():
            raise RuntimeError("selected Xbox unavailable")
        sample = dict(
            received_monotonic_s=clock(),
            axes=[float(joystick.get_axis(i)) for i in range(joystick.get_numaxes())],
            buttons=[int(joystick.get_button(i)) for i in range(joystick.get_numbuttons())],
            hats=[list(joystick.get_hat(i)) for i in range(joystick.get_numhats())],
        )
        if not all(math.isfinite(v) for v in sample["axes"]):
            raise RuntimeError("nonfinite Xbox axis value")
        samples.append(sample)
        emit("sample", **sample)
        sleep(1 / hz)
    if not samples:
        raise RuntimeError("no Xbox samples captured")
    return dict(
        samples=len(samples),
        axes_min=[min(s["axes"][i] for s in samples) for i in range(joystick.get_numaxes())],
        axes_max=[max(s["axes"][i] for s in samples) for i in range(joystick.get_numaxes())],
        pressed_buttons=sorted(
            pressed_events | {i for s in samples for i, v in enumerate(s["buttons"]) if v}
        ),
    )


def measured_buttons(stages):
    if stages["neutral"]["pressed_buttons"]:
        raise ValueError("neutral measurement contains pressed buttons")
    result = {}
    for key in ("hold_button", "emergency_stop_button"):
        pressed = stages[key]["pressed_buttons"]
        if len(pressed) != 1:
            raise ValueError(f"{key}: expected exactly one measured button, got {pressed}")
        result[key] = pressed[0]
    if result["hold_button"] == result["emergency_stop_button"]:
        raise ValueError("shoulder and B measurements selected the same button")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="enumerate Xbox inputs only")
    parser.add_argument("--index", type=int, help="explicit SDL index from --list")
    parser.add_argument("--output", type=Path, help="new measurement directory")
    parser.add_argument(
        "--seconds",
        type=float,
        default=4,
        help="seconds per input measurement, not a control limit",
    )
    parser.add_argument(
        "--hz", type=float, default=60, help="input sampling rate, not a robot control rate"
    )
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.hz)):
        parser.error("seconds and hz must be finite and positive")
    if not args.list and (args.index is None or args.output is None):
        parser.error("measurement requires --index and --output")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    import pygame

    pygame.display.init()
    pygame.joystick.init()
    devices = []
    try:
        for index in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(index)
            joystick.init()
            devices.append(joystick)
        if args.list:
            print(
                json.dumps(
                    [dict(index=i, **identity(j)) for i, j in enumerate(devices)],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.index < 0 or args.index >= len(devices):
            raise ValueError("selected SDL index does not exist; enumerate again")
        joystick = devices[args.index]
        args.output.mkdir(parents=True, exist_ok=False)
        report = dict(
            scope="Xbox input only; no robot/CAN/D435 access",
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            device=identity(joystick),
            stages={},
            status="incomplete",
            robot_hold_acceptance="not_performed",
            mapping_frozen=False,
        )
        with (args.output / "events.jsonl").open("x") as stream:
            stage = None

            def emit(event, **payload):
                stream.write(
                    json.dumps(dict(event=event, stage=stage, **payload), allow_nan=False) + "\n"
                )
                stream.flush()

            try:
                print("只测手柄输入。每轮先按 Enter，然后在采样期间按提示操作。")
                for stage, instruction in STAGES:
                    input(f"{instruction}；准备好后按 Enter，随后采样 {args.seconds:g} 秒：")
                    result = sample_stage(pygame, joystick, args.seconds, args.hz, emit)
                    report["stages"][stage] = result
                    print(json.dumps(dict(stage=stage, **result), ensure_ascii=False))
                report["buttons"] = measured_buttons(report["stages"])
                report["status"] = "input_measured"
                emit("measurement_complete", buttons=report["buttons"])
            except BaseException as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                emit("measurement_failed", error=report["error"])
                raise
            finally:
                report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                (args.output / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        print(f"测量记录：{args.output / 'report.json'}；死区、轴方向和保持参数仍须核定。")
        return 0
    except (ValueError, RuntimeError, OSError, EOFError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        for joystick in devices:
            joystick.quit()
        pygame.joystick.quit()
        pygame.display.quit()


if __name__ == "__main__":
    raise SystemExit(main())
