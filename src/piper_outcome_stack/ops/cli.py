"""Small project CLI; LeRobot owns data, training, and checkpoint lifecycles."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .doctor import doctor_project
from .errors import OpsError, ValidationError
from .robot_doctor import robot_doctor


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piper-outcome-stack")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--root", default=".")
    robot = commands.add_parser("robot")
    robot_actions = robot.add_subparsers(dest="robot_action", required=True)
    robot_actions.add_parser("doctor")
    commands.add_parser("teleoperate", add_help=False)
    commands.add_parser("record", add_help=False)
    return parser


def _run(args: argparse.Namespace) -> Any:
    if args.command == "doctor":
        return doctor_project(args.root)
    if args.command == "robot" and args.robot_action == "doctor":
        return robot_doctor()
    raise ValidationError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in {"teleoperate", "record"}:
        from lerobot_robot_outcome_piper.cli import record_main, teleoperate_main

        command = raw_args[0]
        if command == "teleoperate":
            teleoperate_main(raw_args[1:])
        else:
            record_main(raw_args[1:])
        return 0
    args = _parser().parse_args(raw_args)
    try:
        _emit(_run(args))
        return 0
    except OpsError as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
