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
    audit = commands.add_parser("audit-dataset")
    audit.add_argument("--root", required=True)
    audit.add_argument("--repo-id", required=True)
    audit.add_argument("--raw-source", help="relocated source archive for a teach depth export")
    commands.add_parser("teleoperate", add_help=False)
    commands.add_parser("record", add_help=False)
    commands.add_parser("replay", add_help=False)
    commands.add_parser("train", add_help=False)
    commands.add_parser("prepare-training", add_help=False)
    commands.add_parser("sim", add_help=False)
    commands.add_parser("xbox-input", add_help=False)
    commands.add_parser("xbox-preview", add_help=False)
    commands.add_parser("capture-timing", add_help=False)
    commands.add_parser("teach-record", add_help=False)
    commands.add_parser("teach-collect", add_help=False)
    commands.add_parser("teach-convert", add_help=False)
    commands.add_parser("xbox-convert", add_help=False)
    commands.add_parser("xbox-recover-sealed", add_help=False)
    commands.add_parser("xbox-pauses", add_help=False)
    return parser


def _run(args: argparse.Namespace) -> Any:
    if args.command == "audit-dataset":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot_robot_outcome_piper.action_audit import verify_telemetry

        dataset = LeRobotDataset(args.repo_id, root=args.root)
        if args.raw_source is not None:
            from lerobot_robot_outcome_piper.teach_dataset import audit_teach

            return audit_teach(args.root, dataset, raw_source=args.raw_source)
        return verify_telemetry(args.root, dataset)
    if args.command == "doctor":
        return doctor_project(args.root)
    if args.command == "robot" and args.robot_action == "doctor":
        return robot_doctor()
    raise ValidationError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "prepare-training":
        from piper_outcome_stack.training_selection import main as selection_main

        return selection_main(raw_args[1:])
    if raw_args and raw_args[0] == "xbox-pauses":
        from lerobot_robot_outcome_piper.pause_review import main as pause_main

        return pause_main(raw_args[1:])
    if raw_args and raw_args[0] == "xbox-recover-sealed":
        from lerobot_robot_outcome_piper.xbox_recovery import main as recovery_main

        return recovery_main(raw_args[1:])
    if raw_args and raw_args[0] == "xbox-convert":
        from lerobot_robot_outcome_piper.xbox_raw import main as xbox_convert

        return xbox_convert(raw_args[1:])
    if raw_args and raw_args[0] == "teach-collect":
        from lerobot_robot_outcome_piper.teach_record import collect_main

        return collect_main(raw_args[1:])
    if raw_args and raw_args[0] == "teach-record":
        from lerobot_robot_outcome_piper.teach_record import main as teach_main

        return teach_main(raw_args[1:])
    if raw_args and raw_args[0] == "teach-convert":
        from lerobot_robot_outcome_piper.teach_dataset import main as convert_main

        return convert_main(raw_args[1:])
    if raw_args and raw_args[0] == "capture-timing":
        from .capture_timing import main as timing_main

        return timing_main(raw_args[1:])
    if raw_args and raw_args[0] == "xbox-preview":
        from .xbox_preview import main as preview_main

        return preview_main(raw_args[1:])
    if raw_args and raw_args[0] == "xbox-input":
        from .xbox_input import main as xbox_main

        return xbox_main(raw_args[1:])
    if raw_args and raw_args[0] == "train":
        from piper_outcome_stack.training import train_main

        train_main(raw_args[1:])
        return 0
    if raw_args and raw_args[0] == "sim":
        from piper_outcome_stack.sim.cli import main as sim_main

        return sim_main(raw_args[1:])
    if raw_args and raw_args[0] in {"teleoperate", "record", "replay"}:
        from lerobot_robot_outcome_piper.cli import record_main, teleoperate_main, replay_main

        command = raw_args[0]
        if command == "teleoperate":
            teleoperate_main(raw_args[1:])
        elif command == "replay":
            replay_main(raw_args[1:])
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
