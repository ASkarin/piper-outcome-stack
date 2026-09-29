"""Stable operator commands over the existing CLI. No hardware code at import."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


def load(path):
    return json.loads(Path(path).read_text())


def new_run(profile, kind):
    base = Path(profile["runs"]) / kind
    base.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=base)
    )


def logged(command, run):
    return subprocess.call(
        [
            "script",
            "--quiet",
            "--return",
            "--flush",
            "--command",
            shlex.join(command),
            str(run / "terminal.log"),
        ]
    )


def execute(args, profile):
    cli = str(Path(profile["python"]).parent / "piper-outcome-stack")
    if args.command in ("status", "link", "firmware", "limits", "acceleration"):
        return subprocess.call(["sudo", "-n", "/usr/local/sbin/piper-query", args.command])
    if args.command == "record":
        mount = subprocess.check_output(
            ["findmnt", "-rn", "-M", profile["ssd_mount"], "-o", "UUID"], text=True
        ).strip()
        if mount != profile["ssd_uuid"]:
            raise ValueError("Expected dataset SSD is not mounted")
        run = new_run(profile, "record")
        cfg = load(profile["record_config"])
        root = Path(profile["raw_parent"])
        root.mkdir(parents=True, exist_ok=True)
        cfg["raw_root"] = str(root / run.name)
        cfg["dataset"]["root"] = str(Path(profile["dataset_parent"]) / run.name)
        cfg["dataset"]["repo_id"] = "local/xbox-" + run.name
        cfg["resume"] = False
        if cfg["robot"].get("safety_path"):
            safety = run / "safety.json"
            safety.write_text(json.dumps(load(cfg["robot"]["safety_path"]), indent=2))
            cfg["robot"]["safety_path"] = str(safety)
        path = run / "config.json"
        path.write_text(json.dumps(cfg, indent=2))
        from provenance import capture

        capture(profile, run)  # Before sudo/CAN/camera; never in the live save loop.
        print("Session:", run, flush=True)
        return logged(
            ["sudo", "piper-socketcan", "exec", "--", cli, "record", "--config_path=" + str(path)],
            run,
        )
    if args.command == "recover":
        run = new_run(profile, "recover")
        snapshot = run / "config.json"
        snapshot.write_text(json.dumps(load(profile["record_config"]), indent=2))
        return logged(
            [
                "sudo",
                "piper-socketcan",
                "exec",
                "--",
                profile["python"],
                str(Path(profile["source"]) / "infra/local-controller/operator/recover.py"),
                str(snapshot),
                str(run / "recovery"),
            ],
            run,
        )
    if args.command == "replay":
        cfg = load(args.config)
        if cfg.get("trajectory") is None:
            raise ValueError("Replay config must explicitly select continuous trajectory settings")
        run = new_run(profile, "replay")
        cfg["trajectory_report_path"] = str(run / "result.json")
        path = run / "config.json"
        path.write_text(json.dumps(cfg, indent=2))
        print("Replay requires the selected dataset start pose; Xbox is not used.", flush=True)
        return logged(
            ["sudo", "piper-socketcan", "exec", "--", cli, "replay", "--config_path=" + str(path)],
            run,
        )
    cfg = load(Path(args.session) / "config.json")
    ds = cfg["dataset"]
    if args.command == "convert":
        command = [
            cli,
            "xbox-convert",
            "--raw-root",
            cfg["raw_root"],
            "--output",
            ds["root"],
            "--repo-id",
            ds["repo_id"],
        ]
    elif args.command == "audit":
        command = [cli, "audit-dataset", "--root", ds["root"], "--repo-id", ds["repo_id"]]
    else:
        run = new_run(profile, "pauses")
        command = [
            cli,
            "xbox-pauses",
            "--root",
            ds["root"],
            "--output",
            str(run / "candidates.json"),
        ]
    return subprocess.call(command)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PiPER operator shortcuts; motion/recovery require the operator terminal."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path.home() / ".config/piper-outcome-stack/operator/profile.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "recover", "status", "link", "firmware", "limits", "acceleration"):
        sub.add_parser(name)
    for name in ("convert", "audit", "pauses"):
        sub.add_parser(name).add_argument("session", type=Path)
    sub.add_parser("replay").add_argument("config", type=Path)
    args = parser.parse_args(argv)
    return execute(args, load(args.profile))


if __name__ == "__main__":
    raise SystemExit(main())
