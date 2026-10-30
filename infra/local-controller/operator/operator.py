"""Stable operator commands over the existing CLI. No hardware code at import."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
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


B_LAYOUTS = {
    "1": "相机画面取物区：绿色在上方原取物点，六边形在下方原取物点",
    "2": "相机画面取物区：六边形在上方原取物点，绿色在下方原取物点",
}


def verify_smolvla_bundle(bundle, model):
    """Like ACT verify-only, inspect fixed assets instead of loading GPU models."""
    freeze = load(bundle / "freeze.json")
    if model not in freeze["models"] or not freeze.get("files"):
        raise ValueError("模型或文件清单不在已固化SmolVLA资产中")
    for item in freeze["files"]:
        if item.get("model") not in (None, model):
            continue
        path = Path(item["path"])
        path = path if path.is_absolute() else bundle / path
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen SmolVLA asset changed: {path}")
    return freeze


def configure_b_session(cfg, args):
    split = getattr(args, "split", None)
    if split is None:
        print("本次用途：1训练候选采集（每任务5回合）；2开发检查（每任务1回合）。")
        choice = input("选择用途 1/2：").strip()
        if choice not in ("1", "2"):
            raise ValueError("请选择1或2；尚未连接设备")
        split = {"1": "train", "2": "development"}[choice]
    layout = getattr(args, "layout", None)
    if layout is None:
        for key, description in B_LAYOUTS.items():
            print(f"布局{key}：{description}")
        layout = input("选择布局 1/2：").strip()
    if layout not in B_LAYOUTS or split not in ("train", "development"):
        raise ValueError("未知布局或数据用途；尚未连接设备")
    goal = 5 if split == "train" else 1
    cfg.update(
        recording_position="B-layout" + layout,
        recording_layout_description=B_LAYOUTS[layout],
        recording_split=split,
        recording_task_goal=goal,
    )
    cfg["dataset"]["num_episodes"] = goal * len(cfg["recording_tasks"])
    scene = cfg["robot"].get("scene")
    if scene is not None:
        scene["work_area_notes"] = (
            "B shared ACT/SmolVLA collection; both objects visible; two fixed destination halves. "
            "Left/right means unmirrored raw camera image. Left half is narrower. "
            + B_LAYOUTS[layout]
            + f"; intended split={split}; session stays in one split."
        )
    print(f"本次：{split} / B-layout{layout} / 每任务计划保存{goal}回合。")
    print("左右以未镜像的原始相机画面为准；失败单列，不自动充当成功示范。")
    return cfg


def execute(args, profile):
    cli = str(Path(profile["python"]).parent / "piper-outcome-stack")
    if args.command == "smolvla":
        if "smolvla_bundle" not in profile:
            raise ValueError("尚未配置已固化的 smolvla_bundle")
        bundle = Path(profile["smolvla_bundle"])
        freeze = verify_smolvla_bundle(bundle, args.model)
        if args.model not in freeze["models"] or args.skill not in freeze["tasks"]:
            raise ValueError("模型或技能不在已固化SmolVLA清单中")
        if args.verify_only:
            print("Frozen SmolVLA assets verified; no model loading or hardware I/O.")
            return 0
        command = [
            profile["python"],
            str(bundle / "run-policy.py"),
            "run",
            args.model,
            args.skill,
            args.layout,
        ]
        return subprocess.call(command)
    if args.command == "c":
        run = new_run(profile, "task-c")
        output = run / "trial.json"
        command = [
            profile["python"],
            "-m",
            "piper_outcome_stack.task_c",
            args.mode,
            "--config",
            str(args.config),
            "--skill",
            args.skill,
            "--output",
            str(output),
        ]
        if args.object_id is not None:
            command += ["--object", args.object_id]
        if args.chain_id is not None:
            command += ["--chain-id", args.chain_id]
        if args.mode == "check":
            return subprocess.call(command)
        from provenance import capture

        capture(profile, run)
        code = logged(["sudo", "piper-socketcan", "exec", "--", *command], run)
        if output.exists() and load(output).get("hardware_io"):
            subprocess.call(
                [
                    profile["python"],
                    "-m",
                    "piper_outcome_stack.policy_trials",
                    "prompt",
                    str(output),
                ]
            )
        return code
    if args.command == "act":
        bundle = Path(profile["act_bundle"])
        command = [profile["python"], str(bundle / "run.py"), args.skill, args.layout]
        if args.verify_only:
            command.append("--verify-only")
        return subprocess.call(command)
    if args.command in ("status", "link", "firmware", "limits", "acceleration"):
        return subprocess.call(["sudo", "-n", "/usr/local/sbin/piper-query", args.command])
    if args.command == "record":
        mode = getattr(args, "mode", "a")
        if (mode == "a" and getattr(args, "split", None)) or (
            mode != "b" and getattr(args, "layout", None)
        ):
            raise ValueError("--layout仅用于B采集；--split用于B/C采集")
        if mode != "c" and any(
            getattr(args, key, None) for key in ("skill", "object_id", "chain_id")
        ):
            raise ValueError("--skill/--object/--chain-id仅用于C采集")
        root = Path(profile["raw_parent"])
        mounted = subprocess.run(
            ["findmnt", "-rn", "-M", profile["ssd_mount"], "-o", "UUID"],
            capture_output=True,
            text=True,
            check=False,
        )
        if mounted.returncode != 0 or mounted.stdout.strip() != profile["ssd_uuid"]:
            raise ValueError(
                f"采集SSD未挂载到 {profile['ssd_mount']}（预期UUID {profile['ssd_uuid']}）；"
                "请在控制机终端挂载后再试。尚未连接设备。"
            )
        try:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=root) as probe:
                probe.write(b"piper storage write check\n")
                probe.flush()
                os.fsync(probe.fileno())
        except OSError as exc:
            raise ValueError(f"采集SSD不可写：{root}：{exc}；尚未连接设备。") from exc
        config_key = {"a": "record_config", "b": "record_b_config", "c": "record_c_config"}[mode]
        if config_key not in profile:
            raise ValueError(f"尚未配置采集入口 {config_key}")
        cfg = load(profile[config_key])
        if mode == "b":
            if not cfg.get("recording_tasks"):
                raise ValueError("B采集配置缺少任务菜单")
            cfg = configure_b_session(cfg, args)
        elif mode == "c":
            from piper_outcome_stack.task_c import configure_recording

            cfg = configure_recording(cfg, args.skill, args.object_id, args.split, args.chain_id)
        if mode in ("b", "c"):
            root = root / ("task-" + mode)
            root.mkdir(exist_ok=True)
        run = new_run(profile, "record-" + mode if mode in ("b", "c") else "record")
        cfg["raw_root"] = str(root / run.name)
        cfg["dataset"]["root"] = str(
            Path(profile["dataset_parent"])
            / ("task-" + mode if mode in ("b", "c") else "")
            / run.name
        )
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
    record = sub.add_parser("record")
    record.add_argument("mode", nargs="?", choices=("a", "b", "c"), default="a")
    record.add_argument("--layout", choices=("1", "2"))
    record.add_argument("--split", choices=("train", "development"))
    record.add_argument("--skill", choices=("open_drawer", "store_cup", "close_drawer"))
    record.add_argument("--object", dest="object_id")
    record.add_argument("--chain-id")
    act = sub.add_parser("act", help="Run the selected frozen ACT bundle")
    act.add_argument("skill", choices=("green-left", "green-right", "blue-left", "blue-right"))
    act.add_argument("layout", choices=("1", "2"))
    act.add_argument("--verify-only", action="store_true")
    smolvla = sub.add_parser("smolvla", help="Run the frozen 50k original/flat RTC bundle")
    smolvla.add_argument("model", choices=("original", "flat"))
    smolvla.add_argument("skill", choices=("green-left", "green-right", "blue-left", "blue-right"))
    smolvla.add_argument("layout", choices=("1", "2"))
    smolvla.add_argument("--verify-only", action="store_true")
    task_c = sub.add_parser("c", help="Check or run one C skill from the current pose")
    task_c.add_argument("mode", choices=("check", "run"))
    task_c.add_argument("--config", type=Path, required=True)
    task_c.add_argument(
        "--skill", choices=("open_drawer", "store_cup", "close_drawer"), required=True
    )
    task_c.add_argument("--object", dest="object_id")
    task_c.add_argument("--chain-id")
    for name in ("recover", "status", "link", "firmware", "limits", "acceleration"):
        sub.add_parser(name)
    for name in ("convert", "audit", "pauses"):
        sub.add_parser(name).add_argument("session", type=Path)
    sub.add_parser("replay").add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        return execute(args, load(args.profile))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"[未启动] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
