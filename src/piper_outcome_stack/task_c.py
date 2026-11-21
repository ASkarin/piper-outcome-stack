"""Stage C Push-T capture, data selection and operator-started trials."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

SKILL = "push_to_G"
POSITION = "start-03"
SUCCESS_GOALS = {"train": 50, "development": 6}
INSTRUCTION = "在桌面推动 T 块，使位置和朝向与目标 G 对齐，再退出观察区。"

# The October 27/29 closeout recipe; C checkpoints and scene values are separate.
EXECUTION_RECIPE = {
    "control_hz": 50,
    "request_stride": 25,
    "denoising_steps": 10,
    "initial_delay_ticks": 20,
    "guidance": "native_joint6",
    "joint_lowpass": False,
    "gripper_projection": "zero",
    "action_padding": "t*initial_noise for25structural dimensions",
}


def text_field(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"C configuration requires {name}")
    return value


def select_task(config):
    meta = config.get("task_c") or {}
    scene = (config.get("robot") or {}).get("scene") or {}
    context = {
        "task_family": "C",
        "skill_id": SKILL,
        "start_id": POSITION,
        "scene_id": text_field(scene.get("scene_id"), "robot.scene.scene_id"),
        "start_condition": "按第三种参考图摆放 T 块，保留自然复位误差。",
        "success_description": "推动 T 块对齐 G 的三个角标，撤开夹爪后 T 块保持对齐。",
    }
    if meta.get("position_id") != POSITION:
        raise ValueError(f"C requires position_id={POSITION}")
    for key in ("start_reference", "goal_reference"):
        value = text_field(meta.get(key), f"task_c.{key}")
        if not Path(value).is_file():
            raise FileNotFoundError(f"C {key}: {value}")
        context[key] = value
    duration = meta.get("duration_s")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("C configuration requires positive task_c.duration_s")
    return dict(task=INSTRUCTION, context=context, duration_s=duration, position=POSITION)


def configure_recording(config, split):
    if split not in SUCCESS_GOALS:
        raise ValueError("C recording requires train or development")
    selected = select_task(config)
    cfg = deepcopy(config)
    cfg.pop("task_c")
    cfg.pop("policy", None)
    dataset = cfg["dataset"]
    if dataset.get("fps") != 50:
        raise ValueError("C recording requires the 50Hz dataset contract")
    dataset.update(
        single_task=selected["task"],
        episode_time_s=selected["duration_s"],
        num_episodes=SUCCESS_GOALS[split],
    )
    cfg.update(
        recording_tasks={"3": selected["task"]},
        recording_position=POSITION,
        recording_split=split,
        recording_task_goal=None,
        recording_target_precision="fine",
        recording_context=selected["context"],
        recording_layout_description="第三种起点，终点为桌面三个角标 G。",
    )
    return cfg


def capture_manifest(sessions):
    """Export successful C segments with their original session and chain grouping."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_robot_outcome_piper.action_audit import verify_telemetry
    from .training_selection import is_successful_demonstration, xbox_outcomes

    sources = {"train": [], "development": []}
    session_splits, chain_splits, roots = {}, {}, set()
    for session in sessions:
        config = json.loads((Path(session) / "config.json").read_text())
        root = Path(config["dataset"]["root"]).resolve()
        if root in roots:
            raise ValueError(f"duplicate C dataset: {root}")
        roots.add(root)
        dataset = LeRobotDataset(config["dataset"]["repo_id"], root=root)
        verify_telemetry(root, dataset)
        outcomes = xbox_outcomes(root)
        split = config["recording_split"]
        if split not in sources:
            raise ValueError("C capture manifest supports train/development sessions")
        segments, excluded = [], []
        required_precision = config.get("recording_target_precision")
        for index in range(dataset.num_episodes):
            outcome = outcomes.get(index)
            if outcome is None:
                raise ValueError(f"C episode {index} is missing its outcome label")
            context = outcome.get("task_context")
            if context != config.get("recording_context") or not context:
                raise ValueError("C episode context differs from its capture configuration")
            if (
                context.get("task_family") != "C"
                or context.get("skill_id") != SKILL
                or context.get("start_id") != POSITION
            ):
                raise ValueError("manifest requires C skill metadata")
            if outcome["intended_split"] != split:
                raise ValueError("C episode split differs from its capture configuration")
            source_session = outcome.get("source_session", outcome["session"])
            for groups, key in (
                (session_splits, source_session),
                (chain_splits, context.get("chain_id")),
            ):
                if key is not None:
                    if key in groups and groups[key] != split:
                        raise ValueError(f"C session/chain crosses data splits: {key}")
                    groups[key] = split
            if not is_successful_demonstration(outcome):
                excluded.append(dict(episode=index, outcome=outcome["task_outcome"]))
                continue
            if required_precision is not None and outcome["precision"] != required_precision:
                excluded.append(
                    dict(
                        episode=index,
                        outcome=outcome["task_outcome"],
                        precision=outcome["precision"],
                    )
                )
                continue
            task = outcome["task"]
            if dataset.meta.episodes["tasks"][index] != [task]:
                raise ValueError("C task text differs from converted dataset")
            frames = int(dataset.meta.episodes["dataset_to_index"][index]) - int(
                dataset.meta.episodes["dataset_from_index"][index]
            )
            segments.append(
                dict(
                    dataset_episode=index,
                    first_frame=0,
                    end_frame_exclusive=frames,
                    provenance=dict(
                        source_session=source_session,
                        source_episode=outcome.get("source_episode_index", index),
                        task=task,
                        split=split,
                        task_context=context,
                    ),
                )
            )
        sources[split].append(
            dict(root=str(root), repo_id=dataset.repo_id, segments=segments, excluded=excluded)
        )
    return dict(
        task_family="C", split_unit="source_session_and_chain", sources=sources, hardware_io=False
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("check", "run"):
        command = commands.add_parser(mode)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--session", type=Path, action="append", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "manifest":
        result = capture_manifest(args.session)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    else:
        from .task_c_execution import execute

        result = execute(args.config, args.mode, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
