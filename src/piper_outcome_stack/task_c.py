"""Fixed drawer skills, capture metadata and operator-started single-skill trials."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

SKILLS = {
    "open_drawer": {
        "instruction": "打开抽屉，松开把手并退出。",
        "start_condition": "抽屉关闭，夹爪空闲，机械臂位于本技能约定的起始范围。",
        "success_description": "抽屉达到约定开度，把手已释放，夹爪退出后续取杯区域。",
    },
    "store_cup": {
        "instruction": "把{object}放入抽屉，松开并退出。",
        "start_condition": "抽屉已打开，目标杯位于约定取物范围，机械臂位于本技能起始范围。",
        "success_description": "正确杯子稳定留在抽屉内，夹爪已释放并退出关闭路径。",
    },
    "close_drawer": {
        "instruction": "关闭抽屉并退出。",
        "start_condition": "杯子已放稳，夹爪空闲且位于抽屉外，关闭路径可用。",
        "success_description": "抽屉达到约定关闭状态，杯子未被挤出，机械臂已退出。",
    },
}

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


def select_skill(config, skill, object_id=None, chain_id=None):
    if skill not in SKILLS:
        raise ValueError(f"unknown C skill: {skill}")
    metadata = config.get("task_c")
    if not isinstance(metadata, dict):
        raise ValueError("C configuration requires task_c")
    scene = (config.get("robot") or {}).get("scene") or {}
    context = {
        "task_family": "C",
        "skill_id": skill,
        "scene_id": text_field(scene.get("scene_id"), "robot.scene.scene_id"),
        "drawer_id": text_field(metadata.get("drawer_id"), "task_c.drawer_id"),
        "start_condition": SKILLS[skill]["start_condition"],
        "success_description": SKILLS[skill]["success_description"],
    }
    task = SKILLS[skill]["instruction"]
    if skill == "store_cup":
        objects = metadata.get("objects", {})
        if not isinstance(objects, dict) or object_id not in objects:
            raise ValueError("store_cup requires an object ID from task_c.objects")
        name = text_field(objects[object_id], f"task_c.objects.{object_id}")
        task = task.format(object=name)
        context.update(object_id=object_id, object_name=name)
    elif object_id is not None:
        raise ValueError("--object applies only to store_cup")
    if chain_id is not None:
        context["chain_id"] = text_field(chain_id, "chain_id")
    duration = metadata.get("duration_s", {}).get(skill)
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"C configuration requires a positive task_c.duration_s.{skill}")
    position = text_field(metadata.get("position_id"), "task_c.position_id")
    return dict(task=task, context=context, duration_s=duration, position=position)


def configure_recording(config, skill, object_id, split, chain_id=None):
    if split not in ("train", "development"):
        raise ValueError("C recording requires explicit --split train|development")
    selected = select_skill(config, skill, object_id, chain_id)
    cfg = deepcopy(config)
    cfg.pop("task_c")
    cfg.pop("policy", None)
    dataset = cfg["dataset"]
    if dataset.get("fps") != 50:
        raise ValueError("C recording uses the existing 50Hz dataset contract")
    count = dataset.get("num_episodes")
    if type(count) is not int or count <= 0:
        raise ValueError("set the C session's positive dataset.num_episodes explicitly")
    if chain_id is not None and count != 1:
        raise ValueError("one chain ID labels one real attempt; capture one episode per skill")
    dataset.update(single_task=selected["task"], episode_time_s=selected["duration_s"])
    cfg.update(
        recording_tasks={"1": selected["task"]},
        recording_position=selected["position"],
        recording_split=split,
        recording_task_goal=count,
        recording_context=selected["context"],
        recording_layout_description=selected["context"]["start_condition"],
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
        for index in range(dataset.num_episodes):
            outcome = outcomes.get(index)
            if outcome is None:
                raise ValueError(f"C episode {index} is missing its outcome label")
            context = outcome.get("task_context")
            if context != config.get("recording_context") or not context:
                raise ValueError("C episode context differs from its capture configuration")
            if context.get("task_family") != "C" or context.get("skill_id") not in SKILLS:
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
        command.add_argument("--skill", choices=SKILLS, required=True)
        command.add_argument("--object", dest="object_id")
        command.add_argument("--chain-id")
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

        result = execute(
            args.config, args.skill, args.object_id, args.mode, args.output, args.chain_id
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
