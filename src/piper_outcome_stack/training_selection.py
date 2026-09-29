"""Explicit, audited whole-episode selection for behavior cloning."""

import argparse
import json
from pathlib import Path


def is_successful_demonstration(outcome):
    return outcome.get("task_outcome") == "success" and outcome.get("data_valid") is True


def xbox_outcomes(root):
    from lerobot_robot_outcome_piper.raw_io import read_jsonl

    outcomes = {}
    for path in (Path(root) / "telemetry").glob("*/events.jsonl"):
        events = read_jsonl(path)
        saved = {
            (e["episode_index"], e["attempt"]) for e in events if e["event"] == "episode_saved"
        }
        for e in events:
            if e["event"] == "episode_outcome" and (e["episode_index"], e["attempt"]) in saved:
                index = e["episode_index"]
                if index in outcomes:
                    raise ValueError("duplicate saved episode outcome")
                outcomes[index] = e
    return outcomes


def require_successful_selection(root, episodes):
    if episodes is None or not episodes or len(set(episodes)) != len(episodes):
        raise ValueError(
            "Xbox behavior cloning requires an explicit distinct episode selection; use prepare-training"
        )
    outcomes = xbox_outcomes(root)
    for index in episodes:
        if not is_successful_demonstration(outcomes.get(index, {})):
            raise ValueError(f"episode {index} is not an explicitly successful valid demonstration")


def prepare(config_path, output_config, validation_episodes):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_robot_outcome_piper.action_audit import verify_telemetry

    cfg = json.loads(Path(config_path).read_text())
    root = Path(cfg["dataset"]["root"])
    if not (root / "xbox-conversion.json").exists():
        raise ValueError("this selection command requires an audited Xbox conversion")
    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=root)
    audit = verify_telemetry(root, dataset)
    outcomes = xbox_outcomes(root)
    successful = sorted(i for i, e in outcomes.items() if is_successful_demonstration(e))
    require_successful_selection(root, validation_episodes)
    training = [i for i in successful if i not in validation_episodes]
    if not training:
        raise ValueError("whole-episode split leaves no training demonstrations")
    if cfg["dataset"].get("episodes") is not None and cfg["dataset"]["episodes"] != training:
        raise ValueError("existing explicit episode selection conflicts with proposed split")
    cfg["dataset"]["episodes"] = training
    output_config = Path(output_config)
    manifest = output_config.with_suffix(".episodes.json")
    if output_config.exists() or manifest.exists():
        raise FileExistsError("training output already exists")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            dict(
                source=str(root.resolve()),
                audit=audit,
                train_episodes=training,
                validation_episodes=validation_episodes,
                excluded_episodes=[i for i in range(dataset.num_episodes) if i not in successful],
                outcomes=outcomes,
                split_unit="whole_episode",
            ),
            indent=2,
        )
    )
    output_config.write_text(json.dumps(cfg, indent=2))
    return dict(
        config=str(output_config),
        manifest=str(manifest),
        train_episodes=training,
        validation_episodes=validation_episodes,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--validation-episode", type=int, action="append", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.config, args.output_config, args.validation_episode), indent=2))
    return 0
