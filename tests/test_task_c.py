import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from piper_outcome_stack import task_c
from piper_outcome_stack.policy_trials import label, summarize
from piper_outcome_stack.retained_dataset import require_disjoint_sessions


@pytest.fixture
def c_config(tmp_path):
    start, goal = tmp_path / "start-03.png", tmp_path / "target-G.png"
    start.touch()
    goal.touch()
    return {
        "robot": {"scene": {"scene_id": "synthetic-c"}},
        "teleop": {},
        "dataset": {"num_episodes": 1, "fps": 50},
        "task_c": {
            "start_reference": str(start),
            "goal_reference": str(goal),
            "position_id": "start-03",
            "duration_s": 2.0,
        },
        "policy": {"checkpoint": None},
    }


@pytest.mark.parametrize("split", ["train", "development"])
def test_fixed_start_and_goal_recording(c_config, split):
    before = json.dumps(c_config)
    cfg = task_c.configure_recording(c_config, split)
    assert json.dumps(c_config) == before
    assert "task_c" not in cfg and "policy" not in cfg
    assert cfg["dataset"]["episode_time_s"] == 2
    assert cfg["recording_tasks"] == {"3": cfg["dataset"]["single_task"]}
    assert cfg["recording_position"] == "start-03" and cfg["recording_split"] == split
    assert cfg["recording_context"]["goal_reference"] == c_config["task_c"]["goal_reference"]
    assert cfg["recording_context"]["skill_id"] == "push_to_G"


def test_other_start_missing_reference_or_budget_is_refused(c_config):
    with pytest.raises(ValueError, match="train or development"):
        task_c.configure_recording(c_config, None)
    c_config["task_c"]["position_id"] = "start-02"
    with pytest.raises(ValueError, match="start-03"):
        task_c.select_task(c_config)
    c_config["task_c"]["position_id"] = "start-03"
    c_config["task_c"]["duration_s"] = None
    with pytest.raises(ValueError, match="positive"):
        task_c.select_task(c_config)
    Path(c_config["task_c"]["goal_reference"]).unlink()
    with pytest.raises(FileNotFoundError, match="goal_reference"):
        task_c.select_task(c_config)


def test_empty_template_does_not_claim_ready(tmp_path):
    from piper_outcome_stack.task_c_execution import execute

    path = Path(__file__).parents[1] / "configs/tasks/c.template.json"
    output = tmp_path / "check.json"
    with pytest.raises(ValueError, match="scene_id"):
        execute(path, "check", output)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and not report["hardware_io"]


def test_whole_chain_cannot_cross_split_even_across_different_sessions():
    train = NS(lineage=[{"source_session": "a", "task_context": {"chain_id": "whole01"}}])
    dev = NS(lineage=[{"source_session": "b", "task_context": {"chain_id": "whole01"}}])
    with pytest.raises(ValueError, match="chains overlap"):
        require_disjoint_sessions(train, dev)


def test_outcome_is_independent_of_program_status_and_grouped_by_skill(tmp_path, c_config):
    selected = task_c.select_task(c_config)
    path = tmp_path / "push.json"
    report = dict(
        task=selected["task"],
        task_context=selected["context"],
        mode="run",
        position=selected["position"],
        status="segment_ended_held",
        hold_confirmed=True,
    )
    path.write_text(json.dumps(report))
    label(path, "success")
    assert next(iter(summarize([path.with_suffix(".outcome.json")]).values()))["success"] == 1
    path = tmp_path / "bad.json"
    report["status"] = "failed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="program failure"):
        label(path, "success")
    report["mode"] = "check"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not task trials"):
        label(path, "failure")


def test_batch_manifest_keeps_the_whole_mixed_demonstration_and_older_fine_rows(
    tmp_path, monkeypatch
):
    from lerobot.datasets import lerobot_dataset
    from lerobot_robot_outcome_piper import action_audit
    from piper_outcome_stack import training_selection

    context = dict(task_family="C", start_id=task_c.POSITION, skill_id=task_c.SKILL)
    config = dict(
        dataset=dict(root=str(tmp_path / "dataset"), repo_id="test/pusht"),
        recording_split="train",
        recording_context=context,
        recording_target_precision="fine",
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    base = dict(
        task=task_c.INSTRUCTION,
        task_context=context,
        intended_split="train",
        task_outcome="success",
        data_valid=True,
        session="one",
    )
    mixed = dict(
        base,
        precision="mixed",
        eligible_frames_by_precision={"normal": 5, "fine": 5},
        source_episode_index=0,
    )
    outcomes = {
        0: mixed,
        1: mixed,
        2: dict(base, precision="fine"),
        3: dict(base, precision="normal", eligible_frames_by_precision={"normal": 5}),
    }
    ds = NS(
        repo_id="test/pusht",
        num_episodes=4,
        meta=NS(
            episodes=dict(
                tasks=[[task_c.INSTRUCTION]] * 4,
                dataset_from_index=[0, 5, 10, 15],
                dataset_to_index=[5, 10, 15, 20],
            )
        ),
    )
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", lambda *a, **kw: ds)
    monkeypatch.setattr(action_audit, "verify_telemetry", lambda *a: None)
    monkeypatch.setattr(training_selection, "xbox_outcomes", lambda _: outcomes)
    selection = task_c.capture_manifest([tmp_path])["sources"]["train"][0]
    assert [s["dataset_episode"] for s in selection["segments"]] == [0, 1, 2]
    assert selection["excluded"] == [dict(episode=3, outcome="success", precision="normal")]
