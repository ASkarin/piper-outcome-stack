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
