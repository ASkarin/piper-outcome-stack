import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from piper_outcome_stack import task_c
from piper_outcome_stack.policy_trials import label, summarize
from piper_outcome_stack.retained_dataset import require_disjoint_sessions


@pytest.fixture
def c_config():
    return {
        "robot": {"scene": {"scene_id": "synthetic-c"}},
        "teleop": {},
        "dataset": {"num_episodes": 1, "fps": 50},
        "task_c": {
            "drawer_id": "drawer01",
            "objects": {"cup01": "蓝色杯子", "cup02": "红色杯子"},
            "position_id": "C-p1",
            "duration_s": dict.fromkeys(task_c.SKILLS, 2.0),
        },
        "policy": {"checkpoint": None},
    }


@pytest.mark.parametrize("skill", list(task_c.SKILLS))
def test_single_skill_capture_context_and_instruction(c_config, skill):
    before = json.dumps(c_config)
    object_id = "cup02" if skill == "store_cup" else None
    cfg = task_c.configure_recording(c_config, skill, object_id, "development", "chain01")
    assert json.dumps(c_config) == before
    assert "task_c" not in cfg and "policy" not in cfg
    assert cfg["dataset"]["num_episodes"] == 1
    assert cfg["dataset"]["episode_time_s"] == 2
    assert cfg["recording_tasks"] == {"1": cfg["dataset"]["single_task"]}
    context = cfg["recording_context"]
    assert context["skill_id"] == skill and context["chain_id"] == "chain01"
    if object_id:
        assert "红色杯子" in cfg["dataset"]["single_task"] and context["object_id"] == object_id
    else:
        assert "object_id" not in context


def test_missing_object_budget_scene_or_split_is_not_guessed(c_config):
    for object_id in (None, "unknown"):
        with pytest.raises(ValueError, match="object ID"):
            task_c.configure_recording(c_config, "store_cup", object_id, "train")
    with pytest.raises(ValueError, match="only to store_cup"):
        task_c.select_skill(c_config, "open_drawer", "cup01")
    with pytest.raises(ValueError, match="explicit --split"):
        task_c.configure_recording(c_config, "open_drawer", None, None)
    c_config["task_c"]["duration_s"]["open_drawer"] = None
    with pytest.raises(ValueError, match="positive"):
        task_c.select_skill(c_config, "open_drawer")
    c_config["robot"]["scene"]["scene_id"] = None
    with pytest.raises(ValueError, match="scene_id"):
        task_c.select_skill(c_config, "open_drawer")


def test_waiting_for_delivery_template_does_not_claim_ready(tmp_path):
    from piper_outcome_stack.task_c_execution import execute

    path = Path(__file__).parents[1] / "configs/tasks/c.template.json"
    output = tmp_path / "check.json"
    with pytest.raises(ValueError, match="scene_id"):
        execute(path, "open_drawer", None, "check", output)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and not report["hardware_io"]


def test_whole_chain_cannot_cross_split_even_across_different_sessions():
    train = NS(lineage=[{"source_session": "a", "task_context": {"chain_id": "whole01"}}])
    dev = NS(lineage=[{"source_session": "b", "task_context": {"chain_id": "whole01"}}])
    with pytest.raises(ValueError, match="chains overlap"):
        require_disjoint_sessions(train, dev)


def test_outcome_is_independent_of_program_status_and_grouped_by_skill(tmp_path, c_config):
    paths = []
    for skill in ("open_drawer", "close_drawer"):
        selected = task_c.select_skill(c_config, skill)
        path = tmp_path / (skill + ".json")
        path.write_text(
            json.dumps(
                dict(
                    task=selected["task"],
                    task_context=selected["context"],
                    mode="run",
                    position=selected["position"],
                    status="segment_ended_held",
                    hold_confirmed=True,
                )
            )
        )
        label(path, "success")
        paths.append(path.with_suffix(".outcome.json"))
    assert len(summarize(paths)) == 2
    path = tmp_path / "bad.json"
    report = json.loads((tmp_path / "open_drawer.json").read_text())
    report["status"] = "failed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="program failure"):
        label(path, "success")
    report["mode"] = "check"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not task trials"):
        label(path, "failure")
