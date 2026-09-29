import json

import pytest

from piper_outcome_stack.policy_trials import label, label_path, summarize, wilson


def trial(tmp_path, name, position="P1"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(dict(status="segment_ended_held", position=position)))
    return path


def test_label_writes_sidecar_once_without_touching_trial(tmp_path):
    path = trial(tmp_path, "trial-a")
    before = path.read_text()
    value = label(path, "success")
    assert value["position"] == "P1" and value["program_status"] == "segment_ended_held"
    assert path.read_text() == before
    with pytest.raises(FileExistsError):
        label(path, "failure")


def test_label_requires_position(tmp_path):
    with pytest.raises(ValueError, match="position"):
        label(trial(tmp_path, "trial-b", position=None), "success")


def test_summary_excludes_invalid_from_denominator(tmp_path):
    for i, outcome in enumerate(["success", "success", "failure", "invalid"]):
        label(trial(tmp_path, f"t{i}"), outcome)
    labels = sorted(tmp_path.glob("*.outcome.json"))
    assert labels[0] == label_path(tmp_path / "t0.json")
    row = summarize(labels)["P1"]
    assert (row["attempts"], row["invalid"], row["success_rate"]) == (3, 1, 2 / 3)
    low, high = row["wilson95"]
    assert 0 < low < 2 / 3 < high < 1


def test_wilson_bounds():
    assert wilson(0, 0) is None
    assert wilson(10, 10)[1] == pytest.approx(1.0)
    assert wilson(0, 10)[0] == pytest.approx(0.0)
