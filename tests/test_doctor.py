from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from a3_outcome_stack.ops.cli import _parser
from a3_outcome_stack.ops.doctor import REQUIRED_DIRECTORIES, _source_identity, doctor_project

ROOT = Path(__file__).parents[1]


def test_project_doctor_accepts_non_main_git_checkout():
    report = doctor_project(ROOT)
    assert report["status"] == "ok"
    assert report["source"]["kind"] == "git_checkout"
    assert len(report["source"]["commit"]) == 40
    assert report["dependencies"]["uv_lock"]["el_a3_sdk"]["commit"] == (
        "ea7231f784ebb37e4c4120f7be8e3670514dc9ee"
    )
    assert report["dependencies"]["uv_lock"]["lerobot_robot_a3"]["commit"] == (
        "bf188864ef3922f8caded5cc19cc43b8061c4b22"
    )


def test_source_identity_accepts_git_archive_release(tmp_path: Path):
    commit = "a" * 40
    (tmp_path / ".a3-release-complete").write_text(commit + "\n", encoding="utf-8")
    assert _source_identity(tmp_path) == {
        "kind": "git_archive_release",
        "commit": commit,
        "branch": None,
    }


def test_project_doctor_accepts_complete_git_archive_release(tmp_path: Path):
    release = tmp_path / "release"
    for relative in REQUIRED_DIRECTORIES:
        (release / relative).mkdir(parents=True, exist_ok=True)
    (release / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "configs/project.json", release / "configs/project.json")
    shutil.copy2(ROOT / "uv.lock", release / "uv.lock")
    preregistration = "docs/preregistration/PR-20260722-01.md"
    shutil.copy2(ROOT / preregistration, release / preregistration)
    commit = "b" * 40
    (release / ".a3-release-complete").write_text(commit + "\n", encoding="utf-8")

    report = doctor_project(release)
    assert report["status"] == "ok"
    assert report["source"] == {
        "kind": "git_archive_release",
        "commit": commit,
        "branch": None,
    }


def test_cli_has_no_parallel_experiment_or_dataset_lifecycle():
    parser = _parser()
    assert parser.parse_args(["doctor"]).command == "doctor"
    for removed in ("experiment", "dataset", "asset", "checkpoint", "result", "freeze"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed])
