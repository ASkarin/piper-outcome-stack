from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from piper_outcome_stack.ops import cli as project_cli
from piper_outcome_stack.ops.cli import _parser
from piper_outcome_stack.ops.doctor import (
    REQUIRED_DIRECTORIES,
    _installed_package,
    _source_identity,
    doctor_project,
)
from piper_outcome_stack.ops.errors import IntegrityError

ROOT = Path(__file__).parents[1]


def test_project_doctor_accepts_non_main_git_checkout(monkeypatch):
    _use_valid_installed_distributions(monkeypatch)
    report = doctor_project(ROOT)
    assert report["status"] == "ok"
    assert report["source"]["kind"] == "git_checkout"
    assert len(report["source"]["commit"]) == 40
    assert report["dependencies"]["uv_lock"]["pyagxarm"]["commit"] == (
        "799b8412fbe8b9156bc9892d3dbeb2df7e98be71"
    )
    assert report["dependencies"]["uv_lock"]["lerobot_robot_outcome_piper"] == {
        "version": "0.1.0",
        "source": "workspace",
    }
    assert report["dependencies"]["installed"]["lerobot"]["vcs_commit"] == (
        "30da8e687a6dfc617fcd94afc367ac7071c376ce"
    )


def _distribution(*, version: str, direct_url: dict | str | None):
    def read_text(name: str) -> str | None:
        assert name == "direct_url.json"
        if isinstance(direct_url, dict):
            return json.dumps(direct_url)
        return direct_url

    return SimpleNamespace(version=version, read_text=read_text)


def _use_valid_installed_distributions(monkeypatch) -> None:
    distributions = {
        "pyAgxArm": _distribution(
            version="1.0.0",
            direct_url={
                "url": "https://github.com/agilexrobotics/pyAgxArm.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "799b8412fbe8b9156bc9892d3dbeb2df7e98be71",
                },
            },
        ),
        "lerobot": _distribution(
            version="0.6.0",
            direct_url={
                "url": "https://github.com/huggingface/lerobot.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "30da8e687a6dfc617fcd94afc367ac7071c376ce",
                },
            },
        ),
        "lerobot_robot_outcome_piper": _distribution(version="0.1.0", direct_url=None),
    }
    monkeypatch.setattr(importlib.metadata, "distribution", distributions.__getitem__)


def test_installed_package_requires_distribution(monkeypatch):
    def missing(_name: str):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(IntegrityError, match="is not installed"):
        _installed_package("lerobot", expected_version="0.6.0")


def test_project_doctor_rejects_wrong_runtime_dependency_commit(monkeypatch):
    _use_valid_installed_distributions(monkeypatch)
    valid_distribution = importlib.metadata.distribution

    def distribution(name: str):
        if name == "lerobot":
            return _distribution(
                version="0.6.0",
                direct_url={
                    "url": "https://github.com/huggingface/lerobot.git",
                    "vcs_info": {"vcs": "git", "commit_id": "b" * 40},
                },
            )
        return valid_distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    with pytest.raises(IntegrityError, match="does not match locked commit"):
        doctor_project(ROOT)


def test_installed_package_requires_exact_version(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _name: _distribution(version="0.5.0", direct_url=None),
    )
    with pytest.raises(IntegrityError, match="does not match locked version"):
        _installed_package("lerobot", expected_version="0.6.0")


@pytest.mark.parametrize(
    "direct_url",
    [
        None,
        "not-json",
        {"url": "https://example.invalid/lerobot.whl"},
        {
            "url": "https://github.com/huggingface/lerobot.git",
            "vcs_info": {"vcs": "git", "commit_id": "b" * 40},
        },
    ],
)
def test_installed_git_package_requires_exact_direct_url_commit(monkeypatch, direct_url):
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _name: _distribution(version="0.6.0", direct_url=direct_url),
    )
    with pytest.raises(IntegrityError):
        _installed_package(
            "lerobot",
            expected_version="0.6.0",
            expected_vcs_commit="a" * 40,
        )


def test_source_identity_accepts_git_archive_release(tmp_path: Path):
    commit = "a" * 40
    (tmp_path / ".piper-release-complete").write_text(commit + "\n", encoding="utf-8")
    assert _source_identity(tmp_path) == {
        "kind": "git_archive_release",
        "commit": commit,
        "branch": None,
    }


def test_project_doctor_accepts_complete_git_archive_release(tmp_path: Path, monkeypatch):
    _use_valid_installed_distributions(monkeypatch)
    release = tmp_path / "release"
    for relative in REQUIRED_DIRECTORIES:
        (release / relative).mkdir(parents=True, exist_ok=True)
    (release / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "configs/project.json", release / "configs/project.json")
    shutil.copy2(ROOT / "uv.lock", release / "uv.lock")
    preregistration = "docs/preregistration/PR-20260813-02.md"
    shutil.copy2(ROOT / preregistration, release / preregistration)
    commit = "b" * 40
    (release / ".piper-release-complete").write_text(commit + "\n", encoding="utf-8")

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
    assert parser.parse_args(["teleoperate"]).command == "teleoperate"
    assert parser.parse_args(["record"]).command == "record"
    for removed in ("experiment", "dataset", "asset", "checkpoint", "result", "freeze"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed])


@pytest.mark.parametrize(
    ("command", "entrypoint"),
    [("teleoperate", "teleoperate_main"), ("record", "record_main")],
)
def test_canonical_cli_forwards_fake_lerobot_workflows(monkeypatch, command, entrypoint):
    calls = []
    monkeypatch.setattr(
        f"lerobot_robot_outcome_piper.cli.{entrypoint}",
        lambda argv: calls.append(list(argv)),
    )

    assert project_cli.main([command, "--robot.type=outcome_piper", "--fake-smoke=true"]) == 0
    assert calls == [["--robot.type=outcome_piper", "--fake-smoke=true"]]
