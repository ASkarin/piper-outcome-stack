"""Exercise reuse identity against real Git tree changes."""

import runpy
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = runpy.run_path(str(ROOT / "infra/container/ci/build_inputs.py"))


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    (tmp_path / "uv.lock").write_text("locked dependencies\n")
    (tmp_path / "README.md").write_text("documentation\n")
    git("add", ".")
    git("commit", "-qm", "baseline")
    return tmp_path


def commit():
    git("add", "-A")
    git("commit", "-qm", "update")


def test_unrelated_pr_updates_reuse_the_same_inputs(repository):
    previous = INPUTS["image_input_key"]()
    (repository / "README.md").write_text("new documentation\n")
    source = repository / "src/piper_outcome_stack"
    source.mkdir(parents=True)
    (source / "smolvla_pipeline.py").write_text("# changed orchestration\n")
    commit()
    assert INPUTS["image_input_key"]() == previous


@pytest.mark.parametrize("change", ["content", "add", "delete", "rename", "mode"])
def test_build_input_changes_invalidate_reuse(repository, change):
    previous = INPUTS["image_input_key"]()
    lock = repository / "uv.lock"
    if change == "content":
        lock.write_text("changed dependency\n")
    elif change == "add":
        (repository / ".dockerignore").write_text("artifacts\n")
    elif change == "delete":
        lock.unlink()
        (repository / ".dockerignore").write_text("artifacts\n")
    elif change == "rename":
        lock.rename(repository / "pyproject.toml")
    else:
        git("update-index", "--chmod=+x", "uv.lock")
        git("commit", "-qm", "executable bit")
        assert INPUTS["image_input_key"]() != previous
        return
    commit()
    assert INPUTS["image_input_key"]() != previous


def test_base_branch_input_changes_are_in_the_merge_tree_key(repository):
    baseline = git("rev-parse", "HEAD")
    git("checkout", "-qb", "feature")
    (repository / "README.md").write_text("PR documentation\n")
    commit()
    pr_key = INPUTS["image_input_key"]()
    git("checkout", "-qb", "base", baseline)
    (repository / "uv.lock").write_text("base dependency update\n")
    commit()
    git("merge", "--no-ff", "-m", "PR merge tree", "feature")
    assert INPUTS["image_input_key"]() != pr_key


def test_all_local_docker_copy_sources_invalidate_reuse():
    dockerfile = (ROOT / "infra/container/Dockerfile").read_text().replace("\\\n", " ")
    for line in dockerfile.splitlines():
        parts = shlex.split(line)
        if not parts or parts[0] != "COPY" or parts[1].startswith("--from="):
            continue
        for source in parts[1:-1]:
            assert INPUTS["needs_image_build"]([source]), source
