from __future__ import annotations

import json
import tomllib
from pathlib import Path

import a3_outcome_stack

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_namespace_has_expected_version() -> None:
    assert a3_outcome_stack.__version__ == "0.2.2"
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == a3_outcome_stack.__version__


def test_only_canonical_package_and_cli_are_exposed() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"] == {"a3-outcome-stack": "a3_outcome_stack.ops.cli:main"}
    package_names = {
        path.name
        for path in (PROJECT_ROOT / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert package_names == {"a3_outcome_stack"}
    assert not (PROJECT_ROOT / "plugins").exists()
    assert not (PROJECT_ROOT / "src/a3_outcome_stack/robot").exists()


def test_post_adapter_split_has_no_duplicate_lifecycle_or_configuration_tree() -> None:
    for relative in (
        "configs/robot",
        "experiments/specs",
        "metadata",
        "results",
        "runs/.gitkeep",
    ):
        assert not (PROJECT_ROOT / relative).exists()


def test_project_metadata_uses_only_canonical_identity() -> None:
    project = json.loads((PROJECT_ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    assert project["project_id"] == "a3-outcome-stack"
    assert project["display_name"] == "A3 OutcomeStack"
    assert "legacy_project_ids" not in project
