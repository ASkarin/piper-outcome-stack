from types import SimpleNamespace as NS
import json
import pytest

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from piper_outcome_stack import smolvla_training


def test_multitask_contract_keeps_each_instruction_and_rejects_unknown():
    spec = {"input_contract": {"tasks": ["green left", "green right"]}}
    assert smolvla_training.validate_task(spec, "green right") == "green right"
    with pytest.raises(ValueError, match="frozen input contract"):
        smolvla_training.validate_task(spec, "task A")


def test_presplit_development_satisfies_official_trainer_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        smolvla_training,
        "configured_policy",
        lambda spec: SmolVLAConfig(device="cpu", push_to_hub=False),
    )
    spec = {
        "train_sources": [{"root": str(tmp_path / "source")}],
        "output_dir": str(tmp_path / "output"),
        "run_id": "test",
        "optimization": {"seed": 42, "batch_size": 32, "num_workers": 0},
        "budget": {"max_steps": 10000, "save_interval_steps": 2500, "eval_interval_steps": 1000},
    }
    cfg = smolvla_training.training_config(
        spec, NS(repo_id="local/retained", num_episodes=52), NS(num_episodes=13)
    )
    cfg.validate()
    assert cfg.dataset.eval_split == 0.2
    assert cfg.eval_steps == 1000
    assert not cfg.wandb.enable and not cfg.policy.push_to_hub


def test_repeated_pipeline_does_not_overwrite_completed_results(tmp_path):
    from piper_outcome_stack.smolvla_pipeline import run

    root = tmp_path / "run"
    root.mkdir()
    state = root / "pipeline-status.json"
    original = '{"status":"complete","checkpoint":"preserve"}'
    state.write_text(original)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"output_dir": str(root / "checkpoints-output")}))
    with pytest.raises(FileExistsError, match="already has results"):
        run(spec)
    assert state.read_text() == original


def test_repeated_training_preserves_status_before_loading_data(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    state = root / "training-status.json"
    original = '{"status":"complete"}'
    state.write_text(original)

    def unexpected(spec):
        raise AssertionError("existing run should be rejected before loading data")

    monkeypatch.setattr(smolvla_training, "make_views", unexpected)
    with pytest.raises(FileExistsError, match="already has results"):
        smolvla_training.train({"output_dir": str(root / "checkpoints-output")})
    assert state.read_text() == original
