from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import pwd
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ROOT / "infra/container"
sys.path.insert(0, str(CONTAINER / "lib"))


def load_script(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    module = load_script("gpu_runner", CONTAINER / "bin/piper-gpu-run")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "main.py").write_text('print("baseline")\n')
    subprocess.run(["git", "-C", str(repo), "add", "main.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        check=True,
    )
    user = pwd.getpwuid(os.getuid()).pw_name
    root = tmp_path / "piper"
    (root / "runs" / user).mkdir(parents=True)
    (root / "locks").mkdir()
    monkeypatch.setattr(module, "PIPER_ROOT", root)
    monkeypatch.setattr(
        module, "load_runtime_config", lambda: {"admin_user": user, "collaborator_user": "other"}
    )
    monkeypatch.setattr(module, "gpu_inventory", lambda: [{"index": 0, "uuid": "GPU-test"}])
    return module, repo, root, user


def invoke(runtime, monkeypatch, command, *options):
    module, repo, root, user = runtime
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "piper-gpu-run",
            "--gpus",
            "0",
            "--repo",
            str(repo),
            "--python",
            sys.executable,
            *options,
            "--",
            *command,
        ],
    )
    return module.main()


def test_dirty_development_captures_source_and_actual_runtime(runtime, monkeypatch):
    module, repo, root, user = runtime
    (repo / "main.py").write_text('print("edited")\n')
    (repo / "new.py").write_text('print("new source")\n')
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert (
        invoke(
            runtime,
            monkeypatch,
            [
                "python",
                "-c",
                'import os,sys; print(sys.executable); assert os.environ["HF_HUB_OFFLINE"] == "0"; '
                'assert os.environ["WANDB_MODE"] == "disabled"; assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test"',
            ],
        )
        == 0
    )
    run = next((root / "runs" / user).iterdir())
    environment = json.loads((run / "environment.json").read_text())
    assert environment["python"]["executable"] == sys.executable
    assert environment["mode"] == "development"
    assert environment["git"]["dirty"]
    assert '+print("edited")' in (run / "working-tree.patch").read_text()
    assert (run / "untracked-source/new.py").read_text() == 'print("new source")\n'
    assert sys.executable in (run / "output.log").read_text()


def test_failure_and_duplicate_run_preserve_first_result(runtime, monkeypatch):
    _, _, root, user = runtime
    assert (
        invoke(runtime, monkeypatch, ["python", "-c", "raise SystemExit(7)"], "--run-id", "failed")
        == 7
    )
    run = root / "runs" / user / "failed"
    original = (run / "summary.json").read_bytes()
    assert json.loads(original)["exit_code"] == 7
    assert invoke(runtime, monkeypatch, ["python", "-c", "pass"], "--run-id", "failed") != 0
    assert (run / "summary.json").read_bytes() == original


def test_formal_accepts_unmerged_commit_and_local_manifest(runtime, monkeypatch, tmp_path):
    _, repo, root, user = runtime
    config = tmp_path / "config.json"
    config.write_text('{"seed": 1}')
    manifest = tmp_path / "dataset.json"
    manifest.write_text('{"version": "v1"}')
    options = (
        "--formal",
        "--config",
        str(config),
        "--dataset-manifest",
        str(manifest),
        "--run-id",
        "formal",
    )
    assert invoke(runtime, monkeypatch, ["python", "-c", "pass"], *options) == 0
    assert (
        root / "runs" / user / "formal/dataset-manifest.json"
    ).read_bytes() == manifest.read_bytes()
    (repo / "main.py").write_text('print("dirty")')
    assert invoke(runtime, monkeypatch, ["python", "-c", "pass"], *options) != 0


def test_formal_requires_provenance(runtime, monkeypatch):
    assert invoke(runtime, monkeypatch, ["python", "-c", "pass"], "--formal") != 0


def test_gpu_indexes_uuid_and_alias_duplicates(runtime):
    module = runtime[0]
    inventory = [{"index": 0, "uuid": "GPU-zero"}, {"index": 1, "uuid": "GPU-one"}]
    assert [gpu["uuid"] for gpu in module.parse_gpus("1,GPU-zero", inventory)] == [
        "GPU-one",
        "GPU-zero",
    ]
    with pytest.raises(Exception, match="repeated"):
        module.parse_gpus("0,GPU-zero", inventory)
    with pytest.raises(Exception, match="unknown"):
        module.parse_gpus("3", inventory)


def test_python_console_script_uses_selected_interpreter(runtime, tmp_path):
    module = runtime[0]
    script = tmp_path / "tool"
    script.write_text('#!/wrong/python\nprint("ok")\n')
    script.chmod(0o755)
    assert module.prepare_command([str(script), "--help"], sys.executable) == [
        sys.executable,
        str(script),
        "--help",
    ]
    assert module.prepare_command(["torchrun", "train.py"], sys.executable) == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "train.py",
    ]


def test_profile_preserves_active_venv_and_network_settings(tmp_path):
    env = os.environ.copy()
    env.update(
        VIRTUAL_ENV=str(tmp_path),
        HF_HUB_OFFLINE="0",
        WANDB_MODE="disabled",
        BASH_ENV=str(CONTAINER / "profile.sh"),
    )
    output = subprocess.check_output(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$PATH" "$HF_HUB_OFFLINE" "$WANDB_MODE"',
            "profile-test",
            str(CONTAINER / "profile.sh"),
        ],
        env=env,
        text=True,
    )
    lines = output.splitlines()
    assert lines[0].startswith(str(tmp_path / "bin") + ":")
    assert lines[1:] == ["0", "disabled"]


def test_image_inputs_only_build_when_needed():
    module = load_script("image_inputs", CONTAINER / "ci/build_inputs.py")
    assert not module.needs_image_build(["README.md", "src/piper_outcome_stack/training.py"])
    for path in [
        "uv.lock",
        "infra/container/profile.sh",
        "infra/container/lib/piper_container_common.py",
    ]:
        assert module.needs_image_build([path])


def test_initializer_does_not_overwrite_persistent_tools(tmp_path):
    script = (CONTAINER / "init-shared-python.sh").read_text()
    block = script.split("# Seed missing tools only;", 1)[1].split("\nchown -R", 1)[0]
    block = "# Seed missing tools only;" + block
    import grp

    env = os.environ.copy()
    for key in ("COMMAND_ROOT", "LIB_ROOT", "SOURCE_BIN", "SOURCE_LIB"):
        directory = tmp_path / key
        directory.mkdir()
        env[key] = str(directory)
    env.update(
        PIPER_ADMIN_USER=pwd.getpwuid(os.getuid()).pw_name,
        PIPER_GROUP_NAME=grp.getgrgid(os.getgid()).gr_name,
        PROFILE_SOURCE=str(tmp_path / "profile-source"),
        PROFILE_TARGET=str(tmp_path / "profile-target"),
    )
    names = [
        "piper-artifact-fetch",
        "piper-artifact-promote",
        "piper-env-doctor",
        "piper-gpu-run",
        "piper-python",
    ]
    for name in names:
        (Path(env["SOURCE_BIN"]) / name).write_text("seed")
    (Path(env["SOURCE_LIB"]) / "piper_container_common.py").write_text("seed")
    Path(env["PROFILE_SOURCE"]).write_text("seed")
    subprocess.run(["bash", "-euc", block], env=env, check=True)
    target = Path(env["COMMAND_ROOT"]) / "piper-gpu-run"
    target.write_text("updated")
    Path(env["PROFILE_TARGET"]).write_text("updated profile")
    subprocess.run(["bash", "-euc", block], env=env, check=True)
    assert target.read_text() == "updated"
    assert Path(env["PROFILE_TARGET"]).read_text() == "updated profile"


def test_signal_releases_gpu_lock_and_records_failure(runtime, tmp_path):
    _, repo, root, user = runtime
    harness = tmp_path / "runner.py"
    harness.write_text(f"""
import importlib.machinery, importlib.util
from pathlib import Path
loader = importlib.machinery.SourceFileLoader('runner', {str(CONTAINER / "bin/piper-gpu-run")!r})
spec = importlib.util.spec_from_loader('runner', loader)
m = importlib.util.module_from_spec(spec); loader.exec_module(m)
m.PIPER_ROOT = Path({str(root)!r})
m.load_runtime_config = lambda: {{'admin_user': {user!r}, 'collaborator_user': 'other'}}
m.gpu_inventory = lambda: [{{'index': 0, 'uuid': 'GPU-test'}}]
raise SystemExit(m.main())
""")
    base = [
        sys.executable,
        str(harness),
        "--gpus",
        "0",
        "--repo",
        str(repo),
        "--python",
        sys.executable,
    ]
    with (tmp_path / "runner.log").open("wb") as log:
        process = subprocess.Popen(
            [
                *base,
                "--run-id",
                "signal",
                "--",
                "python",
                "-c",
                'import time; print("ready", flush=True); time.sleep(30)',
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 15
            output = root / "runs" / user / "signal/output.log"
            while not output.exists() or "ready" not in output.read_text():
                if time.monotonic() > deadline or process.poll() is not None:
                    pytest.fail((tmp_path / "runner.log").read_text())
                time.sleep(0.05)
            conflict = subprocess.run(
                [*base, "--run-id", "conflict", "--", "python", "-c", "pass"], capture_output=True
            )
            assert conflict.returncode != 0 and b"already reserved" in conflict.stderr
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == 143
            assert json.loads((output.parent / "summary.json").read_text())["exit_code"] == 143
            free = subprocess.run(
                [*base, "--run-id", "after-signal", "--", "python", "-c", "pass"],
                capture_output=True,
            )
            assert free.returncode == 0, free.stderr
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def test_fetch_resolves_branch_and_honors_endpoint_below_old_disk_gate(tmp_path, monkeypatch):
    import types

    module = load_script("artifact_fetch", CONTAINER / "bin/piper-artifact-fetch")
    user = pwd.getpwuid(os.getuid()).pw_name
    calls = {}

    class Api:
        def __init__(self, endpoint):
            calls["endpoint"] = endpoint

        def repo_info(self, **kwargs):
            calls["request"] = kwargs
            return types.SimpleNamespace(sha="a" * 40)

    def download(**kwargs):
        calls["download"] = kwargs
        kwargs["local_dir"].mkdir()
        (kwargs["local_dir"] / "config.json").write_text("{}")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api, snapshot_download=download)
    )
    monkeypatch.setattr(module, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(module, "free_bytes", lambda _: 1024**3)
    monkeypatch.setattr(
        module, "load_runtime_config", lambda: {"admin_user": user, "collaborator_user": "other"}
    )
    monkeypatch.setenv("HF_ENDPOINT", "https://chosen.example.invalid")
    monkeypatch.setattr(
        sys,
        "argv",
        ["piper-artifact-fetch", "--repo", "owner/model", "--type", "model", "--revision", "dev"],
    )
    assert module.main() == 0
    assert calls["endpoint"] == "https://chosen.example.invalid"
    assert calls["request"]["revision"] == "dev"
    assert calls["download"]["revision"] == "a" * 40
    manifest = next(tmp_path.rglob("piper-artifact-manifest.json"))
    assert json.loads(manifest.read_text())["requested_revision"] == "dev"


def test_doctor_selected_python_and_optional_network(tmp_path, monkeypatch, capsys):
    module = load_script("env_doctor", CONTAINER / "bin/piper-env-doctor")
    user = pwd.getpwuid(os.getuid()).pw_name
    monkeypatch.setattr(module, "SHARED_PYTHON_ENV", tmp_path)
    monkeypatch.setattr(module, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(
        module, "load_runtime_config", lambda: {"admin_user": user, "collaborator_user": "other"}
    )
    monkeypatch.setattr(module, "git_state", lambda _: {"commit": "test", "dirty": True})
    monkeypatch.setattr(module, "free_bytes", lambda _: 1024**3)
    monkeypatch.setattr(module, "total_bytes", lambda _: 1024**3)
    monkeypatch.setattr(module, "gpu_inventory", lambda: [{"index": 0, "uuid": "GPU-test"}])
    seen = []
    monkeypatch.setattr(
        module, "python_environment", lambda p: (seen.append(p) or {"executable": p}, "test==1\n")
    )
    monkeypatch.setattr(
        module, "check_network", lambda _: pytest.fail("unsolicited network request")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["piper-env-doctor", "--repo", str(tmp_path), "--python", sys.executable, "--json"],
    )
    module.main()
    checks = {entry["name"]: entry for entry in json.loads(capsys.readouterr().out)["checks"]}
    assert seen == [sys.executable]
    assert checks["workspace_free"]["status"] == "warn"
    assert checks["shared_memory"]["status"] == "warn"
    assert checks["gpus"]["status"] == "pass"


def test_promotion_accepts_official_endpoint_and_preserves_existing_release(tmp_path, monkeypatch):
    import grp
    from piper_container_common import file_inventory, inventory_identity

    module = load_script("artifact_promote", CONTAINER / "bin/piper-artifact-promote")
    source = tmp_path / "staging/user/model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    files = file_inventory(source)
    manifest = source / "piper-artifact-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "model",
                "repo_id": "owner/model",
                "revision": "a" * 40,
                "source_endpoint": "https://huggingface.co",
                "files": files,
                "content_sha256": inventory_identity(files),
            }
        )
    )
    (tmp_path / "releases/models").mkdir(parents=True)
    monkeypatch.setattr(module, "PIPER_ROOT", tmp_path)
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        module, "load_runtime_config", lambda: {"group_name": grp.getgrgid(os.getgid()).gr_name}
    )
    monkeypatch.setattr(module, "set_release_ownership", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["piper-artifact-promote", "--manifest", str(manifest)])
    assert module.main() == 0
    destination = tmp_path / "releases/models" / ("owner--model@" + "a" * 40)
    original = (destination / "piper-artifact-manifest.json").read_bytes()
    assert module.main() != 0
    assert (destination / "piper-artifact-manifest.json").read_bytes() == original


def test_output_failure_reaps_child_before_releasing_gpu_lock(runtime, monkeypatch):
    import io
    import fcntl
    from types import SimpleNamespace

    module, repo, root, user = runtime

    class Child:
        pid = 987654
        stdout = io.BytesIO(b"child output\n")
        waited = False

        def poll(self):
            return None if not self.waited else 0

        def wait(self, timeout=None):
            # Another owner must not acquire the lock before cleanup finishes.
            with (root / "locks/GPU-test.lock").open("a+") as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.waited = True
            return 0

    child = Child()
    signals = []
    real = module.subprocess
    monkeypatch.setattr(
        module,
        "subprocess",
        SimpleNamespace(
            Popen=lambda *a, **kw: child,
            PIPE=real.PIPE,
            STDOUT=real.STDOUT,
            SubprocessError=real.SubprocessError,
            TimeoutExpired=real.TimeoutExpired,
        ),
    )
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class Output:
        buffer = None

        def __init__(self):
            self.buffer = self

        def write(self, value):
            if isinstance(value, bytes):
                raise BrokenPipeError("closed output")
            return len(value)

        def flush(self):
            pass

    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", Output())
        assert (
            invoke(runtime, monkeypatch, ["python", "-c", "pass"], "--run-id", "pipe-failure") == 3
        )
    assert child.waited and child.stdout.closed
    assert signals == [(child.pid, signal.SIGTERM)]
    with (root / "locks/GPU-test.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_environment_inventory_without_pip(tmp_path):
    import venv
    from piper_container_common import python_environment

    env = tmp_path / "no-pip"
    venv.EnvBuilder(with_pip=False).create(env)
    identity, packages = python_environment(str(env / "bin/python"))
    assert identity["prefix"] == str(env)
    assert not any(line.startswith("pip==") for line in packages.splitlines())
