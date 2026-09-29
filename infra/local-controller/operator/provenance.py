"""Capture editable source/environment before opening any hardware connection."""

import json
from pathlib import Path
import subprocess
import zipfile


def capture(profile, run):
    source = Path(profile["source"]).resolve()
    run = Path(run)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    (run / "source.diff").write_bytes(git("diff", "--binary", "HEAD"))
    (run / "source-status.txt").write_bytes(git("status", "--short"))
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    with zipfile.ZipFile(run / "source-tree.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(paths)):
            path = source / name
            if (
                name
                and path.is_file()
                and (
                    name.startswith(("src/", "packages/", "infra/", "configs/"))
                    or name in ("pyproject.toml", "uv.lock")
                )
                and not any(part in ("__pycache__", ".venv") for part in path.parts)
            ):
                archive.write(path, name)
    probe = """import importlib.metadata as m, json, platform, sys
print(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
packages=sorted([dict(name=d.metadata['Name'],version=d.version,direct_url=d.read_text('direct_url.json'))
for d in m.distributions()],key=lambda d:d['name'].lower()))))"""
    env = json.loads(subprocess.check_output([profile["python"], "-c", probe], text=True))
    env["source"] = str(source)
    env["git_head"] = git("rev-parse", "HEAD").decode().strip()
    (run / "environment.json").write_text(json.dumps(env, indent=2))
