"""Determine whether this change affects the optional base image."""

from __future__ import annotations

import subprocess
import sys

IMAGE_FILES = {
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".dockerignore",
    "packages/lerobot_robot_outcome_piper/pyproject.toml",
    "infra/container/Dockerfile",
    "infra/container/entrypoint.sh",
    "infra/container/init-shared-python.sh",
    "infra/container/profile.sh",
    "infra/container/sshd_config",
    ".github/workflows/remote-training-container.yml",
}


def needs_image_build(paths: list[str]) -> bool:
    return any(
        path in IMAGE_FILES
        or path.startswith(
            (
                "infra/container/bin/",
                "infra/container/lib/",
                "infra/container/ci/",
            )
        )
        for path in paths
    )


if __name__ == "__main__":
    changed = subprocess.run(
        ["git", "diff", "--name-only", "-z", sys.argv[1], sys.argv[2]],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    print("true" if needs_image_build(changed) else "false")
