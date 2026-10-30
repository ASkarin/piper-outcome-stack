"""Select base-image inputs and identify reusable successful build checks."""

from __future__ import annotations

import argparse
import hashlib
import subprocess

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
IMAGE_DIRECTORIES = (
    "packages/piper_policy/",
    "infra/container/bin/",
    "infra/container/lib/",
    "infra/container/ci/",
)


def needs_image_build(paths: list[str]) -> bool:
    return any(path in IMAGE_FILES or path.startswith(IMAGE_DIRECTORIES) for path in paths)


def image_input_key(revision: str = "HEAD") -> str:
    # Tree entries include paths, modes and blob IDs, so additions/deletions and
    # executable-bit changes invalidate the check as well as content changes.
    # Use the actual checkout (the PR merge tree), not just the PR head.
    entries = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", revision, "--", *sorted(IMAGE_FILES), *IMAGE_DIRECTORIES]
    )
    if not entries:
        raise ValueError("no container build inputs found")
    return hashlib.sha256(entries).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    changed_parser = commands.add_parser("changed")
    changed_parser.add_argument("base")
    changed_parser.add_argument("head")
    key_parser = commands.add_parser("key")
    key_parser.add_argument("revision", nargs="?", default="HEAD")
    args = parser.parse_args()
    if args.command == "key":
        print(image_input_key(args.revision))
    else:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "-z", args.base, args.head], text=True
        ).split("\0")
        print("true" if needs_image_build(changed) else "false")
