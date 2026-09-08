#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ "${EUID}" -ne 0 ]] || { echo 'Use your ordinary account, not sudo.' >&2; exit 1; }
export UV_PROJECT_ENVIRONMENT="${root}/.venv-sim"
uv sync --project "${root}" --package piper-outcome-stack --extra simulation \
    --group dev --python 3.12.13 --frozen
"${root}/.venv-sim/bin/python" - <<'PIPER_SIM_PACKAGES'
import importlib.metadata as m
import platform
assert platform.python_version() == '3.12.13'
assert m.version('mujoco') == '3.12.0'
for name in ('pyAgxArm', 'lerobot', 'lerobot-robot-outcome-piper', 'pyrealsense2', 'python-can'):
    try:
        m.distribution(name)
    except m.PackageNotFoundError:
        continue
    raise SystemExit(f'Unexpected hardware/training package in simulation environment: {name}')
print('Independent simulation environment ready; no hardware packages installed.')
PIPER_SIM_PACKAGES
