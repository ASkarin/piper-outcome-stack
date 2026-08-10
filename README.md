# A3 OutcomeStack

A3 OutcomeStack is a reproducible real-robot data, ACT/VLA training, deployment,
evaluation, and action-outcome stack for EduLite A3. This repository is the code and
experiment-evidence source. The control repository holds the roadmap, current status,
decisions, and the canonical preregistration.

## Conventional lifecycle

The project does not maintain a second experiment, dataset, checkpoint, result, or
resume framework. Use the official LeRobot lifecycle directly:

- `LeRobotDataset` v3 for recording, finalization, reload, and replay;
- `lerobot-record`, `lerobot-teleoperate`, and `lerobot-replay` for robot workflows;
- `lerobot-train` for ACT/SmolVLA training and checkpoint resume;
- Hugging Face revisions plus the promoted cross-host artifact boundary for published
  datasets and models.

Historical `experiments/registry.csv`, evidence, preregistration, and Git history stay
inspectable. SHA-256 remains at dependency/image locks, cross-host artifacts,
calibration and safety files, preregistration, and final dataset/model publication
boundaries. Ordinary JSON, frames, permits, checkpoint pointers, and nested result
documents are not self-hashed.

## A3 LeRobot adapter

The actuator-facing adapter is no longer duplicated in this repository. The
`local-controller` extra pins the private, community-maintained third-party repository
`ASkarin/lerobot-robot-edulite-a3` at a full commit. Its distribution remains
`lerobot_robot_a3`, which LeRobot auto-discovers by the standard package prefix. It
registers `A3RobotConfig` as robot type `a3` and exposes:

- observation: `L1.pos` through `L7.pos`, `L1.vel` through `L7.vel`, plus configured
  LeRobot cameras;
- action: `L1.pos` through `L7.pos`;
- direct in-process calls to the commit-pinned official `ELA3Interface`;
- finite-value, joint-limit, fault, timing, and watchdog checks inside the adapter.

The unique highest-privilege administrator runs the plugin directly from an immutable
release. The default `read_only` mode connects without enabling the arm. Calibration
must already be frozen and is never performed interactively. `motion` is an explicit
mode and additionally requires frozen safety settings, a verified physical emergency
stop, all five hardware-gate flags, and matching calibration/safety SHA-256 bindings.
There is no runtime account, Unix socket,
operator permit, or mock control service. Collaborators remain unable to use raw
devices, sudo, or modify an immutable release.

OutcomeStack continues to own camera and controller selection, data collection,
training, evaluation, host permissions, immutable releases, and real validation
evidence. D435 uses LeRobot's RealSense implementation; AR0234 uses OpenCV only after
the complete module enumerates as UVC. Xbox mapping remains deferred until its real
axes are measured and then belongs in a separate LeRobot Teleoperator/processor.

The adapter repository remains private until the separately documented physical gate
passes and the owner explicitly authorizes public visibility. It is installed from a
full Git commit; no PyPI, Hugging Face package, submodule, committed wheel, or second
active source copy is used. Draft calibration, safety, and hardware-gate templates have
one source in that repository's `config/` directory. OutcomeStack does not carry a
second template set; approved real values stay outside Git and are passed explicitly
to `A3RobotConfig`.

## Commands and verification

```bash
a3-outcome-stack doctor --root .
a3-outcome-stack robot doctor
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The top-level doctor accepts either a Git checkout on any branch or a completed
Git-archive release containing `.a3-release-complete`. SDK identity is established by
the Git commit in `uv.lock` and installed distribution metadata, not a second list of
upstream file hashes.

The supported remote training environment is under `infra/container/`; the local
controller deployment is under `infra/local-controller/`. Raw data, videos,
checkpoints, and model weights must not enter Git.
