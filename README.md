# PiPER OutcomeStack

PiPER OutcomeStack is a reproducible real-robot data, ACT/VLA training, deployment,
evaluation, and action-outcome stack for the standard PiPER. This repository is the
code and experiment-evidence source. The control repository holds the roadmap, current
status, decisions, and canonical planning records.

## Conventional lifecycle

The project does not maintain a second experiment, dataset, checkpoint, result, or
resume framework. Use the official LeRobot lifecycle directly:

- `LeRobotDataset` v3 for recording, finalization, reload, and replay;
- `piper-outcome-stack record` and `piper-outcome-stack teleoperate` for Xbox workflows,
  delegating lifecycle management to official LeRobot while injecting the required action
  processor; `lerobot-replay` remains the direct replay entry point;
- `lerobot-train` for ACT/SmolVLA training and checkpoint resume;
- Hugging Face revisions plus the promoted cross-host artifact boundary for published
  datasets and models.

The experiment registry and Git history stay inspectable. SHA-256 remains at
dependency/image locks, cross-host artifacts, safety files, preregistration, and final
dataset/model publication boundaries.

## PiPER LeRobot plugin

The single workspace distribution `lerobot_robot_outcome_piper` is auto-discovered by
LeRobot. It registers robot type `outcome_piper`, teleoperator type
`outcome_piper_xbox`, and exposes one observation/action schema:

- `joint_1.pos` through `joint_6.pos` in radians;
- `gripper.pos` in metres, representing the official gripper's total opening width;
- configured LeRobot cameras as additional observations.

Fault codes, receive frequencies, and timestamps remain telemetry rather than policy
state. The plugin uses only the commit-pinned official `pyAgxArm` SDK. It does not use a
second robot backend, ROS control path, or runtime fallback.

The unique highest-privilege administrator runs the plugin directly from an immutable
release. The default `read_only` mode connects without enabling the arm and rejects all
actions. `motion` additionally requires matching frozen safety and hardware-acceptance
files bound to the exact live firmware identity. The frozen safety file also supplies
the only motion-speed percentage and gripper force used by the SDK. Communication,
watchdog, command, device-disconnect, and hold-to-run release faults issue the
hardware-validated electronic emergency stop and latch the session; a new motion
session is required after operator intervention. Disconnect does not home, reset, or
disable the arm.

There is no runtime account, Unix socket, operator permit, resident control service, or
mock control path. Collaborators cannot enter the target real-CAN namespace, bind its
interface, open the target camera/gamepad nodes, use sudo, or modify an immutable
release; host-namespace `vcan` remains available for software tests.

OutcomeStack continues to own camera and controller selection, data collection,
training, evaluation, host permissions, immutable releases, and real validation
evidence. D435 uses LeRobot's RealSense implementation; AR0234 uses OpenCV only after
the complete module enumerates as UVC. Xbox GUID, axes, directions, trigger endpoints,
deadzone, control rate, step limits, workspace, and safety limits have no guessed
defaults and must be frozen after hardware acceptance.

## Commands and verification

```bash
piper-outcome-stack doctor --root .
piper-outcome-stack robot doctor
piper-outcome-stack teleoperate --robot.type=outcome_piper --teleop.type=outcome_piper_xbox ...
piper-outcome-stack record --robot.type=outcome_piper --teleop.type=outcome_piper_xbox \
  --dataset.push_to_hub=false ...
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The top-level doctor accepts either a Git checkout on any branch or a completed
Git-archive release containing `.piper-release-complete`. Dependency identities are
established by `uv.lock` and installed distribution metadata. See `THIRD_PARTY.md` for
the fixed upstream sources and license notices.

The fixed real-CAN namespace has no veth/NAT. `record` therefore requires
`--dataset.push_to_hub=false`; upload the finalized dataset later from the host namespace.

The supported remote training environment is under `infra/container/`; the local
controller deployment is under `infra/local-controller/`. Raw data, videos,
checkpoints, and model weights must not enter Git.
