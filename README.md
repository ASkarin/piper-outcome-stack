# PiPER OutcomeStack

PiPER OutcomeStack is a reproducible real-robot data, ACT/VLA training, deployment,
evaluation, simulation/sim2real, and action-outcome stack for the standard PiPER. This repository is the
code and experiment-evidence source. The control repository holds the roadmap, current
status, decisions, and canonical planning records.

## Conventional lifecycle

The project does not maintain a second experiment, dataset, checkpoint, result, or
resume framework. Use the official LeRobot lifecycle directly:

- `LeRobotDataset` v3 for recording, finalization, reload, and replay;
- `piper-outcome-stack record` and `piper-outcome-stack teleoperate` for Xbox workflows,
  using official loops/Dataset APIs with the required action processor and per-row
  telemetry sidecars; `lerobot-replay` remains the direct replay entry point;
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
- one D435 RGB observation named `d435` (`observation.images.d435` in the Dataset).

Fault codes, receive frequencies, and timestamps remain telemetry rather than policy
state. The plugin uses only the commit-pinned official `pyAgxArm` SDK. It does not use a
second robot backend, ROS control path, or runtime fallback.

The administrator may debug hardware from a personal editable checkout/environment.
Formal acceptance, formal collection and reproducible experiments use an immutable
release. The default `read_only` mode connects without enabling the arm and rejects all
actions. `motion` additionally requires matching frozen safety and hardware-acceptance
files bound to the exact live firmware identity. The frozen safety file also supplies
the only motion-speed percentage and gripper force used by the SDK. Communication,
command and feedback faults issue the hardware-validated electronic emergency stop
and latch the session. Xbox shoulder release requests a fixed-position hold and
allows deliberate rearming after confirmation; B independently requests a latched
electronic stop. Selected gamepad loss or a stalled control loop attempts a hold
before latching a fault, provided feedback and CAN remain healthy. Failed holds
use electronic stop; this is not a promise of support after CAN/process loss. Disconnect does not home, reset, or
disable the arm.

Motion waits for fresh CAN/J mode confirmation before enable and checks that mode on
subsequent feedback. Firmware must also match the pinned PiPER kinematic model
(S-V1.6-3 or later). See the [official-source comparison](docs/operations/piper_integration_sources.md)
for how the printed manual's SDK/ROS examples apply to this project.

There is no runtime account, Unix socket, operator permit, resident control service, or
mock control path. Collaborators cannot enter the target real-CAN namespace, bind its
interface, open the target camera/gamepad nodes, use sudo, or modify an immutable
release; host-namespace `vcan` remains available for software tests.

OutcomeStack continues to own camera and controller selection, data collection,
training, evaluation, host permissions, immutable releases, and real validation
evidence. The single D435 uses LeRobot's RealSense implementation, its inspected numeric
serial, and explicit RGB resolution/fps. Depth is inspected and calibrated separately;
the first ACT/SmolVLA/outcome-model input uses RGB and the seven state values. Robot-only
bring-up may omit the camera; recording requires it, matching camera/Dataset/Xbox fps
and measured `robot.capture_timing` values.
See [acceptance instructions](docs/operations/piper_bringup.md). Xbox GUID, axes, directions, trigger endpoints,
deadzone, separate shoulder/B indices, hold tolerance/stable-time/timeout, control rate, step limits, workspace, and safety limits have no guessed
defaults and must be frozen after hardware acceptance. See [Xbox pause and input measurement](docs/operations/piper_xbox.md).

For daily development, run `uv sync --frozen --extra local-controller --group dev`
once in the personal controller checkout; ordinary source edits then need only a
debug-process restart. Run the private `.venv/bin` command through `piper-socketcan exec`
when CAN access is needed, using the absolute path. Keep Git baseline/diff, any untracked
source used, interpreter, command and configuration with debug output. Release only at
stable milestones; the same hardware and motion gates apply during development.

The arm and camera have arrived (user confirmation, 2026-09-06); actual hardware gates
are tracked separately from software readiness. The read-only SDK identity probe passed;
five development-plugin read-only sessions and basic joint/gripper commissioning subsequently passed. Formal stop protection and real image/state/action timing acceptance remain open as described in the planning status. Run the doctor with the inspected
`PIPER_D435_SERIAL`; both video and USB nodes require permission verification.

## Commands and verification

```bash
piper-outcome-stack doctor --root .
piper-outcome-stack robot doctor
piper-outcome-stack xbox-input --list
piper-outcome-stack audit-dataset --root <dataset-root> --repo-id <exact-repo-id>
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

## Simulation / sim2real planning (2026-09-08)

Simulation is now a core workstream with equal priority to ACT: model alignment, motion reproduction, a simulated reaching policy tested on the real arm, then task-A ACT comparisons (real-only, sim-only, sim-pretrained plus the same real subset). MuJoCo is the planned starting point, using the pinned official PiPER geometry. No simulator version, code, assets or environment has been installed/implemented by this documentation change.

The user explicitly deferred implementation approval. Proposed `src/piper_outcome_stack/sim/`, `sim2real/`, `assets/piper/` and simulation/experiment configuration directories are designs, not existing runtime paths. ROS is not added to the control path; public task-A/B observations and actions retain one RGB image and seven rad/m values. Synthetic provenance cannot masquerade as physical SDK telemetry.

See [planning amendment](docs/preregistration/PR-20260908-01.md). The canonical workstream and schedule live in the planning repository at `docs/roadmap/piper_sim2real_workstream.md`. The repaired source now binds `configs/project.json` to PR-20260908-01. Original preregistration snapshots and old releases retain their historical identities; the source change does not implement simulation. This document is not evidence of simulation or real-policy completion.

## Existing-stack repair and synchronization

The JSON configuration uses string annotations with explicit allowed-value checks, compatible with the pinned draccus decoder. Real CLI parsing regressions cover record and teleoperate. Motor enable sends once and waits for all six fresh driver flags instead of interpreting the SDK cached return as an acknowledgement.

The maintained operator commissioning entry is `infra/acceptance/piper_joint_commission.py`, with `piper_motion_preflight.py` for read-only controller limits. The earlier J1-only script remains in diagnostic artifacts, not as another active entry. Commissioning is separate from the formal Robot gate and does not start on import or synchronize.

## MuJoCo S0/S1 simulation

The approved first simulation implementation is available through `piper-outcome-stack sim`: independent `.venv-sim`, the pinned standard PiPER model, geometry checks, desktop pose display, and archived-feedback / commanded-servo replay. Follow [the simulation tutorial](docs/operations/piper_simulation.md).

S0/S1 uses an illustrative uncalibrated table/camera and unidentified actuator settings. Recorded feedback excursions are preserved and reported; target validation remains strict. It does not import the real robot plugin or connect to CAN/camera/gamepad, train a policy, implement S2/S3, or deploy a formal release. Earlier planning-only statements above describe the previous approval stage; this S0/S1 scope has now been explicitly approved.
