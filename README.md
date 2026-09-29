# PiPER OutcomeStack

## Current status and entry points (2026-09-29)

| Area | State | Entry |
|---|---|---|
| Task A ACT baseline | 40k correction-mixed model; operator-reported complete pick/place at P1–P4, N1 failed; no measured success rate yet | [process record](docs/reports/ACT实机全过程记录_20260929.md) |
| Policy execution | 50 Hz, re-infer every 3 steps, TE 0.01 on decoded absolute targets, SDK speed 75% | `infra/acceptance/piper_policy_trial.py run --position P1 ...` |
| Trial outcomes | Operator labels as sidecars; per-position rates with Wilson 95% intervals | `python -m piper_outcome_stack.policy_trials label\|summarize` |
| Xbox capture | 50 Hz control/Dataset, 640×480 RGBD60, sealed raw → offline conversion | `piper record`, `piper convert SESSION` ([operator](infra/local-controller/operator/README.md)) |
| Teach capture | Receive-only raw capture, offline N−1 conversion | [teach collection](docs/operations/piper_teach_collection.md) |
| Training | Private editable env on `piper-training`, official `lerobot-train` | [training workflow](infra/container/README.md) |
| Simulation | S0/S1 only (geometry, archived replay) | [simulation](docs/operations/piper_simulation.md) |
| Protocol | Runtime binds PR-20260909-01; latest amendment PR-20260921-01 | [preregistration](docs/preregistration/) |

Sections below record the approved behavior and its dates; later sections supersede earlier
planning-only statements where they say so.

Current operator default (approved 2026-09-20): piper record uses 50 Hz control/Dataset and 640×480 RGBD at 60 fps, with time-based gripper reference and incremental A/Y planning. Save seals raw data; piper convert SESSION creates the independent Dataset offline. See [operator shortcuts](infra/local-controller/operator/README.md). Historical 20 Hz datasets and teaching next-state labels retain their own provenance.

PiPER OutcomeStack is a reproducible real-robot data, ACT/VLA training, deployment,
evaluation, simulation/sim2real, and action-outcome stack for the standard PiPER. This repository is the
code and experiment-evidence source. The control repository holds the roadmap, current
status, decisions, and canonical planning records.

## Conventional lifecycle

The project does not maintain a second experiment, dataset, checkpoint, result, or
resume framework. Use the official LeRobot lifecycle directly:

- `LeRobotDataset` v3 for offline conversion, finalization, reload, and replay; Xbox capture first seals raw episodes;
- `piper-outcome-stack record` and `piper-outcome-stack teleoperate` for Xbox workflows,
  using official loops/Dataset APIs with the required action processor and per-row
  telemetry sidecars; `piper-outcome-stack replay` provides the explicit-enable replay entry point;
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
- configured RealSense RGB/depth observations; one named `d435` is the current example, not a device/name/count restriction.

Fault codes, receive frequencies, and timestamps remain telemetry rather than policy
state. The plugin uses only the commit-pinned official `pyAgxArm` SDK. It does not use a
second robot backend, ROS control path, or runtime fallback.

The administrator may debug hardware from a personal editable checkout/environment.
A separate release build/install/activation is not required for collection, acceptance
or training (operator approval, 2026-09-17). Use validated source with recorded baseline,
actual source snapshot/diff, environment, configuration and results; formal training
still uses a fixed commit and versioned data. The default `read_only` mode connects without enabling the arm and rejects all
actions. `motion` requires explicit numerical safety configuration and checks live
firmware/driver/kinematics compatibility. No acceptance attestations or no-drop claims
are required. The safety file supplies
the only motion-speed percentage and gripper force used by the SDK. Communication,
command and feedback faults issue the SDK electronic emergency stop
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
evidence. The configured RealSense cameras reuse the official backend, with serial or unique-name selection and explicit output profiles. RGBD capture preserves raw Z16, scale and numeric metre depth; default RGB policies select only RGB plus seven state values. Camera rates can exceed the recording rate; runtime frame-age/skew checks remain.
See [acceptance instructions](docs/operations/piper_bringup.md). Xbox GUID, axes, directions, trigger endpoints,
deadzone, separate shoulder/B indices, hold tolerance/stable-time/timeout, control rate, step limits, workspace, and safety limits have no guessed
defaults and must be frozen after hardware acceptance. See [Xbox pause and input measurement](docs/operations/piper_xbox.md).

For daily development, run `uv sync --frozen --extra local-controller --group dev`
once in the personal controller checkout; ordinary source edits then need only a
debug-process restart. Run the private `.venv/bin` command through `piper-socketcan exec`
when CAN access is needed, using the absolute path. Keep Git baseline/diff, any untracked
source used, interpreter, command and configuration with debug output. Release only at
stable milestones; runtime feedback, mode, limits and action checks apply during development.

Validation follows the change: measurements need review of their results; local edits
need syntax and affected tests; connection/mode/hold/fault changes also need their
corresponding on-site validation. Reserve full CI for stage merges, formal releases
and substantial cross-module changes. Synchronizing identical validated code needs
version/import checks, not a repeated suite. The current workflow starts on PR updates
(including drafts); that does not make its unrelated simulation/container jobs a
prerequisite for local debugging. Required merge/release checks remain in force.

The arm and camera have arrived (user confirmation, 2026-09-06); actual hardware results
are tracked separately from software readiness. The read-only SDK identity probe passed;
five development-plugin read-only sessions and basic joint/gripper commissioning subsequently passed. Formal stop protection and real image/state/action timing acceptance remain open as described in the planning status. Run the doctor with the inspected
`PIPER_D435_SERIAL`; both video and USB nodes require permission verification.

## Commands and verification

```bash
piper-outcome-stack doctor --root .
piper-outcome-stack robot doctor
piper-outcome-stack xbox-input --list
piper-outcome-stack audit-dataset --root "<dataset-root>" --repo-id "<exact-repo-id>"
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

## ACT policy execution: current working baseline (2026-09-28)

The operator reported complete pick, carry, place and release on P1–P4 with the 40k
correction-mixed ACT model; N1 failed. This is a working baseline, not a measured success
rate. The full history, evidence and limits are in the
[ACT real-robot process record](docs/reports/ACT实机全过程记录_20260929.md).

- Model: `positive_act.py` wraps official ACT. The six joints and the gripper are predicted
  relative to the observation at generation time, and the physical gripper width is kept
  positive with a smooth softplus parameterization inside the model.
- Execution: `infra/acceptance/piper_policy_trial.py` with `policy_execution.py`,
  `policy_motion.py`, `policy_startup.py` and `policy_rgb_recording.py`. Each chunk is
  decoded to absolute rad/m targets at its own anchor. Temporal ensembling (0.01) runs
  on decoded targets at 50 Hz control time. The current setting re-infers every 3 steps at
  SDK speed 75%. Enter-confirmed move to A, LB start/cancel, B stop, freshness, step and
  workspace checks are unchanged. Automatic cyclic GC is deferred during motion.
- `training.py`/`training_selection.py` select policy inputs and delegate to official
  LeRobot training. The per-run training, compression and resume scripts for the current
  model are still kept with their run evidence, outside this repository.

Operators start every hardware run. The runner does not home, disable or open the gripper
on exit.

## Current route and protocol (2026-09-21)

Task A remains the current green-block pick/place ACT loop; B will add reusable objects, containers and language choices. Task C is now text-instructed drawer storage with bounded recovery from missed grasps or target relocation. Its October 4 start conditions remain; speech, changing-obstacle planning, regrasping and drawer sim2real are extensions, not v1 commitments.

S0/S1 software and GUI evidence already exist. S2/S3 real transfer and formal ACT/SmolVLA/outcome-model results are not implied. The three workstreams, weights and January 14 deadline remain unchanged.

The latest planning amendment is [PR-20260921-01](docs/preregistration/PR-20260921-01.md). Runtime configs/project.json still binds PR-20260909-01; this documentation task does not change runtime provenance or old experiments. Standard ACT has no native language input: B plans an instruction-parser/ACT-skill system comparison with robot-pretrained SmolVLA fine-tuning, with parser errors and total costs disclosed.

## Existing-stack repair and synchronization

The JSON configuration uses string annotations with explicit allowed-value checks, compatible with the pinned draccus decoder. Real CLI parsing regressions cover record and teleoperate. Motor enable sends once and waits for all six fresh driver flags instead of interpreting the SDK cached return as an acknowledgement.

The maintained operator commissioning entry is `infra/acceptance/piper_joint_commission.py`, with `piper_motion_preflight.py` for read-only controller limits. The earlier J1-only script remains in diagnostic artifacts, not as another active entry. Commissioning is distinct from full-workflow validation and does not start on import or synchronization.

## MuJoCo S0/S1 simulation

The approved first simulation implementation is available through `piper-outcome-stack sim`: independent `.venv-sim`, the pinned standard PiPER model, geometry checks, desktop pose display, and archived-feedback / commanded-servo replay. Follow [the simulation tutorial](docs/operations/piper_simulation.md).

S0/S1 uses an illustrative uncalibrated table/camera and unidentified actuator settings. Recorded feedback excursions are preserved and reported; target validation remains strict. It does not import the real robot plugin or connect to CAN/camera/gamepad, train a policy, implement S2/S3, or deploy a formal release. Earlier planning-only statements above describe the previous approval stage; this S0/S1 scope has now been explicitly approved.

See [RGBD capture and policy input selection](docs/operations/camera_rgbd.md).

### Explicit servo lifecycle

`connect()` only connects, validates and observes in both execution modes. It does
not configure CAN/J, enable or move. `enable()` on a connected motion session
configures speed/J, confirms fresh six-axis enable feedback and prepares control;
an already-enabled arm is not sent another enable command. The project teleoperate
and record workflows call these two operations explicitly in sequence. Direct
plugin callers must also call `enable()` before sending actions.

`get_servo_status()` reports six joint bits, a separate gripper bit, driver faults
and host receipt times. Its joint summary is ENABLED/DISABLED/PARTIAL/UNKNOWN.
`CONNECTED` is a connection label, never an assertion of motor torque. After
disconnection or stale feedback the current status is UNKNOWN;
`last_servo_feedback` retains the last complete reading with its timestamps.

`disable(include_gripper=False)` explicitly disables joints and retains gripper
torque; `disable(include_gripper=True)` additionally disables the gripper. The
argument is mandatory. Both confirm new feedback after each sent command, stop
accepting actions before torque removal, and never retry or reset. Failure latches
the session without an extra automatic stop command. Ordinary pause continues to
hold with torque, and `disconnect()` continues to release resources only. Neither
operation implicitly homes or calibrates the arm. No brake/no-drop guarantee is
implied by a confirmed disable.

## New-scene collection

Use [scene preparation and recording](docs/operations/piper_scene_collection.md) for A/work-pose, Y/zero, nonblocking terminal episode decisions, receive-time summaries and scene-separated RGBD trials. Hardware basics are not repeated as startup gates; 30-minute/20-safe-episode observations are collected during trials. Real-time command/feedback checks and Dataset integrity remain mandatory. Relocation templates intentionally leave site-specific values unset; no old workspace or extrinsics are applied automatically.

### 拖动示教采集

Xbox保留；新增`teach-record`只接收原始RGBD/关节/夹爪反馈，`teach-convert`离线生成下一帧目标Dataset，`audit-dataset`按来源审计。示教入口不响应Xbox B、不发送停止或模式指令；详见[示教操作说明](docs/operations/piper_teach_collection.md)。示教候选动作尚未代表实机回放或ACT验证通过。

## Training development

On `piper-training`, install needed tools directly and use a private editable environment. Code edits run after restarting the process; PRs, releases and image builds are not development prerequisites. Formal training records a fixed commit, actual environment and versioned data without requiring a merged PR or project release. See [the training workflow](infra/container/README.md) for setup, GPU commands and collaborator permissions.
