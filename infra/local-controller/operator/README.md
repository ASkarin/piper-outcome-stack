# Stable development operator entry

`~/.local/bin/piper` runs `operator.py` using the controller's existing private
Python. Active configuration lives in `~/.config/piper-outcome-stack/operator/`.
This is a development entry, not a formal release or a hardware stability claim.
No service, CAN setup, new backend or sudo permission is installed.

| Command | Behavior |
|---|---|
| `piper act SKILL 1` | Run the configured fixed ACT bundle; `--verify-only` checks assets without device I/O. |
| `piper record` | New Xbox raw capture; existing LB/X/RB/A/Y/B controls. |
| `piper record b --split train` | B training capture: choose layout, then choose a task for each episode. |
| `piper record b --split development` | B development capture in a separate session: choose layout, then task. |
| `piper status` | Fixed passive status query. |
| `piper link`, `firmware`, `limits`, `acceleration` | Existing fixed query operations. |
| `piper recover` | One confirmed electronic-stop recovery; no enable or position command. |
| `piper convert SESSION` | Convert closed Xbox raw data; exclude protection/marked waits and keep continuous segments. |
| `piper audit SESSION` | Audit that session's converted Dataset. |
| `piper pauses SESSION` | Generate pause candidates only; never delete frames. |
| `piper replay CONFIG.json` | Explicit continuous replay config, unique result path; no automatic approach. |

SESSION is the printed operator run directory, not the raw image directory.
Recovery and replay do not use Xbox buttons. Recovery retains its Enter confirmation.
Replay retains `start`, `stop` and Ctrl+C semantics; it requires the selected
trajectory's actual start pose. These commands must be operator-executed on site.

`record` checks the configured SSD UUID and creates new raw/output paths. Logs and
configuration snapshots are under `~/piper-runs/operator/`. Each new process reads
the active configuration; existing run snapshots and datasets are never updated.
Scene relocation requires updating the active scene/workspace configuration.

In preparation, wait for confirmed hold before `start P1`. The first start
acquires the current legal gripper width and configured holding force; no trigger
tap is required. Use `end`, `save success/failure/cancelled`, then
start the next position or `quit`. Save seals raw files; conversion remains a
separate offline operation. Protection/explicitly marked waits are excluded at
conversion; ordinary holding remains. No automatic replay is performed.

For an already open shell after installation:

```bash
export PATH="$HOME/.local/bin:$PATH"
piper --help
```

Shell routing tests use fake subprocesses and never access hardware.

## Default capture, approved 2026-09-20

Use `piper record` without a trial profile: control and Dataset are 50 Hz, the
D435 streams 640×480 RGBD at 60 fps. The default remains 20 attempts (five per
P1–P4), up to 180 seconds per attempt, with failures retained separately. Each
invocation creates an independent session; it does not resume or mix old 20 Hz data.
New default raw/Dataset sessions use the SSD `piper-data/raw/xbox-50hz` and
`piper-data/datasets/xbox-50hz` directories. Existing datasets and run snapshots
keep their original rates and semantics.

The gripper reference is time-based, capped at 120 mm/s, with 1.2 m/s² ramp and
8 mm lead. A/Y use incremental full-route validation while holding. Per-cycle
XYZ/rotation input is 2.5 mm/0.5 degrees. Approved teleoperation defaults
(2026-09-21): SDK ceiling 75%, ramp 0.5 s, lead_cycles 2.4 at 0.02 s:
125 mm/s per coordinate and 25 degrees/s, with 6 mm per-coordinate and
1.2-degree combined rotation lead. A/Y retain their existing timing and targets;
independent replay configurations are unchanged.
This change does not automatically start capture, replay, conversion or training.

## Fine control and demonstration pauses, 2026-10-10

In the `piper record` or `piper record b` operator terminal, use `fine` or
`normal` in preparation, before selecting the task. Release LB and the mode/pose buttons,
center sticks/triggers, and wait for confirmed hold before switching. No command
is queued if these conditions are not met. The next movement still needs a fresh
LB press. Fine scales Cartesian and orientation input, ramp and lead to 50%;
with the current 50Hz defaults this is 1.25mm/tick (62.5mm/s per coordinate),
0.25degree/tick and 3mm lead per coordinate. The deadzone is applied to the raw
joystick axes before this gain, so selecting fine does not change its threshold.
Gripper generation and A/Y preset
trajectories retain their existing parameters. Normal is the initial setting;
the selected gain persists across episodes and is saved in action telemetry.

During recording, data eligibility follows the existing LB permission without
keyboard interaction. Releasing LB invokes the existing hold and automatically
pauses demonstration data immediately; raw RGB/depth/state/action recording
continues. Center inputs and confirm hold, then press LB again to resume through
the ordinary permission path. Keep LB held and center the sticks when a task
requires genuine holding, gripper-only operation, settling or release waiting;
these frames remain eligible. Initial menu/typing delay and delay before `end`
are excluded while LB permission is not active. A protective input rejection
still excludes frames until a new permitted SDK command succeeds.

`fine`/`normal` are selected before recording and persist across episodes. There
is no `pause`/`resume` keyboard command during recording and no gain switch to
type mid-task. Releasing LB still requires confirmed hold, neutral inputs and a
fresh LB press before movement. B-stop behavior is unchanged.

With LB continuously held, brief stick centering captures one current joint hold
target and clears the arm's velocity reference. New legal stick input can resume
on the next control cycle, before the centering hold has completed its stability
window, only when it continues a running epoch already admitted by the robot.
The new reference starts from fresh feedback and zero velocity; stale arm and
gripper intents remain invalid. Completed holds that drift, protection holds,
mode/pose changes, phase changes, input loss and faults retain the full existing
hold/rearming path. Hold tolerances, stability time, deadzone, motion limits and
gripper generation are unchanged. Admitted continuation is recorded as
`center_resume=true` in the existing action telemetry; inspect `result` and SDK
command results to distinguish actual dispatch from a subsequent rejection/fault.

The terminal reports changes in phase, gain, demonstration eligibility, protection
reason and recovery step through the existing asynchronous output. Unchanged
state, live clearance and protection duration do not trigger repeated output;
ordinary running/centering transitions retain their existing control prompts.
Each status message includes the current calculated minimum tool/table clearance.
The clearance uses
the configured table/tool geometry; it is not object collision detection. The
5mm table lower bound, numerical motion limits, B stop and hold verification
are preserved. A protection wait ends after a new permitted joint or gripper
SDK command succeeds, including a gripper command while joints remain held.

`piper convert SESSION` creates one LeRobot episode per contiguous eligible
segment. It never edits raw frames or overwrites an existing Dataset. The
`demonstration-segments.json` map records original session/episode/attempt and
frame ranges, including excluded waits. Each segment is an episode boundary,
so ACT/SmolVLA future targets use the ordinary episode padding mask and cannot
cross a removed pause. Segments inherit the original outcome for selection;
they are not independently successful complete tasks. `prepare-training`
keeps all segments of one original episode together in the selected split.
Existing dataset/model/run files remain historical evidence. Reconversion of
an old capture needs an explicitly different output path; existing task label
corrections must be retained when preparing a new training version.

These changes have offline regressions. Fine motion, displayed clearance and
pause/recovery interaction still require operator-executed site verification;
this documentation does not report hardware acceptance.

## RB selection while confirming hold

After releasing LB and centering all inputs, RB may select the next mode before
hold confirmation finishes. Wait for the mode-ready prompt before pressing LB.
An early LB press, stick/trigger input, conflict, second RB press or phase/fault
change cancels the pending choice. No new SDK position command or hold-window
restart is caused by selection itself. X keeps the original switching conditions.


## B pilot task menu (2026-09-29)

Run `piper record b` on the controller. It uses `record_b_config` from the operator
profile, keeping the A config intact. The deployed B config has four tasks and
four saved episodes for the initial development pilot, with position `B-layout1`.
Each task choice starts a recording only after the existing hold/neutral checks:

| Input in preparation | Task (fixed camera view) |
|---|---|
| `1` | Green block → left region |
| `2` | Green block → right region |
| `3` | Sky-blue hexagonal block → left region |
| `4` | Sky-blue hexagonal block → right region |

In recording, Enter or `end` ends the episode. In review, `1` saves success,
`2 [note]` saves failure, `3 [note]` saves cancellation, `r` discards for redo,
and `q` discards the current unsaved episode and exits. Blank review input does
nothing. Full `save` commands remain available. These review shortcuts and Enter
to end also work in the A recorder. LB/B and A/Y behavior are unchanged.

Reset **both** blocks to the same two separated pickup locations before each of
the first four combinations. Keep both visible and both destination regions clear;
record a reference layout image. Do not put the selected block alone in the scene.
Left/right mean the fixed camera view, not the robot's base axes. Left is slightly
narrower; the operator confirmed both blocks fit both halves. Keep the outer border
and divider; cover old point labels only if confusing, preserving the old layout
reference. No automatic camera coordinate or success-detector recalibration is implied.

Task text is chosen per episode and supplied to the official record loop; raw
frame task strings survive offline Dataset conversion. Events also record task ID,
text and layout ID. Session config and the existing source snapshot record the menu.
All four pilot combinations in this session are development data; do not split
this session between train and evaluation. Future swapped layouts need a distinct
recording_position/config snapshot. This is human teleoperation, not an ACT rollout.

The launcher checks the configured SSD UUID and performs a temporary write/flush
before sudo or device connection. An unmounted or read-only SSD stops startup;
it does not write capture data to the system disk or mount/remount automatically.
B raw/Dataset paths are under their existing parent directories' `task-b/` child.


## Layout selection and per-task progress (2026-09-30)

These are the standing B capture commands (controller configuration checked
2026-10-16). Supplying `--split train` or `--split development` makes layout the
first menu; task selection follows in the existing recorder. Use the same commands
for targeted recollection: select only the required tasks and quit after the
planned attempts. There is no requirement to collect all four tasks or fill the
session limit. For two development examples per selected task, two separate
development sessions can each collect one example per task.

`piper record b` now asks for intended split and layout **before** connecting devices.
Optional direct commands: `piper record b --split train --layout 1` (or layout 2),
and `piper record b --split development --layout 1` (or 2).
Layout 1 has green at the upper pickup location and blue hexagon at the lower one
in the raw camera view; layout 2 swaps objects between those same locations.
Both objects remain visible. Target left/right always refer to the raw camera view.

One session uses one split/layout. Train candidates plan five saved episodes per
four-way task, 20 per session; development plans one per task, four per session.
Each task displays this-session saved, success, failure and cancelled counts.
Failures count toward saved attempts but not successful demonstrations. Counts
reset with each new session; old pilot episodes are not included. The operator
balances task choices; the menu does not invent cross-session progress or forbid
additional selections. Quitting early retains saved episodes.

The CLI config and raw capture context include layout description, intended split
and planned per-task count. Episode events retain the layout and intended split.
A train marker does not automatically promote failed or unreviewed frames to
training. ACT and SmolVLA share the same source episodes and split grouping.

## Stage2 interface freeze (2026-10-17)

The explicit `act_bundle` profile field selects the immutable controller bundle. See `docs/operations/stage2_interface_contract_20261017.md`. No automatic success labels or motion are created by verification.

## 2026-10-30：固化SmolVLA与C软件准备

收尾的两组050000模型及原RTC源码继续使用原目录，profile增加`smolvla_bundle`。常用入口与ACT一样委托冻结资产：

```bash
piper smolvla original green-left 1 --verify-only
piper smolvla flat blue-right 2 --verify-only
piper smolvla original green-left 1
piper smolvla flat blue-right 2
```

verify-only核对freeze.json中的资产身份，不加载模型或连接设备；实际run仍调用原run-policy.py，保持两模型、四模板、两布局及原操作者启动行为。原56次记录不改写。无默认模型替换，也不把这两组B模型称为已训练的C模型。

C新增`piper record c --skill ... --split ...`与`piper c check|run --config ... --skill ...`，完整参数、数据与验证范围见[任务C说明](../../../docs/operations/piper_task_c.md)。record_c_config只指向C专用配置；待实物配置完成前，模板缺项会明确拒绝。原A/B入口保持。
