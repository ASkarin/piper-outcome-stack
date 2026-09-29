# Xbox dual translation and staged recording

Default 2026-09-20: use piper record, 50 Hz control/Dataset and 640×480 RGBD60. Current target-to-feedback joint limit is 10 degrees; SDK speed setting is 50%. Gripper reference is capped at 120 mm/s with 8 mm lead, within 0–120 mm command range and 15 mm execution step. Physical 120 mm travel is not yet calibrated. Exact operator commands: [shortcuts](../../infra/local-controller/operator/README.md).

Default translation controls the grasp center (flange +Z 105 mm). It first holds
J4-J6 and solves J1-J3; if necessary a position-only solve allows wrist motion
with a 100:1 change penalty. No orientation constraint is imposed in this mode.
Right-stick horizontal is unused but still participates in neutral detection.
X toggles fixed-orientation translation, including right-stick yaw. RB still
toggles translation/orientation. Orientation mode rotates around the grasp center
using fixed base axes. The 135 mm fingertip model remains a workspace check.

X requires released LB, neutral sticks/triggers and confirmed hold. Startup-held
X must be released first. Conflicting X/RB/A/Y requests are rejected; B preempts.
Mode changes invalidate pending motion without dispatching a new hold command.
The measured `translation_switch_button` is mandatory and distinct from other buttons.
Measure with `xbox-input --translation-only --index <device> --seconds 6 --output <new-dir>`;
this is input-only, without Robot, CAN or camera I/O. Do not guess the X index.

## Pause review

`xbox-pauses --root <raw-or-dataset> --output <new-report.json>` detects confirmed
hold intervals of at least 1 s with unchanged command references, measured joint
range at most 0.1 degrees and gripper range at most 0.5 mm. These are review
heuristics, not safety limits. Conversion also emits `pause-candidates.json`.
No frames are deleted: review grasp/placement settling and object visibility
before selecting any interval. Removing a wait is not trajectory smoothing.

## Raw capture and offline Dataset conversion

Interactive `record` requires `raw_root`. Save seals raw RGB/depth, state, actual
seven-value dispatched actions and timing evidence. After seal, `start P2` is
available directly; A/Y preparation remains optional. Raw save does not encode
video, compress depth or reopen a Dataset. Feedback, hold and B remain serviced.

After normal exit, run the printed command separately:

```bash
piper-outcome-stack xbox-convert --raw-root "<raw>" --output "<new-dataset>" --repo-id local/"<name>"
```

Conversion preserves originals and uses official Dataset save/finalize/reload,
lossless compression and source audit. Xbox N frames produce N actual-command
rows; teaching keeps its separate N-1 next-measured-state rule. Interrupted raw
sessions are not complete. Resume requires a normally closed session with the
same capture context. Existing datasets and historical run configs are untouched.

RB response update (2026-09-21): release LB and center inputs, then click RB immediately; hold confirmation need not have completed. A pending choice is displayed, and the effective mode changes only after hold confirmation plus a fresh neutral input sample. Keep LB released until the mode-ready prompt. Early LB/stick input, conflicting X/A/Y, another RB edge, phase changes or faults cancel the pending choice. The 0.3 s hold requirement is unchanged, and X retains its original confirmed-hold rule. No new motion starts without a new neutral LB edge.

The sections below describe the existing hold and input measurement behavior.

The shoulder button controls recoverable teleoperation. B requests the independent
latched electronic emergency stop. This software change does not constitute real-arm
hold, loaded-gripper, disconnect or recovery acceptance.

## Pose movement references

A (work pose), Y (home), and the commissioning move-to-start script share
`JointPoseSequence`. Xbox validates the full route incrementally during confirmed hold, at most 16 points or about 3 ms per tick; no pose command is sent before completion. Fresh start feedback is checked again before dispatch. Releasing LB cancels a pending start without queuing a restart. References follow a dense quintic progression with gentler
start/end increments. Sampling density follows the configured joint-step budget and hold tolerance;
timed planning additionally enforces the configured reference velocity/acceleration.
The next reference is admitted only by fresh feedback, without stopping at every
point. Lag retains the previous target, and only the endpoint requires stable
joint confirmation. LB cancellation and B emergency stop keep their existing
behavior. Intermediate references are debug logs rather than operator messages.
Feedback waits change execution timing, so this is not a physical acceleration
or jerk guarantee. Firmware speed/acceleration and workspace checks are unchanged.

Near an official joint boundary, a session with a previously sent legal target
checks the measured boundary excursion against the existing hold tolerance.
It does not add ordinary tracking lag to that excursion. Actual target-to-feedback
step limits and final hold-arrival checks remain enforced separately. Raw measured
angles are retained; a boundary hold or planning seed explicitly uses the legal
boundary and records that choice. Startup without a legal session target is strict.

## Input-only measurement

In the controller administrator's ordinary terminal, use the current editable environment:

```bash
export PIPER_DEV="$HOME/piper-hardware-acceptance/20260906-capture-timing/development-src"
"$PIPER_DEV/.venv/bin/piper-outcome-stack" xbox-input --list
"$PIPER_DEV/.venv/bin/piper-outcome-stack" xbox-input \
  --index "<enumerated-index>" --output "$HOME/piper-runs/xbox-$(date -u +%Y%m%dT%H%M%SZ)"
```

This tool imports only stdlib and pygame. It does not create a Robot, access CAN or
D435, or need `piper-socketcan`. Connect the Xbox by USB. Each prompt starts a bounded
input sample after Enter: neutral, shoulder, RB, B, four stick directions and two triggers.
The four-second/60 Hz defaults describe measurement only, not robot control limits.
The report contains actual GUID, SDL instance, axis ranges and measured button indices;
`events.jsonl` preserves button edges and monotonic receipt times. A removed selected
controller ends the measurement without reconnecting. An ambiguous button measurement
fails and preserves the raw report. Do not copy synthetic test mappings to hardware.

Use the measured ranges, neutral noise and directed movements to approve axis mapping,
trigger endpoints and deadzone. The tool does not freeze a mapping or generate an
hardware-acceptance record. No real hold parameters can be measured by an input-only test.

## Explicit runtime configuration

`OutcomePiperXboxConfig` requires these new fields, without numeric defaults:

| Field | Meaning |
|---|---|
| `emergency_stop_button` | Measured B index, distinct from `hold_button` |
| `hold_joint_tolerance_rad` | Per-joint absolute hold error tolerance |
| `hold_stable_time_s` | Continuous stable feedback duration |
| `hold_timeout_s` | Fixed confirmation deadline; greater than stable time |

All previous measured axis, trigger, control-rate, step and IK settings remain required.
Acceptance-file/boolean and digest bindings have been removed by the user. Supply
the numerical settings; the software still checks their validity and live feedback.
Record measured behavior honestly, including damping descent; no physical-estop
or no-drop attestation is required to start an authorized session.

## Session behavior

1. Explicitly enabling a motion session establishes one hold target from fresh complete joint
   feedback. There is no startup gripper command. Until hold confirmation, no movement
   input is accepted. Unpressed shoulder at startup does not request emergency stop.
2. Observe shoulder released, sticks centered, triggers released; then press shoulder
   while still neutral. Already holding it at startup or pressing while deflected does
   not arm movement. A failed press requires another release/neutral/press sequence.
3. Release shoulder to cancel pending intent, capture the current joint position once,
   send one hold command and confirm it. The target does not follow subsequent drift.
   Stable-window resets do not extend the attempt deadline. Repeated cached feedback
   does not establish continuous stability. Drift after confirmation reopens a bounded
   confirmation with the same target and no repeated hold write.
4. Once confirmed, release/neutral/new press resumes from the latest joint feedback;
   stale epochs are rejected. Confirmed LB holding rebases the orientation reference to the fixed joint hold target; mode switches retain that reference. Neutral LB rearming retains the arm hold; a fresh effective stick input starts arm motion, while triggers control the gripper independently. Pausing and resuming
   without trigger input retain the last commanded gripper opening and force, including
   when measured opening differs because an object is held.
5. B takes priority over ordinary inputs and IK and requests electronic stop, then
   latches the session. It never automatically resets, disables or resumes the arm.

Watchdog follows valid control-loop ticks, including waiting/holding/paused ticks.
Selected input removal or a stalled loop attempts hold only with healthy arm feedback;
confirmation then latches FAULT and ends the session. Cleanup does not send an additional
electronic stop after that confirmed hold. Explicit B can still request emergency stop.
Failed hold, stale feedback and other control faults use the existing electronic stop;
failed dispatch is recorded as `stop_unknown`. CAN removal or forced process termination
cannot guarantee any software hold. PiPER electronic stop may allow damped descent.

## Recording and acceptance

`intent`, `epoch`, and generation time are process-local action attributes; the public
Dataset/policy action stays seven rad/m values. The pinned official record loop ignores
`send_action()` returns, so `TelemetryDataset` substitutes the effective seven values
only for proven holding frames. Normal motion frames retain strict value equality.
Holding frames refer to the original `hold_move_j` and retained gripper command;
subsequent frames have no invented SDK timestamps. Images and states remain fresh.
Startup without a session command produces events, not training rows. Empty attempts
are recorded as empty; failed/interrupted sessions are not complete training data.

Offline tests cover rearming, fixed-target confirmation, watchdog, stale actions,
in-flight SDK interruption, B, retained grasp, stop failures and official
record/finalize/reload/replay with rerecord/resume. Hardware mapping, unloaded/loaded
hold, disconnect and recovery remain separate pending acceptance items. This change
does not activate a release or authorize real teleoperation.

## Restricted release/resume commissioning

`infra/acceptance/piper_xbox_hold_commission.py` uses the same `JointHold` and
`TeleopControl` as the Robot plugin, through the existing `JointRun` read/dispatch
checks. It is a bounded operator commissioning case, not another robot backend or
a replacement for full-workflow validation. It accepts a completed read-only reference,
a completed start-pose report (hold measurement or joint commissioning), and the measured Xbox mapping.

This case requires the already-enabled CAN/J pose from the hold report. After one
operator confirmation, it closes the empty gripper to 0 mm / 1 N and establishes
initial hold. With sticks/triggers neutral, a fresh shoulder press sends one zero
joint target at 1% speed. Release after movement begins and before arrival captures
one fixed holding target. New received feedback must confirm it before a neutral
release/press can resume the zero target. Completion requires both an observed
mid-movement release and resumed arrival, followed by final hold confirmation.
Releasing before any measured movement, or only after arrival, does not pass.

The case reuses commissioning tolerance 0.1 degree, stable time 0.3 s, timeout 10 s,
feedback age 0.2 s and maximum joint excursion 5 degrees. Deadzone 0.08 is the tested
input candidate. These are explicit commissioning settings; no formal acceptance
file is written. Joint/driver/gripper feedback remains checked, the hold target is
not repeatedly updated, and pause does not rewrite the gripper command. B requests
electronic stop. Detected input loss/loop delay holds with healthy feedback before
faulting; failed holding takes the electronic-stop path and records unknown results.
No automatic disable, reset, mode recovery, re-enable or retry is added.

This checks a single return path and the shared primitives, not full Robot workflow,
Cartesian IK, loaded grasp, forced-process termination, or a background watchdog.
The operator must remain at the arm and use the existing stopping procedure;
electronic stop can allow damped descent. All outputs and failed attempts remain
separate from formal acceptance and training data.

## Joint waypoint planning

A valid IK goal may require more correction than one control tick permits. The processor
subdivides the joint-space segment and emits its first waypoint, then recomputes from
fresh feedback next tick. It does not queue old targets across pauses or change the
committed orientation target. Both the requested Cartesian goal and the emitted waypoint
must satisfy workspace checks; joint limits and the configured dispatch step still
apply. This is joint-space interpolation, not a Cartesian straight-line guarantee.
Telemetry records the full IK goal and segment count in `joint_plan`; the seven action
values remain the actual waypoint sent to the robot and stored in the Dataset.


## Returning from a workspace boundary

A measured pose can slightly overrun the Cartesian box even when the dispatched
waypoint was inside it. After ordinary pause/neutral/shoulder rearming, neutral
input outside the box is a no-dispatch tick; it does not replace the committed orientation or send
an automatic return. An inward Xbox target may approach the box over several
bounded waypoints. No violated axis may worsen or cross the opposite face, no new
axis may leave the box, and at least one violation must improve. Both the IK goal
and dispatched waypoint are checked; direct non-Xbox actions still require an
in-box target. Outward or non-improving inputs request the usual confirmed hold.
Neutral rearm frames reuse the confirmed hold and retained gripper references without new SDK timestamps. Before a valid seven-value command exists they remain control-wait events.


## RB mode switching and six-axis pose control

Default mode is TRANSLATION with WRIST_PRIORITY: left stick X/Y, right stick up/down Z.
Right-stick horizontal is unused until X selects FIXED_ORIENTATION, where it controls yaw.
ORIENTATION uses left stick roll/pitch, right stick left/right yaw, and does not assign
right-stick up/down. All rotation axes are fixed in the arm base frame. LT/RT still
close/open the gripper, LB runs/holds and B requests the latched electronic stop.

RB switches once per new press, only with LB released, all four stick axes centered,
both triggers released, and holding confirmed. This includes the unused axis in
orientation mode. A held RB at startup must be released first. Invalid presses are
reported and discarded, never queued. Switching keeps the pose target and requires
a new neutral LB press before motion. Mode and switch results appear in logs/telemetry.

Raw inputs are six normalized stick/trigger values and hold/neutral/emergency/mode
button flags. The processor owns the mapping; there is no second mode state in the
input driver. `mode_switch_button` must be measured and distinct from LB/B.
`rotation_step_rad` replaces `yaw_step_rad` in active configs and limits the norm of
the combined rotation vector. The current trial values come from the active teleoperation and safety configs;
the launcher prints those values before the operator starts. Historical run copies
retain their original parameters.

Rotation vectors pre-multiply the fresh measured orientation for angular inputs.
In FIXED_ORIENTATION, zero angular input retains the current orientation reference.
WRIST_PRIORITY instead uses position-only IK and records the actual waypoint orientation. When all effective arm stick inputs center, the arm captures a fixed hold and rebases its orientation after confirmation; it no longer chases an unfinished pose. LB release also cancels unfinished motion and requires a fresh neutral LB press. Matrices are used
internally; Euler angles exist only at the pinned SDK FK/IK boundary, whose rotational
residual is a rotation vector. Valid, successful SDK dispatch commits the candidate
only while its epoch still permits running; rejected/interrupted/failed commands do
not advance it. No target-arrival claim is made. Re-enabling initializes orientation
from fresh feedback and starts in TRANSLATION again.

Measure RB without robot access:

```bash
"$PIPER_DEV/.venv/bin/piper-outcome-stack" xbox-input --list
"$PIPER_DEV/.venv/bin/piper-outcome-stack" xbox-input --buttons-only \
  --index "<listed-index>" --output "<new-buttons-directory>"
"$PIPER_DEV/.venv/bin/piper-outcome-stack" xbox-preview \
  --config "<teleoperate.json>" --buttons-report "<new-buttons-directory>"/report.json \
  --output "<new-preview-directory>"
```

The preview uses the measured RB and existing axis calibration. It constructs only
an Xbox input device and the shared control/mapping helpers, not a Robot, camera or
IK solver. Holding confirmation is simulated and labeled in every event. B ends the
preview without sending a stop. It does not silently freeze the measured mapping or
edit the real runtime config: bind the inspected RB index before real teleoperation.
The raw and seven-value Dataset interfaces have distinct roles; no mode or orientation
matrix is appended to the policy action vector.


## Stick-center holding with LB retained

With LB held, centering all effective arm stick inputs cancels pending arm intent and
enters CENTERING. Capture and send one fixed joint hold, then enter CENTERED only
after the existing stability criterion is met. Fresh stick input can resume directly
from CENTERED without releasing LB. Input observed during CENTERING is not queued;
only the current input after confirmation is used. The inactive right-stick vertical
axis in orientation mode does not prevent arm holding, but still participates in
full-neutral checks for LB rearming and RB switching.

Triggers remain independent while LB is held, including during CENTERING: gripper
commands use the existing width/force/step checks and need no IK or new joint target.
Released triggers retain the last gripper target and force. LB release disarms both
channels and follows the ordinary pause/rearm sequence. B, input loss and faults keep
the existing stop semantics; a rejected target does not acquire automatic resume.

Holding frames contain the fixed joint target plus the actual gripper target. When a
gripper command changes, a new joint/gripper reference ID pairs that new command with
the original hold command; its joint SDK timestamps are not rewritten. The action
vector remains seven values. Physical settling still takes sampling, communication
and controller response time; CENTERED is feedback-confirmed, not an instantaneous
stop promise.

### Gripper endpoint travel

Normalized trigger input plans only the available travel toward the configured
opening endpoint. For example, at 2.2 mm with a 5 mm closing increment, the target
is 0 mm; continued closing keeps that target, and opening reverses immediately
without an LB cycle. This applies both during arm motion and stick-centered holding.
A normal endpoint does not pause the arm. Released triggers retain the last target
and force. The approved software operating range is 0–90 mm / 1 N, including the old-table trial
(updated 2026-09-10). As of 2026-09-12, the target-step validation limit is
15 mm; the Xbox trigger increment remains 5 mm and force remains 1 N.
The larger validation limit does not establish smooth hardware tracking.

Action telemetry `gripper_plan` records feedback, requested increment, planned
increment, actual target, endpoint and whether travel was shortened; it adds no
policy/Dataset action dimensions. Dataset actions remain the actual seven-value
targets. Absolute commands still undergo strict bounds and step validation.
Feedback outside the command interval must return inside it in the requested
direction within a legal step; no automatic jump or enlarged range is introduced.
Driver/feedback faults, B stop and stale-action rejection retain their behavior.

## A/Y joint pose execution and recording

A requests the configured work pose and Y requests zero. Both use `joint_pose.py`,
POSE_READY/POSE_MOVING and normal Robot dispatch; there is no home-only executor.
A must be measured and configured; no index or site pose is guessed. Release/center,
request A or Y, release the target button, then hold LB. Release LB or deflect axes
cancels; B preempts; simultaneous requests and requests during recording are rejected.
After arrival the session holds enabled in PAUSED and does not automatically resume.

The relocation software and terminal recording phases are documented in
[piper_scene_collection.md](piper_scene_collection.md). New scene identity and actual
configuration are recorded; preparation and parking do not enter task demonstrations.
The original zero-return trial remains historical evidence, not new-site acceptance.

### Feedback at a command boundary

Joint command bounds are unchanged. A measured joint angle slightly outside those
bounds is accepted only if it remains within the configured hold arrival tolerance
of this session's last successfully sent, legal joint target. No prior target,
invalid targets or larger deviations still fail with joint index, measured angle,
bounds, excess and arrival tolerance in the error. Raw observations are preserved.

When capturing a hold in that condition, affected axes explicitly retain their last
legal targets; other axes use current feedback. The log and hold command record both
measurements and selected targets. IK uses an in-bounds initial seed for such validated
feedback; FK and action step checks still use the original measured angles. This
prevents a tolerated final-zero feedback deviation from becoming an illegal hold
command or an infeasible optimizer seed. It does not expand target bounds, suppress
controller/driver faults, or automatically recover electronic stop.

## Console output

The project teleoperate entry uses the same official `record_loop(dataset=None)`
as recording preparation. It prints no periodic `Teleop loop time`, cursor-up
sequences or motor tables, including when graphical visualization is enabled.
Startup/operator instructions, state transitions, A/Y progress, hold confirmations
and exceptions remain visible. Actual processing overruns still produce warnings.
No stdout redirection, logging suppression, upstream source edit or new quiet-mode
flag is used; timing measurement and Dataset telemetry remain available.


### A/Y 按时长规划对照（2026-09-15）

`teleop.pose_timing` 可指定六轴 `joint_velocity_rad_s`、`joint_acceleration_rad_s2`，以及 `gripper_velocity_m_s`、`gripper_acceleration_m_s2`、`period_s`。所有数值必须显式给出，周期须与 control_hz 一致；它们是外部参考规划参数，不是 SDK 速度百分比换算值，也不会写入驱动加速度。

规划保留原五次关节路径，先按解析速度/加速度上界确定共同时间，再按周期采样；采样点数同时满足原有步长预算。按固定周期时间表推进，正常SDK耗时不重置时间相位；错过整周期后重新安排下一周期，滞后不跳点、不突发追赶，固定超时与末端保持确认仍生效。遥测记录名义时长、参考时间及规划参数。插入跟随等待后，不宣称实际速度/加速度仍满足名义曲线界限，也不宣称机械抖动已消除。

未绑定规划参数的现用配置暂保留为现场对照基线；新的时长规划须在独立对照配置里验证后再替换。第一阶段不改变路径，不启用平滑自适应进度，不恢复手臂/腕部分段。原始 Dataset 不变；现场验证由操作者启动。

### 现用速度设置（2026-09-15）

用户批准现用运动入口的 SDK motion_speed_percent 统一为50。包括日常遥操作、采集准备、到回放起点和当前P3回放；时长规划使用 timed50.sh。5%/25%命名的对照配置、已保存会话和Dataset中的历史参数保留原值。规划速度/加速度、关节步长和8mm间隙不变，规划时长不会因SDK上限提高而自动缩短。50%现场表现待验证。
