# Xbox pause, resume and input measurement

The shoulder button controls recoverable teleoperation. B requests the independent
latched electronic emergency stop. This software change does not constitute real-arm
hold, loaded-gripper, disconnect or recovery acceptance.

## Input-only measurement

In the controller administrator's ordinary terminal, use the current editable environment:

```bash
export PIPER_DEV="$HOME/piper-hardware-acceptance/20260906-capture-timing/development-src"
"$PIPER_DEV/.venv/bin/piper-outcome-stack" xbox-input --list
"$PIPER_DEV/.venv/bin/piper-outcome-stack" xbox-input \
  --index <enumerated-index> --output "$HOME/piper-runs/xbox-$(date -u +%Y%m%dT%H%M%SZ)"
```

This tool imports only stdlib and pygame. It does not create a Robot, access CAN or
D435, or need `piper-socketcan`. Connect the Xbox by USB. Each prompt starts a bounded
input sample after Enter: neutral, shoulder, B, four stick directions and two triggers.
The four-second/60 Hz defaults describe measurement only, not robot control limits.
The report contains actual GUID, SDL instance, axis ranges and measured button indices;
`events.jsonl` preserves button edges and monotonic receipt times. A removed selected
controller ends the measurement without reconnecting. An ambiguous button measurement
fails and preserves the raw report. Do not copy synthetic test mappings to hardware.

Use the measured ranges, neutral noise and directed movements to approve axis mapping,
trigger endpoints and deadzone. The tool does not freeze a mapping or generate an
approved hardware gate. No real hold parameters can be measured by an input-only test.

## Explicit configuration and hold gate

`OutcomePiperXboxConfig` requires these new fields, without numeric defaults:

| Field | Meaning |
|---|---|
| `emergency_stop_button` | Measured B index, distinct from `hold_button` |
| `hold_joint_tolerance_rad` | Per-joint absolute hold error tolerance |
| `hold_stable_time_s` | Continuous stable feedback duration |
| `hold_timeout_s` | Fixed confirmation deadline; greater than stable time |

All previous measured axis, trigger, control-rate, step and IK settings remain required.
Before a real Xbox connection, the existing hardware-acceptance JSON must contain
`teleoperation_hold` with `verified: true` and the exact approved
`joint_tolerance_rad`, `stable_time_s`, `timeout_s`. This section is written only after
the separately authorized on-site hold tests. Do not set it to pass merely because
software tests or basic joint commissioning succeeded.

## Session behavior

1. Connecting a motion session establishes one hold target from fresh complete joint
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
   stale epochs are rejected. Roll/pitch remain session-locked. Pausing and resuming
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
