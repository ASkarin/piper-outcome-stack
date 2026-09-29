# Standard PiPER and camera acceptance

The user confirmed arrival of the arm and camera and the factory-supplied PiPER kit
on 2026-09-06. Firmware, safety and motion require their own acceptance. Use only
`piper-local` for real devices. The planning repository's
`docs/operations/piper_hardware_checklist.md` and `piper_official_alignment.md` define
the current on-site checks and outstanding software work.

## Before real CAN

1. Use the complete factory-supplied PiPER kit confirmed by the user. Record the
   standard PiPER nameplate/serial, official gripper and enumerated USB-CAN identity;
   verify base fastening, tool/load, and the delivered manual revision.
   Record the actual firmware identity. If it is not supplied with the arm, use
   `infra/acceptance/piper_read_only_probe.py`: the pinned PiPER drivers share the
   same firmware-query implementation, so its base profile can query identity once
   before selecting the matching driver for feedback. This is an initial SDK
   inspection, separate from five-cycle plugin acceptance. Do not try driver
   variants or update firmware to find a working combination.
2. Record the delivered stopping behavior: this arm has no brake or independent
   physical emergency stop. Electronic emergency stop permits damped descent;
   the accessible arm-only power switch cuts power without preventing descent.
   Shoulder release uses enabled position hold. Do not confuse these behaviors.
3. For formal acceptance, verify the candidate commit, locks, plugin discovery, and immutable release. The
   release must include the single-camera contract and pass its own acceptance;
   an earlier release's acceptance marker does not validate the updated code.
4. Isolate the inspected real CAN interface while DOWN using `piper-socketcan` in the
   fixed `piper-can` namespace. Verify administrator bind and collaborator denial
   without sending frames. Establish exact USB hotplug rules only from inspected
   identifiers. Do not assume a default interface or create broad device rules.

## Read-only, stopping, then motion

5. For formal five-cycle acceptance, after the administrator authorizes real CAN bring-up, use the immutable release in
   `read_only` for five connect/read/disconnect cycles. It sends firmware queries but
   does not enable, home, reset, or change motion mode. Record actual motor-enable
   status separately: the software name `CONNECTED` is not proof that the
   arm was disabled before connection. Verify units, feedback groups, and freshness.

   Daily administrator debugging may instead use a personal editable checkout and its
   private `.venv`, through the same `piper-socketcan exec` launcher. Preserve the Git
   baseline/diff, untracked source used, command, interpreter and configuration with debug
   output. Source-only changes need a process restart, not a release; these debug runs do
   not replace the formal five-cycle evidence. Device permissions and live runtime checks still apply.
6. Record actual stopping and recovery behavior during explicitly authorized on-site
   tests. An in-process watchdog cannot send a CAN stop after cable removal or
   process termination. Test results describe the tested conditions only.
7. Supply an explicit numeric safety configuration: joint/gripper limits, workspace,
   timing, speed and gripper force. The plugin validates live firmware compatibility,
   complete fresh feedback, controller/driver health and CAN/J mode. Motion requires
   S-V1.6-3 or later for the pinned PiPER MDH model. There is no separate hardware
   acceptance file, checklist attestation or safety-file digest required to start.
8. Use the same Robot and Xbox processor for supervised debugging. Configure hold
   tolerance and timing explicitly; see [Xbox setup](piper_xbox.md). Ordinary invalid
   targets are rejected and request a fixed hold; B requests electronic emergency
   stop. Feedback/control faults end the session. No automatic home/reset/retry.

The explicit `enable()` operation requests J mode once, then waits for fresh CAN/J status within
`feedback_timeout_s` before enabling. Missing confirmation or a controller fault aborts
enable. Subsequent motion feedback must remain in CAN/J mode; a mode change latches
the session before the next action. Receive timestamps are captured around the official
SDK parsing callback using the host monotonic clock. SDK wall timestamps are retained
as provenance, not used for timeout arithmetic. Initial complete feedback has a bounded
wait distinct from runtime staleness; firmware is queried once. These software paths
still require the five normal-plugin read-only cycles on the candidate release.
See the [official-source comparison](piper_integration_sources.md).

### Diagnostic acceleration tools

Operator-run diagnostics, not commissioning or startup gates. Both take explicit
`--interface` and `--firmware` and write a new JSON report (`--output`, never overwritten):

- `infra/acceptance/piper_acceleration_query.py`: one acceleration-limit query per joint;
  no writes, enable or motion. The fixed `piper-query acceleration` operation is the
  normal passive entry; use this script only from an explicit `piper-socketcan exec` debug session.
- `infra/acceptance/piper_j5_acceleration_trial.py apply|restore`: after Enter confirmation,
  writes only the J5 maximum acceleration (5.0→2.5 or 2.5→5.0 rad/s²) when the read-back
  current value matches; one write, read-back verified, no retry or automatic restore.

## Configured RealSense cameras

A single D435 is the current example configuration. Camera names, count, serial or
unique device name, and RGB/depth streams are configurable. Doctor enumerates visible
video nodes and their matching USB nodes; permissions do not establish frame delivery.
Check actual profiles, depth scale, invalid pixels, intrinsics, mounting, exposure and
timestamp domains. See [RGBD configuration and storage](camera_rgbd.md) for the complete
configuration fragment and training input selection.

Robot-only bring-up can use an empty camera map. Recording requires cameras whose
configured fps is at least the Dataset fps, matching Dataset/Xbox control rates, and
`dataset.push_to_hub=false`. Depth is stored as metric TIFF plus raw Z16 and metadata;
RGB model inputs remain separate from the captured modalities.

Stream metadata and images are published together. The common image/state axis is
host receipt time, not an estimate of simultaneous exposure or hardware synchronization.

`OutcomePiperConfig.capture_timing` takes four measured, approved positive seconds:
`camera_max_age_s`, `joint_max_skew_s`, `image_state_max_skew_s`, and
`observation_max_age_s`. Pass them through `--robot.capture_timing.<field>=<approved-value>`.
Read-only sessions may omit them for measurement; motion with a camera and record
require them. Do not copy synthetic test limits to hardware. Robot-only motion still
uses its frozen feedback timeout for observation-to-command age.

Recording calls the official `record_loop` and Dataset APIs, with episode-boundary
orchestration for `telemetry/<session>/events.jsonl` and `complete.json`. Sidecars contain
session/episode/frame/attempt identities, the observation's receive times, frame metadata,
processed action generation time, and SDK call start/end/results. A direct plain-dict
action has no known generation time (null); dispatch time is recorded independently.
An SDK return is not a target-arrival acknowledgement. Policy vectors remain unchanged.
Xbox pause frames retain the fixed joint hold and last gripper target, referencing
the original commands. They do not invent per-frame SDK writes. Before the session
has a valid seven-value command, startup waiting is logged as events only. Pauses
continue to consume fresh images/state and do not extend the episode timer.
Discarded attempts stay in the event log. Failed/interrupted/finalization-incomplete
sessions have no valid complete marker and are rejected by resume and data acceptance.

Before promoting any locally recorded Dataset or using it for training, run:

```bash
piper-outcome-stack audit-dataset --root <dataset-root> --repo-id <exact-repo-id>
```

Keep the Dataset and telemetry directory together. Audit failure blocks promotion;
never manufacture completion markers, discard telemetry, or silently train on an
interrupted recording. Resume additionally verifies exact robot, fps, feature names,
shapes and dtypes using the current Robot.name contract.

Keep the 30-minute concurrent stability, 20 safe episodes, and 20 pilot trajectories
with record/finalize/reload/replay checks. Tests `test_capture_timing.py` and
`test_recording_telemetry.py` exercise synthetic timing and real official Dataset APIs;
they do not freeze hardware thresholds or count as real-robot acceptance.
