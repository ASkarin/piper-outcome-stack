# Standard PiPER and single D435 acceptance

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
2. Identify the independent physical emergency stop and its effect. The teach button
   can start recording/playback; it and the host application's stop button do not
   establish an independent physical stop. Do not test them by starting a trajectory.
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
   status separately: the software name `CONNECTED_DISABLED` is not proof that the
   arm was disabled before connection. Verify units, feedback groups, and freshness.

   Daily administrator debugging may instead use a personal editable checkout and its
   private `.venv`, through the same `piper-socketcan exec` launcher. Preserve the Git
   baseline/diff, untracked source used, command, interpreter and configuration with debug
   output. Source-only changes need a process restart, not a release; these debug runs do
   not replace the formal five-cycle evidence. All device and motion gates still apply.
6. Perform the manufacturer's approved, mechanically supported stop acceptance in
   a separate on-site administrator task. Official electronic emergency stop allows
   damped descent; disable/reset can lose support immediately. An in-process watchdog
   cannot send a CAN stop after the cable is removed or the process exits. Record
   these cases separately and keep the project motion gate closed if hazardous
   descent or unverified stop behavior remains.
7. Only after these checks produce real evidence, create the hardware-acceptance and
   safety documents. Bind exact live firmware, hardware identity and the approved
   safety-file digest. Freeze conservative limits, workspace, timing, speed percent,
   and gripper force. Motion requires S-V1.6-3 or later to match the pinned SDK's
   PiPER MDH model; a compatible CAN driver alone is insufficient. Earlier firmware
   can be inspected in read-only mode, but cannot pass this motion gate. Never fill
   acceptance booleans merely to enter motion.
8. Start with approved single-joint increments, then joints/gripper and Xbox
   hold-to-run. No automatic home/reset/retry. Stop and end the session on unexplained
   motion, incorrect direction/zero, stale feedback, or a failed stop.

The plugin requests J mode once, then waits for fresh CAN/J status within
`feedback_timeout_s` before enabling. Missing confirmation or a controller fault aborts
connection. Subsequent motion feedback must remain in CAN/J mode; a mode change latches
the session before the next action. Receive timestamps are captured around the official
SDK parsing callback using the host monotonic clock. SDK wall timestamps are retained
as provenance, not used for timeout arithmetic. Initial complete feedback has a bounded
wait distinct from runtime staleness; firmware is queried once. These software paths
still require the five normal-plugin read-only cycles on the candidate release.
See the [official-source comparison](piper_integration_sources.md).

## One D435

Independent RGB/depth camera inspection can precede robot motion. Bind the inspected
numeric serial, verify USB topology and both video/USB-node permissions, and use
`PIPER_D435_SERIAL` for doctor. Multiple nodes from one D435 are not multiple cameras.
Check RGB/depth profiles, depth scale, usable working distance, invalid depth pixels,
intrinsics, mounting, exposure and timestamp domains.

The policy input is one fixed external RGB view plus the seven robot state values.
The only camera key is `d435`; the only Dataset image key is
`observation.images.d435`. Configure it with measured values:

```python
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

cameras = {
    "d435": RealSenseCameraConfig(
        serial_number_or_name=verified_serial,
        width=verified_width,
        height=verified_height,
        fps=verified_fps,
        use_rgb=True,
        use_depth=False,
        color_mode="rgb",
    )
}
```

These variables are intentionally supplied by acceptance, not defaults. Robot-only
bring-up may use an empty camera map; `record` requires D435, matching camera/Dataset/
Xbox fps, and `dataset.push_to_hub=false`. Upload after finalization from the host
namespace. Depth is inspected/calibrated separately and is not a policy feature.

Equal fps does not prove synchronization. The RGB publication extension preserves
RealSense frame number/device timestamp/domain and host receive/publish time atomically.
A read consumes a new frame or times out; capture failures do not retry. The common
image/state axis is host receipt time, not an estimate of simultaneous exposure.

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
