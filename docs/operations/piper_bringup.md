# Standard PiPER and single D435 acceptance

The user confirmed arrival of the arm and camera on 2026-09-06. This is shipment
context, not acceptance of identity, wiring, firmware, safety, or motion. Use only
`piper-local` for real devices. The planning repository's
`docs/operations/piper_hardware_checklist.md` and `piper_official_alignment.md` define
the current on-site checks and outstanding software work.

## Before real CAN

1. Inspect the standard PiPER nameplate/serial, official gripper, power and harness,
   USB-CAN adapter, base fastening, tool/load, and the delivered manual revision.
   Record the actual firmware identity supplied with the arm; do not try driver
   variants or update firmware to find a working combination.
2. Identify the independent physical emergency stop and its effect. The teach button
   can start recording/playback; it and the host application's stop button do not
   establish an independent physical stop. Do not test them by starting a trajectory.
3. Verify the candidate commit, locks, plugin discovery, and immutable release. The
   release must include the single-camera contract and pass its own acceptance;
   an earlier release's acceptance marker does not validate the updated code.
4. Isolate the inspected real CAN interface while DOWN using `piper-socketcan` in the
   fixed `piper-can` namespace. Verify administrator bind and collaborator denial
   without sending frames. Establish exact USB hotplug rules only from inspected
   identifiers. Do not assume a default interface or create broad device rules.

## Read-only, stopping, then motion

5. After the administrator authorizes real CAN bring-up, use the immutable release in
   `read_only` for five connect/read/disconnect cycles. It sends firmware queries but
   does not enable, home, reset, or change motion mode. Record actual motor-enable
   status separately: the software name `CONNECTED_DISABLED` is not proof that the
   arm was disabled before connection. Verify units, feedback groups, and freshness.
6. Perform the manufacturer's approved, mechanically supported stop acceptance in
   a separate on-site administrator task. Official electronic emergency stop allows
   damped descent; disable/reset can lose support immediately. An in-process watchdog
   cannot send a CAN stop after the cable is removed or the process exits. Record
   these cases separately and keep the project motion gate closed if hazardous
   descent or unverified stop behavior remains.
7. Only after these checks produce real evidence, create the hardware-acceptance and
   safety documents. Bind exact live firmware, hardware identity and the approved
   safety-file digest. Freeze conservative limits, workspace, timing, speed percent,
   and gripper force. Never fill acceptance booleans merely to enter motion.
8. Start with approved single-joint increments, then joints/gripper and Xbox
   hold-to-run. No automatic home/reset/retry. Stop and end the session on unexplained
   motion, incorrect direction/zero, stale feedback, or a failed stop.

Before accepting motion, close the mode-confirmation gap: the current plugin reads
cached status immediately after setting the mode. Confirmation needs fresh feedback
within a bounded interval, without resending the mode or any motion command.

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

Equal fps does not prove synchronization. Before pilot collection, implement and
validate image/state/action timestamp recording: current SDK feedback uses wall time,
while the camera cache uses `perf_counter()`, and the recorder does not persist their
alignment. Keep the 30-minute concurrent stability, 20 safe episodes, and 20 pilot
trajectories with record/finalize/reload/replay checks. Test the image pipeline with
`infra/acceptance/lerobot_dataset_replay_smoke.py`; it provides synthetic evidence only.
