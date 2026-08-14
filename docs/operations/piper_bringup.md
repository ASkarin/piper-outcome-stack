# Standard PiPER guarded bring-up

Stage 1B defaults to no hardware. Do not run a motor command from the GPU server or any other host until the execution host, robot serial, CAN adapter, power, physical emergency stop, cleared workspace, and on-site administrator are recorded in the local hardware checklist.

## Allowed order

1. Verify `uv.lock`, installed SDK metadata, and LeRobot plugin discovery.
2. Create the versioned hardware-acceptance and safety files from the inspected hardware and low-speed tests; incomplete or guessed values cannot enable motors. The acceptance record must bind the nameplate model, robot serial, official gripper, USB-CAN adapter, physical emergency stop, exact `get_firmware()` identity, and safety-file digest. The safety file must freeze the position-mode speed percentage and gripper force.
3. While the verified real CAN interface is down, use root-owned `piper-socketcan` to
   move it into the fixed `piper-can` network namespace. The host namespace must no
   longer see it. Run the doctor without sending a frame: the administrator process in
   the namespace must be non-root with zero effective capabilities and able to bind;
   the collaborator in the host namespace must not be able to bind or enter the
   namespace. Host `vcan` remains available for software tests. Once the exact adapter
   vendor/product/serial are frozen, install an exact udev match that requests the
   non-resident `piper-socketcan-isolate@<interface>.service` and prove the same boundary
   after unplug/replug; do not create a broad CAN hotplug rule.
4. As the unique administrator, use `piper-socketcan exec --` to connect the immutable
   release in `read_only` mode and record identity/status evidence. The launcher uses
   sudo only to enter the namespace, then runs the Python process as the administrator's
   ordinary UID/GID.
5. With power disabled, verify zero and direction conventions.
6. Obtain explicit approval for conservative numeric limits and a single-joint increment.
7. Verify the damped electronic emergency-stop command under communication loss, watchdog expiry, Xbox disconnect, and hold-to-run release, including that the arm does not drop. The first motion release does not accept a software-only hold strategy. Test the physical emergency stop before any broader motion.

Any unexplained motion, zero drift, CAN fault, feedback timeout, or failed stop ends the session. Stage 1B does not authorize data collection, multi-joint sweeps, unattended motion, or use of official example limits as approved project limits.

## Teleoperation decision

The only formal path is teleoperator type `outcome_piper_xbox` through the
`piper-outcome-stack` workflow, which injects the canonical action processor into
official LeRobot. Until Stage 1B hardware gates pass, only synthetic input mapping and
the safety gate may be tested; hardware availability and real teleoperation remain
unverified.
