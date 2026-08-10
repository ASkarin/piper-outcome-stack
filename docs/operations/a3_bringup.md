# EduLite A3 guarded bring-up

Stage 1B defaults to no hardware. Do not run a motor command from the GPU server or any other host until the execution host, robot serial, CAN adapter, power, physical emergency stop, cleared workspace, and on-site administrator are recorded in the local hardware checklist.

## Allowed order

1. Verify `uv.lock`, installed SDK metadata, and LeRobot plugin discovery.
2. Start from the draft templates in the pinned private adapter repository, then complete and approve versioned calibration and safety files; draft/null values cannot enable motors.
3. As the unique administrator, connect the immutable release in `read_only` mode and
   record identity/status evidence; do not use sudo for the Python process.
4. With power disabled, verify zero and direction conventions.
5. Obtain explicit approval for conservative numeric limits and a single-joint increment.
6. Test disable, software stop, and physical emergency stop before any broader motion.

Any unexplained motion, zero drift, CAN fault, feedback timeout, or failed stop ends the session. Stage 1B does not authorize data collection, multi-joint sweeps, unattended motion, or use of official example limits as approved project limits.

## Teleoperation decision

The primary path is an Xbox-compatible gamepad, following the official interface design. The backup is restricted zero-torque teaching. Until Stage 1B hardware gates pass, only synthetic input mapping and the safety gate may be tested; hardware availability and real teleoperation remain unverified.
