# PiPER integration sources and version locks

The active implementation has one in-repository LeRobot plugin:
`packages/lerobot_robot_outcome_piper`. There is no separate adapter repository and no
runtime backend abstraction.

- LeRobot: `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- pyAgxArm: `799b8412fbe8b9156bc9892d3dbeb2df7e98be71`
- agx_arm_urdf coordinate check: `f6642ce0d7872c686f29c99e9e10cd23d1d49313`
- agx_arm_ros diagnostic reference: `22a9cf6c5ad2fd2e0743531936bc5dab007fa5bc`

Only the first two enter the runtime dependency graph. All runtime Git dependencies use
public HTTPS and full commits. Exact firmware and numeric safety values come from the
delivered standard PiPER and its low-speed acceptance; they are not guessed here.

## Using the printed manual's development links

OutcomeStack is application-level secondary development: the official SDK owns CAN
encoding and device commands; the LeRobot plugin adds the project's observation/action
contract, collection, teleoperation and safety checks. Training and evaluation use the
official LeRobot lifecycle.

The printed manual's [piper_sdk repository](https://github.com/agilexrobotics/piper_sdk)
now points users to **pyAgxArm**, which is already the project's pinned runtime SDK.
The manual's ROS links resolve to [noetic](https://github.com/agilexrobotics/Piper_ros/tree/noetic)
and [foxy](https://github.com/agilexrobotics/Piper_ros/tree/foxy). They are useful reference
material for protocol semantics, coordinates and visualization. They do not add a
runtime dependency or a second control route. Do not launch their examples during
OutcomeStack bring-up: the ROS launch defaults include automatic enable, and setting
`auto_enable=false` does not establish that already-enabled motors are disabled.

The 2026-09-06 code alignment addresses three concrete differences:

- `set_motion_mode()` sends a command while `get_arm_status()` reads asynchronous cached
  feedback. The plugin requests J mode once and waits within `feedback_timeout_s` for a
  status timestamp at or after that request with CAN control and J mode. The deadline
  and receive timestamp comparison use the host monotonic clock; raw SDK wall times
  remain provenance. Missing/stale confirmation, controller faults or a stop request prevent enable.
- Motion sessions also check CAN/J mode on subsequent feedback, including before enable
  and before sending an action. A change to teaching, offline playback or another motion
  mode latches the session instead of silently switching the controller back.
- The pinned SDK's [PiPER MDH constants](https://github.com/agilexrobotics/pyAgxArm/blob/799b8412fbe8b9156bc9892d3dbeb2df7e98be71/pyAgxArm/api/constants.py)
  use J2/J3 offsets of -172.22/-102.78 degrees. The official
  [FK reference](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/kinematics/piper_fk.py)
  identifies those as the offset model; the ROS documentation assigns that model to
  firmware **S-V1.6-3 and later**. Motion acceptance rejects earlier firmware before SDK
  construction because workspace checks and Xbox IK use this fixed model. Read-only
  firmware inspection retains the official driver mapping. No alternate MDH model or
  automatic firmware upgrade is introduced.

The ROS gripper multiplier converts a single-finger coordinate to total opening width.
OutcomeStack already uses total width in metres with `move_gripper_m()`; applying that
multiplier here would incorrectly double the command.

## Single-camera integration

The single D435 uses the pinned LeRobot RealSense implementation with an inspected
serial and RGB stream profile. The policy schema has exactly one image key,
`observation.images.d435`; depth is inspected and calibrated separately. The 2026-09-06
review uses the pinned SDK's [PiPER API](https://github.com/agilexrobotics/pyAgxArm/blob/799b8412fbe8b9156bc9892d3dbeb2df7e98be71/docs/piper/piper_api.md)
and [firmware reference](https://github.com/agilexrobotics/pyAgxArm/blob/799b8412fbe8b9156bc9892d3dbeb2df7e98be71/docs/piper/firmware_reference.md).
See [bring-up](piper_bringup.md) for the current acceptance order and unresolved gaps.

## AgileX's LeRobot integration reference

The 2026-09-06 review also inspected
[`agilexrobotics/lerobot-agilex@8469f3ab67e27987d217e8e1d144b8710a4170a5`](https://github.com/agilexrobotics/lerobot-agilex/tree/8469f3ab67e27987d217e8e1d144b8710a4170a5).
This manufacturer source was missing from the earlier comparison. It was read, not
installed or executed, and is not an additional runtime dependency.

- Its `pyproject.toml` declares LeRobot 0.3.4 and its Dataset code uses v2.1. OutcomeStack
  uses the pinned Hugging Face LeRobot 0.6.0 and Dataset v3; replacing the whole package
  would change those interfaces.
- Its robot factory and robot directories contain no PiPER Robot implementation.
  The README's deployment entry is `scripts/lerobot_inference-ros2.py`, which connects
  policy observations/actions through ROS 2 topics. OutcomeStack's necessary device
  adapter instead connects the official pyAgxArm SDK to the official LeRobot Robot API.
- The ROS 2 script defaults to two arms and three cameras, with seven values per arm.
  Its internals consume configured lists, but its defaults do not match our single arm
  and single `observation.images.d435` input. Do not add unused views or aliases.
- That script's policy selector supports ACT and diffusion; the repository also contains
  upstream SmolVLA model code. Reuse official model, processor, Dataset and training
  implementations rather than copying the vendor's full inference loop.
- Under its default joint-control configuration, `model_inference()` starts interpolation
  toward six zero joint positions and a 0.08 gripper value before asking the operator to
  start inference. This example is not a read-only probe or an accepted parking routine.
- Its `get_frame()` matches observations using ROS header timestamps. This is useful
  reference for our outstanding synchronization work: SDK/camera timestamps, frame
  age/skew checks and episode sidecars are implemented locally; their numeric limits
  and real-data behavior still require hardware acceptance.
  Equal nominal FPS does not establish synchronized observations.

The reference therefore supplements the existing official HF LeRobot + AgileX SDK
foundation. It does not currently supply a drop-in replacement for our PiPER plugin.
No dependency lock, control route or hardware acceptance status changes from this review.

## Simulation source selection — planning only

MuJoCo plus the existing commit-pinned agx_arm_urdf is the approved planning direction. Simulator version, converted assets, actuator/contact/camera parameters, data provenance and tests require a subsequent implementation task. No dependency or runtime source lock changes in this documentation update. Simulation stays separate from real SDK execution and does not introduce an automatic hardware fallback or ROS control route.
