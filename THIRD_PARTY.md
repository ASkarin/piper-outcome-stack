# Third-party sources and license notices

PiPER OutcomeStack dynamically links to or imports the following fixed upstream
dependencies. Their code is not copied into this repository.

| Component | Fixed source | License | Purpose |
|---|---|---|---|
| LeRobot | `huggingface/lerobot@30da8e687a6dfc617fcd94afc367ac7071c376ce` | Apache-2.0 | Dataset, training, replay, robot and teleoperator plugin APIs |
| pyAgxArm | `agilexrobotics/pyAgxArm@799b8412fbe8b9156bc9892d3dbeb2df7e98be71` | LGPL-3.0-only | Standard PiPER CAN SDK and MDH forward kinematics |
| agx_arm_urdf | `agilexrobotics/agx_arm_urdf@f6642ce0d7872c686f29c99e9e10cd23d1d49313` | MIT | Coordinate and gripper-width verification source; not a runtime dependency |
| agx_arm_ros | `agilexrobotics/agx_arm_ros@22a9cf6c5ad2fd2e0743531936bc5dab007fa5bc` | MIT | RViz and diagnostic reference only; not a control, collection, or release dependency |

The pyAgxArm package remains replaceable as a separately installed Python dependency.
The exact upstream LGPL-3.0-only text is retained at
`licenses/pyAgxArm-LGPL-3.0-only.txt`, together with the GPLv3 text it incorporates by
reference at `licenses/GPL-3.0-only.txt`. The exact MIT copyright and permission notices
for both reference repositories are retained at `licenses/agx_arm_urdf-MIT.txt` and
`licenses/agx_arm_ros-MIT.txt`. These files are notices only; no URDF or ROS source is
copied into this project or installed into the runtime.
