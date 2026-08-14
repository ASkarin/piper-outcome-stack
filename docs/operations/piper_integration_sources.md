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
arriving standard PiPER and its low-speed acceptance; they are not guessed here.
