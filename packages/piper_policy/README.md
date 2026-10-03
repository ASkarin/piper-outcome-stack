# piper-policy 0.1.0

Shared pure ACT/SmolVLA prediction, relative processors, temporal ensembling, asynchronous chunk queues, RTC and numerical transforms. No simulation or hardware imports.

Main project uses this workspace package; simulation installs a fixed wheel. Build with `uv build --wheel --no-sources packages/piper_policy`. Preserve serialized registry names `act_positive_gripper` and `piper_sqrt_gripper`. Seven axes are six joints in rad plus total gripper opening in m. `ACTChunkPredictor.predict_batch` accepts float NCHW RGB [0,1] and N×7 states and returns absolute N×H×7 chunks anchored to the input states.

Tests: `python -m pytest packages/piper_policy/tests`. Real observation age, hardware bounds and dispatch remain in the main application.
