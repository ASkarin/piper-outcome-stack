# piper-policy 0.1.13

Shared pure ACT/SmolVLA prediction, relative processors, temporal ensembling, asynchronous chunk queues, RTC and numerical transforms. No simulation or hardware imports.

Main project uses this workspace package; simulation installs a fixed wheel. Build with `uv build --wheel --no-sources packages/piper_policy`. Preserve serialized registry names `act_positive_gripper` and `piper_sqrt_gripper`. Seven axes are six joints in rad plus total gripper opening in m. `ACTChunkPredictor.predict_batch` accepts float NCHW RGB [0,1] and N×7 states and returns absolute N×H×7 chunks anchored to the input states.

Tests: `python -m pytest packages/piper_policy/tests`. Real observation age, hardware bounds and dispatch remain in the main application.

`GripperHeadRefiner` caches ACT decoder features from the actual RGB/state zero-latent inference path and refits only the final gripper output row. It leaves joint weights and default inference unchanged. Refit exports are inference checkpoints, without an upstream optimizer-resume state.

Its training-only width calibration minimizes prediction change under label-derived lower bounds. Since 0.1.13, correlated dual constraints that exhaust coordinate sweeps receive a bounded nonnegative active-set polish, with the same objective and KKT tolerance. Each polish has at most `2 * active_constraints + 1` face steps; the report separates these from coordinate sweeps. Unverified solutions still fail before changing the head. This does not clip inference actions or alter physical/evaluation limits.

`GripperHeadRefiner.fit_ridge` fits the same513 parameters with a single float64 regularized linear solve. It inverts the existing positivity transform on training labels and preserves it at inference; it does not clip actions.

`JointHeadRefiner` fits only the six joint output rows and preserves the gripper row. Both component tools share one frozen-feature/export implementation and one regularized linear solver; default prediction is unchanged.

Joint temporal refitting aligns cached decoder features and their original generation-state anchors with the existing TE rule. It fits executed joint commands while leaving runtime TE and the gripper unchanged; it never reanchors old chunks to the newest state.

`ACTDecoderRefiner` caches encoder memory and shared positional encodings through the existing RGB/state inference path. It supports the single-layer ACT checkpoint, freezes all parameters except the decoder/norm/action head, and keeps eval-mode zero latent and disabled dropout during gradient updates. Its masked normalized L1 combines expert labels with an explicitly separate parent-consistency term on selected old samples. Checkpoint exports retain original processors and registry names; any local optimizer state is a new refit state, not upstream training resume.

`ACTDecoderRefiner.execution_loss` averages three teacher losses over the full chunk, its first five steps, and its first action. Gripper labels use the shared inverse positivity transform before normalized L1, giving the raw head a gradient even near the positive output floor. Old-sample consistency uses the same three temporal windows on normalized physical predictions, retaining full-chunk coverage while emphasizing early execution. The original loss and all inference transforms remain unchanged. Historical experiments keep their recorded fixed wheel version.

`step_with_reference_projection` removes the component of an optimizer's proposed parameter displacement that increases a selected reference loss to first order. It uses the actual AdamW displacement (including momentum/decay), leaves optimizer moments intact, and reports the applied floating-point residual. This is a local training constraint, not an action clamp or a guarantee of nonlinear/validation performance. `execution_loss` exposes the old-input teacher loss separately for this reference; the total objective is unchanged.

The projection also accepts two reference losses and solves their joint halfspace projection by enumerating the at-most-three nonempty active sets. `physical_joint_losses` exposes full-chunk and first-action joint MAE in radians, separately from gripper loss, for fixed training-only reference banks. Log fields contain one entry per reference; historical runs retain their fixed wheels and recorded schemas.
