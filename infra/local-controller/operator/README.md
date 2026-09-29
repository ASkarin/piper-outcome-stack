# Stable development operator entry

`~/.local/bin/piper` runs `operator.py` using the controller's existing private
Python. Active configuration lives in `~/.config/piper-outcome-stack/operator/`.
This is a development entry, not a formal release or a hardware stability claim.
No service, CAN setup, new backend or sudo permission is installed.

| Command | Behavior |
|---|---|
| `piper record` | New Xbox raw capture; existing LB/X/RB/A/Y/B controls. |
| `piper status` | Fixed passive status query. |
| `piper link`, `firmware`, `limits`, `acceleration` | Existing fixed query operations. |
| `piper recover` | One confirmed electronic-stop recovery; no enable or position command. |
| `piper convert SESSION` | Read SESSION/config.json, convert closed Xbox raw data offline. |
| `piper audit SESSION` | Audit that session's converted Dataset. |
| `piper pauses SESSION` | Generate pause candidates only; never delete frames. |
| `piper replay CONFIG.json` | Explicit continuous replay config, unique result path; no automatic approach. |

SESSION is the printed operator run directory, not the raw image directory.
Recovery and replay do not use Xbox buttons. Recovery retains its Enter confirmation.
Replay retains `start`, `stop` and Ctrl+C semantics; it requires the selected
trajectory's actual start pose. These commands must be operator-executed on site.

`record` checks the configured SSD UUID and creates new raw/output paths. Logs and
configuration snapshots are under `~/piper-runs/operator/`. Each new process reads
the active configuration; existing run snapshots and datasets are never updated.
Scene relocation requires updating the active scene/workspace configuration.

In preparation, wait for confirmed hold before `start P1`. The first start
acquires the current legal gripper width and configured holding force; no trigger
tap is required. Use `end`, `save success/failure/cancelled`, then
start the next position or `quit`. Save seals raw files; conversion remains a
separate offline operation. No automatic pause removal or replay is performed.

For an already open shell after installation:

```bash
export PATH="$HOME/.local/bin:$PATH"
piper --help
```

Shell routing tests use fake subprocesses and never access hardware.

## Default capture, approved 2026-09-20

Use `piper record` without a trial profile: control and Dataset are 50 Hz, the
D435 streams 640×480 RGBD at 60 fps. The default remains 20 attempts (five per
P1–P4), up to 180 seconds per attempt, with failures retained separately. Each
invocation creates an independent session; it does not resume or mix old 20 Hz data.
New default raw/Dataset sessions use the SSD `piper-data/raw/xbox-50hz` and
`piper-data/datasets/xbox-50hz` directories. Existing datasets and run snapshots
keep their original rates and semantics.

The gripper reference is time-based, capped at 120 mm/s, with 1.2 m/s² ramp and
8 mm lead. A/Y use incremental full-route validation while holding. Per-cycle
XYZ/rotation input is 2.5 mm/0.5 degrees. Approved teleoperation defaults
(2026-09-21): SDK ceiling 75%, ramp 0.5 s, lead_cycles 2.4 at 0.02 s:
125 mm/s per coordinate and 25 degrees/s, with 6 mm per-coordinate and
1.2-degree combined rotation lead. A/Y retain their existing timing and targets;
independent replay configurations are unchanged.
This change does not automatically start capture, replay, conversion or training.

## RB selection while confirming hold

After releasing LB and centering all inputs, RB may select the next mode before
hold confirmation finishes. Wait for the mode-ready prompt before pressing LB.
An early LB press, stick/trigger input, conflict, second RB press or phase/fault
change cancels the pending choice. No new SDK position command or hold-window
restart is caused by selection itself. X keeps the original switching conditions.
