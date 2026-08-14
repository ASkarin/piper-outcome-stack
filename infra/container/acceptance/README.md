# Host acceptance procedure

Keep raw output under the administrator's persistent run or admin directory. Existing
locked-image verification JSON is historical evidence; it is not a gate for the
current mutable Python environment.

## 1. External SSH boundary

From each member's own workstation, connect with that member's private key and confirm
the reported user. Root and password-only authentication must remain rejected:

```bash
ssh piper-training id
```

The administrator performs the root/password rejection probes without recording
addresses, ports, user-specific key paths, or private material in Git.

## 2. Account, filesystem, and shared Python boundary

Run the doctor once as each account from a personal clone:

```bash
piper-env-doctor --repo "$PWD" --json
```

As the collaborator, all of the following conditions must hold:

```bash
! sudo -n true
! command -v docker
test -r /workspace/piper/python-env
test ! -w /workspace/piper/python-env
test -w "/workspace/piper/staging/$USER"
test -w "/workspace/piper/runs/$USER"
test -w /workspace/piper/cache
test -w /workspace/piper/locks
test ! -w /workspace/piper/releases/datasets
test ! -w /workspace/piper/releases/models
test ! -w /workspace/projects/piper-outcome-stack
test ! -r "/workspace/users/<admin>"
```

The administrator must be able to run `sudo -n true`, modify the shared Python
environment through `piper-python`, and keep both release directories read-only during an
ordinary login.

## 3. Mutable Python cutover

Choose a small package that is absent from the base seed:

```bash
piper-python install <small-package>
piper-python snapshot
```

Confirm that the collaborator can import it but cannot run `piper-python install` or write
the shared environment. Run `./piper-compose restart` on the host, reconnect both users,
and confirm the import still succeeds. Finally uninstall the temporary package if it
is not useful to the project. The install, uninstall, and snapshots must appear in
`/workspace/piper/python-env-history/operations.jsonl`.

## 4. GPU, shared memory, and run records

From the clean canonical checkout, the administrator runs:

```bash
infra/container/acceptance/run_gpu_acceptance.sh
```

The script reserves all three GPU UUIDs through `piper-gpu-run`, then records:

- one tensor allocation on each GPU;
- 100 all-reduce iterations on the preferred first two GPUs;
- 100 all-reduce iterations across all three GPUs;
- a three-rank, two-workers-per-rank DataLoader run lasting 600 seconds;
- `environment.json`, `summary.json`, metrics placeholders, and
  `python-packages.txt`;
- a real TensorBoard event file and a real W&B offline run file.

`environment.json` must record the shared Python executable and the SHA-256 of the
complete package list. Retain raw output even if a check fails.

## 5. Artifact and offline loading

Fetch a deliberately small model or dataset at an exact repository commit:

```bash
piper-artifact-fetch --repo <owner/name> --revision <commit> --type <model|dataset>
```

Review the manifest, promote it with
`sudo piper-artifact-promote --manifest <path>`, and confirm that a second promotion to
the same destination is rejected. Load the promoted local path with
`HF_HUB_OFFLINE=1`; merely listing files is not an offline-load test.

## 6. Restart persistence

Record SSH host-key fingerprints and hashes of small files in each persistent area.
Run `./piper-compose restart`, reconnect both users, and verify that host keys, authorized
keys, personal clones, the shared Python environment and its history, promoted
artifacts, and run records remain unchanged.

Container recreation is a different operation and must be invoked explicitly with
`./piper-compose recreate`.
