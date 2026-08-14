# Remote training container

This directory defines the PiPER OutcomeStack training container. It provides two SSH
accounts, three RTX 3090 GPUs, a persistent shared Python environment, personal Git
clones and run directories, and administrator-controlled model/dataset releases. It
does not include camera, CAN, gamepad, ROS 2, or local-controller dependencies.

## Runtime model

- `/opt/piper/.venv` is the base-image seed used only to initialize a new workspace.
- `/workspace/piper/python-env` is the persistent shared environment used by normal
  logins and formal runs.
- The administrator owns and may modify the shared environment. The collaborator may
  read and execute it but cannot modify it.
- `/workspace/piper/bin` contains the stable PiPER commands and precedes the shared
  environment on `PATH`.
- `piper-python install`, `uninstall`, `list`, and `snapshot` manage the shared
  environment. Mutations and snapshots record the operator, resolved package list,
  timestamp, and `pip freeze --all` SHA-256 under
  `/workspace/piper/python-env-history`.
- A missing Python package is installed in place; it does not require a container
  image build or pull request.

The collaborator may create an experimental venv inside their own home. Such a venv is
personal and is not the environment used by formal `piper-gpu-run` jobs.

## Tracked files

- `Dockerfile`: optional CUDA 12.8 base image and the seed Python 3.12 environment.
- `compose.yaml`: three GPUs, 16 GiB `/dev/shm`, SSH-only ingress, persistent storage,
  and no privileged, host-IPC, or Docker-socket access.
- `entrypoint.sh`: creates accounts, installs public keys, enforces ACLs, initializes
  the shared Python environment, and starts key-only SSH.
- `init-shared-python.sh`: copies the seed environment once and installs the
  persistent commands and profile.
- `bin/`: Python environment management, environment inspection, GPU reservation,
  mirror-only artifact acquisition, and administrator-only release promotion.

## Administrator deployment inputs

1. Copy `env.example` to `.env`.
2. Set `PIPER_IMAGE` to the image reference the administrator wants Compose to use.
3. Replace every `CHANGE_ME` value and set the file mode to `0600`.
4. Create the workspace, SSH host-key, and authorized-key host directories.
5. Put `admin_authorized_keys` and `collaborator_authorized_keys` in the authorized-key
   directory.
6. Run `./piper-compose config`.

Real usernames, IP addresses, ports, host paths, private keys, and service tokens never
belong in Git.

The wrapper commands have deliberately separate effects:

```bash
./piper-compose config
./piper-compose pull       # explicit image download
./piper-compose up         # create/start without an automatic pull
./piper-compose restart    # ordinary restart; no pull or recreation
./piper-compose recreate   # explicit force-recreation
```

## Optional base-image publication

Ordinary pull requests run `validate`, render the Compose configuration, reclaim an
explicit allowlist of unrelated SDKs on the disposable runner, and perform a real
non-publishing build of the locked CUDA image in `container-build`. Merging a dependency,
source, test, or documentation change still does not publish an image.

An administrator may manually dispatch `remote-training-container` with
`publish_image=true` to refresh the optional GHCR base image. That job still produces
SBOM and provenance metadata, but it does not create a second deployment-approval pull
request or deploy automatically. Updating `PIPER_IMAGE` and recreating the container are
separate administrator decisions.

## User workflow

Each user keeps an independent clone:

```text
/workspace/users/<user>/src/piper-outcome-stack
```

Useful commands:

```bash
python --version
piper-python list
piper-env-doctor --repo "$PWD" --json
PYTHONPATH=src python -m pytest
```

The administrator may install a package immediately:

```bash
piper-python install <package>
piper-python snapshot
```

Formal GPU commands use exact GPU UUIDs:

```bash
piper-gpu-run \
  --gpus GPU-UUID-1,GPU-UUID-2 \
  --run-id EXP-EXAMPLE-A001 \
  --repo "$PWD" \
  --dataset-manifest /workspace/piper/releases/datasets/owner--dataset@COMMIT/piper-artifact-manifest.json \
  -- \
  python train.py
```

The wrapper refuses dirty Git worktrees by default, obtains cooperative `flock` locks,
sets offline Hugging Face and W&B paths, and records the actual Python executable plus
the complete live package list and its SHA-256 in every run.

## Artifact acquisition and promotion

Downloads use `hf-mirror.com` only inside the acquisition command and require an exact
40-character revision:

```bash
piper-artifact-fetch \
  --repo owner/model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --type model
```

After review, only the administrator promotes the result:

```bash
sudo piper-artifact-promote \
  --manifest /workspace/piper/staging/<user>/models/<artifact>/piper-artifact-manifest.json
```

Promotion re-hashes the source and copy, refuses symlinks and overwrites, and makes the
new release root-owned and group-read-only. Formal training uses promoted local paths
with `HF_HUB_OFFLINE=1`.

## Decision authority and acceptance

The administrator has final authority for merges, Python package changes, optional
base-image publication, deployment, rollback, and artifact promotion. Collaborator
review is welcome but is not an approval gate. Required CI checks still apply to code
changes.

The exact account, shared-Python, GPU, logging, artifact, and restart checks are in
`acceptance/README.md`.
