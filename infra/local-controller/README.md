# Local controller host

This directory defines the administrator-controlled baseline for `a3-local`. It does
not contain a real identity, network address, private key, or guessed device rule.

There are two human roles:

- the unique highest-privilege administrator owns deployment and raw A3, CAN, camera,
  input, and driver access;
- collaborators use independent accounts and clones, cannot sudo, cannot modify an
  immutable release, and cannot open raw devices.

No ordinary-administrator, operator, hardware-service, or dedicated runtime account
is part of the current architecture. Existing `a3-operator` and `a3-hardware` groups
may remain on an already provisioned host as inactive compatibility residue; the
bootstrap does not create them or assign people to them. The old `a3-local-control.service`, if
already installed, must remain disabled/inactive until an explicitly authorized
administrator migration removes it. This source tree no longer installs that unit.

## Administrator workflow

Run from a physical console or a verified Tailscale SSH session:

```bash
sudo bash infra/local-controller/bootstrap-host.sh base
sudo bash infra/local-controller/bootstrap-host.sh gpu
sudo --preserve-env=SSH_CONNECTION A3_CONFIRM_CONSOLE=YES \
  bash infra/local-controller/bootstrap-host.sh security
sudo --preserve-env=SSH_CONNECTION \
  bash infra/local-controller/bootstrap-host.sh confirm-security
sudo bash infra/local-controller/bootstrap-host.sh check
```

After review, deploy a clean committed checkout as an immutable release:

```bash
sudo --preserve-env=SSH_AUTH_SOCK,A3_PYPI_MIRROR \
  /usr/local/sbin/a3-local-deploy-release install <clean-source-checkout>
```

The release sync installs the auto-discoverable `lerobot_robot_a3` distribution from
its private repository at the full commit in `uv.lock`. Before invoking sudo, the
administrator must use a temporary forwarded SSH agent that can read that repository;
the deployment fails closed when `SSH_AUTH_SOCK` is absent or unusable. Git access is
non-interactive and uses a temporary `known_hosts` file with the repository's pinned
GitHub ED25519 host key and verified fingerprint; it never accepts an unknown key. No
deploy key is stored on `a3-local`. The administrator develops in a personal clone,
but real control, data collection, and policy execution use the immutable release.
Run Python as the administrator, not root; use sudo only for drivers, udev/ACL,
SocketCAN, and deployment administration.

The source archive is moved to its final commit-scoped path before `.venv` is created,
because virtual-environment launchers contain absolute interpreter paths and are not
relocatable. Activation still occurs only after the completion marker is written. If
installation fails before that marker, the incomplete release is moved back to its
hidden `.installing` path for explicit recovery.

`A3_PYPI_MIRROR` may point to an HTTPS package index when the locked registry CDN is
unusable and is explicitly preserved through sudo by the command above. Git
dependencies and pinned PyTorch wheels remain on their locked sources.

## Collaborator lifecycle

No placeholder account is created. Provision exactly the restricted collaborator
role after its individual tailnet Grant is confirmed:

```bash
sudo A3_TAILNET_GRANT_CONFIRMED=YES \
  bash infra/local-controller/manage-collaborator.sh \
  provision <account> <public-key-file>
```

Revocation preserves the home directory and archives the old authorized-key file:

```bash
sudo A3_TAILNET_GRANT_REVOKED=YES \
  bash infra/local-controller/manage-collaborator.sh revoke <account>
```

## Deferred device rules

Until exact hardware arrives, CAN, D435, AR0234, and Xbox access remains
`not_checked`. Test the administrator against the exact nodes without root. Add only
the minimum device group or udev ACL for that same administrator if a test fails; do
not run the whole Python control process as root. Run the inverse test from the
collaborator account. Device access does not authorize motion: the five hardware and
action flags remain false until the physical safety gates have machine evidence.
