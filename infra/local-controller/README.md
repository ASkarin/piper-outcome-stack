# Local controller host

This directory defines the administrator-controlled baseline for `piper-local`. It does
not contain a real identity, network address, private key, or guessed device rule.

There are two human roles:

- the unique highest-privilege administrator owns deployment and raw PiPER, CAN, camera,
  input, and driver access;
- collaborators use independent accounts and clones, cannot sudo, cannot modify an
  immutable release, cannot enter the real-CAN namespace, and cannot open the target
  camera or input devices. Host-namespace `vcan` remains available for software tests.

No secondary operator role, hardware-service role, dedicated runtime account, or
resident robot-control service is part of the architecture.

## Administrator workflow

Run from a physical console or a verified Tailscale SSH session:

```bash
sudo bash infra/local-controller/bootstrap-host.sh base
sudo bash infra/local-controller/bootstrap-host.sh gpu
sudo --preserve-env=SSH_CONNECTION PIPER_CONFIRM_CONSOLE=YES \
  bash infra/local-controller/bootstrap-host.sh security
sudo --preserve-env=SSH_CONNECTION \
  bash infra/local-controller/bootstrap-host.sh confirm-security
sudo bash infra/local-controller/bootstrap-host.sh check
```

After review, deploy a clean committed checkout as an immutable release:

```bash
sudo --preserve-env=PIPER_PYPI_MIRROR \
  /usr/local/sbin/piper-local-deploy-release install <clean-source-checkout>
```

The release sync installs the workspace `lerobot_robot_outcome_piper` distribution and
the public HTTPS, commit-pinned LeRobot and pyAgxArm dependencies from `uv.lock`. It does
not require a deploy key, SSH wrapper, private adapter repository, or source copy outside
the immutable release. The administrator develops in a personal clone, but real control, data
collection, and policy execution use the immutable release. Run Python as the
administrator, not root; use sudo only for drivers, udev/ACL, SocketCAN, and deployment
administration.

The source archive is moved to its final commit-scoped path before `.venv` is created,
because virtual-environment launchers contain absolute interpreter paths and are not
relocatable. Installation, acceptance, and activation are separate commands. Acceptance
creates a temporary environment from the same frozen `uv.lock` with the development test
group; pytest never enters the `--no-dev` runtime. Activation verifies the commit-scoped
acceptance summary, its SHA-256-bound marker, and the accepted lockfile digest. If
installation fails before that marker, the incomplete release is moved back to its
hidden `.installing` path for explicit recovery.

`PIPER_PYPI_MIRROR` may point to an HTTPS package index when the locked registry CDN is
unusable and is explicitly preserved through sudo by the command above. Git
dependencies and pinned PyTorch wheels remain on their locked sources.

## Collaborator lifecycle

No placeholder account is created. Provision exactly the restricted collaborator
role after its individual tailnet Grant is confirmed:

```bash
sudo PIPER_TAILNET_GRANT_CONFIRMED=YES \
  bash infra/local-controller/manage-collaborator.sh \
  provision <account> <public-key-file>
```

Revocation preserves the home directory and archives the old authorized-key file:

```bash
sudo PIPER_TAILNET_GRANT_REVOKED=YES \
  bash infra/local-controller/manage-collaborator.sh revoke <account>
```

## Deferred device rules

Until exact hardware arrives, real CAN, D435, AR0234, and Xbox access remains
`not_checked`. Do not prefill an interface, bitrate, device node, or udev match.

SocketCAN is a network interface, so Unix device groups and udev node ACLs cannot keep
a collaborator from binding it. After the exact adapter/interface and bitrate are
verified, keep the interface down and move it into the fixed `piper-can` network
namespace:

```bash
sudo piper-socketcan isolate <verified-can-interface>
sudo PIPER_CAN_INTERFACE=<verified-can-interface> \
  PIPER_COLLABORATOR_ACCOUNT=<collaborator-account> \
  bash infra/local-controller/bootstrap-host.sh check
sudo piper-socketcan up <verified-can-interface> <verified-bitrate>
sudo piper-socketcan exec -- \
  /opt/piper-outcome-stack/current/.venv/bin/piper-outcome-stack teleoperate ...
```

The launcher enters the namespace and then runs LeRobot as the administrator's normal
UID/GID with zero effective capabilities. It does not create a service or a runtime
account. It also does not bring CAN down when LeRobot exits: automatic communication
loss is not a verified robot stop action. Once the robot is in the separately verified
safe state and no namespace process remains, the administrator may explicitly run
`sudo piper-socketcan down <verified-can-interface>`.

The host namespace must not contain the real interface while a session is accepted.
`bootstrap-host.sh base` installs an inactive systemd `Type=oneshot` isolation template;
it is not a resident control service and does nothing until hardware identity is frozen.
After arrival, create one udev rule that matches the exact adapter vendor, product, and
serial and requests `piper-socketcan-isolate@<interface>.service`. Do not install a broad
CAN rule. Verify an unplug/replug: the oneshot must move the interface while it is down,
exit, and leave it absent from the host namespace. Until that exact rule and hotplug test
pass, every reboot/re-enumeration resets the permission result to `not_checked`, and the
administrator must manually isolate and rerun the positive/negative doctor before CAN
use. Device groups or exact udev ACLs remain appropriate only for the D435, AR0234, and
Xbox nodes. Device access does not authorize motion: the five hardware and action flags
remain false until the physical safety gates have machine evidence.

The namespace has no veth/NAT. Real `record` sessions must pass
`--dataset.push_to_hub=false`; upload the finalized dataset afterwards from the host
namespace.
