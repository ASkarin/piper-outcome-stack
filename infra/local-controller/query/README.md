# Restricted PiPER query access

This optional administrator-installed entry permits noninteractive diagnostics only.
It leaves `piper-socketcan`, CAN configuration, control code, drivers and releases
unchanged. It creates no account or resident service and grants no collaborator access.

## Allowed operations

| Command | Effect |
|---|---|
| `sudo -n /usr/local/sbin/piper-query link` | Read interface metadata; no socket send |
| `sudo -n /usr/local/sbin/piper-query status` | Receive for three seconds; show raw/latest joint, controller, driver and gripper feedback; no send |
| `sudo -n /usr/local/sbin/piper-query firmware` | One fixed `0x4AF#01` request; collect firmware reply |
| `sudo -n /usr/local/sbin/piper-query limits` | One `0x472` angle/speed query per joint, six total; no retry |
| `sudo -n /usr/local/sbin/piper-query acceleration` | One `0x472` search-content=2 query per joint; decode `0x47C` in rad/s²; no writes |

Firmware, limit and acceleration queries are refused when another process is already in `piper-can`.
Queries are serialized against each other. Passive results contain timestamps and may
contain faults or missing groups: they are diagnostics, not synchronized control
observations or evidence that motion is safe. Failures never trigger recovery, stop,
enable/disable, mode changes or movement. Coordinate active queries with other work;
the existing control launcher is unchanged and this is not a new control-session manager.

## One-time installation

Review these four source files and the exact interface/USB serial, then run from the
controller administrator's interactive terminal:

```bash
sudo bash infra/local-controller/query/install-piper-query.sh \
  <verified-interface> <verified-USB-CAN-serial>
```

The interface must already be isolated in `piper-can`. Installation reads its USB
identity, verifies the supplied serial and the factory adapter vendor/product, checks
sudoers syntax, and refuses existing installation files. It performs no device query,
interface UP/DOWN/isolation, dependency installation, motor command or release change.

Installed files:

- `/usr/local/sbin/piper-query`: root-owned fixed shell entry; clean environment.
- `/usr/local/libexec/piper-query.py`: root-owned stdlib-only helper; system Python
  with `-I -S`, no editable project/SDK imports.
- `/etc/piper-outcome-stack/query-access.json`: root-only caller UID/GID and adapter binding.
- `/etc/sudoers.d/piper-query`: only the installing administrator, five literal
  commands with `NOPASSWD: NOSETENV`; no wildcard or arbitrary Python/shell command.

The privileged portion only reads fixed configuration and enters the existing network
namespace. The query worker runs as the configured administrator with no supplementary
groups/capabilities and no-new-privileges. It validates the namespace, current adapter
identity and (before CAN I/O) the existing 1 Mbps/UP configuration. It writes JSON only
to stdout; the unprivileged caller may redirect it into a report file.

After installation, verify `link`, the UID/capability fields, and rejection of an
unsupported operation or extra argument. Passive `status` can then verify reception;
active query verification should wait until other CAN sessions end. Full hardware and
motion gates remain independent. The existing generic `piper-socketcan exec` does not
become passwordless. Do not infer broader permission from a cached sudo password.

## Updating the existing four-command installation

Prepare a reviewed staging copy of these sources plus `previous-worker.txt`, containing
the inspected installed worker. In that staging directory the administrator runs:

```bash
sudo bash install-piper-query.sh --update
```

The update requires an exact match with the reviewed worker, unchanged launcher and
the original four-command sudoers rules. It preserves the existing administrator/USB
binding, takes the existing query lock, validates new sudoers rules, and replaces only
the worker and rules. A failed final sudoers check restores their original bytes.
It performs no device query or CAN change. After installation, separately verify the
new `acceleration` command and argument rejection; do not mark access verified early.

Related software verification only: `pytest tests/test_query_access.py
 tests/test_local_controller_policy.py`, shell syntax and lint on these files. No full
CI or physical motion test is needed for a documentation-only adjustment. Initial
installation, changes to this privileged entry, or removal of its sudoers rule remain
administrator operations. Removing `/etc/sudoers.d/piper-query` revokes this exemption;
do not remove other sudoers entries or change the existing CAN namespace.
