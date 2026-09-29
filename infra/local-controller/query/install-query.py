"""Install new fixed query files; does not alter CAN, motion launcher or dependencies."""

import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import tempfile
import fcntl

ENTRY = Path("/usr/local/sbin/piper-query")
WORKER = Path("/usr/local/libexec/piper-query.py")
CONFIG = Path("/etc/piper-outcome-stack/query-access.json")
SUDOERS = Path("/etc/sudoers.d/piper-query")


MODES = ("link", "status", "firmware", "limits", "acceleration")


def sudoers_text(account, modes=MODES):
    if not re.fullmatch(r"[a-z_][a-z0-9_.-]*\$?", account):
        raise ValueError("administrator name cannot be represented by this sudoers entry")
    return "# Fixed PiPER diagnostics only; no arbitrary command or environment.\n" + "".join(
        f"{account} ALL=(root) NOPASSWD: NOSETENV: {ENTRY} {mode}\n" for mode in modes
    )


def secure_directory(path):
    if not path.exists():
        secure_directory(path.parent)
        path.mkdir(mode=0o755)
    for p in (path.resolve(), *path.resolve().parents):
        st = p.stat()
        if st.st_uid != 0 or st.st_mode & 0o022:
            raise RuntimeError(f"privileged installation parent is writable by non-root: {p}")


def update_payloads(source, account, user):
    """Validate this exact existing installation before preparing its bounded update."""
    cfg = json.loads(CONFIG.read_text())
    if (cfg["administrator"], cfg["uid"], cfg["gid"]) != (account, user.pw_uid, user.pw_gid):
        raise RuntimeError("installed administrator binding differs; not changing identity")
    old_worker, old_rules = WORKER.read_bytes(), SUDOERS.read_bytes()
    if old_worker != (source / "previous-worker.txt").read_bytes():
        raise RuntimeError("installed query worker differs from reviewed update baseline")
    if ENTRY.read_bytes() != (source / "piper-query").read_bytes():
        raise RuntimeError("installed launcher differs; not overwriting it")
    if old_rules.decode() != sudoers_text(account, MODES[:-1]):
        raise RuntimeError("existing sudoers differs from the four-command baseline")
    return cfg, old_worker, old_rules


def replace_file(path, data, mode):
    with tempfile.TemporaryDirectory(prefix=".piper-query-", dir=path.parent) as temporary:
        pending = Path(temporary) / "pending"
        pending.write_bytes(data)
        pending.chmod(mode)
        os.replace(pending, path)


def update_existing(source, account, user):
    for path in (ENTRY, WORKER, CONFIG, SUDOERS):
        secure_directory(path.parent)
        if path.is_symlink() or path.stat().st_uid != 0 or path.stat().st_mode & 0o022:
            raise RuntimeError(f"installed privileged file has unexpected ownership/mode: {path}")
    with CONFIG.open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, old_worker, old_rules = update_payloads(source, account, user)
        rules = sudoers_text(account).encode()
        with tempfile.TemporaryDirectory(prefix=".piper-query-", dir=SUDOERS.parent) as temporary:
            pending = Path(temporary) / "rule"
            pending.write_bytes(rules)
            pending.chmod(0o440)
            subprocess.run(["/usr/sbin/visudo", "-cf", str(pending)], check=True)
        try:
            replace_file(WORKER, (source / "piper-query.py").read_bytes(), 0o644)
            replace_file(SUDOERS, rules, 0o440)
            subprocess.run(["/usr/sbin/visudo", "-c"], check=True)
        except BaseException:
            replace_file(WORKER, old_worker, 0o644)
            replace_file(SUDOERS, old_rules, 0o440)
            raise
    print(
        json.dumps(
            dict(
                status="updated",
                operations=MODES,
                administrator=account,
                interface=cfg["interface"],
                binding_preserved=True,
                can_reconfigured=False,
                device_queries_sent=False,
            ),
            indent=2,
        )
    )


def main():
    update = len(sys.argv) == 3 and sys.argv[2] == "--update"
    if os.geteuid() != 0 or (not update and len(sys.argv) != 4):
        raise RuntimeError("use install-piper-query.sh from the administrator terminal")
    source = Path(sys.argv[1])
    account = os.environ["SUDO_USER"]
    user = pwd.getpwnam(account)
    if user.pw_uid == 0 or str(user.pw_uid) != os.environ.get("SUDO_UID"):
        raise RuntimeError("administrator identity mismatch")
    rules = sudoers_text(account)
    for executable in ("/usr/sbin/ip", "/usr/bin/nsenter", "/usr/bin/setpriv", "/usr/sbin/visudo"):
        if not os.access(executable, os.X_OK):
            raise RuntimeError(f"missing existing system tool: {executable}; no automatic install")
    subprocess.run(["/usr/sbin/visudo", "-c"], check=True)
    if update:
        return update_existing(source, account, user)
    interface, expected_serial = sys.argv[2], sys.argv[3]
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", interface) or not expected_serial:
        raise ValueError("explicit verified interface and serial required")
    paths = (ENTRY, WORKER, CONFIG, SUDOERS)
    if any(p.exists() or p.is_symlink() for p in paths):
        raise RuntimeError("query installation already exists; inspect it instead of overwriting")
    spec = importlib.util.spec_from_file_location(
        "fixed_query_install_source", source / "piper-query.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = module.interface_info(interface, namespace=True)
    usb = module.usb_identity(info)
    if usb["serial"] != expected_serial or (usb["idVendor"], usb["idProduct"]) != ("1d50", "606f"):
        raise RuntimeError("the interface is not the verified factory USB-CAN adapter")
    cfg = {
        "administrator": account,
        "uid": user.pw_uid,
        "gid": user.pw_gid,
        "interface": interface,
        "usb": usb,
    }
    for path in paths:
        secure_directory(path.parent)
    created = []
    try:
        for destination, original, mode in (
            (WORKER, source / "piper-query.py", 0o644),
            (ENTRY, source / "piper-query", 0o755),
        ):
            with destination.open("xb") as out:
                created.append(destination)
                out.write(original.read_bytes())
            os.chown(destination, 0, 0)
            os.chmod(destination, mode)
        with CONFIG.open("x") as out:
            created.append(CONFIG)
            json.dump(cfg, out, indent=2)
        os.chown(CONFIG, 0, 0)
        os.chmod(CONFIG, 0o600)
        subprocess.run(
            [
                "/usr/bin/setpriv",
                "--reuid=" + str(user.pw_uid),
                "--regid=" + str(user.pw_gid),
                "--clear-groups",
                "/usr/bin/test",
                "-r",
                str(WORKER),
            ],
            check=True,
        )
        with tempfile.TemporaryDirectory(prefix=".piper-query-", dir=SUDOERS.parent) as temporary:
            pending = Path(temporary) / "rule"
            pending.write_text(rules)
            pending.chmod(0o440)
            subprocess.run(["/usr/sbin/visudo", "-cf", str(pending)], check=True)
            os.replace(pending, SUDOERS)
            created.append(SUDOERS)
        subprocess.run(["/usr/sbin/visudo", "-c"], check=True)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    print(
        json.dumps(
            {
                "status": "installed",
                "administrator": account,
                "interface": interface,
                "usb_serial": usb["serial"],
                "operations": list(MODES),
                "can_reconfigured": False,
                "device_queries_sent": False,
                "entry": str(ENTRY),
                "sudoers": str(SUDOERS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
