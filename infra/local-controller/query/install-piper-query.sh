#!/bin/bash
set -euo pipefail
[[ $EUID -eq 0 && -n ${SUDO_USER:-} && ${SUDO_USER} != root && -t 0 ]] || {
  echo 'Run this installer once using sudo in the administrator terminal.' >&2
  exit 1
}
[[ $# -eq 2 || ( $# -eq 1 && $1 == --update ) ]] || {
  echo 'usage: install-piper-query.sh <verified-interface> <verified-usb-serial> | --update' >&2; exit 1;
}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  SUDO_USER="$SUDO_USER" SUDO_UID="$SUDO_UID" \
  /usr/bin/python3 -I -S "$source_dir/install-query.py" "$source_dir" "$@"
