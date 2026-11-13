#!/usr/bin/env bash
# Provision a fresh Ubuntu 24.04 box as a dark-factory verify host for the
# benchmark: apt basics, uv, Node 22, rustup. Idempotent; skips what exists.
# Works as root without sudo (a container) and as a sudoer.
set -euo pipefail

log() { echo "[provision $(date -u +%T)] $*"; }
. "$(dirname "$0")/lib.sh"

if command -v apt-get >/dev/null; then
	log "apt packages"
	apt_noninteractive update -qq
	apt_noninteractive install -y -qq "${DF_APT_PACKAGES[@]}" >/dev/null
fi

py_minor=$(python3 -c 'import sys; print(sys.version_info[1])')
((py_minor >= 11)) || log "WARNING: system python3 is 3.$py_minor; the DF lint leg wants >= 3.11 (use Ubuntu 24.04)"

install_user_toolchains
log "done"
