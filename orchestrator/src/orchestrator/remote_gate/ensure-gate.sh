#!/usr/bin/env bash
# Ensure an immutable, handshake-passing gate checkout of $SHA exists under
# $GATE_ROOT, then print {"built": <bool>, "identity": <gate-identity JSON>}.
#
# Task 6565, design D2. Contract (orchestrator/remote_gate.py::ensure_script_text):
#   GATE_ROOT  absolute directory holding one checkout per sha
#   SHA        40-hex commit to run; the dispatcher has pushed it to
#              $SOURCE as refs/gate/$SHA
#   SOURCE     the remote repo the checkout is cloned from
#   PATH       already carries uv and git; `orchestrator` is the host shim,
#              which runs "${ORCH_GATE_BIN:-<shared tree>}"
#
# Invariant: a dir carrying .ready has answered `gate-identity` through its own
# binary. Builds happen in place (a moved uv venv is dead: its shebangs and
# editable .pth entries are absolute) under a per-sha build lock; after .ready
# nothing writes into the dir. stdout carries only the final JSON line; every
# diagnostic goes to stderr. Exit status 255 is the ssh client's own failure
# (remote_gate.py::classify_remote_exit), so this script never exits 255.
set -euo pipefail
trap '[[ $? -ne 255 ]] || exit 70' EXIT

die() {
	echo "ensure-gate: $1" >&2
	exit "${2:-1}"
}

[[ ${GATE_ROOT:-} == /* ]] || die "GATE_ROOT must be an absolute path, got '${GATE_ROOT:-}'" 64
[[ ${SHA:-} =~ ^[0-9a-f]{40}$ ]] || die "SHA must be 40 hex digits, got '${SHA:-}'" 64
[[ -n ${SOURCE:-} ]] || die "SOURCE must name the source repository" 64

dir="$GATE_ROOT/$SHA"
gate_bin="$dir/.venv/bin/orchestrator"

mkdir -p "$GATE_ROOT"
touch "$GATE_ROOT/.dark-factory-gate-root"

to_trash() {
	mv -T "$1" "$(mktemp -u -d "$GATE_ROOT/.trash-XXXXXXXX")"
}

build() {
	git clone --quiet --local --no-checkout "$SOURCE" "$dir" >&2
	git -C "$dir" checkout --quiet --detach "$SHA" >&2
	(cd "$dir" && uv sync --all-packages --frozen) >&2
}

handshake() {
	ORCH_GATE_BIN="$gate_bin" orchestrator gate-identity
}

exec 9>"$GATE_ROOT/.build-$SHA.lock"
flock -w 300 9 || die "build lock for $SHA not acquired within 300s" 75

built=false
if [[ -e $dir && ! -e $dir/.ready ]]; then
	echo "ensure-gate: $dir is not ready; moving it to trash" >&2
	to_trash "$dir"
fi
if [[ ! -e $dir ]]; then
	build
	built=true
fi
if ! identity=$(handshake); then
	[[ $built == false ]] || die "fresh gate $dir failed its handshake" 70
	echo "ensure-gate: ready gate $dir failed its handshake; rebuilding once" >&2
	to_trash "$dir"
	build
	built=true
	identity=$(handshake) || die "rebuilt gate $dir failed its handshake" 70
fi
touch "$dir/.ready"
# The last-used stamp the reaper ages: every ensure refreshes it.
touch -c "$dir"
exec 9>&-

# The clone's detached HEAD keeps the objects; the transfer ref is spent.
git -C "$SOURCE" update-ref -d "refs/gate/$SHA" ||
	echo "ensure-gate: warning: could not delete refs/gate/$SHA in $SOURCE" >&2

printf '{"built": %s, "identity": %s}\n' "$built" "$identity"
