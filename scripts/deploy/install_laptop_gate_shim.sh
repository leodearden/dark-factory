#!/usr/bin/env bash
set -euo pipefail

# install_laptop_gate_shim.sh
#
# SCOPE: workstation-side, idempotent install over ssh of the one-line
# ORCH_GATE_BIN edit into leo-laptop's /usr/local/bin/orchestrator wrapper
# (task 6565, design plans/6565-remote-checkout-sync-design-2026-10-20.md D3).
# It is the before_done.script of a deterministic deploy task that 6565's
# carrier depends on, so the DeterministicRunner runs it once. Any non-zero
# exit makes the runner file a born-at-L2 escalation; it never re-runs it.
#
# THE EDIT: the wrapper's last exec line
#     /home/leo/src/dark-factory/.venv/bin/orchestrator "$@" 2> >(tee -a "$_LOG" >&2)
# becomes
#     "${ORCH_GATE_BIN:-/home/leo/src/dark-factory/.venv/bin/orchestrator}" "$@" 2> >(tee -a "$_LOG" >&2)
# Inert until ORCH_GATE_BIN is set: unset, the wrapper runs the same binary it
# always did. The PATH export, the ruled 600 s watchdog hold (task 5577) and the
# START/EXIT invocation log stay in the wrapper, and the line stays a
# subshell-form call (never exec) so the EXIT log line keeps being written.
#
# WRITE PATH: /usr/local/bin is root-owned, so the file cannot be replaced by
# rename without privilege. Overwriting it in place is NOT acceptable: a
# running wrapper is a bash script read incrementally by a live verify, and the
# edit lengthens a line it has already passed. The new content is therefore
# built aside and renamed over the wrapper (atomic; a running verify keeps the
# old inode), directly when the directory is writable, else through
# passwordless `sudo -n`. With neither, the script states so, prints the one
# line to apply by hand, writes nothing and exits non-zero.
#
# Usage:
#   install_laptop_gate_shim.sh [--check]
#     (no flag)  apply; an already-updated wrapper is a no-op that exits 0
#     --check    read-only contract check; exits 0 iff the installed wrapper
#                carries the edit and, with ORCH_GATE_BIN=/bin/echo, runs that
#                binary (its only side effect is the wrapper's own log line)
#
# Env overrides:
#   SSH           ssh invocation prefix
#                 (default: "ssh -o BatchMode=yes -o ConnectTimeout=15")
#   LAPTOP_HOST   target host (default: leo-laptop)
#   SHIM_PATH     the wrapper ON THE HOST (default: /usr/local/bin/orchestrator)
#   BACKUP_LABEL  suffix of the <wrapper>.bak-<label> backup (default: a timestamp)

SSH="${SSH:-ssh -o BatchMode=yes -o ConnectTimeout=15}"
LAPTOP_HOST="${LAPTOP_HOST:-leo-laptop}"
SHIM_PATH="${SHIM_PATH:-/usr/local/bin/orchestrator}"
BACKUP_LABEL="${BACKUP_LABEL:-$(date +%Y%m%d%H%M%S)}"

MODE="apply"
for arg in "$@"; do
    case "$arg" in
        --check) MODE="check" ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            echo "Usage: $(basename "$0") [--check]" >&2
            exit 2
            ;;
    esac
done

die() { echo "[install_laptop_gate_shim] ERROR: $*" >&2; exit 1; }

$SSH "$LAPTOP_HOST" true || die "ssh $LAPTOP_HOST failed (BatchMode)"

# Single-quoted heredoc: expanded on the HOST. $1 wrapper path, $2 mode, $3 label.
$SSH "$LAPTOP_HOST" bash -s -- "$SHIM_PATH" "$MODE" "$BACKUP_LABEL" <<'REMOTE_EOF'
set -euo pipefail
shim="$1"; mode="$2"; label="$3"
tag="[install_laptop_gate_shim@$(hostname)]"
say() { echo "$tag $*"; }
fail() { echo "$tag ERROR: $*" >&2; exit 1; }

OLD_LINE='/home/leo/src/dark-factory/.venv/bin/orchestrator "$@" 2> >(tee -a "$_LOG" >&2)'
NEW_LINE='"${ORCH_GATE_BIN:-/home/leo/src/dark-factory/.venv/bin/orchestrator}" "$@" 2> >(tee -a "$_LOG" >&2)'
export OLD_LINE NEW_LINE
HAND_EDIT="sudo sed -i 's|^/home/leo/src/dark-factory/.venv/bin/orchestrator \"\$@\"|\"\${ORCH_GATE_BIN:-/home/leo/src/dark-factory/.venv/bin/orchestrator}\" \"\$@\"|' $shim"

count_line() { grep -cxF -- "$1" "$shim" || true; }

[ -f "$shim" ] || fail "$shim does not exist"
[ -r "$shim" ] || fail "$shim is not readable"

# 1. the wrapper still carries the host facts this edit must not disturb
for marker in 'export PATH=' 'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS:-600' '_LOG=' ' START ' ' EXIT rc='; do
    grep -qF -- "$marker" "$shim" || fail "$shim lost '$marker'; refusing to touch an unrecognised wrapper"
done

has_new="$(count_line "$NEW_LINE")"
has_old="$(count_line "$OLD_LINE")"

probe() {
    local out
    out="$(ORCH_GATE_BIN=/bin/echo "$shim" probe 2>/dev/null)" || fail "wrapper failed when run with ORCH_GATE_BIN=/bin/echo"
    [ "$out" = "probe" ] || fail "wrapper ignores ORCH_GATE_BIN (ran something other than /bin/echo; output: '$out')"
}

# 2. read-only contract check
if [ "$has_new" = 1 ] && [ "$has_old" = 0 ]; then
    probe
    say "OK: $shim honours ORCH_GATE_BIN"
    exit 0
fi
if [ "$mode" = check ]; then
    fail "$shim does not honour ORCH_GATE_BIN (run scripts/deploy/install_laptop_gate_shim.sh to install)"
fi
[ "$has_new" = 0 ] && [ "$has_old" = 1 ] \
    || fail "$shim exec line is not recognised (old line x$has_old, new line x$has_new); edit by hand"

# 3. how to write: rename into the wrapper's directory, directly or via sudo -n
dir="$(dirname "$shim")"
if [ -w "$dir" ]; then
    priv=()
elif sudo -n true 2>/dev/null; then
    priv=(sudo -n)
else
    fail "$shim is not writable: $dir is not writable by $(id -un) and passwordless sudo is unavailable. Nothing was written. Apply by hand: $HAND_EDIT"
fi

owner="$(stat -c %U "$shim")"; group="$(stat -c %G "$shim")"; perm="$(stat -c %a "$shim")"
staged="$(mktemp)"
trap 'rm -f "$staged"' EXIT
awk 'ENVIRON["OLD_LINE"] == $0 { print ENVIRON["NEW_LINE"]; next } { print }' "$shim" > "$staged"
bash -n "$staged" || fail "edited wrapper does not parse; nothing was written"
[ "$(grep -cxF -- "$NEW_LINE" "$staged")" = 1 ] || fail "edit did not apply exactly once; nothing was written"

# 4. backup, then atomic rename
backup="$shim.bak-$label"
"${priv[@]}" cp -p -- "$shim" "$backup"
"${priv[@]}" install -o "$owner" -g "$group" -m "$perm" "$staged" "$dir/.$(basename "$shim").new.$$"
"${priv[@]}" mv -f -- "$dir/.$(basename "$shim").new.$$" "$shim"
say "installed; previous wrapper kept at $backup"

# 5. post-check; restore the backup if the installed wrapper misbehaves
if ! ( probe ); then
    "${priv[@]}" cp -p -- "$backup" "$dir/.$(basename "$shim").restore.$$"
    "${priv[@]}" mv -f -- "$dir/.$(basename "$shim").restore.$$" "$shim"
    fail "post-install probe failed; restored $backup"
fi
say "OK: $shim honours ORCH_GATE_BIN"
REMOTE_EOF
