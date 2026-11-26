#!/usr/bin/env bash
set -euo pipefail

# install_laptop_battery_guard.sh
#
# SCOPE: workstation-side, idempotent install of the low-battery clean-shutdown
# guard onto the verify host leo-laptop, over ssh. Usable as the
# before_done.script of a dark_factory deterministic deploy task.
#
# WHY IT EXISTS (task 9051): leo-laptop died uncleanly on 2026-11-08 and
# 2026-11-10 when merge-verify load drained its battery on a USB-C supply that
# had dropped to 30 W. AC read online=1 throughout, so UPower's critical action
# never fired. The guard (scripts/laptop_battery_guard.py) reads BAT* alone.
#
# It is a SYSTEM timer run as root: leo has no lingering, so a user timer would
# stop with the last session, and root can power off without a polkit grant.
#
# ORDER OF OPERATIONS (each step aborts non-zero on any problem)
#   1. The host answers ssh (BatchMode).
#   2. Read-only preflight: passwordless sudo (sudo -n) and the interpreter
#      the committed unit's ExecStart names, its only home.
#   3. The repo's guard runs on that interpreter with --dry-run and prints
#      the current reading. --check / --dry-run stops here.
#   4. Copy the guard and both units to a staging dir in the host user's
#      ~/.cache (never into a checkout).
#   5. sudo -n install them into /usr/local/lib/laptop-battery-guard and
#      /etc/systemd/system, backing up a differing unit first; daemon-reload,
#      enable --now the timer, confirm enabled + active, and dry-run the
#      INSTALLED guard. systemctl edit drop-ins are never touched.
#   6. Run the service once and require Result=success, which proves its
#      ExecStart, root and any drop-in environment together. A battery
#      already Discharging at the floor powers the host off here, as the
#      timer's first tick would.
#
# Usage:
#   install_laptop_battery_guard.sh [--check|--dry-run]
#
# Env overrides:
#   SSH             ssh invocation prefix
#                   (default: "ssh -o BatchMode=yes -o ConnectTimeout=15")
#   LAPTOP_HOST     target host (default: leo-laptop)
#   BACKUP_LABEL    suffix for unit backups (default: a timestamp)
#   REMOTE_DESTDIR  DESTDIR-style prefix for every install path on the host
#                   (default: empty); a test seam

SSH="${SSH:-ssh -o BatchMode=yes -o ConnectTimeout=15}"
LAPTOP_HOST="${LAPTOP_HOST:-leo-laptop}"
BACKUP_LABEL="${BACKUP_LABEL:-$(date +%Y%m%d%H%M%S)}"
REMOTE_DESTDIR="${REMOTE_DESTDIR:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD_SRC="$SCRIPT_DIR/../laptop_battery_guard.py"
UNIT="laptop-battery-guard"
SERVICE_SRC="$SCRIPT_DIR/$UNIT.service"
LIB_DIR="/usr/local/lib/laptop-battery-guard"
UNIT_DIR="/etc/systemd/system"
STAGE_REL=".cache/laptop-battery-guard-staging"

MODE="apply"
for arg in "$@"; do
    case "$arg" in
        --check|--dry-run) MODE="check" ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            echo "Usage: $(basename "$0") [--check|--dry-run]" >&2
            exit 2
            ;;
    esac
done

log() { echo "[install_laptop_battery_guard] $*"; }
die() { echo "[install_laptop_battery_guard] ERROR: $*" >&2; exit 1; }

HOST_PYTHON="$(sed -n 's|^ExecStart=\(/[^ ]*\) .*|\1|p' "$SERVICE_SRC")"
[ -n "$HOST_PYTHON" ] || die "$SERVICE_SRC: ExecStart= names no absolute interpreter"
printf -v host_python_q '%q' "$HOST_PYTHON"

# ── 1. reachability ───────────────────────────────────────────────────────
$SSH "$LAPTOP_HOST" true || die "ssh $LAPTOP_HOST failed (BatchMode)"

# ── 2. read-only preflight ────────────────────────────────────────────────
# Single-quoted heredoc: expanded on the HOST.
# Positional arg: $1 HOST_PYTHON
$SSH "$LAPTOP_HOST" "bash -s -- $host_python_q" <<'PREFLIGHT_EOF' \
    || die "preflight failed on $LAPTOP_HOST"
set -euo pipefail
python="$1"
sudo -n true 2>/dev/null || {
    echo "ERROR: passwordless sudo (sudo -n) is unavailable for $(id -un); nothing was changed" >&2
    exit 1
}
[ -x "$python" ] || {
    echo "ERROR: $python, the unit's ExecStart interpreter, is missing; nothing was changed" >&2
    exit 1
}
echo "preflight OK: passwordless sudo, $("$python" --version 2>&1) at $python"
PREFLIGHT_EOF

# ── 3. the repo's guard runs on the unit's interpreter ────────────────────
$SSH "$LAPTOP_HOST" "$host_python_q - --dry-run" <"$GUARD_SRC" \
    || die "the guard does not run on $LAPTOP_HOST $HOST_PYTHON"

if [ "$MODE" = "check" ]; then
    log "check mode: would install laptop_battery_guard.py into $REMOTE_DESTDIR$LIB_DIR"
    log "  and $UNIT.{service,timer} into $REMOTE_DESTDIR$UNIT_DIR on $LAPTOP_HOST,"
    log "  then enable --now $UNIT.timer. Nothing changed."
    exit 0
fi

# ── 4. copy to a staging dir on the host ──────────────────────────────────
tar -cf - -C "$SCRIPT_DIR/.." laptop_battery_guard.py -C "$SCRIPT_DIR" "$UNIT.service" "$UNIT.timer" \
    | $SSH "$LAPTOP_HOST" "rm -rf $STAGE_REL && mkdir -p $STAGE_REL && tar -xf - -C $STAGE_REL" \
    || die "copying the guard and units to $LAPTOP_HOST failed"

# ── 5-6. install, enable, confirm, dry-run, first run ─────────────────────
# ssh joins its arguments into one remote command line, so each is quoted:
# an empty REMOTE_DESTDIR would otherwise vanish and shift the rest.
printf -v apply_args '%q ' "$UNIT" "$REMOTE_DESTDIR" "$LIB_DIR" "$UNIT_DIR" "$BACKUP_LABEL" \
    "$STAGE_REL" "$HOST_PYTHON"
# Single-quoted heredoc: expanded on the HOST. Positional args:
#   $1 UNIT  $2 REMOTE_DESTDIR  $3 LIB_DIR  $4 UNIT_DIR  $5 BACKUP_LABEL  $6 STAGE_REL
#   $7 HOST_PYTHON
$SSH "$LAPTOP_HOST" "bash -s -- $apply_args" <<'APPLY_EOF' \
    || die "install on $LAPTOP_HOST failed (see output above)"
set -euo pipefail
unit="$1"; lib="$2$3"; ud="$2$4"; label="$5"; stage="$HOME/$6"; python="$7"
sudo -n mkdir -p "$lib" "$ud"

for f in "$unit.service" "$unit.timer"; do
    if [ -f "$ud/$f" ] && ! cmp -s "$stage/$f" "$ud/$f"; then
        sudo -n cp "$ud/$f" "$lib/$f.bak-$label"
        echo "backed up the differing $ud/$f to $lib/$f.bak-$label"
    fi
    sudo -n install -m 0644 "$stage/$f" "$ud/$f"
done
sudo -n install -m 0755 "$stage/laptop_battery_guard.py" "$lib/laptop_battery_guard.py"
rm -rf "$stage"
echo "installed $unit.{service,timer} into $ud and laptop_battery_guard.py into $lib"

sudo -n systemctl daemon-reload
sudo -n systemctl enable --now "$unit.timer"
[ "$(systemctl is-enabled "$unit.timer")" = "enabled" ] \
    || { echo "ERROR: $unit.timer is not enabled" >&2; exit 1; }
[ "$(systemctl is-active "$unit.timer")" = "active" ] \
    || { echo "ERROR: $unit.timer is not active" >&2; exit 1; }
echo "$unit.timer enabled and active"

"$python" "$lib/laptop_battery_guard.py" --dry-run \
    || { echo "ERROR: the installed guard failed its dry run" >&2; exit 1; }

first_run_failed() {
    sudo -n journalctl -u "$unit.service" -n 40 --no-pager >&2 || true
    echo "ERROR: $unit.service $1" >&2
    exit 1
}
# A oneshot start returns once the run ends, including a run the timer began.
sudo -n systemctl start "$unit.service" || first_run_failed "failed its first run"
result="$(systemctl show -p Result --value "$unit.service")"
[ "$result" = "success" ] || first_run_failed "Result=$result"
echo "$unit.service ran as root under its unit environment: Result=success"
APPLY_EOF

log "done: $UNIT.timer installed and verified on $LAPTOP_HOST"
