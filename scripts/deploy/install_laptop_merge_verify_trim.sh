#!/usr/bin/env bash
set -euo pipefail

# install_laptop_merge_verify_trim.sh
#
# SCOPE: workstation-side, idempotent install of reify's merge-verify target
# trim (reify task 9246) onto the remote verify host leo-laptop, over ssh.
# It is the before_done.script of a dark_factory deterministic deploy task that
# depends on reify:9246, so the DeterministicRunner runs it once, after 9246
# lands. Any non-zero exit makes the runner file a born-at-L2 escalation and
# block the task; it never re-runs the script.
#
# WHY IT EXISTS: leo-laptop's root fs filled on 2026-10-16 (esc-5509-2) because
# reify's persistent .worktrees/_merge-verify/target keeps every superseded
# cargo hash generation. reify 9246 ships the trim; this installs it. The
# laptop's own /home/leo/src/reify is a DIRTY checkout that is never updated,
# so the trim's files are copied from the WORKSTATION's reify main checkout
# into ~/.local/lib/reify-verify-host/ on the laptop, never into that checkout.
#
# THE CONTRACT WITH reify 9246 (checked in step 1, not trusted to the dep gate):
#   scripts/prune-unconsulted-target.sh        executable, sources nothing
#   scripts/gc-worktree-targets.sh             exempts _merge-verify by name
#   deploy/systemd/reify-merge-verify-trim.service   ExecStart runs
#       %h/.local/lib/reify-verify-host/prune-unconsulted-target.sh and exits 0
#       when it skips because a verify holds the lane lock
#   deploy/systemd/reify-merge-verify-trim.timer
#
# ORDER OF OPERATIONS (each step aborts non-zero on any problem)
#   1. The four deliverables are present in the reify source checkout and
#      meet the contract above.
#   2. The host answers ssh (BatchMode) and passes a read-only preflight: a
#      systemd --user bus, crontab, the _merge-verify target dir, and a mount
#      that tracks atime (the trim reads atime; noatime would make it unsafe).
#      --check / --dry-run stops here and prints the plan.
#   3. Copy the four files to the host (staging dir, then install).
#   4. Repoint the crontab's gc-worktree-targets.sh line from the dirty
#      checkout's copy to the installed one (crontab backed up first).
#   5. daemon-reload, enable --now the timer, confirm enabled + active.
#   6. Run the service once in the foreground and require Result=success;
#      print free space before and after and the service's journal tail.
#
# Usage:
#   install_laptop_merge_verify_trim.sh [--check|--dry-run]
#
# Env overrides:
#   SSH                ssh invocation prefix
#                      (default: "ssh -o BatchMode=yes -o ConnectTimeout=15")
#   LAPTOP_HOST        target host (default: leo-laptop)
#   REIFY_SOURCE_ROOT  workstation reify checkout the files are copied from
#                      (default: /home/leo/src/reify)
#   REMOTE_REIFY_ROOT  reify checkout path ON THE HOST, used to find the
#                      _merge-verify target and the crontab's old script path
#                      (default: /home/leo/src/reify)
#   BACKUP_LABEL       suffix for crontab / unit backups (default: a timestamp)

SSH="${SSH:-ssh -o BatchMode=yes -o ConnectTimeout=15}"
LAPTOP_HOST="${LAPTOP_HOST:-leo-laptop}"
REIFY_SOURCE_ROOT="${REIFY_SOURCE_ROOT:-/home/leo/src/reify}"
REMOTE_REIFY_ROOT="${REMOTE_REIFY_ROOT:-/home/leo/src/reify}"
BACKUP_LABEL="${BACKUP_LABEL:-$(date +%Y%m%d%H%M%S)}"

UNIT="reify-merge-verify-trim"
LIB_REL=".local/lib/reify-verify-host"
TRIM_SCRIPT="scripts/prune-unconsulted-target.sh"
GC_SCRIPT="scripts/gc-worktree-targets.sh"
SERVICE_SRC="deploy/systemd/$UNIT.service"
TIMER_SRC="deploy/systemd/$UNIT.timer"
DELIVERABLES=("$TRIM_SCRIPT" "$GC_SCRIPT" "$SERVICE_SRC" "$TIMER_SRC")

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

log() { echo "[install_laptop_merge_verify_trim] $*"; }
die() { echo "[install_laptop_merge_verify_trim] ERROR: $*" >&2; exit 1; }

# ── 1. deliverables meet the contract ─────────────────────────────────────
for rel in "${DELIVERABLES[@]}"; do
    [ -f "$REIFY_SOURCE_ROOT/$rel" ] \
        || die "reify 9246 deliverable missing: $REIFY_SOURCE_ROOT/$rel"
done
[ -x "$REIFY_SOURCE_ROOT/$TRIM_SCRIPT" ] \
    || die "$TRIM_SCRIPT is not executable in $REIFY_SOURCE_ROOT"
if grep -nE '^[[:space:]]*(source|\.)[[:space:]]' "$REIFY_SOURCE_ROOT/$TRIM_SCRIPT" >&2; then
    die "$TRIM_SCRIPT sources another file (lines above); only the script itself is installed on the host"
fi
grep -qF "%h/$LIB_REL/prune-unconsulted-target.sh" "$REIFY_SOURCE_ROOT/$SERVICE_SRC" \
    || die "$SERVICE_SRC does not run %h/$LIB_REL/prune-unconsulted-target.sh (the installed copy)"
grep -qF "_merge-verify" "$REIFY_SOURCE_ROOT/$GC_SCRIPT" \
    || die "$GC_SCRIPT does not mention _merge-verify; it would still wipe the warm target"
log "deliverables OK in $REIFY_SOURCE_ROOT"

# ── 2. reachability + read-only preflight ─────────────────────────────────
$SSH "$LAPTOP_HOST" true || die "ssh $LAPTOP_HOST failed (BatchMode)"

TARGET_DIR="$REMOTE_REIFY_ROOT/.worktrees/_merge-verify/target"
CRON_OLD="$REMOTE_REIFY_ROOT/$GC_SCRIPT"

# Single-quoted heredoc: expanded on the HOST. $1 = TARGET_DIR.
$SSH "$LAPTOP_HOST" bash -s -- "$TARGET_DIR" <<'PREFLIGHT_EOF' || die "preflight failed on $LAPTOP_HOST"
set -euo pipefail
target="$1"
systemctl --user show-environment >/dev/null 2>&1 \
    || { echo "ERROR: no systemd --user bus" >&2; exit 1; }
command -v crontab >/dev/null || { echo "ERROR: crontab not found" >&2; exit 1; }
[ -d "$target" ] || { echo "ERROR: missing $target" >&2; exit 1; }
opts="$(findmnt -no OPTIONS --target "$target")"
case ",$opts," in
    *,noatime,*) echo "ERROR: $target is on a noatime mount ($opts)" >&2; exit 1 ;;
esac
echo "preflight OK: $target (mount $opts), free: $(df -h --output=avail "$target" | tail -1 | tr -d ' ')"
PREFLIGHT_EOF

if [ "$MODE" = "check" ]; then
    log "check mode: would install ${DELIVERABLES[*]} into ~/$LIB_REL and the user unit dir on $LAPTOP_HOST,"
    log "  repoint the crontab line $CRON_OLD -> ~/$LIB_REL/gc-worktree-targets.sh,"
    log "  enable --now $UNIT.timer, and run $UNIT.service once. Nothing changed."
    exit 0
fi

# ── 3. copy the deliverables to a staging dir on the host ─────────────────
STAGE_REL="$LIB_REL/.staging"
tar -C "$REIFY_SOURCE_ROOT" -cf - "${DELIVERABLES[@]}" \
    | $SSH "$LAPTOP_HOST" "rm -rf $STAGE_REL && mkdir -p $STAGE_REL && tar -xf - -C $STAGE_REL" \
    || die "copying the deliverables to $LAPTOP_HOST failed"

# ── 4-6. install, repoint cron, enable, run once ──────────────────────────
# Single-quoted heredoc: expanded on the HOST. Positional args:
#   $1 UNIT  $2 LIB_REL  $3 CRON_OLD  $4 BACKUP_LABEL  $5 TARGET_DIR
$SSH "$LAPTOP_HOST" bash -s -- "$UNIT" "$LIB_REL" "$CRON_OLD" "$BACKUP_LABEL" "$TARGET_DIR" <<'APPLY_EOF' \
    || die "install on $LAPTOP_HOST failed (see output above)"
set -euo pipefail
unit="$1"; lib="$HOME/$2"; cron_old="$3"; label="$4"; target="$5"
stage="$lib/.staging"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$unit_dir"

for f in "$unit.service" "$unit.timer"; do
    if [ -f "$unit_dir/$f" ] && ! cmp -s "$stage/deploy/systemd/$f" "$unit_dir/$f"; then
        cp "$unit_dir/$f" "$lib/$f.bak-$label"
    fi
    install -m 0644 "$stage/deploy/systemd/$f" "$unit_dir/$f"
done
install -m 0755 "$stage/scripts/prune-unconsulted-target.sh" "$lib/prune-unconsulted-target.sh"
install -m 0755 "$stage/scripts/gc-worktree-targets.sh" "$lib/gc-worktree-targets.sh"
rm -rf "$stage"
echo "installed $unit.{service,timer} into $unit_dir and scripts into $lib"

cron_new="$lib/gc-worktree-targets.sh"
current="$(crontab -l 2>/dev/null || true)"
if [[ "$current" == *"$cron_old"* ]]; then
    printf '%s\n' "$current" >"$lib/crontab.bak-$label"
    while IFS= read -r line; do
        printf '%s\n' "${line//"$cron_old"/"$cron_new"}"
    done <<<"$current" | crontab -
    [[ "$(crontab -l)" == *"$cron_new"* ]] || { echo "ERROR: crontab repoint did not take" >&2; exit 1; }
    echo "crontab: gc-worktree-targets.sh now runs $cron_new (backup $lib/crontab.bak-$label)"
elif [[ "$current" == *"$cron_new"* ]]; then
    echo "crontab: already runs $cron_new (no-op)"
else
    echo "crontab: no gc-worktree-targets.sh line to repoint"
fi

systemctl --user daemon-reload
systemctl --user enable --now "$unit.timer"
[ "$(systemctl --user is-enabled "$unit.timer")" = "enabled" ] \
    || { echo "ERROR: $unit.timer is not enabled" >&2; exit 1; }
[ "$(systemctl --user is-active "$unit.timer")" = "active" ] \
    || { echo "ERROR: $unit.timer is not active" >&2; exit 1; }
echo "$unit.timer enabled and active"

echo "free before first run: $(df -h --output=avail "$target" | tail -1 | tr -d ' ')"
if ! systemctl --user start "$unit.service"; then
    journalctl --user -u "$unit.service" -n 40 --no-pager >&2 || true
    echo "ERROR: first run of $unit.service failed" >&2
    exit 1
fi
result="$(systemctl --user show -p Result --value "$unit.service")"
if [ "$result" != "success" ]; then
    journalctl --user -u "$unit.service" -n 40 --no-pager >&2 || true
    echo "ERROR: $unit.service Result=$result" >&2
    exit 1
fi
echo "free after first run: $(df -h --output=avail "$target" | tail -1 | tr -d ' ')"
journalctl --user -u "$unit.service" -n 20 --no-pager || true

if command -v loginctl >/dev/null \
        && [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || true)" != "yes" ]; then
    echo "WARN: lingering is off for $(id -un); $unit.timer fires only while the user manager runs" >&2
fi
APPLY_EOF

log "done: $UNIT.timer installed and verified on $LAPTOP_HOST"
