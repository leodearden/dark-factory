#!/usr/bin/env bash
set -euo pipefail

# install_laptop_transcript_secret_sweep.sh
#
# SCOPE: workstation-side, idempotent install of the nightly transcript secret
# sweep (task 10242) onto leo-laptop, over ssh. The laptop runs the SAME
# committed timer, service and wrapper as the workstation, from its own
# dark-factory checkout (main, updated by push), so the sweep sources the
# laptop's own .env and no secret value ever crosses the wire. A committed
# drop-in (scripts/transcript-secret-sweep.laptop.conf) adds the two flags the
# laptop needs: --host-configs-dir (it has no orchestrator units; its verify
# configs in ~/.config/orchestrator name its project roots) and
# --no-escalation (the escalation server binds 127.0.0.1 on the workstation).
# Why, and how to read the laptop's record: OPERATIONS.md "Transcript secret
# sweep (02:30)".
#
# ORDER OF OPERATIONS (each step aborts non-zero on any problem)
#   1. The host answers ssh (BatchMode).
#   2. A read-only preflight on the host: the checkout carries this task's
#      files (the wrapper is executable; the installer, both units and the
#      drop-in exist), its .env exists (tested, never read), a systemd --user
#      bus answers, ~/.config/orchestrator is a directory, uv resolves on the
#      PATH the committed service sets, and the sweep imports in the
#      checkout's venv (the wrapper's own `uv run` with --help, which reads no
#      .env and touches no swept file; the wrapper always exits 0, so a sweep
#      that cannot import would otherwise fail silently every night).
#      --check / --dry-run stops here.
#   3. Install the drop-in FIRST, then run the checkout's own
#      install-transcript-secret-sweep-timer.sh (copies both units,
#      daemon-reloads, enables the timer, kicks no run), so the timer is never
#      enabled without the laptop flags.
#   4. Confirm the timer is enabled, and print how to run or dry-run it.
#
# Usage:
#   install_laptop_transcript_secret_sweep.sh [--check|--dry-run]
#
# Env overrides:
#   SSH          ssh invocation prefix
#                (default: "ssh -o BatchMode=yes -o ConnectTimeout=15")
#   LAPTOP_HOST  target host (default: leo-laptop)
#   REMOTE_REPO  the dark-factory checkout ON THE HOST
#                (default: /home/leo/src/dark-factory)

SSH="${SSH:-ssh -o BatchMode=yes -o ConnectTimeout=15}"
LAPTOP_HOST="${LAPTOP_HOST:-leo-laptop}"
REMOTE_REPO="${REMOTE_REPO:-/home/leo/src/dark-factory}"

UNIT="transcript-secret-sweep"

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

log() { echo "[install_laptop_transcript_secret_sweep] $*"; }
die() { echo "[install_laptop_transcript_secret_sweep] ERROR: $*" >&2; exit 1; }

# ── 1. reachability ───────────────────────────────────────────────────────
$SSH "$LAPTOP_HOST" true || die "ssh $LAPTOP_HOST failed (BatchMode)"

# ── 2. read-only preflight ────────────────────────────────────────────────
# Single-quoted heredoc: expanded on the HOST. $1 = REMOTE_REPO, $2 = UNIT.
$SSH "$LAPTOP_HOST" bash -s -- "$REMOTE_REPO" "$UNIT" <<'PREFLIGHT_EOF' || die "preflight failed on $LAPTOP_HOST"
set -euo pipefail
repo="$1"; unit="$2"
fail() { echo "ERROR: $*" >&2; exit 1; }
[ -x "$repo/scripts/$unit.sh" ] || fail "$repo/scripts/$unit.sh is missing or not executable"
for rel in "scripts/install-$unit-timer.sh" "scripts/$unit.service" "scripts/$unit.timer" "scripts/$unit.laptop.conf"; do
    [ -f "$repo/$rel" ] || fail "$repo/$rel is missing: pull a checkout that carries the laptop sweep"
done
[ -f "$repo/.env" ] || fail "$repo/.env is missing: the sweep would have no values to match"
systemctl --user show-environment >/dev/null 2>&1 || fail "no systemd --user bus"
[ -d "$HOME/.config/orchestrator" ] || fail "$HOME/.config/orchestrator is not a directory: no project roots to sweep"
service_path="$(sed -n 's/^Environment=PATH=//p' "$repo/scripts/$unit.service")"
[ -n "$service_path" ] || fail "$repo/scripts/$unit.service sets no Environment=PATH="
(PATH="$service_path"; command -v uv >/dev/null) || fail "uv does not resolve on the service's PATH ($service_path)"
(PATH="$service_path"; PYTHONDONTWRITEBYTECODE=1 \
    uv run --no-sync --project "$repo/shared" python "$repo/scripts/transcript_secret_sweep.py" --help >/dev/null) \
    || fail "the sweep does not import in $repo's venv (see above): sync it, then re-run"
echo "preflight OK: $repo carries the laptop sweep, and it imports under the service's PATH"
PREFLIGHT_EOF

if [ "$MODE" = "check" ]; then
    log "check mode: would install $REMOTE_REPO/scripts/$UNIT.laptop.conf as the $UNIT.service drop-in on $LAPTOP_HOST,"
    log "  then run $REMOTE_REPO/scripts/install-$UNIT-timer.sh there (no immediate run). Nothing changed."
    exit 0
fi

# ── 3. drop-in first, then the checkout's own timer installer ─────────────
# Single-quoted heredoc: expanded on the HOST. $1 = REMOTE_REPO, $2 = UNIT.
$SSH "$LAPTOP_HOST" bash -s -- "$REMOTE_REPO" "$UNIT" <<'APPLY_EOF' || die "install on $LAPTOP_HOST failed (see output above)"
set -euo pipefail
repo="$1"; unit="$2"
drop_in_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$unit.service.d"
mkdir -p "$drop_in_dir"
install -m 0644 "$repo/scripts/$unit.laptop.conf" "$drop_in_dir/laptop.conf"
echo "installed the laptop drop-in at $drop_in_dir/laptop.conf"
bash "$repo/scripts/install-$unit-timer.sh"
APPLY_EOF

# ── 4. confirm, and say how to run it ─────────────────────────────────────
enabled="$($SSH "$LAPTOP_HOST" systemctl --user is-enabled "$UNIT.timer" || true)"
[ "$enabled" = "enabled" ] || die "$UNIT.timer is not enabled on $LAPTOP_HOST (is-enabled: ${enabled:-nothing})"

log "done: $UNIT.timer enabled on $LAPTOP_HOST with the laptop drop-in (nightly 02:30, no run kicked)"
log "run one now:  ssh $LAPTOP_HOST systemctl --user start $UNIT.service"
log "its record:   ssh $LAPTOP_HOST journalctl --user -u $UNIT.service"
log "dry run:      ssh $LAPTOP_HOST 'PATH=\$HOME/.local/bin:\$PATH $REMOTE_REPO/scripts/$UNIT.sh" \
    "--host-configs-dir /home/leo/.config/orchestrator --no-escalation --dry-run'"
