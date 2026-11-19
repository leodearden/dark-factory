#!/usr/bin/env bash
set -euo pipefail

# install_build_cache_store.sh
#
# SCOPE: idempotent install, on the workstation, of the shared build-cache
# store that reify's canonical build path writes its sccache entries to
# (PRD C13, plans/canonical-build-path-shared-cache-prd.md). Three user units:
#   build-cache-store.service        rclone serve webdav over the store dir,
#                                    bound to 127.0.0.1 and the tailscale
#                                    address only, basic auth from a 0600
#                                    EnvironmentFile (never in argv or the unit)
#   build-cache-store-prune.service  the size-cap pruner, copied out of the
#   build-cache-store-prune.timer    checkout so a branch switch cannot break it
#
# ORDER OF OPERATIONS (each step aborts non-zero on any problem)
#   1. Read-only preflight: rclone on PATH, /usr/bin/python3, lingering on
#      (else the user manager stops with the last session), a tailscale IPv4
#      address that is not 0.0.0.0, a numeric port and cap.
#   2. --check stops here after checking the installed store: every leaf
#      directory, the credentials file's mode, the pruner copy, each unit file
#      against its render, the units' states, and on both addresses a 401
#      without credentials and a 404 for a missing key with them.
#   3. Apply: pre-create the 4,368 leaf dirs <k0>/<k1>/<k2> (concurrent first
#      writes into a new directory fail), create the credentials once, copy
#      the pruner, write each unit only when its text differs, daemon-reload,
#      enable both, restart the store only when its unit or credentials
#      changed or it is not running, wait for it to answer, then run step 2.
#
# Usage:
#   install_build_cache_store.sh [--check]
#
# Env overrides:
#   BUILD_CACHE_STORE_PORT         listen port (default: 4227)
#   BUILD_CACHE_STORE_CAP_BYTES    prune cap (default: 300 GiB)
#   BUILD_CACHE_STORE_DIR          store dir (default: $HOME/.cache/build-cache-store)
#   BUILD_CACHE_STORE_CREDENTIALS  RCLONE_USER=/RCLONE_PASS= file
#                                  (default: $HOME/.config/build-cache-store/credentials)

PORT="${BUILD_CACHE_STORE_PORT:-4227}"
CAP_BYTES="${BUILD_CACHE_STORE_CAP_BYTES:-$((300 << 30))}"
STORE_DIR="${BUILD_CACHE_STORE_DIR:-$HOME/.cache/build-cache-store}"
CREDENTIALS="${BUILD_CACHE_STORE_CREDENTIALS:-$HOME/.config/build-cache-store/credentials}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRUNER_SRC="$SCRIPT_DIR/build_cache_store_prune.py"
LIB_DIR="$HOME/.local/lib/build-cache-store"
PRUNER="$LIB_DIR/build_cache_store_prune.py"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
STORE_UNIT="build-cache-store.service"
PRUNE_UNIT="build-cache-store-prune.service"
PRUNE_TIMER="build-cache-store-prune.timer"
HOST_PYTHON="/usr/bin/python3"
STORE_USER="reify-canon"
LEAF_COUNT=4368
WAIT_SECS=20

usage() { echo "Usage: $(basename "$0") [--check]" >&2; }

MODE="apply"
for arg in "$@"; do
    case "$arg" in
        --check) MODE="check" ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            usage
            exit 2
            ;;
    esac
done

log() { echo "[install_build_cache_store] $*"; }
die() { echo "[install_build_cache_store] ERROR: $*" >&2; exit 1; }

is_dotted_ipv4() {
    [[ "$1" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
    local octet
    for octet in "${BASH_REMATCH[@]:1}"; do
        [ "$((10#$octet))" -le 255 ] || return 1
    done
}

# A path rendered into a unit line must not need systemd quoting or carry a
# % specifier.
require_unit_safe_path() {
    [[ "$2" =~ ^/[A-Za-z0-9._/@+-]+$ ]] \
        || die "$1=$2 is not an absolute path of [A-Za-z0-9._/@+-]; nothing was changed"
}

random_hex() { od -An -N"$1" -tx1 /dev/urandom | tr -d ' \n'; }

# ── 1. read-only preflight ────────────────────────────────────────────────
preflight() {
    RCLONE="$(command -v rclone || true)"
    case "$RCLONE" in
        /*) ;;
        *) die "rclone is not on PATH; nothing was changed" ;;
    esac
    [ -x "$HOST_PYTHON" ] || die "$HOST_PYTHON, the prune unit's interpreter, is missing; nothing was changed"

    local user linger ts_out
    user="$(id -un)"
    linger="$(loginctl show-user "$user" --property=Linger --value 2>&1)" \
        || die "loginctl show-user $user failed: $linger; nothing was changed"
    [ "$linger" = "yes" ] \
        || die "Linger=$linger for $user: the store would stop with the last session (sudo loginctl enable-linger $user); nothing was changed"

    ts_out="$(tailscale ip -4 2>&1)" || die "tailscale ip -4 failed: $ts_out; nothing was changed"
    TS_ADDR="${ts_out%%$'\n'*}"
    is_dotted_ipv4 "$TS_ADDR" \
        || die "tailscale ip -4 printed no IPv4 address ('$TS_ADDR'); nothing was changed"
    [ "$TS_ADDR" != "0.0.0.0" ] \
        || die "tailscale ip -4 printed 0.0.0.0, which would bind every interface; nothing was changed"

    case "$PORT" in ''|*[!0-9]*) die "BUILD_CACHE_STORE_PORT=$PORT is not a port number; nothing was changed" ;; esac
    case "$CAP_BYTES" in ''|*[!0-9]*) die "BUILD_CACHE_STORE_CAP_BYTES=$CAP_BYTES is not a byte count; nothing was changed" ;; esac
    require_unit_safe_path rclone "$RCLONE"
    require_unit_safe_path BUILD_CACHE_STORE_DIR "$STORE_DIR"
    require_unit_safe_path BUILD_CACHE_STORE_CREDENTIALS "$CREDENTIALS"
    require_unit_safe_path pruner "$PRUNER"
    log "preflight OK: rclone at $RCLONE, Linger=yes, tailscale $TS_ADDR, port $PORT"
}

# ── unit renders: the one source of each unit's text, for apply and --check ─
render_store_unit() {
    cat <<EOF
[Unit]
Description=Shared build-cache store for reify's canonical build path (rclone serve webdav)
After=network-online.target tailscaled.service

[Service]
Type=simple
Environment=RCLONE_CONFIG=/dev/null
EnvironmentFile=$CREDENTIALS
ExecStart=$RCLONE serve webdav $STORE_DIR --addr 127.0.0.1:$PORT --addr $TS_ADDR:$PORT
Restart=always
RestartSec=2
TimeoutStopSec=60

[Install]
WantedBy=default.target
EOF
}

render_prune_unit() {
    cat <<EOF
[Unit]
Description=Prune the shared build-cache store to its size cap

[Service]
Type=oneshot
ExecStart=$HOST_PYTHON -I $PRUNER $STORE_DIR --cap-bytes $CAP_BYTES --rclone-unit $STORE_UNIT
Nice=19
IOSchedulingClass=idle
EOF
}

render_prune_timer() {
    cat <<EOF
[Unit]
Description=Daily prune of the shared build-cache store

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1h

[Install]
WantedBy=timers.target
EOF
}

UNITS=("$STORE_UNIT:render_store_unit" "$PRUNE_UNIT:render_prune_unit" "$PRUNE_TIMER:render_prune_timer")

credential() { sed -n "s/^$1=//p" "$CREDENTIALS"; }

# HTTP status of a GET; with a second argument "user:password", sent through
# curl's stdin config so it stays out of argv. 000 when nothing answered.
http_status() {
    local status
    if [ $# -gt 1 ]; then
        status="$(printf 'user = "%s"\n' "$2" \
            | curl -s -o /dev/null -w '%{http_code}' --max-time 5 -K - "$1")" || true
    else
        status="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1")" || true
    fi
    echo "${status:-000}"
}

# ── 2. the installed store, checked ───────────────────────────────────────
CHECK_FAILURES=0
check_failed() { echo "[install_build_cache_store] CHECK FAILED: $*" >&2; CHECK_FAILURES=$((CHECK_FAILURES + 1)); }

check_leaf_dirs() {
    if [ ! -d "$STORE_DIR" ]; then
        check_failed "store directory $STORE_DIR is missing"
        return
    fi
    local missing=0 a b c
    for a in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
        [ -d "$STORE_DIR/$a" ] || missing=$((missing + 1))
        for b in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
            [ -d "$STORE_DIR/$a/$b" ] || missing=$((missing + 1))
            for c in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
                [ -d "$STORE_DIR/$a/$b/$c" ] || missing=$((missing + 1))
            done
        done
    done
    [ "$missing" -eq 0 ] || check_failed "$missing of $LEAF_COUNT leaf directories are missing under $STORE_DIR"
}

check_credentials() {
    if [ ! -f "$CREDENTIALS" ]; then
        check_failed "credentials file $CREDENTIALS is missing"
        return
    fi
    local mode
    mode="$(stat -c %a "$CREDENTIALS")"
    [ "$mode" = "600" ] || check_failed "credentials file $CREDENTIALS has mode $mode, not 600"
    [ -n "$(credential RCLONE_USER)" ] || check_failed "$CREDENTIALS has no RCLONE_USER="
    [ -n "$(credential RCLONE_PASS)" ] || check_failed "$CREDENTIALS has no RCLONE_PASS="
}

check_units() {
    local entry unit render
    for entry in "${UNITS[@]}"; do
        unit="${entry%%:*}"
        render="${entry#*:}"
        if [ ! -f "$UNIT_DIR/$unit" ]; then
            check_failed "unit $UNIT_DIR/$unit is missing"
        elif ! cmp -s <("$render") "$UNIT_DIR/$unit"; then
            check_failed "unit $UNIT_DIR/$unit differs from what this installer renders for $unit"
        fi
    done
    local state
    for unit in "$STORE_UNIT" "$PRUNE_TIMER"; do
        state="$(systemctl --user is-enabled "$unit" 2>&1)" || check_failed "$unit is not enabled ($state)"
        state="$(systemctl --user is-active "$unit" 2>&1)" || check_failed "$unit is not active ($state)"
    done
}

check_http() {
    local auth="" addr key status
    if [ -f "$CREDENTIALS" ]; then
        auth="$(credential RCLONE_USER):$(credential RCLONE_PASS)"
    fi
    key="$(random_hex 16)"
    for addr in 127.0.0.1 "$TS_ADDR"; do
        status="$(http_status "http://$addr:$PORT/")"
        [ "$status" = "401" ] || check_failed "an unauthenticated GET on $addr:$PORT answered $status, not 401"
        if [ -n "$auth" ]; then
            status="$(http_status "http://$addr:$PORT/$key" "$auth")"
            [ "$status" = "404" ] \
                || check_failed "an authenticated GET of a missing key on $addr:$PORT answered $status, not 404"
        fi
    done
}

store_check() {
    CHECK_FAILURES=0
    check_leaf_dirs
    check_credentials
    if ! cmp -s "$PRUNER_SRC" "$PRUNER"; then
        check_failed "the pruner copy $PRUNER is missing or differs from $PRUNER_SRC"
    fi
    check_units
    check_http
    [ "$CHECK_FAILURES" -eq 0 ]
}

# ── 3. apply ──────────────────────────────────────────────────────────────
UNIT_CHANGED="no"
write_unit() {
    local unit="$1" render="$2" target="$UNIT_DIR/$1"
    if [ -f "$target" ] && cmp -s <("$render") "$target"; then
        UNIT_CHANGED="no"
        return
    fi
    "$render" >"$target.tmp"
    mv -f "$target.tmp" "$target"
    UNIT_CHANGED="yes"
    log "wrote $target"
}

wait_for_store() {
    local deadline=$((SECONDS + WAIT_SECS)) status
    while [ "$SECONDS" -lt "$deadline" ]; do
        status="$(http_status "http://127.0.0.1:$PORT/")"
        if [ "$status" = "401" ]; then
            return
        fi
        sleep 0.2
    done
    die "$STORE_UNIT did not answer 401 on 127.0.0.1:$PORT within ${WAIT_SECS}s"
}

apply() {
    mkdir -p "$STORE_DIR"/{0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f}/{0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f}/{0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f}
    log "store directory $STORE_DIR holds its $LEAF_COUNT leaf directories"

    local credentials_created="no"
    if [ ! -f "$CREDENTIALS" ]; then
        (
            umask 077
            mkdir -p "$(dirname "$CREDENTIALS")"
            printf 'RCLONE_USER=%s\nRCLONE_PASS=%s\n' "$STORE_USER" "$(random_hex 24)" >"$CREDENTIALS"
        )
        credentials_created="yes"
        log "created credentials $CREDENTIALS"
    fi
    chmod 600 "$CREDENTIALS"

    if ! cmp -s "$PRUNER_SRC" "$PRUNER"; then
        install -D -m 0755 "$PRUNER_SRC" "$PRUNER"
        log "installed the pruner at $PRUNER"
    fi

    mkdir -p "$UNIT_DIR"
    write_unit "$STORE_UNIT" render_store_unit
    local store_changed="$UNIT_CHANGED"
    write_unit "$PRUNE_UNIT" render_prune_unit
    write_unit "$PRUNE_TIMER" render_prune_timer
    local timer_changed="$UNIT_CHANGED"

    systemctl --user daemon-reload
    systemctl --user enable "$STORE_UNIT"
    systemctl --user enable --now "$PRUNE_TIMER"
    if [ "$timer_changed" = "yes" ]; then
        systemctl --user restart "$PRUNE_TIMER"
    fi
    if [ "$store_changed" = "yes" ] || [ "$credentials_created" = "yes" ] \
        || ! systemctl --user is-active --quiet "$STORE_UNIT"; then
        systemctl --user restart "$STORE_UNIT"
        log "(re)started $STORE_UNIT"
    else
        log "$STORE_UNIT unchanged and running; not restarted"
    fi
    wait_for_store
}

preflight
if [ "$MODE" = "check" ]; then
    store_check || die "$CHECK_FAILURES check(s) failed"
    log "check OK: the store answers on 127.0.0.1:$PORT and $TS_ADDR:$PORT"
    exit 0
fi
apply
store_check || die "the installed store failed $CHECK_FAILURES check(s)"
log "done: $STORE_UNIT serving $STORE_DIR on 127.0.0.1:$PORT and $TS_ADDR:$PORT; $PRUNE_TIMER enabled"
