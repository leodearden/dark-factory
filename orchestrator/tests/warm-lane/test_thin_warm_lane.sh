#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_thin_warm_lane.sh
# Hermetic tests for orchestrator/scripts/warm-lane/thin-warm-lane.sh (task 5174, PRD
# docs/prds/warm-lane-pool-sizing-lifecycle.md §9.3).
#
# Real rm/flock scoped to mktemp lane fixtures (hermetic-by-default, no PATH
# stubbing of coreutils needed — only --seed-script is stubbed, via the
# script's own hermetic test seam).
#
# run_helper captures STDOUT and STDERR SEPARATELY:
#   OUT     — captured stdout from the script
#   ERR_OUT — captured stderr from the script
#   RC      — exit code
#
# Blocks:
#   A — CLI/usage contract + lane_dir existence guard (step-1/step-2)
#   B — precondition-refusal + T3 flock guard (step-3/step-4)
#   C — FREE-FIRST reclaim + T1 source-intact (step-5/step-6)
#   D — --reseed opt-in + free-BEFORE-stage ordering (step-7/step-8)
#   F — live-process-reference gate: a lane whose target/ is referenced by a
#       live process (cwd / open fd / mmap) is PRESERVED even when its flock is
#       free (task 5307, ported from reify task 5823; the flock is a reseed
#       mutex, not a liveness oracle — esc-5334-6)
#
# Labels follow reify's copy of this suite so the two stay diffable. Reify's
# Block E (the seed-abort discard) is deliberately absent: dark-factory's
# thin-warm-lane.sh has no such discard.
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (PORTED_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
SCRIPT="$WARM_LANE_SCRIPTS_DIR/thin-warm-lane.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=tests/infra/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"

echo "=== scripts/thin-warm-lane.sh hermetic tests (task 5174) ==="

# ──────────────────────────────────────────────────────────────────────────────
# Shared temp state + cleanup
# ──────────────────────────────────────────────────────────────────────────────
_TMPDIRS=()
_BGPIDS=()
cleanup() {
    for pid in "${_BGPIDS[@]+${_BGPIDS[@]}}"; do
        kill "$pid" 2>/dev/null || true
    done
    for d in "${_TMPDIRS[@]+${_TMPDIRS[@]}}"; do rm -rf "$d"; done
}
trap cleanup EXIT

# Arm the shared-trash litter guard (task 5612). Sited immediately after
# `trap cleanup EXIT` because the helper registers its per-run root into
# _TMPDIRS and so must follow this file's own `_TMPDIRS=()`.
# Rationale, ordering contract, stem rules and honest scope: see the
# CANONICAL WIRING CONTRACT comment in tests/infra/test_helpers.sh.
init_isolated_lane_root test-thin-warm-lane

# _wait_for_reader_lock <ready-marker> <deadline-seconds>
# Causal ordering (technique R, docs/prds/infra-test-wallclock-deflake.md,
# task #4847): polls for the READY marker file in 0.05s ticks instead of a
# fixed sleep, so a background flock holder's acquisition is causally
# guaranteed complete before the caller's next statement runs -- a fixed
# sleep races the holder under CPU/IO load (the script-under-test's flock -n
# could win first, spuriously turning a T3-refusal assertion green->red).
# Mirrors tests/infra/test_warm_lane_gc.sh's identically-named helper.
_wait_for_reader_lock() {
    local ready_marker="$1"
    local deadline_s="$2"
    local max_ticks=$(( deadline_s * 20 ))
    local tick=0
    while [ "$tick" -lt "$max_ticks" ]; do
        [ -f "$ready_marker" ] && return 0
        sleep 0.05
        tick=$(( tick + 1 ))
    done
    return 1
}

ERR_FILE="$(mktemp /tmp/test-thin-warm-lane-err-XXXXXX)"
_TMPDIRS+=("$ERR_FILE")

# ── run_helper ─────────────────────────────────────────────────────────────────
# Invokes thin-warm-lane.sh, capturing OUT (stdout), ERR_OUT (stderr), RC.
run_helper() {
    local rc=0
    > "$ERR_FILE"
    OUT="$(bash "$SCRIPT" "$@" 2>"$ERR_FILE")" || rc=$?
    ERR_OUT="$(cat "$ERR_FILE")"
    RC=$rc
}

# ──────────────────────────────────────────────────────────────────────────────
# Block A — CLI/usage contract + lane_dir existence guard
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block A: CLI/usage contract + lane_dir existence guard ---"

assert "A1: script exists" test -f "$SCRIPT"
assert "A2: script is executable" test -x "$SCRIPT"
assert "A3: script has the #!/usr/bin/env bash shebang" \
    bash -c 'head -1 "$1" | grep -qx "#!/usr/bin/env bash"' _ "$SCRIPT"

# A4/A5: --help exits 0 and prints a usage line to stderr
run_helper --help
assert "A4: --help exits 0" test "$RC" -eq 0
assert "A5: --help prints a 'Usage' line to stderr" \
    bash -c 'printf "%s\n" "$1" | grep -qi usage' _ "$ERR_OUT"

# A6: an unknown flag exits 2
run_helper /tmp --totally-bogus-flag-xyz
assert "A6: unknown flag exits 2" test "$RC" -eq 2

# A7: a missing positional lane_dir exits 2
run_helper
assert "A7: missing positional lane_dir exits 2" test "$RC" -eq 2

# A8: a nonexistent lane_dir exits 1 with a diagnostic on stderr, touches nothing
A8_LANE="$(mktemp -u /tmp/test-thin-warm-lane-a8-XXXXXX)"
run_helper "$A8_LANE"
assert "A8: nonexistent lane_dir exits 1" test "$RC" -eq 1
assert "A8: nonexistent lane_dir prints a diagnostic on stderr" \
    bash -c '[ -n "$1" ]' _ "$ERR_OUT"
assert "A8: nonexistent lane_dir still does not exist afterward (touches nothing)" \
    bash -c '[ ! -e "$1" ]' _ "$A8_LANE"

# A9: a lane_dir that exists but is a regular file (not a directory) exits 1
A9_LANE="$(mktemp /tmp/test-thin-warm-lane-a9-XXXXXX)"
_TMPDIRS+=("$A9_LANE")
run_helper "$A9_LANE"
assert "A9: lane_dir that is a regular file exits 1" test "$RC" -eq 1
assert "A9: regular-file lane_dir is untouched" test -f "$A9_LANE"

# A10: --reseed without --base is a usage error (exit 2), fired during arg
# parsing before any existence/lock/rm work.
run_helper /tmp --reseed
assert "A10: --reseed without --base exits 2" test "$RC" -eq 2

# A11: --base given with no following value exits 2
run_helper /tmp --base
assert "A11: --base with no value exits 2" test "$RC" -eq 2

# A12: a MISSING sibling lib_live_refs.sh is a fail-LOUD wiring error (task
# 5307). The liveness check must never be silently absent, so the guard that
# enforces its presence cannot itself be untested. Technique copied from
# test_warm_lane_gc.sh A9: the script is copied ALONE into a temp dir and invoked
# with --help, which normally exits 0 (A4 above), so a 2 here also pins that the
# guard fires BEFORE argv is parsed. Exit 2, not 1: an incomplete deployment is
# a WIRING error, and thin_warm_lane logs rc=2 at WARNING, so a broken
# deployment is loud rather than silently fail-open.
A12_DIR="$(mktemp -d /tmp/test-thin-warm-lane-a12-XXXXXX)"
_TMPDIRS+=("$A12_DIR")
cp "$SCRIPT" "$A12_DIR/thin-warm-lane.sh"
assert "A12: fixture — the copy really has no sibling lib_live_refs.sh" \
    bash -c '[ ! -e "$1/lib_live_refs.sh" ]' _ "$A12_DIR"
A12_RC=0
A12_ERR="$(bash "$A12_DIR/thin-warm-lane.sh" --help 2>&1 >/dev/null)" || A12_RC=$?
assert "A12: missing sibling lib_live_refs.sh exits 2 (wiring error, not runtime 1)" \
    test "$A12_RC" -eq 2
assert "A12: stderr names the missing library (fail LOUD, not silent)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "lib_live_refs.sh not found next to"' _ "$A12_ERR"

# ──────────────────────────────────────────────────────────────────────────────
# Block B — precondition-refusal + T3 flock guard
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block B: precondition-refusal + T3 flock guard ---"

# ── B1: lane_dir OUTSIDE REIFY_WARM_LANE_MOUNT is refused ─────────────────────
B1_MOUNT="$(mktemp -d /tmp/test-thin-warm-lane-b1-mount-XXXXXX)"
_TMPDIRS+=("$B1_MOUNT")
B1_OUTSIDE_LANE="$(mktemp -d /tmp/test-thin-warm-lane-b1-outside-XXXXXX)"
_TMPDIRS+=("$B1_OUTSIDE_LANE")
mkdir -p "$B1_OUTSIDE_LANE/target"
touch "$B1_OUTSIDE_LANE/target/MARKER"

REIFY_WARM_LANE_MOUNT="$B1_MOUNT" run_helper "$B1_OUTSIDE_LANE"
assert "B1: lane_dir outside REIFY_WARM_LANE_MOUNT exits 1" test "$RC" -eq 1
assert "B1: target/ untouched (outside-mount refusal)" \
    test -f "$B1_OUTSIDE_LANE/target/MARKER"

# ── B2: lane_dir resolving to the mount's base dir is refused (self-clobber) ──
B2_MOUNT="$(mktemp -d /tmp/test-thin-warm-lane-b2-mount-XXXXXX)"
_TMPDIRS+=("$B2_MOUNT")
mkdir -p "$B2_MOUNT/base/target"
touch "$B2_MOUNT/base/target/MARKER"

REIFY_WARM_LANE_MOUNT="$B2_MOUNT" run_helper "$B2_MOUNT/base"
assert "B2: lane_dir resolving to the mount's base dir exits 1" test "$RC" -eq 1
assert "B2: target/ untouched (self-clobber refusal)" \
    test -f "$B2_MOUNT/base/target/MARKER"

# ── B3: an ASSIGNED lane (external flock held) is refused with EX_TEMPFAIL ────
B3_LANE="$(mktemp -d /tmp/test-thin-warm-lane-b3-lane-XXXXXX)"
_TMPDIRS+=("$B3_LANE")
mkdir -p "$B3_LANE/target"
touch "$B3_LANE/target/MARKER"

B3_LOCK="${B3_LANE}.lock"
_TMPDIRS+=("$B3_LOCK")
B3_READY="${B3_LOCK}.ready-marker"
_TMPDIRS+=("$B3_READY")
touch "$B3_LOCK"
# Causal handshake (see _wait_for_reader_lock above) instead of a fixed sleep:
# the subshell touches B3_READY AFTER acquiring flock -x, so the assertions
# below only run once the lock is provably held.
( flock -x 9 && touch "$B3_READY" && sleep 300 ) 9>"$B3_LOCK" &
B3_LOCK_PID=$!
_BGPIDS+=("$B3_LOCK_PID")
_wait_for_reader_lock "$B3_READY" 30

run_helper "$B3_LANE"
assert "B3: ASSIGNED lane (flock held) exits 75 (EX_TEMPFAIL)" test "$RC" -eq 75
assert "B3: target/ byte-intact under lock contention (T3)" \
    test -f "$B3_LANE/target/MARKER"
assert "B3: stderr mentions the lock/live-consumer refusal" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "lock|consumer|assigned"' _ "$ERR_OUT"
# BOTH refusals exit 75, so the stderr token is the only thing that tells them
# apart in dark-factory's logs, and it is pinned in both directions: F4 below
# pins that a process-reference refusal never carries the lock token, and this
# pins the converse. Pin the TOKEN, not the prose. This arm pins
# non-misattribution only, not gate ORDER: B3_LANE has no live reference, so the
# gates' order is unobservable here. B3b is where it is observable.
assert "B3: stderr carries the LANE_LOCK_CONTENDED token" \
    bash -c 'printf "%s\n" "$1" | grep -qF "LANE_LOCK_CONTENDED:"' _ "$ERR_OUT"
assert "B3: stderr does NOT carry the LANE_LIVE_REF token (the two exit-75 reasons stay distinguishable)" \
    bash -c '! printf "%s\n" "$1" | grep -qF "LANE_LIVE_REF"' _ "$ERR_OUT"

# ── B3b: BOTH gates armed at once — the flock gate must fire FIRST ────────────
# One holder owns <lane>.lock and a second holds its cwd inside <lane>/target, so
# T3 and T4 would each refuse on their own. Only the evaluation ORDER decides
# which token reaches stderr: swap the two gates in thin-warm-lane.sh and this
# arm goes red while every other assert stays green. Flock first is what keeps
# the /proc walk off a flock-held lane and runs it only under a held lock.
B3B_ROOT="$(mktemp -d /tmp/test-thin-warm-lane-b3b-XXXXXX)"
_TMPDIRS+=("$B3B_ROOT")
B3B_LANE="$B3B_ROOT/_lane-b3b"
mkdir -p "$B3B_LANE/target"
touch "$B3B_LANE/target/MARKER"
B3B_LOCK="${B3B_LANE}.lock"
touch "$B3B_LOCK"
B3B_LOCK_READY="$B3B_ROOT/lock-holder.ready"
B3B_CWD_READY="$B3B_ROOT/cwd-holder.ready"

# `exec sleep` keeps each tracked PID the one owning its reference.
( flock -x 9 && touch "$B3B_LOCK_READY" && exec sleep 300 ) 9>"$B3B_LOCK" &
B3B_LOCK_PID=$!
_BGPIDS+=("$B3B_LOCK_PID")
( cd "$B3B_LANE/target" && touch "$B3B_CWD_READY" && exec sleep 300 ) &
B3B_CWD_PID=$!
_BGPIDS+=("$B3B_CWD_PID")
_wait_for_reader_lock "$B3B_LOCK_READY" 30
_wait_for_reader_lock "$B3B_CWD_READY" 30

assert "B3b-fixture: the cwd holder really is under <lane>/target (T4 would fire)" \
    bash -c 'case "$(readlink -f /proc/$1/cwd)/" in "$(readlink -f "$2")"/target/*) exit 0;; *) exit 1;; esac' \
    _ "$B3B_CWD_PID" "$B3B_LANE"
assert "B3b-fixture: the lane flock really is held (T3 would fire)" \
    bash -c '! flock -n 9 2>/dev/null' _ 9<"$B3B_LOCK"

run_helper "$B3B_LANE"
assert "B3b: a lane that is BOTH flock-held and live-referenced exits 75" \
    test "$RC" -eq 75
assert "B3b: target/ byte-intact (neither gate frees anything)" \
    test -f "$B3B_LANE/target/MARKER"
assert "B3b: ORDERING — the flock gate fires first, so stderr carries LANE_LOCK_CONTENDED" \
    bash -c 'printf "%s\n" "$1" | grep -qF "LANE_LOCK_CONTENDED:"' _ "$ERR_OUT"
assert "B3b: ORDERING — the T4 scan is never reached, so stderr carries no LANE_LIVE_REF" \
    bash -c '! printf "%s\n" "$1" | grep -qF "LANE_LIVE_REF"' _ "$ERR_OUT"

kill "$B3B_LOCK_PID" "$B3B_CWD_PID" 2>/dev/null || true
wait "$B3B_LOCK_PID" "$B3B_CWD_PID" 2>/dev/null || true

kill "$B3_LOCK_PID" 2>/dev/null || true
wait "$B3_LOCK_PID" 2>/dev/null || true

# ── B4: lane_dir literally IS the mount root is refused (self-clobber) ────────
# The under-mount guard's trailing-slash case-match ("$_rp_mount"/* against
# "$_rp_lane_dir/") treats the mount root itself as "under" the mount (glob
# * also matches the empty tail), so this case slips past that guard and is
# caught only by the separate self-clobber check's explicit ==mount-root test.
B4_MOUNT="$(mktemp -d /tmp/test-thin-warm-lane-b4-mount-XXXXXX)"
_TMPDIRS+=("$B4_MOUNT")
mkdir -p "$B4_MOUNT/target"
touch "$B4_MOUNT/target/MARKER"

REIFY_WARM_LANE_MOUNT="$B4_MOUNT" run_helper "$B4_MOUNT"
assert "B4: lane_dir == mount root exits 1 (self-clobber)" test "$RC" -eq 1
assert "B4: target/ untouched (mount-root self-clobber refusal)" \
    test -f "$B4_MOUNT/target/MARKER"

# ── B5: a lane_dir literally named 'base' is refused even when
# REIFY_WARM_LANE_MOUNT is unset (the basename=='base' guard is unconditional,
# independent of the under-mount guard) ────────────────────────────────────────
B5_PARENT="$(mktemp -d /tmp/test-thin-warm-lane-b5-parent-XXXXXX)"
_TMPDIRS+=("$B5_PARENT")
B5_LANE="$B5_PARENT/base"
mkdir -p "$B5_LANE/target"
touch "$B5_LANE/target/MARKER"

# Save/restore rather than a bare unset: the OPTIONAL df-delta layer in Block C
# gates on an operator-supplied REIFY_WARM_LANE_MOUNT (private reflink mount),
# and this test must not silently disable that layer for the rest of the run.
_B5_HAD_MOUNT=0
if [ -n "${REIFY_WARM_LANE_MOUNT+x}" ]; then
    _B5_HAD_MOUNT=1
    _B5_SAVED_MOUNT="$REIFY_WARM_LANE_MOUNT"
fi
unset REIFY_WARM_LANE_MOUNT
run_helper "$B5_LANE"
assert "B5: lane_dir named 'base' exits 1 (REIFY_WARM_LANE_MOUNT unset)" test "$RC" -eq 1
assert "B5: target/ untouched ('base'-name refusal, mount unset)" \
    test -f "$B5_LANE/target/MARKER"
if [ "$_B5_HAD_MOUNT" = "1" ]; then
    export REIFY_WARM_LANE_MOUNT="$_B5_SAVED_MOUNT"
fi

# _thin_detect_private_mount() — rung-1-only substrate detector (no loopback
# self-provisioning, unlike test_warm_lane_pool.sh's detect_private_substrate,
# to stay `pool`-safe): returns 0 when REIFY_WARM_LANE_MOUNT is set and
# reflink-capable, 1 otherwise. Gates the OPTIONAL df-delta assertion below;
# never gates (or skips) the hermetic core.
_thin_detect_private_mount() {
    [ -n "${REIFY_WARM_LANE_MOUNT:-}" ] || return 1
    [ -d "${REIFY_WARM_LANE_MOUNT}" ] || return 1
    local probe_src probe_dst
    probe_src="$(mktemp "${REIFY_WARM_LANE_MOUNT}/.thin-reflink-probe-XXXXXX" 2>/dev/null)" || return 1
    probe_dst="${probe_src}.dst"
    if cp --reflink=always "$probe_src" "$probe_dst" 2>/dev/null; then
        rm -f "$probe_src" "$probe_dst" 2>/dev/null || true
        return 0
    fi
    rm -f "$probe_src" "$probe_dst" 2>/dev/null || true
    return 1
}

# ──────────────────────────────────────────────────────────────────────────────
# Block C — FREE-FIRST reclaim + T1 source-intact
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block C: FREE-FIRST reclaim + T1 source-intact ---"

C_LANE="$(mktemp -d /tmp/test-thin-warm-lane-c-lane-XXXXXX)"
_TMPDIRS+=("$C_LANE")

# target/ with real weight (du -sb footprint > 0), so removal is meaningful.
mkdir -p "$C_LANE/target/debug"
dd if=/dev/zero of="$C_LANE/target/debug/blob.bin" bs=1024 count=256 2>/dev/null

C_TARGET_FOOTPRINT_BEFORE="$(du -sb "$C_LANE/target" | cut -f1)"
assert "C1: target/ fixture has nonzero footprint before thinning" \
    test "$C_TARGET_FOOTPRINT_BEFORE" -gt 0

# Source tree + real .git + an uncommitted WIP file (unlanded work) -- all
# must survive thinning byte-intact (T1).
mkdir -p "$C_LANE/src"
echo 'fn main() {}' > "$C_LANE/src/main.rs"
git init -q "$C_LANE"
git -C "$C_LANE" config user.email "test@test.local"
git -C "$C_LANE" config user.name "Test"
git -C "$C_LANE" add src/main.rs
git -C "$C_LANE" commit -q -m "initial"
C_HEAD_BEFORE="$(git -C "$C_LANE" rev-parse HEAD)"
echo "uncommitted work" > "$C_LANE/WIP.txt"

# --seed-script stub: records invocations. Must NOT be called on the default
# (no --reseed) path.
C_SEED_LOG="$(mktemp /tmp/test-thin-warm-lane-c-seedlog-XXXXXX)"
_TMPDIRS+=("$C_SEED_LOG")
C_SEED_STUB="$(mktemp /tmp/test-thin-warm-lane-c-seedstub-XXXXXX)"
_TMPDIRS+=("$C_SEED_STUB")
cat > "$C_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
exit 0
STUB_EOF
chmod +x "$C_SEED_STUB"
export SEED_LOG="$C_SEED_LOG"

run_helper "$C_LANE" --seed-script "$C_SEED_STUB"

assert "C2: exit 0" test "$RC" -eq 0
assert "C3: target/ is GONE" bash -c '[ ! -e "$1" ]' _ "$C_LANE/target"
assert "C4: source tree byte-intact (src/main.rs unchanged)" \
    bash -c '[ "$(cat "$1")" = "fn main() {}" ]' _ "$C_LANE/src/main.rs"
assert "C5: .git/ intact (HEAD unchanged)" \
    bash -c '[ "$(git -C "$1" rev-parse HEAD)" = "$2" ]' _ "$C_LANE" "$C_HEAD_BEFORE"
assert "C6: uncommitted WIP file byte-intact" \
    bash -c '[ "$(cat "$1")" = "uncommitted work" ]' _ "$C_LANE/WIP.txt"
assert "C7: STDOUT equals the realpath-resolved lane_dir (single line, no diagnostics mixed in)" \
    test "$OUT" = "$(realpath -m "$C_LANE")"
assert "C8: seed-script NOT invoked (no reseed staged by default)" \
    bash -c '[ ! -s "$1" ]' _ "$C_SEED_LOG"

# ── OPTIONAL substrate-gated df-delta (B4 literal, direction-only per G6/D8) ───
# Runs only when REIFY_WARM_LANE_MOUNT points at a private reflink mount;
# skips gracefully otherwise (never a frozen GB constant, never false-RED off
# substrate). Only this one assertion is gated -- not the hermetic core above.
if _thin_detect_private_mount; then
    C_DF_LANE="$(mktemp -d "${REIFY_WARM_LANE_MOUNT}/test-thin-warm-lane-df-XXXXXX")"
    _TMPDIRS+=("$C_DF_LANE")
    mkdir -p "$C_DF_LANE/target"
    dd if=/dev/zero of="$C_DF_LANE/target/blob.bin" bs=1M count=2 2>/dev/null

    C_DF_BEFORE="$(df --output=avail -m "$C_DF_LANE" 2>/dev/null | tail -1 | tr -d ' ')"
    run_helper "$C_DF_LANE"
    C_DF_RC="$RC"
    C_DF_AFTER="$(df --output=avail -m "$C_DF_LANE" 2>/dev/null | tail -1 | tr -d ' ')"
    echo "Block C df: before=${C_DF_BEFORE}MiB after=${C_DF_AFTER}MiB" >&2

    assert "C9: df-gated control run exits 0" test "$C_DF_RC" -eq 0
    assert "C9: df available did not decrease after thinning (direction-only, private mount)" \
        test "$C_DF_AFTER" -ge "$C_DF_BEFORE"
else
    echo "SKIP: Block C df-delta assertion -- REIFY_WARM_LANE_MOUNT not set to a private reflink mount" >&2
fi

# ──────────────────────────────────────────────────────────────────────────────
# Block D — --reseed opt-in + free-BEFORE-stage ordering (T2)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block D: --reseed opt-in + free-BEFORE-stage ordering (T2) ---"

D_BASE="$(mktemp -d /tmp/test-thin-warm-lane-d-base-XXXXXX)"
_TMPDIRS+=("$D_BASE")

# --seed-script stub: logs argv AND whether <lane>/target existed AT CALL TIME
# (proves free-before-stage ordering when it's invoked under --reseed).
D_SEED_LOG="$(mktemp /tmp/test-thin-warm-lane-d-seedlog-XXXXXX)"
_TMPDIRS+=("$D_SEED_LOG")
D_SEED_STUB="$(mktemp /tmp/test-thin-warm-lane-d-seedstub-XXXXXX)"
_TMPDIRS+=("$D_SEED_STUB")
cat > "$D_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
{
    echo "ARGV: $*"
    if [ -e "$2/target" ]; then
        echo "PRESENCE: PRESENT"
    else
        echo "PRESENCE: ABSENT"
    fi
} >> "$SEED_LOG"
exit 0
STUB_EOF
chmod +x "$D_SEED_STUB"
export SEED_LOG="$D_SEED_LOG"

# ── D1: default run (no --reseed) leaves the seed-script log EMPTY ────────────
D_LANE_A="$(mktemp -d /tmp/test-thin-warm-lane-d-a-XXXXXX)"
_TMPDIRS+=("$D_LANE_A")
mkdir -p "$D_LANE_A/target"
touch "$D_LANE_A/target/MARKER"

run_helper "$D_LANE_A" --seed-script "$D_SEED_STUB"
assert "D1: default (no --reseed) run exits 0" test "$RC" -eq 0
assert "D1: default run leaves the seed-script log EMPTY (no clone staged)" \
    bash -c '[ ! -s "$1" ]' _ "$D_SEED_LOG"

# ── D2: --reseed invokes the seed-script AFTER the free (T2) ──────────────────
D_LANE_B="$(mktemp -d /tmp/test-thin-warm-lane-d-b-XXXXXX)"
_TMPDIRS+=("$D_LANE_B")
mkdir -p "$D_LANE_B/target"
touch "$D_LANE_B/target/MARKER"

run_helper "$D_LANE_B" --reseed --base "$D_BASE" --seed-script "$D_SEED_STUB"
assert "D2: --reseed run exits 0" test "$RC" -eq 0
assert "D2: seed-script WAS invoked" bash -c '[ -s "$1" ]' _ "$D_SEED_LOG"
assert "D2: seed-script argv includes --fresh-checkout" \
    bash -c 'grep -q -- "--fresh-checkout" "$1"' _ "$D_SEED_LOG"
# thin holds ${LANE_DIR}.lock on FD 9 (T3, line 236) across the seed call, so
# the seed must NOT re-acquire it — else it self-refuses under seed-warm-lane.sh's
# fail-safe lane-lock default (esc-5214/task 5354). thin therefore passes
# --assume-lane-lock-held so the seed skips its own acquire.
assert "D2: seed-script argv includes --assume-lane-lock-held (thin already holds the lane lock on FD 9; task 5354)" \
    bash -c 'grep -q -- "--assume-lane-lock-held" "$1"' _ "$D_SEED_LOG"
assert "D2: seed-script argv includes the lane_dir" \
    bash -c 'grep -qF "$2" "$1"' _ "$D_SEED_LOG" "$D_LANE_B"
assert "D2: seed-script observed target ABSENT at invocation (free-before-stage, T2)" \
    bash -c 'grep -q "PRESENCE: ABSENT" "$1"' _ "$D_SEED_LOG"

# ── D3: --reseed best-effort — a failing seed-script does not flip the exit
# code. The free already succeeded (target/ is gone); that is the operation
# this script guarantees, so a reseed failure is logged, not fatal.
D_LANE_C="$(mktemp -d /tmp/test-thin-warm-lane-d-c-XXXXXX)"
_TMPDIRS+=("$D_LANE_C")
mkdir -p "$D_LANE_C/target"
touch "$D_LANE_C/target/MARKER"

D_FAIL_STUB="$(mktemp /tmp/test-thin-warm-lane-d-failstub-XXXXXX)"
_TMPDIRS+=("$D_FAIL_STUB")
cat > "$D_FAIL_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
exit 1
STUB_EOF
chmod +x "$D_FAIL_STUB"

run_helper "$D_LANE_C" --reseed --base "$D_BASE" --seed-script "$D_FAIL_STUB"
assert "D3: --reseed with a failing seed-script still exits 0 (best-effort)" \
    test "$RC" -eq 0
assert "D3: target/ is still gone despite the reseed failure" \
    bash -c '[ ! -e "$1" ]' _ "$D_LANE_C/target"

# ──────────────────────────────────────────────────────────────────────────────
# Block F — live-process-reference gate (task 5307, ported from reify task 5823)
#
# ROOT CAUSE (esc-5334-6): the lane flock is a reseed MUTEX, not a liveness
# oracle. <lane>.lock is held only across the acquire reseed and across
# run_scoped_verification, NEVER across the implement phase, where an agent runs
# builds for tens of minutes. So Block B's `flock -n 9` gate is blind for most of
# an ASSIGNED lane's life, and thin would `rm -rf <lane>/target` out from under a
# live build.
#
# Every <lane>.lock in this block is FREE throughout — no flock holder is started
# anywhere in Block F — so the T3 gate cannot account for ANY refusal observed
# here. The only thing that can is the /proc live-reference check.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block F: live-process-reference gate (task 5307) ---"

# --seed-script stub: records invocations, so F-reseed3's "the refusal precedes
# the stage" is observable rather than assumed. Same shape as Block C's stub.
F_SEED_LOG="$(mktemp /tmp/test-thin-warm-lane-f-seedlog-XXXXXX)"
_TMPDIRS+=("$F_SEED_LOG")
F_SEED_STUB="$(mktemp /tmp/test-thin-warm-lane-f-seedstub-XXXXXX)"
_TMPDIRS+=("$F_SEED_STUB")
cat > "$F_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
exit 0
STUB_EOF
chmod +x "$F_SEED_STUB"
export SEED_LOG="$F_SEED_LOG"

# ── F-cwd: a live holder whose CWD is <lane>/target ───────────────────────────
# The build-cwd shape a cargo/rustc process holds. Mirrors test_warm_lane_gc.sh's
# P-basic fixture. The holder touches READY only AFTER cd'ing in, so
# _wait_for_reader_lock proves the cwd is established before thin runs; `exec
# sleep` so the tracked PID is the one holding the cwd.
F_ROOT="$(mktemp -d /tmp/test-thin-warm-lane-f-XXXXXX)"
_TMPDIRS+=("$F_ROOT")
F_LANE="$F_ROOT/_lane-f"
mkdir -p "$F_LANE/target"
touch "$F_LANE/target/DIVERGENT_MARKER"
touch "${F_LANE}.lock"
F_READY="$F_ROOT/holder.ready"
( cd "$F_LANE/target" && touch "$F_READY" && exec sleep 300 ) &
F_HOLDER_PID=$!
_BGPIDS+=("$F_HOLDER_PID")
_wait_for_reader_lock "$F_READY" 30

run_helper "$F_LANE" --seed-script "$F_SEED_STUB"

# 75 SPECIFICALLY, not merely non-zero: thin_warm_lane logs rc=75 at DEBUG
# as a benign skip and every other non-zero rc at WARNING, so any other refusal
# code would turn each lingering-reference release into a false fault.
assert "F1: a live-cwd-referenced lane exits 75 (EX_TEMPFAIL), with the flock FREE" \
    test "$RC" -eq 75
assert "F2: target/DIVERGENT_MARKER byte-intact (the live build's cache is untouched)" \
    test -f "$F_LANE/target/DIVERGENT_MARKER"
assert "F3: stderr carries the LANE_LIVE_REF token (the PROCESS-REFERENCE refusal)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "LANE_LIVE_REF:"' _ "$ERR_OUT"
assert "F4: stderr does NOT carry LANE_LOCK_CONTENDED (no misattribution; every lock in Block F is FREE)" \
    bash -c '! printf "%s\n" "$1" | grep -qF "LANE_LOCK_CONTENDED"' _ "$ERR_OUT"
assert "F5: stdout is EMPTY (the resolved lane_dir is printed only on success)" \
    test -z "$OUT"

# ── F-fd: the OPEN-FD leg, holder cwd deliberately OUTSIDE the lane ───────────
# F-cwd drives only the cwd leg of the scanner. This drives the open-fd leg
# alone: a linker holding an output fd under <lane>/target while its cwd sits
# elsewhere. The holder's cwd is FFD_ROOT, outside the lane, and `sleep`'s own
# mappings live in /usr, so neither the cwd nor the mmap leg can explain a
# refusal. `exec 9>` (not a redirect on `exec sleep`) is what makes the
# descriptor survive the exec into the tracked PID.
FFD_ROOT="$(mktemp -d /tmp/test-thin-warm-lane-ffd-XXXXXX)"
_TMPDIRS+=("$FFD_ROOT")
FFD_LANE="$FFD_ROOT/_lane-fd"
mkdir -p "$FFD_LANE/target"
touch "$FFD_LANE/target/DIVERGENT_MARKER"
touch "${FFD_LANE}.lock"
FFD_READY="$FFD_ROOT/fd-holder.ready"
( cd "$FFD_ROOT" && exec 9>"$FFD_LANE/target/held.tmp" \
    && touch "$FFD_READY" && exec sleep 300 ) &
FFD_HOLDER_PID=$!
_BGPIDS+=("$FFD_HOLDER_PID")
_wait_for_reader_lock "$FFD_READY" 30

# Non-vacuity, read via /proc BEFORE the run: were the holder's cwd under the
# lane, this arm would silently degenerate into a second copy of F-cwd.
FFD_HOLDER_CWD="$(readlink "/proc/$FFD_HOLDER_PID/cwd" 2>/dev/null || true)"
FFD_HOLDER_FD9="$(readlink "/proc/$FFD_HOLDER_PID/fd/9" 2>/dev/null || true)"
assert "F-fd1: fixture — holder cwd is OUTSIDE the lane (the cwd leg cannot explain the refusal)" \
    bash -c 'case "$1" in "$2"|"$2"/*) exit 1 ;; *) exit 0 ;; esac' _ \
    "$FFD_HOLDER_CWD" "$(readlink -f "$FFD_LANE")"
assert "F-fd2: fixture — fd 9 survived the exec and points under <lane>/target" \
    bash -c 'case "$1" in "$2"/target/*) exit 0 ;; *) exit 1 ;; esac' _ \
    "$FFD_HOLDER_FD9" "$(readlink -f "$FFD_LANE")"

run_helper "$FFD_LANE" --seed-script "$F_SEED_STUB"

assert "F-fd3: an fd-referenced lane exits 75 (open-fd leg alone refuses)" \
    test "$RC" -eq 75
assert "F-fd4: target/DIVERGENT_MARKER byte-intact" \
    test -f "$FFD_LANE/target/DIVERGENT_MARKER"

# ── F-free: DISCRIMINATION, not a blanket freeze ──────────────────────────────
# Without this the gate could regress to "always refuse" and every other arm of
# Block F would stay green. Same fixture shape, no reference of any kind; the
# name shares no prefix with the referenced lanes above.
FQ_ROOT="$(mktemp -d /tmp/test-thin-warm-lane-fq-XXXXXX)"
_TMPDIRS+=("$FQ_ROOT")
FQ_LANE="$FQ_ROOT/_lane-quiet"
mkdir -p "$FQ_LANE/target"
touch "$FQ_LANE/target/DIVERGENT_MARKER"
touch "${FQ_LANE}.lock"

run_helper "$FQ_LANE" --seed-script "$F_SEED_STUB"

assert "F-free1: an unreferenced lane still thins (exit 0) — the gate discriminates" \
    test "$RC" -eq 0
assert "F-free2: <lane>/target is GONE (the free-first rm still runs)" \
    bash -c '[ ! -e "$1" ]' _ "$FQ_LANE/target"
assert "F-free3: stdout is exactly the resolved lane_dir (single-line contract intact)" \
    test "$OUT" = "$(realpath -m "$FQ_LANE")"

# ── F-temp: preserving is TEMPORARY ───────────────────────────────────────────
# Kill and reap the F-cwd holder (so its /proc entry is provably gone — causal,
# not a sleep), then re-run thin on the SAME lane. Only that PID leaves the
# EXIT-trap list: the F-fd holder is still live and must stay registered, and a
# reaped PID may be recycled onto an unrelated process before the trap fires.
_drop_bgpid() {
    local drop="$1"
    local keep=()
    local p
    for p in "${_BGPIDS[@]+${_BGPIDS[@]}}"; do
        [ "$p" = "$drop" ] || keep+=("$p")
    done
    _BGPIDS=("${keep[@]+${keep[@]}}")
}

# Non-vacuity: target/ must have survived the refused run above, or F-temp2
# would go green for the wrong reason.
assert "F-temp0: fixture — <lane>/target survived the first, REFUSED run" \
    test -f "$F_LANE/target/DIVERGENT_MARKER"

kill "$F_HOLDER_PID" 2>/dev/null || true
wait "$F_HOLDER_PID" 2>/dev/null || true
_drop_bgpid "$F_HOLDER_PID"

run_helper "$F_LANE" --seed-script "$F_SEED_STUB"

assert "F-temp1: the same lane thins once its reference clears (exit 0)" \
    test "$RC" -eq 0
assert "F-temp2: <lane>/target is now GONE — a refusal shields for one cycle, not forever" \
    bash -c '[ ! -e "$1" ]' _ "$F_LANE/target"

# ── F-reseed: the gate covers the --reseed path too ───────────────────────────
# The gate sits immediately before the free, so --reseed inherits it: a refusal
# must precede both the free and the seed invocation.
FRS_ROOT="$(mktemp -d /tmp/test-thin-warm-lane-frs-XXXXXX)"
_TMPDIRS+=("$FRS_ROOT")
FRS_LANE="$FRS_ROOT/_lane-rs"
mkdir -p "$FRS_LANE/target"
touch "$FRS_LANE/target/DIVERGENT_MARKER"
touch "${FRS_LANE}.lock"
FRS_BASE="$FRS_ROOT/base-target.gen.1"
mkdir -p "$FRS_BASE"
FRS_READY="$FRS_ROOT/holder.ready"
( cd "$FRS_LANE/target" && touch "$FRS_READY" && exec sleep 300 ) &
FRS_HOLDER_PID=$!
_BGPIDS+=("$FRS_HOLDER_PID")
_wait_for_reader_lock "$FRS_READY" 30

run_helper "$FRS_LANE" --reseed --base "$FRS_BASE" --seed-script "$F_SEED_STUB"

assert "F-reseed1: --reseed against a live-referenced lane exits 75 (the gate covers this branch too)" \
    test "$RC" -eq 75
assert "F-reseed2: target/DIVERGENT_MARKER byte-intact (the refusal precedes the free)" \
    test -f "$FRS_LANE/target/DIVERGENT_MARKER"
assert "F-reseed3: seed-script log is EMPTY (the refusal precedes the stage)" \
    bash -c '[ ! -s "$1" ]' _ "$F_SEED_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Block TRASH: shared-trash litter guard (task 5612). Two asserts, deliberately
# kept as two independently-reported signals: TRASH2 can realistically only ever
# report "clean", which is indistinguishable from a checker that stopped working
# — TRASH1 is the hermetic control proving the instrument still fires.
# Full rationale and honest scope: the CANONICAL WIRING CONTRACT comment in
# tests/infra/test_helpers.sh.
# ─────────────────────────────────────────────────────────────────────────────
assert "TRASH1: shared-trash litter detector is live (self-test fires on a synthetic bare-/tmp lane)" \
    assert_shared_trash_litter_detector_live
assert "TRASH2: no lane in this suite littered the machine-shared /tmp/.reseed-trash" \
    assert_no_shared_trash_litter

test_summary
