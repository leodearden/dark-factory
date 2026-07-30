#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_warm_lane_sizing_lifecycle.sh
# END-TO-END INTEGRATION GATE for the warm-lane pool sizing lifecycle (task
# 5176, PRD docs/prds/warm-lane-pool-sizing-lifecycle.md §9/§10 — task ζ).
#
# Proves the four landed pillars COMPOSE, not just that each works alone:
#   Block A — integration-surface precondition: the actively-driven
#             primitives exist and are executable.
#   Block B (B8)     — release 2 of 3 divergent lanes; thin-on-release (δ)
#             frees their targets while the still-ASSIGNED lane is refused;
#             the audit (α) HEADROOM line reflects the composition (assigned
#             3->1, free 0->2, reclaimable 0->2). Records the MEASURED
#             free-recovered-on-release delta.
#   Block C          — β's §9.2 ADVISORY budget relation: audit HEADROOM
#             divergent_gib <= budget_gib, budget derived from a stubbed df
#             avail + --safety (not the online-grow op, which is
#             host-exclusive/root-only and covered by
#             test_provision_warm_lane_fs.sh).
#   Block D (B6/B6b) — the REAL warm-lane-disk-guard.sh driven across a
#             descending-avail (stubbed df) sequence: soft pressure (exit 3 +
#             sentinel) fires at a strictly higher avail than the hard floor
#             (exit 75); a df fail/garbage fail-closes to 75 with no
#             sentinel, even under --soft.
#
# Bucket: pool (fully hermetic -- mktemp fixtures, stubbed df, real rm/flock
# scoped to fixtures; never touches the real host warm-lane mount, cargo, or
# a reflink FS).
#
# Consumed primitives (all landed; this task does not modify them):
#   scripts/thin-warm-lane.sh, scripts/warm-lane-audit.sh,
#   scripts/warm-lane-disk-guard.sh -- asserted present/executable in Block A.
#   scripts/provision-warm-lane-fs.sh (β) is consumed indirectly, via the
#   audit's budget FORMULA in Block C -- its own --grow op is host-exclusive/
#   root-only and is covered by test_provision_warm_lane_fs.sh, not here.
#
# Per-primitive run helpers capture STDOUT, STDERR, and RC separately:
#   THIN_OUT/THIN_ERR_OUT/THIN_RC     — scripts/thin-warm-lane.sh
#   AUDIT_OUT/AUDIT_ERR_OUT/AUDIT_RC  — scripts/warm-lane-audit.sh
#   GUARD_OUT/GUARD_ERR_OUT/GUARD_RC  — scripts/warm-lane-disk-guard.sh
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (PORTED_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
REPO_ROOT="$DF_REPO_ROOT"
THIN_SCRIPT="$WARM_LANE_SCRIPTS_DIR/thin-warm-lane.sh"
AUDIT_SCRIPT="$WARM_LANE_SCRIPTS_DIR/warm-lane-audit.sh"
GUARD_SCRIPT="$WARM_LANE_SCRIPTS_DIR/warm-lane-disk-guard.sh"
PROVISION_SCRIPT="$WARM_LANE_SCRIPTS_DIR/provision-warm-lane-fs.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=tests/infra/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"

echo "=== warm-lane sizing lifecycle END-TO-END INTEGRATION GATE (task 5176) ==="
echo "(β/provision-warm-lane-fs.sh consumed via the Block C budget formula, not invoked directly: $PROVISION_SCRIPT)"

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

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture scaffolding
# ──────────────────────────────────────────────────────────────────────────────

# make_lane DIR [BRANCH]
# Creates a minimal standalone git repo at DIR (one initial commit on main).
# BRANCH: task/NNNN -> checkout a new branch; DETACH -> detach HEAD;
# main/"" -> stay on main. Each lane is its OWN independent repo (not a
# linked worktree of a shared primary) -- sufficient for warm-lane-audit.sh's
# per-lane git predicates (merge-base/status/symbolic-ref all operate within
# a single lane's own repo). Verbatim copy of
# tests/infra/test_warm_lane_audit.sh's identically-named helper.
make_lane() {
    local dir="$1" branch="${2:-}"
    git init -q -b main "$dir"
    git -C "$dir" config user.email "test@test.local"
    git -C "$dir" config user.name "Test"
    touch "$dir/README.md"
    git -C "$dir" add README.md
    git -C "$dir" commit -q -m "initial"
    case "$branch" in
        task/*)
            git -C "$dir" checkout -q -b "$branch" ;;
        DETACH)
            git -C "$dir" checkout -q --detach ;;
        main|"")
            : ;;
    esac
}

# _wait_for_reader_lock <ready-marker> <deadline-seconds>
# Causal ordering (technique R, docs/prds/infra-test-wallclock-deflake.md,
# task #4847): polls for the READY marker file in 0.05s ticks, returning 0 as
# soon as it appears, or non-zero once the anti-hang deadline elapses. The
# READY marker is touched by a backgrounded lock holder AFTER it acquires its
# flock, so returning 0 causally guarantees the flock is held at the caller's
# next statement -- replacing a fixed `sleep` that races the background
# subshell's lock acquisition under load. This bounded tick ceiling is an
# anti-hang safeguard, not an assert -- it introduces no wallclock upper
# bound (DD5). Verbatim copy of tests/infra/test_warm_lane_audit.sh's
# identically-named helper (itself mirroring test_warm_lane_gc.sh).
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

# _release_holder_body <ready_marker> <release_marker>
# Body of a RELEASE-marker flock holder. Callers background it as
# `( _release_holder_body READY RELEASE ) 9>lock &` (fd 9 supplied by the
# CALLER's redirect, so the flock is acquired directly by that backgrounded
# subshell -- a DIRECT child of this script, so `wait $!` on it later works).
# Once the exclusive flock is acquired, touches <ready_marker> (the causal
# handshake _wait_for_reader_lock polls for), then polls for <release_marker>
# in bounded 0.05s ticks and returns as soon as it appears -- closing fd 9
# (and thus dropping the flock) the instant the subshell exits.
#
# NOT `sleep N` + `kill`: an orphaned `sleep` child inherits fd 9, so killing
# only the parent subshell can leave fd 9 (and the flock) held after the
# "release", flaking a subsequent thin-warm-lane.sh call to a spurious T3
# (exit 75) refusal. `touch RELEASE; wait $pid` instead causally guarantees
# the flock is dropped before the caller's next statement runs. The bounded
# tick ceiling is an anti-hang safeguard, not an assert -- no wallclock upper
# bound (DD5).
_release_holder_body() {
    local ready_marker="$1"
    local release_marker="$2"
    flock -x 9
    touch "$ready_marker"
    local tick=0
    while [ "$tick" -lt 600 ]; do
        [ -f "$release_marker" ] && return 0
        sleep 0.05
        tick=$(( tick + 1 ))
    done
}

# _pool_target_bytes <mount>
# Sums `du -sB1` over every EXISTING <mount>/_lane-*/target dir (0 bytes for
# a lane whose target/ has already been thinned/never existed). Used to
# measure the pool-wide divergent footprint independent of the audit's own
# (GiB-floored) divergent_gib figure -- the B8 freed-bytes delta is measured
# directly, never inferred from a floored/frozen constant (G6/D8).
_pool_target_bytes() {
    local mount="$1"
    local total=0
    local d bytes du_out
    for d in "$mount"/_lane-*/target; do
        [ -d "$d" ] || continue
        du_out="$(du -sB1 "$d" 2>/dev/null)" || continue
        bytes="$(printf '%s\n' "$du_out" | cut -f1)"
        case "$bytes" in
            ''|*[!0-9]*) continue ;;
        esac
        total=$(( total + bytes ))
    done
    printf '%d' "$total"
}

# _headroom_field <audit_stdout> <field_name>
# Extracts <field_name>=<int> from the "^HEADROOM ..." line of a table-format
# warm-lane-audit.sh run. Every field is preceded by whitespace (including
# the first, which follows the literal "HEADROOM" token), so a leading
# [[:space:]] anchor avoids any accidental substring collision (e.g.
# "free=" never matching inside "free_gib=").
_headroom_field() {
    local out="$1" field="$2"
    printf '%s\n' "$out" | grep '^HEADROOM' | grep -oE "[[:space:]]${field}=[0-9]+" | head -1 | cut -d= -f2
}

# _write_audit_df_stub <path> <avail_bytes>
# Writes a 1-col df stub (mimics `df -B1 --output=avail`) at <path> that
# unconditionally emits the given avail byte count. Mirrors
# tests/infra/test_warm_lane_audit.sh Block G's stub shape byte-for-byte
# (baked value, not env-driven -- Block C only ever needs one avail figure
# per stub instance). Wired via REIFY_WARM_LANE_AUDIT_DF.
_write_audit_df_stub() {
    local path="$1" avail_bytes="$2"
    cat > "$path" << EOF
#!/usr/bin/env bash
printf '     Avail\n'
printf '%s\n' "$avail_bytes"
EOF
    chmod +x "$path"
}

# _write_guard_df_stub <path>
# Writes a 2-col df stub (mimics `df -B1 --output=avail,iavail`) at <path>,
# controlled entirely by env vars at INVOCATION time
# (REIFY_TEST_AVAIL_BYTES/REIFY_TEST_AVAIL_INODES/REIFY_TEST_DF_FAIL/
# REIFY_TEST_DF_GARBAGE) -- one stub script serves every regime driven in
# Block D. Byte-for-byte mirrors tests/infra/test_warm_lane_disk_guard.sh's
# stub. Wired via REIFY_WARM_LANE_DISK_GUARD_DF.
_write_guard_df_stub() {
    local path="$1"
    cat > "$path" << 'STUB_EOF'
#!/usr/bin/env bash
if [ "${REIFY_TEST_DF_FAIL:-}" = "1" ]; then
    echo "df: error: permission denied" >&2
    exit 1
fi
if [ "${REIFY_TEST_DF_GARBAGE:-}" = "1" ]; then
    printf '      Avail      IFree\n'
    printf 'not-an-integer not-an-integer\n'
    exit 0
fi
printf '      Avail      IFree\n'
printf ' %s %s\n' \
    "${REIFY_TEST_AVAIL_BYTES:-107374182400}" \
    "${REIFY_TEST_AVAIL_INODES:-1000000}"
STUB_EOF
    chmod +x "$path"
}

# ── per-primitive run helpers: capture OUT/ERR_OUT/RC separately ─────────────
THIN_ERR_FILE="$(mktemp /tmp/test-warm-lane-sizing-thin-err-XXXXXX)"
_TMPDIRS+=("$THIN_ERR_FILE")
AUDIT_ERR_FILE="$(mktemp /tmp/test-warm-lane-sizing-audit-err-XXXXXX)"
_TMPDIRS+=("$AUDIT_ERR_FILE")
GUARD_ERR_FILE="$(mktemp /tmp/test-warm-lane-sizing-guard-err-XXXXXX)"
_TMPDIRS+=("$GUARD_ERR_FILE")

# run_thin <mount> <lane_dir> [thin-warm-lane.sh flags...]
# Invokes thin-warm-lane.sh with REIFY_WARM_LANE_MOUNT=<mount> (so its
# under-mount guard is exercised for real, not skipped), capturing
# THIN_OUT/THIN_ERR_OUT/THIN_RC.
run_thin() {
    local mount="$1"; shift
    local rc=0
    > "$THIN_ERR_FILE"
    THIN_OUT="$(REIFY_WARM_LANE_MOUNT="$mount" bash "$THIN_SCRIPT" "$@" 2>"$THIN_ERR_FILE")" || rc=$?
    THIN_ERR_OUT="$(cat "$THIN_ERR_FILE")"
    THIN_RC=$rc
}

# run_audit <warm-lane-audit.sh args...>
# Captures AUDIT_OUT/AUDIT_ERR_OUT/AUDIT_RC. Callers may prefix inline env
# vars (e.g. REIFY_WARM_LANE_AUDIT_DF=...) before the call.
run_audit() {
    local rc=0
    > "$AUDIT_ERR_FILE"
    AUDIT_OUT="$(bash "$AUDIT_SCRIPT" "$@" 2>"$AUDIT_ERR_FILE")" || rc=$?
    AUDIT_ERR_OUT="$(cat "$AUDIT_ERR_FILE")"
    AUDIT_RC=$rc
}

# run_guard <warm-lane-disk-guard.sh args...>
# Captures GUARD_OUT/GUARD_ERR_OUT/GUARD_RC. Callers prefix inline env vars
# (REIFY_WARM_LANE_DISK_GUARD_DF=... plus its REIFY_TEST_* controls).
run_guard() {
    local rc=0
    > "$GUARD_ERR_FILE"
    GUARD_OUT="$(bash "$GUARD_SCRIPT" "$@" 2>"$GUARD_ERR_FILE")" || rc=$?
    GUARD_ERR_OUT="$(cat "$GUARD_ERR_FILE")"
    GUARD_RC=$rc
}

# Always-unknown status oracle stub: a hermetic, neutral --status-cmd for
# audit runs whose classification does not depend on backing-task status
# (e.g. Block B's lanes are recoverable=LANDED via no-ahead-commit, which
# short-circuits the RECLAIMABLE classification regardless of status).
# Prevents an ambient REIFY_LANE_LEAK_STATUS_CMD from perturbing this
# hermetic gate's output.
NULL_STATUS_DIR="$(mktemp -d /tmp/test-warm-lane-sizing-nullstatus-XXXXXX)"
_TMPDIRS+=("$NULL_STATUS_DIR")
NULL_STATUS_CMD="$NULL_STATUS_DIR/null-status.sh"
cat > "$NULL_STATUS_CMD" << 'STUB_EOF'
#!/usr/bin/env bash
exit 0
STUB_EOF
chmod +x "$NULL_STATUS_CMD"

# ──────────────────────────────────────────────────────────────────────────────
# Block A — integration-surface precondition
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block A: integration-surface precondition ---"

# The three ACTIVELY-DRIVEN primitives (thin/audit/guard are invoked for real
# in Blocks B-D below) must exist and be executable, or this integration gate
# cannot compose what is absent. β/provision-warm-lane-fs.sh is integrated via
# the audit's budget relation in Block C, not a grow op here, so it is kept
# out of this minimal precondition set (see header comment).
assert "A1: scripts/thin-warm-lane.sh exists" test -f "$THIN_SCRIPT"
assert "A2: scripts/thin-warm-lane.sh is executable" test -x "$THIN_SCRIPT"
assert "A3: scripts/warm-lane-audit.sh exists" test -f "$AUDIT_SCRIPT"
assert "A4: scripts/warm-lane-audit.sh is executable" test -x "$AUDIT_SCRIPT"
assert "A5: scripts/warm-lane-disk-guard.sh exists" test -f "$GUARD_SCRIPT"
assert "A6: scripts/warm-lane-disk-guard.sh is executable" test -x "$GUARD_SCRIPT"

# ──────────────────────────────────────────────────────────────────────────────
# Block B (B8) — release-thin bounds resident-divergent + audit headroom
# reflects it
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block B (B8): release-thin bounds resident-divergent + audit headroom reflects it ---"

B_MOUNT="$(mktemp -d /tmp/test-warm-lane-sizing-b-XXXXXX)"
_TMPDIRS+=("$B_MOUNT")

# Blob size: a few MiB per lane -- large enough that `du -sB1` is
# unambiguously nonzero and the freed-bytes assertion is meaningful, small
# enough to stay a fast pool test. Deliberately NOT GiB-scale: divergent_gib
# would floor to a vacuous 0 for a fast fixture either way (see design
# decision -- "headroom reflects it" is proven via the deterministic COUNT
# fields + this test's own `du` delta below, not via divergent_gib).
B_BLOB_KIB=4096   # 4 MiB

_seed_divergent_lane() {
    local dir="$1" branch="$2"
    make_lane "$dir" "$branch"
    mkdir -p "$dir/target"
    dd if=/dev/zero of="$dir/target/blob.bin" bs=1024 count="$B_BLOB_KIB" 2>/dev/null
}

_seed_divergent_lane "$B_MOUNT/_lane-a" "task/5176101"
_seed_divergent_lane "$B_MOUNT/_lane-b" "task/5176102"
_seed_divergent_lane "$B_MOUNT/_lane-c" "task/5176103"

B_LANE_A_BEFORE="$(du -sB1 "$B_MOUNT/_lane-a/target" | cut -f1)"
B_LANE_B_BEFORE="$(du -sB1 "$B_MOUNT/_lane-b/target" | cut -f1)"
B_LANE_C_BEFORE="$(du -sB1 "$B_MOUNT/_lane-c/target" | cut -f1)"
assert "B0a: _lane-a target fixture has nonzero footprint before thinning" test "$B_LANE_A_BEFORE" -gt 0
assert "B0b: _lane-b target fixture has nonzero footprint before thinning" test "$B_LANE_B_BEFORE" -gt 0
assert "B0c: _lane-c target fixture has nonzero footprint before thinning" test "$B_LANE_C_BEFORE" -gt 0

B_POOL_BEFORE="$(_pool_target_bytes "$B_MOUNT")"
assert "B0d: pool footprint BEFORE equals the sum of the three seeded lane footprints" \
    test "$B_POOL_BEFORE" -eq "$(( B_LANE_A_BEFORE + B_LANE_B_BEFORE + B_LANE_C_BEFORE ))"

# Lane locks: _lane-a/_lane-b get a RELEASE-marker holder (cleanly releasable
# before thinning); _lane-c gets a plain sleep-holder that stays ASSIGNED for
# the whole block (killed at block end). All three use the causal
# READY-marker handshake (_wait_for_reader_lock) before the fixture proceeds.
touch "$B_MOUNT/_lane-a.lock" "$B_MOUNT/_lane-b.lock" "$B_MOUNT/_lane-c.lock"
B_READY_A="$B_MOUNT/_lane-a.lock.ready-marker"
B_RELEASE_A="$B_MOUNT/_lane-a.lock.release-marker"
B_READY_B="$B_MOUNT/_lane-b.lock.ready-marker"
B_RELEASE_B="$B_MOUNT/_lane-b.lock.release-marker"
B_READY_C="$B_MOUNT/_lane-c.lock.ready-marker"

( _release_holder_body "$B_READY_A" "$B_RELEASE_A" ) 9>"$B_MOUNT/_lane-a.lock" &
B_PID_A=$!
_BGPIDS+=("$B_PID_A")

( _release_holder_body "$B_READY_B" "$B_RELEASE_B" ) 9>"$B_MOUNT/_lane-b.lock" &
B_PID_B=$!
_BGPIDS+=("$B_PID_B")

( flock -x 9 && touch "$B_READY_C" && sleep 300 ) 9>"$B_MOUNT/_lane-c.lock" &
B_PID_C=$!
_BGPIDS+=("$B_PID_C")

_wait_for_reader_lock "$B_READY_A" 30
_wait_for_reader_lock "$B_READY_B" 30
_wait_for_reader_lock "$B_READY_C" 30

# BEFORE: all three lanes have a LIVE consumer holding the lane lock; none
# free/reclaimable yet. This block's fixture is a flock holder and nothing
# else -- it writes no .lane-state/ record -- so the honest column to assert
# here is `live`, NOT `assigned`. Post-#5363 those are distinct questions:
# `live` is the shared-flock probe (is a consumer process running?) while
# `assigned` cross-cuts it from the orchestrator's own assignment state (is
# the lane reserved?). Asserting `assigned=3` against a lock-only fixture was
# the same liveness-as-assignment conflation #5363 exists to retire; with no
# state records present the honest reading is assigned=0, live=3.
run_audit --mount "$B_MOUNT" --status-cmd "$NULL_STATUS_CMD"
assert "B1: audit BEFORE exit 0" test "$AUDIT_RC" -eq 0
assert "B2: audit BEFORE HEADROOM live=3" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "live=3"' _ "$AUDIT_OUT"
assert "B3: audit BEFORE HEADROOM free=0" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "free=0"' _ "$AUDIT_OUT"
assert "B4: audit BEFORE HEADROOM reclaimable=0" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "reclaimable=0"' _ "$AUDIT_OUT"
assert "B5: audit BEFORE _lane-a classification LIVE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-a .*classification=LIVE"' _ "$AUDIT_OUT"
assert "B6: audit BEFORE _lane-c classification LIVE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-c .*classification=LIVE"' _ "$AUDIT_OUT"

B_DIVERGENT_GIB_BEFORE="$(_headroom_field "$AUDIT_OUT" divergent_gib)"
assert "B7: audit BEFORE divergent_gib is a well-formed non-negative integer" \
    bash -c '[[ "$1" =~ ^[0-9]+$ ]]' _ "$B_DIVERGENT_GIB_BEFORE"

# Release _lane-a and _lane-b: touch their RELEASE markers, then `wait` each
# PID -- a causal guarantee the flock is dropped (fd 9 closed) before the
# next statement runs (see _release_holder_body). _lane-c stays held.
touch "$B_RELEASE_A" "$B_RELEASE_B"
wait "$B_PID_A"
wait "$B_PID_B"

# Thin the two released lanes via the REAL primitive (REIFY_WARM_LANE_MOUNT
# set so its under-mount guard is exercised for real, not skipped).
run_thin "$B_MOUNT" "$B_MOUNT/_lane-a"
assert "B8: thin _lane-a (released) exits 0" test "$THIN_RC" -eq 0
assert "B9: _lane-a/target is GONE" bash -c '[ ! -e "$1" ]' _ "$B_MOUNT/_lane-a/target"

run_thin "$B_MOUNT" "$B_MOUNT/_lane-b"
assert "B10: thin _lane-b (released) exits 0" test "$THIN_RC" -eq 0
assert "B11: _lane-b/target is GONE" bash -c '[ ! -e "$1" ]' _ "$B_MOUNT/_lane-b/target"

# _lane-c is still ASSIGNED (ready-holder never released) -- thin refuses it
# (T3, exit 75); target/ stays byte-intact. This is the "resident-divergent
# tracks the still-ASSIGNED set" proof: the still-live lane's divergent bytes
# are NEVER touched by a release-triggered thin sweep.
run_thin "$B_MOUNT" "$B_MOUNT/_lane-c"
assert "B12: thin _lane-c (still ASSIGNED) exits 75 (T3 refusal)" test "$THIN_RC" -eq 75
assert "B13: _lane-c/target intact after the refused thin" test -f "$B_MOUNT/_lane-c/target/blob.bin"

B_LANE_C_AFTER="$(du -sB1 "$B_MOUNT/_lane-c/target" | cut -f1)"
assert "B14: _lane-c/target footprint unchanged by the refused thin" \
    test "$B_LANE_C_AFTER" -eq "$B_LANE_C_BEFORE"

B_POOL_AFTER="$(_pool_target_bytes "$B_MOUNT")"
B_FREED=$(( B_POOL_BEFORE - B_POOL_AFTER ))
B_EXPECTED_FREED=$(( B_LANE_A_BEFORE + B_LANE_B_BEFORE ))

assert "B15: pool footprint AFTER equals _lane-c's own (untouched) footprint" \
    test "$B_POOL_AFTER" -eq "$B_LANE_C_BEFORE"
assert "B16: freed bytes == sum of the released a+b footprints (release-thin bounds resident-divergent to the still-assigned set)" \
    test "$B_FREED" -eq "$B_EXPECTED_FREED"
assert "B17: freed bytes >= one full lane footprint (>=1x; ~2x-per-lane B8 headline)" \
    test "$B_FREED" -ge "$B_LANE_A_BEFORE"

echo "RECOVERED-ON-RELEASE-DELTA: before=${B_POOL_BEFORE} after=${B_POOL_AFTER} freed=${B_FREED}"

# AFTER: live 3->1 (a+b dropped their locks; c still holds), free 0->2,
# reclaimable 0->2 (a+b are FREE + LANDED, via make_lane's no-ahead-commit
# branch -- recoverable=LANDED short-circuits RECLAIMABLE regardless of
# backing-task status). Asserted on `live` for the same reason as B2: this
# fixture carries no .lane-state/ record, so `assigned` is 0 throughout and
# the lock drop is only observable on the liveness column.
run_audit --mount "$B_MOUNT" --status-cmd "$NULL_STATUS_CMD"
assert "B18: audit AFTER exit 0" test "$AUDIT_RC" -eq 0
assert "B19: audit AFTER HEADROOM live=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "live=1"' _ "$AUDIT_OUT"
assert "B20: audit AFTER HEADROOM free=2" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "free=2"' _ "$AUDIT_OUT"
assert "B21: audit AFTER HEADROOM reclaimable=2" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "reclaimable=2"' _ "$AUDIT_OUT"
assert "B22: audit AFTER HEADROOM leaked=0" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "leaked=0"' _ "$AUDIT_OUT"
assert "B23: audit AFTER _lane-a classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-a .*classification=RECLAIMABLE"' _ "$AUDIT_OUT"
assert "B24: audit AFTER _lane-b classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-b .*classification=RECLAIMABLE"' _ "$AUDIT_OUT"
assert "B25: audit AFTER _lane-c classification LIVE (still assigned, untouched)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-c .*classification=LIVE"' _ "$AUDIT_OUT"

B_DIVERGENT_GIB_AFTER="$(_headroom_field "$AUDIT_OUT" divergent_gib)"
assert "B26: audit AFTER divergent_gib is a well-formed non-negative integer" \
    bash -c '[[ "$1" =~ ^[0-9]+$ ]]' _ "$B_DIVERGENT_GIB_AFTER"
assert "B27: audit AFTER divergent_gib non-increasing vs BEFORE (headroom reflects the release-thin)" \
    test "$B_DIVERGENT_GIB_AFTER" -le "$B_DIVERGENT_GIB_BEFORE"

# _lane-c's holder is never released mid-block by design; kill it now and
# clear _BGPIDS so the final EXIT trap doesn't try to double-kill an already-
# reaped pid.
kill "$B_PID_C" 2>/dev/null || true
wait "$B_PID_C" 2>/dev/null || true
_BGPIDS=()

# ──────────────────────────────────────────────────────────────────────────────
# Block C — β ADVISORY budget relation (§9.2: "meant to be asserted by ζ")
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block C: β ADVISORY budget relation (audit HEADROOM divergent_gib <= budget_gib) ---"

C_MOUNT="$(mktemp -d /tmp/test-warm-lane-sizing-c-XXXXXX)"
_TMPDIRS+=("$C_MOUNT")

# A small, independent fixture (Block C does not depend on Block B's state):
# one lane with a target/ (nonzero divergent contribution) plus one bare
# lane, exercising the audit's per-lane divergent accumulation.
make_lane "$C_MOUNT/_lane-x"
mkdir -p "$C_MOUNT/_lane-x/target"
dd if=/dev/zero of="$C_MOUNT/_lane-x/target/blob.bin" bs=1024 count=1024 2>/dev/null
make_lane "$C_MOUNT/_lane-y"

# Stub df: emits a KNOWN, deliberately non-round avail byte count (mirrors
# test_warm_lane_audit.sh Block G) so the audit's floor-to-GiB truncation is
# exercised for real, not a coincidental passthrough.
C_DF_STUB_DIR="$(mktemp -d /tmp/test-warm-lane-sizing-c-dfstub-XXXXXX)"
_TMPDIRS+=("$C_DF_STUB_DIR")
C_AVAIL_BYTES=$(( 40 * 1024 * 1024 * 1024 + 987654321 ))
C_DF_STUB="$C_DF_STUB_DIR/df-fake.sh"
_write_audit_df_stub "$C_DF_STUB" "$C_AVAIL_BYTES"

# The test COMPUTES the expected free_gib/budget_gib from the SAME stub
# input + the SAME --safety knob passed to the script -- a derived relation,
# never a hardcoded literal (G6/D8 no-frozen-constant rule).
C_SAFETY=5
C_EXPECTED_FREE_GIB=$(( C_AVAIL_BYTES / 1073741824 ))
C_EXPECTED_BUDGET_GIB=$(( C_EXPECTED_FREE_GIB / C_SAFETY ))
assert "C0: chosen avail/safety yield a non-vacuous budget (budget_gib > 0)" \
    test "$C_EXPECTED_BUDGET_GIB" -gt 0

REIFY_WARM_LANE_AUDIT_DF="$C_DF_STUB" run_audit --mount "$C_MOUNT" --safety "$C_SAFETY" --status-cmd "$NULL_STATUS_CMD"
assert "C1: audit exit 0" test "$AUDIT_RC" -eq 0

C_FREE_GIB="$(_headroom_field "$AUDIT_OUT" free_gib)"
C_BUDGET_GIB="$(_headroom_field "$AUDIT_OUT" budget_gib)"
C_DIVERGENT_GIB="$(_headroom_field "$AUDIT_OUT" divergent_gib)"

assert "C2: HEADROOM free_gib == floor(stub_avail_bytes / 2^30) (derived, not frozen)" \
    test "$C_FREE_GIB" -eq "$C_EXPECTED_FREE_GIB"
assert "C3: HEADROOM budget_gib == floor(free_gib / safety) (β formula, derived from the stub + --safety knob)" \
    test "$C_BUDGET_GIB" -eq "$C_EXPECTED_BUDGET_GIB"
assert "C4: HEADROOM divergent_gib is a well-formed non-negative integer" \
    bash -c '[[ "$1" =~ ^[0-9]+$ ]]' _ "$C_DIVERGENT_GIB"
assert "C5: §9.2 ADVISORY relation holds: divergent_gib <= budget_gib" \
    test "$C_DIVERGENT_GIB" -le "$C_BUDGET_GIB"

# ──────────────────────────────────────────────────────────────────────────────
# Block D (B6 + B6b) — soft floor fires before hard floor, end-to-end
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block D (B6+B6b): soft floor fires before hard floor, end-to-end ---"

D_STUB_DIR="$(mktemp -d /tmp/test-warm-lane-sizing-d-stub-XXXXXX)"
_TMPDIRS+=("$D_STUB_DIR")
D_DF_STUB="$D_STUB_DIR/df_stub"
_write_guard_df_stub "$D_DF_STUB"

D_MOUNT="$(mktemp -d /tmp/test-warm-lane-sizing-d-XXXXXX)"
_TMPDIRS+=("$D_MOUNT")

# Guard floors: hard (H) < soft (S), on both the bytes and inodes axes.
D_HARD_GIB=10
D_SOFT_GIB=20
D_HARD_INODES=100000
D_SOFT_INODES=200000

# Three DERIVED descending-avail regimes (no frozen RED GB constant -- G6/D8):
#   healthy = soft + margin        (>= soft floor on both axes)
#   soft    = midpoint(hard, soft) (H <= avail < S -- guaranteed between)
#   hard    = hard - 1             (< H on both axes)
D_HEALTHY_GIB=$(( D_SOFT_GIB + 5 ))
D_SOFT_AVAIL_GIB=$(( (D_HARD_GIB + D_SOFT_GIB) / 2 ))
D_HARD_AVAIL_GIB=$(( D_HARD_GIB - 1 ))

D_HEALTHY_INODES=$(( D_SOFT_INODES + 50000 ))
D_SOFT_AVAIL_INODES=$(( (D_HARD_INODES + D_SOFT_INODES) / 2 ))
D_HARD_AVAIL_INODES=$(( D_HARD_INODES - 1 ))

# Sanity on the derivation itself (regression guard for the arithmetic, not
# the guard-under-test): the soft regime sits strictly between the two
# floors, and the hard regime sits strictly below the hard floor -- i.e. the
# soft-fires-before-hard ordering the B6 headline requires.
assert "D0a: derived soft-avail regime sits within [hard, soft) (bytes)" \
    bash -c '[ "$1" -ge "$2" ] && [ "$1" -lt "$3" ]' _ "$D_SOFT_AVAIL_GIB" "$D_HARD_GIB" "$D_SOFT_GIB"
assert "D0b: derived hard-avail regime sits below the hard floor (bytes)" \
    test "$D_HARD_AVAIL_GIB" -lt "$D_HARD_GIB"
assert "D0c: soft-avail regime is strictly greater than hard-avail regime (soft fires first as free shrinks)" \
    test "$D_SOFT_AVAIL_GIB" -gt "$D_HARD_AVAIL_GIB"

D_GUARD_FLAGS=(--mount "$D_MOUNT" --min-free-gib "$D_HARD_GIB" --soft-free-gib "$D_SOFT_GIB" \
    --min-free-inodes "$D_HARD_INODES" --soft-free-inodes "$D_SOFT_INODES")

# run_guard_df <avail_gib> <avail_inodes> <guard args...>
# Wires the 2-col df stub with the given avail figures (GiB bytes + raw
# inodes) via REIFY_WARM_LANE_DISK_GUARD_DF, capturing
# GUARD_OUT/GUARD_ERR_OUT/GUARD_RC.
run_guard_df() {
    local avail_gib="$1" avail_inodes="$2"; shift 2
    local avail_bytes=$(( avail_gib * 1024 * 1024 * 1024 ))
    local rc=0
    > "$GUARD_ERR_FILE"
    GUARD_OUT="$(
        REIFY_WARM_LANE_DISK_GUARD_DF="$D_DF_STUB" \
        REIFY_TEST_AVAIL_BYTES="$avail_bytes" \
        REIFY_TEST_AVAIL_INODES="$avail_inodes" \
            bash "$GUARD_SCRIPT" "$@" 2>"$GUARD_ERR_FILE"
    )" || rc=$?
    GUARD_ERR_OUT="$(cat "$GUARD_ERR_FILE")"
    GUARD_RC=$rc
}

# run_guard_dffail / run_guard_dfgarbage <guard args...>
# Drive the E3 fail-closed paths (df exits non-zero / emits unparseable
# output); the stub short-circuits on these branches before ever reading the
# avail env vars.
run_guard_dffail() {
    local rc=0
    > "$GUARD_ERR_FILE"
    GUARD_OUT="$(
        REIFY_WARM_LANE_DISK_GUARD_DF="$D_DF_STUB" REIFY_TEST_DF_FAIL=1 \
            bash "$GUARD_SCRIPT" "$@" 2>"$GUARD_ERR_FILE"
    )" || rc=$?
    GUARD_ERR_OUT="$(cat "$GUARD_ERR_FILE")"
    GUARD_RC=$rc
}
run_guard_dfgarbage() {
    local rc=0
    > "$GUARD_ERR_FILE"
    GUARD_OUT="$(
        REIFY_WARM_LANE_DISK_GUARD_DF="$D_DF_STUB" REIFY_TEST_DF_GARBAGE=1 \
            bash "$GUARD_SCRIPT" "$@" 2>"$GUARD_ERR_FILE"
    )" || rc=$?
    GUARD_ERR_OUT="$(cat "$GUARD_ERR_FILE")"
    GUARD_RC=$rc
}

# healthy: avail >= soft on both axes -> `check --soft` exits 0, stdout empty.
run_guard_df "$D_HEALTHY_GIB" "$D_HEALTHY_INODES" check --soft "${D_GUARD_FLAGS[@]}"
assert "D1: healthy regime, check --soft exits 0" test "$GUARD_RC" -eq 0
assert "D1: healthy regime stdout empty" bash -c '[ -z "$1" ]' _ "$GUARD_OUT"

# soft: hard <= avail < soft -> hard `check` (no --soft) exits 0; `check
# --soft` exits 3 with the throttle-not-requeue sentinel.
run_guard_df "$D_SOFT_AVAIL_GIB" "$D_SOFT_AVAIL_INODES" check "${D_GUARD_FLAGS[@]}"
assert "D2: soft regime, hard check (no --soft) exits 0" test "$GUARD_RC" -eq 0

run_guard_df "$D_SOFT_AVAIL_GIB" "$D_SOFT_AVAIL_INODES" check --soft "${D_GUARD_FLAGS[@]}"
assert "D3: soft regime, check --soft exits 3" test "$GUARD_RC" -eq 3
assert "D3: stdout carries the soft-pressure sentinel" \
    bash -c 'printf "%s\n" "$1" | grep -q "@@REIFY_WARM_LANE_SOFT_PRESSURE@@"' _ "$GUARD_OUT"
assert "D3: sentinel carries free_gib=" \
    bash -c 'printf "%s\n" "$1" | grep -q "free_gib="' _ "$GUARD_OUT"
assert "D3: sentinel carries budget_gib=" \
    bash -c 'printf "%s\n" "$1" | grep -q "budget_gib="' _ "$GUARD_OUT"

# hard: avail < hard -> BOTH `check` and `check --soft` exit 75; sentinel
# ABSENT (fail-closed/below-hard-floor never emits the soft-pressure line).
run_guard_df "$D_HARD_AVAIL_GIB" "$D_HARD_AVAIL_INODES" check "${D_GUARD_FLAGS[@]}"
assert "D4: hard regime, check (no --soft) exits 75" test "$GUARD_RC" -eq 75
assert "D4: stdout empty (no sentinel), hard check" bash -c '[ -z "$1" ]' _ "$GUARD_OUT"

run_guard_df "$D_HARD_AVAIL_GIB" "$D_HARD_AVAIL_INODES" check --soft "${D_GUARD_FLAGS[@]}"
assert "D5: hard regime, check --soft exits 75" test "$GUARD_RC" -eq 75
assert "D5: stdout empty (no sentinel), hard regime under --soft" bash -c '[ -z "$1" ]' _ "$GUARD_OUT"

# B6b (E3 fail-closed): a df failure or garbage output -> BOTH check and
# check --soft exit 75, sentinel absent -- never 0, never the sentinel, even
# under --soft (E3 precedence over the soft gate).
run_guard_dffail check "${D_GUARD_FLAGS[@]}"
assert "D6: df failure, hard check exits 75" test "$GUARD_RC" -eq 75
assert "D6: stdout empty on df failure" bash -c '[ -z "$1" ]' _ "$GUARD_OUT"

run_guard_dffail check --soft "${D_GUARD_FLAGS[@]}"
assert "D7: df failure under --soft exits 75 (never the sentinel)" test "$GUARD_RC" -eq 75
assert "D7: stdout empty on df failure under --soft" bash -c '[ -z "$1" ]' _ "$GUARD_OUT"

run_guard_dfgarbage check "${D_GUARD_FLAGS[@]}"
assert "D8: garbage df output, hard check exits 75" test "$GUARD_RC" -eq 75
assert "D8: stdout empty on garbage df output" bash -c '[ -z "$1" ]' _ "$GUARD_OUT"

run_guard_dfgarbage check --soft "${D_GUARD_FLAGS[@]}"
assert "D9: garbage df output under --soft exits 75 (never the sentinel)" test "$GUARD_RC" -eq 75
assert "D9: stdout empty on garbage df output under --soft" bash -c '[ -z "$1" ]' _ "$GUARD_OUT"

test_summary
