#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_warm_lane_gc_sweep.sh
# Hermetic tests for orchestrator/scripts/warm-lane/warm-lane-gc-sweep.sh (task 4863).
#
# Blocks:
#   A — CLI guard: --help exits 0 and prints usage; unknown flag exits 2; a
#         MISSING sibling scripts/lib_live_refs.sh fails LOUD with exit 2
#         (wiring error) before argv is parsed (task 5572 review)
#   B — fail-open: non-existent --mount dir → exit 0, warn on stderr, gc-script NOT invoked
#   C — happy path: existing --mount dir → exit 0, gc-script invoked as
#         reclaim --mount <dir>
#   V — part-1 verification / green-on-arrival regression guards:
#         warm-lane-disk-guard.sh --help mentions --mount/--min-free-gib/--min-free-inodes
#           (tests --help text only; acceptance via help prose, not live flag invocation)
#   W — cross-script --mount seam (green-on-arrival regression guard):
#         single DF value (worktrees-dir), agreeing semantics across gc.sh + disk-guard.sh
#         pins: exit 1 on unresolvable base-target (not exit 2); stderr names correct
#         sibling-derived path; df volume agreement between --mount value and base-target
#   G — emergency low-water trigger (task 5168): df measurement below a
#         critical free-GiB floor appends --disk-pressure to the reclaim
#         invocation (engaging the 5167 rm-outright mechanism); above the
#         floor or on a df failure, plain reclaim runs unchanged (fail-to-α);
#         end-to-end proof via the real gc.sh that target/ is deleted
#         outright, with the live-consumer flock still honored (a dirty POOL
#         LANE is now reclaimed under always-reclaim, task 5326).
#         Also covers the --critical-free-gib integer-format guard (exit 2
#         on a non-numeric value, via flag or env) and the df-succeeds-but
#         -unparseable-output fallback (a code path distinct from a df exit
#         failure).
#   T — stale .reseed-trash reaper (task 5326): the sweep reaps a stranded
#         $MOUNT/.reseed-trash/<lane>.<pid> entry iff its encoded PID is dead
#         AND no live process references the dir via cwd/fd/maps; a live
#         encoded PID, a live cwd/fd/maps reference, or a malformed (non-
#         numeric) suffix preserves it; a missing .reseed-trash/ dir is a
#         no-op; and an rm failure is fail-open (warn, exit 0, entry intact).
#         The reaper runs every sweep, after the mount-exists fail-open check
#         and before the reclaim exec, independent of --disk-pressure. T7
#         covers the batched single-/proc-scan path: several dead-PID
#         candidates in ONE sweep, exactly the live-cwd-referenced one
#         preserved while the rest are reaped (discrimination within a single
#         scan, not all-or-nothing).
#   U — live-consumer lane guard (task 5378, relocated by task 5572): a lane
#         with an active build is preserved despite a FREE flock — the exact
#         esc-5375-1 gap — while a free unreferenced lane is still reset (no
#         blanket over-preserve) and, once the reference clears, a later sweep
#         reclaims it (preserve is temporary, mirrors T3).
#         U1-U7 drive the REAL gc.sh end to end. They are the COMPOSITION proof
#         and must NOT be deleted as redundant with test_warm_lane_gc.sh Block
#         P: under task 5572 they stay green by a NEW mechanism — the
#         primitive's own per-lane check — rather than the wrapper CSV.
#         U8-U13 are the stub sub-cases over the argv the sweep hands gc.sh.
#         They now assert the argv carries NO --extra-protect-glob AT ALL,
#         whether or not a lane is live: the guard moved into gc.sh, so a
#         wrapper-side snapshot would be both redundant and a reintroduced
#         TOCTOU.
#         The mmap substring-boundary regression (esc-5378 review) MOVED to
#         tests/infra/test_warm_lane_gc.sh Block R with task 5572: its vehicle
#         here was the up-front CSV, which is gone, and the boundary matcher now
#         lives in scripts/lib_live_refs.sh called by gc.sh directly.
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (PORTED_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
SCRIPT="$WARM_LANE_SCRIPTS_DIR/warm-lane-gc-sweep.sh"
DISK_GUARD="$WARM_LANE_SCRIPTS_DIR/warm-lane-disk-guard.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=tests/infra/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"

# The SHIPPED lane-state lib, sourced for ONE value: $LANE_PROTECT_GLOB_FALLBACK
# (see the BRIDGE-COST SEAM comment above run_sweep). Read from the lib rather
# than written here as a literal for the same reason as in the sibling gc suite:
# a literal would be a hand-maintained mirror of PROTECTED_PREFIXES sitting
# outside TestProtectGlobFallbackDrift's gate.
# shellcheck source=orchestrator/scripts/warm-lane/lib_lane_state.sh
source "$WARM_LANE_SCRIPTS_DIR/lib_lane_state.sh"

echo "=== scripts/warm-lane-gc-sweep.sh hermetic tests (task 4863) ==="

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

ERR_FILE="$(mktemp /tmp/test-gc-sweep-err-XXXXXX)"
_TMPDIRS+=("$ERR_FILE")

# ── run_sweep: invoke the sweep script, capture OUT/ERR_OUT/RC ────────────────
#
# BRIDGE-COST SEAM (task 3292) — the same seam as the sibling gc suite's
# run_helper; see the long rationale there (tests/warm-lane/test_warm_lane_gc.sh).
# Kept as one idiom on purpose so the two files do not drift into two unrelated
# ones.
#
# warm-lane-gc.sh's PROTECT_GLOB default now renders dark-factory's
# PROTECTED_PREFIXES through a python3 that imports pydantic, once per gc.sh
# INVOCATION. FOUR cases here drive the REAL gc.sh via --gc-script "$GC_REAL"
# (G4, G5, and the two U cases) and so would each pay a full render; every other
# run_sweep either passes a stub --gc-script or exits at the CLI guard, and is
# unaffected either way. Pinning gc.sh's own already-documented
# REIFY_WARM_LANE_GC_PROTECT_GLOB short-circuits that render before it runs. The
# env propagates unchanged through warm-lane-gc-sweep.sh's terminal
# `exec "$GC_SCRIPT" …` to gc.sh.
#
# NO opt-out helper is needed here, unlike the gc suite's
# run_helper_live_default: NO assert in this file inspects the protect glob. The
# four real-gc cases are proving the sweep -> gc handoff and the disk-pressure
# end-to-end path; the stub cases assert gc.sh's ARGV (`reclaim --mount …`),
# which an environment variable does not appear in. If a future block here does
# assert what the DEFAULT protects, it must bypass this pin — the gc suite's
# run_helper_live_default is the pattern to copy.
run_sweep() {
    local rc=0
    > "$ERR_FILE"
    OUT="$(REIFY_WARM_LANE_GC_PROTECT_GLOB="$LANE_PROTECT_GLOB_FALLBACK" \
        bash "$SCRIPT" "$@" 2>"$ERR_FILE")" || rc=$?
    ERR_OUT="$(cat "$ERR_FILE")"
    RC=$rc
}

# ── shared scaffolding for Block G (emergency low-water trigger, task 5168) ───
# This file does not source tests/infra/test_warm_lane_gc.sh, so the pieces
# needed from it (make_repo, a thinning seed-script stub, the causal
# live-consumer-lock handshake) are reproduced locally, mirroring it exactly.

# make_repo DIR — initialize a minimal git repo with one commit on main.
make_repo() {
    local dir="$1"
    git init -q -b main "$dir"
    git -C "$dir" config user.email "test@test.local"
    git -C "$dir" config user.name "Test"
    touch "$dir/README.md"
    git -C "$dir" add README.md
    git -C "$dir" commit -q -m "initial"
}

# _df_stub OUTFILE AVAIL_BYTES — writes a df-lookalike stub emitting
# `df -B1 --output=avail`-style output (line 1 a header, line 2 the avail
# byte count), ignoring its invocation args. Block G's branch selection is
# driven entirely by this injected value vs an explicit --critical-free-gib,
# never by real disk state.
_df_stub() {
    local outfile="$1" avail="$2"
    cat > "$outfile" << EOF
#!/usr/bin/env bash
printf 'Avail\n%s\n' '$avail'
exit 0
EOF
    chmod +x "$outfile"
}

# _df_fail_stub OUTFILE — writes a df stub that always fails (simulates a
# df measurement failure: wrong path, non-GNU df, transient I/O error, etc.).
_df_fail_stub() {
    local outfile="$1"
    cat > "$outfile" << 'EOF'
#!/usr/bin/env bash
exit 1
EOF
    chmod +x "$outfile"
}

# _df_garbage_stub OUTFILE — writes a df stub that EXITS 0 (succeeds) but
# emits a non-numeric avail field (e.g. a locale/column mismatch), distinct
# from _df_fail_stub's non-zero exit: this drives the sweep's separate
# "df succeeded but output failed the numeric guard" fallback branch.
_df_garbage_stub() {
    local outfile="$1"
    cat > "$outfile" << 'EOF'
#!/usr/bin/env bash
printf 'Avail\nN/A\n'
exit 0
EOF
    chmod +x "$outfile"
}

# _thinning_seed_stub_body — printed to stdout; redirect into a seed-script
# stub file. Logs its argv to $SEED_LOG and removes the lane's divergent
# marker (simulates the α reflink-reseed's thinning effect). Used only to
# prove it is NOT invoked when --disk-pressure short-circuits α.
# Mirrors tests/infra/test_warm_lane_gc.sh's _seed_stub_body.
_thinning_seed_stub_body() {
    cat << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
}

# _wait_for_reader_lock <ready-marker> <deadline-seconds>
# Causal ordering (technique R, docs/prds/infra-test-wallclock-deflake.md,
# task #4847): polls for the READY marker file in 0.05s ticks, returning 0
# as soon as it appears, or non-zero once the anti-hang deadline elapses.
# The READY marker is touched by a backgrounded lock holder AFTER it
# acquires its flock, so callers only proceed once the flock is provably
# held — replacing a fixed `sleep` that would race the background
# subshell's lock acquisition under load. Mirrors
# tests/infra/test_warm_lane_gc.sh's identically-named helper.
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

# ──────────────────────────────────────────────────────────────────────────────
# Block A — CLI guard
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block A: CLI guard ---"

# A1: --help exits 0
run_sweep --help
assert "A1: --help exits 0" test "$RC" -eq 0

# A2: --help prints usage on stderr
assert "A2: --help prints usage on stderr" \
    bash -c 'printf "%s\n" "$1" | grep -qi "usage"' _ "$ERR_OUT"

# A3: unknown flag exits 2
run_sweep --unknown-flag-xyz
assert "A3: unknown flag exits 2" test "$RC" -eq 2

# A4: a MISSING sibling scripts/lib_live_refs.sh is a fail-LOUD wiring error
# (task 5572 review). Mirrors test_warm_lane_gc.sh's A9 — both consumers of the
# shared scanner must report an incomplete deployment the same way, or an
# operator reading one script's exit code learns the wrong taxonomy for the
# other. The sweep is copied ALONE into a temp dir and invoked with --help,
# which normally exits 0: the guard fires before argv parsing, so exit 2 pins
# both the code and the ordering.
A4_DIR="$(mktemp -d /tmp/test-gc-sweep-a4-XXXXXX)"
_TMPDIRS+=("$A4_DIR")
cp "$SCRIPT" "$A4_DIR/warm-lane-gc-sweep.sh"
assert "A4: fixture — the copy really has no sibling lib_live_refs.sh" \
    bash -c '[ ! -e "$1/lib_live_refs.sh" ]' _ "$A4_DIR"
A4_RC=0
A4_ERR="$(bash "$A4_DIR/warm-lane-gc-sweep.sh" --help 2>&1 >/dev/null)" || A4_RC=$?
assert "A4: missing sibling lib_live_refs.sh exits 2 (wiring error, not runtime 1)" \
    test "$A4_RC" -eq 2
assert "A4: stderr names the missing library (fail LOUD, not silent)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "lib_live_refs.sh not found"' _ "$A4_ERR"

# ──────────────────────────────────────────────────────────────────────────────
# Block B — fail-open: non-existent --mount dir
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block B: fail-open (non-existent mount dir) ---"

B_ROOT="$(mktemp -d /tmp/test-gc-sweep-b-XXXXXX)"
_TMPDIRS+=("$B_ROOT")

# gc-script stub that logs calls
B_GC_LOG="$B_ROOT/gc_calls.log"
B_GC_STUB="$B_ROOT/gc_stub.sh"
cat > "$B_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
chmod +x "$B_GC_STUB"

B_NONEXISTENT="$B_ROOT/does/not/exist/worktrees"

GC_LOG="$B_GC_LOG" run_sweep --mount "$B_NONEXISTENT" --gc-script "$B_GC_STUB"

# B1: exits 0 (fail-open)
assert "B1: non-existent --mount dir exits 0 (fail-open)" test "$RC" -eq 0

# B2: warn/skip message on stderr
assert "B2: stderr contains warn/skip message for non-existent mount" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "warn|skip|not exist|no such|missing"' _ "$ERR_OUT"

# B3: gc-script NOT invoked (no calls log file or empty)
assert "B3: gc-script NOT invoked when mount dir does not exist" \
    bash -c '[ ! -f "$1" ] || [ ! -s "$1" ]' _ "$B_GC_LOG"

# ──────────────────────────────────────────────────────────────────────────────
# Block C — happy path: existing --mount dir
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block C: happy path (existing mount dir) ---"

C_ROOT="$(mktemp -d /tmp/test-gc-sweep-c-XXXXXX)"
_TMPDIRS+=("$C_ROOT")

C_MOUNT="$C_ROOT/worktrees"
mkdir -p "$C_MOUNT"

# gc-script stub that logs argv
C_GC_LOG="$C_ROOT/gc_calls.log"
C_GC_STUB="$C_ROOT/gc_stub.sh"
cat > "$C_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
chmod +x "$C_GC_STUB"

GC_LOG="$C_GC_LOG" run_sweep --mount "$C_MOUNT" --gc-script "$C_GC_STUB"

# C1: exits 0
assert "C1: happy path exits 0" test "$RC" -eq 0

# C2: gc-script was invoked
assert "C2: gc-script was invoked" test -f "$C_GC_LOG"

# C3: gc-script invoked as "reclaim --mount <dir>"
assert "C3: gc-script invoked with 'reclaim --mount <mount>'" \
    bash -c 'grep -qE "^reclaim --mount " "$1"' _ "$C_GC_LOG"

# C4: the --mount argument passed to gc-script matches the sweep's --mount
assert "C4: gc-script received correct --mount path" \
    bash -c 'grep -qF "reclaim --mount $2" "$1"' _ "$C_GC_LOG" "$C_MOUNT"

# C5: sweep propagates non-zero exit code from gc.sh (fail-open applies ONLY
# to the missing-mount case; a gc.sh error must propagate so the systemd timer
# marks the unit as failed rather than silently swallowing the error).
C5_ROOT="$(mktemp -d /tmp/test-gc-sweep-c5-XXXXXX)"
_TMPDIRS+=("$C5_ROOT")

C5_MOUNT="$C5_ROOT/worktrees"
mkdir -p "$C5_MOUNT"

# gc-script stub that exits 1 (simulates runtime error in gc.sh)
C5_GC_STUB="$C5_ROOT/gc_stub_fail.sh"
cat > "$C5_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "gc-script: simulated runtime error" >&2
exit 1
STUB_EOF
chmod +x "$C5_GC_STUB"

run_sweep --mount "$C5_MOUNT" --gc-script "$C5_GC_STUB"

assert "C5: sweep propagates rc=1 from failing gc-script (exec-propagation contract)" \
    test "$RC" -eq 1

# ──────────────────────────────────────────────────────────────────────────────
# Block V — part-1 verification / regression guards
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block V: part-1 verification / regression guards ---"

# V1: --help output mentions --mount (tests help text, not live acceptance)
# Note: to upgrade to live acceptance, invoke: disk-guard check --mount <dir>
#       and assert exit 0 or exit 1 (low-disk), but NOT exit 2 (unknown flag).
assert "V1: warm-lane-disk-guard.sh --help mentions --mount" \
    bash -c 'bash "$1" --help 2>&1 | grep -q -- "--mount"' _ "$DISK_GUARD"

# V2: --help output mentions --min-free-gib
assert "V2: warm-lane-disk-guard.sh --help mentions --min-free-gib" \
    bash -c 'bash "$1" --help 2>&1 | grep -q -- "--min-free-gib"' _ "$DISK_GUARD"

# V3: --help output mentions --min-free-inodes
assert "V3: warm-lane-disk-guard.sh --help mentions --min-free-inodes" \
    bash -c 'bash "$1" --help 2>&1 | grep -q -- "--min-free-inodes"' _ "$DISK_GUARD"

# ──────────────────────────────────────────────────────────────────────────────
# Block W — cross-script --mount seam: single DF value, agreeing semantics
# ──────────────────────────────────────────────────────────────────────────────
# This is a GREEN-ON-ARRIVAL regression guard. The current behavior is already
# correct; this block pins the cross-script contract against future drift.
# It asserts RUNTIME behavior (exit codes, stderr error path, df volume identity),
# NOT doc prose.
#
# DF consumer contract (from orchestrator/src/orchestrator/git_ops.py):
#   - _run_warm_lane_gc_reclaim (line 1401): --mount str(self.worktree_base)
#   - _run_warm_lane_disk_guard (line 1364): --mount str(self.worktree_base)
#   where worktree_base = (project_root/.worktrees).resolve()
#                       = /home/leo/src/warm-lanes/worktrees
# DF passes the WORKTREES DIR (not the XFS mount point) to BOTH scripts.
# The two scripts consume it differently:
#   warm-lane-disk-guard.sh: runs df on it (any path on the XFS volume suffices)
#   warm-lane-gc.sh: treats it as the worktrees-dir; derives BASE_TARGET from its PARENT
echo ""
echo "--- Block W: cross-script --mount seam: single DF value, agreeing semantics ---"

# Build hermetic tmp root. WB = the worktrees-dir (same value DF passes as --mount).
# Do NOT create $W_ROOT/base yet — gc's readlink -f of the derived BASE_TARGET
# ($W_ROOT/base/target) will fail deterministically (all components must exist for -f).
W_ROOT="$(mktemp -d /tmp/test-gc-sweep-w-XXXXXX)"
_TMPDIRS+=("$W_ROOT")
GC_REAL="$WARM_LANE_SCRIPTS_DIR/warm-lane-gc.sh"
WB="$W_ROOT/worktrees"
mkdir -p "$WB"

# Capture gc.sh exit code and stderr
W_ERR_FILE="$(mktemp /tmp/test-gc-sweep-w-err-XXXXXX)"
_TMPDIRS+=("$W_ERR_FILE")
W_RC=0
bash "$GC_REAL" reclaim --mount "$WB" 2>"$W_ERR_FILE" || W_RC=$?
W_ERR="$(cat "$W_ERR_FILE")"

# W1: exit 1 (runtime error: base-target unresolvable), NOT exit 2 (unknown flag).
# Proves --mount is accepted and the derivation was ATTEMPTED (readlink -f tried and failed).
assert "W1: gc.sh reclaim --mount exits 1 (runtime, not usage error) when base-target unresolvable" \
    test "$W_RC" -eq 1

# W2: gc stderr names the SIBLING-derived base-target.
# Locks: BASE_TARGET = $(dirname MOUNT)/base/target = $W_ROOT/base/target.
# The error message "Cannot resolve base-target symlink: <path>" must name THIS path.
EXPECTED_BASE_TARGET="$W_ROOT/base/target"
assert "W2: gc stderr names the sibling-derived base-target ($EXPECTED_BASE_TARGET)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "$2"' _ "$W_ERR" "$EXPECTED_BASE_TARGET"

# W3: gc stderr does NOT contain $WB/base/target.
# $WB/base/target = $W_ROOT/worktrees/base/target is what you'd get if gc placed
# BASE_TARGET as a SUBDIR of the mount (the rejected option-a XFS-mount-point
# interpretation). Our derivation places it in the PARENT of the worktrees-dir.
REJECTED_BASE_TARGET="$WB/base/target"
assert "W3: gc stderr does NOT name the rejected worktrees-subdir base-target ($REJECTED_BASE_TARGET)" \
    bash -c '! printf "%s\n" "$1" | grep -qF "$2"' _ "$W_ERR" "$REJECTED_BASE_TARGET"

# W4: disk-guard accepts the SAME --mount value DF passes.
# 0/0 thresholds ensure exit 0 regardless of actual disk state (non-flaky).
# Also verifies that the value DF passes works for both scripts simultaneously.
mkdir -p "$W_ROOT/base/target"  # create base sibling so df can stat it in W5
W4_RC=0
bash "$DISK_GUARD" check --mount "$WB" --min-free-gib 0 --min-free-inodes 0 2>/dev/null || W4_RC=$?
assert "W4: disk-guard check accepts --mount <worktrees-dir> with 0/0 thresholds" \
    bash -c '[ "$1" -eq 0 ] || [ "$1" -eq 1 ]' _ "$W4_RC"

# W5: the single DF value ($WB = worktrees-dir) and the derived base-target
# ($W_ROOT/base/target) land on the SAME filesystem.
# This is the volume-agreement seam-lock: disk-guard's df measurement and gc's
# derived base-target path both refer to the same XFS volume.
# Uses GNU df --output=target (same infra constraint disk-guard already relies on).
WB_FS="$(df --output=target -- "$WB" 2>/dev/null | tail -n1)"
BT_FS="$(df --output=target -- "$W_ROOT/base/target" 2>/dev/null | tail -n1)"
assert "W5: --mount worktrees-dir and derived base-target are on the same filesystem (df volume agreement)" \
    bash -c '[ "$1" = "$2" ]' _ "$WB_FS" "$BT_FS"

# ──────────────────────────────────────────────────────────────────────────────
# Block G — emergency low-water trigger (task 5168)
# ──────────────────────────────────────────────────────────────────────────────
# Below a critical free-GiB floor, the sweep appends --disk-pressure to its
# reclaim invocation, engaging the existing rm-outright reset (task 5167)
# instead of the α reflink-reseed clone that deadlocks at true ENOSPC. All
# branch selection is driven by an INJECTED df stub value vs an explicit
# --critical-free-gib — no real-disk numeric premise.
echo ""
echo "--- Block G: emergency low-water trigger ---"

# ── G1: below-floor → --disk-pressure appended (avail 1 GiB < floor 2 GiB) ────
# RED today: the sweep does not yet parse --critical-free-gib or measure df,
# so it never appends --disk-pressure.
G1_ROOT="$(mktemp -d /tmp/test-gc-sweep-g1-XXXXXX)"
_TMPDIRS+=("$G1_ROOT")

G1_MOUNT="$G1_ROOT/worktrees"
mkdir -p "$G1_MOUNT"

G1_GC_LOG="$G1_ROOT/gc_calls.log"
G1_GC_STUB="$G1_ROOT/gc_stub.sh"
cat > "$G1_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
chmod +x "$G1_GC_STUB"

G1_DF_STUB="$G1_ROOT/df_stub.sh"
_df_stub "$G1_DF_STUB" 1073741824  # 1 GiB avail

GC_LOG="$G1_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_DF="$G1_DF_STUB" \
    run_sweep --mount "$G1_MOUNT" --gc-script "$G1_GC_STUB" --critical-free-gib 2

assert "G1: exit 0" test "$RC" -eq 0
assert "G1: gc-script invoked with reclaim --mount <dir> ... --disk-pressure (below floor)" \
    bash -c 'grep -qE "^reclaim --mount .*--disk-pressure" "$1"' _ "$G1_GC_LOG"

# ── G2: above-floor → plain reclaim, no flag (avail 100 GiB >= floor 2 GiB) ───
G2_ROOT="$(mktemp -d /tmp/test-gc-sweep-g2-XXXXXX)"
_TMPDIRS+=("$G2_ROOT")

G2_MOUNT="$G2_ROOT/worktrees"
mkdir -p "$G2_MOUNT"

G2_GC_LOG="$G2_ROOT/gc_calls.log"
G2_GC_STUB="$G2_ROOT/gc_stub.sh"
cat > "$G2_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
chmod +x "$G2_GC_STUB"

G2_DF_STUB="$G2_ROOT/df_stub.sh"
_df_stub "$G2_DF_STUB" 107374182400  # 100 GiB avail

GC_LOG="$G2_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_DF="$G2_DF_STUB" \
    run_sweep --mount "$G2_MOUNT" --gc-script "$G2_GC_STUB" --critical-free-gib 2

assert "G2: exit 0" test "$RC" -eq 0
assert "G2: gc-script invoked with plain reclaim --mount <dir> (above floor)" \
    bash -c 'grep -qE "^reclaim --mount " "$1"' _ "$G2_GC_LOG"
assert "G2: --disk-pressure NOT appended (above floor)" \
    bash -c '! grep -q -- "--disk-pressure" "$1"' _ "$G2_GC_LOG"

# ── G3: df failure → plain reclaim + exit 0 (fail-to-α) ───────────────────────
G3_ROOT="$(mktemp -d /tmp/test-gc-sweep-g3-XXXXXX)"
_TMPDIRS+=("$G3_ROOT")

G3_MOUNT="$G3_ROOT/worktrees"
mkdir -p "$G3_MOUNT"

G3_GC_LOG="$G3_ROOT/gc_calls.log"
G3_GC_STUB="$G3_ROOT/gc_stub.sh"
cat > "$G3_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
chmod +x "$G3_GC_STUB"

G3_DF_STUB="$G3_ROOT/df_fail_stub.sh"
_df_fail_stub "$G3_DF_STUB"

GC_LOG="$G3_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_DF="$G3_DF_STUB" \
    run_sweep --mount "$G3_MOUNT" --gc-script "$G3_GC_STUB" --critical-free-gib 2

assert "G3: exit 0 (df failure must not abort the sweep under set -euo pipefail)" \
    test "$RC" -eq 0
assert "G3: gc-script invoked with plain reclaim (df failure falls back to α)" \
    bash -c 'grep -qE "^reclaim --mount " "$1"' _ "$G3_GC_LOG"
assert "G3: --disk-pressure NOT appended (df failure → fail-to-α)" \
    bash -c '! grep -q -- "--disk-pressure" "$1"' _ "$G3_GC_LOG"
assert "G3: stderr warns about the df measurement fallback" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "df|measurement|disk-pressure"' _ "$ERR_OUT"

# ── G4: end-to-end below-floor → target/ deleted outright via real gc.sh ──────
# RED today: the sweep runs plain reclaim → α → the seed stub thins the
# marker but keeps the target/ dir → dir present, seed log non-empty.
G4_ROOT="$(mktemp -d /tmp/test-gc-sweep-g4-XXXXXX)"
_TMPDIRS+=("$G4_ROOT")

G4_REPO="$G4_ROOT/repo"
G4_WORKTREES="$G4_ROOT/worktrees"
G4_BASE="$G4_ROOT/base"
mkdir -p "$G4_WORKTREES" "$G4_BASE"
make_repo "$G4_REPO"
mkdir -p "$G4_BASE/target.gen.1"
touch "$G4_BASE/target.gen.1.lock"
ln -sfn "$G4_BASE/target.gen.1" "$G4_BASE/target"

# Reclaimable FREE lane: clean, landed (HEAD==main), holding a divergent marker.
git -C "$G4_REPO" worktree add -q "$G4_WORKTREES/_lane-1"
mkdir -p "$G4_WORKTREES/_lane-1/target"
touch "$G4_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

G4_DF_STUB="$G4_ROOT/df_stub.sh"
_df_stub "$G4_DF_STUB" 1073741824  # 1 GiB avail (below the 2 GiB floor)

G4_SEED_LOG="$G4_ROOT/seed_calls.log"
G4_SEED_STUB="$G4_ROOT/seed_stub.sh"
_thinning_seed_stub_body > "$G4_SEED_STUB"
chmod +x "$G4_SEED_STUB"

REIFY_WARM_LANE_GC_SWEEP_DF="$G4_DF_STUB" \
REIFY_WARM_LANE_GC_SEED_SCRIPT="$G4_SEED_STUB" \
SEED_LOG="$G4_SEED_LOG" \
    run_sweep --mount "$G4_WORKTREES" --gc-script "$GC_REAL" --critical-free-gib 2

assert "G4: exit 0" test "$RC" -eq 0
assert "G4: _lane-1/target deleted outright end-to-end (emergency trigger engaged --disk-pressure)" \
    bash -c '[ ! -d "$1" ]' _ "$G4_WORKTREES/_lane-1/target"
assert "G4: seed stub NOT invoked (--disk-pressure skips the α reseed)" \
    bash -c '[ ! -s "$1" ]' _ "$G4_SEED_LOG"

# ── G5: always-reclaim under --disk-pressure — dirty POOL LANE reclaimed, ─────
# live-consumer lane preserved (task 5326). The dirty pool lane's target/ is
# deleted outright (the flock is free), while the live-consumer lane's flock
# preserves it — proving the flock is the sole remaining Pass-1 preserve gate.
G5_ROOT="$(mktemp -d /tmp/test-gc-sweep-g5-XXXXXX)"
_TMPDIRS+=("$G5_ROOT")

G5_REPO="$G5_ROOT/repo"
G5_WORKTREES="$G5_ROOT/worktrees"
G5_BASE="$G5_ROOT/base"
mkdir -p "$G5_WORKTREES" "$G5_BASE"
make_repo "$G5_REPO"
mkdir -p "$G5_BASE/target.gen.1"
touch "$G5_BASE/target.gen.1.lock"
ln -sfn "$G5_BASE/target.gen.1" "$G5_BASE/target"

# _lane-dirty: uncommitted tracked change POOL LANE. Under always-reclaim
# (task 5326) its flock is free, so it is RECLAIMED — target/ deleted outright
# under --disk-pressure. Dirty tracked changes no longer preserve a pool lane.
git -C "$G5_REPO" worktree add -q "$G5_WORKTREES/_lane-dirty"
echo "dirty" >> "$G5_WORKTREES/_lane-dirty/README.md"
mkdir -p "$G5_WORKTREES/_lane-dirty/target"
touch "$G5_WORKTREES/_lane-dirty/target/DIVERGENT_MARKER"

# _lane-locked: clean, landed, but a live consumer holds the exclusive lock.
git -C "$G5_REPO" worktree add -q "$G5_WORKTREES/_lane-locked"
mkdir -p "$G5_WORKTREES/_lane-locked/target"
touch "$G5_WORKTREES/_lane-locked/target/DIVERGENT_MARKER"

touch "$G5_WORKTREES/_lane-locked.lock"
G5_READY="$G5_WORKTREES/_lane-locked.lock.ready-marker"
( flock -x 9 && touch "$G5_READY" && sleep 300 ) 9>"$G5_WORKTREES/_lane-locked.lock" &
G5_LOCK_PID=$!
_BGPIDS+=("$G5_LOCK_PID")
_wait_for_reader_lock "$G5_READY" 30

G5_DF_STUB="$G5_ROOT/df_stub.sh"
_df_stub "$G5_DF_STUB" 1073741824  # 1 GiB avail (below the 2 GiB floor)

G5_SEED_LOG="$G5_ROOT/seed_calls.log"
G5_SEED_STUB="$G5_ROOT/seed_stub.sh"
_thinning_seed_stub_body > "$G5_SEED_STUB"
chmod +x "$G5_SEED_STUB"

REIFY_WARM_LANE_GC_SWEEP_DF="$G5_DF_STUB" \
REIFY_WARM_LANE_GC_SEED_SCRIPT="$G5_SEED_STUB" \
SEED_LOG="$G5_SEED_LOG" \
    run_sweep --mount "$G5_WORKTREES" --gc-script "$GC_REAL" --critical-free-gib 2

assert "G5: exit 0" test "$RC" -eq 0
assert "G5: dirty pool lane target/ deleted outright (always-reclaim under --disk-pressure)" \
    bash -c '[ ! -d "$1" ]' _ "$G5_WORKTREES/_lane-dirty/target"
assert "G5: live-consumer lane target/ marker intact (flock is the sole preserve gate)" \
    test -f "$G5_WORKTREES/_lane-locked/target/DIVERGENT_MARKER"

# Release the background lock holder.
kill "$G5_LOCK_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ── G6: non-integer --critical-free-gib → exit 2 + stderr warning ────────────
# Regression guard for the CRITICAL_FREE_GIB integer-format validation
# (scripts/warm-lane-gc-sweep.sh: `grep -qE '^[0-9]+$'` guard, ~L139-143). A
# dropped or weakened guard (e.g. accepting "2G" or an empty value) would let
# the sweep silently compare avail_bytes against a bogus/zero floor and never
# engage --disk-pressure, with nothing to catch it. Mirrors Block A's
# unknown-flag CLI-guard assertions (A3).
G6_ROOT="$(mktemp -d /tmp/test-gc-sweep-g6-XXXXXX)"
_TMPDIRS+=("$G6_ROOT")

G6_MOUNT="$G6_ROOT/worktrees"
mkdir -p "$G6_MOUNT"

G6_GC_LOG="$G6_ROOT/gc_calls.log"
G6_GC_STUB="$G6_ROOT/gc_stub.sh"
cat > "$G6_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
chmod +x "$G6_GC_STUB"

# G6a: non-numeric value via --critical-free-gib
GC_LOG="$G6_GC_LOG" run_sweep --mount "$G6_MOUNT" --gc-script "$G6_GC_STUB" --critical-free-gib abc

assert "G6a: --critical-free-gib abc exits 2" test "$RC" -eq 2
assert "G6a: stderr warns of invalid integer" \
    bash -c 'printf "%s\n" "$1" | grep -qi "valid integer"' _ "$ERR_OUT"
assert "G6a: gc-script NOT invoked (usage error precedes exec)" \
    bash -c '[ ! -f "$1" ] || [ ! -s "$1" ]' _ "$G6_GC_LOG"

# G6b: numeric-with-suffix value ("2G") is rejected too — pins the ^[0-9]+$
# anchoring; an unanchored or partial-match regression could let this slip
# through and silently truncate to a bogus floor.
GC_LOG="$G6_GC_LOG" run_sweep --mount "$G6_MOUNT" --gc-script "$G6_GC_STUB" --critical-free-gib 2G

assert "G6b: --critical-free-gib 2G (numeric+suffix) exits 2" test "$RC" -eq 2
assert "G6b: stderr warns of invalid integer" \
    bash -c 'printf "%s\n" "$1" | grep -qi "valid integer"' _ "$ERR_OUT"
assert "G6b: gc-script NOT invoked (usage error precedes exec)" \
    bash -c '[ ! -f "$1" ] || [ ! -s "$1" ]' _ "$G6_GC_LOG"

# G6c: same non-numeric value via the env knob
GC_LOG="$G6_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_CRITICAL_FREE_GIB="abc" \
    run_sweep --mount "$G6_MOUNT" --gc-script "$G6_GC_STUB"

assert "G6c: REIFY_WARM_LANE_GC_SWEEP_CRITICAL_FREE_GIB=abc exits 2" test "$RC" -eq 2
assert "G6c: stderr warns of invalid integer (env-var path)" \
    bash -c 'printf "%s\n" "$1" | grep -qi "valid integer"' _ "$ERR_OUT"
assert "G6c: gc-script NOT invoked (usage error precedes exec)" \
    bash -c '[ ! -f "$1" ] || [ ! -s "$1" ]' _ "$G6_GC_LOG"

# ── G7: df succeeds (exit 0) but reports unparseable avail bytes ─────────────
# Distinct code path from G3 (which covers a non-zero df EXIT CODE): here df
# exits 0 but its second line fails the `^[0-9]+$` numeric guard
# (scripts/warm-lane-gc-sweep.sh: the "else" branch after the numeric check,
# ~L173-175) — e.g. a locale/column mismatch. A mis-anchored regex accepting
# non-numeric input here would feed a bogus value into the arithmetic
# avail_bytes -le critical_bytes comparison.
G7_ROOT="$(mktemp -d /tmp/test-gc-sweep-g7-XXXXXX)"
_TMPDIRS+=("$G7_ROOT")

G7_MOUNT="$G7_ROOT/worktrees"
mkdir -p "$G7_MOUNT"

G7_GC_LOG="$G7_ROOT/gc_calls.log"
G7_GC_STUB="$G7_ROOT/gc_stub.sh"
cat > "$G7_GC_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
chmod +x "$G7_GC_STUB"

G7_DF_STUB="$G7_ROOT/df_garbage_stub.sh"
_df_garbage_stub "$G7_DF_STUB"

GC_LOG="$G7_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_DF="$G7_DF_STUB" \
    run_sweep --mount "$G7_MOUNT" --gc-script "$G7_GC_STUB" --critical-free-gib 2

assert "G7: exit 0 (unparseable avail must not abort the sweep)" test "$RC" -eq 0
assert "G7: gc-script invoked with plain reclaim (unparseable avail falls back to α)" \
    bash -c 'grep -qE "^reclaim --mount " "$1"' _ "$G7_GC_LOG"
assert "G7: --disk-pressure NOT appended (unparseable avail → fail-to-α)" \
    bash -c '! grep -q -- "--disk-pressure" "$1"' _ "$G7_GC_LOG"
assert "G7: stderr warns about unparseable avail bytes" \
    bash -c 'printf "%s\n" "$1" | grep -qi "unparseable"' _ "$ERR_OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block T — stale .reseed-trash reaper (task 5326)
# ──────────────────────────────────────────────────────────────────────────────
# seed-warm-lane.sh --fresh-checkout renames a non-empty target/ to
# $MOUNT/.reseed-trash/<lane>.<pid> then background-rm's it; a SIGKILL between
# the rename and the rm strands the trash until THAT lane is re-acquired (a lane
# never re-acquired strands it forever). The sweep reaps a stranded entry ONLY
# when its encoded PID is dead AND no live process still references the dir via
# cwd/fd/maps (a PID-reuse / in-flight-rm-or-seed guard). The reaper runs every
# sweep, after the mount-exists fail-open check and before the reclaim exec,
# independent of --disk-pressure. Uses a gc-script STUB (Block C pattern) so the
# post-reaper exec is harmless (no real base-target). RED against current
# gc-sweep.sh (no reaper exists).
echo ""
echo "--- Block T: stale .reseed-trash reaper (task 5326) ---"

# Block-T gc-script stub: logs argv to $GC_LOG, exits 0 (Block C pattern). A
# populated $GC_LOG after a sweep proves the reaper ran BEFORE the exec and did
# not abort the sweep.
_t_gc_stub() {
    cat > "$1" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$GC_LOG"
exit 0
STUB_EOF
    chmod +x "$1"
}

# ── T1: dead-PID entry IS reaped; gc-script still runs (reaper precedes exec) ──
T1_ROOT="$(mktemp -d /tmp/test-gc-sweep-t1-XXXXXX)"
_TMPDIRS+=("$T1_ROOT")
T1_MOUNT="$T1_ROOT/worktrees"
mkdir -p "$T1_MOUNT/.reseed-trash"

T1_GC_LOG="$T1_ROOT/gc_calls.log"
T1_GC_STUB="$T1_ROOT/gc_stub.sh"
_t_gc_stub "$T1_GC_STUB"

# DEAD_PID: a backgrounded proc captured then waited → reaped, no longer live.
true &
T1_DEAD_PID=$!
wait "$T1_DEAD_PID" 2>/dev/null || true

T1_ENTRY="$T1_MOUNT/.reseed-trash/_lane-1.$T1_DEAD_PID"
mkdir -p "$T1_ENTRY"
touch "$T1_ENTRY/stranded-file"

GC_LOG="$T1_GC_LOG" run_sweep --mount "$T1_MOUNT" --gc-script "$T1_GC_STUB"

assert "T1: exit 0" test "$RC" -eq 0
assert "T1: dead-PID trash entry reaped (dir gone)" \
    bash -c '[ ! -d "$1" ]' _ "$T1_ENTRY"
assert "T1: gc-script still invoked (reaper precedes the reclaim exec)" \
    bash -c '[ -s "$1" ]' _ "$T1_GC_LOG"

# ── T2: live-PID entry is PRESERVED (encoded PID still alive) ──────────────────
T2_ROOT="$(mktemp -d /tmp/test-gc-sweep-t2-XXXXXX)"
_TMPDIRS+=("$T2_ROOT")
T2_MOUNT="$T2_ROOT/worktrees"
mkdir -p "$T2_MOUNT/.reseed-trash"

T2_GC_LOG="$T2_ROOT/gc_calls.log"
T2_GC_STUB="$T2_ROOT/gc_stub.sh"
_t_gc_stub "$T2_GC_STUB"

# LIVE_PID: a live backgrounded sleep; its PID is the entry's suffix.
sleep 300 &
T2_LIVE_PID=$!
_BGPIDS+=("$T2_LIVE_PID")

T2_ENTRY="$T2_MOUNT/.reseed-trash/_lane-2.$T2_LIVE_PID"
mkdir -p "$T2_ENTRY"
touch "$T2_ENTRY/stranded-file"

GC_LOG="$T2_GC_LOG" run_sweep --mount "$T2_MOUNT" --gc-script "$T2_GC_STUB"

assert "T2: exit 0" test "$RC" -eq 0
assert "T2: live-PID trash entry preserved (encoded PID still alive)" \
    bash -c '[ -d "$1" ]' _ "$T2_ENTRY"

kill "$T2_LIVE_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ── T3: cwd guard — dead encoded PID but a LIVE proc's cwd is the entry dir ───
# The entry's suffix PID is dead (kill -0 fails), so preservation can only come
# from the cwd/fd/maps guard catching the live helper whose cwd is the entry.
# After the helper dies, a second sweep reaps it — isolating the guard.
T3_ROOT="$(mktemp -d /tmp/test-gc-sweep-t3-XXXXXX)"
_TMPDIRS+=("$T3_ROOT")
T3_MOUNT="$T3_ROOT/worktrees"
mkdir -p "$T3_MOUNT/.reseed-trash"

T3_GC_LOG="$T3_ROOT/gc_calls.log"
T3_GC_STUB="$T3_ROOT/gc_stub.sh"
_t_gc_stub "$T3_GC_STUB"

# DEAD encoded PID (so the kill -0 guard falls through to the cwd/fd/maps scan).
true &
T3_DEAD_PID=$!
wait "$T3_DEAD_PID" 2>/dev/null || true

T3_ENTRY="$T3_MOUNT/.reseed-trash/_lane-3.$T3_DEAD_PID"
mkdir -p "$T3_ENTRY"

# Live helper whose CWD is the entry dir. It touches the READY marker AFTER
# cd'ing in, so _wait_for_reader_lock proves the cwd is established before the
# sweep runs (causal ordering — no wall-clock race). exec replaces the subshell
# with sleep so the tracked PID itself holds the cwd.
T3_READY="$T3_ROOT/helper.ready"
( cd "$T3_ENTRY" && touch "$T3_READY" && exec sleep 300 ) &
T3_HELPER_PID=$!
_BGPIDS+=("$T3_HELPER_PID")
_wait_for_reader_lock "$T3_READY" 30

GC_LOG="$T3_GC_LOG" run_sweep --mount "$T3_MOUNT" --gc-script "$T3_GC_STUB"

assert "T3: exit 0 (first sweep)" test "$RC" -eq 0
assert "T3: entry preserved despite dead encoded PID (live cwd reference guards it)" \
    bash -c '[ -d "$1" ]' _ "$T3_ENTRY"

# Kill the helper and wait for it to be reaped, dropping the cwd reference.
kill "$T3_HELPER_PID" 2>/dev/null || true
wait "$T3_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

GC_LOG="$T3_GC_LOG" run_sweep --mount "$T3_MOUNT" --gc-script "$T3_GC_STUB"

assert "T3: exit 0 (second sweep)" test "$RC" -eq 0
assert "T3: entry reaped once the live cwd reference is gone (second sweep)" \
    bash -c '[ ! -d "$1" ]' _ "$T3_ENTRY"

# ── T4: malformed entries (no numeric .pid suffix) are SKIPPED (never reaped) ──
T4_ROOT="$(mktemp -d /tmp/test-gc-sweep-t4-XXXXXX)"
_TMPDIRS+=("$T4_ROOT")
T4_MOUNT="$T4_ROOT/worktrees"
mkdir -p "$T4_MOUNT/.reseed-trash"

T4_GC_LOG="$T4_ROOT/gc_calls.log"
T4_GC_STUB="$T4_ROOT/gc_stub.sh"
_t_gc_stub "$T4_GC_STUB"

# Non-numeric suffix and a name with no dot at all → ambiguous PID → skip.
T4_ENTRY_ALPHA="$T4_MOUNT/.reseed-trash/_lane-4.abc"
T4_ENTRY_NODOT="$T4_MOUNT/.reseed-trash/no-suffix"
mkdir -p "$T4_ENTRY_ALPHA" "$T4_ENTRY_NODOT"

GC_LOG="$T4_GC_LOG" run_sweep --mount "$T4_MOUNT" --gc-script "$T4_GC_STUB"

assert "T4: exit 0" test "$RC" -eq 0
assert "T4: non-numeric-suffix entry preserved (ambiguous PID → never reaped)" \
    bash -c '[ -d "$1" ]' _ "$T4_ENTRY_ALPHA"
assert "T4: no-dot entry preserved (ambiguous PID → never reaped)" \
    bash -c '[ -d "$1" ]' _ "$T4_ENTRY_NODOT"

# ── T5: no .reseed-trash/ dir → reaper is a no-op; sweep still runs gc ─────────
T5_ROOT="$(mktemp -d /tmp/test-gc-sweep-t5-XXXXXX)"
_TMPDIRS+=("$T5_ROOT")
T5_MOUNT="$T5_ROOT/worktrees"
mkdir -p "$T5_MOUNT"  # deliberately NO .reseed-trash/ subdir

T5_GC_LOG="$T5_ROOT/gc_calls.log"
T5_GC_STUB="$T5_ROOT/gc_stub.sh"
_t_gc_stub "$T5_GC_STUB"

GC_LOG="$T5_GC_LOG" run_sweep --mount "$T5_MOUNT" --gc-script "$T5_GC_STUB"

assert "T5: exit 0 (missing .reseed-trash dir → reaper no-op)" test "$RC" -eq 0
assert "T5: gc-script still invoked (no-trash reaper does not abort the sweep)" \
    bash -c '[ -s "$1" ]' _ "$T5_GC_LOG"

# ── T6: fail-open — the reaper's rm fails, yet the sweep warns and exits 0 ─────
# PATH-scoped failing rm stub (Block K4 technique, root-safe). The reaper must
# warn and continue (fail-open), never abort under set -euo pipefail, and must
# leave the entry intact when its own rm fails.
T6_ROOT="$(mktemp -d /tmp/test-gc-sweep-t6-XXXXXX)"
_TMPDIRS+=("$T6_ROOT")
T6_MOUNT="$T6_ROOT/worktrees"
mkdir -p "$T6_MOUNT/.reseed-trash"

T6_GC_LOG="$T6_ROOT/gc_calls.log"
T6_GC_STUB="$T6_ROOT/gc_stub.sh"
_t_gc_stub "$T6_GC_STUB"

# Failing rm stub — its message deliberately avoids the words the reaper's own
# warn uses (trash/reap/reseed) so the T6 stderr assertion pins the REAPER's
# warning, not the stub's echo. Prepended to PATH only for the run_sweep call
# below (bash reverts a prefix assignment on a function call immediately after
# — same idiom as gc Block K4).
T6_STUB_DIR="$T6_ROOT/binstub"
mkdir -p "$T6_STUB_DIR"
cat > "$T6_STUB_DIR/rm" << 'STUB_EOF'
#!/usr/bin/env bash
echo "rm: simulated failure (test stub)" >&2
exit 1
STUB_EOF
chmod +x "$T6_STUB_DIR/rm"

# DEAD-PID entry so the reaper reaches the rm action.
true &
T6_DEAD_PID=$!
wait "$T6_DEAD_PID" 2>/dev/null || true

T6_ENTRY="$T6_MOUNT/.reseed-trash/_lane-6.$T6_DEAD_PID"
mkdir -p "$T6_ENTRY"

PATH="$T6_STUB_DIR:$PATH" GC_LOG="$T6_GC_LOG" \
    run_sweep --mount "$T6_MOUNT" --gc-script "$T6_GC_STUB"

assert "T6: exit 0 (reaper rm failure is fail-open)" test "$RC" -eq 0
assert "T6: entry left intact when the reaper's rm fails" \
    bash -c '[ -d "$1" ]' _ "$T6_ENTRY"
assert "T6: stderr carries the reaper's own trash-reap warning" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "reap|trash|reseed"' _ "$ERR_OUT"

# ── T7: batch scan — several dead-PID candidates in ONE sweep; exactly the ────
# live-cwd-referenced entry is preserved, the rest reaped ─────────────────────
# Exercises the batched single-/proc-scan path: the reaper collects all dead-PID
# candidates, does ONE /proc walk for the whole set, and must discriminate the
# referenced entry from the unreferenced ones (not all-or-nothing). Once the
# live reference drops, a second sweep reaps the last entry.
T7_ROOT="$(mktemp -d /tmp/test-gc-sweep-t7-XXXXXX)"
_TMPDIRS+=("$T7_ROOT")
T7_MOUNT="$T7_ROOT/worktrees"
mkdir -p "$T7_MOUNT/.reseed-trash"

T7_GC_LOG="$T7_ROOT/gc_calls.log"
T7_GC_STUB="$T7_ROOT/gc_stub.sh"
_t_gc_stub "$T7_GC_STUB"

# Live helper started FIRST — so the kernel cannot reuse its PID for any of the
# dead PIDs forked afterwards. Its cwd is a staging dir renamed into the
# referenced trash entry once its dead-PID suffix is known, so the helper holds
# that entry dir's inode as cwd while the entry's ENCODED PID is guaranteed dead
# (isolating the cwd/fd/maps guard, exactly as in T3). READY lives outside the
# staging dir so the rename does not move the handshake marker.
T7_STAGE="$T7_ROOT/stage"
mkdir -p "$T7_STAGE"
T7_READY="$T7_ROOT/helper.ready"
( cd "$T7_STAGE" && touch "$T7_READY" && exec sleep 300 ) &
T7_HELPER_PID=$!
_BGPIDS+=("$T7_HELPER_PID")
_wait_for_reader_lock "$T7_READY" 30

# Three DEAD encoded PIDs (forked while the helper is alive → all distinct from
# it), one per candidate entry.
true & T7_DEAD_A=$!; wait "$T7_DEAD_A" 2>/dev/null || true
true & T7_DEAD_B=$!; wait "$T7_DEAD_B" 2>/dev/null || true
true & T7_DEAD_C=$!; wait "$T7_DEAD_C" 2>/dev/null || true

T7_ENTRY_A="$T7_MOUNT/.reseed-trash/_lane-a.$T7_DEAD_A"   # unreferenced → reaped
T7_ENTRY_B="$T7_MOUNT/.reseed-trash/_lane-b.$T7_DEAD_B"   # live cwd ref → preserved
T7_ENTRY_C="$T7_MOUNT/.reseed-trash/_lane-c.$T7_DEAD_C"   # unreferenced → reaped
mkdir -p "$T7_ENTRY_A" "$T7_ENTRY_C"
touch "$T7_ENTRY_A/stranded" "$T7_ENTRY_C/stranded"
# Rename the helper's live cwd into ENTRY_B: /proc/<helper>/cwd now resolves to
# ENTRY_B, so the batch scan must catch it even though _lane-b's encoded PID is
# dead — while A and C (no live reference) must be reaped in the same sweep.
mv "$T7_STAGE" "$T7_ENTRY_B"

GC_LOG="$T7_GC_LOG" run_sweep --mount "$T7_MOUNT" --gc-script "$T7_GC_STUB"

assert "T7: exit 0 (first sweep)" test "$RC" -eq 0
assert "T7: unreferenced dead-PID entry A reaped in the batch sweep" \
    bash -c '[ ! -d "$1" ]' _ "$T7_ENTRY_A"
assert "T7: unreferenced dead-PID entry C reaped in the batch sweep" \
    bash -c '[ ! -d "$1" ]' _ "$T7_ENTRY_C"
assert "T7: live-cwd-referenced entry B preserved despite dead encoded PID" \
    bash -c '[ -d "$1" ]' _ "$T7_ENTRY_B"

# Drop the live cwd reference; a second sweep now reaps the last entry.
kill "$T7_HELPER_PID" 2>/dev/null || true
wait "$T7_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

GC_LOG="$T7_GC_LOG" run_sweep --mount "$T7_MOUNT" --gc-script "$T7_GC_STUB"

assert "T7: exit 0 (second sweep)" test "$RC" -eq 0
assert "T7: entry B reaped once its live cwd reference is gone (second sweep)" \
    bash -c '[ ! -d "$1" ]' _ "$T7_ENTRY_B"

# ──────────────────────────────────────────────────────────────────────────────
# Block U — live-consumer lane guard (task 5378)
# ──────────────────────────────────────────────────────────────────────────────
# ROOT CAUSE (esc-5375-1): the "one consumer per lane" flock (inv.2) is held
# only across the ACQUIRE reseed, NOT across the consumer's long cargo verify
# build. During that multi-minute build the flock is FREE, so gc.sh's Pass-1
# always-reclaim reset renames/rm's <lane>/target out from under the live build.
# FIX: warm-lane-gc-sweep.sh scans each pool-lane DIR realpath for a live process
# reference (cwd/fd/mmap at or under it — the SAME /proc scanner the reseed-trash
# reaper uses), maps live lanes back to basenames, and passes them to the real
# gc.sh via --extra-protect-glob so a lane with an active build is skipped
# ENTIRELY (Pass 1 reset + Pass 2 remove) despite a FREE flock. Over-preserving
# is SAFE and TEMPORARY: once the reference clears, a later sweep reclaims the
# lane (mirrors the trash reaper's second-sweep behavior). RED against the
# current gc-sweep.sh (no lane scan) — the first sweep resets the live lane.
echo ""
echo "--- Block U: live-consumer lane guard (task 5378) ---"

# ── U1: end-to-end via REAL gc.sh — a live cwd reference at/under a lane dir ───
# preserves it despite a FREE flock, while a free unreferenced lane is reset ────
#
# DO NOT DELETE as "redundant with test_warm_lane_gc.sh Block P" (task 5572).
# U1-U7 drive the REAL gc.sh (GC_REAL) end-to-end and stay green by a NEW
# mechanism: the primitive's own per-lane check, not the wrapper CSV that U8-U13
# now assert is gone. That change of mechanism is exactly what makes them
# valuable — they are the COMPOSITION proof that the sweep → gc.sh path still
# preserves a live lane after the wrapper-side guard was removed. Block P proves
# the primitive in isolation; this proves the two-script path end to end.
U_ROOT="$(mktemp -d /tmp/test-gc-sweep-u-XXXXXX)"
_TMPDIRS+=("$U_ROOT")

U_REPO="$U_ROOT/repo"
U_WORKTREES="$U_ROOT/worktrees"
U_BASE="$U_ROOT/base"
mkdir -p "$U_WORKTREES" "$U_BASE"
make_repo "$U_REPO"
mkdir -p "$U_BASE/target.gen.1"
touch "$U_BASE/target.gen.1.lock"
ln -sfn "$U_BASE/target.gen.1" "$U_BASE/target"

# Two clean+landed (HEAD==main) reclaimable pool lanes, each with a divergent
# marker and a FREE flock (no holder started).
for _u_name in _lane-1 _lane-2; do
    git -C "$U_REPO" worktree add -q "$U_WORKTREES/$_u_name"
    mkdir -p "$U_WORKTREES/$_u_name/target"
    touch "$U_WORKTREES/$_u_name/target/DIVERGENT_MARKER"
done

# Live helper whose CWD is _lane-1/target (a descendant of the lane dir — the
# exact build-fd/cwd shape). It touches READY AFTER cd'ing in, so
# _wait_for_reader_lock proves the cwd is established before the sweep runs
# (causal ordering, Block T3 pattern). exec sleep so the tracked PID holds it.
U_READY="$U_ROOT/helper.ready"
( cd "$U_WORKTREES/_lane-1/target" && touch "$U_READY" && exec sleep 300 ) &
U_HELPER_PID=$!
_BGPIDS+=("$U_HELPER_PID")
_wait_for_reader_lock "$U_READY" 30

U_DF_STUB="$U_ROOT/df_stub.sh"
_df_stub "$U_DF_STUB" 107374182400  # 100 GiB avail (above floor → α path, no --disk-pressure)

U_SEED_LOG="$U_ROOT/seed_calls.log"
U_SEED_STUB="$U_ROOT/seed_stub.sh"
_thinning_seed_stub_body > "$U_SEED_STUB"
chmod +x "$U_SEED_STUB"

REIFY_WARM_LANE_GC_SWEEP_DF="$U_DF_STUB" \
REIFY_WARM_LANE_GC_SEED_SCRIPT="$U_SEED_STUB" \
SEED_LOG="$U_SEED_LOG" \
    run_sweep --mount "$U_WORKTREES" --gc-script "$GC_REAL" --critical-free-gib 2

assert "U1: exit 0 (first sweep)" test "$RC" -eq 0
assert "U2: live-referenced _lane-1 divergent marker INTACT (cwd ref preserves it despite a FREE flock)" \
    test -f "$U_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "U3: _lane-1 seed-script NOT invoked (preserved, not reset)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-1" "$1"' _ "$U_SEED_LOG"
assert "U4: free unreferenced _lane-2 divergent marker REMOVED (reset — no blanket over-preserve)" \
    bash -c '[ ! -f "$1" ]' _ "$U_WORKTREES/_lane-2/target/DIVERGENT_MARKER"
assert "U5: _lane-2 seed-script invoked (reclaimed via α)" \
    bash -c 'test -f "$1" && grep -q "_lane-2" "$1"' _ "$U_SEED_LOG"

# ── U6: preserve is TEMPORARY — once the live reference clears, a SECOND sweep
# reclaims the lane (mirrors Block T3's second-sweep reap) ─────────────────────
kill "$U_HELPER_PID" 2>/dev/null || true
wait "$U_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

REIFY_WARM_LANE_GC_SWEEP_DF="$U_DF_STUB" \
REIFY_WARM_LANE_GC_SEED_SCRIPT="$U_SEED_STUB" \
SEED_LOG="$U_SEED_LOG" \
    run_sweep --mount "$U_WORKTREES" --gc-script "$GC_REAL" --critical-free-gib 2

assert "U6: exit 0 (second sweep)" test "$RC" -eq 0
assert "U7: _lane-1 divergent marker REMOVED once its live cwd reference is gone (preserve is temporary)" \
    bash -c '[ ! -f "$1" ]' _ "$U_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

# ── U8: stub-gc — the sweep hands gc.sh NO --extra-protect-glob at all ────────
# INVERTED by task 5572. The live-consumer guard moved INTO warm-lane-gc.sh,
# per lane, immediately before each lane's reset (tests/infra/test_warm_lane_gc.sh
# Blocks P/Q), so the wrapper no longer computes a live-lane CSV: it would be
# strictly redundant with the primitive's own gate AND it was the TOCTOU source
# — one snapshot taken up front, then consumed across a 25-40 minute lane-by-lane
# traversal, leaving any lane that went live mid-pass unprotected.
#
# This case now PINS that absence. Re-adding a wrapper-side snapshot would
# reintroduce that TOCTOU, so "the sweep passes no --extra-protect-glob" is a
# contract, not an incidental. The same Block-C/T gc-stub logs the argv the sweep
# hands gc.sh, and the same live cwd reference on _lane-1 drives it — only the
# expectation flips. (--extra-protect-glob itself is untouched in gc.sh: it
# remains an explicit-pin escape hatch, covered by that suite's Block O.)
U8_ROOT="$(mktemp -d /tmp/test-gc-sweep-u8-XXXXXX)"
_TMPDIRS+=("$U8_ROOT")
U8_MOUNT="$U8_ROOT/worktrees"
mkdir -p "$U8_MOUNT/_lane-1/target"

U8_GC_LOG="$U8_ROOT/gc_calls.log"
U8_GC_STUB="$U8_ROOT/gc_stub.sh"
_t_gc_stub "$U8_GC_STUB"

U8_DF_STUB="$U8_ROOT/df_stub.sh"
_df_stub "$U8_DF_STUB" 107374182400  # above floor → no --disk-pressure noise

U8_READY="$U8_ROOT/helper.ready"
( cd "$U8_MOUNT/_lane-1/target" && touch "$U8_READY" && exec sleep 300 ) &
U8_HELPER_PID=$!
_BGPIDS+=("$U8_HELPER_PID")
_wait_for_reader_lock "$U8_READY" 30

GC_LOG="$U8_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_DF="$U8_DF_STUB" \
    run_sweep --mount "$U8_MOUNT" --gc-script "$U8_GC_STUB" --critical-free-gib 2

assert "U8: exit 0" test "$RC" -eq 0
# U9 first: without it, U10 would pass vacuously against an empty argv log.
assert "U9: gc-script still invoked (so U10 cannot pass vacuously on an empty log)" \
    bash -c '[ -s "$1" ]' _ "$U8_GC_LOG"
assert "U10: NO --extra-protect-glob in the argv despite a live lane (guard moved into the primitive, task 5572)" \
    bash -c '! grep -q -- "--extra-protect-glob" "$1"' _ "$U8_GC_LOG"

kill "$U8_HELPER_PID" 2>/dev/null || true
wait "$U8_HELPER_PID" 2>/dev/null || true
_BGPIDS=()

# ── U11: no-live-lane — a free unreferenced lane yields an EMPTY scan, so NO ───
# --extra-protect-glob is appended (fail-open, no spurious protection) ─────────
U11_ROOT="$(mktemp -d /tmp/test-gc-sweep-u11-XXXXXX)"
_TMPDIRS+=("$U11_ROOT")
U11_MOUNT="$U11_ROOT/worktrees"
mkdir -p "$U11_MOUNT/_lane-1/target"   # a free lane dir, but NO live reference

U11_GC_LOG="$U11_ROOT/gc_calls.log"
U11_GC_STUB="$U11_ROOT/gc_stub.sh"
_t_gc_stub "$U11_GC_STUB"

U11_DF_STUB="$U11_ROOT/df_stub.sh"
_df_stub "$U11_DF_STUB" 107374182400

GC_LOG="$U11_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_DF="$U11_DF_STUB" \
    run_sweep --mount "$U11_MOUNT" --gc-script "$U11_GC_STUB" --critical-free-gib 2

assert "U11: exit 0" test "$RC" -eq 0
assert "U12: no live reference → --extra-protect-glob NOT appended (no spurious protection, fail-open)" \
    bash -c '! grep -q -- "--extra-protect-glob" "$1"' _ "$U11_GC_LOG"
assert "U13: gc-script still invoked (the lane scan does not abort the sweep)" \
    bash -c '[ -s "$1" ]' _ "$U11_GC_LOG"

# ──────────────────────────────────────────────────────────────────────────────
# Block Y — the BRIDGE-COST SEAM's own contract (task 3655)
# ──────────────────────────────────────────────────────────────────────────────
#
# WHY THIS BLOCK EXISTS. run_sweep's header (above) states outright that "NO
# assert in this file inspects the protect glob". So the pin the whole seam
# exists for — REIFY_WARM_LANE_GC_PROTECT_GLOB short-circuiting gc.sh's python3
# render — was, until this block, verified by NOTHING. A silently-broken pin
# would not fail a single assert here; it would only make the suite slower, and
# task 3298 showed that a wall-clock change on this host is unattributable.
# Every assertion below is therefore a SET-MEMBERSHIP or a COUNT, never a
# duration: a wall-clock assert on this host is a flake generator, which is the
# whole reason task 3655 was filed.
#
# THE DISCRIMINATOR. REIFY_WARM_LANE_IACT_PREFIX moves the interactive band in
# the RENDERED glob — `lane_protect_glob` emits `..._mainsweep-*,_sentinelband-*`
# with it set, in place of `..._mainsweep-*,_iact-*`. It does NOT and cannot
# affect $LANE_PROTECT_GLOB_FALLBACK, which is a static literal. So setting the
# sentinel alongside run_sweep's existing pin splits the two worlds cleanly:
#
#   render SHORT-CIRCUITED (correct) -> applied set is the fallback
#                                       -> `_iact-` protected, `_sentinelband-` NOT
#   render RAN (pin broken)          -> applied set is the render
#                                       -> `_sentinelband-` protected, `_iact-` NOT
#
# Y2 and Y3 are that pair, read straight off gc.sh's own `skipping protected:`
# log line. The entries are plain directories, not git worktrees: gc.sh tests
# the protect glob BEFORE `_is_git_worktree` (warm-lane-gc.sh, Pass-1
# enumeration), so a bare mkdir is enough to observe the applied set — and
# nothing is destroyed, so this block is non-destructive as well as hermetic.
#
# COST. This block adds NO render: it drives the real gc.sh through run_sweep,
# whose pin is the very thing under test. Verified by task 3655's execve
# counters — one full suite run execs `.venv/bin/python3` exactly ONCE, and that
# one is Block W's direct `bash "$GC_REAL"` (line ~415), which bypasses
# run_sweep and so pays the render this pin avoids.
echo ""
echo "--- Block Y: bridge-cost seam contract (pin short-circuits the render) ---"

Y_ROOT="$(mktemp -d /tmp/test-gc-sweep-y-XXXXXX)"
_TMPDIRS+=("$Y_ROOT")
Y_MOUNT="$Y_ROOT/worktrees"
Y_BASE="$Y_ROOT/base"
mkdir -p "$Y_MOUNT/_iact-probe" "$Y_MOUNT/_sentinelband-probe" "$Y_BASE/target.gen.1"
touch "$Y_BASE/target.gen.1.lock"
ln -sfn "$Y_BASE/target.gen.1" "$Y_BASE/target"

Y_DF_STUB="$Y_ROOT/df_stub.sh"
_df_stub "$Y_DF_STUB" 107374182400  # 100 GiB avail — no disk-pressure path

# The sentinel rides ALONGSIDE run_sweep's pin, deliberately: this asserts what
# the PRODUCTION-SHAPED invocation does, not a bespoke one.
REIFY_WARM_LANE_IACT_PREFIX=_sentinelband- \
REIFY_WARM_LANE_GC_SWEEP_DF="$Y_DF_STUB" \
    run_sweep --mount "$Y_MOUNT" --gc-script "$GC_REAL" --critical-free-gib 2

assert "Y1: exit 0 (pinned real-gc sweep)" test "$RC" -eq 0
assert "Y2: _iact- IS protected — gc.sh applied the PINNED fallback set" \
    bash -c 'printf "%s\n" "$1" | grep -qF "skipping protected: _iact-probe"' _ "$ERR_OUT"
assert "Y3: _sentinelband- NOT protected — the python3 render was short-circuited by the pin" \
    bash -c '! printf "%s\n" "$1" | grep -qF "skipping protected: _sentinelband-probe"' _ "$ERR_OUT"

# ── Y4: the RENDER MULTIPLIER, as a count ─────────────────────────────────────
# "Renders per suite run" = "renders per gc.sh invocation" x "gc.sh invocations
# per sweep". Y2/Y3 pin the first factor at ZERO for the pinned path; this pins
# the second at ONE, so the product stays checkable if either factor moves.
# Counted through the existing stub-with-call-log idiom (_t_gc_stub -> $GC_LOG),
# NOT through strace, which needs ptrace permissions the offline/CI lane may not
# grant.
Y4_ROOT="$(mktemp -d /tmp/test-gc-sweep-y4-XXXXXX)"
_TMPDIRS+=("$Y4_ROOT")
Y4_MOUNT="$Y4_ROOT/worktrees"
mkdir -p "$Y4_MOUNT/_lane-1/target"

Y4_GC_LOG="$Y4_ROOT/gc_calls.log"
Y4_GC_STUB="$Y4_ROOT/gc_stub.sh"
_t_gc_stub "$Y4_GC_STUB"

Y4_DF_STUB="$Y4_ROOT/df_stub.sh"
_df_stub "$Y4_DF_STUB" 107374182400

GC_LOG="$Y4_GC_LOG" REIFY_WARM_LANE_GC_SWEEP_DF="$Y4_DF_STUB" \
    run_sweep --mount "$Y4_MOUNT" --gc-script "$Y4_GC_STUB" --critical-free-gib 2

assert "Y4: exactly ONE gc.sh invocation per sweep (the render multiplier, as a count)" \
    bash -c '[ "$(wc -l < "$1")" -eq 1 ]' _ "$Y4_GC_LOG"

test_summary
