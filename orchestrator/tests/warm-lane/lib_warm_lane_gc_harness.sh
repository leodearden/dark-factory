#!/usr/bin/env bash
# orchestrator/tests/warm-lane/lib_warm_lane_gc_harness.sh
# The runner and fixture helpers shared by the two warm-lane-gc.sh suites,
# test_warm_lane_gc.sh and test_warm_lane_gc_native.sh.
#
# Caller contract: before the first run_helper call, the caller sets SCRIPT to
# the warm-lane-gc.sh under test and ERR_FILE to a scratch file registered in
# its own _TMPDIRS. run_helper sets OUT, ERR_OUT and RC.
#
# No EXIT trap, _TMPDIRS or cleanup() lives here, for the reason
# test_helpers.sh gives: bash EXIT traps do not stack, so each suite owns its
# own lifecycle.

_GC_HARNESS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$_GC_HARNESS_LIB_DIR/lib_warm_lane_paths.sh"

# The SHIPPED lane-state lib, sourced for ONE value: $LANE_PROTECT_GLOB_FALLBACK
# (see the BRIDGE-COST SEAM comment above run_helper). Reading it from here
# rather than writing the glob literal into this file is deliberate — a literal
# would be a fresh hand-maintained mirror of dark-factory's PROTECTED_PREFIXES,
# the exact INV-5 lockstep-duplication class task 3292 deletes from
# warm-lane-gc.sh, and it would sit OUTSIDE the drift gate
# (orchestrator/tests/test_lane_state_lib.py::TestProtectGlobFallbackDrift) that
# already covers this constant. Sourcing costs nothing: the lib does pure path
# arithmetic at source time and no python work, and it carries its own
# double-source guard.
# shellcheck source=orchestrator/scripts/warm-lane/lib_lane_state.sh
source "$WARM_LANE_SCRIPTS_DIR/lib_lane_state.sh"

# ── run_helper / run_helper_live_default ──────────────────────────────────────
# Both invoke warm-lane-gc.sh, capturing OUT (stdout), ERR_OUT (stderr), RC.
# ONE capture site (_run_gc) so the two wrappers cannot drift apart.
#
# BRIDGE-COST SEAM (task 3292)
# ----------------------------
# gc.sh's PROTECT_GLOB default is no longer a hardcoded literal: it renders
# dark-factory's PROTECTED_PREFIXES through lib_lane_state.sh's
# lane_protect_glob, which starts a python3 and imports orchestrator.git_ops
# (and transitively pydantic) ONCE PER INVOCATION — measured 1.9-5.6s per
# invocation on the development host, against the ~22 invocations in the gc
# suites that reach glob resolution. That is tens of seconds of added wall-clock for a
# resolution almost none of those blocks is asserting anything about.
#
# run_helper therefore PINS the glob to the shipped static fallback through
# gc.sh's own already-documented REIFY_WARM_LANE_GC_PROTECT_GLOB knob, which
# gc.sh's `[ -n "$PROTECT_GLOB" ] ||` default short-circuits BEFORE the bridge
# runs. No new production surface is added for the tests' benefit: this is the
# knob gc.sh already ships and documents. The pin is an inline command prefix,
# NEVER an `export`, so the test shell's own environment stays clean for every
# other call site (same idiom as test_warm_lane_gc.sh's K4 PATH prefix and
# O-env).
#
# The pin is behaviour-NEUTRAL, not merely believed to be. $LANE_PROTECT_GLOB_FALLBACK
# EQUALS, byte for byte, what gc.sh renders by default, and
# orchestrator/tests/test_lane_state_lib.py::TestProtectGlobFallbackIsAVerbatimRender
# fails the build on any difference in either direction. If pinning ever MOVES
# an existing assert, the seam — not the assert — is what needs revisiting.
#
# >>> A BLOCK THAT ASSERTS WHAT THE **DEFAULT** PROTECTS MUST CALL
# >>> run_helper_live_default, OR IT SILENTLY TESTS THE PIN INSTEAD OF THE
# >>> DEFAULT. Current users: test_warm_lane_gc.sh's M, N-default and O (both
# >>> sub-cases); test_warm_lane_gc_native.sh's X-band.
_run_gc() {
    local mode="$1"; shift
    local rc=0
    > "$ERR_FILE"
    if [ "$mode" = "pin" ]; then
        OUT="$(REIFY_WARM_LANE_GC_PROTECT_GLOB="$LANE_PROTECT_GLOB_FALLBACK" \
            bash "$SCRIPT" "$@" 2>"$ERR_FILE")" || rc=$?
    else
        OUT="$(bash "$SCRIPT" "$@" 2>"$ERR_FILE")" || rc=$?
    fi
    ERR_OUT="$(cat "$ERR_FILE")"
    RC=$rc
}

# Pinned — the default for every block that is NOT asserting the protect glob.
run_helper() { _run_gc pin "$@"; }

# Unpinned — gc.sh resolves its PROTECT_GLOB default exactly as dark-factory's
# ε reclaim call site does (git_ops.py _run_warm_lane_gc_reclaim passes --mount
# and nothing else). Use this, and only this, when the assertion is about what
# the DEFAULT protects.
run_helper_live_default() { _run_gc live "$@"; }

# ── git repo factory ───────────────────────────────────────────────────────────
# make_repo DIR  — create a bare-minimum git repo at DIR with one initial commit.
# Sets global REPO_DIR to the created path.
# Always creates the 'main' branch (requires git >= 2.28; -b flag).
make_repo() {
    local dir="$1"
    git init -q -b main "$dir"
    git -C "$dir" config user.email "test@test.local"
    git -C "$dir" config user.name "Test"
    touch "$dir/README.md"
    git -C "$dir" add README.md
    git -C "$dir" commit -q -m "initial"
}

# make_gc_fixture ROOT — build the layout every gc block starts from:
#   ROOT/repo       a one-commit repo on main (make_repo)
#   ROOT/worktrees  the empty mount gc scans for lanes and orphans
#   ROOT/base       refresh-warm-base.sh's generation layout: target -> target.gen.1,
#                   plus that generation's sibling reader lock, target.gen.1.lock
# base sits BESIDE worktrees because that is where `reclaim --mount` derives the
# base from (the parent of MOUNT, then base/target — Block H), so callers read
# the three paths off ROOT rather than receiving them back.
make_gc_fixture() {
    local root="$1"
    mkdir -p "$root/worktrees" "$root/base/target.gen.1"
    make_repo "$root/repo"
    touch "$root/base/target.gen.1.lock"
    ln -sfn "$root/base/target.gen.1" "$root/base/target"
}

# _seed_stub_body — printed to stdout; redirect into a seed-script stub file.
# Logs its argv to $SEED_LOG and removes the lane's divergent marker (simulates
# the α reflink-reseed's thinning effect). Shared by both gc suites' seeded
# fixtures, test_warm_lane_gc.sh's make_task_lane-based Block K among them.
_seed_stub_body() {
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
# as soon as it appears, or non-zero once the generous anti-hang deadline
# (technique T) elapses. The READY marker is touched by a backgrounded lock
# holder AFTER it acquires its flock, so returning 0 causally guarantees the
# flock is held at the caller's next statement — replacing a fixed `sleep`
# that races the background subshell's lock acquisition under load (the
# subshell may not have won the lock within a short fixed sleep, letting a
# competing acquisition — e.g. the GC sweep under test — win instead).
# Mirrors tests/infra/test_warm_lane_pool.sh's identically-named helper.
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
