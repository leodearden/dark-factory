#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_warm_lane_gc.sh
# Hermetic tests for orchestrator/scripts/warm-lane/warm-lane-gc.sh.
#
# Seed-script stub:
#   Wired via --seed-script flag (overrides the default sibling seed-warm-lane.sh).
#   Records argv to a log file and simulates thinning by removing the divergent
#   marker file from the lane's target/ directory.
#
# run_helper captures STDOUT, STDERR, and RC separately:
#   OUT     — captured stdout from the script
#   ERR_OUT — captured stderr from the script
#   RC      — exit code
#
# Blocks:
#   A — CLI guard: --help, unknown flag, bare invocation, unknown subcommand,
#       reclaim missing --worktrees-dir or --base-target; --status-cmd is now
#       an unknown flag (A8) — the Tier-3 machinery was removed (task 5326);
#       a MISSING sibling scripts/lib_live_refs.sh fails LOUD with exit 2
#       (wiring error) before argv is parsed (A9, task 5572 review)
#   B — reset a divergent FREE lane (seed-script invoked with resolved gen path)
#   C — remove an orphaned-landed clean worktree
#   D — always-reclaim (task 5326): a DIRTY POOL LANE is now RECLAIMED (reset),
#       while a DIRTY ORPHAN stays preserved by Pass-2 _is_reclaimable
#   E — always-reclaim (task 5326): an AHEAD-OF-MAIN POOL LANE is now RECLAIMED
#       (no --status-cmd wired — terminality- AND ahead-independent), while an
#       AHEAD-OF-MAIN ORPHAN stays preserved by Pass-2 _is_reclaimable
#   F — preserve a lane with a live-consumer lock — the SOLE remaining Pass-1
#       preserve gate, holding even for a lane that is ALSO dirty+ahead (5326)
#   G — combined PRD δ signal: pool lanes (clean + dirty) reset, orphan removed,
#       ahead orphan + live-consumer lane + protect-glob preserved + summary line
#   K — disk-pressure fast-path: rm -rf target/ instead of an alpha reseed clone
#       (--disk-pressure / REIFY_WARM_LANE_GC_DISK_PRESSURE); lanes reclaim via
#       always-reclaim, no --status-cmd needed (task 5167, 5326). K5 pins that
#       the live-reference gate covers THIS branch too — Block P only ever
#       drives the α branch, so gc.sh's "before BOTH reset branches" claim was
#       structurally true but unasserted on the destructive one (task 5572
#       review)
#   M — ephemeral verify/sweep worktrees (_mainsweep-*/_mainprobe-*) protected
#       by the DEFAULT protect-glob, DF-faithful (--mount only, no
#       --protect-glob) (task 5221)
#   N — full managed-worktree protect set (_solo-*/_substrate-gate-*/
#       _offline-deep (exact name)/_iact-*) protected by default, plus an
#       explicit --protect-glob override still re-narrows the set (task 5221)
#   O — --extra-protect-glob / REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB ADDS to
#       (never replaces) the default protect set: an extra-matched lane (_lane-9)
#       is preserved alongside a default-protected lane (_mainsweep-x) while a
#       plain lane (_lane-1) is still reset; env-var sub-case drives the same
#       additive protection (task 5378)
#   P — the PRIMITIVE's own per-lane live-consumer check (task 5572): a lane
#       referenced by a live process (cwd/fd/mmap at or under it) is preserved
#       in Pass 1 with NO --extra-protect-glob and a FREE flock — so every
#       caller inherits the guard, including dark-factory's ε reclaim, which
#       never passes that flag. P-basic covers the ε-path gap (live at pass
#       start, plus a free unreferenced control lane and a second-run proof
#       that the preserve is temporary); P-fd pins the OPEN-FD leg of the
#       scanner on its own — cwd outside the mount, so only the fd can explain
#       the preserve (the production shape: a compiler holding an output fd
#       under <lane>/target); P-toctou pins PLACEMENT — a lane that
#       goes live MID-PASS is still preserved, so the check cannot be hoisted
#       into an up-front snapshot without going RED
#   Q — the SAME gate on Pass 2's destructive orphan path (task 5572): a live
#       process reference preserves an orphan worktree from
#       `git worktree remove --force`, while an unreferenced orphan is still
#       removed. Closes the hole that deleting the sweep's wrapper CSV would
#       otherwise open — that CSV enumerated every immediate subdir with a bare
#       */ glob, so it incidentally protected live ORPHANS too
#   R — mmap substring-boundary regression (esc-5378 review), MIGRATED from
#       test_warm_lane_gc_sweep.sh's U14-U18 by task 5572: a live "(deleted)"
#       mmap under a torn-down _lane-10 must NOT protect a free _lane-1 whose
#       basename is merely a name-prefix of it. Lives here now because the
#       boundary matcher is called by gc.sh directly; GREEN on arrival — its
#       job is to keep the regression pinned at the right layer, not to go red
#   S — the DURABLE LANE RECORD is Pass 1's reclaimability gate (task 3075,
#       reify esc-5334-6): a lane whose <worktrees-dir>/.lane-state/<lane>.json
#       reads `assigned`/`in_use` is preserved, with a FREE flock and NO
#       --extra-protect-glob — the dark-factory ε shape. S-basic pins the
#       preserve and that a `released` lane still reclaims; S-reasons pins the
#       reason vocabulary (the raw state stays visible; a null task_id never
#       renders a dangling id), the GATE ORDER (record before the /proc walk),
#       and the appended preserved_assigned= counter
#
# The former Tier-3 blocks I/J/L (terminal-task reclaim + Pass-2 boundary,
# task 5167) were deleted when task 5326 collapsed the Pass-1 gate to the
# live-consumer flock alone (always-reclaim): their essential coverage —
# live-consumer preserves a reclaimable lane, ahead-of-main orphan preserved —
# migrated into the rewritten Blocks F and E.
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (PORTED_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
SCRIPT="$WARM_LANE_SCRIPTS_DIR/warm-lane-gc.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=tests/infra/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"

echo "=== scripts/warm-lane-gc.sh hermetic tests (task 4717) ==="

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
init_isolated_lane_root test-gc

ERR_FILE="$(mktemp /tmp/test-warm-lane-gc-err-XXXXXX)"
_TMPDIRS+=("$ERR_FILE")

# ── run_helper ─────────────────────────────────────────────────────────────────
# Invokes warm-lane-gc.sh, capturing OUT (stdout), ERR_OUT (stderr), RC.
run_helper() {
    local rc=0
    > "$ERR_FILE"
    OUT="$(bash "$SCRIPT" "$@" 2>"$ERR_FILE")" || rc=$?
    ERR_OUT="$(cat "$ERR_FILE")"
    RC=$rc
}

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

# ──────────────────────────────────────────────────────────────────────────────
# Shared scaffolding: task-lane factory + causal live-consumer-lock handshake
# ──────────────────────────────────────────────────────────────────────────────
# The Tier-3 status-oracle stubs (gc-status-oracle{,-fail,-counting}.sh) were
# removed with the Tier-3 machinery in task 5326 (always-reclaim collapses the
# Pass-1 gate to the live-consumer flock alone). make_task_lane is retained: it
# still builds the ahead-of-main / dirty pool-lane fixtures that Blocks E/K now
# reclaim via always-reclaim, and the ahead-of-main ORPHAN fixtures Pass 2 still
# preserves.

# make_task_lane REPO WORKTREES NAME BRANCH [ahead]
# Creates a lane at WORKTREES/NAME as a git worktree checked out on BRANCH
# (task/NNNN), seeded with a target/DIVERGENT_MARKER. When ahead="ahead", adds
# a committed change NOT reachable from main (rebase-orphan simulation: HEAD
# stands in for a task that landed via merge-train REBASE, so its actual work
# is on main under new SHAs while this lane's branch tip is an orphan —
# `git merge-base --is-ancestor HEAD main` is false).
make_task_lane() {
    local repo="$1" worktrees="$2" name="$3" branch="$4" ahead="${5:-}"
    git -C "$repo" worktree add -q -b "$branch" "$worktrees/$name"
    if [ "$ahead" = "ahead" ]; then
        echo "ahead (rebase-orphan simulation)" >> "$worktrees/$name/README.md"
        git -C "$worktrees/$name" add README.md
        git -C "$worktrees/$name" commit -q -m "rebase-orphan simulation: ahead-of-main tip"
    fi
    mkdir -p "$worktrees/$name/target"
    touch "$worktrees/$name/target/DIVERGENT_MARKER"
}

# _seed_stub_body — printed to stdout; redirect into a seed-script stub file.
# Logs its argv to $SEED_LOG and removes the lane's divergent marker (simulates
# the α reflink-reseed's thinning effect). Shared by the make_task_lane-based
# fixtures (Block K).
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

# _wait_for_exec_map <pid> <want-exe-realpath> <deadline-seconds>
# The MMAP analogue of _wait_for_reader_lock: `exec` REPLACES the shell so it
# cannot touch a READY marker, so poll /proc/<pid>/exe until it resolves to the
# exec'd binary — at which point the binary's mapping is provably established —
# before reclaim scans (causal ordering, technique R, no wall-clock sleep).
# Used by Block R. Migrated from tests/infra/test_warm_lane_gc_sweep.sh with the
# substring-boundary regression it serves (task 5572), which now belongs at this
# layer because the boundary logic lives in scripts/lib_live_refs.sh, called by
# gc.sh directly.
_wait_for_exec_map() {
    local pid="$1" want="$2" deadline_s="${3:-30}"
    local max_ticks=$(( deadline_s * 20 )) tick=0 exe
    while [ "$tick" -lt "$max_ticks" ]; do
        exe="$(readlink "/proc/$pid/exe" 2>/dev/null || true)"
        [ "$exe" = "$want" ] && return 0
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

# A1: --help exits 0 and prints usage on stderr
run_helper --help
assert "A1: --help exits 0" test "$RC" -eq 0
assert "A1: --help prints 'usage' or 'Usage' on stderr" \
    bash -c 'printf "%s\n" "$1" | grep -qi "usage"' _ "$ERR_OUT"

# A2: unknown flag exits 2
run_helper --unknown-flag-xyz
assert "A2: unknown flag exits 2" test "$RC" -eq 2

# A3: bare invocation (no subcommand) exits 2
run_helper
assert "A3: bare invocation exits 2" test "$RC" -eq 2

# A4: unknown subcommand exits 2
run_helper frobulate
assert "A4: unknown subcommand exits 2" test "$RC" -eq 2

# A5: reclaim without --worktrees-dir exits 2
run_helper reclaim --base-target /tmp/some-base
assert "A5: reclaim without --worktrees-dir exits 2" test "$RC" -eq 2

# A6: reclaim without --base-target exits 2
run_helper reclaim --worktrees-dir /tmp/some-dir
assert "A6: reclaim without --base-target exits 2" test "$RC" -eq 2

# A7: reclaim with both required flags exits 0 (empty worktrees-dir is valid)
A7_WORKTREES="$(mktemp -d /tmp/test-gc-a7-XXXXXX)"
_TMPDIRS+=("$A7_WORKTREES")
A7_BASE="$(mktemp -d /tmp/test-gc-a7-base-XXXXXX)"
_TMPDIRS+=("$A7_BASE")
# Create a gen dir so the base-target resolution works
mkdir -p "$A7_BASE/target.gen.1"
touch "$A7_BASE/target.gen.1.lock"
ln -sfn "$A7_BASE/target.gen.1" "$A7_BASE/target"
run_helper reclaim --worktrees-dir "$A7_WORKTREES" --base-target "$A7_BASE/target"
assert "A7: empty worktrees-dir exits 0" test "$RC" -eq 0

# A8: --status-cmd is now an UNKNOWN flag (Tier-3 machinery removed, task 5326).
# Locks the flag removal: reclaim --worktrees-dir … --base-target … --status-cmd X
# must exit 2 (usage error, unknown flag), not accept it. RED against the current
# gc.sh, which still parses --status-cmd; GREEN once step-2 deletes the flag.
run_helper reclaim \
    --worktrees-dir "$A7_WORKTREES" \
    --base-target "$A7_BASE/target" \
    --status-cmd /bin/true
assert "A8: --status-cmd rejected as unknown flag (exit 2)" test "$RC" -eq 2

# A9: a MISSING sibling scripts/lib_live_refs.sh is a fail-LOUD wiring error
# (task 5572 review). The whole point of task 5572 is that the liveness guard
# must never be silently absent, so the guard that enforces its presence cannot
# itself be untested. gc.sh is copied ALONE into a temp dir — no sibling lib —
# and invoked with --help, which normally exits 0: the guard fires before argv
# is even parsed, so a 2 here also pins that ordering. Exit 2, not 1: an
# incomplete deployment is a wiring error, matching gc.sh's own exit-code table
# and its "exit 2 is reserved for usage/wiring errors" comment.
A9_DIR="$(mktemp -d /tmp/test-gc-a9-XXXXXX)"
_TMPDIRS+=("$A9_DIR")
cp "$SCRIPT" "$A9_DIR/warm-lane-gc.sh"
assert "A9: fixture — the copy really has no sibling lib_live_refs.sh" \
    bash -c '[ ! -e "$1/lib_live_refs.sh" ]' _ "$A9_DIR"
A9_RC=0
A9_ERR="$(bash "$A9_DIR/warm-lane-gc.sh" --help 2>&1 >/dev/null)" || A9_RC=$?
assert "A9: missing sibling lib_live_refs.sh exits 2 (wiring error, not runtime 1)" \
    test "$A9_RC" -eq 2
assert "A9: stderr names the missing library (fail LOUD, not silent)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "lib_live_refs.sh not found"' _ "$A9_ERR"

# ──────────────────────────────────────────────────────────────────────────────
# Block B — reset a divergent FREE lane
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block B: reset a divergent FREE lane ---"

B_ROOT="$(mktemp -d /tmp/test-gc-b-XXXXXX)"
_TMPDIRS+=("$B_ROOT")

B_REPO="$B_ROOT/repo"
B_WORKTREES="$B_ROOT/worktrees"
B_BASE="$B_ROOT/base"
mkdir -p "$B_WORKTREES" "$B_BASE"

# Set up primary git repo
make_repo "$B_REPO"

# Create base gen directory (simulates the warm base)
mkdir -p "$B_BASE/target.gen.1"
touch "$B_BASE/target.gen.1.lock"
ln -sfn "$B_BASE/target.gen.1" "$B_BASE/target"

# Create _lane-1 as a git worktree (clean, HEAD == main, i.e. landed)
git -C "$B_REPO" worktree add -q "$B_WORKTREES/_lane-1"
# Add a divergent marker in target/ to prove it gets thinned
mkdir -p "$B_WORKTREES/_lane-1/target"
touch "$B_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

# Seed-script stub: records argv to a log file and simulates thinning
B_SEED_LOG="$B_ROOT/seed_calls.log"
B_SEED_STUB="$B_ROOT/seed_stub.sh"
cat > "$B_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
# Seed-script stub: log all argv, simulate thinning
echo "$*" >> "$SEED_LOG"
# Simulate thinning: remove any DIVERGENT_MARKER in the lane's target/
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$B_SEED_STUB"
export SEED_LOG="$B_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$B_WORKTREES" \
    --base-target "$B_BASE/target" \
    --seed-script "$B_SEED_STUB"

assert "B1: exit 0" test "$RC" -eq 0
assert "B2: seed-script was invoked for _lane-1" test -f "$B_SEED_LOG"
assert "B3: seed-script received resolved gen path (not symlink)" \
    bash -c 'grep -q "target.gen.1" "$1"' _ "$B_SEED_LOG"
assert "B4: seed-script received --fresh-checkout" \
    bash -c 'grep -q -- "--fresh-checkout" "$1"' _ "$B_SEED_LOG"
# gc holds the lane flock on FD 8 (line 417) AND the gen lock (flock -s) on
# FD 9 (line 461) itself across the seed call, so the seed must NOT re-acquire
# the lane lock on FD 9 — that would both self-refuse against gc's FD-8 lane
# lock and clobber gc's FD-9 gen lock. gc therefore passes --assume-lane-lock-held
# so the seed skips its own (now default-on) acquire (esc-5214/task 5354).
assert "B4b: seed-script received --assume-lane-lock-held (gc already holds the lane+gen locks; task 5354)" \
    bash -c 'grep -q -- "--assume-lane-lock-held" "$1"' _ "$B_SEED_LOG"
assert "B5: divergent target marker removed (thinned)" \
    bash -c '[ ! -f "$1" ]' _ "$B_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

# ──────────────────────────────────────────────────────────────────────────────
# Block C — remove an orphaned-landed clean worktree
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block C: remove an orphaned-landed clean worktree ---"

C_ROOT="$(mktemp -d /tmp/test-gc-c-XXXXXX)"
_TMPDIRS+=("$C_ROOT")

C_REPO="$C_ROOT/repo"
C_WORKTREES="$C_ROOT/worktrees"
C_BASE="$C_ROOT/base"
mkdir -p "$C_WORKTREES" "$C_BASE"

make_repo "$C_REPO"

mkdir -p "$C_BASE/target.gen.1"
touch "$C_BASE/target.gen.1.lock"
ln -sfn "$C_BASE/target.gen.1" "$C_BASE/target"

# Create _lane-1 (pool lane, reclaimable)
git -C "$C_REPO" worktree add -q "$C_WORKTREES/_lane-1"
mkdir -p "$C_WORKTREES/_lane-1/target"
touch "$C_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

# Create task-9999 (orphan cold worktree, clean, landed in main)
git -C "$C_REPO" worktree add -q "$C_WORKTREES/task-9999"

# Create _merge-verify (protected, must not be touched)
git -C "$C_REPO" worktree add -q "$C_WORKTREES/_merge-verify"
touch "$C_WORKTREES/_merge-verify/PROTECTED_MARKER"

C_SEED_LOG="$C_ROOT/seed_calls.log"
C_SEED_STUB="$C_ROOT/seed_stub.sh"
cat > "$C_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$C_SEED_STUB"
export SEED_LOG="$C_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$C_WORKTREES" \
    --base-target "$C_BASE/target" \
    --seed-script "$C_SEED_STUB"

assert "C1: exit 0" test "$RC" -eq 0
# The orphan should be removed
assert "C2: task-9999 orphan removed (dir gone)" \
    bash -c '[ ! -d "$1" ]' _ "$C_WORKTREES/task-9999"
# task-9999 should no longer appear in worktree list
assert "C3: task-9999 absent from git worktree list" \
    bash -c '! git -C "$1" worktree list | grep -q "task-9999"' _ "$C_REPO"
# _lane-1 should still be reset
assert "C4: _lane-1 seed-script invoked" test -f "$C_SEED_LOG"
# _merge-verify must be untouched (protect-glob)
assert "C5: _merge-verify protected marker intact" \
    test -f "$C_WORKTREES/_merge-verify/PROTECTED_MARKER"

# ──────────────────────────────────────────────────────────────────────────────
# Block D — always-reclaim: dirty POOL LANE reclaimed, dirty ORPHAN preserved
# Under the task-5326 always-reclaim policy, a FREE pool lane whose live-consumer
# flock is free is reclaimed UNCONDITIONALLY — dirty tracked changes no longer
# preserve it (acquire_lane always re-seeds; committed work lives on the branch
# ref; reset touches only target/). A dirty ORPHAN cold worktree (Pass 2) is
# STILL preserved by the unchanged _is_reclaimable predicate.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block D: always-reclaim (dirty pool lane reset, dirty orphan preserved) ---"

D_ROOT="$(mktemp -d /tmp/test-gc-d-XXXXXX)"
_TMPDIRS+=("$D_ROOT")

D_REPO="$D_ROOT/repo"
D_WORKTREES="$D_ROOT/worktrees"
D_BASE="$D_ROOT/base"
mkdir -p "$D_WORKTREES" "$D_BASE"

make_repo "$D_REPO"

mkdir -p "$D_BASE/target.gen.1"
touch "$D_BASE/target.gen.1.lock"
ln -sfn "$D_BASE/target.gen.1" "$D_BASE/target"

# _lane-2: dirty POOL LANE (tracked file modified but not committed) — now RECLAIMED.
git -C "$D_REPO" worktree add -q "$D_WORKTREES/_lane-2"
echo "dirty" >> "$D_WORKTREES/_lane-2/README.md"
mkdir -p "$D_WORKTREES/_lane-2/target"
touch "$D_WORKTREES/_lane-2/target/DIVERGENT_MARKER"

# task-8888: dirty ORPHAN (modified tracked file) — STILL preserved (Pass 2).
git -C "$D_REPO" worktree add -q "$D_WORKTREES/task-8888"
echo "dirty" >> "$D_WORKTREES/task-8888/README.md"

# Thinning seed stub: logs argv AND removes the lane's DIVERGENT_MARKER, so a
# reset is observable both by the seed log and by the marker's disappearance.
D_SEED_LOG="$D_ROOT/seed_calls.log"
D_SEED_STUB="$D_ROOT/seed_stub.sh"
cat > "$D_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$D_SEED_STUB"
export SEED_LOG="$D_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$D_WORKTREES" \
    --base-target "$D_BASE/target" \
    --seed-script "$D_SEED_STUB"

assert "D1: exit 0" test "$RC" -eq 0
# Dirty POOL LANE IS reset: seed-script invoked for it (always-reclaim).
assert "D2: dirty pool lane _lane-2 seed-script invoked (reclaimed)" \
    bash -c 'test -f "$1" && grep -q "_lane-2" "$1"' _ "$D_SEED_LOG"
# Dirty POOL LANE marker removed (thinned by the reset).
assert "D3: dirty pool lane divergent marker removed (reclaimed)" \
    bash -c '[ ! -f "$1" ]' _ "$D_WORKTREES/_lane-2/target/DIVERGENT_MARKER"
# Dirty ORPHAN NOT removed (Pass-2 _is_reclaimable unchanged).
assert "D4: dirty orphan task-8888 still present (preserved by Pass 2)" \
    test -d "$D_WORKTREES/task-8888"
# Stderr should still mention preserving dirty WIP (for the orphan).
assert "D5: stderr mentions dirty WIP preservation (orphan)" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "dirty|preserving|wip|tracked"' _ "$ERR_OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block E — always-reclaim: ahead-of-main POOL LANE reclaimed, ahead ORPHAN preserved
# The ahead-of-main pool lane is reclaimed with NO --status-cmd wired, proving
# the always-reclaim policy is independent of BOTH the ahead-of-main tip AND any
# backing-task terminality (Tier-3 is gone). The ahead-of-main ORPHAN cold
# worktree (Pass 2) is STILL preserved by the unchanged _is_reclaimable predicate.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block E: always-reclaim (ahead pool lane reset, ahead orphan preserved) ---"

E_ROOT="$(mktemp -d /tmp/test-gc-e-XXXXXX)"
_TMPDIRS+=("$E_ROOT")

E_REPO="$E_ROOT/repo"
E_WORKTREES="$E_ROOT/worktrees"
E_BASE="$E_ROOT/base"
mkdir -p "$E_WORKTREES" "$E_BASE"

make_repo "$E_REPO"

mkdir -p "$E_BASE/target.gen.1"
touch "$E_BASE/target.gen.1.lock"
ln -sfn "$E_BASE/target.gen.1" "$E_BASE/target"

# _lane-3: clean POOL LANE with a committed change NOT in main (ahead-of-main) —
# now RECLAIMED via always-reclaim (no --status-cmd wired).
git -C "$E_REPO" worktree add -q "$E_WORKTREES/_lane-3"
echo "ahead" >> "$E_WORKTREES/_lane-3/README.md"
git -C "$E_WORKTREES/_lane-3" add README.md
git -C "$E_WORKTREES/_lane-3" commit -q -m "ahead-of-main commit"
mkdir -p "$E_WORKTREES/_lane-3/target"
touch "$E_WORKTREES/_lane-3/target/DIVERGENT_MARKER"

# task-7777: ahead-of-main ORPHAN — STILL preserved (Pass 2 _is_reclaimable).
git -C "$E_REPO" worktree add -q "$E_WORKTREES/task-7777"
echo "ahead" >> "$E_WORKTREES/task-7777/README.md"
git -C "$E_WORKTREES/task-7777" add README.md
git -C "$E_WORKTREES/task-7777" commit -q -m "ahead-of-main commit"

# Thinning seed stub: logs argv AND removes the lane's DIVERGENT_MARKER.
E_SEED_LOG="$E_ROOT/seed_calls.log"
E_SEED_STUB="$E_ROOT/seed_stub.sh"
cat > "$E_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$E_SEED_STUB"
export SEED_LOG="$E_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$E_WORKTREES" \
    --base-target "$E_BASE/target" \
    --seed-script "$E_SEED_STUB" \
    --main-ref "main"

assert "E1: exit 0" test "$RC" -eq 0
# Ahead-of-main POOL LANE IS reset: seed-script invoked (always-reclaim).
assert "E2: ahead pool lane _lane-3 seed-script invoked (reclaimed)" \
    bash -c 'test -f "$1" && grep -q "_lane-3" "$1"' _ "$E_SEED_LOG"
assert "E3: ahead pool lane divergent marker removed (reclaimed)" \
    bash -c '[ ! -f "$1" ]' _ "$E_WORKTREES/_lane-3/target/DIVERGENT_MARKER"
# Ahead-of-main ORPHAN NOT removed (Pass-2 _is_reclaimable unchanged).
assert "E4: ahead-of-main orphan task-7777 still present (preserved by Pass 2)" \
    test -d "$E_WORKTREES/task-7777"
# Stderr should still mention unlanded/ahead-of-main preservation (for the orphan).
assert "E5: stderr mentions unlanded/ahead-of-main preservation (orphan)" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "unlanded|ahead|preserving"' _ "$ERR_OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block F — live-consumer flock is the SOLE remaining Pass-1 preserve gate
# _lane-4: clean+landed, but a live consumer holds its flock — preserved.
# _lane-5: dirty AND ahead-of-main AND a live consumer holds its flock — STILL
# preserved. Under always-reclaim (task 5326) a dirty+ahead lane WOULD be
# reclaimed if its flock were free (Blocks D/E); the flock alone keeps _lane-5,
# isolating the live-consumer flock as the only Pass-1 preserve gate that survives.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block F: live-consumer flock is the sole Pass-1 preserve gate ---"

F_ROOT="$(mktemp -d /tmp/test-gc-f-XXXXXX)"
_TMPDIRS+=("$F_ROOT")

F_REPO="$F_ROOT/repo"
F_WORKTREES="$F_ROOT/worktrees"
F_BASE="$F_ROOT/base"
mkdir -p "$F_WORKTREES" "$F_BASE"

make_repo "$F_REPO"

mkdir -p "$F_BASE/target.gen.1"
touch "$F_BASE/target.gen.1.lock"
ln -sfn "$F_BASE/target.gen.1" "$F_BASE/target"

# _lane-4: clean, landed, with a live consumer holding the exclusive lock.
git -C "$F_REPO" worktree add -q "$F_WORKTREES/_lane-4"
mkdir -p "$F_WORKTREES/_lane-4/target"
touch "$F_WORKTREES/_lane-4/target/DIVERGENT_MARKER"
touch "$F_WORKTREES/_lane-4.lock"
F4_READY="$F_WORKTREES/_lane-4.lock.ready-marker"
( flock -x 9 && touch "$F4_READY" && sleep 300 ) 9>"$F_WORKTREES/_lane-4.lock" &
F4_LOCK_PID=$!
_BGPIDS+=("$F4_LOCK_PID")

# _lane-5: dirty (uncommitted tracked change) AND ahead-of-main, with a live
# consumer holding the exclusive lock. Both dirty and ahead are now
# non-preserving on their own (Blocks D/E), so only the flock keeps this lane.
git -C "$F_REPO" worktree add -q "$F_WORKTREES/_lane-5"
echo "ahead" >> "$F_WORKTREES/_lane-5/README.md"
git -C "$F_WORKTREES/_lane-5" add README.md
git -C "$F_WORKTREES/_lane-5" commit -q -m "ahead-of-main commit"
echo "dirty uncommitted change" >> "$F_WORKTREES/_lane-5/README.md"
mkdir -p "$F_WORKTREES/_lane-5/target"
touch "$F_WORKTREES/_lane-5/target/DIVERGENT_MARKER"
touch "$F_WORKTREES/_lane-5.lock"
F5_READY="$F_WORKTREES/_lane-5.lock.ready-marker"
( flock -x 9 && touch "$F5_READY" && sleep 300 ) 9>"$F_WORKTREES/_lane-5.lock" &
F5_LOCK_PID=$!
_BGPIDS+=("$F5_LOCK_PID")

# Causal handshake (technique R, #4847): proceed only once BOTH flocks are
# provably held, so the GC sweep under test cannot race in and win either lock.
_wait_for_reader_lock "$F4_READY" 30
_wait_for_reader_lock "$F5_READY" 30

F_SEED_LOG="$F_ROOT/seed_calls.log"
F_SEED_STUB="$F_ROOT/seed_stub.sh"
cat > "$F_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$F_SEED_STUB"
export SEED_LOG="$F_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$F_WORKTREES" \
    --base-target "$F_BASE/target" \
    --seed-script "$F_SEED_STUB" \
    --main-ref "main"

assert "F1: exit 0" test "$RC" -eq 0
# _lane-4 (clean, locked) NOT reset.
assert "F2: live-consumer lane _lane-4 seed-script NOT invoked" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-4" "$1"' _ "$F_SEED_LOG"
assert "F3: live-consumer lane _lane-4 divergent marker intact" \
    test -f "$F_WORKTREES/_lane-4/target/DIVERGENT_MARKER"
assert "F4: stderr mentions live consumer preservation" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "live.consumer|locked|preserving|consumer"' _ "$ERR_OUT"
# _lane-5 (dirty+ahead, locked) STILL NOT reset — the flock is the sole gate.
assert "F5: dirty+ahead live-consumer lane _lane-5 seed-script NOT invoked (flock wins)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-5" "$1"' _ "$F_SEED_LOG"
assert "F6: dirty+ahead live-consumer lane _lane-5 divergent marker intact" \
    test -f "$F_WORKTREES/_lane-5/target/DIVERGENT_MARKER"

# Release the locks
kill "$F4_LOCK_PID" "$F5_LOCK_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Block G — combined PRD δ signal under always-reclaim (task 5326)
# Pool lanes (clean _lane-free + dirty _lane-dirty) are RESET; the clean orphan
# task-free is REMOVED; the ahead-of-main orphan task-ahead, the live-consumer
# lane _lane-locked, and the protected _merge-verify are PRESERVED. Summary:
# reset=2 removed=1 preserved=3.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block G: combined PRD delta signal (always-reclaim) ---"

G_ROOT="$(mktemp -d /tmp/test-gc-g-XXXXXX)"
_TMPDIRS+=("$G_ROOT")

G_REPO="$G_ROOT/repo"
G_WORKTREES="$G_ROOT/worktrees"
G_BASE="$G_ROOT/base"
mkdir -p "$G_WORKTREES" "$G_BASE"

make_repo "$G_REPO"

mkdir -p "$G_BASE/target.gen.1"
touch "$G_BASE/target.gen.1.lock"
ln -sfn "$G_BASE/target.gen.1" "$G_BASE/target"

# Fixture 1: reclaimable FREE lane (_lane-free)
git -C "$G_REPO" worktree add -q "$G_WORKTREES/_lane-free"
mkdir -p "$G_WORKTREES/_lane-free/target"
touch "$G_WORKTREES/_lane-free/target/DIVERGENT_MARKER"

# Fixture 2: reclaimable orphan worktree (task-free)
git -C "$G_REPO" worktree add -q "$G_WORKTREES/task-free"

# Fixture 3: dirty-WIP POOL LANE (_lane-dirty) — now RECLAIMED (always-reclaim)
git -C "$G_REPO" worktree add -q "$G_WORKTREES/_lane-dirty"
echo "dirty" >> "$G_WORKTREES/_lane-dirty/README.md"
mkdir -p "$G_WORKTREES/_lane-dirty/target"
touch "$G_WORKTREES/_lane-dirty/target/DIVERGENT_MARKER"

# Fixture 4: unlanded-ahead orphan (task-ahead)
git -C "$G_REPO" worktree add -q "$G_WORKTREES/task-ahead"
echo "ahead" >> "$G_WORKTREES/task-ahead/README.md"
git -C "$G_WORKTREES/task-ahead" add README.md
git -C "$G_WORKTREES/task-ahead" commit -q -m "ahead of main"

# Fixture 5: live-consumer-locked lane (_lane-locked)
git -C "$G_REPO" worktree add -q "$G_WORKTREES/_lane-locked"
mkdir -p "$G_WORKTREES/_lane-locked/target"
touch "$G_WORKTREES/_lane-locked/target/DIVERGENT_MARKER"
touch "$G_WORKTREES/_lane-locked.lock"
( flock -x 9 && sleep 300 ) 9>"$G_WORKTREES/_lane-locked.lock" &
G_LOCK_PID=$!
_BGPIDS+=("$G_LOCK_PID")
sleep 0.1

# Fixture 6: protected _merge-verify (protect-glob)
git -C "$G_REPO" worktree add -q "$G_WORKTREES/_merge-verify"
touch "$G_WORKTREES/_merge-verify/PROTECTED_MARKER"

G_SEED_LOG="$G_ROOT/seed_calls.log"
G_SEED_STUB="$G_ROOT/seed_stub.sh"
cat > "$G_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$G_SEED_STUB"
export SEED_LOG="$G_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$G_WORKTREES" \
    --base-target "$G_BASE/target" \
    --seed-script "$G_SEED_STUB" \
    --main-ref "main"

assert "G1: exit 0" test "$RC" -eq 0

# Reclaimable clean pool lane was reset
assert "G2: reclaimable lane _lane-free was reset (seed-script invoked)" \
    bash -c 'grep -q "_lane-free" "$1"' _ "$G_SEED_LOG"
assert "G3: reclaimable lane divergent marker removed" \
    bash -c '[ ! -f "$1" ]' _ "$G_WORKTREES/_lane-free/target/DIVERGENT_MARKER"

# Reclaimable orphan was removed
assert "G4: reclaimable orphan task-free was removed" \
    bash -c '[ ! -d "$1" ]' _ "$G_WORKTREES/task-free"

# Dirty POOL LANE now reclaimed too (always-reclaim, task 5326)
assert "G5: dirty pool lane _lane-dirty divergent marker removed (reclaimed)" \
    bash -c '[ ! -f "$1" ]' _ "$G_WORKTREES/_lane-dirty/target/DIVERGENT_MARKER"

# Preserved fixtures: ahead ORPHAN (Pass 2), live-consumer lane, protected glob
assert "G6: ahead orphan task-ahead still present" \
    test -d "$G_WORKTREES/task-ahead"
assert "G7: locked lane _lane-locked marker intact" \
    test -f "$G_WORKTREES/_lane-locked/target/DIVERGENT_MARKER"

# Protected glob untouched
assert "G8: _merge-verify protected marker intact" \
    test -f "$G_WORKTREES/_merge-verify/PROTECTED_MARKER"

# Summary line on stdout with reset/removed/preserved counts
assert "G9: stdout contains machine-readable summary" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reclaim:.*reset=.*removed=.*preserved="' _ "$OUT"
assert "G10: summary shows reset=2 (_lane-free + _lane-dirty)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=2"' _ "$OUT"
assert "G11: summary shows removed=1" \
    bash -c 'printf "%s\n" "$1" | grep -qE "removed=1"' _ "$OUT"
assert "G12: summary shows preserved=3 (task-ahead + _lane-locked + _merge-verify)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=3"' _ "$OUT"

# Release lock
kill "$G_LOCK_PID" 2>/dev/null || true
_BGPIDS=()

# ──────────────────────────────────────────────────────────────────────────────
# Block H — --mount derivation: DF consumer contract
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block H: --mount derivation (DF consumer contract) ---"

H_ROOT="$(mktemp -d /tmp/test-gc-h-XXXXXX)"
_TMPDIRS+=("$H_ROOT")

H_REPO="$H_ROOT/repo"
H_WORKTREES="$H_ROOT/worktrees"
H_BASE="$H_ROOT/base"
mkdir -p "$H_WORKTREES" "$H_BASE"

# Set up primary git repo
make_repo "$H_REPO"

# Create base gen directory (simulates the warm base)
# Layout: <root>/worktrees/ (mount) + <root>/base/target.gen.1 (sibling)
mkdir -p "$H_BASE/target.gen.1"
touch "$H_BASE/target.gen.1.lock"
ln -sfn "$H_BASE/target.gen.1" "$H_BASE/target"

# Create _lane-1 as a git worktree (clean, HEAD == main, i.e. landed)
git -C "$H_REPO" worktree add -q "$H_WORKTREES/_lane-1"
# Add a divergent marker in target/ to prove it gets thinned
mkdir -p "$H_WORKTREES/_lane-1/target"
touch "$H_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

# Seed-script stub: records argv to a log file and simulates thinning
H_SEED_LOG="$H_ROOT/seed_calls.log"
H_SEED_STUB="$H_ROOT/seed_stub.sh"
cat > "$H_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
# Seed-script stub: log all argv, simulate thinning
echo "$*" >> "$SEED_LOG"
# Simulate thinning: remove any DIVERGENT_MARKER in the lane's target/
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$H_SEED_STUB"
export SEED_LOG="$H_SEED_LOG"

# H1–H5: reclaim --mount <root>/worktrees (NO --worktrees-dir/--base-target)
run_helper reclaim \
    --mount "$H_WORKTREES" \
    --seed-script "$H_SEED_STUB"

assert "H1: --mount exits 0" test "$RC" -eq 0
assert "H2: seed-stub invoked for _lane-1 (WORKTREES_DIR derived = \$MOUNT)" \
    bash -c 'test -f "$1" && grep -q "_lane-1" "$1"' _ "$H_SEED_LOG"
assert "H3: seed-stub received resolved base/target.gen.1 path (BASE_TARGET derived = \$(dirname \$MOUNT)/base/target, symlink-resolved)" \
    bash -c 'grep -q "target.gen.1" "$1"' _ "$H_SEED_LOG"
assert "H4: --fresh-checkout passed through to seed-stub" \
    bash -c 'grep -q -- "--fresh-checkout" "$1"' _ "$H_SEED_LOG"
assert "H5: divergent target marker removed (thinned)" \
    bash -c '[ ! -f "$1" ]' _ "$H_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

# H6: explicit --base-target together with --mount overrides the derived base
H6_ROOT="$(mktemp -d /tmp/test-gc-h6-XXXXXX)"
_TMPDIRS+=("$H6_ROOT")
H6_REPO="$H6_ROOT/repo"
H6_WORKTREES="$H6_ROOT/worktrees"
H6_ALT_BASE="$H6_ROOT/alt_base"
mkdir -p "$H6_WORKTREES" "$H6_ALT_BASE"
make_repo "$H6_REPO"
mkdir -p "$H6_ALT_BASE/target.gen.99"
touch "$H6_ALT_BASE/target.gen.99.lock"
ln -sfn "$H6_ALT_BASE/target.gen.99" "$H6_ALT_BASE/target"
# Lane: landed and clean
git -C "$H6_REPO" worktree add -q "$H6_WORKTREES/_lane-1"
mkdir -p "$H6_WORKTREES/_lane-1/target"
touch "$H6_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

H6_SEED_LOG="$H6_ROOT/seed_calls.log"
H6_SEED_STUB="$H6_ROOT/seed_stub.sh"
cat > "$H6_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$H6_SEED_STUB"
export SEED_LOG="$H6_SEED_LOG"

run_helper reclaim \
    --mount "$H6_WORKTREES" \
    --base-target "$H6_ALT_BASE/target" \
    --seed-script "$H6_SEED_STUB"

assert "H6: explicit --base-target overrides --mount derived base (uses alt gen.99)" \
    bash -c 'grep -q "target.gen.99" "$1"' _ "$H6_SEED_LOG"

# ──────────────────────────────────────────────────────────────────────────────
# Block K — disk-pressure fast-path
# Under --disk-pressure, a reclaimable lane's target/ is deleted outright
# (rm -rf) instead of going through the alpha reflink-reseed clone — valid
# because acquire_lane always re-seeds from base. Mirrors the manual
# 2026-07-10 remediation for the 768G/404G leaked lanes.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block K: disk-pressure fast-path ---"

K_ROOT="$(mktemp -d /tmp/test-gc-k-XXXXXX)"
_TMPDIRS+=("$K_ROOT")

# ── K1: --disk-pressure deletes target/ outright, no seed-script invocation ────
K1_REPO="$K_ROOT/k1-repo"
K1_WORKTREES="$K_ROOT/k1-worktrees"
K1_BASE="$K_ROOT/k1-base"
mkdir -p "$K1_WORKTREES" "$K1_BASE"
make_repo "$K1_REPO"
mkdir -p "$K1_BASE/target.gen.1"
touch "$K1_BASE/target.gen.1.lock"
ln -sfn "$K1_BASE/target.gen.1" "$K1_BASE/target"

make_task_lane "$K1_REPO" "$K1_WORKTREES" "_lane-1" "task/4827" "ahead"

K1_SEED_LOG="$K_ROOT/k1-seed-calls.log"
K1_SEED_STUB="$K_ROOT/k1-seed-stub.sh"
_seed_stub_body > "$K1_SEED_STUB"
chmod +x "$K1_SEED_STUB"
export SEED_LOG="$K1_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$K1_WORKTREES" \
    --base-target "$K1_BASE/target" \
    --seed-script "$K1_SEED_STUB" \
    --disk-pressure

assert "K1: exit 0" test "$RC" -eq 0
assert "K1: target/ directory deleted outright (disk-pressure fast-path)" \
    bash -c '[ ! -d "$1" ]' _ "$K1_WORKTREES/_lane-1/target"
assert "K1: seed-script NOT invoked (no reflink clone under disk-pressure)" \
    bash -c '[ ! -f "$1" ]' _ "$K1_SEED_LOG"
assert "K1: summary line parses as reclaim: reset=.. removed=.. preserved=.." \
    bash -c 'printf "%s\n" "$1" | grep -qE "reclaim:.*reset=.*removed=.*preserved="' _ "$OUT"
assert "K1: summary shows reset=1 (disk-pressure delete still counts as reset)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1"' _ "$OUT"

# ── K2: control — same fixture WITHOUT --disk-pressure resets via alpha ────────
K2_REPO="$K_ROOT/k2-repo"
K2_WORKTREES="$K_ROOT/k2-worktrees"
K2_BASE="$K_ROOT/k2-base"
mkdir -p "$K2_WORKTREES" "$K2_BASE"
make_repo "$K2_REPO"
mkdir -p "$K2_BASE/target.gen.1"
touch "$K2_BASE/target.gen.1.lock"
ln -sfn "$K2_BASE/target.gen.1" "$K2_BASE/target"

make_task_lane "$K2_REPO" "$K2_WORKTREES" "_lane-1" "task/4827" "ahead"

K2_SEED_LOG="$K_ROOT/k2-seed-calls.log"
K2_SEED_STUB="$K_ROOT/k2-seed-stub.sh"
_seed_stub_body > "$K2_SEED_STUB"
chmod +x "$K2_SEED_STUB"
export SEED_LOG="$K2_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$K2_WORKTREES" \
    --base-target "$K2_BASE/target" \
    --seed-script "$K2_SEED_STUB"

assert "K2: exit 0" test "$RC" -eq 0
assert "K2: seed-script invoked (normal alpha reset path, no disk-pressure)" \
    bash -c 'test -f "$1" && grep -q "_lane-1" "$1"' _ "$K2_SEED_LOG"
assert "K2: target/ directory still present (thinned by alpha, not deleted)" \
    test -d "$K2_WORKTREES/_lane-1/target"
assert "K2: summary shows reset=1" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1"' _ "$OUT"

# ── K3: landed-clean NON-task-branch lane under --disk-pressure ────────────────
# --disk-pressure is documented (usage()/header) as applying to EVERY
# reclaimable Pass-1 lane. This locks that pool-wide scope: an ordinary
# landed/clean lane (no task/NNNN branch) also gets its target/ deleted
# outright under --disk-pressure rather than alpha-reseeded.
K3_REPO="$K_ROOT/k3-repo"
K3_WORKTREES="$K_ROOT/k3-worktrees"
K3_BASE="$K_ROOT/k3-base"
mkdir -p "$K3_WORKTREES" "$K3_BASE"
make_repo "$K3_REPO"
mkdir -p "$K3_BASE/target.gen.1"
touch "$K3_BASE/target.gen.1.lock"
ln -sfn "$K3_BASE/target.gen.1" "$K3_BASE/target"

# Landed, clean, non-task-branch lane (mirrors Block B's fixture).
git -C "$K3_REPO" worktree add -q "$K3_WORKTREES/_lane-1"
mkdir -p "$K3_WORKTREES/_lane-1/target"
touch "$K3_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

K3_SEED_LOG="$K_ROOT/k3-seed-calls.log"
K3_SEED_STUB="$K_ROOT/k3-seed-stub.sh"
_seed_stub_body > "$K3_SEED_STUB"
chmod +x "$K3_SEED_STUB"
export SEED_LOG="$K3_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$K3_WORKTREES" \
    --base-target "$K3_BASE/target" \
    --seed-script "$K3_SEED_STUB" \
    --disk-pressure

assert "K3: exit 0" test "$RC" -eq 0
assert "K3: landed-clean non-task-branch lane target/ deleted outright (pool-wide disk-pressure scope)" \
    bash -c '[ ! -d "$1" ]' _ "$K3_WORKTREES/_lane-1/target"
assert "K3: seed-script NOT invoked (no reflink clone under disk-pressure)" \
    bash -c '[ ! -f "$1" ]' _ "$K3_SEED_LOG"
assert "K3: summary shows reset=1" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1"' _ "$OUT"

# ── K4: rm failure under --disk-pressure is preserved, not silently reset ──────
# The rm-failure branch (warn + preserve, scripts/warm-lane-gc.sh) is the
# failure-mode handling for the exact scenario --disk-pressure exists to
# remediate (ENOSPC / disk pressure) — an untested regression here would
# silently mis-count reclaim results. Stubs `rm` via PATH to fail
# deterministically: unlike a chmod-0-parent-dir permission trick, this is
# root-safe (permission bits are bypassed when the test process runs as
# root, which a filesystem-based trick cannot guarantee against).
K4_REPO="$K_ROOT/k4-repo"
K4_WORKTREES="$K_ROOT/k4-worktrees"
K4_BASE="$K_ROOT/k4-base"
mkdir -p "$K4_WORKTREES" "$K4_BASE"
make_repo "$K4_REPO"
mkdir -p "$K4_BASE/target.gen.1"
touch "$K4_BASE/target.gen.1.lock"
ln -sfn "$K4_BASE/target.gen.1" "$K4_BASE/target"

make_task_lane "$K4_REPO" "$K4_WORKTREES" "_lane-1" "task/4827" "ahead"

K4_SEED_LOG="$K_ROOT/k4-seed-calls.log"
K4_SEED_STUB="$K_ROOT/k4-seed-stub.sh"
_seed_stub_body > "$K4_SEED_STUB"
chmod +x "$K4_SEED_STUB"
export SEED_LOG="$K4_SEED_LOG"

# rm stub: always fails — simulates rm -rf hitting EACCES / a busy mount /
# an immutable file under real disk pressure. Prepended to PATH only for
# the run_helper call below (bash's temporary-environment semantics for a
# prefix assignment on a function call: visible for that call's duration,
# including subprocesses it spawns, and reverted immediately after).
K4_RMSTUB_DIR="$(mktemp -d /tmp/test-gc-k4-rmstub-XXXXXX)"
_TMPDIRS+=("$K4_RMSTUB_DIR")
cat > "$K4_RMSTUB_DIR/rm" << 'STUB_EOF'
#!/usr/bin/env bash
echo "rm: cannot remove target: simulated failure (test stub)" >&2
exit 1
STUB_EOF
chmod +x "$K4_RMSTUB_DIR/rm"

PATH="$K4_RMSTUB_DIR:$PATH" run_helper reclaim \
    --worktrees-dir "$K4_WORKTREES" \
    --base-target "$K4_BASE/target" \
    --seed-script "$K4_SEED_STUB" \
    --disk-pressure

assert "K4: exit 0 (rm failure does not abort the sweep)" test "$RC" -eq 0
assert "K4: seed-script NOT invoked (disk-pressure path has no alpha fallback)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-1" "$1"' _ "$K4_SEED_LOG"
assert "K4: target/ NOT removed (rm failed, lane left untouched)" \
    test -d "$K4_WORKTREES/_lane-1/target"
assert "K4: divergent marker intact (nothing was reset)" \
    test -f "$K4_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "K4: summary counts the failed reset as preserved, not reset" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=0 removed=0 preserved=1"' _ "$OUT"
assert "K4: stderr captures the rm failure detail (not swallowed)" \
    bash -c 'printf "%s\n" "$1" | grep -qi "simulated failure"' _ "$ERR_OUT"

# ── K5: --disk-pressure ALSO honours the live-reference gate (task 5572) ───────
# gc.sh's Pass-1 comment claims the gate sits "before BOTH reset branches ... so
# neither path can bypass it". Block P only ever drives the α branch, so that
# claim was structurally true but never asserted — and the branch it left
# unasserted is the destructive one: --disk-pressure `rm -rf <lane>/target`
# obliterates a live build's output outright, with no seed-script in the way to
# make the mistake visible. This is also the emergency path (ENOSPC response),
# i.e. exactly when an operator is least able to notice a corrupted build
# (task 5572 review).
K5_REPO="$K_ROOT/k5-repo"
K5_WORKTREES="$K_ROOT/k5-worktrees"
K5_BASE="$K_ROOT/k5-base"
mkdir -p "$K5_WORKTREES" "$K5_BASE"
make_repo "$K5_REPO"
mkdir -p "$K5_BASE/target.gen.1"
touch "$K5_BASE/target.gen.1.lock"
ln -sfn "$K5_BASE/target.gen.1" "$K5_BASE/target"

# _lane-1 live (cwd holder), _lane-2 free — same discrimination shape as P-basic,
# but under --disk-pressure and with a FREE flock on both.
for _k5_name in _lane-1 _lane-2; do
    git -C "$K5_REPO" worktree add -q "$K5_WORKTREES/$_k5_name"
    mkdir -p "$K5_WORKTREES/$_k5_name/target"
    touch "$K5_WORKTREES/$_k5_name/target/DIVERGENT_MARKER"
done

K5_READY="$K_ROOT/k5-holder.ready"
( cd "$K5_WORKTREES/_lane-1/target" && touch "$K5_READY" && exec sleep 300 ) &
K5_HELPER_PID=$!
_BGPIDS+=("$K5_HELPER_PID")
_wait_for_reader_lock "$K5_READY" 30

K5_SEED_LOG="$K_ROOT/k5-seed-calls.log"
K5_SEED_STUB="$K_ROOT/k5-seed-stub.sh"
_seed_stub_body > "$K5_SEED_STUB"
chmod +x "$K5_SEED_STUB"
export SEED_LOG="$K5_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$K5_WORKTREES" \
    --base-target "$K5_BASE/target" \
    --seed-script "$K5_SEED_STUB" \
    --disk-pressure

assert "K5: exit 0" test "$RC" -eq 0
assert "K5: live-referenced _lane-1 target/ SURVIVES --disk-pressure rm -rf (gate covers the destructive branch)" \
    test -d "$K5_WORKTREES/_lane-1/target"
assert "K5: live-referenced _lane-1 divergent marker intact (nothing was deleted)" \
    test -f "$K5_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "K5: free _lane-2 target/ DELETED outright (disk-pressure still works; not a blanket freeze)" \
    bash -c '[ ! -d "$1" ]' _ "$K5_WORKTREES/_lane-2/target"
assert "K5: stderr names the process-reference preserve reason for _lane-1" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-1: live consumer (process reference)"' _ "$ERR_OUT"
assert "K5: summary attributes the preserve to the live-reference gate (reset=1, preserved_live_ref=1)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1 removed=0 preserved=1 preserved_live_ref=1"' _ "$OUT"

kill "$K5_HELPER_PID" 2>/dev/null || true
wait "$K5_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ──────────────────────────────────────────────────────────────────────────────
# Block M — ephemeral verify/sweep worktrees protected by default
# _mainsweep-*/_mainprobe-* are dark-factory's ephemeral background-verify and
# merge/main-probe worktrees, minted directly under the warm-lane mount (the
# same directory gc.sh scans). DF's gc reclaim call site
# (git_ops.py _run_warm_lane_gc_reclaim) passes ONLY --mount — no
# --protect-glob — so gc's DEFAULT protect-glob is the only thing standing
# between a live sweep/probe worktree and Pass-2 `git worktree remove --force`.
# Their sole other guard, a per-worktree advisory flock, demonstrably failed
# twice in production (task 5221 analysis), removing a live worktree out from
# under a running background integrity sweep. This block is DF-faithful:
# --mount only, no --protect-glob, exercising the default exactly as invoked
# in production.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block M: ephemeral verify/sweep worktrees protected by default ---"

M_ROOT="$(mktemp -d /tmp/test-gc-m-XXXXXX)"
_TMPDIRS+=("$M_ROOT")

M_REPO="$M_ROOT/repo"
M_WORKTREES="$M_ROOT/worktrees"
M_BASE="$M_ROOT/base"
mkdir -p "$M_WORKTREES" "$M_BASE"

make_repo "$M_REPO"

mkdir -p "$M_BASE/target.gen.1"
touch "$M_BASE/target.gen.1.lock"
ln -sfn "$M_BASE/target.gen.1" "$M_BASE/target"

# Fixture 1: _mainsweep-abcd1234 — DF's background main-tip integrity-sweep
# worktree (clean, landed — as it is immediately after DF mints it at main_sha).
git -C "$M_REPO" worktree add -q "$M_WORKTREES/_mainsweep-abcd1234"
touch "$M_WORKTREES/_mainsweep-abcd1234/PROTECTED_MARKER"
mkdir -p "$M_WORKTREES/_mainsweep-abcd1234/target"
touch "$M_WORKTREES/_mainsweep-abcd1234/target/DIVERGENT_MARKER"

# Fixture 2: _mainprobe-abcd1234 — DF's merge/main-probe worktree, same shape.
git -C "$M_REPO" worktree add -q "$M_WORKTREES/_mainprobe-abcd1234"
touch "$M_WORKTREES/_mainprobe-abcd1234/PROTECTED_MARKER"
mkdir -p "$M_WORKTREES/_mainprobe-abcd1234/target"
touch "$M_WORKTREES/_mainprobe-abcd1234/target/DIVERGENT_MARKER"

# Control: task-9999 — genuine cold orphan (non-underscore, clean, landed),
# proving reclaim capability stays intact and the fix does not over-protect.
git -C "$M_REPO" worktree add -q "$M_WORKTREES/task-9999"

M_SEED_LOG="$M_ROOT/seed_calls.log"
M_SEED_STUB="$M_ROOT/seed_stub.sh"
cat > "$M_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$M_SEED_STUB"
export SEED_LOG="$M_SEED_LOG"

# DF-faithful invocation: --mount only, NO --protect-glob (exercises the DEFAULT).
run_helper reclaim \
    --mount "$M_WORKTREES" \
    --seed-script "$M_SEED_STUB"

assert "M1: exit 0" test "$RC" -eq 0
assert "M2: _mainsweep-abcd1234 dir preserved" \
    test -d "$M_WORKTREES/_mainsweep-abcd1234"
assert "M3: _mainsweep-abcd1234 PROTECTED_MARKER intact" \
    test -f "$M_WORKTREES/_mainsweep-abcd1234/PROTECTED_MARKER"
assert "M4: _mainprobe-abcd1234 dir preserved" \
    test -d "$M_WORKTREES/_mainprobe-abcd1234"
assert "M5: _mainprobe-abcd1234 PROTECTED_MARKER intact" \
    test -f "$M_WORKTREES/_mainprobe-abcd1234/PROTECTED_MARKER"
assert "M6: _mainsweep-abcd1234 seed-script NOT invoked (skipped, not reset)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_mainsweep-abcd1234" "$1"' _ "$M_SEED_LOG"
assert "M7: _mainprobe-abcd1234 seed-script NOT invoked (skipped, not reset)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_mainprobe-abcd1234" "$1"' _ "$M_SEED_LOG"
assert "M8: control orphan task-9999 removed (dir gone; reclaim capability intact)" \
    bash -c '[ ! -d "$1" ]' _ "$M_WORKTREES/task-9999"
assert "M9: control orphan task-9999 absent from git worktree list" \
    bash -c '! git -C "$1" worktree list | grep -q "task-9999"' _ "$M_REPO"
assert "M10: summary shows preserved=2 (the two managed worktrees)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=2"' _ "$OUT"
assert "M11: summary shows removed=1 (control orphan)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "removed=1"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block N — full managed-worktree protect set + explicit override intact
# Extends Block M's coverage to the remaining orchestrator-managed non-pool
# worktree kinds (mirroring dark-factory's PROTECTED_PREFIXES inventory):
# _solo-*, _substrate-gate-*, _offline-deep (persistent, exact name), _iact-*.
# Sub-case N-default proves the DEFAULT protect-glob covers the full set;
# sub-case N-override proves an explicit --protect-glob still re-narrows the
# protect set exactly as before — the fix changes only the DEFAULT, never the
# override mechanism.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block N: full managed-worktree protect set + explicit override intact ---"

# ── N-default: the full managed set survives the DEFAULT protect-glob ─────────
N_ROOT="$(mktemp -d /tmp/test-gc-n-XXXXXX)"
_TMPDIRS+=("$N_ROOT")

N_REPO="$N_ROOT/repo"
N_WORKTREES="$N_ROOT/worktrees"
N_BASE="$N_ROOT/base"
mkdir -p "$N_WORKTREES" "$N_BASE"

make_repo "$N_REPO"

mkdir -p "$N_BASE/target.gen.1"
touch "$N_BASE/target.gen.1.lock"
ln -sfn "$N_BASE/target.gen.1" "$N_BASE/target"

# Four remaining orchestrator-managed non-pool worktree kinds, each clean/landed.
for _n_name in _solo-1234 _substrate-gate-1234 _offline-deep _iact-demo; do
    git -C "$N_REPO" worktree add -q "$N_WORKTREES/$_n_name"
    touch "$N_WORKTREES/$_n_name/PROTECTED_MARKER"
done

# Control: task-8888 — genuine cold orphan, must still be removed.
git -C "$N_REPO" worktree add -q "$N_WORKTREES/task-8888"

N_SEED_LOG="$N_ROOT/seed_calls.log"
N_SEED_STUB="$N_ROOT/seed_stub.sh"
cat > "$N_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$N_SEED_STUB"
export SEED_LOG="$N_SEED_LOG"

run_helper reclaim \
    --mount "$N_WORKTREES" \
    --seed-script "$N_SEED_STUB"

assert "N-default: exit 0" test "$RC" -eq 0
for _n_name in _solo-1234 _substrate-gate-1234 _offline-deep _iact-demo; do
    assert "N-default: $_n_name dir preserved" \
        test -d "$N_WORKTREES/$_n_name"
    assert "N-default: $_n_name PROTECTED_MARKER intact" \
        test -f "$N_WORKTREES/$_n_name/PROTECTED_MARKER"
    assert "N-default: $_n_name seed-script NOT invoked (skipped, not reset)" \
        bash -c '[ ! -f "$1" ] || ! grep -q "$2" "$1"' _ "$N_SEED_LOG" "$_n_name"
done
assert "N-default: control orphan task-8888 removed (dir gone; reclaim capability intact)" \
    bash -c '[ ! -d "$1" ]' _ "$N_WORKTREES/task-8888"
assert "N-default: control orphan task-8888 absent from git worktree list" \
    bash -c '! git -C "$1" worktree list | grep -q "task-8888"' _ "$N_REPO"
assert "N-default: summary shows preserved=4 (the four managed worktrees)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=4"' _ "$OUT"
assert "N-default: summary shows removed=1 (control orphan)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "removed=1"' _ "$OUT"

# ── N-override: explicit --protect-glob re-narrows the protect set ────────────
# On a separate mount, an explicit --protect-glob "_merge-*" re-narrows the
# set exactly as before the fix, so _mainsweep-abcd1234 (no longer covered by
# the narrowed glob) IS removed by Pass 2 — proving the fix changes only the
# DEFAULT, never the override mechanism.
NOV_ROOT="$(mktemp -d /tmp/test-gc-nov-XXXXXX)"
_TMPDIRS+=("$NOV_ROOT")

NOV_REPO="$NOV_ROOT/repo"
NOV_WORKTREES="$NOV_ROOT/worktrees"
NOV_BASE="$NOV_ROOT/base"
mkdir -p "$NOV_WORKTREES" "$NOV_BASE"

make_repo "$NOV_REPO"

mkdir -p "$NOV_BASE/target.gen.1"
touch "$NOV_BASE/target.gen.1.lock"
ln -sfn "$NOV_BASE/target.gen.1" "$NOV_BASE/target"

git -C "$NOV_REPO" worktree add -q "$NOV_WORKTREES/_mainsweep-abcd1234"
touch "$NOV_WORKTREES/_mainsweep-abcd1234/PROTECTED_MARKER"

NOV_SEED_LOG="$NOV_ROOT/seed_calls.log"
NOV_SEED_STUB="$NOV_ROOT/seed_stub.sh"
cat > "$NOV_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
exit 0
STUB_EOF
chmod +x "$NOV_SEED_STUB"
export SEED_LOG="$NOV_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$NOV_WORKTREES" \
    --base-target "$NOV_BASE/target" \
    --protect-glob "_merge-*" \
    --seed-script "$NOV_SEED_STUB"

assert "N-override: exit 0" test "$RC" -eq 0
assert "N-override: explicit --protect-glob _merge-* re-narrows — _mainsweep-abcd1234 IS removed" \
    bash -c '[ ! -d "$1" ]' _ "$NOV_WORKTREES/_mainsweep-abcd1234"
assert "N-override: _mainsweep-abcd1234 absent from git worktree list" \
    bash -c '! git -C "$1" worktree list | grep -q "_mainsweep-abcd1234"' _ "$NOV_REPO"
assert "N-override: summary shows removed=1" \
    bash -c 'printf "%s\n" "$1" | grep -qE "removed=1"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block O — --extra-protect-glob ADDS to (never replaces) the default protect set
# The live-consumer lane guard (task 5378) needs the sweep to protect specific
# lane basenames it discovers hold a live build, WITHOUT restating gc.sh's full
# default protect list (that duplication would be a G7 lockstep-duplication hit).
# --extra-protect-glob / REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB is that additive
# primitive: entries it matches are skipped exactly like --protect-glob entries
# (Pass 1 reset AND Pass 2 remove) and counted as preserved, while the DEFAULT
# protect set stays active alongside it (unlike --protect-glob, which REPLACES
# the default). Fixture mirrors Block B/M: base/target.gen.1 (+ .lock + symlink),
# an argv-logging thinning seed stub, and three clean+landed (HEAD==main) lanes
# each with a target/DIVERGENT_MARKER:
#   _lane-1      — plain reclaimable pool lane      → MUST still be reset
#   _lane-9      — reclaimable, passed via --extra-protect-glob → preserved
#   _mainsweep-x — default-protected (_mainsweep-*) → preserved (default set on)
# Sub-case O-env drives the SAME protection through the env var instead of the
# flag. RED today: --extra-protect-glob is an unknown flag (exit 2) and the env
# var is ignored (so _lane-9 is reset instead of preserved).
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block O: --extra-protect-glob adds to the default protect set ---"

# ── O-flag: additive protection driven by the --extra-protect-glob FLAG ───────
O_ROOT="$(mktemp -d /tmp/test-gc-o-XXXXXX)"
_TMPDIRS+=("$O_ROOT")

O_REPO="$O_ROOT/repo"
O_WORKTREES="$O_ROOT/worktrees"
O_BASE="$O_ROOT/base"
mkdir -p "$O_WORKTREES" "$O_BASE"

make_repo "$O_REPO"

mkdir -p "$O_BASE/target.gen.1"
touch "$O_BASE/target.gen.1.lock"
ln -sfn "$O_BASE/target.gen.1" "$O_BASE/target"

# Three clean+landed (HEAD==main) lanes, each with a divergent target marker.
for _o_name in _lane-1 _lane-9 _mainsweep-x; do
    git -C "$O_REPO" worktree add -q "$O_WORKTREES/$_o_name"
    mkdir -p "$O_WORKTREES/$_o_name/target"
    touch "$O_WORKTREES/$_o_name/target/DIVERGENT_MARKER"
done

O_SEED_LOG="$O_ROOT/seed_calls.log"
O_SEED_STUB="$O_ROOT/seed_stub.sh"
cat > "$O_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$O_SEED_STUB"
export SEED_LOG="$O_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$O_WORKTREES" \
    --base-target "$O_BASE/target" \
    --seed-script "$O_SEED_STUB" \
    --main-ref main \
    --extra-protect-glob _lane-9

assert "O1: exit 0" test "$RC" -eq 0
# _lane-1: plain reclaimable → reset (seed invoked, divergent marker gone).
assert "O2: plain lane _lane-1 reset (seed-script invoked)" \
    bash -c 'test -f "$1" && grep -q "_lane-1" "$1"' _ "$O_SEED_LOG"
assert "O3: plain lane _lane-1 divergent marker removed (reset)" \
    bash -c '[ ! -f "$1" ]' _ "$O_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
# _lane-9: extra-protected → preserved (seed NOT invoked, marker intact).
assert "O4: extra-protected lane _lane-9 seed-script NOT invoked (preserved)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-9" "$1"' _ "$O_SEED_LOG"
assert "O5: extra-protected lane _lane-9 divergent marker intact" \
    test -f "$O_WORKTREES/_lane-9/target/DIVERGENT_MARKER"
# _mainsweep-x: default-protected → preserved (default set still active alongside extra).
assert "O6: default-protected lane _mainsweep-x seed-script NOT invoked (preserved)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_mainsweep-x" "$1"' _ "$O_SEED_LOG"
assert "O7: default-protected lane _mainsweep-x divergent marker intact (default set still on)" \
    test -f "$O_WORKTREES/_mainsweep-x/target/DIVERGENT_MARKER"
# Summary: reset=1 (_lane-1), preserved=2 (extra _lane-9 + default _mainsweep-x).
assert "O8: summary shows reset=1 (only _lane-1)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1"' _ "$OUT"
assert "O9: summary preserved count includes BOTH protected entries (preserved=2)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=2"' _ "$OUT"

# ── O-env: SAME additive protection via REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB ──
# Parallel sub-case: NO --extra-protect-glob flag; the env var carries _lane-9,
# proving the env knob is an equal driver of the additive protect set (mirrors
# the --disk-pressure / REIFY_WARM_LANE_GC_DISK_PRESSURE flag/env convention).
OE_ROOT="$(mktemp -d /tmp/test-gc-oe-XXXXXX)"
_TMPDIRS+=("$OE_ROOT")

OE_REPO="$OE_ROOT/repo"
OE_WORKTREES="$OE_ROOT/worktrees"
OE_BASE="$OE_ROOT/base"
mkdir -p "$OE_WORKTREES" "$OE_BASE"

make_repo "$OE_REPO"

mkdir -p "$OE_BASE/target.gen.1"
touch "$OE_BASE/target.gen.1.lock"
ln -sfn "$OE_BASE/target.gen.1" "$OE_BASE/target"

for _oe_name in _lane-1 _lane-9 _mainsweep-x; do
    git -C "$OE_REPO" worktree add -q "$OE_WORKTREES/$_oe_name"
    mkdir -p "$OE_WORKTREES/$_oe_name/target"
    touch "$OE_WORKTREES/$_oe_name/target/DIVERGENT_MARKER"
done

OE_SEED_LOG="$OE_ROOT/seed_calls.log"
OE_SEED_STUB="$OE_ROOT/seed_stub.sh"
cat > "$OE_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
exit 0
STUB_EOF
chmod +x "$OE_SEED_STUB"
export SEED_LOG="$OE_SEED_LOG"

REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB="_lane-9" run_helper reclaim \
    --worktrees-dir "$OE_WORKTREES" \
    --base-target "$OE_BASE/target" \
    --seed-script "$OE_SEED_STUB" \
    --main-ref main

assert "O-env1: exit 0" test "$RC" -eq 0
# Env-var-protected lane preserved (seed NOT invoked, marker intact).
assert "O-env2: env-protected lane _lane-9 seed-script NOT invoked (preserved)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-9" "$1"' _ "$OE_SEED_LOG"
assert "O-env3: env-protected lane _lane-9 divergent marker intact" \
    test -f "$OE_WORKTREES/_lane-9/target/DIVERGENT_MARKER"
# Plain lane still reset (env protection is additive, not blanket).
assert "O-env4: plain lane _lane-1 still reset (marker removed)" \
    bash -c '[ ! -f "$1" ]' _ "$OE_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
# Summary: preserved=2 (env-driven _lane-9 + default _mainsweep-x).
assert "O-env5: summary preserved=2 (env _lane-9 + default _mainsweep-x)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=2"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block P — the PRIMITIVE preserves a live-consumer lane, with NO
# --extra-protect-glob (task 5572)
# ──────────────────────────────────────────────────────────────────────────────
# ROOT CAUSE: two entry points reach one operation. warm-lane-gc-sweep.sh (the δ
# systemd backstop) computed a live-lane CSV and handed it to gc.sh via
# --extra-protect-glob; dark-factory's ε path (git_ops.py
# _run_warm_lane_gc_reclaim) invokes `warm-lane-gc.sh reclaim --mount <base>`
# with NO --extra-protect-glob, ever. A guard living in the WRAPPER therefore
# covered only one of the two callers, and ε — the hot, per-acquire path — was
# the uncovered one (the esc-5375-1 gap, reopened).
#
# FIX: the liveness check moves INTO this primitive, per-lane, immediately
# before the reset and under the lane flock this loop already holds. Both cases
# below therefore invoke reclaim DELIBERATELY WITHOUT --extra-protect-glob and
# with NO flock holder (every <lane>.lock is FREE), so neither the wrapper CSV
# nor the flock gate can account for any preservation observed.
echo ""
echo "--- Block P: per-lane live-consumer check inside the primitive (task 5572) ---"

# ── P-basic: the dark-factory ε-path gap ──────────────────────────────────────
# _lane-1 has a live helper whose cwd is _lane-1/target (the exact build-cwd/fd
# shape); _lane-2 is free and unreferenced. The guard must preserve the former
# and still reset the latter — discrimination, not blanket over-preserve.
P_ROOT="$(mktemp -d /tmp/test-gc-p-XXXXXX)"
_TMPDIRS+=("$P_ROOT")

P_REPO="$P_ROOT/repo"
P_WORKTREES="$P_ROOT/worktrees"
P_BASE="$P_ROOT/base"
mkdir -p "$P_WORKTREES" "$P_BASE"

make_repo "$P_REPO"

mkdir -p "$P_BASE/target.gen.1"
touch "$P_BASE/target.gen.1.lock"
ln -sfn "$P_BASE/target.gen.1" "$P_BASE/target"

# Two clean+landed (HEAD==main) reclaimable pool lanes, each with a divergent
# marker and a FREE flock (no holder started anywhere in this block).
for _p_name in _lane-1 _lane-2; do
    git -C "$P_REPO" worktree add -q "$P_WORKTREES/$_p_name"
    mkdir -p "$P_WORKTREES/$_p_name/target"
    touch "$P_WORKTREES/$_p_name/target/DIVERGENT_MARKER"
done

# Live helper whose CWD is _lane-1/target — a DESCENDANT of the lane dir. It
# touches READY only AFTER cd'ing in, so _wait_for_reader_lock proves the cwd is
# established before reclaim runs (causal ordering, technique R — no wall-clock
# sleep). exec sleep so the tracked PID is the one holding the cwd.
P_READY="$P_ROOT/helper.ready"
( cd "$P_WORKTREES/_lane-1/target" && touch "$P_READY" && exec sleep 300 ) &
P_HELPER_PID=$!
_BGPIDS+=("$P_HELPER_PID")
_wait_for_reader_lock "$P_READY" 30

P_SEED_LOG="$P_ROOT/seed_calls.log"
P_SEED_STUB="$P_ROOT/seed_stub.sh"
_seed_stub_body > "$P_SEED_STUB"
chmod +x "$P_SEED_STUB"
export SEED_LOG="$P_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$P_WORKTREES" \
    --base-target "$P_BASE/target" \
    --seed-script "$P_SEED_STUB" \
    --main-ref main

assert "P1: exit 0" test "$RC" -eq 0
assert "P2: live-referenced _lane-1 divergent marker INTACT (no --extra-protect-glob, FREE flock)" \
    test -f "$P_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "P3: _lane-1 seed-script NOT invoked (preserved, not reset)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-1" "$1"' _ "$P_SEED_LOG"
assert "P4: free unreferenced _lane-2 divergent marker REMOVED (no blanket over-preserve)" \
    bash -c '[ ! -f "$1" ]' _ "$P_WORKTREES/_lane-2/target/DIVERGENT_MARKER"
assert "P5: _lane-2 seed-script invoked (reclaimed via α)" \
    bash -c 'test -f "$1" && grep -q "_lane-2" "$1"' _ "$P_SEED_LOG"
# The two preserve reasons must stay DISTINGUISHABLE in dark-factory's logs: a
# process-reference preserve must not be reported as a flock preserve (and vice
# versa), or an operator reading ε logs cannot tell which gate fired.
assert "P6: stderr names the process-reference preserve reason for _lane-1" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-1: live consumer (process reference)"' _ "$ERR_OUT"
assert "P7: stderr does NOT misattribute _lane-1 to the flock gate" \
    bash -c '! printf "%s\n" "$1" | grep -qF "preserving _lane-1: live consumer (flock held)"' _ "$ERR_OUT"
assert "P8: summary counts the live lane as preserved (preserved=1)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=1"' _ "$OUT"
assert "P9: summary counts the free lane as reset (reset=1)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1"' _ "$OUT"

# Preserve is TEMPORARY — the over-preserve bias is bounded. Once the reference
# clears, the NEXT reclaim resets the lane (mirrors the sweep suite's U6/U7 and
# the trash reaper's second-sweep behaviour).
kill "$P_HELPER_PID" 2>/dev/null || true
wait "$P_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

run_helper reclaim \
    --worktrees-dir "$P_WORKTREES" \
    --base-target "$P_BASE/target" \
    --seed-script "$P_SEED_STUB" \
    --main-ref main

assert "P10: exit 0 (second reclaim)" test "$RC" -eq 0
assert "P11: _lane-1 reset once its live cwd reference is gone (preserve is temporary)" \
    bash -c '[ ! -f "$1" ]' _ "$P_WORKTREES/_lane-1/target/DIVERGENT_MARKER"

# ── P-fd: a BARE OPEN FD, with the cwd OUTSIDE the mount ──────────────────────
# The scanner advertises THREE reference kinds — cwd, open fd, mmap — and until
# this sub-case existed only two were driven anywhere: P-basic/Q use a cwd
# holder and R uses an mmap (negatively). The `-path '*/fd/*'` leg was therefore
# unpinned at every layer, and it is the leg most likely to matter in
# production: a rustc/linker holding an output fd under <lane>/target while its
# own cwd sits elsewhere. A regression that dropped it would leave every other
# assertion in this file green (task 5572 review).
#
# The holder's cwd is deliberately $PF_ROOT — the fixture root, OUTSIDE
# $PF_WORKTREES — so the cwd leg CANNOT account for the preserve, and `sleep`'s
# own mappings live in /usr, so the mmap leg cannot either. The open fd is the
# sole reference. `exec 9>` (not a redirect on `exec sleep`) is what makes the
# descriptor survive the exec into the tracked PID.
PF_ROOT="$(mktemp -d /tmp/test-gc-pfd-XXXXXX)"
_TMPDIRS+=("$PF_ROOT")

PF_REPO="$PF_ROOT/repo"
PF_WORKTREES="$PF_ROOT/worktrees"
PF_BASE="$PF_ROOT/base"
mkdir -p "$PF_WORKTREES" "$PF_BASE"

make_repo "$PF_REPO"

mkdir -p "$PF_BASE/target.gen.1"
touch "$PF_BASE/target.gen.1.lock"
ln -sfn "$PF_BASE/target.gen.1" "$PF_BASE/target"

for _pf_name in _lane-1 _lane-2; do
    git -C "$PF_REPO" worktree add -q "$PF_WORKTREES/$_pf_name"
    mkdir -p "$PF_WORKTREES/$_pf_name/target"
    touch "$PF_WORKTREES/$_pf_name/target/DIVERGENT_MARKER"
done

PF_READY="$PF_ROOT/fd-holder.ready"
( cd "$PF_ROOT" && exec 9>"$PF_WORKTREES/_lane-1/target/held.tmp" \
    && touch "$PF_READY" && exec sleep 300 ) &
PF_HELPER_PID=$!
_BGPIDS+=("$PF_HELPER_PID")
_wait_for_reader_lock "$PF_READY" 30

PF_SEED_LOG="$PF_ROOT/seed_calls.log"
PF_SEED_STUB="$PF_ROOT/seed_stub.sh"
_seed_stub_body > "$PF_SEED_STUB"
chmod +x "$PF_SEED_STUB"
export SEED_LOG="$PF_SEED_LOG"

# Non-vacuity, captured BEFORE the run: if the holder's cwd were (or were under)
# the lane, this sub-case would silently degenerate into a second copy of
# P-basic and prove nothing about the fd leg.
PF_HELPER_CWD="$(readlink "/proc/$PF_HELPER_PID/cwd" 2>/dev/null || true)"
PF_HELPER_FD9="$(readlink "/proc/$PF_HELPER_PID/fd/9" 2>/dev/null || true)"

run_helper reclaim \
    --worktrees-dir "$PF_WORKTREES" \
    --base-target "$PF_BASE/target" \
    --seed-script "$PF_SEED_STUB" \
    --main-ref main

assert "P-fd1: fixture — holder cwd is OUTSIDE the worktrees dir (cwd leg cannot explain the preserve)" \
    bash -c 'case "$1" in "$2"/*) exit 1 ;; *) exit 0 ;; esac' _ "$PF_HELPER_CWD" "$(readlink -f "$PF_WORKTREES")"
assert "P-fd2: fixture — fd 9 survived the exec and points under _lane-1/target" \
    bash -c 'case "$1" in "$2"/target/*) exit 0 ;; *) exit 1 ;; esac' _ "$PF_HELPER_FD9" "$(readlink -f "$PF_WORKTREES/_lane-1")"
assert "P-fd3: exit 0" test "$RC" -eq 0
assert "P-fd4: fd-referenced _lane-1 divergent marker INTACT (open-fd leg alone preserves)" \
    test -f "$PF_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "P-fd5: _lane-1 seed-script NOT invoked (preserved, not reset)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-1" "$1"' _ "$PF_SEED_LOG"
assert "P-fd6: free unreferenced _lane-2 divergent marker REMOVED (discrimination, not blanket)" \
    bash -c '[ ! -f "$1" ]' _ "$PF_WORKTREES/_lane-2/target/DIVERGENT_MARKER"
assert "P-fd7: stderr names the process-reference preserve reason for _lane-1" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-1: live consumer (process reference)"' _ "$ERR_OUT"
# The live-reference share of preserved= is reported separately so an
# indefinitely-shielded pool is visible in DF's logs (task 5572 review).
assert "P-fd8: summary attributes the preserve to the live-reference gate (preserved_live_ref=1)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=1 preserved_live_ref=1"' _ "$OUT"

kill "$PF_HELPER_PID" 2>/dev/null || true
wait "$PF_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ── P-toctou: the up-front-snapshot TOCTOU (defect 5) ─────────────────────────
# PLACEMENT test, not merely a liveness test. NOTHING is live when the pass
# starts; _lane-2 goes live MID-PASS. So this case stays RED against ANY
# implementation that computes liveness once up front (the wrapper CSV shape it
# replaces) — it only goes GREEN when the check runs per-lane, immediately
# before that lane's own reset.
#
# Mechanism: gc.sh invokes --seed-script SYNCHRONOUSLY inside the Pass-1 loop,
# and bash `*/` glob expansion is LC_COLLATE-sorted, so _lane-1 is always
# visited before _lane-2. _lane-1's seed stub therefore spawns the _lane-2 cwd
# holder and blocks until it is established.
PT_ROOT="$(mktemp -d /tmp/test-gc-pt-XXXXXX)"
_TMPDIRS+=("$PT_ROOT")

PT_REPO="$PT_ROOT/repo"
PT_WORKTREES="$PT_ROOT/worktrees"
PT_BASE="$PT_ROOT/base"
mkdir -p "$PT_WORKTREES" "$PT_BASE"

make_repo "$PT_REPO"

mkdir -p "$PT_BASE/target.gen.1"
touch "$PT_BASE/target.gen.1.lock"
ln -sfn "$PT_BASE/target.gen.1" "$PT_BASE/target"

for _pt_name in _lane-1 _lane-2; do
    git -C "$PT_REPO" worktree add -q "$PT_WORKTREES/$_pt_name"
    mkdir -p "$PT_WORKTREES/$_pt_name/target"
    touch "$PT_WORKTREES/$_pt_name/target/DIVERGENT_MARKER"
done

PT_SEED_LOG="$PT_ROOT/seed_calls.log"
PT_SEED_STUB="$PT_ROOT/seed_stub.sh"
PT_READY2="$PT_ROOT/lane2-holder.ready"
PT_PIDFILE="$PT_ROOT/lane2-holder.pid"

# The trigger stub: behaves like _seed_stub_body for every lane, and
# ADDITIONALLY, when invoked for _lane-1, makes _lane-2 live before returning.
#   - stdio MUST go to /dev/null: gc.sh runs the seed under
#     `( ... ) 2>&1 | while IFS= read -r line; ...`, and an inherited pipe write
#     end would keep that `while read` open for the holder's whole 300s life,
#     hanging the pass.
#   - the PID is written to a file because the holder reparents to init when the
#     stub exits, so the TEST cannot learn it any other way.
#   - the poll for READY2 is causal ordering (technique R), not a wall-clock
#     sleep: the stub returns only once the cwd provably exists.
cat > "$PT_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
if [ "${LANE_DIR##*/}" = "_lane-1" ]; then
    ( cd "$TOCTOU_LANE2/target" && touch "$TOCTOU_READY2" && exec sleep 300 ) \
        </dev/null >/dev/null 2>&1 &
    echo "$!" > "$TOCTOU_PIDFILE"
    _tick=0
    while [ "$_tick" -lt 600 ]; do
        [ -f "$TOCTOU_READY2" ] && break
        sleep 0.05
        _tick=$(( _tick + 1 ))
    done
fi
exit 0
STUB_EOF
chmod +x "$PT_SEED_STUB"

export SEED_LOG="$PT_SEED_LOG"
export TOCTOU_LANE2="$PT_WORKTREES/_lane-2"
export TOCTOU_READY2="$PT_READY2"
export TOCTOU_PIDFILE="$PT_PIDFILE"

run_helper reclaim \
    --worktrees-dir "$PT_WORKTREES" \
    --base-target "$PT_BASE/target" \
    --seed-script "$PT_SEED_STUB" \
    --main-ref main

# Adopt the reparented holder so the suite's cleanup() reaps it.
if [ -s "$PT_PIDFILE" ]; then
    PT_HOLDER_PID="$(cat "$PT_PIDFILE")"
    _BGPIDS+=("$PT_HOLDER_PID")
else
    PT_HOLDER_PID=""
fi

unset TOCTOU_LANE2 TOCTOU_READY2 TOCTOU_PIDFILE

assert "P-toctou1: fixture — the _lane-1 seed stub established the mid-pass _lane-2 cwd holder" \
    bash -c 'test -f "$1" && test -s "$2"' _ "$PT_READY2" "$PT_PIDFILE"
assert "P-toctou2: exit 0" test "$RC" -eq 0
assert "P-toctou3: trigger lane _lane-1 WAS reset (behaves normally)" \
    bash -c '[ ! -f "$1" ]' _ "$PT_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "P-toctou4: _lane-2 divergent marker INTACT — went live MID-PASS and was still preserved" \
    test -f "$PT_WORKTREES/_lane-2/target/DIVERGENT_MARKER"
assert "P-toctou5: _lane-2 seed-script NOT invoked (preserved, not reset)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-2" "$1"' _ "$PT_SEED_LOG"
assert "P-toctou6: stderr names the process-reference preserve reason for _lane-2" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-2: live consumer (process reference)"' _ "$ERR_OUT"

kill "$PT_HOLDER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ──────────────────────────────────────────────────────────────────────────────
# Block Q — Pass 2 destructive orphan removal ALSO honours a live process
# reference (task 5572)
# ──────────────────────────────────────────────────────────────────────────────
# This is not a bonus case; it closes a coverage hole the wrapper-CSV deletion
# would otherwise open. warm-lane-gc-sweep.sh's _live_consumer_protect_csv (now
# deleted by task 5572 — named here as history, not as a live symbol)
# enumerated EVERY immediate subdir under the mount with a bare */ glob — it did
# NOT filter by lane-glob — and --extra-protect-glob skips a match ENTIRELY, in
# Pass 2 as well as Pass 1. So the wrapper's CSV was, incidentally, also
# protecting a live ORPHAN worktree from `git worktree remove --force`. Removing
# the CSV without adding this gate would silently regress that protection on the
# single most destructive path in the script — strictly worse than the bug being
# fixed. Block Q makes the regression visible BEFORE the deletion lands.
echo ""
echo "--- Block Q: Pass-2 orphan removal honours a live process reference (task 5572) ---"

Q_ROOT="$(mktemp -d /tmp/test-gc-q-XXXXXX)"
_TMPDIRS+=("$Q_ROOT")

Q_REPO="$Q_ROOT/repo"
Q_WORKTREES="$Q_ROOT/worktrees"
Q_BASE="$Q_ROOT/base"
mkdir -p "$Q_WORKTREES" "$Q_BASE"

make_repo "$Q_REPO"

mkdir -p "$Q_BASE/target.gen.1"
touch "$Q_BASE/target.gen.1.lock"
ln -sfn "$Q_BASE/target.gen.1" "$Q_BASE/target"

# Two ORPHAN worktrees: names matching neither --lane-glob nor --protect-glob,
# both clean and landed (HEAD == main) so _is_reclaimable returns 0 and Pass 2
# would remove them. Both locks are FREE — no flock holder anywhere in this
# block, so the flock gate cannot account for any preservation. target/ is
# untracked, and _is_reclaimable runs `git status --untracked-files=no`, so it
# does not make either orphan dirty.
for _q_name in task-live task-free; do
    git -C "$Q_REPO" worktree add -q "$Q_WORKTREES/$_q_name"
    mkdir -p "$Q_WORKTREES/$_q_name/target"
done

# Live helper with a cwd under task-live only. task-free is the discriminator:
# without it, a guard that simply refused to remove any orphan would pass.
Q_READY="$Q_ROOT/helper.ready"
( cd "$Q_WORKTREES/task-live/target" && touch "$Q_READY" && exec sleep 300 ) &
Q_HELPER_PID=$!
_BGPIDS+=("$Q_HELPER_PID")
_wait_for_reader_lock "$Q_READY" 30

Q_SEED_LOG="$Q_ROOT/seed_calls.log"
Q_SEED_STUB="$Q_ROOT/seed_stub.sh"
_seed_stub_body > "$Q_SEED_STUB"
chmod +x "$Q_SEED_STUB"
export SEED_LOG="$Q_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$Q_WORKTREES" \
    --base-target "$Q_BASE/target" \
    --seed-script "$Q_SEED_STUB" \
    --main-ref main

assert "Q1: exit 0" test "$RC" -eq 0
assert "Q2: live-referenced orphan task-live NOT removed (dir still present)" \
    test -d "$Q_WORKTREES/task-live"
assert "Q3: live-referenced orphan task-live still registered as a git worktree" \
    bash -c 'git -C "$1" worktree list | grep -q "task-live"' _ "$Q_REPO"
# The Pass-2 success path rm -f's the orphan's sibling lock file, so the lock's
# SURVIVAL independently confirms removal never ran — not merely that a later
# step recreated the directory.
assert "Q4: task-live sibling lock file survives (Pass-2 removal never ran)" \
    test -f "$Q_WORKTREES/task-live.lock"
assert "Q5: unreferenced orphan task-free WAS removed (no blanket over-preserve)" \
    bash -c '[ ! -d "$1" ]' _ "$Q_WORKTREES/task-free"
assert "Q6: task-free absent from git worktree list" \
    bash -c '! git -C "$1" worktree list | grep -q "task-free"' _ "$Q_REPO"
assert "Q7: stderr names the process-reference preserve reason for task-live" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving task-live: live consumer (process reference)"' _ "$ERR_OUT"
assert "Q8: summary shows removed=1 (only task-free)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "removed=1"' _ "$OUT"
assert "Q9: summary counts the live orphan as preserved (preserved=1)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=1"' _ "$OUT"

kill "$Q_HELPER_PID" 2>/dev/null || true
wait "$Q_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ──────────────────────────────────────────────────────────────────────────────
# Block R — mmap substring-boundary regression, at the layer that now owns it
# (esc-5378 review; MIGRATED here by task 5572)
# ──────────────────────────────────────────────────────────────────────────────
# The /proc scanner's mmap pass once matched a candidate lane realpath as a bare
# fixed-string SUBSTRING, so a maps line ".../_lane-10/target/... (deleted)"
# from an _lane-10 build spuriously matched the free candidate ".../_lane-1" —
# perpetually shielding the low-numbered lane's divergent target/ from reclaim
# and partially defeating the disk-space purpose of the whole mechanism. The fix
# is the trailing-"/" boundary pattern in live_referenced_paths.
#
# This regression used to be pinned through warm-lane-gc-sweep.sh's up-front CSV
# (its Blocks U14-U18). That CSV is being deleted, so the coverage MIGRATES here,
# to the layer where the logic now lives: gc.sh calls the boundary matcher
# directly, per lane. Expect GREEN on arrival — the boundary fix rides along in
# the extracted lib. Its job is to keep the regression pinned, not to go red.
#
# Reproduced via a lane-teardown race: _lane-10's build binary is exec'd, then
# _lane-10 is torn down so it is no longer an enumerated candidate (which also
# defeats GNU grep's longest-match tie-break, which HIDES the bug while both
# lanes are candidates); the live process's "(deleted)" mapping still names
# ".../_lane-10/..." in /proc/<pid>/maps. Only a real MMAP exercises this pass —
# a cwd/fd ref would use the already-boundary-correct cwd/fd pass — so the
# reference is an ELF exec'd from UNDER _lane-10/target with cwd OUTSIDE the
# mount, making the mmap the SOLE lane reference.
echo ""
echo "--- Block R: mmap substring-boundary regression against gc.sh (esc-5378, migrated by 5572) ---"

R_ROOT="$(mktemp -d /tmp/test-gc-r-XXXXXX)"
_TMPDIRS+=("$R_ROOT")

R_REPO="$R_ROOT/repo"
R_WORKTREES="$R_ROOT/worktrees"
R_BASE="$R_ROOT/base"
mkdir -p "$R_WORKTREES" "$R_BASE" "$R_ROOT/stage"

make_repo "$R_REPO"

mkdir -p "$R_BASE/target.gen.1"
touch "$R_BASE/target.gen.1.lock"
ln -sfn "$R_BASE/target.gen.1" "$R_BASE/target"

# _lane-1 and _lane-10: both clean+landed pool lanes with divergent markers and
# FREE flocks. _lane-1's basename is a NAME-PREFIX of _lane-10's.
for _r_name in _lane-1 _lane-10; do
    git -C "$R_REPO" worktree add -q "$R_WORKTREES/$_r_name"
    mkdir -p "$R_WORKTREES/$_r_name/target"
    touch "$R_WORKTREES/$_r_name/target/DIVERGENT_MARKER"
done

# A standalone ELF under _lane-10/target; exec'ing it maps it as
# ".../_lane-10/target/live-bin" in the helper's /proc/<pid>/maps. cwd is the
# stage dir OUTSIDE the mount, so NO cwd/fd reference touches any lane.
R_BIN="$R_WORKTREES/_lane-10/target/live-bin"
cp "$(command -v sleep)" "$R_BIN"
chmod +x "$R_BIN"
R_BIN_RP="$(readlink -f "$R_BIN")"
( cd "$R_ROOT/stage" && exec "$R_BIN" 300 ) &
R_HELPER_PID=$!
_BGPIDS+=("$R_HELPER_PID")
_wait_for_exec_map "$R_HELPER_PID" "$R_BIN_RP" 30 && R_MAPPED=1 || R_MAPPED=0

assert "R1: fixture — helper exec'd the lane binary (live mmap established)" \
    test "$R_MAPPED" -eq 1
assert "R2: fixture — the _lane-10 build binary is mmap'd in the live helper (pre-teardown)" \
    bash -c 'grep -qF "/_lane-10/target/live-bin" "/proc/$1/maps"' _ "$R_HELPER_PID"

# Tear down _lane-10 so it is no longer an enumerated candidate (its "(deleted)"
# mapping lingers in the live process's maps). Now _lane-1 is the ONLY candidate
# whose realpath is a bare substring of that ".../_lane-10/..." maps line.
rm -f "$R_BIN"
rm -rf "$R_WORKTREES/_lane-10"

R_SEED_LOG="$R_ROOT/seed_calls.log"
R_SEED_STUB="$R_ROOT/seed_stub.sh"
_seed_stub_body > "$R_SEED_STUB"
chmod +x "$R_SEED_STUB"
export SEED_LOG="$R_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$R_WORKTREES" \
    --base-target "$R_BASE/target" \
    --seed-script "$R_SEED_STUB" \
    --main-ref main

assert "R3: exit 0" test "$RC" -eq 0
assert "R4: free _lane-1 (name-prefix of a live _lane-10 mmap) IS reset — marker removed" \
    bash -c '[ ! -f "$1" ]' _ "$R_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "R5: _lane-1 seed-script invoked (not spuriously protected by the name-prefix mmap)" \
    bash -c 'test -f "$1" && grep -q "_lane-1" "$1"' _ "$R_SEED_LOG"
assert "R6: stderr does NOT claim a process reference for _lane-1" \
    bash -c '! printf "%s\n" "$1" | grep -qF "preserving _lane-1: live consumer (process reference)"' _ "$ERR_OUT"

kill "$R_HELPER_PID" 2>/dev/null || true
wait "$R_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ──────────────────────────────────────────────────────────────────────────────
# Block S — the durable lane record is the Pass-1 reclaimability gate (task 3075)
# ──────────────────────────────────────────────────────────────────────────────
# ROOT CAUSE (reify esc-5334-6): gc.sh's header claimed "FREE/idle ≈ no live
# consumer holding the lane flock". That approximation is FALSE for most of an
# ASSIGNED lane's life — the inv.2 flock is held only across the acquire reseed
# and across run_scoped_verification, never across the implement phase — so task
# 5326's always-reclaim, written for "a FREE pool lane", fires on lanes that
# dark-factory has assigned to a live task.
#
# FIX: consult dark-factory's OWN durable lane record —
# <worktrees-dir>/.lane-state/<lane>.json, read per lane through
# lib_lane_state.sh — inside the Pass-1 loop, under the flock the loop already
# holds. Every sub-case below is invoked DF-FAITHFULLY: no --extra-protect-glob,
# and every <lane>.lock FREE, so neither the wrapper CSV nor the flock gate can
# account for any preservation observed.
#
# The state dir is DOT-PREFIXED and gc.sh's candidate loop is
# `"$WORKTREES_DIR"/*/` with no `shopt -s dotglob` anywhere, so `.lane-state/`
# is never enumerated as a candidate and cannot inflate `preserved=`.
echo ""
echo "--- Block S: the durable lane record gates Pass-1 reclaim (task 3075) ---"

# ── S-basic: an ASSIGNED lane is preserved from the dark-factory entry point ──
# _lane-5 carries an `assigned` record naming task 5334 (the esc-5334-6 lane);
# _lane-1 carries a `released` record. Both are clean+landed with a FREE flock,
# so under the pre-γ rule BOTH would be reset. Discrimination, not blanket
# over-preserve.
S_ROOT="$(mktemp -d /tmp/test-gc-s-XXXXXX)"
_TMPDIRS+=("$S_ROOT")

S_REPO="$S_ROOT/repo"
S_WORKTREES="$S_ROOT/worktrees"
S_BASE="$S_ROOT/base"
mkdir -p "$S_WORKTREES" "$S_BASE"

make_repo "$S_REPO"

mkdir -p "$S_BASE/target.gen.1"
touch "$S_BASE/target.gen.1.lock"
ln -sfn "$S_BASE/target.gen.1" "$S_BASE/target"

for _s_name in _lane-1 _lane-5; do
    git -C "$S_REPO" worktree add -q "$S_WORKTREES/$_s_name"
    mkdir -p "$S_WORKTREES/$_s_name/target"
    touch "$S_WORKTREES/$_s_name/target/DIVERGENT_MARKER"
done

# Records built through the SHARED factory (tests/warm-lane/test_helpers.sh),
# the same one warm-lane-audit.sh's suite uses — no third hand-copied mirror.
make_lane_state "$S_WORKTREES" _lane-5 assigned 5334
make_lane_state "$S_WORKTREES" _lane-1 released

S_SEED_LOG="$S_ROOT/seed_calls.log"
S_SEED_STUB="$S_ROOT/seed_stub.sh"
_seed_stub_body > "$S_SEED_STUB"
chmod +x "$S_SEED_STUB"
export SEED_LOG="$S_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$S_WORKTREES" \
    --base-target "$S_BASE/target" \
    --seed-script "$S_SEED_STUB" \
    --main-ref main

assert "S1: exit 0" test "$RC" -eq 0
assert "S2: assigned _lane-5 seed-script NOT invoked (preserved, not reset)" \
    bash -c '[ ! -f "$1" ] || ! grep -q "_lane-5" "$1"' _ "$S_SEED_LOG"
assert "S3: assigned _lane-5 divergent marker INTACT (FREE flock, no --extra-protect-glob)" \
    test -f "$S_WORKTREES/_lane-5/target/DIVERGENT_MARKER"
# The PRD/task-pinned reason string. Asserted with grep -F on the exact prefix so
# a later suffix (the raw state, step-3) keeps this substring match green.
assert "S4: stderr names the record-driven preserve reason for _lane-5" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-5: assigned to task 5334"' _ "$ERR_OUT"
assert "S5: released _lane-1 still reset (fail-open on a non-assigned record)" \
    bash -c 'test -f "$1" && grep -q "_lane-1" "$1" && [ ! -f "$2" ]' _ \
    "$S_SEED_LOG" "$S_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "S6: summary reports reset=1 (only the released lane was reclaimed)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1"' _ "$OUT"

# ── S-reasons: the reason vocabulary, the gate ORDER, and the counter ─────────
# Three preserved lanes in ONE pass:
#   _lane-6  state `in_use` — leaf δ has no writer yet, so the record is
#            hand-written here; this is what makes the raw state's visibility
#            testable NOW rather than after δ lands.
#   _lane-7  state `assigned` with task_id JSON `null` — the record must never
#            render a dangling "assigned to task " with an empty id.
#   _lane-8  state `assigned` AND a live process reference. It would be
#            preserved either way, so the assertion is about ATTRIBUTION: the
#            RECORD must be the gate that fired, which is what pins the ~1ms
#            read ahead of the ~1.9s /proc walk.
SR_ROOT="$(mktemp -d /tmp/test-gc-sr-XXXXXX)"
_TMPDIRS+=("$SR_ROOT")

SR_REPO="$SR_ROOT/repo"
SR_WORKTREES="$SR_ROOT/worktrees"
SR_BASE="$SR_ROOT/base"
mkdir -p "$SR_WORKTREES" "$SR_BASE"

make_repo "$SR_REPO"

mkdir -p "$SR_BASE/target.gen.1"
touch "$SR_BASE/target.gen.1.lock"
ln -sfn "$SR_BASE/target.gen.1" "$SR_BASE/target"

for _sr_name in _lane-6 _lane-7 _lane-8; do
    git -C "$SR_REPO" worktree add -q "$SR_WORKTREES/$_sr_name"
    mkdir -p "$SR_WORKTREES/$_sr_name/target"
    touch "$SR_WORKTREES/$_sr_name/target/DIVERGENT_MARKER"
done

make_lane_state "$SR_WORKTREES" _lane-6 in_use 5551
make_lane_state "$SR_WORKTREES" _lane-7 assigned
make_lane_state "$SR_WORKTREES" _lane-8 assigned 5416

# Live cwd holder under _lane-8/target, established causally (technique R — no
# wall-clock sleep), exactly as P-basic does.
SR_READY="$SR_ROOT/lane8-holder.ready"
( cd "$SR_WORKTREES/_lane-8/target" && touch "$SR_READY" && exec sleep 300 ) &
SR_HELPER_PID=$!
_BGPIDS+=("$SR_HELPER_PID")
_wait_for_reader_lock "$SR_READY" 30

SR_SEED_LOG="$SR_ROOT/seed_calls.log"
SR_SEED_STUB="$SR_ROOT/seed_stub.sh"
_seed_stub_body > "$SR_SEED_STUB"
chmod +x "$SR_SEED_STUB"
export SEED_LOG="$SR_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$SR_WORKTREES" \
    --base-target "$SR_BASE/target" \
    --seed-script "$SR_SEED_STUB" \
    --main-ref main

assert "S7: in_use _lane-6 preserved (seed NOT invoked, marker intact)" \
    bash -c '([ ! -f "$1" ] || ! grep -q "_lane-6" "$1") && [ -f "$2" ]' _ \
    "$SR_SEED_LOG" "$SR_WORKTREES/_lane-6/target/DIVERGENT_MARKER"
# The RAW state stays visible in the reason, so `assigned` and `in_use` are
# distinguishable in dark-factory's logs the moment leaf δ starts writing the
# latter — leaf ε depends on that distinction.
assert "S8: stderr surfaces the RAW state for _lane-6 (in_use, not flattened to assigned)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-6: assigned to task 5551 (state=in_use)"' _ "$ERR_OUT"
assert "S9: a null task_id gets its own wording for _lane-7" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-7: assigned, no task id in record"' _ "$ERR_OUT"
assert "S9b: ...and NEVER renders a dangling 'assigned to task ' with an empty id" \
    bash -c '! printf "%s\n" "$1" | grep -qE "preserving _lane-7: assigned to task( |$)"' _ "$ERR_OUT"
# GATE ORDER. _lane-8 is both assigned AND live-referenced; the record must be
# the gate that fires, ahead of the walk.
assert "S10: _lane-8 preserved by the RECORD gate" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-8: assigned to task 5416"' _ "$ERR_OUT"
assert "S10b: ...and NOT attributed to the /proc live-reference gate (record is checked first)" \
    bash -c '! printf "%s\n" "$1" | grep -qF "preserving _lane-8: live consumer (process reference)"' _ "$ERR_OUT"
# The full adjacent triple pins BOTH the new field's value AND that it was
# APPENDED after preserved_live_ref rather than interposed — the header's stated
# rule, and what keeps K5's and P-fd8's `preserved=1 preserved_live_ref=1` greps
# green.
assert "S11: summary APPENDS preserved_assigned=3 after preserved_live_ref (never interposed)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=3 preserved_live_ref=0 preserved_assigned=3"' _ "$OUT"

kill "$SR_HELPER_PID" 2>/dev/null || true
wait "$SR_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

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
