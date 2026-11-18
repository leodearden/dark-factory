#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_warm_lane_gc_native.sh
# Hermetic tests for orchestrator/scripts/warm-lane/warm-lane-gc.sh — the
# dark-factory-native coverage, which has no reify counterpart.
# The reify-ported blocks live in test_warm_lane_gc.sh.
#
# Blocks:
#   A — CLI guard, the native cases: A10, A12, A11a-n/l. A MISSING sibling
#       scripts/lib_lane_state.sh fails LOUD with exit 2, with
#       lib_live_refs.sh PRESENT so the guard test_warm_lane_gc.sh's A9 pins
#       cannot account for it (A10, task 3075), and so does
#       scripts/lib_lane_seed.sh with both earlier siblings present (A12, task
#       7859); --max-record-age-days is validated and a misconfiguration is
#       FATAL, because EVERY wrong bound is invisible in the summary line:
#       missing value, non-integer and negative all exit 2 (A11a-A11d), and so
#       do the two doors the `^[0-9]+$` first cut ACCEPTED — a LEADING ZERO read
#       as octal and an OVERFLOWING magnitude that wrapped the multiply negative
#       (A11e-A11k, which also pin the env var as the guard's second door). A11l
#       is the non-vacuity case. What each door measurably did lives in
#       orchestrator/scripts/warm-lane/README.md "Delta 10"; what is pinned HERE
#       is only that the guard rejects it (task 5504)
#   A11-boundary — the bound is validated BEFORE any lane is touched, which no
#       Block A case can reach because A11's fixture has no lanes. Glob order
#       puts a reclaimable lane ahead of the assigned one, so a boundary guard
#       and a mid-pass abort are distinguishable from the first lane alone: a
#       misconfigured bound must reclaim NOTHING and exit 2, not reclaim half a
#       pool and die with exit 1 and no summary line (task 5504)
#   S — the DURABLE LANE RECORD is Pass 1's reclaimability gate (task 3075,
#       reify esc-5334-6): a lane whose <worktrees-dir>/.lane-state/<lane>.json
#       reads `assigned`/`in_use` is preserved, with a FREE flock and NO
#       --extra-protect-glob — the dark-factory ε shape. S-basic pins the
#       preserve and that a `released` lane still reclaims; S-reasons pins the
#       reason vocabulary (the raw state stays visible; a null task_id never
#       renders a dangling id), the GATE ORDER (record before the /proc walk),
#       and the appended preserved_assigned= counter; S-fallback pins that the
#       task-5572 /proc scan is INHERITED as the recordless fallback (PRD D7),
#       that released/quarantined/corrupt all fall OPEN, and that a corrupt
#       record is loud while the ordinary recordless case stays quiet;
#       S-toctou pins PLACEMENT — a lane assigned MID-PASS is still preserved,
#       so the read cannot be hoisted into an up-front classification pass
#       without going RED (GREEN on arrival, like test_warm_lane_gc.sh's
#       Block R).
#       Task 5504 BOUNDS that preserve, because an unbounded one composed with
#       the gate's position before the --disk-pressure branch held the ENOSPC
#       valve shut with the very failure it responds to (a release whose
#       durable write hit ENOSPC and was swallowed leaves `assigned` on disk
#       while the pool reads FREE). Two orthogonal downgrade reasons, OR'd:
#       S-pressure pins the ACUTE one — --disk-pressure downgrades the gate
#       to warn-and-fall-through, with a two-arm fixture proving the change is
#       scoped to the emergency path, and with a live-referenced lane proving
#       "downgrade" means FALL THROUGH to the /proc gate, never reclaim
#       outright; S-age pins the CHRONIC one — a record whose `updated_at` is
#       older than --max-record-age-days (default 14) is downgraded in every
#       mode, with a fresh sibling preserved in the same pass so it is
#       discrimination and not blanket expiry, plus `0` as the escape hatch
#       and the REIFY_WARM_LANE_GC_MAX_RECORD_AGE_DAYS knob; S-age-degrade
#       pins the DELIBERATE ASYMMETRY — an unreadable STATE reclaims (S15/S17)
#       while an unreadable AGE preserves loudly, and Arm B pins that the
#       acute valve is never subject to that fail-safe because the
#       --disk-pressure leg is evaluated first
#   V — Pass 1 SKIPS a lane whose cache is unchanged since its last seed on the
#       current base generation (task 7859): counted in the appended
#       skipped_unchanged=, never in reset or preserved. Every doubt — a write
#       since the seed, a moved generation, a missing or untrustworthy stamp,
#       no target/, no python3 — resets as before; --disk-pressure and an assigned record
#       both take precedence over the skip (README.md "Delta 14")
#   X — the DEFAULT protect-glob is RENDERED from dark-factory's
#       PROTECTED_PREFIXES via lib_lane_state.sh's lane_protect_glob, not
#       hand-mirrored as a literal (task 3292). X-band pins the observable
#       payoff — the interactive band is config-driven, so
#       REIFY_WARM_LANE_IACT_PREFIX moves which band is protected, asserted in
#       BOTH directions (the renamed band survives, the stock `_iact-*` one
#       does not); X-degrade pins the `||` fallback in situ, driving a
#       RELOCATED copy of gc.sh whose resolved repo root carries no
#       dark-factory checkout, so the bridge fails and gc.sh must degrade to
#       LANE_PROTECT_GLOB_FALLBACK — loudly (exactly one [warn]) and without
#       aborting its own `set -euo pipefail` sweep
#   TRASH — this suite's own shared-trash litter guard pair (task 5612)
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (NATIVE_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
SCRIPT="$WARM_LANE_SCRIPTS_DIR/warm-lane-gc.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=orchestrator/tests/warm-lane/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"
# run_helper/run_helper_live_default, make_repo, make_gc_fixture, _seed_stub_body
# and _wait_for_reader_lock come from the harness both gc suites share.
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_gc_harness.sh
source "$SCRIPT_DIR/lib_warm_lane_gc_harness.sh"

echo "=== scripts/warm-lane-gc.sh dark-factory-native hermetic tests (task 6931) ==="

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

# Arm the shared-trash litter guard (task 5612): see the CANONICAL WIRING
# CONTRACT comment in test_helpers.sh.
init_isolated_lane_root test-native-gc

ERR_FILE="$(mktemp /tmp/test-native-gc-err-XXXXXX)"
_TMPDIRS+=("$ERR_FILE")

# ──────────────────────────────────────────────────────────────────────────────
# Block A — CLI guard (dark-factory-native cases)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block A: CLI guard (dark-factory-native cases) ---"

# A10: the SAME contract for the new sibling scripts/lib_lane_state.sh (task
# 3075). A silently-absent lane-state reader degrades reclaim back to exactly
# the `FREE ≈ flock-free` approximation this leaf removes — invisible, and the
# defect reify esc-5334-6 exists to close — so the guard that enforces its
# presence cannot itself be untested.
#
# The fixture copies BOTH gc.sh AND lib_live_refs.sh, deliberately withholding
# only lib_lane_state.sh. Copying lib_live_refs.sh is what makes the assertion
# SPECIFIC: without it, the guard test_warm_lane_gc.sh's A9 pins would fire
# first and A10 would prove nothing about the new dependency.
A10_DIR="$(mktemp -d /tmp/test-gc-a10-XXXXXX)"
_TMPDIRS+=("$A10_DIR")
cp "$SCRIPT" "$A10_DIR/warm-lane-gc.sh"
cp "$WARM_LANE_SCRIPTS_DIR/lib_live_refs.sh" "$A10_DIR/lib_live_refs.sh"
assert "A10: fixture — the copy HAS lib_live_refs.sh and has NO lib_lane_state.sh" \
    bash -c '[ -f "$1/lib_live_refs.sh" ] && [ ! -e "$1/lib_lane_state.sh" ]' _ "$A10_DIR"
A10_RC=0
A10_ERR="$(bash "$A10_DIR/warm-lane-gc.sh" --help 2>&1 >/dev/null)" || A10_RC=$?
# --help normally exits 0, so a 2 here also pins that the guard fires BEFORE
# argv is parsed — the same ordering property test_warm_lane_gc.sh's A9 asserts.
assert "A10a: missing sibling lib_lane_state.sh exits 2 (wiring error, not runtime 1)" \
    test "$A10_RC" -eq 2
# The VERBATIM fragment already registered in
# orchestrator/tests/test_warm_lane_scripts_shipped.py FAIL_LOUD_FRAGMENTS,
# established by warm-lane-audit.sh's guard in task 3074 — so the
# shipped-and-wired contract test covers gc.sh's new guard with no change to its
# assertion data.
assert "A10b: stderr names the missing library with the registered fragment" \
    bash -c 'printf "%s\n" "$1" | grep -qF "lib_lane_state.sh not found next to"' _ "$A10_ERR"

# A12: the SAME contract for scripts/lib_lane_seed.sh (task 7859), A10's shape
# one sibling further on: both earlier libs travel with the copy, so only the
# new guard can account for an exit 2.
A12_DIR="$(mktemp -d /tmp/test-gc-a12-XXXXXX)"
_TMPDIRS+=("$A12_DIR")
cp "$SCRIPT" "$A12_DIR/warm-lane-gc.sh"
cp "$WARM_LANE_SCRIPTS_DIR/lib_live_refs.sh" "$A12_DIR/lib_live_refs.sh"
cp "$WARM_LANE_SCRIPTS_DIR/lib_lane_state.sh" "$A12_DIR/lib_lane_state.sh"
assert "A12: fixture — the copy HAS lib_live_refs.sh and lib_lane_state.sh and has NO lib_lane_seed.sh" \
    bash -c '[ -f "$1/lib_live_refs.sh" ] && [ -f "$1/lib_lane_state.sh" ] && [ ! -e "$1/lib_lane_seed.sh" ]' _ "$A12_DIR"
A12_RC=0
A12_ERR="$(bash "$A12_DIR/warm-lane-gc.sh" --help 2>&1 >/dev/null)" || A12_RC=$?
assert "A12a: missing sibling lib_lane_seed.sh exits 2 (wiring error, not runtime 1)" \
    test "$A12_RC" -eq 2
assert "A12b: stderr names the missing library with the registered fragment" \
    bash -c 'printf "%s\n" "$1" | grep -qF "lib_lane_seed.sh not found next to warm-lane-gc.sh"' _ "$A12_ERR"

# A11: --max-record-age-days is validated, and a misconfiguration is FATAL
# (task 5504). A bound is a safety valve, and both silent failure directions
# are invisible in the summary line: silently defaulting would leave an
# operator believing they had widened it, and silently zeroing would disable
# the valve outright while `downgraded_assigned=0` reads as "nothing was
# stale". Neither is discoverable from the output, so the misconfiguration
# must be loud at the boundary. Exit 2, the usage/WIRING class — nothing about
# the invocation could have avoided it.
A11_ROOT="$(mktemp -d /tmp/test-gc-a11-XXXXXX)"
_TMPDIRS+=("$A11_ROOT")
make_gc_fixture "$A11_ROOT"
run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days
assert "A11a: --max-record-age-days with NO value exits 2 (requires-a-value guard)" \
    test "$RC" -eq 2

run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days notanumber
assert "A11b: --max-record-age-days notanumber exits 2" test "$RC" -eq 2
# `grep -qF --` is load-bearing: without the end-of-options marker grep parses
# the pattern itself as an unrecognized long option and exits 2, which reads as
# "no match" and would make this assertion silently un-assertable.
assert "A11c: ...and stderr names the flag and the offending value" \
    bash -c 'printf "%s\n" "$1" | grep -qF -- "--max-record-age-days" && printf "%s\n" "$1" | grep -qF -- "notanumber"' _ "$ERR_OUT"

# A negative would make MAX_RECORD_AGE_SECS negative, and `age > negative` is
# true for every record — a blanket downgrade of the whole pool from a typo.
run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days -3
assert "A11d: a NEGATIVE --max-record-age-days exits 2 (never a blanket downgrade)" \
    test "$RC" -eq 2

# A11e-A11k: the two doors A11b/A11d do not reach. `^[0-9]+$` rejects junk and
# negatives, but a LEADING ZERO (`08` is the loud one — it dies in the octal
# multiply, leaving MAX_RECORD_AGE_SECS unset mid-pass; `010` is the dangerous
# one — it quietly means 8 days) and an OVERFLOWING MAGNITUDE both sail through
# it and then mean something other than what the operator typed. Neither is
# discoverable from the summary line, which is why the guard must reject rather
# than normalize. The pinned property here is exactly that rejection.
run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days 08
assert "A11e: a LEADING-ZERO --max-record-age-days 08 exits 2 (never read as octal)" \
    test "$RC" -eq 2
# Two halves, and the second is what keeps this assert honest if the guard is
# ever re-loosened: a raw bash arithmetic diagnostic reaching an operator is the
# tell that the value got as far as the multiply, i.e. the guard was bypassed
# rather than enforced. `grep -qF --` for the same reason A11c documents.
assert "A11f: ...and stderr names the flag and the value 08, with NO raw bash arithmetic diagnostic" \
    bash -c 'printf "%s\n" "$1" | grep -qF -- "--max-record-age-days" \
        && printf "%s\n" "$1" | grep -qF -- "08" \
        && ! printf "%s\n" "$1" | grep -qF -- "value too great for base"' _ "$ERR_OUT"

run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days 010
assert "A11g: --max-record-age-days 010 exits 2 (it would SILENTLY mean 8 days, not 10)" \
    test "$RC" -eq 2

# The disable escape hatch is documented as exactly `0`. `00` is the same
# ambiguous-token class and must not become a second, undocumented spelling of
# a switch that turns the valve off.
run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days 00
assert "A11h: --max-record-age-days 00 exits 2 (the disable hatch is spelled 0, and only 0)" \
    test "$RC" -eq 2

# The env var is a REAL second door into the same guard, so it must be guarded
# identically — and the message must name the door the operator actually used,
# not a flag they never passed. Inline command prefix, NEVER an export, per the
# gc suites' established REIFY_* idiom (test_warm_lane_gc.sh's K4 PATH prefix
# and O-env).
REIFY_WARM_LANE_GC_MAX_RECORD_AGE_DAYS=09 run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target"
assert "A11i: a leading-zero value via the ENV VAR exits 2 too (the same guard, the other door)" \
    test "$RC" -eq 2
assert "A11j: ...and the error names the env var, not only the flag the operator never passed" \
    bash -c 'printf "%s\n" "$1" | grep -qF -- "REIFY_WARM_LANE_GC_MAX_RECORD_AGE_DAYS"' _ "$ERR_OUT"

# Overflow is the same forbidden SILENT direction as 010, reached through a
# different door, and A11d's negative case cannot reach it because the typed
# value is positive: `9223372036854775807 * 86400` WRAPS to -86400, which then
# fails the `-gt 0` test and disables the bound outright while
# `downgraded_assigned=0` reads as "nothing was stale".
run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days 9223372036854775807
assert "A11k: an OVERFLOWING --max-record-age-days exits 2 (a wrap would silently disable the valve)" \
    test "$RC" -eq 2

# grep's line-oriented `^…$` accepted a value whose first line is valid.
run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days $'14\nx'
assert "A11m: a multi-line --max-record-age-days exits 2" test "$RC" -eq 2
assert "A11n: ...and stderr names the flag, with NO raw bash arithmetic diagnostic" \
    bash -c 'grep -qF -- "--max-record-age-days" <<<"$1" && ! grep -qF -- "syntax error" <<<"$1"' _ "$ERR_OUT"

# NON-VACUITY. Without this the block would pass a guard that rejected
# everything, and A11's reader cannot see Block S-age's accepting cases from
# here (heuristic 13 — a block makes internal sense in isolation).
run_helper reclaim \
    --worktrees-dir "$A11_ROOT/worktrees" \
    --base-target "$A11_ROOT/base/target" \
    --max-record-age-days 14
assert "A11l: NON-VACUITY — a plain --max-record-age-days 14 still exits 0 on the same fixture" \
    test "$RC" -eq 0

# ──────────────────────────────────────────────────────────────────────────────
# Block A11-boundary — the bound is validated BEFORE any lane is touched
# ──────────────────────────────────────────────────────────────────────────────
# The property that makes the finding's severity concrete, and one no A11 case
# above can reach because A11's fixture has no lanes at all. A guard that fires
# at the boundary makes a misconfigured bound reclaim NOTHING; a guard that
# fires mid-pass makes it reclaim HALF A POOL and then die without a summary
# line to say so.
#
# Glob order is deliberately exploited: `_lane-1` (reclaimable) sorts before
# `_lane-2` (assigned, and so the first lane that reads MAX_RECORD_AGE_SECS),
# so a boundary guard and a mid-pass abort are distinguishable by looking at
# `_lane-1` alone. All <lane>.lock FREE, no live process references, no
# --extra-protect-glob.
echo ""
echo "--- Block A11-boundary: a misconfigured bound reclaims NOTHING (task 5504) ---"

A11B_ROOT="$(mktemp -d /tmp/test-gc-a11b-XXXXXX)"
_TMPDIRS+=("$A11B_ROOT")
mkdir -p "$A11B_ROOT/worktrees" "$A11B_ROOT/base"
make_repo "$A11B_ROOT/repo"
mkdir -p "$A11B_ROOT/base/target.gen.1"
touch "$A11B_ROOT/base/target.gen.1.lock"
ln -sfn "$A11B_ROOT/base/target.gen.1" "$A11B_ROOT/base/target"
for _a11b_name in _lane-1 _lane-2; do
    git -C "$A11B_ROOT/repo" worktree add -q "$A11B_ROOT/worktrees/$_a11b_name"
    mkdir -p "$A11B_ROOT/worktrees/$_a11b_name/target"
    touch "$A11B_ROOT/worktrees/$_a11b_name/target/DIVERGENT_MARKER"
done
make_lane_state "$A11B_ROOT/worktrees" _lane-1 released
make_lane_state "$A11B_ROOT/worktrees" _lane-2 assigned 5504
_seed_stub_body > "$A11B_ROOT/seed_stub.sh"
chmod +x "$A11B_ROOT/seed_stub.sh"
export SEED_LOG="$A11B_ROOT/seed_calls.log"

run_helper reclaim \
    --worktrees-dir "$A11B_ROOT/worktrees" \
    --base-target "$A11B_ROOT/base/target" \
    --seed-script "$A11B_ROOT/seed_stub.sh" \
    --main-ref main \
    --max-record-age-days 08

# Exit 2, the usage/WIRING class — not the runtime 1 a mid-pass `set -u` abort
# produces, and the distinction is load-bearing: an operator (or the sweep's
# own error handling) reads the class to decide whether the invocation could
# have avoided it.
assert "A11-boundary1: a misconfigured bound exits 2 (usage/wiring), not 1 (runtime)" \
    test "$RC" -eq 2
assert "A11-boundary2: the seed stub was NEVER invoked — no lane was reclaimed" \
    bash -c '[ ! -s "$1" ]' _ "$A11B_ROOT/seed_calls.log"
assert "A11-boundary3: the reclaimable _lane-1 is INTACT (a mid-pass abort would have reset it first)" \
    test -f "$A11B_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "A11-boundary4: the assigned _lane-2 is likewise untouched" \
    test -f "$A11B_ROOT/worktrees/_lane-2/target/DIVERGENT_MARKER"
# The LITERAL is load-bearing, and the obvious one is the wrong one:
# `reclaim complete:` is `ok`'s wording and every log helper in warm-lane-gc.sh
# writes to STDERR, so grepping $OUT for it would pass no matter what the script
# did — including in the exact mid-pass abort this case exists to catch. The
# stdout summary is the `reclaim: reset=…` printf, the line a consumer
# prefix-matches; its ABSENCE is what says no pass was half-run.
assert "A11-boundary5: NO reclaim: summary line on stdout (a misconfigured bound must not half-run a pass)" \
    bash -c '! printf "%s\n" "$1" | grep -qF "reclaim: reset="' _ "$OUT"

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
#   _lane-6  state `in_use` — hand-written here even though leaf δ (task
#            3076) now writes it, because this bash test must not depend on
#            Python; the writer is pinned in orchestrator/tests/test_lane_in_use.py.
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
# wall-clock sleep), exactly as test_warm_lane_gc.sh's P-basic does.
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
# distinguishable in dark-factory's logs now that leaf δ writes the latter
# (orchestrator/tests/test_lane_in_use.py) — leaf ε depends on that distinction.
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
# rule, and what keeps test_warm_lane_gc.sh's K5 and P-fd8
# `preserved=1 preserved_live_ref=1` greps green.
assert "S11: summary APPENDS preserved_assigned=3 after preserved_live_ref (never interposed)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=3 preserved_live_ref=0 preserved_assigned=3"' _ "$OUT"

kill "$SR_HELPER_PID" 2>/dev/null || true
wait "$SR_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ── S-fallback: the /proc gate is INHERITED, and every other class falls open ─
# The state dir EXISTS here and only _lane-2's record is missing — sharper than
# "no .lane-state at all", and the real shape of an `_iact-*` or manual operator
# worktree living alongside pool lanes.
#   _lane-1  released      — control, must reset.
#   _lane-2  NO record + a live process reference — reaches the /proc gate via
#            the recordless fall-through, proving the 5572 scan is SUPERSEDED,
#            not deleted (PRD D7).
#   _lane-3  quarantined   — falls open. Reset touches only target/, never the
#            source tree or branch (sizing-lifecycle T1), so a quarantined
#            lane's forensic state survives; only its disk is reclaimed.
#   _lane-4  present, readable, but carrying no `state` string —
#            LANE_STATE_CAUSE=unparseable-record. Must be LOUD and must still
#            fall open: failing closed on a corrupt record would let one bad
#            write freeze a lane out of reclaim forever.
SF_ROOT="$(mktemp -d /tmp/test-gc-sf-XXXXXX)"
_TMPDIRS+=("$SF_ROOT")

SF_REPO="$SF_ROOT/repo"
SF_WORKTREES="$SF_ROOT/worktrees"
SF_BASE="$SF_ROOT/base"
mkdir -p "$SF_WORKTREES" "$SF_BASE"

make_repo "$SF_REPO"

mkdir -p "$SF_BASE/target.gen.1"
touch "$SF_BASE/target.gen.1.lock"
ln -sfn "$SF_BASE/target.gen.1" "$SF_BASE/target"

for _sf_name in _lane-1 _lane-2 _lane-3 _lane-4; do
    git -C "$SF_REPO" worktree add -q "$SF_WORKTREES/$_sf_name"
    mkdir -p "$SF_WORKTREES/$_sf_name/target"
    touch "$SF_WORKTREES/$_sf_name/target/DIVERGENT_MARKER"
done

make_lane_state "$SF_WORKTREES" _lane-1 released
# _lane-2: deliberately NO record — the state dir exists (make_lane_state above
# created it), only this lane's entry is absent.
make_lane_state "$SF_WORKTREES" _lane-3 quarantined
make_lane_state_raw "$SF_WORKTREES" _lane-4 '{"lane": "_lane-4", "note": "truncated write'

SF_READY="$SF_ROOT/lane2-holder.ready"
( cd "$SF_WORKTREES/_lane-2/target" && touch "$SF_READY" && exec sleep 300 ) &
SF_HELPER_PID=$!
_BGPIDS+=("$SF_HELPER_PID")
_wait_for_reader_lock "$SF_READY" 30

SF_SEED_LOG="$SF_ROOT/seed_calls.log"
SF_SEED_STUB="$SF_ROOT/seed_stub.sh"
_seed_stub_body > "$SF_SEED_STUB"
chmod +x "$SF_SEED_STUB"
export SEED_LOG="$SF_SEED_LOG"

run_helper reclaim \
    --worktrees-dir "$SF_WORKTREES" \
    --base-target "$SF_BASE/target" \
    --seed-script "$SF_SEED_STUB" \
    --main-ref main

assert "S12: recordless _lane-2 reaches the INHERITED /proc gate (PRD D7)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-2: live consumer (process reference)"' _ "$ERR_OUT"
assert "S13: recordless _lane-2 preserved (marker intact, seed NOT invoked)" \
    bash -c '[ -f "$1" ] && ([ ! -f "$2" ] || ! grep -q "_lane-2" "$2")' _ \
    "$SF_WORKTREES/_lane-2/target/DIVERGENT_MARKER" "$SF_SEED_LOG"
assert "S14: released _lane-1 reset (fail open on RELEASED)" \
    bash -c 'grep -q "_lane-1" "$1" && [ ! -f "$2" ]' _ \
    "$SF_SEED_LOG" "$SF_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "S15: quarantined _lane-3 reset (QUARANTINED does not preserve)" \
    bash -c 'grep -q "_lane-3" "$1" && [ ! -f "$2" ]' _ \
    "$SF_SEED_LOG" "$SF_WORKTREES/_lane-3/target/DIVERGENT_MARKER"
# INV-2, structured-facts-at-failure: a record that IS there and is wrong means
# an assigned lane has silently degraded back to the pre-γ FREE≈flock-free
# regime. That has to be visible AND attributable to the lane.
assert "S16: a corrupt record is LOUD and names both the lane and the cause" \
    bash -c 'printf "%s\n" "$1" | grep -qE "_lane-4.*unparseable-record"' _ "$ERR_OUT"
assert "S17: ...and still falls open — _lane-4 is reset (one bad write cannot freeze a lane)" \
    bash -c 'grep -q "_lane-4" "$1" && [ ! -f "$2" ]' _ \
    "$SF_SEED_LOG" "$SF_WORKTREES/_lane-4/target/DIVERGENT_MARKER"
# The asymmetry is the point: no-readable-record is the ORDINARY reading for
# every recordless _iact-*/manual worktree, so warning on it would emit a line
# per entry per pass and train operators to ignore the channel S16 exists to
# raise.
assert "S18: the ordinary recordless case stays QUIET (no no-readable-record spam)" \
    bash -c '! printf "%s\n" "$1" | grep -qF "no-readable-record"' _ "$ERR_OUT"
assert "S19: summary reset=3 with the preserve attributed to the /proc gate, not the record" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=3 .*preserved_live_ref=1 preserved_assigned=0"' _ "$OUT"

kill "$SF_HELPER_PID" 2>/dev/null || true
wait "$SF_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ── S-toctou: the up-front-snapshot TOCTOU (B2, INV-3) ───────────────────────
# PLACEMENT test, not a record-reading test. BOTH lanes read `released` when the
# pass starts, so ANY implementation that snapshots the state dir up front — a
# precomputed map, an `--assigned-lanes CSV`, a hoisted read — classifies both
# as reclaimable and resets both. It goes GREEN only when the read runs per
# lane, from THIS lane's own path, immediately before that lane's reset.
#
# That is not a theoretical failure mode: it is exactly what
# warm-lane-gc-sweep.sh's protect CSV did. It was computed once and consumed
# across a sweep that ran 06:32-06:59 BST while _lane-5 dropped out of the
# protect set at 06:32 (PRD §2).
#
# Mechanism, mirroring test_warm_lane_gc.sh's P-toctou: gc.sh invokes
# --seed-script SYNCHRONOUSLY inside the Pass-1 loop, and bash `*/` glob
# expansion is LC_COLLATE-sorted, so _lane-1 is ALWAYS visited before _lane-2.
# The ordering is RELIED UPON, not incidental. _lane-1's stub therefore
# overwrites _lane-2's record — the only place in a hermetic bash test that
# provably runs between one lane's reset and the next lane's gate.
#
# Expected disposition: GREEN ON ARRIVAL, because step-2 already placed the read
# inside the loop — recorded here the way test_warm_lane_gc.sh's Block R records
# its own. Its job is to go RED against any LATER optimisation that hoists the
# read into an up-front classification pass, which is precisely what PRD D5
# forbids.
ST_ROOT="$(mktemp -d /tmp/test-gc-st-XXXXXX)"
_TMPDIRS+=("$ST_ROOT")

ST_REPO="$ST_ROOT/repo"
ST_WORKTREES="$ST_ROOT/worktrees"
ST_BASE="$ST_ROOT/base"
mkdir -p "$ST_WORKTREES" "$ST_BASE"

make_repo "$ST_REPO"

mkdir -p "$ST_BASE/target.gen.1"
touch "$ST_BASE/target.gen.1.lock"
ln -sfn "$ST_BASE/target.gen.1" "$ST_BASE/target"

for _st_name in _lane-1 _lane-2; do
    git -C "$ST_REPO" worktree add -q "$ST_WORKTREES/$_st_name"
    mkdir -p "$ST_WORKTREES/$_st_name/target"
    touch "$ST_WORKTREES/$_st_name/target/DIVERGENT_MARKER"
done

# Both FREE at pass start.
make_lane_state "$ST_WORKTREES" _lane-1 released
make_lane_state "$ST_WORKTREES" _lane-2 released

ST_SEED_LOG="$ST_ROOT/seed_calls.log"
ST_SEED_STUB="$ST_ROOT/seed_stub.sh"

# The trigger stub: behaves like _seed_stub_body for every lane, and
# ADDITIONALLY, when invoked for _lane-1, reassigns _lane-2 before returning.
# The record is written in the SAME byte shape make_lane_state emits, so the
# fixture and the mid-pass mutation cannot disagree about the record format --
# including updated_at, which the stub takes from $TOCTOU_UPDATED_AT stamped
# from the same clock the factory uses. A frozen past literal here would read
# as STALE to the Pass-1 record gate's age bound (task 5504) and the mid-pass
# reassignment would be downgraded rather than honoured, which is the opposite
# of what S21-S23 exist to pin.
cat > "$ST_SEED_STUB" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "$SEED_LOG"
LANE_DIR="$2"
rm -rf "$LANE_DIR/target/DIVERGENT_MARKER" 2>/dev/null || true
if [ "${LANE_DIR##*/}" = "_lane-1" ]; then
    printf '{\n  "state": "assigned",\n  "task_id": "%s",\n  "title": "lane fixture task %s",\n  "branch": null,\n  "seeded_from_sha": null,\n  "updated_at": "%s"\n}' \
        "$TOCTOU_TASK_ID" "$TOCTOU_TASK_ID" "$TOCTOU_UPDATED_AT" \
        > "$TOCTOU_STATE_DIR/_lane-2.json"
fi
exit 0
STUB_EOF
chmod +x "$ST_SEED_STUB"

export SEED_LOG="$ST_SEED_LOG"
export TOCTOU_STATE_DIR="$ST_WORKTREES/.lane-state"
export TOCTOU_TASK_ID="5334"
export TOCTOU_UPDATED_AT="$(date -u +%Y-%m-%dT%H:%M:%S.000000+00:00)"

run_helper reclaim \
    --worktrees-dir "$ST_WORKTREES" \
    --base-target "$ST_BASE/target" \
    --seed-script "$ST_SEED_STUB" \
    --main-ref main

unset TOCTOU_STATE_DIR TOCTOU_TASK_ID TOCTOU_UPDATED_AT

# Non-vacuity: without S20 the whole sub-case could pass because the stub never
# ran and the mutation never fired.
assert "S20: trigger lane _lane-1 WAS reset (the stub ran, so the mutation fired)" \
    bash -c 'grep -q "_lane-1" "$1" && [ ! -f "$2" ]' _ \
    "$ST_SEED_LOG" "$ST_WORKTREES/_lane-1/target/DIVERGENT_MARKER"
assert "S21: _lane-2 assigned MID-PASS is still preserved (no up-front snapshot)" \
    bash -c '[ -f "$1" ] && ([ ! -f "$2" ] || ! grep -q "_lane-2" "$2")' _ \
    "$ST_WORKTREES/_lane-2/target/DIVERGENT_MARKER" "$ST_SEED_LOG"
assert "S22: ...and preserved for the RIGHT reason (the record, read at gate time)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-2: assigned to task 5334"' _ "$ERR_OUT"
assert "S23: summary reset=1 with the preserve attributed to the record gate" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1 .*preserved_assigned=1"' _ "$OUT"

# ── S-pressure: --disk-pressure DOWNGRADES the record gate (task 5504) ────────
# test_warm_lane_gc.sh's Block K5 pins one half of the "before BOTH reset
# branches" contract: under --disk-pressure the LIVE-REFERENCE gate still fires.
# This is the other half, and it is the half the composition in task 5504 turns
# on.
#
# THE DEFECT. The record gate `continue`d unconditionally on ASSIGNED, and it
# sits BEFORE the --disk-pressure branch. So a lane whose durable record says
# `assigned` was unreachable by the emergency rm — including a lane that is
# assigned ONLY because the release that should have cleared the record hit
# ENOSPC and WarmLanePool._note_released_durable swallowed the OSError
# (fail-open release, invariant I3). That composition holds the ENOSPC valve
# shut with the very failure it exists to respond to, and it does so on the
# UNATTENDED path: warm-lane-gc-sweep.sh measures `df -B1 --output=avail` once
# per sweep and APPENDS --disk-pressure below the critical floor.
#
# THE CONTRACT. --disk-pressure downgrades the record gate to
# warn-and-fall-through. "Downgrade" means FALL THROUGH, never "reclaim
# outright": the lane still meets the live-reference gate, so the emergency
# trades an AUTHORITATIVE preserve for a PROBED one, not for none. Arm A is the
# emergency; Arm B is the SAME fixture without the flag, proving the change is
# scoped to the emergency path and Block S's ordinary contract is untouched.
#
# DF-faithful throughout: every <lane>.lock FREE and NO --extra-protect-glob, so
# neither the flock nor a protect glob can account for any preservation seen.
echo ""
echo "--- Block S-pressure: --disk-pressure downgrades the record gate (task 5504) ---"

# _make_sp_fixture <root-var-prefix-dir>
# Builds the three-lane fixture both arms share, into the directory given.
# A function, not a copy-paste: the two arms MUST be identical apart from the
# flag, or the discrimination the block claims is not discrimination.
_make_sp_fixture() {
    local root="$1"
    mkdir -p "$root/worktrees" "$root/base"
    make_repo "$root/repo"
    mkdir -p "$root/base/target.gen.1"
    touch "$root/base/target.gen.1.lock"
    ln -sfn "$root/base/target.gen.1" "$root/base/target"
    local _sp_name
    for _sp_name in _lane-1 _lane-2 _lane-3; do
        git -C "$root/repo" worktree add -q "$root/worktrees/$_sp_name"
        mkdir -p "$root/worktrees/$_sp_name/target"
        touch "$root/worktrees/$_sp_name/target/DIVERGENT_MARKER"
    done
    # Both assigned records are FRESH (the factory stamps NOW), so nothing in
    # this block can be explained by the staleness bound — the only variable
    # between the two arms is --disk-pressure.
    make_lane_state "$root/worktrees" _lane-1 assigned 5504
    make_lane_state "$root/worktrees" _lane-2 assigned 5505
    make_lane_state "$root/worktrees" _lane-3 released
    _seed_stub_body > "$root/seed_stub.sh"
    chmod +x "$root/seed_stub.sh"
}

# ── Arm A: WITH --disk-pressure ───────────────────────────────────────────────
SPA_ROOT="$(mktemp -d /tmp/test-gc-spa-XXXXXX)"
_TMPDIRS+=("$SPA_ROOT")
_make_sp_fixture "$SPA_ROOT"

# _lane-2 also carries a live cwd holder, established causally (technique R, no
# wall-clock sleep) exactly as test_warm_lane_gc.sh's K5 does. It is the lane
# that proves the downgrade hands the record's verdict to the /proc gate rather
# than discarding it.
SPA_READY="$SPA_ROOT/lane2-holder.ready"
( cd "$SPA_ROOT/worktrees/_lane-2/target" && touch "$SPA_READY" && exec sleep 300 ) &
SPA_HELPER_PID=$!
_BGPIDS+=("$SPA_HELPER_PID")
_wait_for_reader_lock "$SPA_READY" 30

export SEED_LOG="$SPA_ROOT/seed_calls.log"

run_helper reclaim \
    --worktrees-dir "$SPA_ROOT/worktrees" \
    --base-target "$SPA_ROOT/base/target" \
    --seed-script "$SPA_ROOT/seed_stub.sh" \
    --main-ref main \
    --disk-pressure

assert "S-pressure1: exit 0" test "$RC" -eq 0
assert "S-pressure2: assigned _lane-1 target/ DELETED (the record gate was downgraded and the rm branch was reached)" \
    bash -c '[ ! -d "$1" ]' _ "$SPA_ROOT/worktrees/_lane-1/target"
assert "S-pressure3: stderr names _lane-1's downgrade and attributes it to --disk-pressure" \
    bash -c 'printf "%s\n" "$1" | grep -qF "downgrading record gate for _lane-1: --disk-pressure emergency override"' _ "$ERR_OUT"
assert "S-pressure4: ...and the downgraded lane does NOT also print the ordinary assigned-preserve line" \
    bash -c '! printf "%s\n" "$1" | grep -qF "preserving _lane-1: assigned to task 5504"' _ "$ERR_OUT"
# THE LOAD-BEARING HALF. Downgrade means fall through, not reclaim: _lane-2 is
# assigned AND live-referenced, and the emergency must still stop at the /proc
# gate. If this goes red the valve has become a wipe.
assert "S-pressure5: live-referenced _lane-2 target/ SURVIVES (downgrade falls through to the live-reference gate, it does not reclaim outright)" \
    test -d "$SPA_ROOT/worktrees/_lane-2/target"
assert "S-pressure6: live-referenced _lane-2 divergent marker intact (nothing under it was deleted)" \
    test -f "$SPA_ROOT/worktrees/_lane-2/target/DIVERGENT_MARKER"
assert "S-pressure7: ...and _lane-2's preserve is attributed to the /proc gate, the backstop the downgrade hands it to" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-2: live consumer (process reference)"' _ "$ERR_OUT"
assert "S-pressure8: released _lane-3 target/ DELETED (non-vacuity: the pass really ran the rm branch)" \
    bash -c '[ ! -d "$1" ]' _ "$SPA_ROOT/worktrees/_lane-3/target"
assert "S-pressure9: the seed stub was NOT invoked for any lane (the disk-pressure branch has no α fallback)" \
    bash -c '[ ! -s "$1" ]' _ "$SPA_ROOT/seed_calls.log"
# Full adjacent tail, the way S11 does it: pins BOTH downgraded_assigned's value
# AND that it was APPENDED after preserved_assigned rather than interposed.
assert "S-pressure10: summary APPENDS downgraded_assigned=2 after preserved_assigned (never interposed)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=2 removed=0 preserved=1 preserved_live_ref=1 preserved_assigned=0 downgraded_assigned=2"' _ "$OUT"

kill "$SPA_HELPER_PID" 2>/dev/null || true
wait "$SPA_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ── Arm B: the SAME fixture, WITHOUT --disk-pressure ──────────────────────────
# The control that proves the downgrade is scoped to the emergency path. Every
# byte of the fixture is identical (same factory function); the flag is the only
# difference, so any divergence in outcome is attributable to it alone.
SPB_ROOT="$(mktemp -d /tmp/test-gc-spb-XXXXXX)"
_TMPDIRS+=("$SPB_ROOT")
_make_sp_fixture "$SPB_ROOT"

SPB_READY="$SPB_ROOT/lane2-holder.ready"
( cd "$SPB_ROOT/worktrees/_lane-2/target" && touch "$SPB_READY" && exec sleep 300 ) &
SPB_HELPER_PID=$!
_BGPIDS+=("$SPB_HELPER_PID")
_wait_for_reader_lock "$SPB_READY" 30

export SEED_LOG="$SPB_ROOT/seed_calls.log"

run_helper reclaim \
    --worktrees-dir "$SPB_ROOT/worktrees" \
    --base-target "$SPB_ROOT/base/target" \
    --seed-script "$SPB_ROOT/seed_stub.sh" \
    --main-ref main

assert "S-pressure11: exit 0" test "$RC" -eq 0
assert "S-pressure12: off the emergency path _lane-1 is PRESERVED with Block S's existing wording" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-1: assigned to task 5504 (state=assigned)"' _ "$ERR_OUT"
assert "S-pressure13: ...and its target/ and divergent marker are intact" \
    test -f "$SPB_ROOT/worktrees/_lane-2/target/DIVERGENT_MARKER" -a -f "$SPB_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "S-pressure14: _lane-2 preserved by the RECORD gate, not the /proc gate (record is still checked first off the emergency path)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-2: assigned to task 5505 (state=assigned)"' _ "$ERR_OUT"
assert "S-pressure15: no downgrade was reported for any lane" \
    bash -c '! printf "%s\n" "$1" | grep -qF "downgrading record gate"' _ "$ERR_OUT"
assert "S-pressure16: released _lane-3 still resets via the seed stub (the α branch, not the rm branch)" \
    bash -c 'test -f "$1" && grep -q "_lane-3" "$1" && [ ! -f "$2" ]' _ \
    "$SPB_ROOT/seed_calls.log" "$SPB_ROOT/worktrees/_lane-3/target/DIVERGENT_MARKER"
assert "S-pressure17: summary reports preserved_assigned=2 downgraded_assigned=0 (Block S's contract, untouched off the emergency path)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=1 removed=0 preserved=2 preserved_live_ref=0 preserved_assigned=2 downgraded_assigned=0"' _ "$OUT"

kill "$SPB_HELPER_PID" 2>/dev/null || true
wait "$SPB_HELPER_PID" 2>/dev/null || true
_BGPIDS=()  # clear so EXIT cleanup does not re-kill a possibly-reused PID

# ── S-age: an ASSIGNED record too OLD to believe is downgraded (task 5504) ────
# The chronic half of the same defect, on the ORDINARY (non-emergency) path.
#
# THE PRODUCER. A lane record is stranded `assigned` whenever the transition
# that should have cleared it never landed — the ENOSPC-at-release swallow, or
# a task that is still pending/in-progress/blocked and so is skipped BY DESIGN
# by harness.py's terminal-status reclaim. dark-factory already knows about
# such lanes: _stale_lane_assignment_census bounds exactly this field
# (`updated_at`) by exactly this predicate (age > a days threshold) and prints
# them under the digest's `## Stale lane assignments`. It is report-only by
# explicit design, so the age bound terminated in a digest line and never in a
# reclaim. This block pins the bound REUSED rather than re-derived: same field,
# same predicate, same documented handling of an unreadable value — different
# ACTION.
#
# Nothing here has a live process reference and every <lane>.lock is FREE, with
# no --extra-protect-glob, so neither the /proc gate nor the flock nor a glob
# can account for any outcome observed. The ONLY variable is record age.
echo ""
echo "--- Block S-age: the record's updated_at bounds an ASSIGNED preserve (task 5504) ---"

# 400 days against a 14-day default leaves a ~386-day margin, so no clock skew
# and no pass duration can flip the verdict. Computed ONCE so all three
# sub-cases judge the same record.
SA_STALE_TS="$(date -u -d '-400 days' +%Y-%m-%dT%H:%M:%S.000000+00:00)"

# _make_sa_fixture <root>
# The three-lane fixture all three sub-cases share, built identically each time
# so the flag/env knob under test is the only variable between them.
_make_sa_fixture() {
    local root="$1"
    mkdir -p "$root/worktrees" "$root/base"
    make_repo "$root/repo"
    mkdir -p "$root/base/target.gen.1"
    touch "$root/base/target.gen.1.lock"
    ln -sfn "$root/base/target.gen.1" "$root/base/target"
    local _sa_name
    for _sa_name in _lane-1 _lane-2 _lane-3; do
        git -C "$root/repo" worktree add -q "$root/worktrees/$_sa_name"
        mkdir -p "$root/worktrees/$_sa_name/target"
        touch "$root/worktrees/$_sa_name/target/DIVERGENT_MARKER"
    done
    # _lane-1: stale, via make_lane_state's 6th argument (the branch argument is
    # empty, the shape an unassigned-branch record carries).
    make_lane_state "$root/worktrees" _lane-1 assigned 5504 '' "$SA_STALE_TS"
    # _lane-2: the factory's default — stamped NOW. The discrimination control.
    make_lane_state "$root/worktrees" _lane-2 assigned 5505
    make_lane_state "$root/worktrees" _lane-3 released
    _seed_stub_body > "$root/seed_stub.sh"
    chmod +x "$root/seed_stub.sh"
}

# ── S-age-fires: the bound acts, and acts on ONE lane ─────────────────────────
SAF_ROOT="$(mktemp -d /tmp/test-gc-saf-XXXXXX)"
_TMPDIRS+=("$SAF_ROOT")
_make_sa_fixture "$SAF_ROOT"
export SEED_LOG="$SAF_ROOT/seed_calls.log"

run_helper reclaim \
    --worktrees-dir "$SAF_ROOT/worktrees" \
    --base-target "$SAF_ROOT/base/target" \
    --seed-script "$SAF_ROOT/seed_stub.sh" \
    --main-ref main

assert "S-age1: exit 0" test "$RC" -eq 0
assert "S-age2: the stale assigned _lane-1 IS reset (the seed stub ran for it and its marker is gone)" \
    bash -c 'test -f "$1" && grep -q "_lane-1" "$1" && [ ! -f "$2" ]' _ \
    "$SAF_ROOT/seed_calls.log" "$SAF_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "S-age3: stderr names _lane-1's downgrade with the record's own timestamp and the effective bound" \
    bash -c 'printf "%s\n" "$1" | grep -qF "downgrading record gate for _lane-1: record updated_at=$2 is 400d old, past the --max-record-age-days 14 bound"' _ "$ERR_OUT" "$SA_STALE_TS"
assert "S-age4: ...and the downgraded lane does NOT also print the ordinary assigned-preserve line" \
    bash -c '! printf "%s\n" "$1" | grep -qF "preserving _lane-1: assigned to task 5504"' _ "$ERR_OUT"
# DISCRIMINATION, not a blanket expiry: the two assigned lanes differ ONLY in
# their record's timestamp, and they take opposite branches in ONE pass.
assert "S-age5: the fresh assigned _lane-2 is PRESERVED with Block S's existing wording" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-2: assigned to task 5505 (state=assigned)"' _ "$ERR_OUT"
assert "S-age6: ...and its divergent marker is intact (the seed stub never ran for it)" \
    bash -c '[ -f "$1" ] && ([ ! -f "$2" ] || ! grep -q "_lane-2" "$2")' _ \
    "$SAF_ROOT/worktrees/_lane-2/target/DIVERGENT_MARKER" "$SAF_ROOT/seed_calls.log"
assert "S-age7: released _lane-3 still resets (non-vacuity: the pass ran the reset branch)" \
    bash -c 'grep -q "_lane-3" "$1" && [ ! -f "$2" ]' _ \
    "$SAF_ROOT/seed_calls.log" "$SAF_ROOT/worktrees/_lane-3/target/DIVERGENT_MARKER"
assert "S-age8: summary reports reset=2 with the two assigned lanes split 1 preserved / 1 downgraded" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=2 removed=0 preserved=1 preserved_live_ref=0 preserved_assigned=1 downgraded_assigned=1"' _ "$OUT"

# ── S-age-disabled: 0 is the documented escape hatch ──────────────────────────
# An operator who wants pre-5504 behaviour must have a way to get it that is
# explicit and visible in the invocation, not a source edit.
SAD_ROOT="$(mktemp -d /tmp/test-gc-sad-XXXXXX)"
_TMPDIRS+=("$SAD_ROOT")
_make_sa_fixture "$SAD_ROOT"
export SEED_LOG="$SAD_ROOT/seed_calls.log"

run_helper reclaim \
    --worktrees-dir "$SAD_ROOT/worktrees" \
    --base-target "$SAD_ROOT/base/target" \
    --seed-script "$SAD_ROOT/seed_stub.sh" \
    --main-ref main \
    --max-record-age-days 0

assert "S-age9: exit 0 under --max-record-age-days 0" test "$RC" -eq 0
assert "S-age10: with the bound disabled the 400-day-stale _lane-1 is PRESERVED again" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-1: assigned to task 5504 (state=assigned)"' _ "$ERR_OUT"
assert "S-age11: ...and its divergent marker is intact (0 really means no bound, not a large one)" \
    test -f "$SAD_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "S-age12: summary reports both assigned lanes preserved and none downgraded" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved_assigned=2 downgraded_assigned=0"' _ "$OUT"

# ── S-age-env: the knob is wired as every sibling REIFY_WARM_LANE_GC_* is ─────
# The flag's default comes from the env var, so a deployment can widen the bound
# without editing the sweep's argv. Asserted with a value that CHANGES the
# verdict (1000 > 400), so a knob that were read but ignored goes red.
SAE_ROOT="$(mktemp -d /tmp/test-gc-sae-XXXXXX)"
_TMPDIRS+=("$SAE_ROOT")
_make_sa_fixture "$SAE_ROOT"
export SEED_LOG="$SAE_ROOT/seed_calls.log"
export REIFY_WARM_LANE_GC_MAX_RECORD_AGE_DAYS=1000

run_helper reclaim \
    --worktrees-dir "$SAE_ROOT/worktrees" \
    --base-target "$SAE_ROOT/base/target" \
    --seed-script "$SAE_ROOT/seed_stub.sh" \
    --main-ref main

unset REIFY_WARM_LANE_GC_MAX_RECORD_AGE_DAYS

assert "S-age13: exit 0 under REIFY_WARM_LANE_GC_MAX_RECORD_AGE_DAYS=1000" test "$RC" -eq 0
assert "S-age14: a 1000-day bound preserves the 400-day-stale _lane-1 (the env knob is the flag's default)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-1: assigned to task 5504 (state=assigned)"' _ "$ERR_OUT"
assert "S-age15: ...and its divergent marker is intact" \
    test -f "$SAE_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "S-age16: summary reports both assigned lanes preserved and none downgraded" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved_assigned=2 downgraded_assigned=0"' _ "$OUT"

# ── S-age-degrade: the AGE fails SAFE where the STATE fails OPEN (task 5504) ──
# The two reads answer different questions, and the gate treats them in OPPOSITE
# directions on purpose:
#   an unreadable STATE  -> reclaim  (S15/S17 pin it: failing closed on a corrupt
#                                     record would freeze a lane out of reclaim
#                                     forever)
#   an unreadable AGE    -> preserve (this block)
# A readable `assigned` state is a trustworthy claim about STATE, so a bad
# timestamp leaves only the AGE unknown. Reclaiming on that would let a
# malformed field delete a genuinely live lane's build — strictly worse than one
# extra sweep of preservation. It is also verbatim the policy
# harness.py::_stale_lane_assignment_census already documents for the same
# field ("a record with an empty/unparseable `updated_at` is skipped"), which is
# the reuse this task is about.
#
# Arm B is the half that keeps the fail-safe from re-creating the very defect:
# under --disk-pressure an unreadable timestamp must NOT hold the valve shut.
# The chronic bound may fail closed; the acute valve never can.
echo ""
echo "--- Block S-age-degrade: an unjudgeable record age preserves, except under --disk-pressure (task 5504) ---"

# _make_sag_fixture <root>
# Both degraded records go through make_lane_state_raw, which exists for exactly
# the records make_lane_state cannot express. Both keep a VALID `state` and
# task_id, so the only thing wrong with them is the timestamp.
_make_sag_fixture() {
    local root="$1"
    mkdir -p "$root/worktrees" "$root/base"
    make_repo "$root/repo"
    mkdir -p "$root/base/target.gen.1"
    touch "$root/base/target.gen.1.lock"
    ln -sfn "$root/base/target.gen.1" "$root/base/target"
    local _sag_name
    for _sag_name in _lane-1 _lane-2 _lane-3; do
        git -C "$root/repo" worktree add -q "$root/worktrees/$_sag_name"
        mkdir -p "$root/worktrees/$_sag_name/target"
        touch "$root/worktrees/$_sag_name/target/DIVERGENT_MARKER"
    done
    # _lane-1: valid assigned record with NO `updated_at` key at all.
    make_lane_state_raw "$root/worktrees" _lane-1 \
        '{"state": "assigned", "task_id": "5504", "title": null, "branch": null, "seeded_from_sha": null}'
    # _lane-2: `updated_at` present but unparseable by `date -d`.
    make_lane_state_raw "$root/worktrees" _lane-2 \
        '{"state": "assigned", "task_id": "5505", "title": null, "branch": null, "seeded_from_sha": null, "updated_at": "not-a-timestamp"}'
    make_lane_state "$root/worktrees" _lane-3 released
    _seed_stub_body > "$root/seed_stub.sh"
    chmod +x "$root/seed_stub.sh"
}

# ── Arm A: the ordinary path — both degraded records PRESERVE, loudly ─────────
SAGA_ROOT="$(mktemp -d /tmp/test-gc-saga-XXXXXX)"
_TMPDIRS+=("$SAGA_ROOT")
_make_sag_fixture "$SAGA_ROOT"
export SEED_LOG="$SAGA_ROOT/seed_calls.log"

run_helper reclaim \
    --worktrees-dir "$SAGA_ROOT/worktrees" \
    --base-target "$SAGA_ROOT/base/target" \
    --seed-script "$SAGA_ROOT/seed_stub.sh" \
    --main-ref main

assert "S-age-degrade1: exit 0" test "$RC" -eq 0
assert "S-age-degrade2: fixture — both degraded records still read as ASSIGNED (only the timestamp is wrong)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "preserving _lane-1: assigned to task 5504" && printf "%s\n" "$1" | grep -qF "preserving _lane-2: assigned to task 5505"' _ "$ERR_OUT"
assert "S-age-degrade3: the record with NO updated_at key is PRESERVED, markers intact, seed stub never invoked" \
    bash -c '[ -f "$1" ] && ([ ! -f "$2" ] || ! grep -q "_lane-1" "$2")' _ \
    "$SAGA_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER" "$SAGA_ROOT/seed_calls.log"
assert "S-age-degrade4: the record with a GARBAGE updated_at is PRESERVED likewise" \
    bash -c '[ -f "$1" ] && ([ ! -f "$2" ] || ! grep -q "_lane-2" "$2")' _ \
    "$SAGA_ROOT/worktrees/_lane-2/target/DIVERGENT_MARKER" "$SAGA_ROOT/seed_calls.log"
# LOUD, per lane, and attributable. The channel is anomaly-only (a real producer
# always stamps the field), so it cannot train operators to ignore it — the same
# asymmetry argument the unparseable-record vs no-readable-record comment makes.
assert "S-age-degrade5: stderr says the bound could not be applied to _lane-1, naming the empty value" \
    bash -c 'printf "%s\n" "$1" | grep -qF "_lane-1: cannot judge lane-state record age (updated_at='"''"')"' _ "$ERR_OUT"
assert "S-age-degrade6: ...and names _lane-2's actual garbage value, not a generic message" \
    bash -c 'printf "%s\n" "$1" | grep -qF "_lane-2: cannot judge lane-state record age (updated_at='"'"'not-a-timestamp'"'"')"' _ "$ERR_OUT"
assert "S-age-degrade7: released _lane-3 still resets (non-vacuity: the pass ran the reset branch)" \
    bash -c 'grep -q "_lane-3" "$1" && [ ! -f "$2" ]' _ \
    "$SAGA_ROOT/seed_calls.log" "$SAGA_ROOT/worktrees/_lane-3/target/DIVERGENT_MARKER"
assert "S-age-degrade8: summary reports preserved_assigned=2 downgraded_assigned=0 (an unreadable AGE preserves)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved_assigned=2 downgraded_assigned=0"' _ "$OUT"

# ── Arm B: the SAME fixture under --disk-pressure — the valve is immune ───────
SAGB_ROOT="$(mktemp -d /tmp/test-gc-sagb-XXXXXX)"
_TMPDIRS+=("$SAGB_ROOT")
_make_sag_fixture "$SAGB_ROOT"
export SEED_LOG="$SAGB_ROOT/seed_calls.log"

run_helper reclaim \
    --worktrees-dir "$SAGB_ROOT/worktrees" \
    --base-target "$SAGB_ROOT/base/target" \
    --seed-script "$SAGB_ROOT/seed_stub.sh" \
    --main-ref main \
    --disk-pressure

assert "S-age-degrade9: exit 0" test "$RC" -eq 0
assert "S-age-degrade10: under --disk-pressure the no-updated_at record IS reclaimed" \
    bash -c '[ ! -d "$1" ]' _ "$SAGB_ROOT/worktrees/_lane-1/target"
assert "S-age-degrade11: ...and so is the garbage-updated_at record — no unreadable timestamp holds the ENOSPC valve shut" \
    bash -c '[ ! -d "$1" ]' _ "$SAGB_ROOT/worktrees/_lane-2/target"
assert "S-age-degrade12: both are attributed to the emergency override, not the staleness bound" \
    bash -c 'printf "%s\n" "$1" | grep -qF "downgrading record gate for _lane-1: --disk-pressure emergency override" && printf "%s\n" "$1" | grep -qF "downgrading record gate for _lane-2: --disk-pressure emergency override"' _ "$ERR_OUT"
# ORDERING, asserted rather than left to be inferred from line order in the
# source: the acute leg returns BEFORE the chronic leg runs, so the fail-safe's
# warn is never even reached under --disk-pressure.
assert "S-age-degrade13: the chronic leg never ran (no unjudgeable-age warn under --disk-pressure)" \
    bash -c '! printf "%s\n" "$1" | grep -qF "cannot judge lane-state record age"' _ "$ERR_OUT"
assert "S-age-degrade14: summary reports downgraded_assigned=2 preserved_assigned=0" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved_assigned=0 downgraded_assigned=2"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block V — skip a lane unchanged since its last seed on the current generation
# (task 7859)
#
# Resetting a lane nothing has built in since its seed frees nothing and costs a
# full α reseed. The project-owned seed stamp, <realpath lane>.seed-stamp.json
# (reify scripts/verify-ledger.py::write_seed_stamp, plus reify task 9611's
# base_gen), says when the lane was last seeded and from which gen; the mtimes of
# a fixed set of directories under <lane>/target say whether anything was written
# since. Only a lane for which every reading is trustworthy AND unchanged is
# skipped; every doubt resets, which is today's behaviour. The probe set and its
# blind spot: orchestrator/scripts/warm-lane/README.md "Delta 14".
#
# "Unchanged" is made deterministic with no sleep: every dir under target/ is aged
# to V_AGED and the stamp says the seed exited at V_EXITED, 50s later, so anything
# a case writes afterwards is newer than the seed's exit.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block V: skip a lane unchanged since its last seed on the current gen (task 7859) ---"

V_ROOT="$(mktemp -d /tmp/test-gc-v-XXXXXX)"
_TMPDIRS+=("$V_ROOT")
V_NOW="$(date +%s)"
V_AGED=$((V_NOW - 100))
V_EXITED=$((V_NOW - 50))

V_SEED_STUB="$V_ROOT/seed_stub.sh"
_seed_stub_body > "$V_SEED_STUB"
chmod +x "$V_SEED_STUB"

# make_seed_stamp LANE BASE_GEN EXITED_AT — write LANE's stamp where reify's
# seed_stamp_path puts it, in write_seed_stamp's one-line json.dumps shape plus
# base_gen. A BASE_GEN of `-` omits the key. EXITED_AT is interpolated as a RAW
# JSON value, so a case can pass `true` or a quoted string.
make_seed_stamp() {
    local lane="$1" base_gen="$2" exited_at="$3" real gen_field=''
    real="$(readlink -f -- "$lane")"
    if [ "$base_gen" != "-" ]; then
        gen_field=", \"base_gen\": \"$base_gen\""
    fi
    printf '{"schema": "reify.lane-seed/1", "seed_id": "0123456789abcdef", "lane_dir": "%s", "mode": "fresh-checkout", "started_at": %s, "exited_at": %s%s}\n' \
        "$real" "$((V_EXITED - 10))" "$exited_at" "$gen_field" > "$real.seed-stamp.json"
}

# make_seed_stamp_raw LANE TEXT — write arbitrary bytes as LANE's stamp.
make_seed_stamp_raw() {
    printf '%s' "$2" > "$(readlink -f -- "$1").seed-stamp.json"
}

# _make_v_lane ROOT — make_gc_fixture plus _lane-1, whose target/ carries the
# probed cargo layout and a DIVERGENT_MARKER (the "target/ was left alone"
# witness), with every directory under target/ aged to V_AGED.
_make_v_lane() {
    local root="$1" lane="$1/worktrees/_lane-1"
    make_gc_fixture "$root"
    git -C "$root/repo" worktree add -q "$lane"
    mkdir -p "$lane/target/debug/deps" "$lane/target/debug/.fingerprint" \
        "$lane/target/debug/build" "$lane/target/debug/incremental" "$lane/target/tmp"
    touch "$lane/target/DIVERGENT_MARKER"
    find "$lane/target" -type d -exec touch -d "@$V_AGED" {} +
}

# _current_gen ROOT — the exact string gc compares base_gen against.
_current_gen() { readlink -f "$1/base/target"; }

# _run_v ROOT [EXTRA-ARGS...] — reclaim ROOT's mount with the logging seed stub,
# which logs to ROOT/seed_calls.log.
_run_v() {
    local root="$1"; shift
    export SEED_LOG="$root/seed_calls.log"
    run_helper reclaim \
        --worktrees-dir "$root/worktrees" \
        --base-target "$root/base/target" \
        --seed-script "$V_SEED_STUB" "$@"
}

# _assert_v_reset LABEL ROOT — the outcome every case that must NOT skip shares.
_assert_v_reset() {
    local label="$1" root="$2"
    assert "$label: exit 0" test "$RC" -eq 0
    assert "$label: the seed WAS invoked for _lane-1" \
        bash -c 'test -f "$1" && grep -q "_lane-1" "$1"' _ "$root/seed_calls.log"
    assert "$label: summary counts the lane reset (reset=1 removed=0 preserved=0)" \
        bash -c 'printf "%s\n" "$1" | grep -qE "reset=1 removed=0 preserved=0"' _ "$OUT"
    assert "$label: summary shows skipped_unchanged=0" \
        bash -c 'printf "%s\n" "$1" | grep -qE " skipped_unchanged=0( |$)"' _ "$OUT"
}

# ── V1: unchanged since a seed from the current gen ⇒ skipped, target/ intact ──
V1_ROOT="$V_ROOT/v1"
_make_v_lane "$V1_ROOT"
make_seed_stamp "$V1_ROOT/worktrees/_lane-1" "$(_current_gen "$V1_ROOT")" "$V_EXITED"
_run_v "$V1_ROOT"

assert "V1a: exit 0" test "$RC" -eq 0
assert "V1b: the seed was NOT invoked — an unchanged lane has nothing to reclaim" \
    bash -c '[ ! -s "$1" ]' _ "$V1_ROOT/seed_calls.log"
assert "V1c: target/ is untouched (divergent marker still present)" \
    test -f "$V1_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "V1d: the lane is in neither reset nor preserved (reset=0 removed=0 preserved=0)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=0 removed=0 preserved=0"' _ "$OUT"
assert "V1e: summary carries the appended skipped_unchanged=1" \
    bash -c 'printf "%s\n" "$1" | grep -qE " skipped_unchanged=1( |$)"' _ "$OUT"
assert "V1f: an [info] line names _lane-1 as skipped unchanged" \
    bash -c 'printf "%s\n" "$1" | grep -F "[info]" | grep -F "skipping unchanged lane" | grep -qF "_lane-1"' _ "$ERR_OUT"

# ── V2: anything written under a probed dir since the seed's exit ⇒ reset ─────
# One arm per probed shape: a new artifact in deps/ (every compile — rustc
# replaces its outputs), a new unit dir in .fingerprint/ (cargo check), a new file
# directly under a profile-level dir (target/tmp, CARGO_TARGET_TMPDIR), and the
# BOUNDARY: a mtime EXACTLY equal to exited_at resets, because only strictly older
# proves the write preceded the seed's exit.
for _v2_arm in deps-file fingerprint-dir tmp-file boundary; do
    _v2_root="$V_ROOT/v2-$_v2_arm"
    _v2_lane="$_v2_root/worktrees/_lane-1"
    _make_v_lane "$_v2_root"
    make_seed_stamp "$_v2_lane" "$(_current_gen "$_v2_root")" "$V_EXITED"
    case "$_v2_arm" in
        deps-file)       touch "$_v2_lane/target/debug/deps/libnew-0123.rlib" ;;
        fingerprint-dir) mkdir "$_v2_lane/target/debug/.fingerprint/new-unit-0123" ;;
        tmp-file)        touch "$_v2_lane/target/tmp/written-by-a-test" ;;
        boundary)        touch -d "@$V_EXITED" "$_v2_lane/target/debug/deps" ;;
    esac
    _run_v "$_v2_root"
    _assert_v_reset "V2 ($_v2_arm)" "$_v2_root"
done

# ── V3: the base generation moved since the seed ⇒ reset, from the NEW gen ────
V3_ROOT="$V_ROOT/v3"
_make_v_lane "$V3_ROOT"
make_seed_stamp "$V3_ROOT/worktrees/_lane-1" "$(_current_gen "$V3_ROOT")" "$V_EXITED"
mkdir "$V3_ROOT/base/target.gen.2"
touch "$V3_ROOT/base/target.gen.2.lock"
ln -sfn "$V3_ROOT/base/target.gen.2" "$V3_ROOT/base/target"
_run_v "$V3_ROOT"

_assert_v_reset "V3 (gen moved)" "$V3_ROOT"
assert "V3: the seed received target.gen.2, the gen live now" \
    bash -c 'grep -q "target\.gen\.2" "$1"' _ "$V3_ROOT/seed_calls.log"

# ── V4: no stamp ⇒ exactly today's behaviour, and no skip-gate diagnostic ─────
# The ordinary reading for every project that writes no stamp: the eight
# non-reify units. (a) with a seed, the lane resets as in test_warm_lane_gc.sh's
# Block B. (b) their production shape, `--mount` with no --seed-script and no
# sibling seed: the lane is preserved by the seed-presence guard as in
# test_warm_lane_gc.sh's S-discard5c, neither skipped nor failing the pass.
V4A_ROOT="$V_ROOT/v4a"
_make_v_lane "$V4A_ROOT"
_run_v "$V4A_ROOT"

_assert_v_reset "V4a (no stamp, seed present)" "$V4A_ROOT"
assert "V4a: the seed thinned target/ (divergent marker removed), as in Block B" \
    bash -c '[ ! -e "$1" ]' _ "$V4A_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "V4a: a missing stamp is quiet — no skip-gate line on stderr" \
    bash -c '! printf "%s\n" "$1" | grep -qE "skipping unchanged lane|not skipped"' _ "$ERR_OUT"

V4B_ROOT="$V_ROOT/v4b"
_make_v_lane "$V4B_ROOT"
assert "V4b: fixture — the sibling default seed-warm-lane.sh really is absent" \
    bash -c '[ ! -e "$1" ]' _ "$WARM_LANE_SCRIPTS_DIR/seed-warm-lane.sh"
REIFY_WARM_LANE_GC_SEED_SCRIPT='' run_helper reclaim --mount "$V4B_ROOT/worktrees"

assert "V4b: exit 0" test "$RC" -eq 0
assert "V4b: target/ is untouched (divergent marker still present)" \
    test -f "$V4B_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "V4b: the lane is preserved with the 'seed-script error' warn" \
    bash -c 'printf "%s\n" "$1" | grep -F "[warn]" | grep -qF "reset failed for _lane-1 (seed-script error)"' _ "$ERR_OUT"
assert "V4b: summary line present, the lane preserved and not skipped (preserved=1, skipped_unchanged=0)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^reclaim: reset=0 removed=0 preserved=1 .* skipped_unchanged=0( |$)"' _ "$OUT"
assert "V4b: a missing stamp is quiet — no skip-gate line on stderr" \
    bash -c '! printf "%s\n" "$1" | grep -qE "skipping unchanged lane|not skipped"' _ "$ERR_OUT"

# ── V5: an unreadable or untrustworthy stamp ⇒ reset ──────────────────────────
# Each arm is otherwise a V1 skip, so the one named defect is the only thing that
# can explain a reset. `symlink-spelled` matters most: a stamp naming base/target
# rather than the concrete gen must NOT be normalised, or every such stamp would
# read as current. It is spelled as realpath(base) + /target, so a realpath
# normalisation would land exactly on the current gen and skip.
#
# `exited-at-nan` is the one arm the finiteness check alone stands behind: json
# accepts a bare NaN, it is not in the future, and every `mtime >= NaN` is
# False, so without that check a NaN stamp reads as unchanged and SKIPS.
# `exited-at-inf` is also refused as a future exit. `python3-absent` puts a
# PATH stub first that answers as bash does for a missing command (rc 127), so
# the probe cannot run at all; its log is the witness that the stub was reached.
V5_NO_PY_DIR="$V_ROOT/v5-no-python3-bin"
mkdir -p "$V5_NO_PY_DIR"
cat > "$V5_NO_PY_DIR/python3" << 'STUB_EOF'
#!/usr/bin/env bash
echo "$*" >> "${0%/*}/calls.log"
echo "python3: command not found (test stub)" >&2
exit 127
STUB_EOF
chmod +x "$V5_NO_PY_DIR/python3"

for _v5_arm in truncated not-an-object schema-v2 no-base-gen empty-base-gen \
               symlink-spelled exited-at-string exited-at-bool exited-at-future \
               exited-at-nan exited-at-inf python3-absent; do
    _v5_root="$V_ROOT/v5-$_v5_arm"
    _v5_lane="$_v5_root/worktrees/_lane-1"
    _v5_path="$PATH"
    _make_v_lane "$_v5_root"
    _v5_gen="$(_current_gen "$_v5_root")"
    case "$_v5_arm" in
        truncated)        make_seed_stamp_raw "$_v5_lane" '{"schema": "reify.lane-seed/1", "exited_at": 1' ;;
        not-an-object)    make_seed_stamp_raw "$_v5_lane" '[]' ;;
        schema-v2)        make_seed_stamp_raw "$_v5_lane" "{\"schema\": \"reify.lane-seed/2\", \"exited_at\": $V_EXITED, \"base_gen\": \"$_v5_gen\"}" ;;
        no-base-gen)      make_seed_stamp "$_v5_lane" - "$V_EXITED" ;;
        empty-base-gen)   make_seed_stamp "$_v5_lane" "" "$V_EXITED" ;;
        symlink-spelled)  make_seed_stamp "$_v5_lane" "$(readlink -f "$_v5_root/base")/target" "$V_EXITED" ;;
        exited-at-string) make_seed_stamp "$_v5_lane" "$_v5_gen" "\"$V_EXITED\"" ;;
        exited-at-bool)   make_seed_stamp "$_v5_lane" "$_v5_gen" true ;;
        exited-at-future) make_seed_stamp "$_v5_lane" "$_v5_gen" "$((V_NOW + 3600))" ;;
        exited-at-nan)    make_seed_stamp "$_v5_lane" "$_v5_gen" NaN ;;
        exited-at-inf)    make_seed_stamp "$_v5_lane" "$_v5_gen" Infinity ;;
        python3-absent)   make_seed_stamp "$_v5_lane" "$_v5_gen" "$V_EXITED"
                          _v5_path="$V5_NO_PY_DIR:$PATH" ;;
    esac
    PATH="$_v5_path" _run_v "$_v5_root"
    _assert_v_reset "V5 ($_v5_arm)" "$_v5_root"
done
assert "V5 (python3-absent): the stamp probe reached the PATH stub, not a real python3" \
    bash -c 'grep -qF "_lane-1.seed-stamp.json" "$1"' _ "$V5_NO_PY_DIR/calls.log"

# ── V6: a valid stamp but no target/ ⇒ reset ──────────────────────────────────
V6_ROOT="$V_ROOT/v6"
_make_v_lane "$V6_ROOT"
rm -rf "$V6_ROOT/worktrees/_lane-1/target"
make_seed_stamp "$V6_ROOT/worktrees/_lane-1" "$(_current_gen "$V6_ROOT")" "$V_EXITED"
_run_v "$V6_ROOT"

_assert_v_reset "V6 (no target/)" "$V6_ROOT"

# ── V7: --disk-pressure bypasses the skip — the emergency path is unchanged ───
V7_ROOT="$V_ROOT/v7"
_make_v_lane "$V7_ROOT"
make_seed_stamp "$V7_ROOT/worktrees/_lane-1" "$(_current_gen "$V7_ROOT")" "$V_EXITED"
_run_v "$V7_ROOT" --disk-pressure

assert "V7: exit 0" test "$RC" -eq 0
assert "V7: target/ is deleted outright — the emergency rm ran on an unchanged lane" \
    bash -c '[ ! -e "$1" ]' _ "$V7_ROOT/worktrees/_lane-1/target"
assert "V7: summary counts the lane reset, not skipped (reset=1, skipped_unchanged=0)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^reclaim: reset=1 removed=0 preserved=0 .* skipped_unchanged=0( |$)"' _ "$OUT"

# ── V8: an assigned record is decided BEFORE the skip ─────────────────────────
# The record gate keeps its authoritative preserved_assigned attribution; a lane
# it preserves never reaches the skip.
V8_ROOT="$V_ROOT/v8"
_make_v_lane "$V8_ROOT"
make_seed_stamp "$V8_ROOT/worktrees/_lane-1" "$(_current_gen "$V8_ROOT")" "$V_EXITED"
make_lane_state "$V8_ROOT/worktrees" _lane-1 assigned 7859
_run_v "$V8_ROOT"

assert "V8: exit 0" test "$RC" -eq 0
assert "V8: the record preserves the lane (preserved=1 preserved_live_ref=0 preserved_assigned=1)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "reset=0 removed=0 preserved=1 preserved_live_ref=0 preserved_assigned=1"' _ "$OUT"
assert "V8: ...and it is not also counted skipped (skipped_unchanged=0)" \
    bash -c 'printf "%s\n" "$1" | grep -qE " skipped_unchanged=0( |$)"' _ "$OUT"

# ── V9: a mount reached through a symlink still finds the realpath stamp ──────
V9_ROOT="$V_ROOT/v9"
_make_v_lane "$V9_ROOT"
make_seed_stamp "$V9_ROOT/worktrees/_lane-1" "$(_current_gen "$V9_ROOT")" "$V_EXITED"
ln -s "$V9_ROOT/worktrees" "$V9_ROOT/worktrees-link"
export SEED_LOG="$V9_ROOT/seed_calls.log"
run_helper reclaim \
    --worktrees-dir "$V9_ROOT/worktrees-link" \
    --base-target "$V9_ROOT/base/target" \
    --seed-script "$V_SEED_STUB"

assert "V9: exit 0" test "$RC" -eq 0
assert "V9: the seed was NOT invoked" \
    bash -c '[ ! -s "$1" ]' _ "$V9_ROOT/seed_calls.log"
assert "V9: target/ is untouched (divergent marker still present)" \
    test -f "$V9_ROOT/worktrees/_lane-1/target/DIVERGENT_MARKER"
assert "V9: summary carries skipped_unchanged=1" \
    bash -c 'printf "%s\n" "$1" | grep -qE " skipped_unchanged=1( |$)"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block X — the PROTECT_GLOB default is RENDERED from dark-factory's registry
# (task 3292)
#
# Until now gc.sh's default protect-glob was a hardcoded literal carrying a
# comment that admitted it mirrored PROTECTED_PREFIXES by hand — an INV-5
# lockstep duplication whose failure mode is silent: a band added to the
# registry that nobody mirrors here becomes a live managed worktree Pass 2 will
# happily `git worktree remove --force`. lib_lane_state.sh's lane_protect_glob
# shipped the faithful render ahead of any consumer; this block pins gc.sh
# actually consuming it.
#
# Two sub-cases, covering the two halves of lane_protect_glob's contract:
#
#   X-band    — the OBSERVABLE PAYOFF. The interactive band is the one registry
#               key that is not a constant (it is dark-factory's `git.iact_prefix`
#               config field), so a deployment that renamed it had a hardcoded
#               `_iact-*` protecting a band it never mints while OMITTING the
#               band it does — the dangerous direction. With
#               REIFY_WARM_LANE_IACT_PREFIX honoured, the rendered glob tracks
#               the deployment. Asserted in BOTH directions: the renamed band is
#               protected AND the stock `_iact-*` band is no longer.
#   X-degrade — the `||` FALLBACK, pinned in situ rather than as a contract
#               claim. lane_protect_glob fails LOUD (one [warn], empty stdout,
#               non-zero) and gc.sh runs `set -euo pipefail`, so a caller that
#               omitted the `||` would ABORT a reclaim sweep instead of
#               degrading it. Driving a relocated copy — from which the repo
#               root resolves to `/`, which carries no dark-factory checkout —
#               is what makes the degrade branch reachable hermetically.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block X: PROTECT_GLOB default rendered from the registry (task 3292) ---"

# ── X-band: REIFY_WARM_LANE_IACT_PREFIX moves the protected interactive band ──
# DF-faithful invocation (--mount only, no --protect-glob) via
# run_helper_live_default, because this sub-case is ABOUT the default.
XB_ROOT="$(mktemp -d /tmp/test-gc-xb-XXXXXX)"
_TMPDIRS+=("$XB_ROOT")

XB_REPO="$XB_ROOT/repo"
XB_WORKTREES="$XB_ROOT/worktrees"
XB_BASE="$XB_ROOT/base"
mkdir -p "$XB_WORKTREES" "$XB_BASE"

make_repo "$XB_REPO"

mkdir -p "$XB_BASE/target.gen.1"
touch "$XB_BASE/target.gen.1.lock"
ln -sfn "$XB_BASE/target.gen.1" "$XB_BASE/target"

# _myiact-live — an interactive worktree of THIS deployment's renamed band.
# Clean and landed, i.e. indistinguishable from a cold orphan to every gate
# EXCEPT the protect glob: only the rendered band can explain its survival.
git -C "$XB_REPO" worktree add -q "$XB_WORKTREES/_myiact-live"
touch "$XB_WORKTREES/_myiact-live/PROTECTED_MARKER"

# _iact-stale — the STOCK band this deployment does not mint. The old hardcoded
# default protected it; a faithful render for this deployment must not.
git -C "$XB_REPO" worktree add -q "$XB_WORKTREES/_iact-stale"

XB_SEED_LOG="$XB_ROOT/seed_calls.log"
XB_SEED_STUB="$XB_ROOT/seed_stub.sh"
_seed_stub_body > "$XB_SEED_STUB"
chmod +x "$XB_SEED_STUB"
export SEED_LOG="$XB_SEED_LOG"

# Inline command prefix, not an `export`: the band override must reach gc.sh's
# child (and through it lane_protect_glob's python) for THIS call only, leaving
# the test shell's environment clean for every later block.
REIFY_WARM_LANE_IACT_PREFIX="_myiact-" run_helper_live_default reclaim \
    --mount "$XB_WORKTREES" \
    --seed-script "$XB_SEED_STUB"

assert "X-band1: exit 0" test "$RC" -eq 0
assert "X-band2: _myiact-live preserved (the deployment's OWN interactive band is protected)" \
    test -d "$XB_WORKTREES/_myiact-live"
assert "X-band3: _myiact-live PROTECTED_MARKER intact" \
    test -f "$XB_WORKTREES/_myiact-live/PROTECTED_MARKER"
assert "X-band4: _iact-stale removed (the band this deployment never mints is NOT protected)" \
    bash -c '[ ! -d "$1" ]' _ "$XB_WORKTREES/_iact-stale"
assert "X-band5: _iact-stale absent from git worktree list" \
    bash -c '! git -C "$1" worktree list | grep -q "_iact-stale"' _ "$XB_REPO"
assert "X-band6: summary shows preserved=1 (only the renamed band)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "preserved=1"' _ "$OUT"
assert "X-band7: summary shows removed=1 (the stale stock band)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "removed=1"' _ "$OUT"

# ── X-degrade: a failed render degrades to the fallback, LOUDLY, without ──────
# ── aborting the sweep ────────────────────────────────────────────────────────
# Fixture idiom is A9/A10's (test_warm_lane_gc.sh's A9 and Block A's A10 here:
# copy the shipped script plus selected siblings into a temp dir), aimed
# differently: A9/A10 WITHHOLD a lib to reach gc.sh's own sourcing guard,
# whereas this relocates the copy so lib_lane_state.sh's environment-free root
# arithmetic (`<lib dir>/../../..`) lands on `/`, whose missing
# orchestrator/src/orchestrator/git_ops.py witness makes the bridge return
# non-zero. gc.sh does NOT source lib_portable.sh (that is
# warm-lane-audit.sh's), so the three siblings copied here are exactly the three
# it needs.
#
# The mount is deliberately MINIMAL — one protected entry, one control orphan,
# no pool lanes and no .lane-state records — so that the bridge is the only
# thing in the run that can emit a [warn], which is what makes "exactly one"
# an assertion about the degrade rather than about arithmetic.
#
# NOTE: no REIFY_WARM_LANE_GC_PROTECT_GLOB pin here, and $SCRIPT is not used, so
# run_helper is not applicable — the point is to reach the resolution path in a
# relocated copy, with its own rc/stderr capture.
XD_DIR="$(mktemp -d /tmp/test-gc-x-XXXXXX)"
_TMPDIRS+=("$XD_DIR")
cp "$SCRIPT" "$XD_DIR/warm-lane-gc.sh"
cp "$WARM_LANE_SCRIPTS_DIR/lib_live_refs.sh" "$XD_DIR/lib_live_refs.sh"
cp "$WARM_LANE_SCRIPTS_DIR/lib_lane_state.sh" "$XD_DIR/lib_lane_state.sh"
cp "$WARM_LANE_SCRIPTS_DIR/lib_lane_seed.sh" "$XD_DIR/lib_lane_seed.sh"

assert "X-degrade: fixture — all three siblings gc.sh sources are present (so no sourcing guard fires)" \
    bash -c '[ -f "$1/lib_live_refs.sh" ] && [ -f "$1/lib_lane_state.sh" ] && [ -f "$1/lib_lane_seed.sh" ]' _ "$XD_DIR"
assert "X-degrade: fixture — no dark-factory checkout at the relocated copy's resolved root (the bridge MUST degrade)" \
    bash -c '[ ! -f "$(cd "$1/../../.." && pwd)/orchestrator/src/orchestrator/git_ops.py" ]' _ "$XD_DIR"

XD_ROOT="$(mktemp -d /tmp/test-gc-xd-XXXXXX)"
_TMPDIRS+=("$XD_ROOT")

XD_REPO="$XD_ROOT/repo"
XD_WORKTREES="$XD_ROOT/worktrees"
XD_BASE="$XD_ROOT/base"
mkdir -p "$XD_WORKTREES" "$XD_BASE"

make_repo "$XD_REPO"

mkdir -p "$XD_BASE/target.gen.1"
touch "$XD_BASE/target.gen.1.lock"
ln -sfn "$XD_BASE/target.gen.1" "$XD_BASE/target"

# _merge-verify — covered by the FALLBACK (both as `_merge-*` and by name).
# Its survival is what proves the fallback was substituted rather than an EMPTY
# glob: an empty glob returned with a zero exit reads downstream as "nothing is
# protected", the silent widening lib_lane_state.sh's contract exists to block.
git -C "$XD_REPO" worktree add -q "$XD_WORKTREES/_merge-verify"
touch "$XD_WORKTREES/_merge-verify/PROTECTED_MARKER"

# Control orphan — must still be removed, proving the sweep RAN to completion
# rather than aborting early under `set -euo pipefail`.
git -C "$XD_REPO" worktree add -q "$XD_WORKTREES/task-7777"

XD_SEED_LOG="$XD_ROOT/seed_calls.log"
XD_SEED_STUB="$XD_ROOT/seed_stub.sh"
_seed_stub_body > "$XD_SEED_STUB"
chmod +x "$XD_SEED_STUB"
export SEED_LOG="$XD_SEED_LOG"

XD_ERR_FILE="$(mktemp /tmp/test-gc-xd-err-XXXXXX)"
_TMPDIRS+=("$XD_ERR_FILE")
XD_RC=0
bash "$XD_DIR/warm-lane-gc.sh" reclaim \
    --mount "$XD_WORKTREES" \
    --seed-script "$XD_SEED_STUB" \
    >/dev/null 2>"$XD_ERR_FILE" || XD_RC=$?
XD_ERR="$(cat "$XD_ERR_FILE")"

assert "X-degrade1: exit 0 — the || degraded the failed render instead of aborting under set -e" \
    test "$XD_RC" -eq 0
assert "X-degrade2: stderr carries EXACTLY ONE [warn] line" \
    bash -c '[ "$(printf "%s\n" "$1" | grep -c "\[warn\]")" -eq 1 ]' _ "$XD_ERR"
assert "X-degrade3: ...and it is the lane_protect_glob DEGRADED warning (fail LOUD, not silent)" \
    bash -c 'printf "%s\n" "$1" | grep -qF "lane_protect_glob DEGRADED"' _ "$XD_ERR"
assert "X-degrade4: _merge-verify preserved (LANE_PROTECT_GLOB_FALLBACK substituted, not an empty glob)" \
    test -d "$XD_WORKTREES/_merge-verify"
assert "X-degrade5: _merge-verify PROTECTED_MARKER intact" \
    test -f "$XD_WORKTREES/_merge-verify/PROTECTED_MARKER"
assert "X-degrade6: control orphan task-7777 removed (the sweep completed, it did not exit early)" \
    bash -c '[ ! -d "$1" ]' _ "$XD_WORKTREES/task-7777"
assert "X-degrade7: control orphan task-7777 absent from git worktree list" \
    bash -c '! git -C "$1" worktree list | grep -q "task-7777"' _ "$XD_REPO"

# ─────────────────────────────────────────────────────────────────────────────
# Block TRASH: shared-trash litter guard (task 5612). Two asserts, deliberately
# kept as two independently-reported signals: TRASH2 can realistically only ever
# report "clean", which is indistinguishable from a checker that stopped working
# — TRASH1 is the hermetic control proving the instrument still fires.
# Full rationale and honest scope: the CANONICAL WIRING CONTRACT comment in
# test_helpers.sh.
# ─────────────────────────────────────────────────────────────────────────────
assert "TRASH1: shared-trash litter detector is live (self-test fires on a synthetic bare-/tmp lane)" \
    assert_shared_trash_litter_detector_live
assert "TRASH2: no lane in this suite littered the machine-shared /tmp/.reseed-trash" \
    assert_no_shared_trash_litter

test_summary
