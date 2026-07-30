#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_warm_lane_degenerate_ref.sh
# Hermetic tests for orchestrator/scripts/warm-lane/warm-lane-degenerate-ref-check.sh (task #5006).
#
# run_helper captures STDOUT, STDERR, and RC separately:
#   OUT     — captured stdout from the script
#   ERR_OUT — captured stderr from the script
#   RC      — exit code
#
# build_fixture inits a hermetic `git init -b main` repo (no linked worktree
# needed — the classifier only reads refs + commit ancestry/messages, never a
# working tree) and the fixture_* helpers below add task branches at the
# three tip shapes the classifier must distinguish:
#   fixture_branch_at_foreign_merge — tip cites a DIFFERENT task id
#                                      (the degenerate-ref shape)
#   fixture_branch_at_self_merge    — tip cites its OWN task id (landed)
#   fixture_branch_own_commit       — one commit of its own ahead of main (live)
#
# Blocks (added incrementally across task #5006's TDD steps):
#   step-1  — arg-parsing / usage taxonomy
#   step-3  — structural-error taxonomy (exit 3)
#   step-5  — single-ref DEGENERATE classification + read-only invariant
#   step-7  — single-ref LIVE/LANDED/ABSENT + substring-safety
#   step-9  — fleet-audit mode (no status oracle)
#   step-11 — fleet-audit status-oracle (advisory filter)
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (PORTED_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
SCRIPT="$WARM_LANE_SCRIPTS_DIR/warm-lane-degenerate-ref-check.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=tests/infra/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"

echo "=== scripts/warm-lane-degenerate-ref-check.sh hermetic tests (task 5006) ==="

# ─────────────────────────────────────────────────────────────────────────────
# Shared temp state
# ─────────────────────────────────────────────────────────────────────────────
_TMPDIRS=()
cleanup() {
    for d in "${_TMPDIRS[@]+${_TMPDIRS[@]}}"; do rm -rf "$d"; done
}
trap cleanup EXIT

ERR_FILE="$(mktemp /tmp/test-warm-lane-degen-ref-err-XXXXXX)"
_TMPDIRS+=("$ERR_FILE")

# ── run_helper ────────────────────────────────────────────────────────────────
# Invokes the script under test with no PATH stub.
# Sets OUT (stdout), ERR_OUT (stderr), RC (exit code) as globals.
run_helper() {
    local rc=0
    > "$ERR_FILE"
    OUT="$(bash "$SCRIPT" "$@" 2>"$ERR_FILE")" || rc=$?
    ERR_OUT="$(cat "$ERR_FILE")"
    RC=$rc
}

# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────

# build_fixture <dir>
# Inits a hermetic main-checkout git repo at <dir>: `git init -b main`, test
# user config, and one initial commit. Task branches are added directly to
# this repo's refs/heads/ namespace by the fixture_* helpers below.
build_fixture() {
    local dir="$1"
    git init -q -b main "$dir"
    git -C "$dir" config user.email "test@test.local"
    git -C "$dir" config user.name "Test"
    git -C "$dir" commit -q --allow-empty -m "initial"
}

# fixture_merge_commit <repo> <cite_id> [prefix]
# Creates a throwaway side branch, commits once, merges it into main with an
# explicit "Merge <prefix><cite_id> into main" message, then deletes the side
# branch. Leaves `main` checked out. Prints the resulting merge-commit SHA on
# stdout — ONLY the SHA; all git chatter is redirected away so callers can
# capture it via command substitution.
fixture_merge_commit() {
    local repo="$1" cite_id="$2" prefix="${3:-task/}"
    local side="_fixture-side-${cite_id}-$$-${RANDOM}"
    git -C "$repo" checkout -q -b "$side" main >/dev/null
    git -C "$repo" commit -q --allow-empty -m "work for ${prefix}${cite_id}" >/dev/null
    git -C "$repo" checkout -q main >/dev/null
    git -C "$repo" merge -q --no-ff -m "Merge ${prefix}${cite_id} into main" "$side" >/dev/null
    git -C "$repo" branch -D "$side" >/dev/null
    git -C "$repo" rev-parse HEAD
}

# fixture_branch_at_foreign_merge <repo> <branch_task_id> <cited_task_id> [prefix]
# Points refs/heads/<prefix><branch_task_id> at a NEW merge commit that cites
# a DIFFERENT id — the degenerate-ref shape: parked on a foreign
# main-ancestor, zero of its own commits, tip does not cite its own id.
fixture_branch_at_foreign_merge() {
    local repo="$1" branch_id="$2" cited_id="$3" prefix="${4:-task/}"
    local sha
    sha="$(fixture_merge_commit "$repo" "$cited_id" "$prefix")"
    git -C "$repo" branch -q "${prefix}${branch_id}" "$sha"
}

# fixture_branch_at_self_merge <repo> <task_id> [prefix]
# Points refs/heads/<prefix><task_id> at a NEW merge commit that cites its
# OWN id — the landed shape.
fixture_branch_at_self_merge() {
    local repo="$1" task_id="$2" prefix="${3:-task/}"
    local sha
    sha="$(fixture_merge_commit "$repo" "$task_id" "$prefix")"
    git -C "$repo" branch -q "${prefix}${task_id}" "$sha"
}

# fixture_branch_own_commit <repo> <task_id> [prefix]
# Creates refs/heads/<prefix><task_id> with ONE commit of its own ahead of
# main (count>0) — the live shape. Leaves `main` checked out.
fixture_branch_own_commit() {
    local repo="$1" task_id="$2" prefix="${3:-task/}"
    git -C "$repo" checkout -q -b "${prefix}${task_id}" main >/dev/null
    git -C "$repo" commit -q --allow-empty -m "own work for ${prefix}${task_id}" >/dev/null
    git -C "$repo" checkout -q main >/dev/null
}

# ─────────────────────────────────────────────────────────────────────────────
# step-1 — arg-parsing / usage taxonomy
#
# U1 — no mode (neither --task nor --audit)   -> exit 2, usage on stderr
# U2 — --help                                  -> exit 0, usage on stderr
# U3 — -h                                      -> exit 0, usage on stderr
# U4 — unknown flag                            -> exit 2
# U5 — --task with no value                    -> exit 2
# U6 — --task N and --audit together           -> exit 2 (mutually exclusive)
# U7 — --task with a non-numeric id            -> exit 2
# All error cases (U1, U4-U7): empty stdout.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- step-1: arg-parsing / usage taxonomy ---"

run_helper
assert "U1: no mode exits 2" test "$RC" -eq 2
assert "U1: stderr carries usage" \
    bash -c 'printf "%s\n" "$1" | grep -qi "usage"' _ "$ERR_OUT"
assert "U1: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

run_helper --help
assert "U2: --help exits 0" test "$RC" -eq 0
assert "U2: stderr carries usage" \
    bash -c 'printf "%s\n" "$1" | grep -qi "usage"' _ "$ERR_OUT"
assert "U2: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

run_helper -h
assert "U3: -h exits 0" test "$RC" -eq 0
assert "U3: stderr carries usage" \
    bash -c 'printf "%s\n" "$1" | grep -qi "usage"' _ "$ERR_OUT"
assert "U3: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

run_helper --bogus-flag
assert "U4: unknown flag exits 2" test "$RC" -eq 2
assert "U4: stderr is non-empty" bash -c '[ -n "$1" ]' _ "$ERR_OUT"
assert "U4: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

run_helper --task
assert "U5: --task with no value exits 2" test "$RC" -eq 2
assert "U5: stderr is non-empty" bash -c '[ -n "$1" ]' _ "$ERR_OUT"
assert "U5: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

run_helper --task 123 --audit
assert "U6: --task and --audit together exits 2" test "$RC" -eq 2
assert "U6: stderr is non-empty" bash -c '[ -n "$1" ]' _ "$ERR_OUT"
assert "U6: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

run_helper --task abc
assert "U7: --task with non-numeric id exits 2" test "$RC" -eq 2
assert "U7: stderr is non-empty" bash -c '[ -n "$1" ]' _ "$ERR_OUT"
assert "U7: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# step-3 — structural-error taxonomy (exit 3)
#
# S1 — --repo is NOT inside a git work tree -> exit 3, empty stdout
# S2 — unresolvable --main-ref against a valid repo -> exit 3, empty stdout
# Exit 3 is distinct from usage (2) and from any classification code.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- step-3: structural-error taxonomy (exit 3) ---"

S3_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-s3-XXXXXX)"
_TMPDIRS+=("$S3_TMP")

# S1: --repo is NOT inside a git work tree -> exit 3
S3_NOT_A_REPO="$S3_TMP/not-a-repo"
mkdir -p "$S3_NOT_A_REPO"
run_helper --task 1 --repo "$S3_NOT_A_REPO"
assert "S1: not-a-git-work-tree exits 3" test "$RC" -eq 3
assert "S1: exit code 3 is distinct from usage (2)" bash -c '[ "$1" -ne 2 ]' _ "$RC"
assert "S1: stderr names the work-tree/provisioning condition" \
    bash -c 'printf "%s\n" "$1" | grep -qi "work tree\|worktree\|git repo\|not a git"' _ "$ERR_OUT"
assert "S1: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

# S2: unresolvable --main-ref against a valid repo -> exit 3
S3_REPO="$S3_TMP/repo"
build_fixture "$S3_REPO"
run_helper --task 1 --repo "$S3_REPO" --main-ref does/not/exist
assert "S2: unresolvable --main-ref exits 3" test "$RC" -eq 3
assert "S2: exit code 3 is distinct from usage (2)" bash -c '[ "$1" -ne 2 ]' _ "$RC"
assert "S2: stderr names the main-ref condition" \
    bash -c 'printf "%s\n" "$1" | grep -qi "main-ref\|main ref\|does/not/exist"' _ "$ERR_OUT"
assert "S2: stdout is empty" bash -c '[ -z "$1" ]' _ "$OUT"

# ─────────────────────────────────────────────────────────────────────────────
# step-5 — single-ref DEGENERATE classification + read-only invariant
#
# D1 — task/9001 parked on a FOREIGN merge (cites 8888, not 9001; count==0
#      vs main) -> exit 0, stdout "degenerate <tip sha>", and git show-ref is
#      byte-identical before/after (the classifier never mutates a ref).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- step-5: single-ref DEGENERATE classification ---"

D5_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-d5-XXXXXX)"
_TMPDIRS+=("$D5_TMP")
D5_REPO="$D5_TMP/repo"
build_fixture "$D5_REPO"
fixture_branch_at_foreign_merge "$D5_REPO" 9001 8888
D5_TIP="$(git -C "$D5_REPO" rev-parse refs/heads/task/9001)"

D5_SHOWREF_BEFORE="$(git -C "$D5_REPO" show-ref 2>/dev/null || true)"
run_helper --task 9001 --repo "$D5_REPO"
D5_SHOWREF_AFTER="$(git -C "$D5_REPO" show-ref 2>/dev/null || true)"

assert "D1: degenerate ref exits 0" test "$RC" -eq 0
assert "D1: stdout is 'degenerate <tip sha>'" \
    bash -c '[ "$1" = "degenerate $2" ]' _ "$OUT" "$D5_TIP"
assert "D1: show-ref is byte-identical before/after (read-only invariant)" \
    bash -c '[ "$1" = "$2" ]' _ "$D5_SHOWREF_BEFORE" "$D5_SHOWREF_AFTER"

# ─────────────────────────────────────────────────────────────────────────────
# step-7 — single-ref taxonomy: LIVE / LANDED / ABSENT / substring-safety
#
# L1 — task/9002 with one commit of its own ahead of main -> exit 1, "live <sha>"
# L2 — task/9003 self-merge, ancestor of main, cites itself -> exit 4, "landed <sha>"
# L3 — --task 9999 (no such ref)                            -> exit 5, "absent -"
# L4 — task/900 whose tip cites 9001 (count==0)              -> exit 0, "degenerate
#      <sha>" NOT landed — proves the citation boundary (900 != 9001)
# L5 — show-ref is byte-identical across all four runs (read-only invariant)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- step-7: single-ref taxonomy — LIVE / LANDED / ABSENT / substring-safety ---"

T7_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-t7-XXXXXX)"
_TMPDIRS+=("$T7_TMP")
T7_REPO="$T7_TMP/repo"
build_fixture "$T7_REPO"
fixture_branch_own_commit "$T7_REPO" 9002
fixture_branch_at_self_merge "$T7_REPO" 9003
fixture_branch_at_foreign_merge "$T7_REPO" 900 9001

T7_9002_TIP="$(git -C "$T7_REPO" rev-parse refs/heads/task/9002)"
T7_9003_TIP="$(git -C "$T7_REPO" rev-parse refs/heads/task/9003)"
T7_900_TIP="$(git -C "$T7_REPO" rev-parse refs/heads/task/900)"

T7_SHOWREF_BEFORE="$(git -C "$T7_REPO" show-ref 2>/dev/null || true)"

# L1: LIVE
run_helper --task 9002 --repo "$T7_REPO"
assert "L1: live ref (own commit ahead) exits 1" test "$RC" -eq 1
assert "L1: stdout is 'live <tip sha>'" \
    bash -c '[ "$1" = "live $2" ]' _ "$OUT" "$T7_9002_TIP"

# L2: LANDED
run_helper --task 9003 --repo "$T7_REPO"
assert "L2: landed ref (self-merge, ancestor of main) exits 4" test "$RC" -eq 4
assert "L2: stdout is 'landed <tip sha>'" \
    bash -c '[ "$1" = "landed $2" ]' _ "$OUT" "$T7_9003_TIP"

# L3: ABSENT
run_helper --task 9999 --repo "$T7_REPO"
assert "L3: absent ref exits 5" test "$RC" -eq 5
assert "L3: stdout is 'absent -'" bash -c '[ "$1" = "absent -" ]' _ "$OUT"

# L4: SUBSTRING-SAFETY — task/900's tip cites 9001, NOT 900 -> degenerate
run_helper --task 900 --repo "$T7_REPO"
assert "L4: task/900 (tip cites 9001, count==0) exits 0 (degenerate, not landed)" \
    test "$RC" -eq 0
assert "L4: stdout is 'degenerate <tip sha>'" \
    bash -c '[ "$1" = "degenerate $2" ]' _ "$OUT" "$T7_900_TIP"

T7_SHOWREF_AFTER="$(git -C "$T7_REPO" show-ref 2>/dev/null || true)"
assert "L5: show-ref is byte-identical across all four runs (read-only invariant)" \
    bash -c '[ "$1" = "$2" ]' _ "$T7_SHOWREF_BEFORE" "$T7_SHOWREF_AFTER"

# ─────────────────────────────────────────────────────────────────────────────
# Amendment (review) — `#<id>` citation-form coverage
#
# _cites_task has two disjuncts: the merge-subject form (exercised above by
# fixture_branch_at_self_merge / fixture_branch_at_foreign_merge) and the
# "#<id>" reference form, which step-7 never exercised. fixture_branch_at_hash_commit
# commits directly on main (no merge) with a "#<id>"-bearing message, so only
# the reference-form disjunct can make the match.
#
# H1 — task/46 tip is a non-merge commit whose message contains "#46",
#      ancestor of main (count==0) -> exit 4, "landed <sha>" (the `#<id>`
#      citation path).
# H2 — task/45 tip's message contains "#4588" (NOT "#45"), count==0 -> exit 0,
#      "degenerate <sha>" — proves the `#<id>` digit-boundary safety for the
#      reference form (id=45 must not match within "#4588"), mirroring L4's
#      merge-subject boundary proof.
# H3 — show-ref is byte-identical across both runs (read-only invariant).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- amendment: '#<id>' citation-form coverage ---"

# fixture_branch_at_hash_commit <repo> <branch_task_id> <cited_id> [prefix]
# Creates a NEW commit directly on main (never a merge, so the
# "^Merge <prefix><id> into " subject form can never match) whose body
# contains "#<cited_id>", then points refs/heads/<prefix><branch_task_id> at
# it. Exercises the "#<id>" citation disjunct of _cites_task in isolation
# from the merge-subject disjunct covered by fixture_branch_at_self_merge /
# fixture_branch_at_foreign_merge. Leaves `main` checked out (commits land
# directly on the currently-checked-out branch, same as build_fixture).
fixture_branch_at_hash_commit() {
    local repo="$1" branch_id="$2" cited_id="$3" prefix="${4:-task/}"
    local sha
    git -C "$repo" commit -q --allow-empty -m "fix stuff, closes #${cited_id}" >/dev/null
    sha="$(git -C "$repo" rev-parse HEAD)"
    git -C "$repo" branch -q "${prefix}${branch_id}" "$sha"
}

H_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-hash-XXXXXX)"
_TMPDIRS+=("$H_TMP")
H_REPO="$H_TMP/repo"
build_fixture "$H_REPO"
fixture_branch_at_hash_commit "$H_REPO" 46 46
fixture_branch_at_hash_commit "$H_REPO" 45 4588

H_46_TIP="$(git -C "$H_REPO" rev-parse refs/heads/task/46)"
H_45_TIP="$(git -C "$H_REPO" rev-parse refs/heads/task/45)"

H_SHOWREF_BEFORE="$(git -C "$H_REPO" show-ref 2>/dev/null || true)"

# H1: LANDED via "#<id>" reference form
run_helper --task 46 --repo "$H_REPO"
assert "H1: '#46'-citing tip (ancestor of main) exits 4" test "$RC" -eq 4
assert "H1: stdout is 'landed <tip sha>'" \
    bash -c '[ "$1" = "landed $2" ]' _ "$OUT" "$H_46_TIP"

# H2: SUBSTRING-SAFETY for the "#<id>" reference form — tip cites "#4588",
# NOT "#45" -> degenerate
run_helper --task 45 --repo "$H_REPO"
assert "H2: task 45 (tip cites '#4588', count==0) exits 0 (degenerate, not landed)" \
    test "$RC" -eq 0
assert "H2: stdout is 'degenerate <tip sha>'" \
    bash -c '[ "$1" = "degenerate $2" ]' _ "$OUT" "$H_45_TIP"

H_SHOWREF_AFTER="$(git -C "$H_REPO" show-ref 2>/dev/null || true)"
assert "H3: show-ref is byte-identical across both runs (read-only invariant)" \
    bash -c '[ "$1" = "$2" ]' _ "$H_SHOWREF_BEFORE" "$H_SHOWREF_AFTER"

# ─────────────────────────────────────────────────────────────────────────────
# Amendment (review) — --branch-prefix and -C alias coverage
#
# The fixture helpers all accept a prefix parameter, but until now nothing
# invoked the script itself with a non-default --branch-prefix or with the
# -C alias for --repo.
#
# P1 — custom --branch-prefix "wip/": a degenerate fixture built under a
#      non-default prefix classifies identically to the default-prefix case,
#      proving the `_id` strip / `refs/heads/<prefix><id>` resolution honors
#      the flag.
# P2 — -C <repo> (short alias for --repo) yields an identical classification
#      to the equivalent --repo <repo> invocation.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- amendment: --branch-prefix and -C alias coverage ---"

P_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-prefix-XXXXXX)"
_TMPDIRS+=("$P_TMP")
P_REPO="$P_TMP/repo"
build_fixture "$P_REPO"
fixture_branch_at_foreign_merge "$P_REPO" 9301 8801 "wip/"
P_TIP="$(git -C "$P_REPO" rev-parse refs/heads/wip/9301)"

run_helper --task 9301 --repo "$P_REPO" --branch-prefix "wip/"
assert "P1: custom --branch-prefix classifies degenerate" test "$RC" -eq 0
assert "P1: stdout is 'degenerate <tip sha>' under wip/ prefix" \
    bash -c '[ "$1" = "degenerate $2" ]' _ "$OUT" "$P_TIP"

run_helper -C "$P_REPO" --task 9301 --branch-prefix "wip/"
assert "P2: -C alias exits identically to --repo" test "$RC" -eq 0
assert "P2: -C alias stdout matches --repo stdout" \
    bash -c '[ "$1" = "degenerate $2" ]' _ "$OUT" "$P_TIP"

# ─────────────────────────────────────────────────────────────────────────────
# Amendment (review) — --branch-prefix regex-metacharacter safety
#
# BRANCH_PREFIX is interpolated into _cites_task's `grep -E` pattern. A
# caller-supplied prefix containing an ERE metacharacter (e.g. "+") must be
# matched LITERALLY, not interpreted as regex syntax. "wip+" is a
# self-citing (landed) branch: its own merge subject is "Merge wip+5 into
# main". Unescaped, the ERE `wip+5` means "wi" + one-or-more "p" + "5",
# which does NOT match the literal string "wip+5" (there is exactly one
# literal "p") — a false NEGATIVE that would misclassify a genuinely landed
# ref as degenerate. Escaping the prefix before interpolation fixes this.
#
# R1 — --branch-prefix "wip+" self-citing tip classifies "landed" (proves
#      the "+" is matched literally, not as an ERE quantifier).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- amendment: --branch-prefix regex-metacharacter safety ---"

R1_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-metachar-XXXXXX)"
_TMPDIRS+=("$R1_TMP")
R1_REPO="$R1_TMP/repo"
build_fixture "$R1_REPO"
fixture_branch_at_self_merge "$R1_REPO" 5 "wip+"
R1_TIP="$(git -C "$R1_REPO" rev-parse refs/heads/wip+5)"

run_helper --task 5 --repo "$R1_REPO" --branch-prefix "wip+"
assert "R1: '+'-bearing --branch-prefix self-citation exits 4 (landed)" test "$RC" -eq 4
assert "R1: stdout is 'landed <tip sha>' (prefix matched literally, not as ERE quantifier)" \
    bash -c '[ "$1" = "landed $2" ]' _ "$OUT" "$R1_TIP"

# ─────────────────────────────────────────────────────────────────────────────
# step-9 — fleet-audit mode (no status oracle)
#
# Fixture mix: two degenerate (9101, 9102; foreign merges), one live (9103;
# own commit), one landed (9104; self-merge), plus a non-numeric task/foo
# that must be skipped without aborting the sweep.
#
# AU1 — exit 0; one row per numeric ref `<task_id> <class> <tip_sha>`;
#       summary "audit: degenerate=2 live=1 landed=1 absent=0 total=4
#       flagged=2" (flagged==degenerate when no oracle); read-only invariant.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- step-9: fleet-audit mode (no status oracle) ---"

A9_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-a9-XXXXXX)"
_TMPDIRS+=("$A9_TMP")
A9_REPO="$A9_TMP/repo"
build_fixture "$A9_REPO"
fixture_branch_at_foreign_merge "$A9_REPO" 9101 8801
fixture_branch_at_foreign_merge "$A9_REPO" 9102 8802
fixture_branch_own_commit "$A9_REPO" 9103
fixture_branch_at_self_merge "$A9_REPO" 9104
git -C "$A9_REPO" branch -q task/foo main

A9_SHOWREF_BEFORE="$(git -C "$A9_REPO" show-ref 2>/dev/null || true)"
run_helper --audit --repo "$A9_REPO"
A9_SHOWREF_AFTER="$(git -C "$A9_REPO" show-ref 2>/dev/null || true)"

assert "AU1: --audit exits 0" test "$RC" -eq 0
assert "AU1: exactly 4 data rows" \
    bash -c 'printf "%s\n" "$1" | grep -cE "^[0-9]+ (degenerate|live|landed|absent) " | grep -qx 4' _ "$OUT"
assert "AU1: 9101 row is degenerate" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9101 degenerate "' _ "$OUT"
assert "AU1: 9102 row is degenerate" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9102 degenerate "' _ "$OUT"
assert "AU1: 9103 row is live" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9103 live "' _ "$OUT"
assert "AU1: 9104 row is landed" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9104 landed "' _ "$OUT"
assert "AU1: task/foo (non-numeric) is skipped, not a row" \
    bash -c '! printf "%s\n" "$1" | grep -q "foo"' _ "$OUT"
assert "AU1: summary line matches" \
    bash -c 'printf "%s\n" "$1" | grep -qxF "audit: degenerate=2 live=1 landed=1 absent=0 total=4 flagged=2"' \
    _ "$OUT"
assert "AU1: show-ref is byte-identical (read-only invariant)" \
    bash -c '[ "$1" = "$2" ]' _ "$A9_SHOWREF_BEFORE" "$A9_SHOWREF_AFTER"

# ─────────────────────────────────────────────────────────────────────────────
# step-11 — fleet-audit status-oracle (advisory filter)
#
# O1 — 9101/9102/9103 all degenerate; oracle maps 9101->blocked,
#      9102->done, 9103->in-progress. Each row gains a trailing status
#      column; flagged=2 (done excluded as terminal).
# O2 — oracle exits non-zero for 9205 and prints empty for 9206 (both
#      degenerate); both are treated as "unknown" (non-terminal) and
#      still flagged, mirroring warm-lane-preflight.sh Check 6.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- step-11: fleet-audit status-oracle (advisory filter) ---"

O11_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-o11-XXXXXX)"
_TMPDIRS+=("$O11_TMP")
O11_REPO="$O11_TMP/repo"
build_fixture "$O11_REPO"
fixture_branch_at_foreign_merge "$O11_REPO" 9101 8801
fixture_branch_at_foreign_merge "$O11_REPO" 9102 8802
fixture_branch_at_foreign_merge "$O11_REPO" 9103 8803

O11_STATUS_CMD="$O11_TMP/status-cmd.sh"
cat > "$O11_STATUS_CMD" << 'STATUS_EOF'
#!/usr/bin/env bash
case "$1" in
    9101) echo blocked ;;
    9102) echo done ;;
    9103) echo in-progress ;;
    *) echo "" ;;
esac
STATUS_EOF
chmod +x "$O11_STATUS_CMD"

run_helper --audit --repo "$O11_REPO" --status-cmd "$O11_STATUS_CMD"
assert "O1: --audit with status-cmd exits 0" test "$RC" -eq 0
assert "O1: 9101 row has trailing status 'blocked'" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9101 degenerate [0-9a-f]{40} blocked$"' _ "$OUT"
assert "O1: 9102 row has trailing status 'done'" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9102 degenerate [0-9a-f]{40} done$"' _ "$OUT"
assert "O1: 9103 row has trailing status 'in-progress'" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9103 degenerate [0-9a-f]{40} in-progress$"' _ "$OUT"
assert "O1: summary flagged=2 (done excluded, blocked+in-progress counted)" \
    bash -c 'printf "%s\n" "$1" | grep -qxF "audit: degenerate=3 live=0 landed=0 absent=0 total=3 flagged=2"' \
    _ "$OUT"

O11B_TMP="$(mktemp -d /tmp/test-warm-lane-degen-ref-o11b-XXXXXX)"
_TMPDIRS+=("$O11B_TMP")
O11B_REPO="$O11B_TMP/repo"
build_fixture "$O11B_REPO"
fixture_branch_at_foreign_merge "$O11B_REPO" 9205 8901
fixture_branch_at_foreign_merge "$O11B_REPO" 9206 8902

O11B_STATUS_CMD="$O11B_TMP/status-cmd.sh"
cat > "$O11B_STATUS_CMD" << 'STATUS_EOF'
#!/usr/bin/env bash
case "$1" in
    9205) exit 1 ;;
    9206) printf '' ;;
    *) echo blocked ;;
esac
STATUS_EOF
chmod +x "$O11B_STATUS_CMD"

run_helper --audit --repo "$O11B_REPO" --status-cmd "$O11B_STATUS_CMD"
assert "O2: --audit with failing/empty oracle exits 0" test "$RC" -eq 0
assert "O2: 9205 (oracle exits non-zero) row status is 'unknown'" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9205 degenerate [0-9a-f]{40} unknown$"' _ "$OUT"
assert "O2: 9206 (oracle prints empty) row status is 'unknown'" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^9206 degenerate [0-9a-f]{40} unknown$"' _ "$OUT"
assert "O2: summary flagged=2 (both unknown treated as non-terminal)" \
    bash -c 'printf "%s\n" "$1" | grep -qxF "audit: degenerate=2 live=0 landed=0 absent=0 total=2 flagged=2"' \
    _ "$OUT"

test_summary
