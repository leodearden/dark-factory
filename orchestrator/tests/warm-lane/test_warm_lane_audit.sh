#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_warm_lane_audit.sh
# Hermetic tests for orchestrator/scripts/warm-lane/warm-lane-audit.sh.
#
# PRD: docs/prds/warm-lane-pool-sizing-lifecycle.md §9.1 (task α); boundary
# tests B1/B2 (§10). Reuses the task-4749 status-oracle seam
# (REIFY_LANE_LEAK_STATUS_CMD) and the <lane_dir>.lock flock convention from
# scripts/warm-lane-gc.sh / tests/infra/test_warm_lane_gc.sh.
#
# run_helper captures STDOUT, STDERR, and RC separately:
#   OUT     — captured stdout from the script
#   ERR_OUT — captured stderr from the script
#   RC      — exit code
#
# Blocks:
#   A — CLI guard: --help, unknown flag
#   (additional blocks land in subsequent commits as the script grows: the
#    ASSIGNED/FREE probe + LIVE + headroom, backing-task-status +
#    RECLAIMABLE-via-terminal, recoverable LANDED/PUSHED/ORPHAN,
#    residue-only-dirty, LEAKED + stale, measured disk figures, --format
#    json, the combined B1 fixture + A3 degradation, and B2 read-only.)
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (PORTED_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
REPO_ROOT="$DF_REPO_ROOT"
SCRIPT="$WARM_LANE_SCRIPTS_DIR/warm-lane-audit.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=tests/infra/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"

echo "=== scripts/warm-lane-audit.sh hermetic tests (task 5172) ==="

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

ERR_FILE="$(mktemp /tmp/test-warm-lane-audit-err-XXXXXX)"
_TMPDIRS+=("$ERR_FILE")

# ── run_helper ─────────────────────────────────────────────────────────────────
# Invokes warm-lane-audit.sh, capturing OUT (stdout), ERR_OUT (stderr), RC.
run_helper() {
    local rc=0
    > "$ERR_FILE"
    OUT="$(bash "$SCRIPT" "$@" 2>"$ERR_FILE")" || rc=$?
    ERR_OUT="$(cat "$ERR_FILE")"
    RC=$rc
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
run_helper --nope
assert "A2: unknown flag exits 2" test "$RC" -eq 2

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture scaffolding: make_lane + _wait_for_reader_lock
# ──────────────────────────────────────────────────────────────────────────────

# make_lane DIR [BRANCH]
# Creates a minimal standalone git repo at DIR (one initial commit on main).
# BRANCH: task/NNNN -> checkout a new branch; DETACH -> detach HEAD;
# main/"" -> stay on main. Mirrors tests/infra/test_warm_lane_preflight.sh's
# identically-named helper (Block D) — each lane is its OWN independent repo
# (not a linked worktree of a shared primary), which is sufficient for
# warm-lane-audit.sh's per-lane git predicates (merge-base/status/symbolic-ref
# all operate within a single lane's own repo).
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
# subshell's lock acquisition under load. Mirrors tests/infra/test_warm_lane_gc.sh's
# identically-named helper.
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

# _hold_lane_lock <mount> <lane>
# Marks <lane> LIVE: creates <mount>/<lane>.lock, backgrounds a consumer that
# takes the EXCLUSIVE flock, touches its READY marker and then parks, and
# blocks until _wait_for_reader_lock observes that marker -- so the flock is
# causally guaranteed held at the caller's next statement (technique R,
# docs/prds/infra-test-wallclock-deflake.md; never a fixed `sleep`).
#
# Publishes the background pid in the GLOBAL `LANE_LOCK_PID` rather than on
# stdout deliberately: a `pid=$(_hold_lane_lock ...)` command substitution
# would run the body in a subshell, so the `_BGPIDS` registration below would
# be discarded and the 300s sleeper would outlive the suite's cleanup trap.
# Callers that need a per-lane handle copy LANE_LOCK_PID immediately.
LANE_LOCK_PID=""
_hold_lane_lock() {
    local mount="$1" lane="$2"
    local lock="$mount/$lane.lock"
    local ready="$lock.ready-marker"
    touch "$lock"
    ( flock -x 9 && touch "$ready" && sleep 300 ) 9>"$lock" &
    LANE_LOCK_PID=$!
    _BGPIDS+=("$LANE_LOCK_PID")
    _wait_for_reader_lock "$ready" 30
}

# _hold_lane_lock_shared <mount> <lane>
# Block Q's SHARED counterpart to _hold_lane_lock above (§9.1 Invariant A2):
# marks <lane> "held by a shared reader" instead of "held by an exclusive
# consumer" -- the case scripts/warm-lane-audit.sh's own probe (_probe_live,
# line 330-331) must read as IDLE, not LIVE.
#
# Two deliberate differences from _hold_lane_lock, both load-bearing:
#   - `flock -s 9`, not `-x` -- this is the entire point of the helper.
#   - the fd is opened READ-only (`9<"$lock"`, after the `touch`), not `9>` as
#     the exclusive helper uses. This mirrors the production probe's own
#     read-only open (`exec 7<"$lock"`, scripts/warm-lane-audit.sh:330), so
#     the fixture models a real shared READER rather than an artificial
#     write-opened shared lock, and it avoids relying on Linux's (correct but
#     non-obvious, and not POSIX-fcntl-portable) acceptance of LOCK_SH on an
#     O_WRONLY fd.
# The ready-marker uses a DISTINCT suffix (`.shared-ready-marker`) so a shared
# and an exclusive holder on two different lanes can never collide on one
# lock file's marker.
#
# Publishes the pid via the GLOBAL `LANE_LOCK_PID` (and registers it in
# `_BGPIDS`) for the same reason as _hold_lane_lock: a `$( )` capture would
# run this body in a subshell and orphan the 300s sleeper past the cleanup
# trap. Uses _wait_for_reader_lock (technique R) for the causal handshake --
# never a fixed `sleep` (DD5).
_hold_lane_lock_shared() {
    local mount="$1" lane="$2"
    local lock="$mount/$lane.lock"
    local ready="$lock.shared-ready-marker"
    touch "$lock"
    ( flock -s 9 && touch "$ready" && sleep 300 ) 9<"$lock" &
    LANE_LOCK_PID=$!
    _BGPIDS+=("$LANE_LOCK_PID")
    _wait_for_reader_lock "$ready" 30
}

# make_lane_state <mount> <lane> <state> [task_id] [branch]
# Writes <mount>/.lane-state/<lane>.json in the EXACT byte shape dark-factory's
# LaneRecord.to_json() emits (orchestrator/src/orchestrator/lane_lifecycle.py:
# json.dumps(to_dict(), indent=2) -- dataclass field order state/task_id/title/
# branch/seeded_from_sha/updated_at, unquoted `null` for a None field, and NO
# trailing newline). Creates the state dir on first use.
#
# An omitted/empty task_id or branch is emitted as JSON `null`, not `""` --
# that is the real producer's shape for an unassigned lane, and it is the
# fixture the record-vs-branch pin fallback is asserted against.
#
# updated_at is a FIXED timestamp, not `date`: no fixture in this suite may
# depend on wall-clock (DD5); the audit never reads this field.
make_lane_state() {
    local mount="$1" lane="$2" state="$3" task_id="${4:-}" branch="${5:-}"
    local state_dir="$mount/.lane-state"
    mkdir -p "$state_dir"
    # Plain `if`, not `[ -n .. ] && ..`: an AND-list whose left side fails
    # yields a non-zero list status, which `set -e` may treat as a function
    # failure at the call site. Never worth the subtlety in fixture code.
    local task_json='null' branch_json='null' title_json='null'
    if [ -n "$task_id" ]; then
        task_json="\"$task_id\""
        title_json="\"lane fixture task $task_id\""
    fi
    if [ -n "$branch" ]; then
        branch_json="\"$branch\""
    fi
    printf '{\n  "state": "%s",\n  "task_id": %s,\n  "title": %s,\n  "branch": %s,\n  "seeded_from_sha": null,\n  "updated_at": "2026-07-26T12:43:10.704531+00:00"\n}' \
        "$state" "$task_json" "$title_json" "$branch_json" \
        > "$state_dir/$lane.json"
}

# make_lane_state_raw <mount> <lane> <text>
# Writes arbitrary bytes to <mount>/.lane-state/<lane>.json -- the corrupt- and
# compact-record cases, which by construction cannot go through make_lane_state.
make_lane_state_raw() {
    local mount="$1" lane="$2" text="$3"
    local state_dir="$mount/.lane-state"
    mkdir -p "$state_dir"
    printf '%s' "$text" > "$state_dir/$lane.json"
}

# ── summary-line readers (shared: first used in Block L, again in N/O/P) ──────
# _report_field <audit-stdout> <line-prefix> <key>
# Prints the integer value of <key> on the <line-prefix> summary line, or
# NOTHING when the field is absent. Deliberately does not default to 0: a
# missing field must read as missing, so an assertion can never pass against a
# field the script does not actually emit.
_report_field() {
    local out="$1" prefix="$2" key="$3"
    printf '%s\n' "$out" | grep "^${prefix} " | tr ' ' '\n' \
        | sed -n -E "s/^${key}=([0-9]+)$/\1/p" || true
    return 0
}

# _headroom_field <audit-stdout> <key> — the HEADROOM line's <key>.
_headroom_field() { _report_field "$1" HEADROOM "$2"; }

# _pinned_field <audit-stdout> <key> — the PINNED breakdown line's <key>.
_pinned_field() { _report_field "$1" PINNED "$2"; }

# _sum_holds <total> <part>...
# True iff every argument is a non-empty integer AND <total> equals the sum of
# the parts. An ABSENT field fails (empty is NOT 0): a summing identity must be
# proven against fields that actually exist, or it passes vacuously against a
# report that never emitted them.
_sum_holds() {
    local total="$1"; shift
    local v sum=0
    for v in "$total" "$@"; do
        case "$v" in
            ''|*[!0-9]*) return 1 ;;
        esac
    done
    for v in "$@"; do
        sum=$(( sum + v ))
    done
    [ "$total" -eq "$sum" ]
}

# _partition_holds <resident> <live> <pinned> <quarantined> <free>
_partition_holds() { _sum_holds "$@"; }

# _strict_max <candidate> <other>...
# True iff every argument is a non-empty integer AND <candidate> is STRICTLY
# greater than each of the others. Absent fields fail, for _sum_holds' reason.
_strict_max() {
    local cand="$1"; shift
    local v
    for v in "$cand" "$@"; do
        case "$v" in
            ''|*[!0-9]*) return 1 ;;
        esac
    done
    for v in "$@"; do
        [ "$cand" -gt "$v" ] || return 1
    done
    return 0
}

# ──────────────────────────────────────────────────────────────────────────────
# Block B — ASSIGNED/FREE probe + LIVE + headroom counts
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block B: ASSIGNED/FREE probe + LIVE + headroom counts ---"

B_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-b-XXXXXX)"
_TMPDIRS+=("$B_MOUNT")

# _lane-live: a live consumer holds the lane's exclusive flock (ASSIGNED -> LIVE).
# Causal READY-marker handshake instead of a fixed sleep (de-flake convention,
# tests/infra/test_warm_lane_gc.sh Block I6 precedent).
make_lane "$B_MOUNT/_lane-live"
_hold_lane_lock "$B_MOUNT" "_lane-live"
B_LOCK_PID="$LANE_LOCK_PID"

# _lane-free: unheld, pre-created lock file (FREE).
make_lane "$B_MOUNT/_lane-free"
touch "$B_MOUNT/_lane-free.lock"

run_helper --mount "$B_MOUNT"

assert "B1: exit 0" test "$RC" -eq 0
assert "B2: _lane-live row shows live=LIVE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-live .*live=LIVE"' _ "$OUT"
assert "B3: _lane-live classification LIVE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-live .*classification=LIVE"' _ "$OUT"
assert "B4: _lane-free row shows live=IDLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-free .*live=IDLE"' _ "$OUT"
assert "B5: HEADROOM line shows resident=2" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "resident=2"' _ "$OUT"
assert "B6: HEADROOM line shows live=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -qE "(^| )live=1( |$)"' _ "$OUT"
assert "B7: HEADROOM line shows free=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "free=1"' _ "$OUT"

# Release the lock
kill "$B_LOCK_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Shared scaffolding: status-oracle stub + ahead-of-main commit helper
# ──────────────────────────────────────────────────────────────────────────────

# _add_ahead_commit DIR — add a committed change NOT reachable from main.
_add_ahead_commit() {
    local dir="$1"
    echo "ahead" >> "$dir/README.md"
    git -C "$dir" add README.md
    git -C "$dir" commit -q -m "ahead-of-main commit"
}

# leak-oracle.sh: given task-id $1, looks up status in ORACLE_MAP file (one
# "id status" pair per line). Exits 0 with empty output for unknown ids.
# Mirrors tests/infra/test_warm_lane_preflight.sh Block D / test_warm_lane_gc.sh
# byte-for-byte (the shared task-4749 status-oracle contract).
ORACLE_STUB_DIR="$(mktemp -d /tmp/test-warm-lane-audit-oracle-stub-XXXXXX)"
_TMPDIRS+=("$ORACLE_STUB_DIR")
cat > "$ORACLE_STUB_DIR/leak-oracle.sh" << 'STUB_EOF'
#!/usr/bin/env bash
_qid="$1"
if [ -f "${ORACLE_MAP:-}" ]; then
    while IFS=' ' read -r _mid _mst; do
        if [ "$_mid" = "$_qid" ]; then
            printf '%s\n' "$_mst"
            exit 0
        fi
    done < "$ORACLE_MAP"
fi
exit 0
STUB_EOF
chmod +x "$ORACLE_STUB_DIR/leak-oracle.sh"

# leak-oracle-fail.sh: always exits non-zero -- drives the A3 hardening test
# (oracle failure must NOT abort; unknown = neither terminal nor non-terminal).
cat > "$ORACLE_STUB_DIR/leak-oracle-fail.sh" << 'STUB_EOF'
#!/usr/bin/env bash
exit 1
STUB_EOF
chmod +x "$ORACLE_STUB_DIR/leak-oracle-fail.sh"

# ──────────────────────────────────────────────────────────────────────────────
# Block C — backing-task-status via the 4749 seam + terminal -> RECLAIMABLE
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block C: backing-task-status + terminal -> RECLAIMABLE ---"

C_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-c-XXXXXX)"
_TMPDIRS+=("$C_MOUNT")

C_MAP="$(mktemp /tmp/test-warm-lane-audit-c-map-XXXXXX)"
_TMPDIRS+=("$C_MAP")
printf '100 done\n200 pending\n' > "$C_MAP"

# _lane-done: FREE, task/100, oracle=done, ahead-of-main.
make_lane "$C_MOUNT/_lane-done" "task/100"
_add_ahead_commit "$C_MOUNT/_lane-done"

# _lane-pending: FREE, task/200, oracle=pending, ahead-of-main.
make_lane "$C_MOUNT/_lane-pending" "task/200"
_add_ahead_commit "$C_MOUNT/_lane-pending"

ORACLE_MAP="$C_MAP" run_helper --mount "$C_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "C1: exit 0" test "$RC" -eq 0
assert "C2: _lane-done backing-task-status=terminal" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-done .*status=terminal"' _ "$OUT"
assert "C3: _lane-done classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-done .*classification=RECLAIMABLE"' _ "$OUT"
assert "C4: HEADROOM line shows reclaimable=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "reclaimable=1"' _ "$OUT"
assert "C5: _lane-pending backing-task-status=non-terminal" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-pending .*status=non-terminal"' _ "$OUT"
assert "C6: _lane-pending classification PRESERVED-OK (not reclaimable-via-terminal)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-pending .*classification=PRESERVED-OK"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block D — recoverable LANDED|PUSHED|ORPHAN + recoverable -> RECLAIMABLE
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block D: recoverable LANDED|PUSHED|ORPHAN ---"

# D(a): HEAD is an ancestor of main (no ahead commit) -> LANDED -> RECLAIMABLE.
DA_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-da-XXXXXX)"
_TMPDIRS+=("$DA_MOUNT")
DA_MAP="$(mktemp /tmp/test-warm-lane-audit-da-map-XXXXXX)"
_TMPDIRS+=("$DA_MAP")
printf '300 pending\n' > "$DA_MAP"
make_lane "$DA_MOUNT/_lane-landed" "task/300"

ORACLE_MAP="$DA_MAP" run_helper --mount "$DA_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "D1: exit 0" test "$RC" -eq 0
assert "D2: _lane-landed recoverable=LANDED" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-landed .*recoverable=LANDED"' _ "$OUT"
assert "D3: _lane-landed classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-landed .*classification=RECLAIMABLE"' _ "$OUT"

# D(b): pushed to a bare origin remote (origin/<branch> contains HEAD), ahead
# of local main -> PUSHED -> RECLAIMABLE.
DB_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-db-XXXXXX)"
_TMPDIRS+=("$DB_MOUNT")
DB_MAP="$(mktemp /tmp/test-warm-lane-audit-db-map-XXXXXX)"
_TMPDIRS+=("$DB_MAP")
printf '400 pending\n' > "$DB_MAP"
git init -q --bare "$DB_MOUNT/origin.git"
make_lane "$DB_MOUNT/_lane-pushed" "task/400"
_add_ahead_commit "$DB_MOUNT/_lane-pushed"
git -C "$DB_MOUNT/_lane-pushed" remote add origin "$DB_MOUNT/origin.git"
git -C "$DB_MOUNT/_lane-pushed" push -q origin task/400

ORACLE_MAP="$DB_MAP" run_helper --mount "$DB_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "D4: exit 0" test "$RC" -eq 0
assert "D5: _lane-pushed recoverable=PUSHED" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-pushed .*recoverable=PUSHED"' _ "$OUT"
assert "D6: _lane-pushed classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-pushed .*classification=RECLAIMABLE"' _ "$OUT"

# D(c): ahead of main, no origin -> ORPHAN; not stale -> PRESERVED-OK.
DC_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-dc-XXXXXX)"
_TMPDIRS+=("$DC_MOUNT")
DC_MAP="$(mktemp /tmp/test-warm-lane-audit-dc-map-XXXXXX)"
_TMPDIRS+=("$DC_MAP")
printf '500 pending\n' > "$DC_MAP"
make_lane "$DC_MOUNT/_lane-orphan" "task/500"
_add_ahead_commit "$DC_MOUNT/_lane-orphan"

ORACLE_MAP="$DC_MAP" run_helper --mount "$DC_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "D7: exit 0" test "$RC" -eq 0
assert "D8: _lane-orphan recoverable=ORPHAN" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-orphan .*recoverable=ORPHAN"' _ "$OUT"
assert "D9: _lane-orphan classification PRESERVED-OK (not stale)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-orphan .*classification=PRESERVED-OK"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block E — residue-only-dirty -> RECLAIMABLE vs genuine WIP
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block E: residue-only-dirty vs genuine WIP ---"

# E(a): FREE, pending, ahead-of-main, ORPHAN lane whose ONLY dirty tracked path
# is the residue file data/queue/write_queue.db (committed first, then
# modified in place) -> dirty=residue-only -> classification RECLAIMABLE,
# despite being non-terminal and ORPHAN.
EA_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-ea-XXXXXX)"
_TMPDIRS+=("$EA_MOUNT")
EA_MAP="$(mktemp /tmp/test-warm-lane-audit-ea-map-XXXXXX)"
_TMPDIRS+=("$EA_MAP")
printf '600 pending\n' > "$EA_MAP"
make_lane "$EA_MOUNT/_lane-residue" "task/600"
_add_ahead_commit "$EA_MOUNT/_lane-residue"
mkdir -p "$EA_MOUNT/_lane-residue/data/queue"
printf 'db-v1\n' > "$EA_MOUNT/_lane-residue/data/queue/write_queue.db"
git -C "$EA_MOUNT/_lane-residue" add data/queue/write_queue.db
git -C "$EA_MOUNT/_lane-residue" commit -q -m "add residue db"
printf 'db-v2\n' > "$EA_MOUNT/_lane-residue/data/queue/write_queue.db"

ORACLE_MAP="$EA_MAP" run_helper --mount "$EA_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "E1: exit 0" test "$RC" -eq 0
assert "E2: _lane-residue dirty=residue-only" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-residue .*dirty=residue-only"' _ "$OUT"
assert "E3: _lane-residue classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-residue .*classification=RECLAIMABLE"' _ "$OUT"

# E(b): FREE, pending, ahead-of-main, ORPHAN lane with a dirty tracked SOURCE
# file (README.md) -> dirty=wip -> classification PRESERVED-OK (genuine
# unrecoverable WIP; not yet stale — Block F introduces the LEAKED carve-out).
EB_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-eb-XXXXXX)"
_TMPDIRS+=("$EB_MOUNT")
EB_MAP="$(mktemp /tmp/test-warm-lane-audit-eb-map-XXXXXX)"
_TMPDIRS+=("$EB_MAP")
printf '700 pending\n' > "$EB_MAP"
make_lane "$EB_MOUNT/_lane-wip" "task/700"
_add_ahead_commit "$EB_MOUNT/_lane-wip"
printf 'uncommitted wip change\n' >> "$EB_MOUNT/_lane-wip/README.md"

ORACLE_MAP="$EB_MAP" run_helper --mount "$EB_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "E4: exit 0" test "$RC" -eq 0
assert "E5: _lane-wip dirty=wip" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-wip .*dirty=wip"' _ "$OUT"
assert "E6: _lane-wip classification PRESERVED-OK (genuine WIP, not stale)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-wip .*classification=PRESERVED-OK"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block F — LEAKED + stale A4 relation
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block F: LEAKED + stale A4 relation ---"

F_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-f-XXXXXX)"
_TMPDIRS+=("$F_MOUNT")
F_MAP="$(mktemp /tmp/test-warm-lane-audit-f-map-XXXXXX)"
_TMPDIRS+=("$F_MAP")
printf '800 pending\n' > "$F_MAP"

# FREE, non-terminal (task/800, oracle=pending), ORPHAN (ahead-of-main commit,
# no origin) -- built FIRST; the lane dir's mtime is aged past the default
# 60-minute knob LAST (touch -d), so the staleness measurement reflects only
# the deliberate backdate, not repo setup (git commit only touches paths
# inside .git/ and README.md's own mtime, never the lane dir's own entries).
make_lane "$F_MOUNT/_lane-leaked" "task/800"
_add_ahead_commit "$F_MOUNT/_lane-leaked"
touch -d '90 minutes ago' "$F_MOUNT/_lane-leaked"

ORACLE_MAP="$F_MAP" run_helper --mount "$F_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "F1: exit 0" test "$RC" -eq 0
assert "F2: _lane-leaked classification LEAKED (default stale-age-min=60, age~90min)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-leaked .*classification=LEAKED"' _ "$OUT"
assert "F3: HEADROOM line shows leaked=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "leaked=1"' _ "$OUT"

# A4 relation (not a frozen age): the SAME lane, re-run with --stale-age-min
# raised far above its ~90-minute age -> PRESERVED-OK, leaked=0.
ORACLE_MAP="$F_MAP" run_helper --mount "$F_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh" --stale-age-min 100000
assert "F4: exit 0" test "$RC" -eq 0
assert "F5: _lane-leaked classification PRESERVED-OK under --stale-age-min 100000" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-leaked .*classification=PRESERVED-OK"' _ "$OUT"
assert "F6: HEADROOM line shows leaked=0 under --stale-age-min 100000" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "leaked=0"' _ "$OUT"

# Same relation via the env knob (no --stale-age-min flag) -- confirms the
# knob is read from REIFY_WARM_LANE_AUDIT_STALE_AGE_MIN, not just the flag.
REIFY_WARM_LANE_AUDIT_STALE_AGE_MIN=100000 ORACLE_MAP="$F_MAP" run_helper --mount "$F_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "F7: exit 0" test "$RC" -eq 0
assert "F8: _lane-leaked classification PRESERVED-OK under env REIFY_WARM_LANE_AUDIT_STALE_AGE_MIN=100000" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-leaked .*classification=PRESERVED-OK"' _ "$OUT"

# ──────────────────────────────────────────────────────────────────────────────
# Block G — measured divergent_gib / free_gib / budget_gib (DD3, no frozen constant)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block G: measured disk figures (DD3, no frozen GB constant) ---"

G_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-g-XXXXXX)"
_TMPDIRS+=("$G_MOUNT")

# A lane with no target/ -> divergent_gib must report 0.
make_lane "$G_MOUNT/_lane-g"

# Stub df: ignores all args, emits a KNOWN avail byte count in the same shape
# as `df -B1 --output=avail -- <mount>` (a header line + one data line). The
# byte count is deliberately non-round (+123456789) so the floor-to-GiB
# truncation is exercised for real, not a coincidental passthrough.
G_DF_STUB_DIR="$(mktemp -d /tmp/test-warm-lane-audit-g-dfstub-XXXXXX)"
_TMPDIRS+=("$G_DF_STUB_DIR")
G_AVAIL_BYTES=$(( 200 * 1024 * 1024 * 1024 + 123456789 ))
cat > "$G_DF_STUB_DIR/df-fake.sh" << EOF
#!/usr/bin/env bash
printf '     Avail\n'
printf '%s\n' "$G_AVAIL_BYTES"
EOF
chmod +x "$G_DF_STUB_DIR/df-fake.sh"

# The test COMPUTES the expected free_gib/budget_gib from the SAME stub input
# + the SAME --safety knob passed to the script -- a derived relation, not a
# hardcoded literal (DD3/G6/D8 no-frozen-constant rule).
G_SAFETY=4
G_EXPECTED_FREE_GIB=$(( G_AVAIL_BYTES / 1073741824 ))
G_EXPECTED_BUDGET_GIB=$(( G_EXPECTED_FREE_GIB / G_SAFETY ))

REIFY_WARM_LANE_AUDIT_DF="$G_DF_STUB_DIR/df-fake.sh" run_helper --mount "$G_MOUNT" --safety "$G_SAFETY"

assert "G1: exit 0" test "$RC" -eq 0
assert "G2: _lane-g row has a non-negative integer divergent_gib" \
    bash -c 'printf "%s\n" "$1" | grep -qE "lane=_lane-g .*divergent_gib=[0-9]+"' _ "$OUT"
assert "G3: _lane-g divergent_gib=0 (no target/)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-g .*divergent_gib=0"' _ "$OUT"
assert "G4: HEADROOM divergent_gib is a non-negative integer" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -qE "divergent_gib=[0-9]+"' _ "$OUT"
assert "G5: HEADROOM free_gib == floor(stub_avail_bytes / 2^30)" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "free_gib=$2 budget_gib="' _ "$OUT" "$G_EXPECTED_FREE_GIB"
assert "G6: HEADROOM budget_gib == floor(free_gib / safety)" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -qE "budget_gib=$2\$"' _ "$OUT" "$G_EXPECTED_BUDGET_GIB"

# ──────────────────────────────────────────────────────────────────────────────
# Block H — --format json
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block H: --format json ---"

H_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-h-XXXXXX)"
_TMPDIRS+=("$H_MOUNT")
make_lane "$H_MOUNT/_lane-h1"
make_lane "$H_MOUNT/_lane-h2"

# Small python3 checker scripts (avoids nested-quoting hazards of inlining a
# multi-line python one-liner inside a single-quoted bash -c inside assert).
H_PY_KEYS="$(mktemp /tmp/test-warm-lane-audit-h-keys-XXXXXX.py)"
_TMPDIRS+=("$H_PY_KEYS")
cat > "$H_PY_KEYS" << 'PYEOF'
import json, sys
data = json.load(sys.stdin)
lanes = data["lanes"] if isinstance(data, dict) and "lanes" in data else data
assert isinstance(lanes, list) and len(lanes) == 2, lanes
expected_keys = {"lane", "role", "live", "assigned", "branch", "status",
                  "recoverable", "dirty", "divergent_gib", "age_min",
                  "classification"}
for obj in lanes:
    assert expected_keys.issubset(obj.keys()), obj
names = {obj["lane"] for obj in lanes}
assert names == {"_lane-h1", "_lane-h2"}, names
PYEOF

H_PY_HEADROOM="$(mktemp /tmp/test-warm-lane-audit-h-headroom-XXXXXX.py)"
_TMPDIRS+=("$H_PY_HEADROOM")
cat > "$H_PY_HEADROOM" << 'PYEOF'
import json, sys
data = json.load(sys.stdin)
headroom = data["headroom"] if isinstance(data, dict) and "headroom" in data else data
assert headroom["resident"] == 2, headroom
PYEOF

# Default (no --format) is still the table.
run_helper --mount "$H_MOUNT"
assert "H1: exit 0 (default format)" test "$RC" -eq 0
assert "H2: default format is the table (a lane= row is present)" \
    bash -c 'printf "%s\n" "$1" | grep -q "^lane=_lane-h1 "' _ "$OUT"

# --format json: parses via python3 json.load; carries one object per lane
# with the full column set, plus a headroom summary.
run_helper --mount "$H_MOUNT" --format json
assert "H3: exit 0 (json format)" test "$RC" -eq 0
assert "H4: --format json output parses as JSON (python3 json.load)" \
    bash -c 'printf "%s" "$1" | python3 -c "import json,sys; json.load(sys.stdin)"' _ "$OUT"
assert "H5: json output contains a classification key" \
    bash -c 'printf "%s\n" "$1" | grep -q "\"classification\""' _ "$OUT"
assert "H6: json carries 2 lane objects with the full expected key set" \
    bash -c 'printf "%s" "$1" | python3 "$2"' _ "$OUT" "$H_PY_KEYS"
assert "H7: json headroom object has resident=2" \
    bash -c 'printf "%s" "$1" | python3 "$2"' _ "$OUT" "$H_PY_HEADROOM"

# ──────────────────────────────────────────────────────────────────────────────
# Block I — B1 headline boundary (4-lane pool) + A3 exit-0 degradation
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block I: B1 headline boundary + A3 degradation ---"

I_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-i-XXXXXX)"
_TMPDIRS+=("$I_MOUNT")
I_MAP="$(mktemp /tmp/test-warm-lane-audit-i-map-XXXXXX)"
_TMPDIRS+=("$I_MAP")
printf '900 done\n901 pending\n902 pending\n' > "$I_MAP"

# (1) _lane-live: a live consumer holds the lane's exclusive flock (causal
# READY-marker handshake, no fixed sleep) -> ASSIGNED -> LIVE.
make_lane "$I_MOUNT/_lane-live"
_hold_lane_lock "$I_MOUNT" "_lane-live"
I_LOCK_PID="$LANE_LOCK_PID"

# (2) _lane-done: FREE, task/900, oracle=done, ahead-of-main -> RECLAIMABLE
# (via terminal status).
make_lane "$I_MOUNT/_lane-done" "task/900"
_add_ahead_commit "$I_MOUNT/_lane-done"

# (3) _lane-residue: FREE, task/901, oracle=pending (non-terminal),
# ahead-of-main, residue-only-dirty (data/queue/write_queue.db committed then
# modified in place) -> RECLAIMABLE (via residue-only-dirty, independent of
# status).
make_lane "$I_MOUNT/_lane-residue" "task/901"
_add_ahead_commit "$I_MOUNT/_lane-residue"
mkdir -p "$I_MOUNT/_lane-residue/data/queue"
printf 'db-v1\n' > "$I_MOUNT/_lane-residue/data/queue/write_queue.db"
git -C "$I_MOUNT/_lane-residue" add data/queue/write_queue.db
git -C "$I_MOUNT/_lane-residue" commit -q -m "add residue db"
printf 'db-v2\n' > "$I_MOUNT/_lane-residue/data/queue/write_queue.db"

# (4) _lane-leaked: FREE, task/902 then DETACHED (backing task resolved via
# the containing-branch enumeration path, not symbolic-ref -- this is the
# first fixture to exercise that path), oracle=pending (non-terminal),
# ahead-of-main/no-origin -> ORPHAN, aged past the default stale-age-min knob
# (60) -- built FIRST, backdated LAST (mirrors Block F) -> LEAKED.
make_lane "$I_MOUNT/_lane-leaked" "task/902"
_add_ahead_commit "$I_MOUNT/_lane-leaked"
git -C "$I_MOUNT/_lane-leaked" checkout -q --detach
touch -d '90 minutes ago' "$I_MOUNT/_lane-leaked"

ORACLE_MAP="$I_MAP" run_helper --mount "$I_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "I1: exit 0" test "$RC" -eq 0
assert "I2: _lane-live classification LIVE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-live .*classification=LIVE"' _ "$OUT"
assert "I3: _lane-done classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-done .*classification=RECLAIMABLE"' _ "$OUT"
assert "I4: _lane-residue classification RECLAIMABLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-residue .*classification=RECLAIMABLE"' _ "$OUT"
assert "I5: _lane-leaked classification LEAKED" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-leaked .*classification=LEAKED"' _ "$OUT"
assert "I6: HEADROOM resident=4" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "resident=4"' _ "$OUT"
assert "I7: HEADROOM live=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -qE "(^| )live=1( |$)"' _ "$OUT"
assert "I8: HEADROOM free=3" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "free=3"' _ "$OUT"
assert "I9: HEADROOM reclaimable=2" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "reclaimable=2"' _ "$OUT"
assert "I10: HEADROOM leaked=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "leaked=1"' _ "$OUT"
assert "I10a: HEADROOM leak_unknown=0 when the status oracle resolves cleanly" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "leak_unknown=0"' _ "$OUT"

# A3: a status oracle that exits non-zero degrades the task lanes to
# `unknown`; the run still exits 0; no lane is reclassified
# RECLAIMABLE-via-terminal (the previously-terminal _lane-done demotes to
# PRESERVED-OK, and the previously-LEAKED _lane-leaked -- whose backing task
# is resolved via the DETACHED-HEAD containing-branch path -- never falls
# into a false LEAKED under an unknown status); the residue-only-dirty
# _lane-residue's RECLAIMABLE holds regardless (that reclaim path is
# independent of status).
ORACLE_MAP="$I_MAP" run_helper --mount "$I_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle-fail.sh"

assert "I11: exit 0 (oracle failure degrades, never aborts)" test "$RC" -eq 0
assert "I12: _lane-done status=unknown under oracle failure" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-done .*status=unknown"' _ "$OUT"
assert "I13: _lane-done classification PRESERVED-OK (not reclassified RECLAIMABLE-via-terminal)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-done .*classification=PRESERVED-OK"' _ "$OUT"
assert "I14: _lane-leaked status=unknown under oracle failure" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-leaked .*status=unknown"' _ "$OUT"
assert "I15: _lane-leaked classification PRESERVED-OK (never falsely LEAKED under unknown status)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-leaked .*classification=PRESERVED-OK"' _ "$OUT"
assert "I16: _lane-residue classification still RECLAIMABLE (residue-only reclaim is status-independent)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-residue .*classification=RECLAIMABLE"' _ "$OUT"

# A3 observability: an unresolvable status that suppresses a would-be LEAKED
# verdict (_lane-leaked: ORPHAN + stale, but status=unknown under the failing
# oracle) must be surfaced -- not silently folded into "no leaks" -- via a
# distinct HEADROOM leak_unknown count and a stderr warning naming the lane.
# _lane-done is unknown+ORPHAN too but NOT stale, so it must NOT be counted.
assert "I16a: HEADROOM leak_unknown=1 under oracle failure (distinguishes 'no leaks' from 'unverifiable')" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -q "leak_unknown=1"' _ "$OUT"
assert "I16b: stderr warns _lane-leaked's LEAKED verdict is unconfirmable under unknown status" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-leaked.*status unknown"' _ "$ERR_OUT"

# A3: a failing df seam also degrades gracefully -- exit 0, HEADROOM line
# still emitted, free_gib/budget_gib degrade to 0 (never abort; PRD §9.5
# inv.12).
I_DF_FAIL_DIR="$(mktemp -d /tmp/test-warm-lane-audit-i-dffail-XXXXXX)"
_TMPDIRS+=("$I_DF_FAIL_DIR")
cat > "$I_DF_FAIL_DIR/df-fail.sh" << 'STUB_EOF'
#!/usr/bin/env bash
echo "df: simulated failure" >&2
exit 1
STUB_EOF
chmod +x "$I_DF_FAIL_DIR/df-fail.sh"

ORACLE_MAP="$I_MAP" REIFY_WARM_LANE_AUDIT_DF="$I_DF_FAIL_DIR/df-fail.sh" \
    run_helper --mount "$I_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "I17: exit 0 (df failure degrades, never aborts)" test "$RC" -eq 0
assert "I18: HEADROOM line still emitted under df failure" \
    bash -c 'printf "%s\n" "$1" | grep -q "^HEADROOM"' _ "$OUT"
assert "I19: HEADROOM free_gib=0 budget_gib=0 under df failure" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -qE "free_gib=0 budget_gib=0$"' _ "$OUT"

# Release the live-flock lock.
kill "$I_LOCK_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Block J — B2 read-only / byte-identical
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block J: B2 read-only / byte-identical ---"

J_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-j-XXXXXX)"
_TMPDIRS+=("$J_MOUNT")
J_MAP="$(mktemp /tmp/test-warm-lane-audit-j-map-XXXXXX)"
_TMPDIRS+=("$J_MAP")
printf '910 done\n911 pending\n912 pending\n' > "$J_MAP"

# The same B1-shape 4-lane pool as Block I (live/done/residue/leaked), so the
# read-only guarantee is proven across every git-state variety the script
# handles (clean, ahead-only, residue-dirty, detached-stale) -- PLUS a
# target/ directory on one lane (none of Block I's fixtures had one), so the
# content/size/mtime manifest check is exercised for real, not vacuously.
make_lane "$J_MOUNT/_lane-live"
_hold_lane_lock "$J_MOUNT" "_lane-live"
J_LOCK_PID="$LANE_LOCK_PID"

make_lane "$J_MOUNT/_lane-done" "task/910"
_add_ahead_commit "$J_MOUNT/_lane-done"
mkdir -p "$J_MOUNT/_lane-done/target"
printf 'built-artifact\n' > "$J_MOUNT/_lane-done/target/artifact.bin"

make_lane "$J_MOUNT/_lane-residue" "task/911"
_add_ahead_commit "$J_MOUNT/_lane-residue"
mkdir -p "$J_MOUNT/_lane-residue/data/queue"
printf 'db-v1\n' > "$J_MOUNT/_lane-residue/data/queue/write_queue.db"
git -C "$J_MOUNT/_lane-residue" add data/queue/write_queue.db
git -C "$J_MOUNT/_lane-residue" commit -q -m "add residue db"
printf 'db-v2\n' > "$J_MOUNT/_lane-residue/data/queue/write_queue.db"

make_lane "$J_MOUNT/_lane-leaked" "task/912"
_add_ahead_commit "$J_MOUNT/_lane-leaked"
git -C "$J_MOUNT/_lane-leaked" checkout -q --detach
touch -d '90 minutes ago' "$J_MOUNT/_lane-leaked"

# _snapshot_lane <lane_dir> — prints HEAD sha, the full `git status
# --porcelain` (untracked entries included, so an unexpected new/removed
# file anywhere in the tree -- not just target/ -- would show up), and a
# sorted path/size/mtime manifest of target/ (or a sentinel when absent).
# Two snapshots of the SAME lane compare byte-identical iff nothing in its
# git state or target/ changed -- the A1/B2 read-only proof. This only
# compares two point-in-time strings for equality -- no wall-clock bound
# anywhere (DD5).
_snapshot_lane() {
    local dir="$1"
    printf 'HEAD=%s\n' "$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo NONE)"
    printf '== status ==\n'
    git -C "$dir" status --porcelain 2>/dev/null
    printf '== target manifest ==\n'
    if [ -e "$dir/target" ]; then
        find "$dir/target" -printf '%p %s %T@\n' 2>/dev/null | sort
    else
        printf '(no target)\n'
    fi
}

J_LANES=(_lane-live _lane-done _lane-residue _lane-leaked)
declare -A J_BEFORE
for j_lane in "${J_LANES[@]}"; do
    J_BEFORE["$j_lane"]="$(_snapshot_lane "$J_MOUNT/$j_lane")"
done

# A FREE lane (_lane-done) with NO pre-created <dir>.lock -- confirm the
# probe never creates one (A1: a missing lock is FREE and is NEVER created).
assert "J0: _lane-done has no .lock before the run" \
    bash -c '[ ! -e "$1" ]' _ "$J_MOUNT/_lane-done.lock"

ORACLE_MAP="$J_MAP" run_helper --mount "$J_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "J1: exit 0" test "$RC" -eq 0

for j_lane in "${J_LANES[@]}"; do
    j_after="$(_snapshot_lane "$J_MOUNT/$j_lane")"
    assert "J2: $j_lane is byte-identical before/after the audit run (A1/B2)" \
        bash -c '[ "$1" = "$2" ]' _ "${J_BEFORE[$j_lane]}" "$j_after"
done

assert "J3: _lane-done still has no .lock after the run (probe never creates a sidecar)" \
    bash -c '[ ! -e "$1" ]' _ "$J_MOUNT/_lane-done.lock"

# _is_locked <lock_file> — a TEST-side (not script-under-test) non-blocking
# probe, entirely within an isolated subshell so its fd manipulation never
# leaks into this test script's own shell: exit 0 if the lock is currently
# held by someone else, exit 1 if free/missing. Proves A2 -- the audit's own
# probe released its own fd immediately and never stole or retained the
# background job's lock.
_is_locked() {
    local lock="$1"
    (
        exec 8<"$lock" 2>/dev/null || exit 1
        if flock -n -x 8 2>/dev/null; then
            flock -u 8
            exit 1
        else
            exit 0
        fi
    )
}
assert "J4: _lane-live's flock is still held by the background job after the run (A2)" \
    _is_locked "$J_MOUNT/_lane-live.lock"

# Release the live-flock lock.
kill "$J_LOCK_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Block K — the flock probe is a LIVENESS column (live=LIVE|IDLE), not an
#           assignment one
# ──────────────────────────────────────────────────────────────────────────────
# The <lane>.lock flock probe measures exactly one thing: whether a consumer
# PROCESS is running and holding the lane's exclusive lock right now. Reporting
# that under the key `assigned=` conflated liveness with the orchestrator's
# assignment state, which is the category error behind both pool misreads this
# task exists to fix. Here the probe's own column is asserted under its honest
# name; `assigned=` is re-sourced from the real assignment record in Block L.
echo ""
echo "--- Block K: flock probe reports liveness (live=LIVE|IDLE) ---"

K_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-k-XXXXXX)"
_TMPDIRS+=("$K_MOUNT")

# _lane-live: a live consumer holds the exclusive flock (causal handshake).
make_lane "$K_MOUNT/_lane-live"
_hold_lane_lock "$K_MOUNT" "_lane-live"
K_LOCK_PID="$LANE_LOCK_PID"

# _lane-idle: an unheld, pre-created lock file -- no consumer process.
make_lane "$K_MOUNT/_lane-idle"
touch "$K_MOUNT/_lane-idle.lock"

run_helper --mount "$K_MOUNT"

assert "K1: exit 0" test "$RC" -eq 0
assert "K2: _lane-live row reports live=LIVE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-live .*live=LIVE"' _ "$OUT"
assert "K3: _lane-idle row reports live=IDLE" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-idle .*live=IDLE"' _ "$OUT"
assert "K4: HEADROOM reports live=1" \
    bash -c 'printf "%s\n" "$1" | grep "^HEADROOM" | grep -qE "(^| )live=1( |$)"' _ "$OUT"
# The retired value must be GONE, not silently repurposed: any unknown consumer
# still grepping `assigned=FREE` now matches nothing (fail-loud) rather than
# matching a string whose meaning changed underneath it.
assert "K5: the retired value 'assigned=FREE' appears nowhere in the output" \
    bash -c '! printf "%s\n" "$1" | grep -q "assigned=FREE"' _ "$OUT"

K_PY_LIVE="$(mktemp /tmp/test-warm-lane-audit-k-live-XXXXXX.py)"
_TMPDIRS+=("$K_PY_LIVE")
cat > "$K_PY_LIVE" << 'PYEOF'
import json, sys
data = json.load(sys.stdin)
by_lane = {o["lane"]: o for o in data["lanes"]}
assert set(by_lane) == {"_lane-live", "_lane-idle"}, sorted(by_lane)
for name, obj in by_lane.items():
    assert "live" in obj, f"{name} has no 'live' key: {sorted(obj)}"
assert by_lane["_lane-live"]["live"] == "LIVE", by_lane["_lane-live"]
assert by_lane["_lane-idle"]["live"] == "IDLE", by_lane["_lane-idle"]
PYEOF

run_helper --mount "$K_MOUNT" --format json
assert "K6a: exit 0 (json)" test "$RC" -eq 0
assert "K6b: json lane objects carry a 'live' key with the same LIVE/IDLE values" \
    bash -c 'printf "%s" "$1" | python3 "$2"' _ "$OUT" "$K_PY_LIVE"

# Release the live-flock lock.
kill "$K_LOCK_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Block L — `assigned=` is sourced from the orchestrator's OWN assignment
#           record at <state-dir>/<lane>.json
# ──────────────────────────────────────────────────────────────────────────────
# Liveness (Block K) cannot answer "has the pool reserved this lane?". Only the
# orchestrator's durable record can, so `assigned=` reads it directly. Every
# LaneState the producer can write is pinned here to its reported column, and
# every way the read can fail degrades to UNKNOWN with exit 0 -- the read is
# advisory, so it must never abort and must never invent an assignment.
echo ""
echo "--- Block L: assigned= from the .lane-state record ---"

L_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-l-XXXXXX)"
_TMPDIRS+=("$L_MOUNT")

# lane:raw-record-state:expected-column:expected-unknown-cause. The raw states
# are exactly dark-factory's LaneState enum (lane_lifecycle.py: seed/registered/
# assigned/in_use/released/quarantined) plus the three unresolvable cases. The
# cause field is `-` for a resolvable record, and otherwise the A5 cause the
# stderr warning must name -- the three UNKNOWN lanes are UNKNOWN for three
# DIFFERENT reasons, and a warning that cannot tell them apart sends triage
# after a missing file that is sitting right there.
#
# The cause is LAST because `unrecognized-state:wat` itself contains the
# delimiter: `read` assigns the unsplit remainder to its final variable.
L_CASES=(
    "_lane-assigned:assigned:ASSIGNED:-"
    "_lane-inuse:in_use:ASSIGNED:-"
    "_lane-released:released:RELEASED:-"
    "_lane-seed:seed:RELEASED:-"
    "_lane-registered:registered:RELEASED:-"
    "_lane-quarantined:quarantined:QUARANTINED:-"
    "_lane-norecord::UNKNOWN:no-readable-record"
    "_lane-corrupt:CORRUPT:UNKNOWN:unparseable-record"
    "_lane-badstate:wat:UNKNOWN:unrecognized-state:wat"
    "_lane-compact:COMPACT:ASSIGNED:-"
)

for l_case in "${L_CASES[@]}"; do
    IFS=':' read -r l_lane l_state l_expected l_cause <<< "$l_case"
    make_lane "$L_MOUNT/$l_lane"
    case "$l_state" in
        "")        : ;;  # no record at all
        CORRUPT)   make_lane_state_raw "$L_MOUNT" "$l_lane" 'not json{' ;;
        # A compact, single-line record: the reader must not depend on the
        # producer's current indent=2 pretty-printing.
        COMPACT)   make_lane_state_raw "$L_MOUNT" "$l_lane" \
                       '{"state":"assigned","task_id":"7001","title":null,"branch":null,"seeded_from_sha":null,"updated_at":"2026-07-26T12:43:10.704531+00:00"}' ;;
        *)         make_lane_state "$L_MOUNT" "$l_lane" "$l_state" ;;
    esac
done

# Every aggregate below is DERIVED from L_CASES, never typed: no lane in this
# block is live, so the four-way partition collapses onto the assigned column
# alone (live=0, ASSIGNED=>pinned, QUARANTINED=>quarantined, the rest=>free).
# That makes `free` the direct proof of A5's conservative-accounting claim --
# an UNKNOWN lane is counted free -- for all three unresolvable causes, not
# just the no-record one.
l_exp_live=0
l_exp_pinned=0
l_exp_quarantined=0
l_exp_free=0
l_exp_state_unknown=0
for l_case in "${L_CASES[@]}"; do
    IFS=':' read -r l_lane l_state l_expected l_cause <<< "$l_case"
    case "$l_expected" in
        ASSIGNED)    l_exp_pinned=$(( l_exp_pinned + 1 )) ;;
        QUARANTINED) l_exp_quarantined=$(( l_exp_quarantined + 1 )) ;;
        RELEASED)    l_exp_free=$(( l_exp_free + 1 )) ;;
        UNKNOWN)     l_exp_free=$(( l_exp_free + 1 ))
                     l_exp_state_unknown=$(( l_exp_state_unknown + 1 )) ;;
    esac
done
l_exp_resident="${#L_CASES[@]}"

run_helper --mount "$L_MOUNT"

assert "L1: exit 0 over a pool spanning every LaneState + every unresolvable record" \
    test "$RC" -eq 0

for l_case in "${L_CASES[@]}"; do
    IFS=':' read -r l_lane l_state l_expected l_cause <<< "$l_case"
    assert "L2: $l_lane reports assigned=$l_expected" \
        bash -c 'printf "%s\n" "$1" | grep -q "lane=$2 .*assigned=$3"' _ "$OUT" "$l_lane" "$l_expected"
done

# The state dir is dot-prefixed, so the resident glob must already skip it --
# asserted rather than assumed, since a counted .lane-state would inflate every
# headroom figure. resident is DERIVED from the fixture array, never typed.
assert "L3: HEADROOM resident==seeded lane count (.lane-state is not a resident)" \
    bash -c '[ "$1" = "$2" ]' _ "$(_headroom_field "$OUT" resident)" "$l_exp_resident"

# L4: the pool-wide figures, so each of the three unresolvable causes is proven
# at the HEADROOM cross-cut too -- not only at its own row's column. Without
# `state_unknown` here, a reader that resolved the column correctly but forgot
# to COUNT the corrupt/unrecognized lanes would pass every per-lane assertion.
L_FIELDS=(
    "live:$l_exp_live"
    "pinned:$l_exp_pinned"
    "quarantined:$l_exp_quarantined"
    "free:$l_exp_free"
    "state_unknown:$l_exp_state_unknown"
)
for l_f in "${L_FIELDS[@]}"; do
    l_key="${l_f%%:*}"
    l_want="${l_f##*:}"
    assert "L4: HEADROOM $l_key=$l_want (derived from L_CASES)" \
        bash -c '[ "$1" = "$2" ]' _ "$(_headroom_field "$OUT" "$l_key")" "$l_want"
done
assert "L5: partition identity holds across every LaneState + unresolvable record" \
    _partition_holds \
    "$(_headroom_field "$OUT" resident)" \
    "$(_headroom_field "$OUT" live)" \
    "$(_headroom_field "$OUT" pinned)" \
    "$(_headroom_field "$OUT" quarantined)" \
    "$(_headroom_field "$OUT" free)"

# L6: A5's stderr warning must name the lane AND the cause that actually fired.
# Asserting the DISTINCT text per lane is what keeps the three causes separable:
# a single hardcoded message satisfies "a warning was emitted" while telling an
# operator to go look for a file that is present and readable.
for l_case in "${L_CASES[@]}"; do
    IFS=':' read -r l_lane l_state l_expected l_cause <<< "$l_case"
    [ "$l_expected" = "UNKNOWN" ] || continue
    assert "L6: stderr names $l_lane with cause ($l_cause)" \
        bash -c 'printf "%s\n" "$1" | grep -qF "lane=$2: assignment state unknown ($3)"' \
        _ "$ERR_OUT" "$l_lane" "$l_cause"
done

# L7: `pin` is gated on assigned==ASSIGNED, so the QUARANTINED and UNKNOWN arms
# of that gate must report `-` just as the LIVE (M5) and RELEASED (M6) arms do.
# A gate that leaked a pin for a withheld or unresolvable lane would name a
# holder for a lane nothing has reserved.
for l_case in "${L_CASES[@]}"; do
    IFS=':' read -r l_lane l_state l_expected l_cause <<< "$l_case"
    case "$l_expected" in
        QUARANTINED|UNKNOWN) : ;;
        *) continue ;;
    esac
    assert "L7: $l_lane is $l_expected, not ASSIGNED -- not pinned (pin=-)" \
        bash -c 'printf "%s\n" "$1" | grep -qE "lane=$2 .*pin=-( |\$)"' _ "$OUT" "$l_lane"
done

# L8: the JSON emitter's own `assigned` VALUES. The table row and the JSON
# object are two separate interpolations, so a JSON path that hardcoded the
# column or interpolated the wrong variable would satisfy every table-side
# assertion above plus H6's key-set check. `assigned` is the column this whole
# change exists to add; its JSON values must be pinned, not merely present.
L_PY_ASSIGNED="$(mktemp /tmp/test-warm-lane-audit-l-assigned-XXXXXX.py)"
_TMPDIRS+=("$L_PY_ASSIGNED")
cat > "$L_PY_ASSIGNED" << 'PYEOF'
import json, sys
data = json.load(sys.stdin)
by_lane = {o["lane"]: o for o in data["lanes"]}
want = dict(kv.split("=", 1) for kv in sys.argv[1:])
assert set(by_lane) == set(want), (sorted(by_lane), sorted(want))
for lane, expected in want.items():
    got = by_lane[lane].get("assigned")
    assert got == expected, f"{lane}: assigned == {got!r}, want {expected!r}"
# Not every lane may carry the same value -- a JSON path that hardcoded one
# constant would otherwise satisfy a fixture that happened to be uniform.
assert len(set(want.values())) > 1, want
PYEOF

L_PY_ARGS=()
for l_case in "${L_CASES[@]}"; do
    IFS=':' read -r l_lane l_state l_expected l_cause <<< "$l_case"
    L_PY_ARGS+=("$l_lane=$l_expected")
done
run_helper --mount "$L_MOUNT" --format json
assert "L8a: exit 0 (json)" test "$RC" -eq 0
assert "L8b: json lane objects carry the same assigned= values as the table rows" \
    bash -c 'printf "%s" "$1" | python3 "$2" "${@:3}"' _ "$OUT" "$L_PY_ASSIGNED" "${L_PY_ARGS[@]}"

# ── L9: a present-but-UNREADABLE record is the third way _lane_record's
# readability guard can fire, and the only one no other fixture reaches.
# Skipped under root, for whom mode 000 is not a barrier.
if [ "$(id -u)" -ne 0 ]; then
    L9_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-l9-XXXXXX)"
    _TMPDIRS+=("$L9_MOUNT")
    make_lane "$L9_MOUNT/_lane-unreadable"
    make_lane_state "$L9_MOUNT" "_lane-unreadable" "assigned" "7200"
    chmod 000 "$L9_MOUNT/.lane-state/_lane-unreadable.json"
    run_helper --mount "$L9_MOUNT"
    # Restore before any assertion can abort the suite, so cleanup's rm -rf works.
    chmod 644 "$L9_MOUNT/.lane-state/_lane-unreadable.json"
    assert "L9a: exit 0 with an unreadable record (degrades, never aborts)" test "$RC" -eq 0
    assert "L9b: an unreadable record reports assigned=UNKNOWN (never the state inside it)" \
        bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-unreadable .*assigned=UNKNOWN"' _ "$OUT"
    assert "L9c: stderr names the unreadable record's cause" \
        bash -c 'printf "%s\n" "$1" | grep -qF "lane=_lane-unreadable: assignment state unknown (no-readable-record)"' \
        _ "$ERR_OUT"
else
    echo "  SKIP: L9 (running as root — mode 000 does not make a file unreadable)"
fi

# ── L10: with NO status oracle at all, a pinned lane's holder is unresolvable.
# That is _task_status_raw's `unset STATUS_CMD` early return, a DIFFERENT branch
# from M9's failing-oracle path, and it must reach the same pin=unknown sentinel
# rather than a blank column or an aborted run.
REIFY_LANE_LEAK_STATUS_CMD= run_helper --mount "$L_MOUNT"
assert "L10a: exit 0 with no status oracle configured at all" test "$RC" -eq 0
assert "L10b: a pinned lane with no oracle reports pin=unknown (never blank)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "lane=_lane-assigned .*pin=unknown( |\$)"' _ "$OUT"

# ── L11: REIFY_WARM_LANE_AUDIT_STATE_DIR override, pointed OUTSIDE the mount ─
# Run twice against the SAME lane: once bare (no record findable -> UNKNOWN),
# once with the override (record found -> ASSIGNED). The pair proves the
# override is what resolved the state, not a coincidence of the fixture.
L11_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-l11-XXXXXX)"
_TMPDIRS+=("$L11_MOUNT")
L11_STATE="$(mktemp -d /tmp/test-warm-lane-audit-l11-state-XXXXXX)"
_TMPDIRS+=("$L11_STATE")
make_lane "$L11_MOUNT/_lane-ext"
# make_lane_state writes <arg>/.lane-state/<lane>.json, so pass the parent and
# point the override at the .lane-state dir it creates.
make_lane_state "$L11_STATE" "_lane-ext" "assigned" "7100" "task/7100"

run_helper --mount "$L11_MOUNT"
assert "L11a: without the override the out-of-mount record is not found (assigned=UNKNOWN)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-ext .*assigned=UNKNOWN"' _ "$OUT"

REIFY_WARM_LANE_AUDIT_STATE_DIR="$L11_STATE/.lane-state" run_helper --mount "$L11_MOUNT"
assert "L11b: exit 0 under REIFY_WARM_LANE_AUDIT_STATE_DIR" test "$RC" -eq 0
assert "L11c: REIFY_WARM_LANE_AUDIT_STATE_DIR outside the mount is honoured (assigned=ASSIGNED)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-ext .*assigned=ASSIGNED"' _ "$OUT"

# ── L12: a state dir that is absent ENTIRELY warns ONCE for the directory ─────
# The default shape of every pool until dark-factory's LaneLifecycle has written
# its first record: every resident resolves the SAME cause, so a per-lane line
# for each is N copies of "the feature is not deployed here" -- and it drowns
# the per-lane naming, whose whole value is saying "THIS lane, unlike its
# neighbours". The count is what makes this a real assertion: one warning for a
# multi-lane pool cannot be satisfied by the retired per-lane loop.
L12_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-l12-XXXXXX)"
_TMPDIRS+=("$L12_MOUNT")
L12_LANES=(_lane-l12-a _lane-l12-b _lane-l12-c)
for l12_lane in "${L12_LANES[@]}"; do
    make_lane "$L12_MOUNT/$l12_lane"
done
l12_exp_resident="${#L12_LANES[@]}"

run_helper --mount "$L12_MOUNT"
assert "L12a: exit 0 with no state dir at all" test "$RC" -eq 0
assert "L12b: every lane still reports state_unknown (accounting stays PER LANE)" \
    bash -c '[ "$1" = "$2" ]' _ "$(_headroom_field "$OUT" state_unknown)" "$l12_exp_resident"
assert "L12c: exactly ONE warning names the absent dir (not one per lane)" \
    bash -c '[ "$(printf "%s\n" "$1" | grep -cF "state dir $2 does not exist")" = "1" ]' \
    _ "$ERR_OUT" "$L12_MOUNT/.lane-state"
# The PER-LANE form is `lane=<name>: assignment state unknown (...)`. The
# dir-level line deliberately reuses the "assignment state unknown" phrase (an
# operator greps the concept, and must hit something in BOTH shapes), so the
# discriminator here is the `lane=` prefix, not the phrase.
assert "L12d: no per-lane warning fires when the whole dir is missing" \
    bash -c '! printf "%s\n" "$1" | grep -qE "lane=[^ ]+: assignment state unknown"' _ "$ERR_OUT"
# The converse, so L12d cannot be satisfied by suppressing the per-lane warning
# outright: with the dir PRESENT, a single missing record still names its lane.
# (L6 asserts the cause text; this asserts the dir-level line stays absent.)
make_lane_state "$L12_MOUNT" "${L12_LANES[0]}" "assigned" "7300"
run_helper --mount "$L12_MOUNT"
assert "L12e: with the dir present, the missing records warn PER LANE again" \
    bash -c '[ "$(printf "%s\n" "$1" | grep -cE "lane=[^ ]+: assignment state unknown")" = "$2" ]' \
    _ "$ERR_OUT" "$(( l12_exp_resident - 1 ))"
assert "L12f: ...and the dir-level warning is not emitted when the dir exists" \
    bash -c '! printf "%s\n" "$1" | grep -qF "state dir $2 does not exist"' \
    _ "$ERR_OUT" "$L12_MOUNT/.lane-state"

# ──────────────────────────────────────────────────────────────────────────────
# Block M — `pin=`: WHO is holding a reserved-but-idle lane, and in what state
# ──────────────────────────────────────────────────────────────────────────────
# A lane that is ASSIGNED but not LIVE is reserved by a task that is not
# running. `pin=` names that task's RAW backing status, because that raw value
# is exactly what an operator triaging one specific lane needs: `in-progress`
# with no live consumer means a crashed agent, `done` means a terminal task
# still holding a reservation, `pending` means work that never started. The
# fixed bucketing exists only in the aggregate rollup (Block O), never here.
echo ""
echo "--- Block M: pin= (raw backing status of a reserved-but-idle lane) ---"

M_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-m-XXXXXX)"
_TMPDIRS+=("$M_MOUNT")
M_MAP="$(mktemp /tmp/test-warm-lane-audit-m-map-XXXXXX)"
_TMPDIRS+=("$M_MAP")
printf '8001 pending\n8002 infra-hold\n8003 blocked\n8004 in-progress\n8005 done\n8006 done\n8007 pending\n8008 blocked\n8009 pending\n8010 pending\n' > "$M_MAP"

# Raw-fidelity cases: idle + ASSIGNED, pin id carried by the RECORD. These
# lanes stay on main (no task/ branch), so the record is the ONLY possible
# source of the id -- the branch cannot silently supply it.
M_RAW_CASES=(
    "_lane-pin-pending:8001:pending"
    "_lane-pin-infrahold:8002:infra-hold"
    "_lane-pin-blocked:8003:blocked"
    "_lane-pin-inprogress:8004:in-progress"
    "_lane-pin-done:8005:done"
)
for m_case in "${M_RAW_CASES[@]}"; do
    m_lane="${m_case%%:*}"
    m_rest="${m_case#*:}"
    m_id="${m_rest%%:*}"
    make_lane "$M_MOUNT/$m_lane"
    make_lane_state "$M_MOUNT" "$m_lane" "assigned" "$m_id"
done

# Record-vs-branch disagreement: branch says task/8006 (done), record says
# 8007 (pending). The RECORD is the authoritative pin holder -- a lane's
# branch name can be stale, the reservation record cannot.
make_lane "$M_MOUNT/_lane-pin-mismatch" "task/8006"
make_lane_state "$M_MOUNT" "_lane-pin-mismatch" "assigned" "8007" "task/8007"

# Record with an explicitly null task_id -> fall back to the branch-derived id.
make_lane "$M_MOUNT/_lane-pin-nullid" "task/8008"
make_lane_state "$M_MOUNT" "_lane-pin-nullid" "assigned"

# A LIVE lane is not pinned (someone IS using it) -> pin=-
make_lane "$M_MOUNT/_lane-pin-live"
make_lane_state "$M_MOUNT" "_lane-pin-live" "assigned" "8009"
_hold_lane_lock "$M_MOUNT" "_lane-pin-live"
M_LOCK_PID="$LANE_LOCK_PID"

# A RELEASED lane is not pinned (nothing reserves it) -> pin=-
make_lane "$M_MOUNT/_lane-pin-released"
make_lane_state "$M_MOUNT" "_lane-pin-released" "released" "8010"

ORACLE_MAP="$M_MAP" run_helper --mount "$M_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "M1: exit 0" test "$RC" -eq 0

for m_case in "${M_RAW_CASES[@]}"; do
    m_lane="${m_case%%:*}"
    m_expected="${m_case##*:}"
    assert "M2: $m_lane reports the RAW status pin=$m_expected (no per-lane bucketing)" \
        bash -c 'printf "%s\n" "$1" | grep -qE "lane=$2 .*pin=$3( |\$)"' _ "$OUT" "$m_lane" "$m_expected"
done

assert "M3: record task_id wins over a disagreeing branch id (pin=pending, not done)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "lane=_lane-pin-mismatch .*pin=pending( |\$)"' _ "$OUT"
assert "M4: a null record task_id falls back to the branch-derived id (pin=blocked)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "lane=_lane-pin-nullid .*pin=blocked( |\$)"' _ "$OUT"
assert "M5: a LIVE lane is not pinned (pin=-)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "lane=_lane-pin-live .*pin=-( |\$)"' _ "$OUT"
assert "M6: a RELEASED lane is not pinned (pin=-)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "lane=_lane-pin-released .*pin=-( |\$)"' _ "$OUT"

M_PY_PIN="$(mktemp /tmp/test-warm-lane-audit-m-pin-XXXXXX.py)"
_TMPDIRS+=("$M_PY_PIN")
cat > "$M_PY_PIN" << 'PYEOF'
import json, sys
data = json.load(sys.stdin)
by_lane = {o["lane"]: o for o in data["lanes"]}
for name, obj in by_lane.items():
    assert "pin" in obj, f"{name} has no 'pin' key: {sorted(obj)}"
assert by_lane["_lane-pin-inprogress"]["pin"] == "in-progress", by_lane["_lane-pin-inprogress"]
assert by_lane["_lane-pin-live"]["pin"] == "-", by_lane["_lane-pin-live"]
PYEOF

ORACLE_MAP="$M_MAP" run_helper --mount "$M_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh" --format json
assert "M7: json lane objects carry a 'pin' key with the same raw values" \
    bash -c 'printf "%s" "$1" | python3 "$2"' _ "$OUT" "$M_PY_PIN"

# A failing oracle must not abort and must not invent a status: a pinned lane
# whose holder cannot be resolved reports pin=unknown (A3's treatment, applied
# to the pin column).
ORACLE_MAP="$M_MAP" run_helper --mount "$M_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle-fail.sh"
assert "M8: exit 0 under a failing status oracle" test "$RC" -eq 0
assert "M9: an unresolvable pin holder reports pin=unknown (never invented)" \
    bash -c 'printf "%s\n" "$1" | grep -qE "lane=_lane-pin-pending .*pin=unknown( |\$)"' _ "$OUT"

kill "$M_LOCK_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Block N — classification=PINNED + the HEADROOM four-way partition
# ──────────────────────────────────────────────────────────────────────────────
# A lane the pool has RESERVED but which no consumer process is running against
# is neither live nor free: it is PINNED, and it is a standing capacity loss.
# The old single ASSIGNED/FREE column had no way to say that, which is why 53
# reserved lanes read as free on 2026-07-22. PINNED becomes a first-class
# classification ranked immediately below LIVE, and the HEADROOM line becomes a
# genuine PARTITION -- resident = live + pinned + quarantined + free -- so a
# pinned lane can never again be counted as free.
echo ""
echo "--- Block N: classification=PINNED + the HEADROOM partition ---"

N_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-n-XXXXXX)"
_TMPDIRS+=("$N_MOUNT")
N_MAP="$(mktemp /tmp/test-warm-lane-audit-n-map-XXXXXX)"
_TMPDIRS+=("$N_MAP")
printf '9101 pending\n9102 pending\n9103 done\n9104 pending\n9105 pending\n9106 pending\n9107 pending\n' > "$N_MAP"

# lane : liveness : record state ('' = no record at all) : task id : git shape :
#        expected assigned column : expected classification
#
# `orphan-stale` builds the LEAKED predicate's EXACT shape (non-terminal status
# + ahead-of-main with no origin => ORPHAN + aged past the default 60-minute
# knob). Three lanes here are byte-for-byte identical in that respect and differ
# ONLY in their assignment record -- pinned / quarantined / leaked -- which is
# what makes "the reserved one reports PINNED, the withheld one QUARANTINED, and
# only the unreserved one is still LEAKED" a real discrimination rather than a
# fixture artifact. Dropping the LEAKED control would leave nothing proving the
# predicate can still fire at all, so the two suppressions would pass vacuously.
#
# _lane-n-live-unknown is the LIVE x UNKNOWN partition cell: a live consumer on a
# lane with no readable record. It is counted `live`, NOT `free`, so a warning
# that claims a fixed accounting contradicts the HEADROOM line beside it.
N_CASES=(
    "_lane-n-live:LIVE:assigned:9101:plain:ASSIGNED:LIVE"
    "_lane-n-pinned:IDLE:assigned:9102:orphan-stale:ASSIGNED:PINNED"
    "_lane-n-released:IDLE:released:9103:plain:RELEASED:RECLAIMABLE"
    "_lane-n-quarantined:IDLE:quarantined:9104:orphan-stale:QUARANTINED:QUARANTINED"
    "_lane-n-unknown:IDLE::9105:plain:UNKNOWN:RECLAIMABLE"
    "_lane-n-live-unknown:LIVE::9106:plain:UNKNOWN:LIVE"
    "_lane-n-leaked:IDLE:released:9107:orphan-stale:RELEASED:LEAKED"
)

N_LIVE_PIDS=()
for n_case in "${N_CASES[@]}"; do
    IFS=':' read -r n_lane n_liveness n_state n_task n_shape n_assigned n_class <<< "$n_case"
    make_lane "$N_MOUNT/$n_lane" "task/$n_task"
    if [ "$n_shape" = "orphan-stale" ]; then
        _add_ahead_commit "$N_MOUNT/$n_lane"
    fi
    if [ -n "$n_state" ]; then
        make_lane_state "$N_MOUNT" "$n_lane" "$n_state" "$n_task" "task/$n_task"
    fi
    if [ "$n_liveness" = "LIVE" ]; then
        _hold_lane_lock "$N_MOUNT" "$n_lane"
        N_LIVE_PIDS+=("$LANE_LOCK_PID")
    else
        touch "$N_MOUNT/$n_lane.lock"
    fi
    # Backdate LAST for this lane (mirrors Block F/I): every git and lock
    # mutation above is done, and nothing a later iteration does touches this
    # lane's directory (records live in <mount>/.lane-state, locks in
    # <mount>/<lane>.lock -- neither is inside the lane dir).
    if [ "$n_shape" = "orphan-stale" ]; then
        touch -d '90 minutes ago' "$N_MOUNT/$n_lane"
    fi
done

# Every expected count is DERIVED from N_CASES, never typed as a literal: a
# fixture edit that changes the pool shape must move the expectations with it.
n_exp_live=0
n_exp_pinned=0
n_exp_quarantined=0
n_exp_free=0
n_exp_assigned=0
n_exp_state_unknown=0
n_exp_leaked=0
n_exp_reclaimable=0
n_exp_live_unknown_lanes=()
n_exp_idle_unknown_lanes=()
for n_case in "${N_CASES[@]}"; do
    IFS=':' read -r n_lane n_liveness n_state n_task n_shape n_assigned n_class <<< "$n_case"
    # The partition is ORDERED and mutually exclusive -- live > pinned >
    # quarantined > free -- mirroring the classification rank, so the four
    # buckets sum to resident by construction rather than by coincidence.
    if [ "$n_liveness" = "LIVE" ]; then
        n_exp_live=$((n_exp_live + 1))
    elif [ "$n_assigned" = "ASSIGNED" ]; then
        n_exp_pinned=$((n_exp_pinned + 1))
    elif [ "$n_assigned" = "QUARANTINED" ]; then
        n_exp_quarantined=$((n_exp_quarantined + 1))
    else
        n_exp_free=$((n_exp_free + 1))
    fi
    # Cross-cuts: independent of the partition, so they may overlap it.
    if [ "$n_assigned" = "ASSIGNED" ]; then
        n_exp_assigned=$((n_exp_assigned + 1))
    fi
    if [ "$n_assigned" = "UNKNOWN" ]; then
        n_exp_state_unknown=$((n_exp_state_unknown + 1))
        # Split by liveness: A5's warning must report the bucket the lane was
        # ACTUALLY counted in, and the two arms differ (see N5).
        if [ "$n_liveness" = "LIVE" ]; then
            n_exp_live_unknown_lanes+=("$n_lane")
        else
            n_exp_idle_unknown_lanes+=("$n_lane")
        fi
    fi
    case "$n_class" in
        LEAKED)      n_exp_leaked=$((n_exp_leaked + 1)) ;;
        RECLAIMABLE) n_exp_reclaimable=$((n_exp_reclaimable + 1)) ;;
    esac
done
n_exp_resident="${#N_CASES[@]}"

ORACLE_MAP="$N_MAP" run_helper --mount "$N_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "N1: exit 0" test "$RC" -eq 0

for n_case in "${N_CASES[@]}"; do
    IFS=':' read -r n_lane n_liveness n_state n_task n_shape n_assigned n_class <<< "$n_case"
    assert "N2: $n_lane reports classification=$n_class" \
        bash -c 'printf "%s\n" "$1" | grep -qE "lane=$2 .*classification=$3( |\$)"' \
        _ "$OUT" "$n_lane" "$n_class"
done

# N3 covers both the partition fields and the two cross-cuts, plus `leaked` --
# whose expected value (1, not 2) is the assertion that a PINNED lane matching
# the LEAKED predicate is NOT additionally counted as a leak.
N_FIELDS=(
    "resident:$n_exp_resident"
    "live:$n_exp_live"
    "pinned:$n_exp_pinned"
    "quarantined:$n_exp_quarantined"
    "free:$n_exp_free"
    "assigned:$n_exp_assigned"
    "state_unknown:$n_exp_state_unknown"
    "leaked:$n_exp_leaked"
    "reclaimable:$n_exp_reclaimable"
)
for n_f in "${N_FIELDS[@]}"; do
    n_key="${n_f%%:*}"
    n_want="${n_f##*:}"
    n_got="$(_headroom_field "$OUT" "$n_key")"
    assert "N3: HEADROOM $n_key=$n_want (derived from N_CASES)" \
        bash -c '[ "$1" = "$2" ]' _ "$n_got" "$n_want"
done

assert "N4: partition identity resident == live + pinned + quarantined + free" \
    _partition_holds \
    "$(_headroom_field "$OUT" resident)" \
    "$(_headroom_field "$OUT" live)" \
    "$(_headroom_field "$OUT" pinned)" \
    "$(_headroom_field "$OUT" quarantined)" \
    "$(_headroom_field "$OUT" free)"

# A5 observability, mirroring A3's leak_unknown treatment: a lane whose
# assignment state could not be resolved is named on stderr so "no pins" stays
# distinguishable from "pins could not be evaluated".
#
# The ACCOUNTING CLAUSE is asserted per liveness arm, not as one fixed string.
# "counted free (conservative)" is true only of an IDLE unknown lane; a LIVE one
# is counted in `live`, and a warning claiming otherwise contradicts the
# HEADROOM line printed directly beneath it. N3's derived live/free counts prove
# which bucket the script really used; these two prove the warning agrees.
for n_lane in "${n_exp_idle_unknown_lanes[@]+${n_exp_idle_unknown_lanes[@]}}"; do
    assert "N5a: stderr names $n_lane (idle, unknown) as counted free (conservative)" \
        bash -c 'printf "%s\n" "$1" | grep -qF "lane=$2: assignment state unknown (no-readable-record) at $3/$2.json; counted free (conservative)."' \
        _ "$ERR_OUT" "$n_lane" "$N_MOUNT/.lane-state"
done
for n_lane in "${n_exp_live_unknown_lanes[@]+${n_exp_live_unknown_lanes[@]}}"; do
    assert "N5b: stderr names $n_lane (live, unknown) as counted live -- never free" \
        bash -c 'printf "%s\n" "$1" | grep -qF "lane=$2: assignment state unknown (no-readable-record) at $3/$2.json; counted live."' \
        _ "$ERR_OUT" "$n_lane" "$N_MOUNT/.lane-state"
done
# ...and the fixture must actually exercise BOTH arms, or the pair above passes
# vacuously the day someone drops a case from N_CASES.
assert "N5c: the fixture covers both the live and idle UNKNOWN cells" \
    bash -c '[ "$1" -gt 0 ] && [ "$2" -gt 0 ]' _ \
    "${#n_exp_live_unknown_lanes[@]}" "${#n_exp_idle_unknown_lanes[@]}"

N_PY_HEADROOM="$(mktemp /tmp/test-warm-lane-audit-n-headroom-XXXXXX.py)"
_TMPDIRS+=("$N_PY_HEADROOM")
cat > "$N_PY_HEADROOM" << 'PYEOF'
import json, sys
data = json.load(sys.stdin)
h = data["headroom"]
want = dict(kv.split("=", 1) for kv in sys.argv[1:])
for k, v in want.items():
    assert k in h, f"headroom has no {k!r} key: {sorted(h)}"
    assert h[k] == int(v), f"headroom[{k!r}] == {h[k]!r}, want {v}"
assert h["resident"] == h["live"] + h["pinned"] + h["quarantined"] + h["free"], h
PYEOF

ORACLE_MAP="$N_MAP" run_helper --mount "$N_MOUNT" \
    --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh" --format json
assert "N6a: exit 0 (json)" test "$RC" -eq 0
assert "N6b: json headroom carries the same partition + cross-cut fields" \
    bash -c 'printf "%s" "$1" | python3 "$2" "${@:3}"' _ "$OUT" "$N_PY_HEADROOM" \
    "resident=$n_exp_resident" "live=$n_exp_live" "pinned=$n_exp_pinned" \
    "quarantined=$n_exp_quarantined" "free=$n_exp_free" \
    "assigned=$n_exp_assigned" "state_unknown=$n_exp_state_unknown"

for n_pid in "${N_LIVE_PIDS[@]+${N_LIVE_PIDS[@]}}"; do
    kill "$n_pid" 2>/dev/null || true
done
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Block O — the PINNED breakdown line (why the pins are held)
# ──────────────────────────────────────────────────────────────────────────────
# `pinned=N` says how much capacity is standing idle under a reservation; it
# does not say what to DO about it. The breakdown does: a `terminal` pin is
# reclaimable right now, a `pending`/`blocked`/`infra-hold` pin is a reservation
# held by work that never started, and an `other` pin covering `in-progress`
# is a likely-crashed consumer. On 2026-07-26 (esc-5556-1) that split was
# 27 pending + 1 infra-hold, which is a scheduling problem, not a leak -- and
# the audit had no column able to say so.
#
# The buckets are a FIXED, closed vocabulary emitted in a FIXED order, zeros
# included, so an operator's grep has a stable shape and a zero is assertable
# rather than absent.
echo ""
echo "--- Block O: the PINNED breakdown line ---"

O_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-o-XXXXXX)"
_TMPDIRS+=("$O_MOUNT")
O_MAP="$(mktemp /tmp/test-warm-lane-audit-o-map-XXXXXX)"
_TMPDIRS+=("$O_MAP")
printf '9201 pending\n9202 infra-hold\n9203 blocked\n9204 done\n9205 cancelled\n9206 in-progress\n9207 deferred\n' > "$O_MAP"

# lane : task id : raw status ('' = id absent from ORACLE_MAP) : expected bucket
#
# Every lane is ASSIGNED and idle, so every lane is PINNED and every lane lands
# in exactly one bucket. Two lanes per non-singleton bucket (done+cancelled ->
# terminal, in-progress+deferred -> other) so a bucket count of 2 distinguishes
# a real accumulation from an off-by-one that happens to match 1.
O_CASES=(
    "_lane-o-pending:9201:pending:pending"
    "_lane-o-infrahold:9202:infra-hold:infra-hold"
    "_lane-o-blocked:9203:blocked:blocked"
    "_lane-o-done:9204:done:terminal"
    "_lane-o-cancelled:9205:cancelled:terminal"
    "_lane-o-inprogress:9206:in-progress:other"
    "_lane-o-deferred:9207:deferred:other"
    "_lane-o-unresolvable:9299::unknown"
)

for o_case in "${O_CASES[@]}"; do
    IFS=':' read -r o_lane o_task o_raw o_bucket <<< "$o_case"
    # Stay on main: the record's task_id is then the ONLY possible source of
    # the pin holder, so a bucket can never be sourced from a branch name.
    make_lane "$O_MOUNT/$o_lane"
    make_lane_state "$O_MOUNT" "$o_lane" "assigned" "$o_task"
    touch "$O_MOUNT/$o_lane.lock"
done

# Bucket expectations DERIVED from O_CASES, never typed.
declare -A O_EXPECTED=(
    [pending]=0 [infra-hold]=0 [blocked]=0 [terminal]=0 [other]=0 [unknown]=0
)
o_exp_total=0
for o_case in "${O_CASES[@]}"; do
    IFS=':' read -r o_lane o_task o_raw o_bucket <<< "$o_case"
    O_EXPECTED["$o_bucket"]=$(( O_EXPECTED["$o_bucket"] + 1 ))
    o_exp_total=$(( o_exp_total + 1 ))
done

ORACLE_MAP="$O_MAP" run_helper --mount "$O_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "O1: exit 0" test "$RC" -eq 0

# The FIXED key set, in the FIXED order, with every value an integer -- the
# stable grep shape an operator (and any future consumer) can rely on.
assert "O2: PINNED line carries total/pending/infra-hold/blocked/terminal/other/unknown in that order" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^PINNED total=[0-9]+ pending=[0-9]+ infra-hold=[0-9]+ blocked=[0-9]+ terminal=[0-9]+ other=[0-9]+ unknown=[0-9]+$"' \
    _ "$OUT"

for o_bucket in pending infra-hold blocked terminal other unknown; do
    o_want="${O_EXPECTED[$o_bucket]}"
    o_got="$(_pinned_field "$OUT" "$o_bucket")"
    assert "O3: PINNED $o_bucket=$o_want (derived from O_CASES)" \
        bash -c '[ "$1" = "$2" ]' _ "$o_got" "$o_want"
done

assert "O4: PINNED total=$o_exp_total equals the sum of the six buckets" \
    _sum_holds \
    "$(_pinned_field "$OUT" total)" \
    "$(_pinned_field "$OUT" pending)" "$(_pinned_field "$OUT" infra-hold)" \
    "$(_pinned_field "$OUT" blocked)" "$(_pinned_field "$OUT" terminal)" \
    "$(_pinned_field "$OUT" other)" "$(_pinned_field "$OUT" unknown)"

# A done-backed pin is `terminal`, never `other`: that distinction is the
# difference between "reclaim this lane now" and "a consumer probably crashed".
# Asserted on a single-lane pool so the verdict cannot be masked by any other
# lane's contribution to either bucket.
O_DONE_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-o-done-XXXXXX)"
_TMPDIRS+=("$O_DONE_MOUNT")
make_lane "$O_DONE_MOUNT/_lane-o-solo"
make_lane_state "$O_DONE_MOUNT" "_lane-o-solo" "assigned" "9204"
ORACLE_MAP="$O_MAP" run_helper --mount "$O_DONE_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "O5: a lone done-backed pin buckets terminal=1 other=0 (never 'other')" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^PINNED total=1 pending=0 infra-hold=0 blocked=0 terminal=1 other=0 unknown=0$"' \
    _ "$OUT"

ORACLE_MAP="$O_MAP" run_helper --mount "$O_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "O6: PINNED total equals HEADROOM pinned (the breakdown covers every pin)" \
    bash -c '[ -n "$1" ] && [ "$1" = "$2" ]' _ \
    "$(_pinned_field "$OUT" total)" "$(_headroom_field "$OUT" pinned)"

# ── O7: the line is emitted with all-zeros on a pool with NO pinned lanes ────
# A zero must be assertable, not absent: an operator grepping `^PINNED ` on a
# healthy pool has to see zeros, otherwise "no pins" is indistinguishable from
# "the audit did not run".
O2_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-o2-XXXXXX)"
_TMPDIRS+=("$O2_MOUNT")
make_lane "$O2_MOUNT/_lane-o2"

run_helper --mount "$O2_MOUNT"
assert "O7a: exit 0 on a pool with no pinned lanes" test "$RC" -eq 0
assert "O7b: PINNED line still emitted, all buckets zero" \
    bash -c 'printf "%s\n" "$1" | grep -qE "^PINNED total=0 pending=0 infra-hold=0 blocked=0 terminal=0 other=0 unknown=0$"' \
    _ "$OUT"

# ── O8: the JSON counterpart ─────────────────────────────────────────────────
O_PY_PINNED="$(mktemp /tmp/test-warm-lane-audit-o-pinned-XXXXXX.py)"
_TMPDIRS+=("$O_PY_PINNED")
cat > "$O_PY_PINNED" << 'PYEOF'
import json, sys
data = json.load(sys.stdin)
assert "pinned_by_status" in data, f"no pinned_by_status key: {sorted(data)}"
p = data["pinned_by_status"]
want = dict(kv.split("=", 1) for kv in sys.argv[1:])
assert set(p) == set(want), f"pinned_by_status keys {sorted(p)}, want {sorted(want)}"
for k, v in want.items():
    assert p[k] == int(v), f"pinned_by_status[{k!r}] == {p[k]!r}, want {v}"
# The breakdown must account for every pin the headroom partition reports.
assert sum(p.values()) == data["headroom"]["pinned"], (p, data["headroom"])
PYEOF

ORACLE_MAP="$O_MAP" run_helper --mount "$O_MOUNT" \
    --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh" --format json
assert "O8a: exit 0 (json)" test "$RC" -eq 0
assert "O8b: json pinned_by_status carries exactly the six buckets, summing to headroom.pinned" \
    bash -c 'printf "%s" "$1" | python3 "$2" "${@:3}"' _ "$OUT" "$O_PY_PINNED" \
    "pending=${O_EXPECTED[pending]}" "infra-hold=${O_EXPECTED[infra-hold]}" \
    "blocked=${O_EXPECTED[blocked]}" "terminal=${O_EXPECTED[terminal]}" \
    "other=${O_EXPECTED[other]}" "unknown=${O_EXPECTED[unknown]}"

# ──────────────────────────────────────────────────────────────────────────────
# Block P — the two incident regressions + the state-dir read-only proof
# ──────────────────────────────────────────────────────────────────────────────
# Both pool misreads this task exists to fix are reproduced here as fixtures, so
# the accounting that mis-answered them can never silently return. Neither
# incident's headline numbers (53 / 33 / 30) are frozen as literals: every
# expectation is computed from the fixture's own arrays, so these stay
# regression tests of the RELATION, not of one day's pool.
echo ""
echo "--- Block P: incident regressions + state-dir read-only ---"

# ── P-a — 2026-07-22: a fully reserved pool must not read as fully free ──────
# Every lane carries a `state: assigned` record and NO lane holds a flock: this
# is exactly the pool that reported 53 lanes FREE while the orchestrator had
# every one of them reserved. Under the retired accounting (free = resident -
# live) this pool reports free == resident; the assertion that would have caught
# the incident on the day is free == 0.
PA_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-pa-XXXXXX)"
_TMPDIRS+=("$PA_MOUNT")
PA_MAP="$(mktemp /tmp/test-warm-lane-audit-pa-map-XXXXXX)"
_TMPDIRS+=("$PA_MAP")

PA_LANES=(_lane-pa-1 _lane-pa-2 _lane-pa-3 _lane-pa-4 _lane-pa-5 _lane-pa-6)
: > "$PA_MAP"
pa_id=9400
for pa_lane in "${PA_LANES[@]}"; do
    pa_id=$(( pa_id + 1 ))
    make_lane "$PA_MOUNT/$pa_lane"
    make_lane_state "$PA_MOUNT" "$pa_lane" "assigned" "$pa_id"
    # An unheld lock file: the incident's exact shape -- the sidecar exists,
    # nothing holds it.
    touch "$PA_MOUNT/$pa_lane.lock"
    printf '%s pending\n' "$pa_id" >> "$PA_MAP"
done
pa_exp_resident="${#PA_LANES[@]}"

ORACLE_MAP="$PA_MAP" run_helper --mount "$PA_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "P-a1: exit 0" test "$RC" -eq 0
assert "P-a2: free=0 -- a fully reserved pool reports NO free capacity (the 2026-07-22 regression)" \
    bash -c '[ "$1" = "0" ]' _ "$(_headroom_field "$OUT" free)"
assert "P-a3: live=0 (no consumer holds any lock)" \
    bash -c '[ "$1" = "0" ]' _ "$(_headroom_field "$OUT" live)"
assert "P-a4: pinned=$pa_exp_resident equals the lane count" \
    bash -c '[ -n "$1" ] && [ "$1" = "$2" ]' _ \
    "$(_headroom_field "$OUT" pinned)" "$pa_exp_resident"
assert "P-a5: free != resident (the retired free = resident - live rule is gone)" \
    bash -c '[ -n "$1" ] && [ -n "$2" ] && [ "$1" != "$2" ]' _ \
    "$(_headroom_field "$OUT" free)" "$(_headroom_field "$OUT" resident)"
for pa_lane in "${PA_LANES[@]}"; do
    assert "P-a6: $pa_lane reports classification=PINNED" \
        bash -c 'printf "%s\n" "$1" | grep -qE "lane=$2 .*classification=PINNED( |\$)"' \
        _ "$OUT" "$pa_lane"
done

# ── P-b — 2026-07-26 (esc-5556-1): pins dominated by not-running work ────────
# A assigned lanes of which L hold a live flock, plus R released lanes. The
# incident's own shape: a handful of genuinely live consumers, and the rest of
# the reservations held by tasks that are not running -- overwhelmingly
# `pending`, with a single `infra-hold`. `pinned` names that standing capacity
# loss; the breakdown names its cause.
PB_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-pb-XXXXXX)"
_TMPDIRS+=("$PB_MOUNT")
PB_MAP="$(mktemp /tmp/test-warm-lane-audit-pb-map-XXXXXX)"
_TMPDIRS+=("$PB_MAP")

# lane : liveness : record state : task id : backing status
PB_CASES=(
    "_lane-pb-live1:LIVE:assigned:9501:in-progress"
    "_lane-pb-live2:LIVE:assigned:9502:in-progress"
    "_lane-pb-live3:LIVE:assigned:9503:in-progress"
    "_lane-pb-pin1:IDLE:assigned:9504:pending"
    "_lane-pb-pin2:IDLE:assigned:9505:pending"
    "_lane-pb-pin3:IDLE:assigned:9506:pending"
    "_lane-pb-pin4:IDLE:assigned:9507:infra-hold"
    "_lane-pb-rel1:IDLE:released:9508:done"
    "_lane-pb-rel2:IDLE:released:9509:done"
)

: > "$PB_MAP"
PB_LIVE_PIDS=()
for pb_case in "${PB_CASES[@]}"; do
    IFS=':' read -r pb_lane pb_liveness pb_state pb_task pb_status <<< "$pb_case"
    printf '%s %s\n' "$pb_task" "$pb_status" >> "$PB_MAP"
    make_lane "$PB_MOUNT/$pb_lane"
    make_lane_state "$PB_MOUNT" "$pb_lane" "$pb_state" "$pb_task"
    if [ "$pb_liveness" = "LIVE" ]; then
        _hold_lane_lock "$PB_MOUNT" "$pb_lane"
        PB_LIVE_PIDS+=("$LANE_LOCK_PID")
    else
        touch "$PB_MOUNT/$pb_lane.lock"
    fi
done

pb_exp_assigned=0
pb_exp_live=0
pb_exp_pinned=0
pb_exp_free=0
pb_exp_pin_pending=0
pb_exp_pin_infrahold=0
for pb_case in "${PB_CASES[@]}"; do
    IFS=':' read -r pb_lane pb_liveness pb_state pb_task pb_status <<< "$pb_case"
    if [ "$pb_state" = "assigned" ]; then
        pb_exp_assigned=$(( pb_exp_assigned + 1 ))
    fi
    if [ "$pb_liveness" = "LIVE" ]; then
        pb_exp_live=$(( pb_exp_live + 1 ))
        continue
    fi
    if [ "$pb_state" = "assigned" ]; then
        pb_exp_pinned=$(( pb_exp_pinned + 1 ))
        case "$pb_status" in
            pending)    pb_exp_pin_pending=$(( pb_exp_pin_pending + 1 )) ;;
            infra-hold) pb_exp_pin_infrahold=$(( pb_exp_pin_infrahold + 1 )) ;;
        esac
    else
        pb_exp_free=$(( pb_exp_free + 1 ))
    fi
done
pb_exp_resident="${#PB_CASES[@]}"

# ── P-c baselines, captured HERE — before the FIRST audit run over this mount.
# _snapshot_state_dir <mount> — a sorted path/size/mtime/content manifest of
# every record under <mount>/.lane-state, or a sentinel when the dir is absent.
# Two snapshots compare byte-identical iff no record was created, removed,
# resized, re-stamped, or rewritten. Complements Block J's per-lane
# _snapshot_lane proof at the SAME fidelity (J's manifest carries %T@ too, so an
# mtime-touching read cannot hide in either): the audit now reads a second
# on-disk surface, and that surface gets the same guarantee.
#
# The ordering is load-bearing, and is why these live above the P-b run rather
# than beside the P-c assertions that consume them. A baseline taken AFTER an
# earlier run over the same mount cannot see a FIRST-RUN-ONLY mutation of an
# existing record — a normalizing rewrite, a trailing-newline append, a backfill
# of a missing field: run #1 performs it, the baseline bakes it in, run #2 is
# idempotent, and the comparison goes green while A1 is violated. Block J takes
# its baselines before its first run for exactly this reason. (P-c4-P-c8 cover
# the other half of A1 — never CREATING a record or dir — on a fresh mount.)
_snapshot_state_dir() {
    local mount="$1"
    local dir="$mount/.lane-state"
    if [ ! -d "$dir" ]; then
        printf '(no .lane-state)\n'
        return 0
    fi
    local line f
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        f="${line%% *}"
        printf '%s %s\n' "$line" "$(cat "$f")"
    done < <(find "$dir" -type f -printf '%p %s %T@\n' 2>/dev/null | sort)
    return 0
}

pc_state_before="$(_snapshot_state_dir "$PB_MOUNT")"
declare -A PC_LANE_BEFORE
for pb_case in "${PB_CASES[@]}"; do
    IFS=':' read -r pb_lane _ _ _ _ <<< "$pb_case"
    PC_LANE_BEFORE["$pb_lane"]="$(_snapshot_lane "$PB_MOUNT/$pb_lane")"
done

ORACLE_MAP="$PB_MAP" run_helper --mount "$PB_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"

assert "P-b1: exit 0" test "$RC" -eq 0
PB_FIELDS=(
    "resident:$pb_exp_resident"
    "assigned:$pb_exp_assigned"
    "live:$pb_exp_live"
    "pinned:$pb_exp_pinned"
    "free:$pb_exp_free"
)
for pb_f in "${PB_FIELDS[@]}"; do
    pb_key="${pb_f%%:*}"
    pb_want="${pb_f##*:}"
    assert "P-b2: HEADROOM $pb_key=$pb_want (derived from PB_CASES)" \
        bash -c '[ "$1" = "$2" ]' _ "$(_headroom_field "$OUT" "$pb_key")" "$pb_want"
done
assert "P-b3: partition identity holds on the incident-shaped pool" \
    _partition_holds \
    "$(_headroom_field "$OUT" resident)" \
    "$(_headroom_field "$OUT" live)" \
    "$(_headroom_field "$OUT" pinned)" \
    "$(_headroom_field "$OUT" quarantined)" \
    "$(_headroom_field "$OUT" free)"
# `assigned` is a CROSS-CUT, so it must exceed `live` here without breaking the
# partition -- reservations outnumber running consumers, which IS the incident.
assert "P-b4: assigned > live (reservations outnumber running consumers)" \
    bash -c '[ -n "$1" ] && [ -n "$2" ] && [ "$1" -gt "$2" ]' _ \
    "$(_headroom_field "$OUT" assigned)" "$(_headroom_field "$OUT" live)"
assert "P-b5: PINNED pending=$pb_exp_pin_pending (derived from PB_CASES)" \
    bash -c '[ "$1" = "$2" ]' _ "$(_pinned_field "$OUT" pending)" "$pb_exp_pin_pending"
assert "P-b6: PINNED infra-hold=$pb_exp_pin_infrahold (derived from PB_CASES)" \
    bash -c '[ "$1" = "$2" ]' _ "$(_pinned_field "$OUT" infra-hold)" "$pb_exp_pin_infrahold"
assert "P-b7: PINNED total equals HEADROOM pinned" \
    bash -c '[ -n "$1" ] && [ "$1" = "$2" ]' _ \
    "$(_pinned_field "$OUT" total)" "$(_headroom_field "$OUT" pinned)"
# "Dominated by pending" is the incident's signature: pending is the strict
# maximum bucket, so the pool is losing capacity to work that never started
# rather than to leaks or crashes.
assert "P-b8: pending is the strictly dominant pin bucket" \
    _strict_max \
    "$(_pinned_field "$OUT" pending)" \
    "$(_pinned_field "$OUT" infra-hold)" "$(_pinned_field "$OUT" blocked)" \
    "$(_pinned_field "$OUT" terminal)" "$(_pinned_field "$OUT" other)" \
    "$(_pinned_field "$OUT" unknown)"

# ── P-c — the state dir is read strictly read-only, and never created ────────
# The baselines were captured BEFORE P-b's run above, not here: see the comment
# at their capture site for why that ordering is the whole proof.
ORACLE_MAP="$PB_MAP" run_helper --mount "$PB_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle.sh"
assert "P-c1: exit 0" test "$RC" -eq 0
assert "P-c2: the .lane-state manifest is byte-identical across BOTH runs (A1/A5)" \
    bash -c '[ "$1" = "$2" ]' _ "$pc_state_before" "$(_snapshot_state_dir "$PB_MOUNT")"
for pb_case in "${PB_CASES[@]}"; do
    IFS=':' read -r pb_lane _ _ _ _ <<< "$pb_case"
    assert "P-c3: $pb_lane is byte-identical across BOTH runs over a state-bearing pool" \
        bash -c '[ "$1" = "$2" ]' _ \
        "${PC_LANE_BEFORE[$pb_lane]}" "$(_snapshot_lane "$PB_MOUNT/$pb_lane")"
done

# A mount with NO state dir: the audit must not conjure one. The state read is
# non-creating for exactly the reason the <dir>.lock probe is (A1) -- an
# advisory reader that materializes pool state is no longer a reader.
PC_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-pc-XXXXXX)"
_TMPDIRS+=("$PC_MOUNT")
make_lane "$PC_MOUNT/_lane-pc"
assert "P-c4: the fixture mount has no .lane-state dir before the run" \
    bash -c '[ ! -e "$1" ]' _ "$PC_MOUNT/.lane-state"
run_helper --mount "$PC_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle-fail.sh"
assert "P-c5: exit 0" test "$RC" -eq 0
assert "P-c6: the run did NOT create a .lane-state dir" \
    bash -c '[ ! -e "$1" ]' _ "$PC_MOUNT/.lane-state"
# ...nor a record under an explicitly-pointed-at, nonexistent state dir.
PC_MISSING_STATE="$PC_MOUNT/nowhere/.lane-state"
REIFY_WARM_LANE_AUDIT_STATE_DIR="$PC_MISSING_STATE" \
    run_helper --mount "$PC_MOUNT" --status-cmd "$ORACLE_STUB_DIR/leak-oracle-fail.sh"
assert "P-c7: exit 0 under a nonexistent explicit state dir" test "$RC" -eq 0
assert "P-c8: a nonexistent explicit state dir is not created either" \
    bash -c '[ ! -e "$1" ]' _ "$PC_MISSING_STATE"

for pb_pid in "${PB_LIVE_PIDS[@]+${PB_LIVE_PIDS[@]}}"; do
    kill "$pb_pid" 2>/dev/null || true
done
_BGPIDS=()  # clear so cleanup doesn't double-kill

# ──────────────────────────────────────────────────────────────────────────────
# Block Q — the liveness probe is SHARED (flock -s), so a shared-lock holder
#           reads IDLE (§9.1 Invariant A2)
# ──────────────────────────────────────────────────────────────────────────────
# Block K (above) pins live=LIVE|IDLE, but its only lock-holding fixture is
# _hold_lane_lock, which takes an EXCLUSIVE flock -- and an EXCLUSIVE holder
# blocks both a `-x` and a `-s` probe identically. So Block K cannot tell
# _probe_live's real `flock -n -s 7` (scripts/warm-lane-audit.sh:331) apart
# from a regressed `flock -n -x 7`: the exact regression this block exists to
# catch. Block Q closes that gap with a SHARED-lock lane (must read IDLE --
# the A2 pin) alongside an EXCLUSIVE-lock lane in the SAME run (must stay
# LIVE -- the negative control that keeps the pin from passing vacuously).
echo ""
echo "--- Block Q: liveness probe is SHARED (flock -s) -- a shared lock reads IDLE (A2) ---"

Q_MOUNT="$(mktemp -d /tmp/test-warm-lane-audit-q-XXXXXX)"
_TMPDIRS+=("$Q_MOUNT")

# _lane-shared: a shared-lock holder -- must probe IDLE (the A2 pin).
make_lane "$Q_MOUNT/_lane-shared"
touch "$Q_MOUNT/_lane-shared.lock"
_hold_lane_lock_shared "$Q_MOUNT" "_lane-shared"
Q_SHARED_PID="$LANE_LOCK_PID"

# _lane-excl: an exclusive-lock holder -- must stay LIVE (the negative
# control; see the Q6-Q8 comment below for why it's needed).
make_lane "$Q_MOUNT/_lane-excl"
_hold_lane_lock "$Q_MOUNT" "_lane-excl"
Q_EXCL_PID="$LANE_LOCK_PID"

# Both holders are established BEFORE this single run_helper call so one
# audit run observes both lanes -- the side-by-side comparison in one report
# IS the pin (same probe, same run, two lock modes, two answers).
run_helper --mount "$Q_MOUNT"

assert "Q1: exit 0" test "$RC" -eq 0

# Q2/Q3 are fixture-integrity controls, not a restatement of the READY-marker
# handshake: an assertion that something is ABSENT (no LIVE) passes vacuously
# if the background holder silently died or opened the wrong path -- the lane
# would then read IDLE for the wrong reason and the pin would evaporate
# without any test going red. Q2 (blocked exclusive request) proves the lock
# is genuinely held; Q3 (successful shared request) proves it is held in
# SHARED mode, not exclusive -- together they uniquely characterize "a shared
# lock is held right now", and Q3 is what fails if a future edit flips this
# fixture's `flock -s` back to `-x`.
assert "Q2: an independent exclusive flock request on the lock is BLOCKED (the shared lock is genuinely held)" \
    bash -c 'exec 8<"$1"; ! flock -n -x 8' _ "$Q_MOUNT/_lane-shared.lock"
assert "Q3: an independent shared flock request on the same lock SUCCEEDS (the fixture's lock is SHARED, not exclusive)" \
    bash -c 'exec 8<"$1"; flock -n -s 8' _ "$Q_MOUNT/_lane-shared.lock"

assert "Q4: _lane-shared row reports live=IDLE (A2: the probe is -s, so a shared holder is IDLE)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-shared .*live=IDLE"' _ "$OUT"
assert "Q5: _lane-shared classifies RECLAIMABLE, not LIVE (the A2 consequence downstream)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-shared .*classification=RECLAIMABLE"' _ "$OUT"

# Q6-Q8 pin the other half of the same run: _lane-excl (an EXCLUSIVE holder)
# is the negative control that keeps Q4/Q5 from passing vacuously -- a probe
# degenerated to "always IDLE" would satisfy Q4/Q5 without ever exercising
# the -s/-x distinction, but it would also wrongly read _lane-excl as IDLE.
# Measured mutation table (shipped `flock -n -s 7` vs. regressed
# `flock -n -x 7`, same two-lane mount): shipped => live=1 free=1
# reclaimable=1; regressed => live=2 free=0 reclaimable=0.
assert "Q6: _lane-excl row still reports live=LIVE (negative control, invariant under the -s/-x mutation)" \
    bash -c 'printf "%s\n" "$1" | grep -q "lane=_lane-excl .*live=LIVE"' _ "$OUT"
assert "Q7: HEADROOM reports live=1 (only the exclusive lane counts live)" \
    bash -c '[ "$1" = "1" ]' _ "$(_headroom_field "$OUT" live)"
assert "Q8: HEADROOM reports free=1 and reclaimable=1 (the shared lane's freed capacity is visible in the aggregate)" \
    bash -c '[ "$1" = "1" ] && [ "$2" = "1" ]' _ "$(_headroom_field "$OUT" free)" "$(_headroom_field "$OUT" reclaimable)"

kill "$Q_SHARED_PID" "$Q_EXCL_PID" 2>/dev/null || true
_BGPIDS=()  # clear so cleanup doesn't double-kill

test_summary
