#!/usr/bin/env bash
# scripts/warm-lane-audit.sh — Standalone, timer-friendly audit/telemetry
# report for the warm-lane CoW pool. Read-only observability: never mutates a
# lane and never gates dispatch/reclaim/merge (PRD §9.5 inv.12).
#
# Part of PRD docs/prds/warm-lane-pool-sizing-lifecycle.md §9.1, §10 B1/B2.
# Consumer: reify ζ (integration gate), dark-factory θ (soft-floor dispatch
# throttle), the operator, and a health timer (§13 open-Q 3).
#
# Usage:
#   scripts/warm-lane-audit.sh [--mount WORKTREES_DIR] [--format table|json] \
#       [--status-cmd CMD] [--stale-age-min N] [--main-ref REF] [--safety N]
#
# For each resident worktree under --mount (lanes _lane-*/_spec-* and orphan
# git-worktree dirs), emits: lane · role · live (LIVE|IDLE) · assigned
# (ASSIGNED|RELEASED|QUARANTINED|UNKNOWN) · pin (the RAW backing status of a
# reserved-but-idle lane's holder, `unknown` if unresolvable, `-` when the
# lane is not pinned) · branch ·
# backing-task-status (terminal|non-terminal|unknown) · recoverable
# (LANDED|PUSHED|ORPHAN) · dirty (clean|residue-only|wip) · divergent_gib ·
# age_min · classification (LIVE|PINNED|QUARANTINED|RECLAIMABLE|LEAKED|
# PRESERVED-OK, ranked in that order).
#
# `live`, `assigned` and `pin` answer THREE independent questions, and are
# three columns for that reason: is a consumer PROCESS holding this lane's
# flock; has the pool RESERVED it (read from the orchestrator's own record at
# <state-dir>/<lane>.json, never inferred from the lock); and, when it is
# reserved with nothing running, WHO holds it.
#
# Two summary lines follow the per-lane rows:
#
#     HEADROOM resident=… live=… pinned=… quarantined=… free=… assigned=…
#              state_unknown=… reclaimable=… leaked=… leak_unknown=…
#              divergent_gib=… free_gib=… budget_gib=…
#     PINNED   total=… pending=… infra-hold=… blocked=… terminal=… other=…
#              unknown=…
#
# HEADROOM's occupancy figures are an ORDERED, mutually exclusive PARTITION --
# `resident = live + pinned + quarantined + free`, with `free` the residue --
# while `assigned` and `state_unknown` are CROSS-CUTS never added into it (see
# the counter block at the resident walk, which is where that identity is
# enforced). PINNED breaks `pinned` down by WHY each pin is held: a fixed,
# closed bucket vocabulary in a fixed order, always emitted, zeros included
# (JSON: a sibling "pinned_by_status" object whose total is headroom.pinned).
#
# Why the columns are split this way, the two pool misreads that motivated it,
# and how to read a PINNED-heavy pool: docs/notes/warm-lane-audit-runbook.md.
#
# Options (env defaults shown):
#   --mount DIR           Warm-lane worktrees dir (env: REIFY_WARM_LANE_MOUNT;
#                         shared with warm-lane-preflight.sh / warm-lane-gc.sh).
#                         A nonexistent/empty mount reports resident=0 (not
#                         an error — this script is advisory-only).
#   --format table|json  Output format (default: table).
#   --status-cmd CMD     Backing-task status oracle: invoked as `<cmd> <id>`,
#                         expected to print a status (e.g. done/cancelled/
#                         pending) to stdout. Empty output or a non-zero exit
#                         is treated as unknown. Default: REIFY_LANE_LEAK_STATUS_CMD
#                         (the same oracle warm-lane-preflight.sh Check 6 /
#                         warm-lane-gc.sh consume — D6, no new status plumbing).
#   --stale-age-min N     Minutes; a LEAKED candidate is stale when
#                         age_min >= N (env: REIFY_WARM_LANE_AUDIT_STALE_AGE_MIN;
#                         default: 60).
#   --main-ref REF        Git ref for "main" (env: REIFY_WARM_LANE_AUDIT_MAIN_REF;
#                         default: main).
#   --safety N            Dimensionless divisor for budget_gib = floor(free_gib
#                         / safety) (env: REIFY_WARM_LANE_AUDIT_SAFETY; default:
#                         1.5, mirroring the illustrative safety factor in PRD
#                         §9.2's worked example). Must be > 0.
#   -h, --help            Print this message and exit.
#
# Additional env knobs (no dedicated CLI flag):
#   REIFY_WARM_LANE_AUDIT_DF             df command override (default: df),
#                                         mirrors REIFY_WARM_LANE_DISK_GUARD_DF.
#   REIFY_WARM_LANE_AUDIT_RESIDUE_GLOB   Glob (or comma-separated globs)
#                                         matching "residue" dirty paths that
#                                         don't count as unrecoverable WIP
#                                         (default: data/queue/*.db*, the
#                                         §2/D1 write_queue.db residue).
#   REIFY_WARM_LANE_AUDIT_STATE_DIR      Directory holding the orchestrator's
#                                         durable per-lane assignment records
#                                         <lane>.json (default:
#                                         <mount>/.lane-state, dark-factory's
#                                         LANE_STATE_DIRNAME). Read-only; a
#                                         missing dir is not an error (every
#                                         lane simply reports
#                                         assigned=UNKNOWN -- see A5).
#
# Exit codes:
#   0  — Always, on every valid invocation (advisory/observability — this
#        script must never gate anything; PRD §9.5 inv.12). A status-lookup
#        failure, a df/du measurement failure, or a nonexistent/empty mount
#        all degrade gracefully rather than aborting.
#   2  — Usage/WIRING error: unknown flag, missing flag value, invalid
#        --format/--stale-age-min/--safety, or a missing sibling library
#        (scripts/lib_portable.sh, scripts/lib_lane_state.sh). The library
#        case is a wiring error, not a runtime one: nothing about the
#        invocation could have avoided it and no retry will fix it — the
#        deployment is incomplete. Same class as an unknown flag, hence 2.
#        It does not weaken the exit-0 rule above, which binds every VALID
#        invocation; a wiring abort means there was no valid invocation to
#        report on.
#
# Invariants:
#   A1 — read-only: never mutates a lane (no reset/rm/reclaim). This binds
#        BOTH on-disk surfaces the audit touches, and identically:
#          · the LIVE/IDLE probe opens an EXISTING <dir>.lock read-only and
#            never creates a missing one;
#          · the assignment-state read opens an EXISTING
#            <state-dir>/<lane>.json read-only and never creates the record
#            OR the directory -- neither a default <mount>/.lane-state nor an
#            explicitly-pointed-at REIFY_WARM_LANE_AUDIT_STATE_DIR.
#        No `>`-open, touch, or mkdir occurs anywhere on either path; an
#        advisory reader that materializes pool state is not a reader.
#   A2 — the flock -n -s (shared) probe is non-blocking and released
#        immediately. A shared request still correctly detects LIVE (a
#        live consumer's exclusive flock blocks a new shared request too),
#        but never contends with another concurrent reader (e.g. a second
#        audit run) — only a genuine writer's non-blocking attempt could
#        ever be perturbed, and only for the instant this probe's fd is
#        open.
#   A3 — a status-lookup failure degrades that lane to `unknown` (never
#        aborts, never reclassifies as reclaimable/leaked). When this
#        suppresses what would otherwise be a LEAKED verdict, the lane is
#        still reported PRESERVED-OK (conservative default), but a stderr
#        warning fires and the lane is counted in the HEADROOM
#        `leak_unknown` field, so "no leaks" stays distinguishable from
#        "leaks could not be evaluated".
#   A4 — `stale` is always the relation age_min >= stale_age_min against the
#        declared knob — never an inline/undeclared literal.
#   A5 — the assignment-state read is fail-safe: a missing state dir, a
#        missing/unreadable record, corrupt JSON, or an unrecognized `state`
#        value all degrade that lane to `assigned=UNKNOWN`. It never aborts
#        and never invents an assignment. An UNKNOWN lane is never counted
#        pinned, quarantined or assigned; an idle one falls to the partition
#        residue `free` (the conservative reading), a live one is counted
#        `live` like any other. Such lanes are surfaced separately -- a stderr
#        warning naming the lane, which of those causes fired
#        (no-readable-record | unparseable-record | unrecognized-state:<raw>;
#        see _read_lane_assignment), and which bucket it was counted in, plus
#        the HEADROOM state_unknown field -- so "no pins" stays distinguishable
#        from "pins could not be evaluated", the same treatment A3 gives an
#        unresolvable backing-task status. A state dir that is absent ENTIRELY
#        warns ONCE for the directory instead of once per lane, so the per-lane
#        naming keeps meaning "this lane, unlike its neighbours".

set -euo pipefail

# dirname by PARAMETER EXPANSION, not a fork: a PATH without dirname yields an
# EMPTY substitution, `cd ""` succeeds as a no-op, and SCRIPT_DIR silently becomes
# the CALLER'S CWD. See README.md "Delta 7".
_src="${BASH_SOURCE[0]}"
_dir='.'
case "$_src" in
    */*) _dir="${_src%/*}"
         [ -n "$_dir" ] || _dir='/' ;;
esac
SCRIPT_DIR="$(cd "$_dir" && pwd)"
unset _src _dir

# reify's portable shims (portable_mtime et al.). Fail LOUDLY if the lib is
# missing: portable_mtime is called unconditionally on every lane (see
# _age_min below, line ~651), so an absent lib_portable.sh is an incomplete
# DEPLOYMENT, never a data problem -- exit 2 (usage/WIRING), not the 1 bash's
# own bare `source` failure produces under `set -e`. Exit 1 sends an operator
# or timer hunting a runtime/data fault that is not there, and attributes the
# failure to bash rather than to this script.
#
# Ordered FIRST, above the lib_lane_state.sh guard below, deliberately: a copy
# carrying NEITHER sibling must report this one, mirroring
# warm-lane-gc.sh:305-307's live-refs-before-lane-state rule. See README.md
# "Delta 8".
if [ ! -f "$SCRIPT_DIR/lib_portable.sh" ]; then
    echo "warm-lane-audit.sh: ERROR — scripts/lib_portable.sh not found next to warm-lane-audit.sh" >&2
    exit 2
fi
# shellcheck source=scripts/lib_portable.sh
source "$SCRIPT_DIR/lib_portable.sh"

# dark-factory's lane-state reader (lane_state_read + lane_state_class), the
# one definition of the record parse and of the raw-state -> column table this
# script's `assigned` column reports. Fail LOUDLY if it is missing.
#
# Ordered SECOND, after the lib_portable.sh guard above, deliberately:
# reordering the pair would flip which message a copy carrying NEITHER sibling
# emits, which test_warm_lane_scripts_shipped.py's
# TestAuditFailsLoudOnAMissingLibPortable pins by asserting THIS message is
# absent from that run.
#
# Exit 2 (usage/WIRING), not a degrade-to-UNKNOWN, and not 1 (runtime): this
# script's "never abort" rule is about lane-level DATA problems -- an unreadable
# or corrupt record must degrade that lane to UNKNOWN and never kill the run --
# not about deployment wiring. Nothing about the invocation could have avoided
# an absent sibling and no retry fixes it, which is exactly the class
# warm-lane-gc.sh already exits 2 for on a missing lib_live_refs.sh. Degrading
# instead would silently report EVERY lane as UNKNOWN, indistinguishable from a
# real pool-wide state-dir outage -- a triage trap.
if [ ! -f "$SCRIPT_DIR/lib_lane_state.sh" ]; then
    echo "warm-lane-audit.sh: ERROR — scripts/lib_lane_state.sh not found next to warm-lane-audit.sh" >&2
    exit 2
fi
# shellcheck source=scripts/lib_lane_state.sh
source "$SCRIPT_DIR/lib_lane_state.sh"

# ── log helpers (all write to stderr) ─────────────────────────────────────────
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*" >&2; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
err()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# ── usage ──────────────────────────────────────────────────────────────────────
_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [--mount DIR] [--format table|json] [--status-cmd CMD]
                       [--stale-age-min N] [--main-ref REF] [--safety N]

  Standalone, timer-friendly audit/telemetry report for the warm-lane CoW
  pool. Read-only: never mutates a lane; never gates dispatch/reclaim/merge.

  Per-lane columns:
    live      LIVE|IDLE — is a consumer PROCESS holding the lane's exclusive
              flock right now? Liveness only; NOT the assignment state.
    assigned  ASSIGNED|RELEASED|QUARANTINED|UNKNOWN — has the pool RESERVED
              this lane? Read from the orchestrator's own record at
              \$REIFY_WARM_LANE_AUDIT_STATE_DIR/<lane>.json (default:
              <mount>/.lane-state). Unresolvable => UNKNOWN, never an error.
    pin       For a lane that is ASSIGNED but not LIVE (reserved, nothing
              running): the RAW backing status of the task holding it, e.g.
              pending / infra-hold / in-progress / done. \`unknown\` when it
              cannot be resolved; \`-\` when the lane is not pinned.

  Classification, ranked: LIVE > PINNED > QUARANTINED >
  RECLAIMABLE|LEAKED|PRESERVED-OK -- the HEADROOM partition's order exactly.
  A PINNED or QUARANTINED lane is never additionally reported LEAKED.

  Summary lines:
    HEADROOM  occupancy is an ordered, exclusive PARTITION --
              resident = live + pinned + quarantined + free -- so \`free\` is
              the residue and never absorbs a reserved-but-idle lane.
              \`assigned\` and \`state_unknown\` are cross-cuts, not partition
              members.
    PINNED    why the pins are held: total + the fixed buckets pending,
              infra-hold, blocked, terminal, other, unknown. Always emitted,
              zeros included.

  Options:
    --mount DIR           Warm-lane worktrees dir (default: \$REIFY_WARM_LANE_MOUNT).
    --format table|json   Output format (default: table).
    --status-cmd CMD      Backing-task status oracle (default:
                          \$REIFY_LANE_LEAK_STATUS_CMD).
    --stale-age-min N     Minutes before a candidate is stale (default:
                          \$REIFY_WARM_LANE_AUDIT_STALE_AGE_MIN or 60).
    --main-ref REF        Git ref for "main" (default: \$REIFY_WARM_LANE_AUDIT_MAIN_REF
                          or main).
    --safety N            Divisor for budget_gib = floor(free_gib / safety)
                          (default: \$REIFY_WARM_LANE_AUDIT_SAFETY or 1.5).
    -h, --help            Print this message and exit.

  Exit codes:
    0  — Always, on every valid invocation (advisory-only; never gates).
    2  — Usage/wiring error (incl. a missing sibling scripts/lib_portable.sh
         or scripts/lib_lane_state.sh).
EOF
}

# ── defaults ───────────────────────────────────────────────────────────────────
MOUNT="${REIFY_WARM_LANE_MOUNT:-}"
FORMAT="table"
STATUS_CMD="${REIFY_LANE_LEAK_STATUS_CMD:-}"
STALE_AGE_MIN="${REIFY_WARM_LANE_AUDIT_STALE_AGE_MIN:-60}"
MAIN_REF="${REIFY_WARM_LANE_AUDIT_MAIN_REF:-main}"
SAFETY="${REIFY_WARM_LANE_AUDIT_SAFETY:-1.5}"
DF="${REIFY_WARM_LANE_AUDIT_DF:-df}"
RESIDUE_GLOB="${REIFY_WARM_LANE_AUDIT_RESIDUE_GLOB:-data/queue/*.db*}"
# Resolved after arg parsing (the default is derived from the final --mount).
STATE_DIR="${REIFY_WARM_LANE_AUDIT_STATE_DIR:-}"

# ── arg parsing ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            _usage; exit 0 ;;
        --mount)
            [ $# -ge 2 ] || { err "--mount requires a value"; exit 2; }
            MOUNT="$2"; shift 2 ;;
        --format)
            [ $# -ge 2 ] || { err "--format requires a value"; exit 2; }
            FORMAT="$2"; shift 2 ;;
        --status-cmd)
            [ $# -ge 2 ] || { err "--status-cmd requires a value"; exit 2; }
            STATUS_CMD="$2"; shift 2 ;;
        --stale-age-min)
            [ $# -ge 2 ] || { err "--stale-age-min requires a value"; exit 2; }
            STALE_AGE_MIN="$2"; shift 2 ;;
        --main-ref)
            [ $# -ge 2 ] || { err "--main-ref requires a value"; exit 2; }
            MAIN_REF="$2"; shift 2 ;;
        --safety)
            [ $# -ge 2 ] || { err "--safety requires a value"; exit 2; }
            SAFETY="$2"; shift 2 ;;
        -*)
            err "Unknown flag: $1"
            err "Run '$(basename "$0") --help' for usage."
            exit 2 ;;
        *)
            err "Unexpected argument: $1"
            err "Run '$(basename "$0") --help' for usage."
            exit 2 ;;
    esac
done

# ── validate ───────────────────────────────────────────────────────────────────
case "$FORMAT" in
    table|json) : ;;
    *)
        err "Unknown --format: '$FORMAT' (expected table or json)."
        err "Run '$(basename "$0") --help' for usage."
        exit 2 ;;
esac

if ! printf '%s\n' "$STALE_AGE_MIN" | grep -qE '^[0-9]+$'; then
    err "--stale-age-min (or REIFY_WARM_LANE_AUDIT_STALE_AGE_MIN) is not a valid non-negative integer: '$STALE_AGE_MIN'."
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

if ! printf '%s\n' "$SAFETY" | grep -qE '^[0-9]+(\.[0-9]+)?$' || ! awk -v s="$SAFETY" 'BEGIN{exit !(s>0)}'; then
    err "--safety (or REIFY_WARM_LANE_AUDIT_SAFETY) is not a valid positive number: '$SAFETY'."
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

# The state dir defaults to <mount>/.lane-state, so it can only be resolved
# once --mount is final. An explicit REIFY_WARM_LANE_AUDIT_STATE_DIR wins and
# may point anywhere, including outside the mount.
if [ -z "$STATE_DIR" ] && [ -n "$MOUNT" ]; then
    STATE_DIR="$MOUNT/.lane-state"
fi

info "warm-lane-audit.sh: mount=$MOUNT format=$FORMAT stale_age_min=$STALE_AGE_MIN main_ref=$MAIN_REF safety=$SAFETY state_dir=$STATE_DIR"

# ── helper: is this dir a git worktree? ───────────────────────────────────────
_is_git_worktree() {
    local dir="$1"
    [ -d "$dir" ] || return 1
    git -C "$dir" rev-parse --git-dir >/dev/null 2>&1
}

# ── helper: role from the resident dir's basename ─────────────────────────────
# _lane-* -> lane; _spec-* -> spec; anything else (that is still a git
# worktree) -> orphan. Mirrors warm-lane-gc.sh's lane-glob/orphan bucketing.
_lane_role() {
    local name="$1"
    case "$name" in
        _lane-*) printf 'lane' ;;
        _spec-*) printf 'spec' ;;
        *) printf 'orphan' ;;
    esac
}

# ── helper: non-mutating LIVE/IDLE liveness probe (A1/A2/DD1) ─────────────────
# Measures LIVENESS and nothing else, which is why it feeds `live=` and NOT
# `assigned=`: a lane reserved for a task whose consumer is not running probes
# IDLE while remaining very much assigned (see _read_lane_assignment, the
# separate answer).
#
# Opens an EXISTING <dir>.lock read-only and attempts a non-blocking SHARED
# flock on that read-only fd: success (lock acquired) => IDLE, released
# immediately; failure (blocked by a live consumer's exclusive flock) =>
# LIVE. A missing lock file is IDLE and is NEVER created (no `>`-open/
# truncation, no `flock <file> <cmd>` convenience form -- both would mutate
# the pool).
#
# Shared (-s), not exclusive (-x): this probe is a pure reader, and every
# real lane-assignment consumer holds an EXCLUSIVE flock while live
# (mirroring warm-lane-gc.sh's own `flock -n <lock>` reclaim-eligibility
# check, which defaults to exclusive because IT proceeds to a real mutation
# on success -- this script never does). A shared request still correctly
# fails against a live consumer's exclusive lock, so LIVE detection is
# unchanged; but two readers (e.g. two concurrent audit runs) never contend
# with each other. Only a genuine writer's non-blocking attempt could still
# be transiently perturbed, and only for the instant this fd is open (A2) --
# an unavoidable characteristic of any momentary lock-state probe, and
# benign for this script's advisory-only output.
_probe_live() {
    local lock="$1"
    local result='IDLE'
    if [ -e "$lock" ]; then
        if exec 7<"$lock" 2>/dev/null; then
            if flock -n -s 7 2>/dev/null; then
                flock -u 7 2>/dev/null || true
            else
                result='LIVE'
            fi
            exec 7<&- 2>/dev/null || true
        fi
    fi
    printf '%s' "$result"
    return 0
}

# ── assignment state: read the orchestrator's own durable record (A5) ─────────
# _read_lane_assignment <lane>
# Resolves EVERYTHING this script needs from <lane>'s assignment record --
# <STATE_DIR>/<lane>.json, the pool's RESERVATION truth, wholly independent of
# the liveness probe -- from a SINGLE read, and publishes it in three globals:
#
#   LANE_ASSIGNED_STATE  ASSIGNED|RELEASED|QUARANTINED|UNKNOWN
#   LANE_UNKNOWN_CAUSE   why the state is UNKNOWN; EMPTY when it is not
#   LANE_RECORD_TASK_ID  the record's task_id ('' when absent, null, or unread)
#
# A THIN ADAPTER over lib_lane_state.sh since dark-factory task 3074 (PRD
# warm-lane-infra-repatriation-prd.md §8, "extract and unify"). What MOVED into
# the lib is everything that was never really this script's to own: the record
# parse (_record_text/_record_scalar/_lane_record), the non-creating guarantee,
# the single-slurp discipline, and the raw-state -> column table. Those describe
# dark-factory's OWN durable format -- the `state` values ARE its LaneState enum
# (orchestrator/src/orchestrator/lane_lifecycle.py) -- and holding a private
# copy of them in a project-agnostic script is the lockstep duplication this
# extraction removes. The lib's enum-exhaustiveness test is now the drift gate.
# What stayed HERE is only this script's published vocabulary, unchanged at the
# caller boundary; every downstream consumer below is untouched.
#
# Globals, and one slurp, because the three values must describe ONE observation
# of ONE record. The orchestrator rewrites these records on every acquire and
# release, so a second read is a DIFFERENT INSTANT: re-deriving the cause would
# let a record that was absent during the first read and valid during the second
# report `assigned=UNKNOWN` paired with `unrecognized-state:<raw>` -- a cause the
# runbook tells operators to read as dark-factory LaneState schema drift, aiming
# triage at a schema problem that does not exist. lane_state_read publishes all
# three of ITS values from one in-memory snapshot, and the three republished
# below are derived from those alone, so the triple is still self-consistent by
# construction.
#
# A5 folds three DISTINCT causes into the one UNKNOWN column value, and they
# send an operator to three different places, so the stderr warning names which
# one fired rather than asserting a single guess:
#   no-readable-record        no state dir, or no readable <lane>.json there --
#                             the only cause that is a filesystem/permissions
#                             question, and the only one where the file an
#                             operator is told to look for may not exist.
#   unparseable-record        the record IS present and readable, but no `state`
#                             string could be read out of it -- a corrupt,
#                             truncated, or reshaped write.
#   unrecognized-state:<raw>  the record parsed and named a state the mapping
#                             does not know. Reported VERBATIM, because a mass
#                             state_unknown spike carrying one repeated raw
#                             value is the SCHEMA-DRIFT signal (a new
#                             dark-factory LaneState member), and no other cause
#                             looks like that.
# The first two come from the lib (which is where the record is actually read,
# so it is the only layer that can tell them apart). The third is derived HERE,
# from lane_state_class returning UNKNOWN for a NON-EMPTY raw -- deliberately
# not duplicated into the lib, so there is still exactly one copy of the
# recognized-state table anywhere.
LANE_ASSIGNED_STATE='UNKNOWN'
LANE_UNKNOWN_CAUSE=''
LANE_RECORD_TASK_ID=''
_read_lane_assignment() {
    # Preserves _lane_record's `[ -n "$STATE_DIR" ]` guard. Unreachable today
    # (an empty STATE_DIR implies an empty --mount, which skips the resident
    # walk entirely), but kept explicit: without it an empty STATE_DIR would
    # fall through to lane_state_read's derived default, which for the bare
    # lane NAME passed here resolves CWD-relative to ./.lane-state.
    if [ -z "$STATE_DIR" ]; then
        LANE_ASSIGNED_STATE='UNKNOWN'
        LANE_UNKNOWN_CAUSE='no-readable-record'
        LANE_RECORD_TASK_ID=''
        return 0
    fi

    # ONE read. lane_state_read resets its own three globals at entry, so a lane
    # whose record cannot be read never inherits its predecessor's values.
    lane_state_read "$1" "$STATE_DIR" >/dev/null

    LANE_RECORD_TASK_ID="$LANE_STATE_TASK_ID"
    LANE_ASSIGNED_STATE="$(lane_state_class "$LANE_STATE_RAW")"
    LANE_UNKNOWN_CAUSE=''
    if [ "$LANE_ASSIGNED_STATE" = 'UNKNOWN' ]; then
        LANE_UNKNOWN_CAUSE="${LANE_STATE_CAUSE:-unrecognized-state:$LANE_STATE_RAW}"
    fi
    return 0
}

# ── helper: resolve the lane's raw branch (empty string when detached) ────────
# Callers needing a display value substitute "(detached)" for an empty result
# (e.g. the table row); recoverable's PUSHED check needs the raw (possibly
# empty) value so a detached lane never spuriously matches refs/remotes/origin/.
_lane_branch_raw() {
    local dir="$1"
    git -C "$dir" symbolic-ref --short HEAD 2>/dev/null || true
}

# ── backing-task resolution (D6 reuse, lifted from warm-lane-gc.sh) ──────────
# _backing_task_id <dir>
# Prints the numeric task id backing <dir>'s HEAD, or nothing if none can be
# resolved. Never fails under set -e (all git calls guarded with `|| true`).
# attached  — HEAD is on a branch named task/NNNN (purely numeric NNNN).
# detached  — enumerate refs/heads/task/* branches that CONTAIN HEAD; use the
#             id ONLY when exactly one DISTINCT task/NNNN branch matches.
_backing_task_id() {
    local dir="$1"
    local br id
    br="$(git -C "$dir" symbolic-ref --short HEAD 2>/dev/null || true)"
    if [ -n "$br" ]; then
        case "$br" in
            task/*)
                id="${br#task/}"
                case "$id" in
                    ''|*[!0-9]*) return 0 ;;  # non-numeric id — no match
                esac
                printf '%s' "$id"
                ;;
        esac
        return 0  # attached to a non-task (or non-numeric-task) branch — no id
    fi

    # Detached HEAD: resolve via containing task/* branches.
    local -a ids=()
    local ref rid
    while IFS= read -r ref; do
        [ -n "$ref" ] || continue
        rid="${ref#task/}"
        case "$rid" in
            ''|*[!0-9]*) continue ;;  # non-numeric id — skip
        esac
        ids+=("$rid")
    done < <(git -C "$dir" for-each-ref --format='%(refname:short)' --contains HEAD refs/heads/task/* 2>/dev/null || true)

    if [ "${#ids[@]}" -eq 1 ]; then
        printf '%s' "${ids[0]}"
    fi
    return 0
}

# _task_status_raw <id>
# The SINGLE site in this script that invokes the status oracle. Prints the
# whitespace-trimmed RAW status string (e.g. pending / infra-hold / done), or
# nothing when it cannot be resolved. Oracle contract mirrors
# warm-lane-preflight.sh Check 6 / warm-lane-gc.sh _backing_task_terminal
# byte-for-byte: unset STATUS_CMD, empty id, non-zero exit, or empty output
# all yield empty (A3 -- never aborts, never invents a status).
_task_status_raw() {
    local id="$1"
    [ -n "$STATUS_CMD" ] || return 0
    [ -n "$id" ] || return 0
    # `|| true` covers the whole pipeline: under `set -o pipefail` a non-zero
    # oracle exit fails the pipeline even though `tr` itself succeeds.
    "$STATUS_CMD" "$id" 2>/dev/null | tr -d '[:space:]' || true
    return 0
}

# _status_bucket <raw_status>
# The SINGLE definition of the terminal/non-terminal/unknown mapping. Kept
# apart from _task_status_raw so the raw string stays available to callers
# that need full fidelity (the `pin` column) while classification consumers
# share one predicate.
_status_bucket() {
    case "$1" in
        done|cancelled) printf 'terminal' ;;
        '')             printf 'unknown' ;;
        *)              printf 'non-terminal' ;;
    esac
}

# _pin_bucket <pin>
# Buckets a PINNED lane's `pin` column value into the fixed, closed vocabulary
# the aggregate PINNED line reports. The per-lane column keeps full raw
# fidelity; only this rollup bucketizes, and it does so because the buckets are
# the TRIAGE, not a summary:
#   terminal              the holder is done/cancelled -- reclaim now.
#   pending|blocked|      a reservation held by work that is not running; the
#     infra-hold          lane is lost to scheduling, not to a leak.
#   other                 the RESIDUE -- any status outside the buckets above,
#                         and deliberately not a single reading. `in-progress`
#                         here is a probably-crashed agent (running task, no
#                         live consumer), but `deferred`/`review` are
#                         not-running states in `pending`'s family. Triage reads
#                         the per-lane `pin` column, which keeps the raw value
#                         precisely so this rollup never has to guess.
#   unknown               the holder could not be resolved (A3).
#
# `terminal` delegates to _status_bucket so the done|cancelled predicate keeps
# exactly ONE definition in this script. The literal `unknown` is the pin
# column's own sentinel for an unresolvable holder, so it buckets as unknown
# rather than falling through to `other`.
_pin_bucket() {
    local pin="$1"
    case "$pin" in
        ''|unknown) printf 'unknown'; return 0 ;;
    esac
    if [ "$(_status_bucket "$pin")" = "terminal" ]; then
        printf 'terminal'; return 0
    fi
    case "$pin" in
        pending)    printf 'pending' ;;
        infra-hold) printf 'infra-hold' ;;
        blocked)    printf 'blocked' ;;
        *)          printf 'other' ;;
    esac
    return 0
}

# ── recoverable: LANDED | PUSHED | ORPHAN ─────────────────────────────────────
# _recoverable <dir> <branch>
# LANDED <=> HEAD is an ancestor of --main-ref. Else PUSHED <=> HEAD is an
# ancestor of refs/remotes/origin/<branch> (attached branches only -- a
# detached lane, branch=="", is never PUSHED; the merge-base call against a
# nonexistent remote ref fails harmlessly and falls through). Else ORPHAN.
_recoverable() {
    local dir="$1" branch="$2"
    if git -C "$dir" merge-base --is-ancestor HEAD "$MAIN_REF" 2>/dev/null; then
        printf 'LANDED'; return 0
    fi
    if [ -n "$branch" ] && git -C "$dir" merge-base --is-ancestor HEAD "refs/remotes/origin/$branch" 2>/dev/null; then
        printf 'PUSHED'; return 0
    fi
    printf 'ORPHAN'
    return 0
}

# ── helper: name/path matches a glob pattern ──────────────────────────────────
# _matches_glob <value> <comma-separated-globs>
# Lifted verbatim from warm-lane-gc.sh's identically-named helper.
_matches_glob() {
    local value="$1"
    local globs="$2"
    local g
    # Split comma-separated globs
    local IFS=","
    for g in $globs; do
        # shellcheck disable=SC2254
        case "$value" in
            $g) return 0 ;;
        esac
    done
    return 1
}

# ── dirty-state: clean | residue-only | wip ───────────────────────────────────
# _dirty_state <dir>
# git status --porcelain --untracked-files=no (reuse gc.sh's _is_reclaimable
# dirty predicate verbatim: --untracked-files=no excludes untracked artifacts
# like target/, so only uncommitted changes to TRACKED files are considered).
# Empty -> clean. Non-empty AND every changed path matches RESIDUE_GLOB
# (default data/queue/*.db*, the §2/D1 write_queue.db residue) -> residue-only.
# Any non-residue path -> wip (genuine unrecoverable WIP). A `git status`
# failure degrades fail-closed to wip (never mistakenly reclaimable).
_dirty_state() {
    local dir="$1"
    local status_out
    status_out="$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)" || { printf 'wip'; return 0; }
    [ -n "$status_out" ] || { printf 'clean'; return 0; }

    local line path
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        # Porcelain v1 short format: "XY path" (renames/copies: "XY orig -> new"
        # -- use the new path). Strip the 2-char status code + separating space.
        path="${line:3}"
        case "$path" in
            *' -> '*) path="${path#*' -> '}" ;;
        esac
        if ! _matches_glob "$path" "$RESIDUE_GLOB"; then
            printf 'wip'
            return 0
        fi
    done <<< "$status_out"

    printf 'residue-only'
    return 0
}

# ── age_min: whole minutes since <dir>'s mtime ────────────────────────────────
# _age_min <dir>
# floor((now - mtime(dir)) / 60), via lib_portable.sh's portable_mtime. A
# missing/unreadable dir degrades to 0 (never stale) rather than aborting.
_age_min() {
    local dir="$1"
    local mtime now
    mtime="$(portable_mtime "$dir" 2>/dev/null)" || { printf '0'; return 0; }
    now="$(date +%s)"
    printf '%d' $(( (now - mtime) / 60 ))
}

# ── divergent_bytes: measured (real du), never a frozen constant (DD3/G6/D8) ─
# _divergent_bytes <dir>
# `du -sB1 <dir>/target`, in raw bytes (0 when target/ is absent). A du
# failure or unparseable output degrades this figure to 0 with a stderr
# note -- fail-open, this script must never abort (PRD §9.5 inv.12).
#
# Callers floor this to GiB for the per-lane row. The pool-wide HEADROOM
# total is accumulated from these RAW bytes and floored exactly ONCE at
# emission time (sum-then-floor) -- summing already-floored per-lane GiB
# values (floor-then-sum) would systematically undercount: e.g. four lanes
# each holding ~0.9 GiB would each floor to 0 individually and sum to a
# false divergent_gib=0, silently hiding ~3.6 real GiB (performance_accuracy
# fix).
_divergent_bytes() {
    local dir="$1"
    local target="$dir/target"
    [ -e "$target" ] || { printf '0'; return 0; }

    local du_out bytes
    du_out="$(du -sB1 "$target" 2>/dev/null)" || {
        warn "du failed for $target; divergent size degraded to 0"
        printf '0'
        return 0
    }
    bytes="$(printf '%s\n' "$du_out" | awk '{print $1}')"
    if ! printf '%s\n' "$bytes" | grep -qE '^[0-9]+$'; then
        warn "du reported non-integer size for $target; divergent size degraded to 0"
        printf '0'
        return 0
    fi
    printf '%d' "$bytes"
}

# ── classification (monotonically refined across the walk's build-out) ───────
# _classify live assigned_state status recoverable dirty age_min
#
# The rank is LIVE > PINNED > QUARANTINED > (RECLAIMABLE | LEAKED |
# PRESERVED-OK), which is the HEADROOM occupancy partition's order exactly: the
# first three arms are one-for-one the partition's three occupied buckets, and
# the FREE ladder is one-for-one its residue. That correspondence is the point,
# not a coincidence -- a lane counted `quarantined=` on the summary line while
# its own row read `LEAKED` would be the report contradicting itself.
#
# PINNED and QUARANTINED both rank above the whole FREE ladder, which is what
# keeps such a lane out of `leaked=` even when it satisfies the LEAKED predicate
# exactly. The reason is the same for both, and it is about RECLAIM: a
# reservation the pool still holds is a scheduling problem, and a quarantine is
# state the pool deliberately withheld for inspection -- reporting either as a
# leak would invite reclaiming state that a consumer may return to, or that an
# operator is not finished with. (Runbook: "Classification".)
_classify() {
    local live="$1" assigned_state="$2" status="$3" recoverable="$4" dirty="$5" age_min="$6"
    if [ "$live" = "LIVE" ]; then
        printf 'LIVE'; return 0
    fi
    if [ "$assigned_state" = "ASSIGNED" ]; then
        printf 'PINNED'; return 0
    fi
    if [ "$assigned_state" = "QUARANTINED" ]; then
        printf 'QUARANTINED'; return 0
    fi
    # Idle lane that is neither reserved nor withheld.
    if [ "$status" = "terminal" ] || [ "$recoverable" = "LANDED" ] || [ "$recoverable" = "PUSHED" ] || [ "$dirty" = "residue-only" ]; then
        printf 'RECLAIMABLE'; return 0
    fi
    # LEAKED (A4: stale is always the age_min >= STALE_AGE_MIN relation vs the
    # declared knob, never an inline literal). recoverable==ORPHAN is, by
    # _recoverable's construction (HEAD is not an ancestor of MAIN_REF),
    # always "ahead-of-main" in the PRD's sense, so the "(genuine-WIP ∨
    # ahead-of-main)" disjunct is satisfied automatically here.
    if [ "$status" = "non-terminal" ] && [ "$recoverable" = "ORPHAN" ] && [ "$age_min" -ge "$STALE_AGE_MIN" ]; then
        printf 'LEAKED'; return 0
    fi
    printf 'PRESERVED-OK'
}

# ── leak-unknown suspect (A3 observability) ───────────────────────────────────
# _is_leak_unknown_suspect status recoverable age_min
# Mirrors _classify's LEAKED predicate exactly, substituting status=="unknown"
# for status=="non-terminal": true iff the ONLY reason an idle lane isn't
# classified LEAKED is that its backing-task status could not be resolved.
# Never changes the reported classification -- PRESERVED-OK remains A3's
# conservative default -- this purely flags the ambiguity (stderr warning +
# HEADROOM leak_unknown count) so "no leaks" stays distinguishable from
# "leaks could not be evaluated" (an unresolvable status must never silently
# zero the headline leaked metric).
_is_leak_unknown_suspect() {
    local status="$1" recoverable="$2" age_min="$3"
    [ "$status" = "unknown" ] && [ "$recoverable" = "ORPHAN" ] && [ "$age_min" -ge "$STALE_AGE_MIN" ]
}

# ── minimal JSON string escaping (backslash, double-quote, control chars) ────
# _json_escape <string>
# Only lane/branch are free-form (every other column is a fixed enum or
# integer); this covers the characters that would otherwise break the emitted
# JSON.
_json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

# ── resident walk ──────────────────────────────────────────────────────────────
# HEADROOM's occupancy figures are an ORDERED, mutually exclusive PARTITION of
# the resident set -- live > pinned > quarantined > free, mirroring the
# classification rank -- so `resident = live + pinned + quarantined + free`
# holds by construction rather than by coincidence. That identity is the whole
# point: under the old accounting `free` was `resident - live`, so every lane
# the pool had reserved but nothing was running against was counted as
# available capacity (the 2026-07-22 misread). `free` is now the partition
# RESIDUE -- neither live, nor reserved, nor withheld -- and nothing else.
#
# ASSIGNED_COUNT and STATE_UNKNOWN_COUNT are CROSS-CUTS, not partition members:
# they may overlap any bucket (an ASSIGNED lane is live or pinned; an idle
# UNKNOWN-state lane falls to free, a live one is counted live) and must never
# be added into the identity.
RESIDENT=0
LIVE_COUNT=0
PINNED_COUNT=0
QUARANTINED_COUNT=0
FREE_COUNT=0
ASSIGNED_COUNT=0
STATE_UNKNOWN_COUNT=0
# The PINNED breakdown's six buckets (see _pin_bucket). They partition
# PINNED_COUNT exactly -- every pinned lane increments exactly one -- and are
# emitted unconditionally, zeros included, so "no pins" is a readable value
# rather than an absent line.
PIN_PENDING_COUNT=0
PIN_INFRA_HOLD_COUNT=0
PIN_BLOCKED_COUNT=0
PIN_TERMINAL_COUNT=0
PIN_OTHER_COUNT=0
PIN_UNKNOWN_COUNT=0
RECLAIMABLE_COUNT=0
LEAKED_COUNT=0
LEAK_UNKNOWN_COUNT=0
DIVERGENT_TOTAL_BYTES=0
TABLE_OUT=""
JSON_LANE_OBJS=()

# ── A5 observability, at DIRECTORY scope ──────────────────────────────────────
# Resolved ONCE, before the walk: does the state dir exist at all? When it does
# not -- the shape of every pool until dark-factory's LaneLifecycle has written
# its first record, and the shape a first rollout hits -- EVERY resident
# resolves `no-readable-record`, and a per-lane line for each would bury the
# per-lane naming that cause vocabulary exists for: 56 identical warnings say
# "the feature is not deployed here", while the genuinely interesting signal is
# ONE lane's record missing while its neighbours' are present. So the absent-dir
# case warns once, naming the dir, and the per-lane warnings are reserved for a
# dir that IS there. state_unknown accounting is untouched -- still per lane.
#
# The single warning is emitted LAZILY, on the first lane that would have warned
# rather than here, so a mount with no lanes at all (resident=0, nothing to say
# anything about) stays silent.
STATE_DIR_PRESENT=0
if [ -n "$STATE_DIR" ] && [ -d "$STATE_DIR" ]; then
    STATE_DIR_PRESENT=1
fi
STATE_DIR_ABSENT_WARNED=0

# A nonexistent/empty --mount is NOT an error (advisory-only script): the walk
# below simply visits nothing and resident stays 0.
if [ -n "$MOUNT" ] && [ -d "$MOUNT" ]; then
    for entry in "$MOUNT"/*/; do
        entry="${entry%/}"
        [ -d "$entry" ] || continue
        name="${entry##*/}"
        _is_git_worktree "$entry" || continue

        RESIDENT=$((RESIDENT + 1))
        role="$(_lane_role "$name")"

        live="$(_probe_live "$MOUNT/$name.lock")"

        # Independent of `live`: the pool's own reservation truth (A5). ONE read
        # publishes the state, its UNKNOWN cause, and the record's task_id, so
        # all three describe the same instant of the same record.
        _read_lane_assignment "$name"
        assigned_state="$LANE_ASSIGNED_STATE"

        # The four-way partition (see the counter block above). The if/elif
        # chain IS the exclusivity proof -- exactly one bucket is incremented
        # per resident, so the identity cannot drift out of true.
        if [ "$live" = "LIVE" ]; then
            LIVE_COUNT=$((LIVE_COUNT + 1))
        elif [ "$assigned_state" = "ASSIGNED" ]; then
            PINNED_COUNT=$((PINNED_COUNT + 1))
        elif [ "$assigned_state" = "QUARANTINED" ]; then
            QUARANTINED_COUNT=$((QUARANTINED_COUNT + 1))
        else
            FREE_COUNT=$((FREE_COUNT + 1))
        fi

        # Cross-cuts (may overlap the partition; never part of the identity).
        if [ "$assigned_state" = "ASSIGNED" ]; then
            ASSIGNED_COUNT=$((ASSIGNED_COUNT + 1))
        fi
        if [ "$assigned_state" = "UNKNOWN" ]; then
            STATE_UNKNOWN_COUNT=$((STATE_UNKNOWN_COUNT + 1))
            # A5 observability, mirroring A3's leak_unknown warning: the lane is
            # named here so "no pins" stays distinguishable from "pins could not
            # be evaluated". Two things are REPORTED rather than assumed. The
            # CAUSE, because naming a single one would send triage after a
            # missing file that is often sitting right there (see
            # _read_lane_assignment). And the bucket the lane was actually
            # counted in, read off the partition above rather than asserted: an
            # UNKNOWN lane that is LIVE lands in `live`, so a fixed "counted
            # free" would contradict the very HEADROOM line it points at.
            if [ "$live" = "LIVE" ]; then
                unknown_accounting='counted live'
            else
                unknown_accounting='counted free (conservative)'
            fi
            if [ "$STATE_DIR_PRESENT" -eq 1 ]; then
                warn "lane=$name: assignment state unknown ($LANE_UNKNOWN_CAUSE) at ${STATE_DIR:-<unset>}/$name.json; ${unknown_accounting}. See HEADROOM state_unknown."
            elif [ "$STATE_DIR_ABSENT_WARNED" -eq 0 ]; then
                # Whole-directory absence: ONE warning for the dir, not one per
                # lane. No accounting clause -- it differs per lane, and the
                # claim that holds for all of them is the one made here.
                STATE_DIR_ABSENT_WARNED=1
                warn "state dir ${STATE_DIR:-<unset>} does not exist: every lane reports assignment state unknown (no-readable-record) and none is counted pinned, quarantined or assigned. See HEADROOM state_unknown."
            fi
        fi

        raw_branch="$(_lane_branch_raw "$entry")"
        branch="${raw_branch:-(detached)}"

        # Resolve the backing task's raw status ONCE, then bucket it. The raw
        # value is kept because the `pin` column needs full fidelity while
        # `status` needs only the terminal/non-terminal/unknown bucket.
        backing_id="$(_backing_task_id "$entry")"
        backing_raw="$(_task_status_raw "$backing_id")"
        status="$(_status_bucket "$backing_raw")"

        # pin: WHO holds a reserved-but-idle lane, and in what state. Only a
        # lane that is both ASSIGNED and not LIVE is pinned -- a live lane is
        # in use, and a released lane is not reserved at all.
        pin='-'
        if [ "$assigned_state" = "ASSIGNED" ] && [ "$live" != "LIVE" ]; then
            # WHO holds the reservation. The record's task_id is authoritative
            # when present -- a lane's branch name can be stale (or absent, on a
            # detached HEAD), whereas the reservation record is written by
            # whoever made the reservation. Falls back to the branch-derived id
            # when the record carries no usable id, notably `"task_id": null`,
            # which lib_lane_state.sh's scalar reader reports as empty because it
            # requires a quoted value. Both halves come from reads already done
            # above, so nothing
            # here re-opens the record or re-walks the refs.
            pin_id="${LANE_RECORD_TASK_ID:-$backing_id}"
            if [ "$pin_id" = "$backing_id" ]; then
                pin_raw="$backing_raw"   # same task -- don't ask the oracle twice
            else
                pin_raw="$(_task_status_raw "$pin_id")"
            fi
            pin="${pin_raw:-unknown}"
        fi

        recoverable="$(_recoverable "$entry" "$raw_branch")"
        dirty="$(_dirty_state "$entry")"
        divergent_bytes="$(_divergent_bytes "$entry")"
        divergent_gib=$(( divergent_bytes / 1073741824 ))
        age_min="$(_age_min "$entry")"
        DIVERGENT_TOTAL_BYTES=$((DIVERGENT_TOTAL_BYTES + divergent_bytes))

        classification="$(_classify "$live" "$assigned_state" "$status" "$recoverable" "$dirty" "$age_min")"
        case "$classification" in
            RECLAIMABLE) RECLAIMABLE_COUNT=$((RECLAIMABLE_COUNT + 1)) ;;
            LEAKED) LEAKED_COUNT=$((LEAKED_COUNT + 1)) ;;
            PINNED)
                # Keyed off the CLASSIFICATION, not the raw predicate, so the
                # breakdown can never drift out of sync with PINNED_COUNT.
                case "$(_pin_bucket "$pin")" in
                    pending)    PIN_PENDING_COUNT=$((PIN_PENDING_COUNT + 1)) ;;
                    infra-hold) PIN_INFRA_HOLD_COUNT=$((PIN_INFRA_HOLD_COUNT + 1)) ;;
                    blocked)    PIN_BLOCKED_COUNT=$((PIN_BLOCKED_COUNT + 1)) ;;
                    terminal)   PIN_TERMINAL_COUNT=$((PIN_TERMINAL_COUNT + 1)) ;;
                    unknown)    PIN_UNKNOWN_COUNT=$((PIN_UNKNOWN_COUNT + 1)) ;;
                    *)          PIN_OTHER_COUNT=$((PIN_OTHER_COUNT + 1)) ;;
                esac
                ;;
        esac

        # A3 observability: surface (never reclassify) a lane whose LEAKED
        # verdict is suppressed solely because its status is unknown.
        if [ "$classification" = "PRESERVED-OK" ] && _is_leak_unknown_suspect "$status" "$recoverable" "$age_min"; then
            LEAK_UNKNOWN_COUNT=$((LEAK_UNKNOWN_COUNT + 1))
            warn "lane=$name: backing-task status unknown -- cannot confirm LEAKED (would classify LEAKED if status resolved non-terminal); reported PRESERVED-OK. See HEADROOM leak_unknown."
        fi

        TABLE_OUT="${TABLE_OUT}lane=${name} role=${role} live=${live} assigned=${assigned_state} pin=${pin} branch=${branch} status=${status} recoverable=${recoverable} dirty=${dirty} divergent_gib=${divergent_gib} age_min=${age_min} classification=${classification}
"
        JSON_LANE_OBJS+=("{\"lane\":\"$(_json_escape "$name")\",\"role\":\"${role}\",\"live\":\"${live}\",\"assigned\":\"${assigned_state}\",\"pin\":\"$(_json_escape "$pin")\",\"branch\":\"$(_json_escape "$branch")\",\"status\":\"${status}\",\"recoverable\":\"${recoverable}\",\"dirty\":\"${dirty}\",\"divergent_gib\":${divergent_gib},\"age_min\":${age_min},\"classification\":\"${classification}\"}")
    done
fi

# Floor the pool-wide divergent total ONCE from the raw summed bytes
# (sum-then-floor) -- never from a sum of already-floored per-lane GiB
# values (see _divergent_bytes).
DIVERGENT_TOTAL_GIB=$((DIVERGENT_TOTAL_BYTES / 1073741824))

# ── free_gib / budget_gib: measured via the stubbable df seam (DD3/G6/D8) ────
# A df failure or unparseable output degrades both figures to 0 with a stderr
# note (fail-open -- this script must never abort; PRD §9.5 inv.12). Skipped
# (stays 0) for a nonexistent/empty --mount, mirroring the resident walk's
# own guard.
FREE_GIB=0
BUDGET_GIB=0
if [ -n "$MOUNT" ] && [ -d "$MOUNT" ]; then
    df_out="$("$DF" -B1 --output=avail -- "$MOUNT" 2>/dev/null)" || df_out=""
    avail_bytes="$(printf '%s\n' "$df_out" | tail -n +2 | head -n 1 | awk '{print $1}')"
    if printf '%s\n' "$avail_bytes" | grep -qE '^[0-9]+$'; then
        FREE_GIB=$((avail_bytes / 1073741824))
        BUDGET_GIB="$(awk -v f="$FREE_GIB" -v s="$SAFETY" 'BEGIN{printf "%d", (f/s)}')"
    else
        warn "df ($DF) failed or returned unparseable avail bytes for mount '$MOUNT'; free_gib/budget_gib degraded to 0."
    fi
fi

# ── emit: table (default) or json ─────────────────────────────────────────────
# The machine report goes to stdout; all diagnostics above went to stderr
# (unchanged stdout-contract convention, mirroring warm-lane-disk-guard.sh).
if [ "$FORMAT" = "json" ]; then
    lanes_json=""
    if [ "${#JSON_LANE_OBJS[@]}" -gt 0 ]; then
        IFS=,
        lanes_json="${JSON_LANE_OBJS[*]}"
        unset IFS
    fi
    # pinned_by_status sits ALONGSIDE headroom rather than inside it, and
    # carries only the six buckets: their total is headroom.pinned, so
    # repeating it here would be a second copy of a number that must agree.
    printf '{"lanes":[%s],"headroom":{"resident":%d,"live":%d,"pinned":%d,"quarantined":%d,"free":%d,"assigned":%d,"state_unknown":%d,"reclaimable":%d,"leaked":%d,"leak_unknown":%d,"divergent_gib":%d,"free_gib":%d,"budget_gib":%d},"pinned_by_status":{"pending":%d,"infra-hold":%d,"blocked":%d,"terminal":%d,"other":%d,"unknown":%d}}\n' \
        "$lanes_json" "$RESIDENT" "$LIVE_COUNT" "$PINNED_COUNT" "$QUARANTINED_COUNT" "$FREE_COUNT" "$ASSIGNED_COUNT" "$STATE_UNKNOWN_COUNT" "$RECLAIMABLE_COUNT" "$LEAKED_COUNT" "$LEAK_UNKNOWN_COUNT" "$DIVERGENT_TOTAL_GIB" "$FREE_GIB" "$BUDGET_GIB" \
        "$PIN_PENDING_COUNT" "$PIN_INFRA_HOLD_COUNT" "$PIN_BLOCKED_COUNT" "$PIN_TERMINAL_COUNT" "$PIN_OTHER_COUNT" "$PIN_UNKNOWN_COUNT"
else
    printf '%s' "$TABLE_OUT"
    printf 'HEADROOM resident=%d live=%d pinned=%d quarantined=%d free=%d assigned=%d state_unknown=%d reclaimable=%d leaked=%d leak_unknown=%d divergent_gib=%d free_gib=%d budget_gib=%d\n' \
        "$RESIDENT" "$LIVE_COUNT" "$PINNED_COUNT" "$QUARANTINED_COUNT" "$FREE_COUNT" "$ASSIGNED_COUNT" "$STATE_UNKNOWN_COUNT" "$RECLAIMABLE_COUNT" "$LEAKED_COUNT" "$LEAK_UNKNOWN_COUNT" "$DIVERGENT_TOTAL_GIB" "$FREE_GIB" "$BUDGET_GIB"
    # Always emitted, zeros included: a stable grep shape means "no pins" reads
    # as a value an operator can see, never as an absent line.
    printf 'PINNED total=%d pending=%d infra-hold=%d blocked=%d terminal=%d other=%d unknown=%d\n' \
        "$PINNED_COUNT" "$PIN_PENDING_COUNT" "$PIN_INFRA_HOLD_COUNT" "$PIN_BLOCKED_COUNT" "$PIN_TERMINAL_COUNT" "$PIN_OTHER_COUNT" "$PIN_UNKNOWN_COUNT"
fi

exit 0
