#!/usr/bin/env bash
# orchestrator/scripts/warm-lane/lib_lane_state.sh — the two dark-factory-
# authoritative facts the warm-lane scripts need, readable from bash.
# Designed to be sourced, not executed directly.
#
# Usage:  source "$(dirname "${BASH_SOURCE[0]}")/lib_lane_state.sh"
#
# The warm-lane scripts are project-agnostic (they ship for any project the
# orchestrator drives), but two of the facts they act on are owned by
# dark-factory and by nothing else:
#
#   1. A LANE'S LIFECYCLE STATE — the orchestrator's own durable record at
#      <worktree_base>/.lane-state/<lane>.json, whose `state` values are the
#      LaneState enum in orchestrator/src/orchestrator/lane_lifecycle.py.
#   2. WHICH WORKTREE BANDS ARE PROTECTED from a reclaim sweep —
#      PROTECTED_PREFIXES in orchestrator/src/orchestrator/git_ops.py.
#
# Both used to be duplicated into the scripts: (1) as a private reader inside
# warm-lane-audit.sh, (2) as a hand-copied glob literal in warm-lane-gc.sh
# carrying a comment that admitted the coupling. That is an INV-5
# (no-lockstep-duplication) violation whose failure mode is SILENT — a band
# added to PROTECTED_PREFIXES that nobody mirrors becomes a live managed
# worktree the reaper will happily remove. This lib is the one definition
# both halves now read.
#
# WHY THE TWO HALVES USE DIFFERENT MECHANISMS
# -------------------------------------------
# The lane-state half is pure bash: NO jq, NO python3. It is called once per
# lane from warm-lane-audit.sh, which runs from a systemd timer and from the
# disk-pressure paths and is forbidden from ever aborting — a hard jq/python3
# requirement there would be a new environmental failure mode for an
# advisory-only read of two flat top-level string scalars.
#
# The protected-prefix half DOES shell out to python3, because
# PROTECTED_PREFIXES cannot be faithfully text-scraped: five of its keys are
# computed constants and one more is config-driven, so any sed/awk parse would
# silently UNDER-render and the failure mode is "a live managed worktree is no
# longer protected". See lane_protect_glob below for its fail-open contract.

# Source guard — prevent double-sourcing.
if [ "${_DF_LIB_LANE_STATE_SH_SOURCED:-}" = "1" ]; then
    return 0 2>/dev/null || true
fi
_DF_LIB_LANE_STATE_SH_SOURCED=1

# ── lane state: read the orchestrator's own durable record ───────────────────

# _lane_state_scalar <record-text> <key>
# Prints the value of a flat top-level STRING scalar in an already-slurped
# record, or nothing on any miss (absent key, or a non-string value such as
# `null`).
#
# THE ONLY site in the shipped warm-lane scripts that extracts a scalar from a
# lane-state record — warm-lane-audit.sh's `_record_scalar` was folded in here
# (PRD §8 extract-and-unify), and the single-definition-site property is pinned
# by orchestrator/tests/test_lane_state_lib.py.
#
# Two properties the regex depends on: the BARE double quote before <key> is
# what makes it safe against a value CONTAINING the key text (json escapes any
# inner quote as \", so `"state"` can only be a real key); and the required
# quotes make a `null` value yield empty — the desired reading for an
# unassigned task_id.
_lane_state_scalar() {
    local text="$1" key="$2"
    printf '%s' "$text" \
        | sed -n -E "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"([^\"]*)\".*/\1/p" \
        || true
    return 0
}

# _lane_state_record <state-dir> <lane>
# THE ONLY site that composes a state-record path. Prints
# <state-dir>/<lane>.json when that record exists and is readable, and NOTHING
# otherwise — so every caller's access is guarded by construction, and the
# non-creating guarantee has exactly ONE place it could be broken rather than
# one per caller. Purely existence/readability tests: no `>`-open, no touch, no
# mkdir, on either the directory or the record. That matters more here than it
# did in warm-lane-audit.sh, because a reclaim sweep is about to read through
# this lib and a read that MINTED an empty record would manufacture the very
# artifact the sweep consults to decide whether a lane is free.
_lane_state_record() {
    local state_dir="$1" lane="$2"
    [ -n "$state_dir" ] && [ -d "$state_dir" ] || return 0
    local record="$state_dir/$lane.json"
    [ -f "$record" ] && [ -r "$record" ] || return 0
    printf '%s' "$record"
    return 0
}

# lane_state_read <lane-dir|lane-name> [<state-dir>]
#
# Resolves EVERYTHING a caller needs about <lane>'s assignment from a SINGLE
# read of <state-dir>/<lane>.json, publishing it in three globals and echoing
# the headline pair on stdout:
#
#   LANE_STATE_RAW      the record's raw `state` string; 'unknown' when none
#                       could be read
#   LANE_STATE_TASK_ID  the record's task_id ('' when absent, null, or unread)
#   LANE_STATE_CAUSE    why the state is unknown; EMPTY when it is not
#   stdout              '<raw>' or '<raw> <task_id>'
#
# <state-dir> defaults to `<dirname of lane-dir>/.lane-state`; an explicit
# second argument WINS and is honoured verbatim, including a path outside the
# mount (warm-lane-audit.sh's --state-dir / REIFY_WARM_LANE_AUDIT_STATE_DIR
# override may point anywhere). The lane argument may be a full directory path
# or a bare lane name.
#
# Globals, and ONE slurp, because the values must describe ONE observation of
# ONE record. The orchestrator rewrites these records on every acquire and
# release, so a second read is a DIFFERENT INSTANT and can report a pair of
# values that never coexisted.
#
# FAILS OPEN to 'unknown', always — callers include a systemd-timer audit and
# the disk-pressure paths, which must never abort on one bad lane. The two
# causes are kept DISTINCT because they send an operator to two different
# places:
#   no-readable-record   no state dir, or no readable <lane>.json there. The
#                        only cause that is a filesystem/permissions question,
#                        and the ordinary reading for every recordless _iact-*
#                        and manual operator worktree (which leaf γ routes to
#                        the /proc liveness fallback).
#   unparseable-record   the record IS present and readable, but no `state`
#                        string could be read out of it — a corrupt, truncated,
#                        or reshaped write.
# A third cause, `unrecognized-state:<raw>`, is NOT set here: it is derived by
# the caller from lane_state_class returning UNKNOWN for a non-empty raw, so
# there is no second copy of the recognized-state table to drift out of sync.
lane_state_read() {
    # Reset FIRST: every lane's triple is resolved from scratch, so a lane whose
    # record cannot be read can never inherit its predecessor's values. Without
    # this the audit's `pin` column would report "lane X is held by task N" —
    # a claim about a lane that X's record never made.
    LANE_STATE_RAW='unknown'
    LANE_STATE_TASK_ID=''
    LANE_STATE_CAUSE=''

    local raw_arg="${1:-}" explicit_dir="${2:-}"

    # dirname/basename by pure parameter expansion — no fork, because this is
    # called once per lane in the audit's resident walk.
    local trimmed="${raw_arg%/}"
    local lane="${trimmed##*/}"
    local parent='.'
    case "$trimmed" in
        */*) parent="${trimmed%/*}"; [ -n "$parent" ] || parent='/' ;;
    esac

    local state_dir="$explicit_dir"
    [ -n "$state_dir" ] || state_dir="$parent/.lane-state"

    local record text
    record="$(_lane_state_record "$state_dir" "$lane")"
    # `2>/dev/null` precedes the input redirection deliberately: bash applies
    # redirections left to right, so the stderr redirect must already be in
    # place to suppress the shell's own "No such file or directory" for a
    # record that vanished between the guard and this read.
    if [ -z "$record" ] || ! text="$(tr -d '\n' 2>/dev/null < "$record")"; then
        LANE_STATE_CAUSE='no-readable-record'
        printf '%s\n' "$LANE_STATE_RAW"
        return 0
    fi

    LANE_STATE_TASK_ID="$(_lane_state_scalar "$text" task_id)"
    local raw
    raw="$(_lane_state_scalar "$text" state)"
    if [ -n "$raw" ]; then
        LANE_STATE_RAW="$raw"
    else
        LANE_STATE_CAUSE='unparseable-record'
    fi

    if [ -n "$LANE_STATE_TASK_ID" ]; then
        printf '%s %s\n' "$LANE_STATE_RAW" "$LANE_STATE_TASK_ID"
    else
        printf '%s\n' "$LANE_STATE_RAW"
    fi
    return 0
}

# lane_state_class <raw>
# The ONE normative raw-state -> column mapping. Prints exactly one of
# ASSIGNED | RELEASED | QUARANTINED | UNKNOWN.
#
# The table lives HERE, in the code — not only in a comment — and this is its
# single definition site across the shipped warm-lane scripts
# (warm-lane-audit.sh's copy was folded in, PRD §8 extract-and-unify):
#
#   assigned, in_use              -> ASSIGNED     (reserved for a task)
#   released, seed, registered    -> RELEASED     (in the pool, not reserved)
#   quarantined                   -> QUARANTINED  (withheld from the pool)
#   anything else, including ''   -> UNKNOWN      (fail open)
#
# The raw values are dark-factory's LaneState enum,
# orchestrator/src/orchestrator/lane_lifecycle.py. That coupling is MACHINE
# CHECKED, not documented-and-hoped: the drift gate in
# orchestrator/tests/test_lane_state_lib.py
# (TestLaneStateClass::test_every_lane_state_enum_member_maps_to_a_known_column)
# imports LaneState and fails the build if any member falls through to UNKNOWN.
# It has to, because the silent failure is severe — a new member would degrade
# every lane carrying it to UNKNOWN, and a pool-wide UNKNOWN spike is
# indistinguishable from a real state-dir outage.
#
# No case folding: the raw values are the enum's lowercase strings, so an
# uppercase spelling is a genuinely unrecognized state and must read that way.
lane_state_class() {
    case "${1:-}" in
        assigned|in_use)          printf 'ASSIGNED\n' ;;
        released|seed|registered) printf 'RELEASED\n' ;;
        quarantined)              printf 'QUARANTINED\n' ;;
        *)                        printf 'UNKNOWN\n' ;;
    esac
    return 0
}

# Published defaults, so a caller that sources this lib and inspects the
# globals before any read sees the same fail-open shape a failed read yields.
LANE_STATE_RAW='unknown'
LANE_STATE_TASK_ID=''
LANE_STATE_CAUSE=''
