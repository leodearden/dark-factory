#!/usr/bin/env bash
# scripts/warm-lane-degenerate-ref-check.sh — Read-only warm-lane degenerate
# task-branch-pointer classifier + fleet-audit primitive.
#
# Cross-repo seam primitive: reify ships this classifier; dark-factory wires
# BOTH fix angles that consume it (see
# docs/design/warm-lane-degenerate-ref-seam.md for full root-cause + wiring
# documentation):
#   angle B — DF's citation-missing / phantom-done reconciliation sweep skips
#             a ref this script classifies `degenerate` (never a landed
#             phantom; re-firing on it is the observed escalation-storm bug).
#   angle A — DF's re-block path deletes a ref this script classifies
#             `degenerate` (DF owns the delete; this script never mutates a
#             ref, mirroring the warm-lane-ref-check.sh / #4855 precedent).
#
# Root cause addressed (task #5006):
#   refs/heads/task/N is created by dark-factory's acquire path
#   (`git worktree add -b task/N <lane> <base>`), where <base> is a recent
#   main commit that is frequently ANOTHER task's no-ff merge commit ("Merge
#   task/<other> into main"). When the task faults/re-blocks before
#   producing its first commit, the ref is left parked on that foreign
#   main-ancestor with ZERO of its own commits — indistinguishable from a
#   landed-but-uncited phantom-done branch to a naive `is_ancestor(task/N,
#   main)` check. DF's citation-missing sweep re-fires on this ref every
#   reconciliation pass while the task stays blocked.
#
# Discriminant (see the seam doc for the full rationale):
#   degenerate <=> rev-list --count <main>..task/N == 0
#                  AND the branch tip does NOT cite task N
#   count==0 is mathematically equivalent to is_ancestor(task/N, main), so
#   the only thing distinguishing "degenerate, parked on a foreign ancestor"
#   from "genuinely landed" is whether the tip commit cites ITS OWN task id.
#
# The citation predicate is NOT defined here. It lives in
# scripts/lib_task_citation.sh, the single copy of a grammar that is normative
# across the reify/dark-factory seam, which this script sources: a
# `Merge <prefix><id> into …` line, a conventional-commit subject citing the id
# (`impl(<id>): …`, dark-factory's kind list), OR a `#<id>` reference, each
# boundary-safe (task/1 does not match "Merge task/10 into main"; impl(5) does
# not match impl(50); #45 does not match #4588). Do not re-inline it here — see
# that file's header, which also records where it agrees with dark-factory.
# That lib is the SINGLE copy; dark-factory ports no separate lib test, as for
# lib_live_refs.sh and lib_portable.sh, so re-inlining either ERE here would
# reintroduce the drift the lib exists to prevent with nothing to catch it.
#
# The conventional-commit arm is what makes `landed` reachable for the usual
# tip: a branch whose last commit reads "fix(<id>): …" and is on main. Without
# it, 81 of the 427 refs this script reported degenerate over the live pool
# were such tips (2026-09-17, esc-7244-16: every one of the 81 flipped to
# landed, and no landed ref flipped back) — and dark-factory treats degenerate
# as "zero task work", downgrading a MARK_DONE recovery to a
# revert-and-redispatch.
#
# The claim this replaced — that the predicate mirrored
# orchestrator/git_ops.py::DEFAULT_COMMIT_CITATION_PATTERN byte-for-byte — was
# FALSE IN BOTH DIRECTIONS: that pattern has no `#<id>` arm at all, and its
# conventional-commit alternative was absent from this copy.
#
# Usage — two mutually exclusive modes:
#
#   1) Single-ref classify:
#        scripts/warm-lane-degenerate-ref-check.sh --task <id> \
#            [--main-ref <ref>] [--branch-prefix <pfx>] [--repo <dir>|-C <dir>]
#      Resolves refs/heads/<prefix><id> and prints `<class> <tip_sha>` on
#      stdout (or `absent -` if the ref does not exist).
#
#   2) Fleet audit:
#        scripts/warm-lane-degenerate-ref-check.sh --audit \
#            [--main-ref <ref>] [--branch-prefix <pfx>] [--repo <dir>|-C <dir>] \
#            [--status-cmd <cmd>]
#      Enumerates every refs/heads/<prefix>* ref, classifies each, and prints
#      one machine-readable row per ref plus a summary line.
#
# Options:
#   --task <id>             Task id to classify (numeric only). Mutually
#                           exclusive with --audit.
#   --audit                 Fleet-audit mode. Mutually exclusive with --task.
#   --main-ref <ref>        Ref to diff/ancestor-check against (default: main).
#   --branch-prefix <pfx>   Branch-name prefix (default: "task/").
#   --repo <dir>, -C <dir>  Repo/worktree to operate in (default: CWD).
#   --status-cmd <cmd>      (--audit only) Optional advisory status oracle:
#                           invoked as `<cmd> <task_id>`, expected to print a
#                           task status (e.g. done/cancelled/blocked) to
#                           stdout. Empty output or non-zero exit is treated
#                           as `unknown` (non-terminal). Also settable via
#                           REIFY_DEGENERATE_REF_STATUS_CMD. Mirrors
#                           warm-lane-preflight.sh Check 6's
#                           REIFY_LANE_LEAK_STATUS_CMD contract.
#   -h, --help              Print this message and exit 0.
#
# Stdout contract:
#   Single-ref mode: exactly one line, `<class> <tip_sha>` (or `absent -`).
#   Audit mode: one `<task_id> <class> <tip_sha> [status]` row per ref,
#     followed by one `audit: degenerate=.. live=.. landed=.. absent=..
#     total=.. flagged=..` summary line.
#   All diagnostics go to stderr. This script NEVER creates, moves, or
#   deletes a ref (read-only diagnostic).
#
# Exit codes (single-ref mode; audit mode always exits 0 on a completed sweep):
#   0  — degenerate  (count==0, tip does not cite N — skip-sweep / prune-safe)
#   1  — live        (count>0 — own commits ahead of main)
#   2  — usage/WIRING error: an unknown flag, a missing flag value, OR a
#        missing sibling library (scripts/lib_task_citation.sh). The library
#        case is a wiring error, not a runtime one: nothing about the
#        invocation could have avoided it and no retry will fix it — the
#        deployment is incomplete. Same class as an unknown flag, hence 2.
#   3  — structural  (not a git work tree, or --main-ref unresolvable)
#   4  — landed      (count==0, tip DOES cite N — genuinely merged)
#   5  — absent      (no such ref)
#
# See: docs/design/warm-lane-degenerate-ref-seam.md

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

# ── the citation grammar (sourced, never re-inlined) ─────────────────────────
# scripts/lib_task_citation.sh is the SINGLE copy of the "cites task N"
# grammar; see its header for why that matters across the reify/dark-factory
# seam. Re-inlining either ERE here is what the lib exists to prevent.
#
# Fail LOUDLY if the lib is missing: without it EVERY classification is wrong,
# and an absent sibling is an incomplete DEPLOYMENT, never a data problem --
# exit 2 (usage/WIRING), not the 1 bash's own bare `source` failure produces
# under `set -e`. reify's copy does a bare `source`; this guard is the
# dark-factory divergence recorded in README.md "Delta 11", in the shape task
# 3370 ("Delta 8") established for lib_portable.sh.
if [ ! -f "$SCRIPT_DIR/lib_task_citation.sh" ]; then
    echo "warm-lane-degenerate-ref-check.sh: ERROR — scripts/lib_task_citation.sh not found next to warm-lane-degenerate-ref-check.sh" >&2
    exit 2
fi
# shellcheck source=orchestrator/scripts/warm-lane/lib_task_citation.sh
source "$SCRIPT_DIR/lib_task_citation.sh"

# ── log helpers (all write to stderr) ─────────────────────────────────────────
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*" >&2; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
err()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
hint()  { printf '\033[1;33m[hint]\033[0m  %s\n' "$*" >&2; }

# ── usage ──────────────────────────────────────────────────────────────────────
_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") --task <id> [OPTIONS]
   or: $(basename "$0") --audit [OPTIONS]

  Read-only warm-lane degenerate task-branch-pointer classifier + fleet-audit
  primitive. See the script header for the full contract.

  Options:
    --task <id>             Classify refs/heads/<prefix><id> (numeric only).
    --audit                  Fleet-audit every refs/heads/<prefix>* ref.
    --main-ref <ref>         Ref to compare against (default: main).
    --branch-prefix <pfx>   Branch-name prefix (default: "task/").
    --repo <dir>, -C <dir>   Repo/worktree to operate in (default: CWD).
    --status-cmd <cmd>       (--audit only) Advisory status oracle.
    -h, --help               Print this message and exit 0.

  Exit codes: 0=degenerate 1=live 2=usage/wiring 3=structural 4=landed
              5=absent
    2 also covers a missing sibling scripts/lib_task_citation.sh.

  See: docs/design/warm-lane-degenerate-ref-seam.md
EOF
}

# ── arg parsing ────────────────────────────────────────────────────────────────
TASK_ID=""
AUDIT_MODE=0
MAIN_REF="main"
BRANCH_PREFIX="task/"
REPO_DIR=""
STATUS_CMD="${REIFY_DEGENERATE_REF_STATUS_CMD:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            _usage; exit 0 ;;
        --task)
            [ $# -ge 2 ] || { err "--task requires a value"; exit 2; }
            TASK_ID="$2"; shift 2 ;;
        --audit)
            AUDIT_MODE=1; shift ;;
        --main-ref)
            [ $# -ge 2 ] || { err "--main-ref requires a value"; exit 2; }
            MAIN_REF="$2"; shift 2 ;;
        --branch-prefix)
            [ $# -ge 2 ] || { err "--branch-prefix requires a value"; exit 2; }
            BRANCH_PREFIX="$2"; shift 2 ;;
        --repo|-C)
            [ $# -ge 2 ] || { err "--repo requires a value"; exit 2; }
            REPO_DIR="$2"; shift 2 ;;
        --status-cmd)
            [ $# -ge 2 ] || { err "--status-cmd requires a value"; exit 2; }
            STATUS_CMD="$2"; shift 2 ;;
        *)
            err "Unknown flag: $1"
            err "Run '$(basename "$0") --help' for usage."
            exit 2 ;;
    esac
done

# ── mode validation: exactly one of --task / --audit ──────────────────────────
if [ -n "$TASK_ID" ] && [ "$AUDIT_MODE" -eq 1 ]; then
    err "--task and --audit are mutually exclusive"
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi
if [ -z "$TASK_ID" ] && [ "$AUDIT_MODE" -ne 1 ]; then
    err "Exactly one of --task <id> or --audit is required"
    _usage
    exit 2
fi

# ── numeric --task validation ─────────────────────────────────────────────────
if [ -n "$TASK_ID" ] && ! printf '%s\n' "$TASK_ID" | grep -qE '^[0-9]+$'; then
    err "--task must be a positive integer (got: '$TASK_ID')"
    exit 2
fi

# ── structural preflight (shared by single-ref and audit modes) ──────────────
[ -n "$REPO_DIR" ] || REPO_DIR="."

if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    err "Not inside a git work tree: $REPO_DIR"
    hint "Pass --repo/-C <dir> to point at a git checkout or worktree."
    exit 3
fi

MAIN_SHA="$(git -C "$REPO_DIR" rev-parse --verify "$MAIN_REF" 2>/dev/null || true)"
if [ -z "$MAIN_SHA" ]; then
    err "Cannot resolve --main-ref '$MAIN_REF' in repo: $REPO_DIR"
    hint "Check that the ref exists (e.g. 'git -C $REPO_DIR rev-parse $MAIN_REF')."
    exit 3
fi

# ── shared classify core (single-ref and audit modes both call this) ─────────

# _regex_escape <string>
# Thin local alias for the library's escape, kept so the call sites below read
# unchanged. See task_citation_regex_escape for the contract.
_regex_escape() {
    task_citation_regex_escape "$1"
}

# BRANCH_PREFIX pre-escaped for safe interpolation into _cites_task's ERE
# (computed once; --branch-prefix is fixed after arg parsing).
BRANCH_PREFIX_RE="$(_regex_escape "$BRANCH_PREFIX")"

# _cites_task <commit> <id>
# True iff <commit>'s message cites task <id>. This function owns only the
# FETCH — resolving <commit> to its message text; the grammar itself lives in
# scripts/lib_task_citation.sh, so this copy and reify's consumers of the same
# lib can never drift apart on what "cites task N" means.
_cites_task() {
    local commit="$1" id="$2" msg
    msg="$(git -C "$REPO_DIR" log -1 --format=%B "$commit" 2>/dev/null || true)"
    task_citation_message_cites "$msg" "$id" "$BRANCH_PREFIX_RE"
}

# _classify_ref <id>
# Prints "<class> <tip_sha>" to stdout and returns 0 — never exits, so both
# single-ref and audit mode can call it. Class is one of: absent | live |
# landed | degenerate. Never invokes a ref-mutating git command.
_classify_ref() {
    local id="$1" ref tip count
    ref="refs/heads/${BRANCH_PREFIX}${id}"
    # Capture the resolve RC explicitly (not `set -e`-fatal): an absent ref
    # is a valid classification outcome, not a script error.
    tip="$(git -C "$REPO_DIR" rev-parse --verify "$ref" 2>/dev/null || true)"
    if [ -z "$tip" ]; then
        printf 'absent -\n'
        return 0
    fi
    count="$(git -C "$REPO_DIR" rev-list --count "${MAIN_SHA}..${tip}" 2>/dev/null || true)"
    if [ "$count" != "0" ]; then
        printf 'live %s\n' "$tip"
        return 0
    fi
    if _cites_task "$tip" "$id"; then
        printf 'landed %s\n' "$tip"
    else
        printf 'degenerate %s\n' "$tip"
    fi
    return 0
}

# _ref_status <task_id>
# Invokes the optional advisory status oracle ($STATUS_CMD <task_id>),
# trims whitespace, and treats a non-zero exit or empty output as "unknown"
# (non-terminal). Mirrors warm-lane-preflight.sh Check 6's
# REIFY_LANE_LEAK_STATUS_CMD contract byte-for-byte so the two advisory
# oracles behave identically. Only called when STATUS_CMD is non-empty.
_ref_status() {
    local id="$1" st
    st="$("$STATUS_CMD" "$id" 2>/dev/null | tr -d '[:space:]' || true)"
    printf '%s' "${st:-unknown}"
}

if [ "$AUDIT_MODE" -eq 1 ]; then
    # ── fleet-audit mode ───────────────────────────────────────────────────
    degenerate_n=0; live_n=0; landed_n=0; absent_n=0; total_n=0; flagged_n=0

    while IFS= read -r _ref; do
        [ -n "$_ref" ] || continue
        _id="${_ref#"${BRANCH_PREFIX}"}"
        case "$_id" in
            ''|*[!0-9]*)
                warn "audit: skipping non-numeric branch: $_ref"
                continue ;;
        esac
        _result="$(_classify_ref "$_id")"
        _class="${_result%% *}"
        _tip="${_result#* }"
        total_n=$((total_n + 1))
        case "$_class" in
            degenerate) degenerate_n=$((degenerate_n + 1)) ;;
            live)       live_n=$((live_n + 1)) ;;
            landed)     landed_n=$((landed_n + 1)) ;;
            absent)     absent_n=$((absent_n + 1)) ;;
        esac
        if [ -n "$STATUS_CMD" ]; then
            _status="$(_ref_status "$_id")"
            printf '%s %s %s %s\n' "$_id" "$_class" "$_tip" "$_status"
            if [ "$_class" = "degenerate" ]; then
                case "$_status" in
                    done|cancelled) ;;
                    *) flagged_n=$((flagged_n + 1)) ;;
                esac
            fi
        else
            printf '%s %s %s\n' "$_id" "$_class" "$_tip"
            if [ "$_class" = "degenerate" ]; then
                flagged_n=$((flagged_n + 1))
            fi
        fi
    done < <(git -C "$REPO_DIR" for-each-ref --format='%(refname:short)' "refs/heads/${BRANCH_PREFIX}*" 2>/dev/null)

    printf 'audit: degenerate=%d live=%d landed=%d absent=%d total=%d flagged=%d\n' \
        "$degenerate_n" "$live_n" "$landed_n" "$absent_n" "$total_n" "$flagged_n"
    exit 0
fi

# ── single-ref classify mode ────────────────────────────────────────────────
RESULT="$(_classify_ref "$TASK_ID")"
printf '%s\n' "$RESULT"
CLASS="${RESULT%% *}"
case "$CLASS" in
    absent)     exit 5 ;;
    live)       exit 1 ;;
    landed)     exit 4 ;;
    degenerate) exit 0 ;;
esac
