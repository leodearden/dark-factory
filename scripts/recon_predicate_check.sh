#!/usr/bin/env bash
# recon_predicate_check.sh — RECON_PREDICATE_CHECK reference runner (task 2779).
#
# A committed, executable reference target for a recon-stage predicate task
# filed via build_predicate_contradiction_task (see
# fused-memory/src/fused_memory/reconciliation/predicate_contradiction.py). It
# settles a recon-stage contradiction that only LIVE command execution can
# decide — a specific pytest test, or a git grep/log presence check — instead
# of an ad-hoc Mem0 investigation note that silently ages across cycles.
#
# EXIT-CODE CONTRACT (the DeterministicRunner parses the exit code ONLY, never
# stdout — mirroring scripts/check_merge_flakiness.sh):
#   0        predicate settled AS EXPECTED  -> task done
#                                              (done_provenance.kind='deterministic-milestone')
#   non-zero predicate MISMATCH / not found -> born-at-L2 milestone_check_failed
#                                              escalation + task blocked
#   2        usage error (unknown/missing mode)
#
# The builder stays generic — a caller may point before_done.script at ANY
# project-relative committed executable; this runner is the concrete reference
# implementation for the check shapes named in task 2779.
#
# Usage:
#   recon_predicate_check.sh --pytest <pytest-arg>...
#   recon_predicate_check.sh --git-grep <pattern> [-- <path>...]
#   recon_predicate_check.sh --git-log-grep <pattern>
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: recon_predicate_check.sh <mode> [args...]
  --pytest <pytest-arg>...          run `python -m pytest` with the given args;
                                    predicate holds (exit 0) iff pytest passes
  --git-grep <pattern> [-- <path>]  exit 0 iff the pattern is found in tracked
                                    files (git grep), exit 1 if absent
  --git-log-grep <pattern>          exit 0 iff `git log --grep=<pattern>` is
                                    non-empty, exit 1 otherwise
EOF
}

mode="${1:-}"
case "$mode" in
    --pytest)
        shift
        # Run pytest with the remaining args; its exit status IS the verdict.
        exec python -m pytest "$@"
        ;;
    --git-grep)
        shift
        [ $# -ge 1 ] || { echo "recon_predicate_check.sh: --git-grep requires a <pattern>" >&2; exit 2; }
        pattern="$1"
        shift
        # Preserve git grep's exit code faithfully: 0 = found, 1 = not found,
        # 2+ = error. Capture explicitly (mirrors check_merge_flakiness.sh's
        # awk idiom) so `set -e` does not swallow the not-found verdict.
        git grep -q -e "$pattern" "$@" && rc=0 || rc=$?
        exit "$rc"
        ;;
    --git-log-grep)
        shift
        [ $# -ge 1 ] || { echo "recon_predicate_check.sh: --git-log-grep requires a <pattern>" >&2; exit 2; }
        pattern="$1"
        if [ -n "$(git log --grep="$pattern" --oneline)" ]; then
            exit 0
        else
            exit 1
        fi
        ;;
    *)
        if [ -n "$mode" ]; then
            echo "recon_predicate_check.sh: unknown mode: $mode" >&2
        else
            echo "recon_predicate_check.sh: a mode argument is required" >&2
        fi
        usage
        exit 2
        ;;
esac
