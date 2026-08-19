#!/usr/bin/env bash
# scripts/check_sandbox_soak.sh — OS-sandbox rollout soak predicate (PRD γ1/γ5).
#
# The before_done.kind='predicate' check consumed by the γ5 soak gate task of
# plans/os-sandbox-worktree-containment-prd.md. A thin wrapper delegating to the
# orchestrator module `orchestrator.sandbox_soak`, which derives the verdict
# from STRUCTURED queries over the event store + task records — never
# transcript-grep (INV-2).
#
# EXIT-CODE CONTRACT (the DeterministicRunner parses the exit code ONLY; mirrors
# scripts/recon_predicate_check.sh / scripts/check_merge_flakiness.sh):
#   0  GREEN — all three hold: >=10 DISTINCT tasks with a `sandbox_applied`
#      event reached `done`; the containment probe report is tracked on main at
#      docs/sandbox-containment-probe-report.md; and 0 sandbox-attributable
#      blocks. -> task done.
#   1  measured-but-NOT-yet-green (the legitimate pre-soak state: too few done
#      sandboxed tasks, report absent, or >=1 attributable block). γ5 `resume`
#      re-runs the check = wait longer.
#   2  usage/infra error (missing/unreadable events or tasks DB, bad args, git
#      error resolving the report) — kept DISTINCT from the exit-1 verdict so an
#      infra failure is never misread as a soak verdict.
# Both non-zero codes surface as milestone_check_failed born-at-L2 on γ5. Every
# non-zero exit prints one reason line (0/1 verdict to stdout, 2 error to
# stderr).
#
# The python invocation is overridable via CHECK_SANDBOX_SOAK_PY (mirrors
# fused-memory-flag-marker-check.sh's FLAG_MARKER_SWEEP_CMD) so tests can
# substitute a lightweight stub command without spinning up uv.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# shellcheck disable=SC2086  # CHECK_SANDBOX_SOAK_PY is a command word list.
exec ${CHECK_SANDBOX_SOAK_PY:-uv run --frozen --project "$REPO_ROOT/orchestrator" python -m orchestrator.sandbox_soak} \
    --repo-root "$REPO_ROOT" "$@"
