#!/usr/bin/env bash
# scripts/check_dashboard_payload_budgets.sh — the dashboard payload budget predicate.
#
# The before_done.kind='predicate' check of the `dashboard-payload-budgets`
# recurrence chain, seeded by task 5835 (OPERATIONS.md, "Payload budgets"). A
# thin launcher for dashboard/src/dashboard/payload_check.py, which measures
# every request in dashboard/src/dashboard/payload_budgets.json against the
# running dashboard.
#
# EXIT-CODE CONTRACT (the DeterministicRunner reads the exit code only):
#   0  every budgeted request is within its budget -> the link is done.
#   1  at least one request is over budget. The output names each one with its
#      largest leaf fields. Remedy: shrink the payload, or merge an appended,
#      reasoned raise to its history, then resolve the escalation to re-run.
#   2  the instrument failed and nothing measured over: the dashboard was
#      unreachable, a request timed out or answered non-200, a request was not
#      reached before the run deadline, or the table could not be read. Never
#      reported as an overage.
# Both non-zero codes surface as a born-at-L2 milestone_check_failed.
#
# Arguments are forwarded unchanged (--base-url, --timeout, --run-deadline,
# --table). The python invocation is overridable via CHECK_DASHBOARD_PAYLOAD_PY
# so tests can substitute a stub command without spinning up uv.
#
# This path is a contract with task 5835's seed link: do not rename it.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# shellcheck disable=SC2086  # CHECK_DASHBOARD_PAYLOAD_PY is a command word list.
exec ${CHECK_DASHBOARD_PAYLOAD_PY:-uv run --frozen --project "$REPO_ROOT/dashboard" python -m dashboard.payload_check} "$@"
