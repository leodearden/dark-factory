#!/usr/bin/env bash
# scripts/check_sandbox_soak.sh — OS-sandbox rollout soak predicate (PRD γ1/γ5).
#
# Exit-code contract (before_done.kind='predicate', consumed by the γ5 soak
# gate task of plans/os-sandbox-worktree-containment-prd.md): exit 0 iff the
# DF sandbox canary soak is green —
#   * >=10 distinct tasks with sandbox_applied events reached done, AND
#   * the containment probe report is present on main at
#     docs/sandbox-containment-probe-report.md, AND
#   * 0 sandbox-attributable blocks
# — via structured queries over the event store + task records (never
# transcript-grep; INV-2). Non-zero exit => milestone_check_failed born-at-L2.
#
# STUB-NOT-IMPLEMENTED: this is the decompose-time fail-safe stub, committed
# so the γ5 predicate gate could be filed (the submit_task guard requires
# before_done.script to exist and be executable at filing time). Task γ1
# replaces this file with the real implementation (and removes the marker
# above — the delivered-check on γ1 greps for its absence). Until then this
# stub always exits 2: a premature run escalates loudly rather than
# false-passing (fail-safe, INV-4).

echo "check_sandbox_soak: NOT IMPLEMENTED — decompose-time stub. PRD task γ1 (plans/os-sandbox-worktree-containment-prd.md) must land the real soak predicate before this gate can pass." >&2
exit 2
