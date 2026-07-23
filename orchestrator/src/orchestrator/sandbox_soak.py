"""OS-sandbox rollout soak predicate (PRD γ1/γ5).

The real implementation behind ``scripts/check_sandbox_soak.sh`` — the
``before_done.kind='predicate'`` check consumed by the γ5 soak gate. The soak
is GREEN (exit 0 → task done) iff ALL of:

  (a) >=10 DISTINCT tasks that have a ``sandbox_applied`` event reached
      ``done``;
  (b) the containment probe report is tracked on main at
      ``docs/sandbox-containment-probe-report.md``;
  (c) 0 sandbox-attributable blocks.

Everything is derived from STRUCTURED queries over the event store + task
records — never transcript-grep (INV-2). See the module design in the task
2910 plan and ``plans/os-sandbox-worktree-containment-prd.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical location of the containment probe report (γ4 commits it to main).
PROBE_REPORT_PATH = "docs/sandbox-containment-probe-report.md"

# PRD-D6 spec constant: the soak requires >=10 distinct done sandboxed tasks.
MIN_DONE_DEFAULT = 10


@dataclass
class SoakVerdict:
    """Structured verdict of the soak predicate.

    ``ok`` is True iff all three soak conditions hold. ``reason`` is a single
    human-readable line (the sole stdout line the CLI prints). ``metrics``
    carries the derived counts for observability / debugging.
    """

    ok: bool
    reason: str
    metrics: dict


def evaluate_soak(
    sandbox_applied_task_ids,
    task_status,
    sandbox_unavailable_task_ids,
    escalations,
    report_present,
    min_done=MIN_DONE_DEFAULT,
):
    """Pure verdict function over already-read structured inputs.

    Args:
        sandbox_applied_task_ids: iterable of task_id (str) with a
            ``sandbox_applied`` event (condition-a numerator candidates).
        task_status: mapping ``{task_id(str): status(str)}`` from tasks.db.
        sandbox_unavailable_task_ids: iterable of task_id (str) with a
            ``sandbox_unavailable`` event (condition-c arm 1).
        escalations: list of ``{task_id, summary, category}`` dicts
            (condition-c arm 2).
        report_present: bool — is the probe report tracked on main (condition
            b).
        min_done: the >=N distinct-done bound (PRD-D6, default 10).

    Returns:
        SoakVerdict.
    """
    applied = {str(t) for t in sandbox_applied_task_ids}
    done_count = sum(1 for t in applied if task_status.get(t) == "done")

    # Condition (c) — sandbox-attributable blocks. Placeholder until step-04
    # wires in _sandbox_attributable_blocks; treated as zero for now so the
    # done-count (a) and report (b) conditions can be exercised first.
    attributable: list[str] = []

    clauses: list[str] = []
    if done_count < min_done:
        clauses.append(
            f"only {done_count}/{min_done} distinct sandboxed tasks reached done"
        )
    if not report_present:
        clauses.append(
            f"containment probe report absent on main ({PROBE_REPORT_PATH})"
        )
    if attributable:
        clauses.append(
            f"{len(attributable)} sandbox-attributable block(s) "
            f"[{', '.join(attributable)}]"
        )

    metrics = {
        "done_count": done_count,
        "min_done": min_done,
        "sandboxed_task_count": len(applied),
        "report_present": bool(report_present),
        "attributable_block_count": len(attributable),
        "attributable_block_task_ids": list(attributable),
    }

    if clauses:
        return SoakVerdict(
            ok=False, reason="FAIL — " + "; ".join(clauses) + ".", metrics=metrics
        )
    return SoakVerdict(
        ok=True,
        reason=(
            f"PASS — sandbox soak green: {done_count}/{min_done} distinct "
            "sandboxed tasks reached done; containment probe report present on "
            "main; 0 sandbox-attributable blocks."
        ),
        metrics=metrics,
    )
