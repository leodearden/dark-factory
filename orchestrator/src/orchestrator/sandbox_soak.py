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

import re
from dataclasses import dataclass

# Canonical location of the containment probe report (γ4 commits it to main).
PROBE_REPORT_PATH = "docs/sandbox-containment-probe-report.md"

# PRD-D6 spec constant: the soak requires >=10 distinct done sandboxed tasks.
MIN_DONE_DEFAULT = 10

# Arm-2 attribution token: an out-of-set path denial (D9 errnos) surfaced
# through a structured escalation summary — matched on the errno token only
# (never the word "sandbox"), keeping arm 2 precise and non-overlapping with
# arm 1.
_ERRNO_RE = re.compile(r"\b(EACCES|EROFS)\b")


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


def _sandbox_attributable_blocks(task_status, sandbox_unavailable_task_ids, escalations):
    r"""Return the sorted task_ids that are a sandbox-attributable block.

    A block is sandbox-attributable iff the task's CURRENT status is ``blocked``
    AND either:
      * arm 1 — its task_id has a ``sandbox_unavailable`` event (a fail-closed
        refusal), or
      * arm 2 — it has an ``escalation_created`` event whose structured summary
        matches ``\b(EACCES|EROFS)\b`` (a denial on an out-of-set path).

    Correlation is strictly by task_id — both event types carry task_id as a
    first-class column, so no fuzzy timestamp window is needed (INV-2). A
    ``sandbox_unavailable`` event for a task that later recovered (not currently
    ``blocked``) is not counted.
    """
    unavailable = {str(t) for t in sandbox_unavailable_task_ids}
    errno_task_ids = set()
    for esc in escalations:
        tid = esc.get("task_id")
        if tid is None:
            continue
        if _ERRNO_RE.search(esc.get("summary") or ""):
            errno_task_ids.add(str(tid))
    blocked = {str(t) for t, status in task_status.items() if status == "blocked"}
    attributable = {t for t in blocked if t in unavailable or t in errno_task_ids}
    # Numeric ids sort numerically (5 before 10); non-numeric ids fall after.
    return sorted(
        attributable, key=lambda t: (0, int(t)) if t.isdigit() else (1, t)
    )


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

    # Condition (c) — sandbox-attributable blocks (PRD Open Q2).
    attributable = _sandbox_attributable_blocks(
        task_status, sandbox_unavailable_task_ids, escalations
    )

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
