"""Shared filing gate for single-shot CRITICAL escalations (task 2558).

A single observation is never sufficient to file a destructive-intervention
CRITICAL — the kind that recommends a ref move / rewind / "roll back to
last-green".  The survey §1.7 precedent: a "last-green" rewind recommendation
named a commit that ALSO failed when it was finally run — the evidence was never
executed, or had since mutated.  So before such an alarm fires, the observation
must be corroborated by EITHER:

- a positive re-run confirmation on the current state (``rerun_confirmed``), OR
- a consecutive-failure streak reaching an explicit threshold
  (``streak >= threshold``).

Two live callers delegate here:

- ``MergeQueue._record_runner_unavailable`` — counts a persistent per-host
  consecutive-failure streak and passes an int ``threshold``
  (``verify_host_unreachable_escalate_after_n``, default 3).  The streak arm
  reproduces its original ``streak >= self._unreachable_escalate_after_n``
  predicate exactly.
- ``Harness._run_main_tip_sweep`` — has no cross-cycle streak (it skips
  re-verifying a non-advancing SHA, so a streak can never accumulate on a stuck
  red main).  It passes ``rerun_confirmed`` (task 2370's subset re-run AND the
  current tip still being the observed bad SHA) with ``threshold=None`` so the
  streak arm is inert.
"""

from __future__ import annotations


def critical_filing_gate(
    *,
    rerun_confirmed: bool = False,
    streak: int = 0,
    threshold: int | None = None,
) -> bool:
    """Return whether a single-shot CRITICAL escalation is corroborated enough to file.

    ``rerun_confirmed`` short-circuits to ``True`` regardless of the streak/threshold
    (a positive re-run on the current state is sufficient corroboration on its own).

    Otherwise the streak arm applies: it fires only when ``threshold`` is an explicit
    int AND ``streak >= threshold``.  ``threshold=None`` is the sentinel for "no
    streak gate" — it disables the streak arm entirely, so a rerun-only caller can
    pass ``streak=0`` without accidentally tripping ``0 >= 0``.
    """
    return rerun_confirmed or (threshold is not None and streak >= threshold)
