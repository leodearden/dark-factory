"""Pure helpers that route a Stage-3 finding naming a task id onto that task's
ORCHESTRATOR escalation queue (task 4821, arm 3 of task 4764).

## Why this exists

A Stage-3 integrity finding that names a task id currently DEAD-ENDS in memory.
``ReconciliationHarness._escalate`` files it to the RECON queue
(``config.escalation_queue_dir``, drained by the port-8103 watcher) under a
SYNTHETIC ``recon-<run8>`` task id — so the real task id survives only inside
the JSON ``detail`` blob and the dedupe fingerprint's ``affected_ids``. Nothing
on the task's own ladder ever sees it.

The motivating incident (reify task 4458): an operator resolved ``esc-4458-87``
choosing Option A, and an implementer then committed the opposite. Reconciliation
DID flag the divergence — and the flag went to a queue nobody reading task 4458
would ever look at.

This module supplies the two PURE halves of the fix: deciding WHICH task a
finding names, and building the escalation payload. The impure half — queue
construction, dedupe and submit — lives on the harness as
``ReconciliationHarness._file_finding_task_escalation``, because it needs the
harness's ``_known_projects`` registry and its guarded ``HAS_ESCALATION``
import.

Explicitly OUT OF SCOPE (per task 4764's own wording): semantic contradiction
DETECTION. This arm is the plumbing only.

## Purity

This module performs NO filesystem or network I/O and imports nothing from the
``escalation`` package (which is an optional workspace member — see the guarded
imports in ``reconciliation/targeted.py`` and ``reconciliation/harness.py``).
That keeps it hermetically testable from plain dicts, mirroring the stated
purity contract of the sibling module ``reconciliation/predicate_contradiction.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    'resolve_finding_task_target',
]


def resolve_finding_task_target(
    finding: Mapping,
    project_id: str,
) -> str | None:
    """Return the task id *finding* names within *project_id*, or None.

    ``finding`` is a ``flagged_items`` finding dict as produced by the
    ``finding_dict`` projection in ``fused_memory/server/recon_report.py`` (and
    as carried through the harness's remediation pass).

    The bare ``finding['task_id']`` field is LLM-authored and copied verbatim by
    that projection, so it is coerced with ``str()`` and stripped; an absent,
    None, or blank-after-strip value yields None.
    """
    raw = finding.get('task_id')
    if raw is not None:
        candidate = str(raw).strip()
        if candidate:
            return candidate
    return None
