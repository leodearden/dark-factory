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

    Two branches, in precedence order:

    1. The bare ``finding['task_id']`` field, INTERPRETED as belonging to
       *project_id* without being matched against it.
    2. Otherwise, the first ``finding['cited_tasks']`` entry whose
       ``project_id`` EQUALS *project_id*.

    The asymmetry between them is deliberate and is the guard that keeps a
    foreign project's task id out of this project's queue.  The task-4185
    operator ruling recorded in ``fused_memory/server/recon_report.py`` (the
    projectless-signature comment on the in-run signature index) states it
    plainly: an ``add_finding(task_id='42', ...)`` call "carries no project
    whatsoever", so a run containing two projects' findings about task 42 can
    collapse there and no guard at that layer can tell that from a genuine
    duplicate.  A bare ``task_id`` therefore NAMES no project and cannot be
    project-matched — only interpreted.  A ``cited_tasks`` entry, by contrast,
    IS project-qualified (``{project_id, task_id, title}``, written by
    ``cite_task``), so it can be matched — and is, because routing a foreign
    project's id onto this project's same-numbered task would file a record
    that looks entirely well-formed while pointing at an unrelated task.

    Cross-project routing (filing into the OTHER project's queue) is
    deliberately not attempted: it would require resolving a second project's
    root and reasoning about a second orchestrator's liveness.  A finding whose
    only citations are foreign resolves to None and is simply not routed.

    Both branches are LLM-authored input, so every value is coerced with
    ``str()`` and stripped, and a malformed ``cited_tasks`` entry is SKIPPED
    rather than raised — a bad citation must not abort the remediation pass.
    """
    raw = finding.get('task_id')
    if raw is not None:
        candidate = str(raw).strip()
        if candidate:
            return candidate

    cited = finding.get('cited_tasks')
    if not isinstance(cited, (list, tuple)):
        return None
    for entry in cited:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get('project_id') or '').strip() != project_id:
            continue
        entry_task_id = entry.get('task_id')
        if entry_task_id is None:
            continue
        candidate = str(entry_task_id).strip()
        if candidate:
            return candidate
    return None
