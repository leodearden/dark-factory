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

import json
from collections.abc import Mapping
from typing import Any

__all__ = [
    'FINDING_TASK_ESCALATION_CATEGORY',
    'build_finding_task_escalation_kwargs',
    'resolve_finding_task_target',
]

# Escalation category for a Stage-3 finding routed onto its named task's
# ORCHESTRATOR queue.  A new, honest, greppable name that deliberately does NOT
# over-claim contradiction semantics: full semantic contradiction DETECTION is
# out of scope for this arm (task 4764's own wording), so a category like
# `recon_contradiction` would promise a calibration this plumbing does not have.
#
# REFACTOR TRIGGER, forwarded: `Escalation.category` in
# `escalation/src/escalation/models.py` carries a task-3709 comment saying the
# NEXT category addition should promote that prose vocabulary to an enum (or a
# submit-time lint) rather than growing another line.  THIS IS THAT ADDITION.
# Promoting it is out of scope here — it touches the shared `escalation`
# package, whose filer/reader spelling contract is depended on by every
# categorized detector — so the trigger is recorded rather than discharged.
# Because nothing rejects a typo'd category at submit time, the name is
# single-sourced through this constant: the filer and the `has_open_l1` dedupe
# read the SAME symbol, which is the property the trigger's comment says the
# dedup correctness of a categorized detector depends on.
FINDING_TASK_ESCALATION_CATEGORY = 'recon_task_finding'

# Fixed routing fields for every record this module builds.
#
# `level=1` is required TWICE OVER: it routes to the auto-watcher (which
# promotes to L2 when human judgement is needed — the ladder this arm exists to
# reach), and `EscalationQueue.has_open_l1` reads level-1 records ONLY, so the
# cross-cycle dedupe simply does not function at level 0.
_ESCALATION_LEVEL = 0

# `severity='info'`, NOT the 'blocking' used by the `_sweep_escalate_l1`
# template this filer is otherwise transcribed from.  An open L1 is documented
# (`EscalationQueue.has_open_l1`) as signalling that "the workflow must not
# auto-requeue the task".  Asserting that for every persistent recon finding
# that happens to name a task id would silently change task-blocking semantics
# fleet-wide as a side effect of a PLUMBING change.  'info' still lands the
# record queued and triaged — the entire acceptance criterion — while the
# follow-up semantic-contradiction arm can file at 'blocking' on the narrower
# population where it is warranted.  Matches the closest fused-memory precedent
# for a code-path orchestrator-queue filing, `middleware/scope_violation_
# escalator.py` (level 1, severity 'info').
#
# Severity is deliberately NOT derived from the finding's own
# 'minor'/'moderate'/'serious' field: that vocabulary is LLM-authored free text
# with no calibration against the escalation ladder's semantics.
_ESCALATION_SEVERITY = 'info'

_AGENT_ROLE = 'reconciliation-harness'


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


def build_finding_task_escalation_kwargs(
    finding: Mapping,
    *,
    task_id: str,
    project_id: str,
    run_id: str,
    persistence: int,
) -> dict[str, Any]:
    """Build the ``Escalation(...)`` keyword arguments for *finding*.

    *task_id* is the ALREADY-RESOLVED target from
    :func:`resolve_finding_task_target` — the REAL task id, never the synthetic
    ``recon-<run8>`` id that ``ReconciliationHarness._escalate`` puts on the
    recon queue.  Carrying the real id is the entire point of this arm: it is
    what makes ``get_by_task`` (which filters on the STORED ``task_id`` field)
    surface the record on the task's own ladder.

    Returns a dict whose keys are a strict subset of
    ``Escalation.__dataclass_fields__`` MINUS ``id``:

    - No ``id``: it must come from ``queue.make_id(...)``, a durable per-key
      counter, which a pure function has no access to.
    - No key outside the dataclass: ``Escalation.to_dict`` is a bare ``asdict``
      and ``from_dict`` filters to ``__dataclass_fields__``, while
      ``EscalationQueue.resolve`` rewrites the record from ``esc.to_json()`` —
      so an extra key on disk is DESTROYED on the first resolve.  That is the
      defect ``BacklogPolicy._restore_policy_keys`` exists to work around; ALL
      provenance goes into ``detail`` (a real field) as JSON so we never need
      that workaround.
    - No ``dedupe_fingerprint``: nothing on the orchestrator queue folds on one
      for this category — cross-cycle dedupe is ``has_open_l1``'s job — and
      setting one risks unintended folding should a future ``submit_or_dedupe``
      config ever name the category.  The recon-side fingerprint is preserved in
      ``detail`` for correlation instead.
    - No ``suggested_action``: it is left at its ``''`` default rather than
      guessed from the ``expand_scope|create_followup_task|abort_task``
      vocabulary, because the correct disposition is exactly what the ladder
      exists to decide.  The finding's OWN ``suggested_action`` text is carried
      in ``detail``.

    Pure: no I/O, and *finding* is not mutated.
    """
    finding_category = str(finding.get('category') or '') or 'unknown'
    description = str(finding.get('description') or '') or '(no description)'
    summary = (
        f'Recon finding on task {task_id} ({finding_category}, '
        f'{persistence} cycles): {" ".join(description.split())}'
    )

    detail = json.dumps(
        {
            'finding_id': finding.get('finding_id'),
            'category': finding.get('category'),
            'severity': finding.get('severity'),
            'description': finding.get('description'),
            'suggested_action': finding.get('suggested_action'),
            'actionable': finding.get('actionable'),
            'affected_ids': finding.get('affected_ids'),
            'cited_tasks': finding.get('cited_tasks'),
            'run_id': run_id,
            'project_id': project_id,
            'persistence': persistence,
        },
        default=str,
        indent=2,
    )

    return {
        'task_id': task_id,
        'agent_role': _AGENT_ROLE,
        'severity': _ESCALATION_SEVERITY,
        'category': FINDING_TASK_ESCALATION_CATEGORY,
        'summary': summary,
        'detail': detail,
        'level': _ESCALATION_LEVEL,
    }
