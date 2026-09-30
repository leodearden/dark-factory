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
    'FINDING_TASK_ESCALATION_LEVEL',
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
# `level=0`, and the argument for it is STRUCTURAL, not a matter of degree.
#
# Every orchestrator guard that a recon-authored record could hijack reaches
# the escalation queue through ONE helper — `EscalationQueue.has_open_l1`
# (`escalation/src/escalation/queue.py`), which filters on `level` + `status`
# and an OPTIONAL `category`, and is documented as answering "a human is
# already on this task, so the workflow must not auto-requeue it".  Crucially
# it never reads `.severity` at all, so a severity choice cannot mitigate the
# collision — only the level can.  It is level-1-ONLY, so a level-0 record is
# invisible to every one of them by construction:
#
#   - `orchestrator/harness.py::Harness._block_and_escalate_external_dep`, and
#     its siblings `::Harness._block_and_escalate_cross_repo` and
#     `::Harness._block_and_escalate_substrate_flip`, each of which SUPPRESSES
#     its own human-facing L1 and downgrades to a WARNING log when one is
#     already open;
#   - `orchestrator/harness.py::Harness._reap_orphan_l0_escalations`, which
#     DISMISSES a pending orphan L0 on the task rather than promoting it when
#     an L1 is open;
#   - `orchestrator/workflow.py::TaskWorkflow._wait_for_resolution`, which
#     raises `_StewardReescalated` and diverts the workflow on the strength of
#     an open L1;
#   - `orchestrator/workflow.py::TaskWorkflow._await_steward_completion`, the
#     requeue-diversion override, which holds a requeue-producing steward
#     outcome back when an L1 is already open.
#
# BOTH workflow.py sites are uncategorized, and an earlier revision of this
# block listed only the first. That was a measurement error, corrected here
# (esc-4821-5): `_await_steward_completion` was mistakenly reported as NOT
# holding an uncategorized read, when it holds exactly the requeue-diversion
# one. Grep `has_open_l1(` and subtract the `category=` call sites before
# editing this list -- there are two uncategorized readers in workflow.py, not
# one. The omission never changed the DESIGN (level-0 is invisible to every
# level-1-only reader alike), only this list's claim to be complete.
#
# That is the population this closes.  Two classes of `has_open_l1` reader are
# deliberately NOT in it, and were re-measured rather than assumed: the
# sentinel filers (`_ARCHIVAL_STORM_SENTINEL`, `_SCHEDULER_PAUSE_SENTINEL`, the
# `main-sweep-<sha12>` tip sweep, ...) key on SYNTHETIC ids that no real task id
# can collide with, and the CATEGORIZED reads in `orchestrator/workflow.py`
# pass their own `category=`, which `FINDING_TASK_ESCALATION_CATEGORY` never
# matches.  Keep this list in `path::symbol` form (CLAUDE.md) — an earlier
# revision carried bare line pins and two of them had already drifted onto
# unrelated statements.  A passive observation must not start
# gating dispatch as a side effect of a PLUMBING change, and level 0 is the
# only spelling that guarantees it — pinned executably by
# `test_routed_record_is_invisible_to_the_orchestrator_l1_guards`.
#
# Routing note: filed at `severity='info', level=0`.  Level-0 records are
# surfaced via the steward's pending-escalation view and any external monitor
# polling the queue.  UNLIKE the notes on
# `orchestrator/harness.py::_file_starvation_info` and
# `::_file_warm_base_hard_down_notice`, which claim level-0 records "do NOT
# auto-promote", this one does not: `Harness._reap_orphan_l0_escalations` DOES
# promote a pending L0 to L1 once it is older than
# `config.orphan_l0_timeout_secs`, gated on the task having no running workflow
# (`_escalation_events`), not being `scheduler.is_actively_held`, and having no
# already-open L1 — in which last case it DISMISSES the L0 instead, as a
# duplicate of the record the human is already handling.
#
# The consequence is the good one, and it is why no escalate-if-unattended
# behaviour needs asserting here: an unattended routed finding reaches L1 on
# its own, delivered by the component that OWNS strand-detection semantics
# (liveness, hold state, duplicate suppression), while an attended one is
# correctly folded away.  Re-deriving any of that at this filer would mean
# re-deriving all three signals from a process that cannot see them.
#
# PUBLIC (and in `__all__`), like `FINDING_TASK_ESCALATION_CATEGORY` above:
# the level a routed record is filed at is part of this module's CROSS-MODULE
# contract, not an implementation detail.  It is what
# `ReconciliationHarness._file_finding_task_escalation`'s docstring, the
# call-site comment in `_run_remediation_pass` and
# `test_routed_record_is_invisible_to_the_orchestrator_l1_guards` all reason
# about.  A leading underscore would have signalled module-local and invited a
# future reader to inline the literal — and, worse, to weaken it without
# noticing the seven guard sites enumerated above.
FINDING_TASK_ESCALATION_LEVEL = 0

# `severity='info'`, NOT the 'blocking' used by the `_sweep_escalate_l1`
# template this filer is otherwise transcribed from.  This is an OBSERVATION
# about a task, not a claim that the task is broken: the finding is
# LLM-authored and this arm ships the plumbing only, with semantic
# contradiction DETECTION explicitly out of scope (task 4764's own wording).
# The follow-up arm can file at 'blocking' on the narrower population where
# that is warranted.  Matches the closest fused-memory precedent for a
# code-path orchestrator-queue filing, `middleware/scope_violation_
# escalator.py`, in its severity choice.
#
# Note that severity is NOT what keeps this record off the orchestrator's guard
# surface — `has_open_l1` never reads it.  `FINDING_TASK_ESCALATION_LEVEL`
# above does that work, alone.
#
# Severity is deliberately NOT derived from the finding's own
# 'minor'/'moderate'/'serious' field: that vocabulary is LLM-authored free text
# with no calibration against the escalation ladder's semantics.
_ESCALATION_SEVERITY = 'info'

_AGENT_ROLE = 'reconciliation-harness'


def _sole_task_id_part(value: object) -> str | None:
    """Return the single task id inside *value*, or None if it is not exactly one.

    A ``task_id`` field is NOT guaranteed to hold one id. A comma-joined value
    (e.g. ``'5040,5149'``) is a DOCUMENTED supported shape: ``add_finding``
    stores the value canonicalized by
    ``fused_memory/server/recon_report.py::_canonicalize_task_id_string``, which
    splits on ``','``, sorts and REJOINS with ``','`` — and ``flagged_items``
    copies that straight through. So a finding about two tasks arrives here as
    a single joined string.

    Routing such a value verbatim would be worse than not routing it. The id is
    load-bearing twice over: ``EscalationQueue.make_id`` builds
    ``esc-{task_id}-{seq}`` from it, and ``EscalationQueue.get_by_task`` filters
    on EXACT equality of the stored field. A record stored under
    ``'5040,5149'`` is therefore invisible to ``get_by_task('5040')`` AND to
    ``get_by_task('5149')`` — it surfaces on no task's ladder at all, which is
    precisely the dead-end this module exists to close, only now wearing a
    well-formed-looking record and consuming a queue slot. Worse,
    ``Harness._reap_orphan_l0_escalations`` scans ``get_pending()`` without any
    task-existence check and would eventually promote that L0 to L1 under a
    task id that does not exist.

    A multi-part value therefore returns None and is simply NOT ROUTED. This is
    deliberately conservative rather than clever: picking one part would file
    against an arbitrary task and silently drop the others, and splitting into N
    records would breach the volume-parity guarantee documented on
    ``_file_finding_task_escalation``. Deciding which of several tasks a
    contradiction belongs to is semantic work, and semantic detection is
    explicitly out of scope for this arm. Not routing is no worse than today's
    behaviour: the recon-queue filing via ``_escalate`` still fires either way.

    Splitting also normalizes the single-id case, so a value that merely carries
    stray whitespace or a trailing comma (``' 5040 , '``) still resolves to
    ``'5040'`` instead of being routed verbatim.
    """
    if value is None:
        return None
    # Same split shape as recon_report.py::_split_task_id_parts, kept as a local
    # literal rather than an import: this module's purity contract (see the
    # module docstring) keeps it free of server/ imports, and the expression is
    # one line.
    parts = {p.strip() for p in str(value).split(',') if p.strip()}
    if len(parts) != 1:
        return None
    return parts.pop()


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

    Both branches are LLM-authored input, so every value is passed through
    :func:`_sole_task_id_part` (coerce, strip, and reject a comma-joined
    multi-id value — see that docstring for why routing one verbatim is worse
    than not routing it), and a malformed ``cited_tasks`` entry is SKIPPED
    rather than raised — a bad citation must not abort the remediation pass.
    """
    raw = finding.get('task_id')
    if raw is not None:
        candidate = _sole_task_id_part(raw)
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
        # Same multi-part rejection as the bare branch. `cite_task` is expected
        # to write one id per entry, so this is defence in depth rather than a
        # known shape -- but a joined value here would be just as unroutable,
        # and skipping the entry lets a well-formed later citation still win.
        candidate = _sole_task_id_part(entry_task_id)
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
      for this category — cross-cycle dedupe is the filer's own pending-scan
      on ``(level, category)``, see ``_file_finding_task_escalation`` — and
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
        'level': FINDING_TASK_ESCALATION_LEVEL,
    }
