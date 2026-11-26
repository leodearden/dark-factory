"""Files `recon_missing_index` escalations for index-provisioning drift (task 3709, PRD δ).

The I/O half of the FalkorDB index-drift detector.  `index_health` decides WHAT
drifted; this module decides whether to go loud about it, and files exactly one
open level-1 escalation per drifted graph.

Follows `stage1_stall_detector.escalate_stalled_gate_backlog_tasks` line for
line: a module-level category constant, a category-scoped `has_open_l1` dedup
with an INFO-logged suppression, and construction + submit inside ONE try whose
failure logs a WARNING rather than escaping.

Four contracts this module exists to honour:

D11 — `graph:{group_id}` occupies the `task_id` slot as an OPAQUE DEDUP KEY and
is NOT a task reference.  Nothing downstream resolves it to a task: verified
against the storage layer before it was adopted — `EscalationQueue.get_by_task`
filters on the STORED `task_id` field rather than deriving one from a filename
or id, `make_id`'s sidecar counter (`esc-graph:dark_factory.seq`) is unaffected
by the colon (a legal Linux filename byte), and no code in `escalation/`,
`dashboard/` or `orchestrator/` splits an escalation id back into a task_id (the
only id pattern, `authority.py`'s ``r'esc-\\S+'``, matches).  Had any of those
parsed the id, dedup would have silently no-opped and the storm escape would
have failed OPEN — filing one escalation per cycle forever.

INV-2 — `group_id` and the missing specs are carried as first-class
`index_health` fields, so no consumer parses `summary`/`detail` prose to recover
a fact the emitter already had in a variable.  `evidence` still carries one raw
observation, keeping measurement separate from the structured payload.

INV-4 — the standing loud signal is the escalation REMAINING OPEN.  That is what
lets a persistently-missing index file ONCE rather than once per recon cycle.
The dedup is category-scoped so an unrelated open L1 on the same graph key
cannot silently suppress a genuine index finding.

INV-7 — `suggested_action` names a machine-attributable exit owner: the
recon-escalation-watcher on the reconciliation escalation queue (port 8103).

REFACTOR TRIGGER: this is the SECOND non-task dedup key in the escalation queue.
A THIRD grows `has_open_l1` a generic `key=` parameter instead of accumulating
synthetic task_ids — a signature change every caller inherits, which is why it
is deferred until a third caller actually justifies it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    _HAS_ESCALATION = True
except ImportError:  # pragma: no cover - exercised only without the optional package
    Escalation = None  # type: ignore[assignment,misc]
    _HAS_ESCALATION = False


_MISSING_INDEX_ESCALATION_CATEGORY = 'recon_missing_index'
"""``Escalation.category`` for FalkorDB index-provisioning drift (task 3709).

Kept distinct from the Stage 1 aging categories so index findings dedup and
filter independently, and so an open escalation of another kind on the same
graph cannot suppress one of these.
"""

_EXIT_OWNER = (
    'the recon-escalation-watcher on the reconciliation escalation queue (port 8103)'
)


def escalate_missing_indices(
    escalation_queue,
    group_id: str,
    health: dict | None,
    *,
    project_id: str,
    run_id: str | None = None,
) -> str | None:
    """File one level-1 `recon_missing_index` escalation for a drifted graph.

    Args:
        escalation_queue: An `EscalationQueue` (or compatible) to file into.
        group_id: The graph whose index state drifted.
        health: A `summarize_index_health()` record, or None when the graph was
            unreadable or absent (which is NOT drift).
        project_id: Owning project, recorded in the detail block.
        run_id: Reconciliation run that observed this, when there is one — the
            startup sweep has no run.

    Returns:
        The filed escalation id, or None when nothing was filed (healthy graph,
        unknown health, an already-open escalation, or a fail-soft error).

    Never raises: a diagnostics filer must not be able to break the caller.
    That covers the DEDUP READ as well as the filing — `has_open_l1` fans out
    to `get_by_task`, which globs the queue root and parses every pending
    record, so an OSError (fd exhaustion, a permission flap, a disk error) is a
    real failure mode there and not only on `submit`.  Unguarded it would
    propagate into `_detect_index_drift` and cost the caller a health record it
    had ALREADY computed successfully — a queue-read hiccup silently erasing a
    good graph read from the Stage 1 report.
    """
    if health is None or health.get('healthy', False):
        return None

    # Guard on `Escalation is None` rather than `not _HAS_ESCALATION`: the two
    # are equivalent by construction, but only the identity check NARROWS the
    # optionally-imported symbol, so the constructor call below type-checks.
    # This is `stage1_stall_detector`'s idiom verbatim.
    if Escalation is None or escalation_queue is None:
        return None

    dedup_key = f'graph:{group_id}'

    # The dedup read is guarded on its own so its failure mode reads
    # distinctly in the logs from a filing failure.  Failing CLOSED here (not
    # filing) is the deliberate choice: filing blind with no dedup answer would
    # risk exactly the once-per-cycle storm the dedup exists to prevent, and the
    # drift is re-observed every cycle anyway, so a transient read error costs
    # at most a delay — never a lost finding.
    try:
        already_open = escalation_queue.has_open_l1(
            dedup_key, category=_MISSING_INDEX_ESCALATION_CATEGORY
        )
    except Exception as exc:
        logger.warning(
            'index_drift_detector: dedup read failed for group_id=%s; not filing '
            'this cycle (the drift is re-observed next cycle): %s',
            group_id,
            exc,
            extra={'project_id': project_id, 'group_id': group_id},
        )
        return None

    if already_open:
        logger.info(
            'reconciliation.index_drift_escalation_suppressed: graph=%s already has '
            'an open level-1 index-drift escalation',
            group_id,
            extra={'project_id': project_id, 'group_id': group_id},
        )
        return None

    try:
        # Payload construction, make_id() and Escalation() are all inside the
        # try so a surprising health record, an id-generation failure or a
        # constructor failure is caught and logged rather than escaping.
        missing = list(health.get('missing') or [])
        unexpected = list(health.get('unexpected') or [])
        expected_total = health.get('expected_total', 0)

        summary = (
            f'Graph {group_id} is missing {len(missing)} of {expected_total} expected '
            f'FalkorDB indices'
        )
        detail = '\n'.join(
            [
                f'project_id: {project_id}',
                f'run_id: {run_id}',
                f'group_id: {group_id}',
                f'expected_total: {expected_total}',
                f'actual_total: {health.get("actual_total")}',
                f'missing_count: {len(missing)}',
                'missing:',
                *(f'  - {spec}' for spec in missing),
            ]
        )
        suggested_action = (
            f'Provision the missing indices with ensure_indices(group_id={group_id!r}); '
            f'{_EXIT_OWNER} owns closing this escalation once the graph reports healthy. '
            'Unexpected (operator-added) indices are reported for context only and are '
            'never acted on.'
        )

        esc = Escalation(
            id=escalation_queue.make_id(dedup_key),
            task_id=dedup_key,
            agent_role='reconciliation-index-drift',
            severity='blocking',
            category=_MISSING_INDEX_ESCALATION_CATEGORY,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            level=1,
            index_health={
                'group_id': group_id,
                # Specs are stored as JSON LISTS, never tuples: a tuple
                # deserialises back as a list, so storing tuples would make the
                # on-disk record fail to round-trip identically.
                'missing': [list(spec) for spec in missing],
                'unexpected': [list(spec) for spec in unexpected],
                'expected_total': expected_total,
            },
            evidence=[
                {
                    'observation': (
                        f'graph {group_id}: {len(missing)} of {expected_total} '
                        f'expected indices absent'
                    ),
                    'measured_at': datetime.now(UTC).isoformat(),
                    'ref': 'CALL db.indexes()',
                }
            ],
        )
        escalation_queue.submit(esc)
        logger.info(
            'reconciliation.index_drift_escalated: graph=%s missing=%d id=%s',
            group_id,
            len(missing),
            esc.id,
            extra={
                'project_id': project_id,
                'group_id': group_id,
                'missing_count': len(missing),
            },
        )
        return esc.id
    except Exception as exc:
        logger.warning(
            'index_drift_detector: failed to escalate missing indices for group_id=%s '
            '(payload, id-gen, construction, or submit): %s',
            group_id,
            exc,
            extra={'project_id': project_id, 'group_id': group_id},
        )
        return None
