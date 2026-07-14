"""Curator replay corpus builder (T5): tickets.db -> frontier-adjudicated +
human-spot-checked labeled CuratorReplayItems, split 2:1:7.

See plans/tier1-prompt-optimization-prd.md T5. Mirrors reviewer_trial's
corpus/mining/adjudication machinery (task 2495) but stays decoupled from
fused_memory: tickets.db rows are read as plain dicts by column name via
stdlib sqlite3 (read-only), and target_fingerprint/target_id use a local
minimal action representation -- no import of
``fused_memory.middleware.task_curator``. Every external effect (the
frontier label proposer) is dependency-injected so the builder is fully
hermetic in tests; see ``__main__.py`` for the operator-facing CLI that
wires the real tickets.db + a real frontier proposer.

Decisions != ground truth (PRD D-6): a ticket's RECORDED action/target
(persisted at task-creation time by the live curator) is retained only as
provenance -- gold labels always come from an injected frontier-adjudication
proposer, further checked by a deterministic, action-stratified human
spot-check subset (the Open-Q Sec9 tactical decision this task owns).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'RecordedDecision',
    'read_curator_decisions',
    'recover_recorded_action',
]

_RECOVERABLE_ACTIONS = ('drop', 'combine', 'create')


@dataclass(frozen=True)
class RecordedDecision:
    """One recovered ticket decision -- provenance/weak signal, NOT a gold label.

    See PRD D-6: the recorded action is what the live curator historically
    decided, unverified -- ``build_curator_corpus`` always obtains the GOLD
    label from an injected frontier-adjudication proposer instead.
    """

    ticket_id: str
    candidate: dict
    action: str
    target_fingerprint: str | None
    target_id: str | None


def recover_recorded_action(
    status: str,
    result_json: str | None,
    task_id: str | None,
) -> tuple[str, str | None, str | None] | None:
    """Recover ``(action, target_fingerprint, target_id)`` from a ticket row.

    ``result_json['action']`` is authoritative when present and valid: drop
    AND combine BOTH persist ``status='combined'``, so status alone cannot
    disambiguate them -- only the embedded action can. Falls back to
    ``status == 'created' -> 'create'`` when *result_json* carries no
    recoverable action (the real add_task-result JSON has no ``'action'``
    key). ``target_id`` prefers ``result_json['id']`` (or the less common
    ``result_json['target_id']``), falling back to the ticket row's own
    *task_id* column when neither is present -- but only for drop/combine,
    since 'create' has no "target being combined into".

    Returns ``None`` (un-actionable, e.g. ``status='failed'``/``'pending'``,
    or a missing/unparseable *result_json* that isn't rescued by the
    'created' fallback) rather than raising -- callers skip these rows.
    """
    parsed: dict | None = None
    if result_json:
        try:
            candidate = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            candidate = None
        if isinstance(candidate, dict):
            parsed = candidate

    action: str | None = None
    target_fingerprint: str | None = None
    target_id: str | None = None

    if parsed is not None:
        raw_action = parsed.get('action')
        if raw_action in _RECOVERABLE_ACTIONS:
            action = raw_action
            target_fingerprint = parsed.get('target_fingerprint')
            target_id = parsed.get('id') or parsed.get('target_id')

    if action is None and status == 'created':
        action = 'create'

    if action is None:
        return None

    if action in ('drop', 'combine') and target_id is None:
        target_id = task_id

    return (action, target_fingerprint, target_id)


def read_curator_decisions(db_path: Path) -> list[RecordedDecision]:
    """Read every recoverable curator decision from *db_path* (read-only).

    Opens the tickets.db directly via stdlib ``sqlite3`` (no writes, no
    ``fused_memory`` import) and skips rows :func:`recover_recorded_action`
    deems un-actionable. A row whose ``candidate_json`` fails to parse as a
    dict degrades to an empty candidate ``{}`` rather than dropping the
    row entirely -- the recovered action is still a meaningful signal even
    when the candidate payload itself is malformed.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT ticket_id, candidate_json, status, task_id, result_json FROM tickets'
        ).fetchall()
    finally:
        conn.close()

    decisions: list[RecordedDecision] = []
    for row in rows:
        recovered = recover_recorded_action(row['status'], row['result_json'], row['task_id'])
        if recovered is None:
            continue
        action, target_fingerprint, target_id = recovered

        try:
            candidate = json.loads(row['candidate_json'])
        except (json.JSONDecodeError, TypeError):
            candidate = None
        if not isinstance(candidate, dict):
            logger.warning(
                'read_curator_decisions: unparseable candidate_json for ticket %s', row['ticket_id'],
            )
            candidate = {}

        decisions.append(RecordedDecision(
            ticket_id=row['ticket_id'],
            candidate=candidate,
            action=action,
            target_fingerprint=target_fingerprint,
            target_id=target_id,
        ))

    return decisions
