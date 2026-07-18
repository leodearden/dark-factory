"""verify_checkpoint — the durable verified-green checkpoint predicate (task 2752).

``_execute_verify_review_loop`` re-enters VERIFY on every fresh dispatch, so a
long pipeline replay can be cancelled by an 8h fleet redeploy before it ever
reaches the merge queue — even when the branch is already green and its tip
has not moved since the last green verify. This module answers the single
question that lets the loop skip that redundant re-verify:

    "Has this task's branch already been verified green at its CURRENT tip,
     in this OR a prior orchestrator run?"

``green_checkpoint_at_tip(event_store, event_type, task_id, tip_sha)`` returns
True iff a durable ``workflow_verify`` row for *task_id* has BOTH a truthy
``data['passed']`` AND ``data['tip_sha'] == tip_sha``. The row is read via
``event_store.fetch_events_by_type_all_runs`` — the run-agnostic reader — so a
green recorded before a restart still counts (that cross-run durability is the
whole point; the run-scoped ``fetch_events_by_type`` cannot see it).

Keying on the branch TIP SHA alone (not also base_sha) is deliberate and safe:
a rebase is the only operation that moves the branch base, and it rewrites
every branch commit → a new tip SHA. So an unchanged tip already implies an
unchanged base. An exact tip match to a prior-recorded green cannot manufacture
a false green; a tip change simply misses the checkpoint and a normal verify
runs. base_sha is carried in the payload for telemetry only, never matched here.

Strict fail-closed, mirroring ``merge_completion.merge_completion_eligible``:
a None-or-empty ``event_store`` / ``task_id`` / ``tip_sha``, no matching row,
or ANY read error returns False and NEVER raises — the safe direction is always
"run the verify", never "skip it on thin evidence". An EMPTY ``tip_sha`` is
treated as absent (not a valid key): ``_get_head_commit`` returns ``''`` when
``git rev-parse`` fails without raising, and matching ``'' == ''`` against a
green whose tip was likewise recorded empty would manufacture a false skip — so
an empty tip fails closed exactly like None. ``event_store`` is duck-typed
(only ``fetch_events_by_type_all_runs`` is used), and this module imports
nothing beyond ``EventType`` + stdlib logging/typing, keeping it
dependency-light like ``orchestrator.merge_completion`` / ``orchestrator.b3_gate``.
"""

from __future__ import annotations

import logging
from typing import Any

from orchestrator.event_store import EventType

logger = logging.getLogger(__name__)


def green_checkpoint_at_tip(
    event_store: Any,
    event_type: EventType,
    task_id: str | None,
    tip_sha: str | None,
) -> bool:
    """Return True iff a durable *event_type* green exists for *task_id* at *tip_sha*.

    A "green at tip" is a row whose ``data['passed']`` is truthy AND whose
    ``data['tip_sha']`` equals *tip_sha*. Rows are read run-agnostically via
    ``event_store.fetch_events_by_type_all_runs(event_type, task_id=task_id)``,
    so a green recorded in a PRIOR run (pre-restart) still counts — this is the
    cross-restart durability the checkpoint relies on.

    Strict fail-closed: returns False when *event_store* or *task_id* is None,
    when *tip_sha* is None OR empty (an empty tip is a failed/absent
    ``git rev-parse``, never a valid match key), and on any read error (never
    raises) — see the module docstring for the tip-only-key rationale and the
    fail-closed contract.
    *event_store* is duck-typed: only
    ``fetch_events_by_type_all_runs(event_type, *, task_id=...) -> list[dict]``
    is called, matching ``EventStore.fetch_events_by_type_all_runs``'s shape
    (each row a dict with at least 'task_id' and a JSON-parsed 'data').
    """
    if event_store is None or task_id is None or not tip_sha:
        return False
    try:
        rows = event_store.fetch_events_by_type_all_runs(event_type, task_id=task_id)
        return any(
            bool((row.get('data') or {}).get('passed'))
            and (row.get('data') or {}).get('tip_sha') == tip_sha
            for row in rows
        )
    except Exception:
        logger.warning(
            'green_checkpoint_at_tip: event-store read failed for task_id=%s '
            'tip_sha=%s; degrading to False (evidence absent, fail-safe)',
            task_id,
            tip_sha,
            exc_info=True,
        )
        return False
