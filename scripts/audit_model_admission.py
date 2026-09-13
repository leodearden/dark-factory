#!/usr/bin/env python3
"""Audit whether an admitted model is actually being DISPATCHED, and safely.

Answers one question in five parts, for a model M admitted to a set of roles R
at a known time: is M actually being resolved and run for R, are its runs
completing, are the per-model ceiling and the scoped account cap behaving, and
has M leaked onto a role outside R.

Written for the D6 claude-fable-5-1 admission milestone checks (tasks 5440 and
5441), but parameterized by --model / --expect-roles / --since / --window /
--ceiling so any future model admission is the same command with different
arguments, rather than a fresh set of hand-written SQL that quietly disagrees
with the last one.

STRICTLY READ-ONLY.  Every connection is a `mode=ro` SQLite URI
(:func:`_connect_ro`); this script writes nothing, files nothing, and emits no
events.  It is safe to run against a live store while the orchestrator is
merging.

WHAT THIS DOES NOT DO: it asserts no threshold.  It prints what it measures and
FLAGS observations — at-or-over-ceiling, at-or-over-wall-clock, unexpected-role
— for a human narrative to interpret.  The close-or-escalate judgement belongs
to the reader and to the milestone task's own conditional, not to a pass/fail
hardcoded inside a read-only audit.  Each rendered section is labelled with the
concrete resolved window and target model it was computed over, so a report that
pastes the output cannot drift from the query that produced it.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The orchestrator's own data/ is gitignored (.gitignore:9 "/data/"), so a task
# worktree has no store of its own and a worktree-relative default would never
# resolve.  The live store exists only in the main checkout.
DEFAULT_RUNS_DB = Path('/home/leo/src/dark-factory/data/orchestrator/runs.db')

# orchestrator/src/orchestrator/routing.py::_model_rejection_reason returns
# exactly these three; resolve_route namespaces each as "<layer>:<reason>"
# before appending it to the decision's `rejected` list.  Kept here as the one
# place the vocabulary is written down.
#
# NOT included, deliberately: 'model-not-in-ladder'.  resolve_route appends that
# one when a '+N' ladder-relative spec cannot be resolved, BEFORE any candidate
# model is formed or validated — so it is not evidence that a model was
# attempted and refused, which is the only thing this audit reads rejections for.
MODEL_REJECTION_REASONS = frozenset({
    'model-not-in-allowlist',
    'model-ceiling-exhausted',
    'model-capacity-exhausted',
})


def _connect_ro(path: str | Path) -> sqlite3.Connection:
    """Open *path* strictly read-only, via a `mode=ro` SQLite URI.

    `mode=ro` (rather than a bare `sqlite3.connect`) is the house convention for
    a script that reads a LIVE store — see `scripts/merge_lane_throughput.py`,
    `scripts/audit_wiped_metadata_files.py` and `scripts/census_tagger_debris.py`.
    It buys two things a bare connect does not: a write attempted through this
    connection fails loudly instead of mutating an orchestrator's event store,
    and a typo'd path raises rather than silently creating an empty database
    that would then report every section as "no data".
    """
    uri = f'file:{Path(path).resolve()}?mode=ro'
    return sqlite3.connect(uri, uri=True)


def _iso(moment: datetime) -> str:
    """Format *moment* in the exact ISO-8601 spelling the store writes.

    `event_store.py::EventStore.emit` and the cost store both write
    ``datetime.now(UTC).isoformat()``, which renders the offset as ``+00:00``
    (never ``Z``).  Those columns are TEXT and the spelling is fixed, so the
    result can be used directly as a SQL comparand and the comparison is a
    correct chronological one.
    """
    return moment.astimezone(UTC).isoformat()


def _loads_object(raw: Any) -> dict[str, Any] | None:
    """Parse *raw* as a JSON object, or return None if it is not one.

    Tolerant by design: the live store holds at least one ``account_events.
    details`` that is a bare non-JSON string (``'Escalation watcher (auto)'``),
    and a 181 MB store must not be abortable by one malformed row.  Every
    caller counts what it skips, so a dropped row stays visible in the output.
    """
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_model_rejection(entry: Any) -> bool:
    """True when *entry* is a ``"<layer>:<reason>"`` naming a model rejection.

    CAVEAT, and it decides the shape of this whole check: the producer's
    rejection string names the LAYER and the REASON but never the candidate
    MODEL (routing.py::resolve_route appends ``f'config:{reason}'`` and
    friends).  So a rejection cannot be attributed to a particular model from
    the event payload alone.  This audit therefore reports every model
    rejection on every role and leaves attribution to the reader — deliberately
    over-inclusive, because the failure that matters is missing one.
    """
    if not isinstance(entry, str):
        return False
    _, _, reason = entry.rpartition(':')
    return reason in MODEL_REJECTION_REASONS


@dataclass(frozen=True)
class RoutingSelection:
    """One decision that resolved to the target model."""

    timestamp: str
    task_id: str | None
    role: str
    source_layer: str
    rule_id: str | None
    routing_tier: int | None


@dataclass(frozen=True)
class RoutingRejection:
    """One decision that recorded a model rejection, and what it resolved to instead."""

    timestamp: str
    task_id: str | None
    role: str
    resolved_model: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RoutingScan:
    """What the resolver DECIDED since a given moment (task checks 1 and 3)."""

    selections: tuple[RoutingSelection, ...]
    rejections: tuple[RoutingRejection, ...]
    skipped_rows: int


def scan_routing_decisions(
    conn: sqlite3.Connection, *, model: str, since: datetime
) -> RoutingScan:
    """Scan `routing_decision` events at or after *since*, once, for two answers.

    SELECTIONS are the decisions that resolved to *model*, each carrying the
    ``source_layer``/``rule_id``/``routing_tier`` that say WHY — which is the
    only way to answer "did rule <X> actually match?".

    REJECTIONS span ALL roles, not just the roles that selected *model*: a
    rejection is recorded on a decision that then resolved to a DIFFERENT
    model, so filtering to selections would systematically miss every rejection
    there is.  See :func:`_is_model_rejection` for why they cannot be filtered
    to *model* itself.
    """
    cursor = conn.execute(
        'SELECT timestamp, task_id, data FROM events '
        'WHERE event_type = ? AND timestamp >= ? ORDER BY timestamp, id',
        ('routing_decision', _iso(since)),
    )
    selections: list[RoutingSelection] = []
    rejections: list[RoutingRejection] = []
    skipped = 0
    for timestamp, task_id, raw in cursor:
        payload = _loads_object(raw)
        if payload is None:
            skipped += 1
            continue
        role = payload.get('role') or ''
        resolved = payload.get('model') or ''
        if resolved == model:
            selections.append(RoutingSelection(
                timestamp=timestamp,
                task_id=task_id,
                role=role,
                source_layer=payload.get('source_layer') or '',
                rule_id=payload.get('rule_id'),
                routing_tier=payload.get('routing_tier'),
            ))
        rejected = payload.get('rejected')
        reasons = tuple(
            entry for entry in rejected if _is_model_rejection(entry)
        ) if isinstance(rejected, list) else ()
        if reasons:
            rejections.append(RoutingRejection(
                timestamp=timestamp,
                task_id=task_id,
                role=role,
                resolved_model=resolved,
                reasons=reasons,
            ))
    return RoutingScan(
        selections=tuple(selections),
        rejections=tuple(rejections),
        skipped_rows=skipped,
    )
