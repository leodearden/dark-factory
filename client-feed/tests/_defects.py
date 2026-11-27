"""Client defect fixtures in their producers' shape: the inbox lines a portal writes, the task metadata and gate edges the filer writes, and the carrier the publisher reads back."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from _stores import TasksDb

from client_feed.contract import (
    DefectReported,
    DefectRuled,
    Quarantined,
    QuarantineReason,
    Restated,
    Restater,
    RulingKind,
)
from client_feed.defect_ref import parse_portal_ref
from client_feed.sources.tasks import TASK_TAG, DefectCarrier


def restated(observed: str = 'o', expected: str = 'e', steps: tuple[str, ...] = ('s',)) -> Restated:
    return Restated(observed=observed, expected=expected, steps=steps)


def quarantined(*reasons: QuarantineReason) -> Quarantined:
    return Quarantined(reasons=reasons)


def reported(
    block: int,
    report: int,
    *,
    at: datetime,
    seq: int = 1,
    criterion_id: str = 'C-1',
    outcome: Restated | Quarantined | None = None,
) -> DefectReported:
    """A defect_reported inbox line; restated as restated() writes it unless `outcome` says otherwise."""
    return DefectReported(
        seq=seq,
        at=at,
        block=block,
        report=report,
        criterion_id=criterion_id,
        restater=Restater(prompt_version='restate-v1'),
        outcome=restated() if outcome is None else outcome,
    )


def ruled(
    block: int, report: int, *, at: datetime, seq: int = 2, kind: RulingKind = RulingKind.NOT_A_DEFECT
) -> DefectRuled:
    return DefectRuled(seq=seq, at=at, block=block, report=report, kind=kind)


def declared(ref: object, *, origin: str = 'client') -> dict[str, object]:
    """One metadata.defects entry as declare_defect writes it; `ref` is any value, so a malformed one can be stored."""
    return {
        'ref': ref,
        'origin': origin,
        'summary': 'SECRET-SUMMARY',
        'declared_at': '2026-10-21T11:00:00+00:00',
        'declared_by': 'client-defect-filer',
        'status': 'open',
        'sightings': 1,
    }


def carrying(*refs: str) -> dict[str, object]:
    """The metadata of a task holding a client defect entry for each of `refs`."""
    return {'defects': [declared(ref) for ref in refs]}


def gate_for(ref: str) -> dict[str, object]:
    """The metadata of the human gate the filer files for `ref`."""
    return {'source': 'client-defect-filer', 'x_client_defect_gate_for': ref}


def add_dependency(tasks: TasksDb, task_id: int, depends_on: int, *, tag: str = TASK_TAG) -> None:
    """An edge inserted as fused-memory's same-project add_dependency inserts it: the task's updated_at untouched."""
    conn = sqlite3.connect(tasks.path)
    try:
        conn.execute(
            'INSERT INTO dependencies (tag, task_id, depends_on) VALUES (?, ?, ?)',
            (tag, task_id, depends_on),
        )
        conn.commit()
    finally:
        conn.close()


def carrier_for(
    ref: str,
    *,
    project_id: str,
    task_id: int,
    gated: bool = False,
    waits_on_deferred_gate: bool = False,
) -> DefectCarrier:
    """The carrier of `ref`, its block and report read back through the one ref rule."""
    named = parse_portal_ref(ref, project_id=project_id)
    assert named is not None, f'{ref} is no portal ref of {project_id}'
    return DefectCarrier(
        ref=ref,
        block=named.block,
        report=named.report,
        task_id=task_id,
        gated=gated,
        waits_on_deferred_gate=waits_on_deferred_gate,
    )
