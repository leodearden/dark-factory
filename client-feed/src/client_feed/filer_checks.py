"""The publisher's two filer checks (plans/client-feed-prd.md §14.2 decisions 8 and 14, and the §12 amendment of 2026-11-26 from task 11885): reports read on the inbox wait in cursor.json until a carrier exists, filed reports whose unfinished carrier waits on a deferred gate are held from the pass that first saw them so, and the heartbeat names `filer` for each kind once one has waited past the grace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from datetime import datetime, timedelta

from client_feed.contract import DefectReported, DefectRuled, Degradation, InboxLine, Source
from client_feed.defect_ref import portal_ref


def still_unfiled(
    held: Mapping[str, datetime],
    lines: Iterable[InboxLine],
    *,
    project_id: str,
    filed: AbstractSet[str] | None,
) -> dict[str, datetime]:
    """The reports still waiting for a carrier, each since it was first read.

    `held` gains each newly reported ref and loses each ruled one, since a ruled report is never filed.
    Each ref in `filed` is then dropped; with `filed` None, the carriers could not be read and nothing is.
    """
    pending = dict(held)
    for line in lines:
        if isinstance(line, DefectReported):
            pending.setdefault(portal_ref(project_id, line.block, line.report), line.at)
        elif isinstance(line, DefectRuled):
            pending.pop(portal_ref(project_id, line.block, line.report), None)
    if filed is None:
        return pending
    return {ref: since for ref, since in pending.items() if ref not in filed}


def still_behind_deferred_gates(
    held: Mapping[str, datetime], waiting: AbstractSet[str] | None, *, now: datetime
) -> dict[str, datetime]:
    """The refs whose carrier waits on a deferred gate, each since the first pass that saw it so.

    With `waiting` None, the carriers could not be read and `held` stands as it is.
    """
    if waiting is None:
        return dict(held)
    return {ref: held.get(ref, now) for ref in waiting}


def _overdue(
    waiting: Mapping[str, datetime], *, state: str, now: datetime, grace_seconds: int
) -> Degradation | None:
    """`filer`, naming oldest first every report `state` strictly longer than the grace; None while none is."""
    grace = timedelta(seconds=grace_seconds)
    overdue = sorted((since, ref) for ref, since in waiting.items() if now - since > grace)
    if not overdue:
        return None
    refs = ', '.join(ref for _, ref in overdue)
    return Degradation(
        source=Source.FILER,
        reason=f'{len(overdue)} defect report(s) {state} past {grace_seconds}s: {refs}',
    )


def unfiled_degradation(
    pending: Mapping[str, datetime], *, now: datetime, grace_seconds: int
) -> Degradation | None:
    return _overdue(pending, state='unfiled', now=now, grace_seconds=grace_seconds)


def deferred_gate_degradation(
    held: Mapping[str, datetime], *, now: datetime, grace_seconds: int
) -> Degradation | None:
    return _overdue(
        held, state='waiting on a deferred gate', now=now, grace_seconds=grace_seconds
    )
