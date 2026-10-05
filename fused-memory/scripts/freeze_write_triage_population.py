#!/usr/bin/env python3
"""Freeze the fresh write-triage population with its production slates.

Task 6151 (π), plans/write-triage-flip-readiness-prd.md §11 D14: every Mem0
write in the three Mem0-primary categories of each population project, created
on or after :data:`POPULATION_SINCE` and no later than the moment the freeze
starts, paired with the slate and band production retrieval gives it from the
live store. The snapshot is the input of the judge arms
(``run_write_triage_population_arms.py``) and is written under a gitignored
``--out-root``.

READ-ONLY against the store: it scrolls, searches and reads points, and
writes nothing but the snapshot files.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, TypedDict

from fused_memory.models.enums import MEM0_PRIMARY
from fused_memory.models.scope import Scope
from fused_memory.reconciliation.prompts import (
    FLAG_FOR_STAGE2_MARKER_KIND,
    STAGE2_SUPPRESS_GUARD_KIND,
)
from fused_memory.server.write_triage import declares_attach_keys

POPULATION_SINCE = datetime(2026, 9, 29, tzinfo=UTC)
POPULATION_PROJECTS = ('dark_factory', 'reify')
POPULATION_CATEGORIES = tuple(sorted(category.value for category in MEM0_PRIMARY))

#: The boolean flags the reconciliation stage prompts (reconciliation/prompts/)
#: wrote on their marker records before χ gave those markers a declared kind.
_RECON_MARKER_FLAGS = ('flag_for_stage2', 'stage2_suppress')
_RECON_MARKER_KINDS = frozenset({FLAG_FOR_STAGE2_MARKER_KIND, STAGE2_SUPPRESS_GUARD_KIND})


class PopulationWrite(TypedDict):
    """One frozen write, before its slate is attached."""

    memory_id: str
    project_id: str
    category: str
    created_at: str
    content: str
    metadata: dict[str, Any]
    recon_marker: bool
    declares_attach_keys: bool


def is_recon_marker(metadata: Mapping[str, Any]) -> bool:
    """Whether *metadata* is a reconciliation marker, by its flag or its declared kind."""
    return (
        any(metadata.get(flag) is True for flag in _RECON_MARKER_FLAGS)
        or metadata.get('kind') in _RECON_MARKER_KINDS
    )


def _parse_created_at(value: object) -> datetime | None:
    """*value* as an aware instant, or ``None`` when it is naive, absent or unparseable."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _population_write(
    record: Mapping[str, Any], project_id: str, category: str, stored: Mapping[str, Any],
) -> PopulationWrite:
    metadata = dict(stored.get('metadata') or {})
    return PopulationWrite(
        memory_id=str(record['id']),
        project_id=project_id,
        category=category,
        created_at=record['created_at'],
        content=str(stored.get('content') or ''),
        metadata=metadata,
        recon_marker=is_recon_marker(metadata),
        declares_attach_keys=declares_attach_keys(metadata),
    )


async def enumerate_population(
    memory_service: Any,
    *,
    projects: Sequence[str],
    since: datetime,
    frozen_at: datetime,
) -> tuple[list[PopulationWrite], dict[str, int]]:
    """Every write of *projects* created in ``[since, frozen_at]``, and what was excluded.

    Each project's Mem0-primary categories are scrolled exhaustively; content
    and metadata come from the public point read. A record whose
    ``created_at`` is not an aware timestamp is ``undated`` and one gone
    before its read is ``vanished``: both are excluded and counted, never
    guessed. Ordered by project, then instant, then id.
    """
    excluded = {'undated': 0, 'vanished': 0}
    keyed: list[tuple[tuple[str, datetime, str], PopulationWrite]] = []
    for project_id in projects:
        scope = Scope(project_id=project_id)
        for category in POPULATION_CATEGORIES:
            records = memory_service.mem0.scroll_all_by_metadata(scope, {'category': category})
            async for record in records:
                instant = _parse_created_at(record.get('created_at'))
                if instant is None:
                    excluded['undated'] += 1
                    continue
                if not since <= instant <= frozen_at:
                    continue
                stored = await memory_service.get_memory_by_id(project_id, str(record['id']))
                if stored is None:
                    excluded['vanished'] += 1
                    continue
                write = _population_write(record, project_id, category, stored)
                keyed.append(((project_id, instant, write['memory_id']), write))
    keyed.sort(key=lambda pair: pair[0])
    return [write for _, write in keyed], excluded
