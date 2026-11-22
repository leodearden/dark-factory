"""Restart-time re-test of the project_root predicate behind a preserved merge-halt record.

A preserved record re-asserts its merge-queue halt only while project_root still
has tracked uncommitted WIP or unmerged entries, or while that cannot be
measured; a record measured clean is stamped released and never re-owns a halt
(``plans/scheduler-pause-halt-retirement-prd.md`` leaf η, task 3876).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from escalation.models import Escalation

logger = logging.getLogger(__name__)

__all__ = [
    'HALT_NOT_REHYDRATED',
    'HaltRecords',
    'HaltSink',
    'HygieneProbe',
    'ProjectRootReading',
    'read_project_root',
    'rehydrate_merge_halt',
]

HALT_NOT_REHYDRATED = 'Merge halt NOT rehydrated'
_PATHS_SHOWN = 3


class HygieneProbe(Protocol):
    @property
    def project_root(self) -> Path: ...

    async def unmerged_paths(self, cwd: Path) -> tuple[str, ...]: ...

    async def has_dirty_working_tree(self) -> str: ...


class HaltSink(Protocol):
    def halt_for_wip(self, reason: str) -> None: ...

    def set_halt_owner(self, esc_id: str) -> None: ...


class HaltRecords(Protocol):
    def release_merge_halt(self, escalation_id: str, *, note: str) -> Escalation | None: ...


def _shown(paths: tuple[str, ...]) -> str:
    if not paths:
        return ''
    more = ', …' if len(paths) > _PATHS_SHOWN else ''
    return f' ({", ".join(paths[:_PATHS_SHOWN])}{more})'


@dataclass(frozen=True)
class ProjectRootReading:
    path: str
    dirty: tuple[str, ...] = ()
    unmerged: tuple[str, ...] = ()
    error: str | None = None

    @property
    def holds(self) -> bool:
        return self.error is not None or bool(self.dirty or self.unmerged)

    def describe(self) -> str:
        if self.error is not None:
            return f'project_root {self.path} could not be measured: {self.error}'
        return (
            f'project_root {self.path} has {len(self.unmerged)} unmerged path(s)'
            f'{_shown(self.unmerged)} and {len(self.dirty)} tracked dirty path(s)'
            f'{_shown(self.dirty)}'
        )


async def read_project_root(probe: HygieneProbe) -> ProjectRootReading:
    """Measure project_root's hygiene; a failed measurement is recorded, never raised."""
    path = str(probe.project_root)
    try:
        unmerged = await probe.unmerged_paths(probe.project_root)
        dirty = await probe.has_dirty_working_tree()
    except (RuntimeError, OSError) as exc:
        return ProjectRootReading(path=path, error=f'{type(exc).__name__}: {exc}')
    return ProjectRootReading(
        path=path,
        dirty=tuple(line for line in dirty.splitlines() if line),
        unmerged=unmerged,
    )


async def rehydrate_merge_halt(
    candidates: Sequence[Escalation],
    *,
    probe: HygieneProbe,
    worker: HaltSink,
    records: HaltRecords,
) -> str | None:
    """Re-assert the halt owned by *candidates* only while project_root's predicate holds.

    Returns the id of the record that now owns the halt, or ``None`` when none does.
    """
    unreleased = [esc for esc in candidates if not _already_released(esc)]
    if not unreleased:
        return None
    reading = await read_project_root(probe)
    if reading.holds:
        return _reassert_halt(unreleased, reading, worker)
    _release_halts(unreleased, reading, records)
    return None


def _already_released(esc: Escalation) -> bool:
    if esc.halt_released_at is None:
        return False
    logger.info(
        'Merge halt not rehydrated from %s: released at %s (%s)',
        esc.id, esc.halt_released_at, esc.halt_release_note,
    )
    return True


def _reassert_halt(
    unreleased: Sequence[Escalation], reading: ProjectRootReading, worker: HaltSink,
) -> str:
    if len(unreleased) > 1:
        logger.warning(
            '_rehydrate_merge_halt: %d qualifying level-≥1 record(s) '
            'found; registering only the most recent as halt owner.  The '
            'merge queue will resume as soon as that owner is resolved — '
            'even though %d older record(s) remain pending.  Resolve the '
            'most-recent one last to avoid premature queue resumption.',
            len(unreleased),
            len(unreleased) - 1,
        )
    esc = max(unreleased, key=lambda e: datetime.fromisoformat(e.timestamp))
    reason = (
        f'Rehydrated merge halt from preserved L{esc.level} {esc.id} '
        f'(category={esc.category}) after restart: {reading.describe()}'
    )
    worker.halt_for_wip(reason)
    worker.set_halt_owner(esc.id)
    logger.warning('%s', reason)
    return esc.id


def _release_halts(
    unreleased: Sequence[Escalation], reading: ProjectRootReading, records: HaltRecords,
) -> None:
    for esc in unreleased:
        note = (
            f'{HALT_NOT_REHYDRATED}: preserved L{esc.level} {esc.id} '
            f'(category={esc.category}) — project_root predicate re-tested clean: '
            f'{reading.describe()}'
        )
        logger.warning('%s', note)
        try:
            records.release_merge_halt(esc.id, note=note)
        except OSError:
            logger.warning('Could not stamp the merge-halt release on %s', esc.id, exc_info=True)
