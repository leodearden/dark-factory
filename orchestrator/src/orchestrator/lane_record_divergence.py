"""Digest-time cross-check of the warm-lane pool's in-memory assignment map
against the durable ``.lane-state`` records (PRD
``plans/warm-lane-exhaustion-hardening-prd.md`` δ). The durable record is
canonical and the map is its cache, so every disagreement is a defect signal.
This module counts each lane's divergence streak across digests and alarms on
a persistent one; it acts on no lane and never repairs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from escalation.categories import EscalationCategory
from pydantic import AwareDatetime, PositiveInt, TypeAdapter
from shared import safe_io

from orchestrator.born_at_l2 import BornAtL2Alarm, file_deduped_born_at_l2
from orchestrator.lane_lifecycle import LaneRecord, LaneState

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

    from orchestrator.event_store import EventStore

logger = logging.getLogger(__name__)

LANE_RECORD_DIVERGENCE: Final = 'lane_record_divergence'
PERSISTENT_DIVERGENCE_DIGESTS: Final = 2
PERSISTENT_DIVERGENCE_MIN_AGE: Final = timedelta(hours=1)
STREAKS_FILENAME: Final = '.lane-record-divergence-streaks.json'

_HOLDING_STATES: Final = frozenset({LaneState.ASSIGNED, LaneState.IN_USE})

_ALARM: Final = BornAtL2Alarm(
    root_cause=LANE_RECORD_DIVERGENCE,
    sentinel_task='__lane_record_divergence__',
    role='orchestrator-lane-record-divergence',
    category=EscalationCategory.INFRA_ISSUE,
    suggested_action='investigate_lane_record_divergence',
    created_reason='lane-record-divergence-persistent',
)
_ALARM_GUIDANCE: Final = (
    'This is a read-only alarm: the orchestrator repairs nothing. The durable '
    '.lane-state record is canonical (PRD W2b) and the in-memory assignment map '
    'is its cache. Corroborate each lane\'s .lane-state record against the live '
    'pool before acting; the digest section lists every divergent lane and '
    'shows _none_ once they have all cleared.'
)


@dataclass(frozen=True)
class DivergenceStreak:
    """How long one lane has diverged: digests in a row, first and latest."""

    consecutive_digests: PositiveInt
    first_seen: AwareDatetime
    last_seen: AwareDatetime

    @classmethod
    def starting(cls, now: datetime) -> DivergenceStreak:
        return cls(consecutive_digests=1, first_seen=now, last_seen=now)

    def continued(self, now: datetime) -> DivergenceStreak:
        return replace(
            self, consecutive_digests=self.consecutive_digests + 1, last_seen=now,
        )

    @property
    def is_persistent(self) -> bool:
        # The age floor rides out GitOps's lane re-seed and steal-retry, which
        # hold map and record apart across awaits while an escalation burst
        # can fire two digests.
        return (
            self.consecutive_digests >= PERSISTENT_DIVERGENCE_DIGESTS
            and self.last_seen - self.first_seen >= PERSISTENT_DIVERGENCE_MIN_AGE
        )


@dataclass(frozen=True)
class LaneDivergence:
    """One lane whose map entries and durable record disagree."""

    lane: str
    map_task_ids: tuple[str, ...]
    record_state: LaneState | None
    record_task_id: str | None
    streak: DivergenceStreak

    def render(self) -> str:
        return (
            f'{self.lane}: map {self._map_view()}, record {self._record_view()} '
            f'({self._streak_view()})'
        )

    def _map_view(self) -> str:
        if not self.map_task_ids:
            return 'unassigned'
        noun = 'task' if len(self.map_task_ids) == 1 else 'tasks'
        return f'{noun} {", ".join(self.map_task_ids)}'

    def _record_view(self) -> str:
        if self.record_state is None:
            return 'no readable record'
        if self.record_task_id is None:
            return self.record_state.value
        return f'{self.record_state.value} task {self.record_task_id}'

    def _streak_view(self) -> str:
        count = self.streak.consecutive_digests
        plural = '' if count == 1 else 's'
        since = self.streak.first_seen.isoformat(timespec='minutes')
        return f'{count} consecutive digest{plural} since {since}'


def cross_check_lane_records(
    assignments: Mapping[str, Path],
    records: Mapping[str, LaneRecord],
    previous_streaks: Mapping[str, DivergenceStreak] | None = None,
    *,
    now: datetime,
) -> tuple[LaneDivergence, ...]:
    """Every lane where the map's tasks differ from the record's holding task.

    *assignments* maps task id to lane path; *records* maps lane name to its
    readable durable record. A record holds a lane only in an ASSIGNED or
    IN_USE state, and a lane that several map tasks point at always diverges.
    """
    streaks = previous_streaks or {}
    map_view = _task_ids_by_lane(assignments)
    held = {
        lane: (record.task_id,)
        for lane, record in records.items()
        if record.state in _HOLDING_STATES and record.task_id is not None
    }
    divergences = []
    for lane in sorted(map_view.keys() | held.keys()):
        map_task_ids = map_view.get(lane, ())
        if map_task_ids == held.get(lane, ()):
            continue
        record = records.get(lane)
        previous = streaks.get(lane)
        divergences.append(LaneDivergence(
            lane=lane,
            map_task_ids=map_task_ids,
            record_state=None if record is None else record.state,
            record_task_id=None if record is None else record.task_id,
            streak=DivergenceStreak.starting(now) if previous is None else previous.continued(now),
        ))
    return tuple(divergences)


def _task_ids_by_lane(assignments: Mapping[str, Path]) -> dict[str, tuple[str, ...]]:
    by_lane: dict[str, list[str]] = {}
    for task_id, lane in assignments.items():
        by_lane.setdefault(lane.name, []).append(task_id)
    return {lane: tuple(sorted(task_ids)) for lane, task_ids in by_lane.items()}


_STREAKS: Final = TypeAdapter(dict[str, DivergenceStreak])


def advance_lane_record_divergence(
    assignments: Mapping[str, Path],
    records: Mapping[str, LaneRecord],
    *,
    streaks_path: Path,
    now: datetime,
) -> tuple[LaneDivergence, ...]:
    """Cross-check one digest, continuing and then saving the per-lane streaks."""
    divergences = cross_check_lane_records(
        assignments, records, _load_streaks(streaks_path), now=now,
    )
    _save_streaks(streaks_path, divergences)
    return divergences


def _load_streaks(path: Path) -> dict[str, DivergenceStreak]:
    try:
        return _STREAKS.validate_json(path.read_bytes(), strict=True)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning(
            'lane_record_divergence: ignoring streaks file %s, counting afresh: %s',
            path, exc,
        )
        return {}


def _save_streaks(path: Path, divergences: Sequence[LaneDivergence]) -> None:
    streaks = {d.lane: d.streak for d in divergences}
    try:
        safe_io.atomic_write_text(path, _STREAKS.dump_json(streaks).decode(), mkdir=True)
    except OSError:
        logger.warning(
            'lane_record_divergence: could not save streaks to %s', path,
            exc_info=True,
        )


def file_persistent_divergence_alarm(
    escalation_queue: EscalationQueue | None,
    divergences: Sequence[LaneDivergence],
    *,
    event_store: EventStore | None,
    filing_claimant_run_id: str | None,
) -> Escalation | None:
    """File one deduped born-at-L2 once any lane diverges persistently.

    Fail-open: any filing error is logged and ``None`` returned, so the
    digest never breaks on it.
    """
    persistent = [d for d in divergences if d.streak.is_persistent]
    if not persistent:
        return None
    lanes = ', '.join(d.lane for d in persistent)
    min_age_minutes = int(PERSISTENT_DIVERGENCE_MIN_AGE.total_seconds() // 60)
    detail = '\n'.join(
        [
            f'Lanes diverged for at least {PERSISTENT_DIVERGENCE_DIGESTS} consecutive '
            f'digests over at least {min_age_minutes} minutes:',
        ]
        + [f'- {d.render()}' for d in persistent]
        + ['', _ALARM_GUIDANCE]
    )
    return file_deduped_born_at_l2(
        escalation_queue, _ALARM,
        summary=(
            f'warm-lane map/record divergence persisted on {len(persistent)} '
            f'lane(s): {lanes}'
        ),
        detail=detail,
        event_store=event_store,
        filing_claimant_run_id=filing_claimant_run_id,
    )
