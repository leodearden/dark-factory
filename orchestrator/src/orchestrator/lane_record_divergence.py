"""Read-only digest-time cross-check of the warm-lane pool's in-memory
assignment map against the durable ``.lane-state`` records (PRD
``plans/warm-lane-exhaustion-hardening-prd.md`` δ). The durable record is
canonical and the map is its cache, so every disagreement is a defect signal;
this module reports it and never repairs it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from shared import safe_io

from orchestrator.lane_lifecycle import LaneRecord, LaneState

logger = logging.getLogger(__name__)

LANE_RECORD_DIVERGENCE: Final = 'lane_record_divergence'
PERSISTENT_DIVERGENCE_DIGESTS: Final = 2
STREAKS_FILENAME: Final = '.lane-record-divergence-streaks.json'

_HOLDING_STATES: Final = frozenset({LaneState.ASSIGNED, LaneState.IN_USE})


@dataclass(frozen=True)
class LaneDivergence:
    """One lane whose map entry and durable record disagree."""

    lane: str
    map_task_id: str | None
    record_state: LaneState | None
    record_task_id: str | None
    consecutive_digests: int = 1

    @property
    def is_persistent(self) -> bool:
        return self.consecutive_digests >= PERSISTENT_DIVERGENCE_DIGESTS

    def render(self) -> str:
        map_view = 'unassigned' if self.map_task_id is None else f'task {self.map_task_id}'
        if self.record_state is None:
            record_view = 'no record'
        elif self.record_task_id is None:
            record_view = self.record_state.value
        else:
            record_view = f'{self.record_state.value} task {self.record_task_id}'
        plural = '' if self.consecutive_digests == 1 else 's'
        return (
            f'{self.lane}: map {map_view}, record {record_view} '
            f'({self.consecutive_digests} consecutive digest{plural})'
        )


def cross_check_lane_records(
    assignments: Mapping[str, Path],
    records: Mapping[str, LaneRecord],
    previous_streaks: Mapping[str, int] | None = None,
) -> tuple[LaneDivergence, ...]:
    """Every lane where the map's task differs from the record's holding task.

    *assignments* maps task id to lane path; *records* maps lane name to its
    durable record. A record holds a lane only in an ASSIGNED or IN_USE state.
    """
    streaks = previous_streaks or {}
    map_view = {lane.name: task_id for task_id, lane in assignments.items()}
    held = {
        lane: record.task_id
        for lane, record in records.items()
        if record.state in _HOLDING_STATES and record.task_id is not None
    }
    divergences = []
    for lane in sorted(map_view.keys() | held.keys()):
        if map_view.get(lane) == held.get(lane):
            continue
        record = records.get(lane)
        divergences.append(LaneDivergence(
            lane=lane,
            map_task_id=map_view.get(lane),
            record_state=None if record is None else record.state,
            record_task_id=None if record is None else record.task_id,
            consecutive_digests=streaks.get(lane, 0) + 1,
        ))
    return tuple(divergences)


def advance_lane_record_divergence(
    assignments: Mapping[str, Path],
    records: Mapping[str, LaneRecord],
    *,
    streaks_path: Path,
) -> tuple[LaneDivergence, ...]:
    """Cross-check one digest, continuing and then saving the per-lane streaks."""
    divergences = cross_check_lane_records(
        assignments, records, _load_streaks(streaks_path),
    )
    _save_streaks(streaks_path, divergences)
    return divergences


def _load_streaks(path: Path) -> dict[str, int]:
    try:
        return _parse_streaks(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning(
            'lane_record_divergence: ignoring streaks file %s, counting afresh: %s',
            path, exc,
        )
        return {}


def _parse_streaks(text: str) -> dict[str, int]:
    data = json.loads(text)
    if not isinstance(data, dict) or not all(
        _is_streak(lane, count) for lane, count in data.items()
    ):
        raise ValueError(f'not a lane -> streak map: {text[:80]!r}')
    return data


def _is_streak(lane: object, count: object) -> bool:
    return isinstance(lane, str) and type(count) is int and count >= 1


def _save_streaks(path: Path, divergences: Sequence[LaneDivergence]) -> None:
    streaks = {d.lane: d.consecutive_digests for d in divergences}
    try:
        safe_io.atomic_write_text(path, json.dumps(streaks, sort_keys=True), mkdir=True)
    except OSError:
        logger.warning(
            'lane_record_divergence: could not save streaks to %s', path,
            exc_info=True,
        )
