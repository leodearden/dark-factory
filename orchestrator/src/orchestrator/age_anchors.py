"""One age anchor per task in the scheduler's active fetch.

An anchor is the max task id at which a task's wait began, so its legacy
dispatch age is the count of tasks filed since (the I-6 unarmed-tick fallback,
task 3819).  Anchors are restart-durable (task 5352), and only a changed
``pending_since`` resets one: D3 in
``plans/scheduler-dispatch-scoring-and-lock-layer-prd.md`` is the rule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from orchestrator.guard_state import PersistentMap, guard_path
from orchestrator.task_status import TERMINAL_STATUSES

AGE_ANCHORS_FILE = 'scheduler_age_anchors.json'

# How long a record outlives the last write that recorded it.  Every write
# re-records every held anchor, so only a departed task's record expires, or a
# live one through a whole TTL with no write; a restart then re-anchors that
# task to its own id.
AGE_ANCHOR_TTL = timedelta(days=30)


def _as_stamp(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _pending_since(task: Mapping) -> str | None:
    """The task's D3 ``metadata.pending_since`` stamp; None when absent or
    malformed, never raising, because every tick reads it."""
    metadata = task.get('metadata')
    return _as_stamp(metadata.get('pending_since')) if isinstance(metadata, dict) else None


@dataclass(frozen=True)
class AgeAnchor:
    """A task's age anchor and the ``pending_since`` stamp it was taken at.

    A later stamp that differs is D3's cancelled -> pending, which resets the
    anchor (:meth:`AgeAnchors.observe`).
    """

    anchor: int
    pending_since: str | None

    def is_reset_by(self, stamp: str | None) -> bool:
        """Whether *stamp* differs from a known stamp.  A first stamp, where
        none was known, is not a reset, and neither is a missing one."""
        return bool(self.pending_since and stamp and stamp != self.pending_since)

    def to_record(self) -> dict:
        return {'anchor': self.anchor, 'pending_since': self.pending_since}

    @classmethod
    def from_record(cls, raw: object) -> AgeAnchor | None:
        """Parse a guard-file record; a malformed one reads as absent, keeping
        guard_state's fail-open contract."""
        if not isinstance(raw, dict):
            return None
        anchor = raw.get('anchor')
        if not isinstance(anchor, int) or isinstance(anchor, bool):
            return None
        return cls(anchor, _as_stamp(raw.get('pending_since')))


class AgeAnchors:
    """The held anchors of the active fetch, and the durable record of every
    anchor, held or departed.  A ``None`` *project_root* keeps it in memory."""

    def __init__(
        self,
        project_root: str | Path | None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._held: dict[str, AgeAnchor] = {}
        self._records = PersistentMap(
            guard_path(project_root, AGE_ANCHORS_FILE), ttl=AGE_ANCHOR_TTL, now=now,
        )

    def observe(self, tasks: Iterable[Mapping], max_id: int) -> Mapping[str, AgeAnchor]:
        """Hold one anchor per active task in *tasks*; return the resets applied.

        - A task with an empty or terminal status is skipped: it is not an
          observation of the active population, and recording its departure
          is :meth:`record_departures`' job alone.
        - Any other task keeps its known anchor: the held one, else the
          recorded one, else its own numeric id (so a genuinely-old task
          carries its accumulated age), else *max_id*.  So a requeue, an
          unblock, an undefer, an infra-hold resume, a done reopen and a
          restart all keep the wait.
        - If the task's stamp resets the known anchor
          (:meth:`AgeAnchor.is_reset_by`), the anchor moves to *max_id*.

        An anchor is recorded as soon as it differs from its record, so the
        stamp it was taken at survives a restart.  A cancel and reopen while
        no process runs is therefore still reset at the next start, and a
        reset is applied ONCE, because its record carries the new stamp.  An
        observation that changes no anchor writes nothing.
        """
        changed: dict[str, AgeAnchor] = {}
        resets: dict[str, AgeAnchor] = {}
        for t in tasks:
            tid = str(t.get('id', ''))
            status = t.get('status', '')
            if not tid or not status or status in TERMINAL_STATUSES:
                continue
            stamp = _pending_since(t)
            stored = self._held.get(tid) or AgeAnchor.from_record(self._records.get(tid))
            known = stored or AgeAnchor(int(tid) if tid.isdecimal() else max_id, stamp)
            if known.is_reset_by(stamp):
                current = AgeAnchor(max_id, stamp)
                resets[tid] = current
            else:
                current = AgeAnchor(known.anchor, stamp or known.pending_since)
            if current != stored:
                changed[tid] = current
            self._held[tid] = current
        self._record(changed)
        return resets

    def record_departures(self, task_ids: Iterable[str]) -> None:
        """Stop holding each of *task_ids* and record its anchor, so a return
        restores it.  An id not held is ignored."""
        departed = {tid: self._held.pop(tid) for tid in task_ids if tid in self._held}
        self._record(departed)

    def age(self, task_id: str, max_id: int) -> int:
        """The task's age in newer-task-count units; 0 when it is not held."""
        held = self._held.get(task_id)
        if held is None:
            return 0
        return max(0, max_id - held.anchor)

    def tracked_ids(self) -> frozenset[str]:
        return frozenset(self._held)

    def _record(self, records: Mapping[str, AgeAnchor]) -> None:
        """Write *records*, and every held anchor with them, in ONE write.

        One write, because each write rewrites the whole file: a cold start
        or a mass departure of N tasks would otherwise perform N writes of an
        N-entry file.  Re-recording the held anchors adds no write and
        refreshes their TTL, so a task that stays in the fetch keeps its
        record while any write lands within ``AGE_ANCHOR_TTL``.  No records,
        no write.
        """
        if not records:
            return
        self._records.put_many({tid: a.to_record() for tid, a in {**self._held, **records}.items()})
