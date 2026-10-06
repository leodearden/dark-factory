"""The session window one census run mines.

A run mines the sessions dated in the last ``census.ledger_retention_days``
UTC days before today, minus those already ledgered as coded and those with
no confusion signal. Until the ledger holds a census row, the window starts
no earlier than the last census, so the first ledgered run does not re-mine
history that predates the ledger. Contract: plans/census-incremental-prd.md
§4.2 (C2, decision D2).

Imports only downward; it never imports census.py.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from legibility import digest, inventory, sampling, session_ledger
from legibility.config import LegibilityConfig

logger = logging.getLogger('legibility.census_window')

DEFAULT_BATCH_SIZE = 20
"""Sessions per mining batch."""


def _utc_midnight(day: date) -> datetime:
    return datetime.combine(day, time(), UTC)


@dataclass(frozen=True)
class MiningWindow:
    """The half-open UTC interval ``[start, end)`` of session dates mined."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for name, value in (('start', self.start), ('end', self.end)):
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f'MiningWindow.{name} must be timezone-aware: {value!r}')
        if self.start > self.end:
            raise ValueError(
                f'MiningWindow.start {self.start.isoformat()} is after '
                f'end {self.end.isoformat()}',
            )

    @property
    def first_day(self) -> date:
        return self.start.date()

    @property
    def last_day(self) -> date:
        return (self.end - timedelta(days=1)).date()

    @property
    def is_empty(self) -> bool:
        return self.start == self.end

    def to_record(self) -> list[str]:
        return [self.start.isoformat(), self.end.isoformat()]


def retention_window(now: datetime, retention_days: int) -> MiningWindow:
    end = _utc_midnight(now.astimezone(UTC).date())
    return MiningWindow(start=end - timedelta(days=retention_days), end=end)


def transition_window(window: MiningWindow, *, last_census_at: date | None) -> MiningWindow:
    """*window* with its start raised to the last census's day, never past
    its end."""
    if last_census_at is None:
        return window
    floor = min(max(window.start, _utc_midnight(last_census_at)), window.end)
    return MiningWindow(start=floor, end=window.end)


@dataclass(frozen=True)
class SessionSelection:
    """Which sessions one census run chose to mine, and why the rest were
    skipped."""

    window: MiningWindow
    ledger: session_ledger.LedgerSnapshot
    sessions_enumerated: int
    skipped_coded: int
    skipped_zero_signal: int

    def __post_init__(self) -> None:
        counts = (self.sessions_enumerated, self.skipped_coded, self.skipped_zero_signal)
        if min(counts) < 0 or self.skipped_coded + self.skipped_zero_signal > self.sessions_enumerated:
            raise ValueError(
                f'SessionSelection counts are inconsistent: '
                f'sessions_enumerated={self.sessions_enumerated} '
                f'skipped_coded={self.skipped_coded} '
                f'skipped_zero_signal={self.skipped_zero_signal}',
            )


def _stratified_random_order(by_stratum: dict, *, rng: random.Random) -> list:
    """Interleave *by_stratum* (``{stratum: [ScoredRecord, ...]}``)
    round-robin, each stratum's own list independently shuffled first, so
    every batch mine_to_saturation draws is a representative random
    cross-section of every active stratum rather than front-loading one
    stratum's sessions ahead of another's (the "stratified-RANDOM"
    sampling this task's design decisions call for)."""
    queues = [list(records) for records in by_stratum.values()]
    for queue in queues:
        rng.shuffle(queue)

    order = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop())
    return order


def _render_batch(records: Sequence[sampling.ScoredRecord]) -> list[str]:
    digests = []
    for record in records:
        try:
            digests.append(digest.build_digest(record.path, agent_class_override=record.stratum))
        except Exception as exc:  # noqa: BLE001 - isolate one bad transcript, keep mining
            logger.warning('census: failed to build digest for %s: %s', record.path, exc)
    return digests


class WindowBatchSource:
    """The census's batch source: digest batches of the sessions in its
    mining window that are neither ledgered as coded nor zero-signal, in
    stratified-random order.

    Lazy: nothing is opened, enumerated or rendered until the first
    iteration, so a run deferred before mining touches nothing. The first
    iteration opens (creating or pruning) the ledger and publishes
    :attr:`selection`. A ledgered session is recognised from its first
    ``sessionId`` (:func:`sampling.peek_session_id`), so it costs no full
    transcript scan.
    """

    def __init__(
        self,
        cfg: LegibilityConfig,
        *,
        projects_root: Path | str,
        now: datetime,
        ledger_path: Path,
        last_census_at: date | None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rng: random.Random | None = None,
    ) -> None:
        self._cfg = cfg
        self._projects_root = projects_root
        self._now = now
        self._ledger_path = ledger_path
        self._last_census_at = last_census_at
        self._batch_size = batch_size
        self._rng = rng if rng is not None else random.Random()
        self._selection: SessionSelection | None = None

    @property
    def selection(self) -> SessionSelection | None:
        """``None`` until the source has been iterated."""
        return self._selection

    def __iter__(self) -> Iterator[list[str]]:
        by_stratum: dict[str, list[sampling.ScoredRecord]] = {}
        for record in self._select():
            by_stratum.setdefault(record.stratum, []).append(record)
        ordered = _stratified_random_order(by_stratum, rng=self._rng)
        for start in range(0, len(ordered), self._batch_size):
            digests = _render_batch(ordered[start:start + self._batch_size])
            if digests:
                yield digests

    def _select(self) -> list[sampling.ScoredRecord]:
        full = retention_window(self._now, self._cfg.census.ledger_retention_days)
        snapshot = session_ledger.open_for_census(self._ledger_path, prune_before=full.start)
        window = (
            full if snapshot.has_census_rows
            else transition_window(full, last_census_at=self._last_census_at)
        )
        if snapshot.state is session_ledger.LedgerState.UNREADABLE:
            logger.warning(
                'census: session ledger unreadable, so no already-coded session is '
                'skipped this run: %s',
                snapshot.error,
            )
        sessions = [] if window.is_empty else inventory.enumerate_sessions_in_range(
            self._projects_root, self._cfg.cwd_prefixes,
            window.first_day, window.last_day,
            agent_transcript_roots=inventory.resolve_agent_transcript_roots(
                self._cfg.project_root, self._cfg.agent_transcript_roots,
            ),
        )
        uncoded = [
            session for session in sessions
            if sampling.peek_session_id(session.path) not in snapshot.sessions
        ]
        scored = [sampling.score_session(session) for session in uncoded]
        eligible = [record for record in scored if record.score > 0]
        self._selection = SessionSelection(
            window=window,
            ledger=snapshot,
            sessions_enumerated=len(sessions),
            skipped_coded=len(sessions) - len(uncoded),
            skipped_zero_signal=len(scored) - len(eligible),
        )
        return eligible
