"""The derived-state store of recorded verify greens: deleting its file is a
cold start, never an error (plans/verify-result-cache-prd.md § Registry)."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, TypeVar

from shared.project_layout import orchestrator_data_dir
from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_evidence import UnitTier

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

#: I5: the cache is disabled only on the evidence of at least this many leaks.
MIN_DISABLING_LEAKS = 2

#: What a damaged file or a row breaking an invariant raises while being read or written.
_UNREADABLE = (sqlite3.Error, OSError, RuntimeError, ValueError, TypeError, KeyError)

_ENTRY_COLUMNS = (
    'entry_id, unit_id, key, tier, spec_hash, inputs, evidence_event_id, run_id, sha, recorded_at'
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id TEXT NOT NULL,
    key TEXT NOT NULL,
    tier TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    inputs TEXT NOT NULL,
    evidence_event_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    sha TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    UNIQUE (unit_id, key)
);
CREATE INDEX IF NOT EXISTS entries_by_key ON entries (key);
CREATE TABLE IF NOT EXISTS consumed (
    landing_sha TEXT NOT NULL,
    entry_id INTEGER NOT NULL,
    unit_id TEXT NOT NULL,
    key TEXT NOT NULL,
    tier TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    inputs TEXT NOT NULL,
    evidence_event_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    sha TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    consumed_at TEXT NOT NULL,
    PRIMARY KEY (landing_sha, entry_id)
);
CREATE TABLE IF NOT EXISTS policy_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    disabled_until TEXT,
    reason TEXT,
    streak INTEGER NOT NULL,
    leaks TEXT NOT NULL,
    updated_at TEXT
);
"""


class RegistryError(RuntimeError):
    """The registry file at *path* could not be read or written; *cause* says why."""

    def __init__(self, path: Path, cause: str) -> None:
        super().__init__(f'verify-cache registry {path}: {cause}')
        self.path = path
        self.cause = cause


def registry_db_path(project_root: Path) -> Path:
    return orchestrator_data_dir(project_root) / 'verify_cache.db'


@dataclass(frozen=True)
class Entry:
    """One recorded green; its key fields obey every UnitKey invariant."""

    entry_id: int
    unit_id: str
    key: str
    tier: UnitTier
    spec_hash: str
    inputs: tuple[tuple[str, str], ...]
    evidence_event_id: int
    run_id: str
    sha: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        UnitKey(self.unit_id, self.tier, self.key, self.spec_hash, self.inputs)
        _require_aware('entry.recorded_at', self.recorded_at)


@dataclass(frozen=True)
class PolicyState:
    """The kill switch and its leak streak (I5). Built at write and again at
    read, so a hand-edited row fails with the writer's message."""

    disabled_until: datetime | None = None
    reason: str | None = None
    streak: int = 0
    leaks: tuple[str, ...] = ()
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_aware('policy_state.disabled_until', self.disabled_until)
        _require_aware('policy_state.updated_at', self.updated_at)
        if self.streak < 0:
            raise ValueError(f'policy_state.streak must be >= 0, got {self.streak}')
        if self.disabled_until is None:
            return
        if not self.reason:
            raise ValueError(f'policy_state.disabled_until is set but reason is {self.reason!r}')
        if len(self.leaks) < MIN_DISABLING_LEAKS:
            raise ValueError(
                f'policy_state.disabled_until needs >= {MIN_DISABLING_LEAKS} leak records, got {self.leaks!r}',
            )

    def is_disabled(self, now: datetime) -> bool:
        return self.disabled_until is not None and now < self.disabled_until


@dataclass(frozen=True)
class RegistryView:
    """The entries recorded for a verify's keys, by ``(unit_id, key)``, and the policy row."""

    entries: Mapping[tuple[str, str], Entry]
    policy: PolicyState


class Pruned(NamedTuple):
    entries: int
    consumed: int


@dataclass(frozen=True)
class RegistrySummary:
    path: Path
    exists: bool
    entries_by_tier: Mapping[UnitTier, int]
    consumed: int
    policy: PolicyState = field(default_factory=PolicyState)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Registry:
    """Async access to one registry file; each method is one thread hop over one
    connection, so a verify's reads never block the event loop (I7)."""

    def __init__(
        self, db_path: Path, *, busy_timeout_ms: int = 5000, clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._db_path = db_path
        self._busy_timeout_ms = busy_timeout_ms
        self._clock = clock

    async def view(self, keys: Iterable[UnitKey]) -> RegistryView:
        wanted = [(key.unit_id, key.key) for key in keys]
        return await self._call(lambda conn: _view(conn, wanted))

    async def record_green(self, key: UnitKey, *, evidence_event_id: int, run_id: str, sha: str) -> Entry:
        now = self._clock()
        return await self._call(lambda conn: _record_green(conn, key, evidence_event_id, run_id, sha, now))

    async def consume(self, landing_sha: str, entries: Iterable[Entry]) -> None:
        consumed, now = tuple(entries), self._clock()
        await self._call(lambda conn: _consume(conn, landing_sha, consumed, now))

    async def consumed_for(self, landing_sha: str) -> tuple[Entry, ...]:
        return await self._call(lambda conn: _consumed_for(conn, landing_sha))

    async def evict_key(self, key: str, reason: str) -> int:
        removed = await self._call(lambda conn: conn.execute('DELETE FROM entries WHERE key = ?', (key,)).rowcount)
        logger.warning('verify_cache.evict key=%s entries=%d reason=%s', key, removed, reason)
        return removed

    async def prune(self, *, entries_recorded_before: datetime, consumed_before: datetime) -> Pruned:
        """Delete greens last recorded before *entries_recorded_before*, and
        consumption records older than *consumed_before* (the audit horizon).
        A pruned green is re-earned by the next run of its unit."""
        _require_aware('entries_recorded_before', entries_recorded_before)
        _require_aware('consumed_before', consumed_before)
        pruned = await self._call(lambda conn: _prune(conn, entries_recorded_before, consumed_before))
        logger.info('verify_cache.prune entries=%d consumed=%d', pruned.entries, pruned.consumed)
        return pruned

    async def policy_state(self) -> PolicyState:
        return await self._call(_policy_state)

    async def write_policy_state(self, state: PolicyState) -> None:
        """Replace the policy row. The only sanctioned writer is the audit's apply step (I5)."""
        await self._call(lambda conn: _write_policy_state(conn, state))

    async def summary(self) -> RegistrySummary:
        """Counts for an operator; reads only, and never creates a missing file."""
        return await asyncio.to_thread(self._summary)

    async def _call(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        return await asyncio.to_thread(self._in_connection, operation)

    def _in_connection(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        try:
            conn = _open(self._db_path, self._busy_timeout_ms)
            try:
                with conn:
                    return operation(conn)
            finally:
                conn.close()
        except _UNREADABLE as exc:
            raise RegistryError(self._db_path, str(exc)) from exc

    def _summary(self) -> RegistrySummary:
        if not self._db_path.exists():
            return RegistrySummary(self._db_path, False, dict.fromkeys(UnitTier, 0), 0)
        try:
            conn = sqlite3.connect(f'{self._db_path.absolute().as_uri()}?mode=ro', uri=True)
            try:
                conn.execute(f'PRAGMA busy_timeout={self._busy_timeout_ms}')
                return _summarize(conn, self._db_path)
            finally:
                conn.close()
        except _UNREADABLE as exc:
            raise RegistryError(self._db_path, str(exc)) from exc


def _open(db_path: Path, busy_timeout_ms: int) -> sqlite3.Connection:
    """A connection with the durability pragmas and the schema; closed if either fails."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=busy_timeout_ms)
        conn.executescript(_SCHEMA)
    except BaseException:
        conn.close()
        raise
    return conn


def _view(conn: sqlite3.Connection, wanted: list[tuple[str, str]]) -> RegistryView:
    entries = {}
    for unit_id, key in wanted:
        row = conn.execute(
            f'SELECT {_ENTRY_COLUMNS} FROM entries WHERE unit_id = ? AND key = ?', (unit_id, key),
        ).fetchone()
        if row is not None:
            entries[(unit_id, key)] = _entry(row)
    return RegistryView(MappingProxyType(entries), _policy_state(conn))


def _record_green(
    conn: sqlite3.Connection, key: UnitKey, evidence_event_id: int, run_id: str, sha: str, now: datetime,
) -> Entry:
    conn.execute(
        'INSERT INTO entries (unit_id, key, tier, spec_hash, inputs, evidence_event_id, run_id, sha, recorded_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) '
        'ON CONFLICT (unit_id, key) DO UPDATE SET evidence_event_id = excluded.evidence_event_id, '
        'run_id = excluded.run_id, sha = excluded.sha, recorded_at = excluded.recorded_at',
        (
            key.unit_id, key.key, str(key.tier), key.spec_hash, json.dumps(key.inputs),
            evidence_event_id, run_id, sha, now.isoformat(),
        ),
    )
    row = conn.execute(
        f'SELECT {_ENTRY_COLUMNS} FROM entries WHERE unit_id = ? AND key = ?', (key.unit_id, key.key),
    ).fetchone()
    return _entry(row)


def _consume(conn: sqlite3.Connection, landing_sha: str, entries: tuple[Entry, ...], now: datetime) -> None:
    conn.executemany(
        f'INSERT OR IGNORE INTO consumed (landing_sha, {_ENTRY_COLUMNS}, consumed_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [
            (
                landing_sha, entry.entry_id, entry.unit_id, entry.key, str(entry.tier), entry.spec_hash,
                json.dumps(entry.inputs), entry.evidence_event_id, entry.run_id, entry.sha,
                entry.recorded_at.isoformat(), now.isoformat(),
            )
            for entry in entries
        ],
    )


def _prune(conn: sqlite3.Connection, entries_before: datetime, consumed_before: datetime) -> Pruned:
    entries = conn.execute(
        'DELETE FROM entries WHERE julianday(recorded_at) < julianday(?)', (entries_before.isoformat(),),
    ).rowcount
    consumed = conn.execute(
        'DELETE FROM consumed WHERE julianday(consumed_at) < julianday(?)', (consumed_before.isoformat(),),
    ).rowcount
    return Pruned(entries, consumed)


def _consumed_for(conn: sqlite3.Connection, landing_sha: str) -> tuple[Entry, ...]:
    rows = conn.execute(
        f'SELECT {_ENTRY_COLUMNS} FROM consumed WHERE landing_sha = ? ORDER BY entry_id', (landing_sha,),
    ).fetchall()
    return tuple(_entry(row) for row in rows)


def _policy_state(conn: sqlite3.Connection) -> PolicyState:
    row = conn.execute(
        'SELECT disabled_until, reason, streak, leaks, updated_at FROM policy_state WHERE id = 1',
    ).fetchone()
    if row is None:
        return PolicyState()
    disabled_until, reason, streak, leaks, updated_at = row
    return PolicyState(
        disabled_until=_datetime_or_none(disabled_until),
        reason=reason,
        streak=streak,
        leaks=tuple(str(leak) for leak in json.loads(leaks)),
        updated_at=_datetime_or_none(updated_at),
    )


def _write_policy_state(conn: sqlite3.Connection, state: PolicyState) -> None:
    conn.execute(
        'INSERT INTO policy_state (id, disabled_until, reason, streak, leaks, updated_at) '
        'VALUES (1, ?, ?, ?, ?, ?) '
        'ON CONFLICT (id) DO UPDATE SET disabled_until = excluded.disabled_until, reason = excluded.reason, '
        'streak = excluded.streak, leaks = excluded.leaks, updated_at = excluded.updated_at',
        (
            _iso_or_none(state.disabled_until), state.reason, state.streak,
            json.dumps(list(state.leaks)), _iso_or_none(state.updated_at),
        ),
    )


def _summarize(conn: sqlite3.Connection, path: Path) -> RegistrySummary:
    tables = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    by_tier = dict.fromkeys(UnitTier, 0)
    if 'entries' in tables:
        for tier, count in conn.execute('SELECT tier, COUNT(*) FROM entries GROUP BY tier'):
            by_tier[UnitTier(tier)] = count
    consumed = conn.execute('SELECT COUNT(*) FROM consumed').fetchone()[0] if 'consumed' in tables else 0
    policy = _policy_state(conn) if 'policy_state' in tables else PolicyState()
    return RegistrySummary(path, True, by_tier, consumed, policy)


def _entry(row: tuple[Any, ...]) -> Entry:
    entry_id, unit_id, key, tier, spec_hash, inputs, evidence_event_id, run_id, sha, recorded_at = row
    return Entry(
        entry_id=entry_id,
        unit_id=unit_id,
        key=key,
        tier=UnitTier(tier),
        spec_hash=spec_hash,
        inputs=tuple((str(name), str(value)) for name, value in json.loads(inputs)),
        evidence_event_id=evidence_event_id,
        run_id=run_id,
        sha=sha,
        recorded_at=datetime.fromisoformat(recorded_at),
    )


def _require_aware(name: str, value: datetime | None) -> None:
    if value is not None and value.utcoffset() is None:
        raise ValueError(f'{name} must be timezone-aware, got {value}')


def _datetime_or_none(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


def _iso_or_none(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()
