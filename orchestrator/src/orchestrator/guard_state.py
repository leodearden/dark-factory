"""Restart-durable, TTL-bounded state for the orchestrator's one-strike guards.

Several guards across the fleet exist to say "never do this again" — a capped
escalation the steward must stop re-adopting, a train that derailed once and
must not re-form, a red offline-lane run whose fix task is already open, a
resurrected task that must not re-claim an age bonus it did not earn.  Each
held that memory in process RAM while the fleet redeploys every ~8-15h, so
each was really "never do this again *until the next redeploy*": a permanent
verdict decayed into a per-redeploy one, and the thing the guard was supposed
to prevent recurred on a fixed cadence forever.  This module is the durable
substrate those guards now share — a key->value JSON sidecar that survives a
restart and expires on its own.

It generalises ``orchestrator/src/orchestrator/merge_drift.py::DriftCheckState``,
the one guard already remediated this way, and its rule that EVERY mutation
writes (there is no "at exit" flush to miss, and no no-fire path that forgets).
Two things are deliberately improved rather than copied.  The non-atomic
``path.write_text`` becomes ``shared/src/shared/safe_io.py::atomic_write_text``.
And a ``path`` of ``None`` yields a fully functional IN-MEMORY store rather
than a second code path: ``DriftCheckState``'s owner keeps an in-memory mirror
in lockstep purely to serve its ``_drift_state_path is None`` fallback, which
means the guard has two shapes, only one of which production ever exercises.
Absorbing ``None`` here means every call site reads as if persistence is
unconditional, bare-harness construction gets exactly today's behaviour, and
there is one path to read and one path to test.

The two public names are facades over that store, conforming to the stdlib
collection ABCs so the guards keep the vocabulary they already speak:

* :class:`PersistentSet` — a ``MutableSet`` of keys ("this happened").
* :class:`PersistentMap` — a ``MutableMapping`` of key -> JSON value
  ("this happened N times", "this fingerprint owns that task id").

Every entry carries a ``ttl`` supplied by the guard that owns it.  That value
is an upper bound rather than a tuning dial — it asserts "this state must not
outlive its subject", and each guard's subject (an escalation record, a task's
residency in the merge queue, an open fix task, a resurrected task's pending
window) has a lifetime its own module already knows, which is why the number
lives next to the guard and not in this one.  Expiry is applied on read, on
load and on flush, so an aged-out guard genuinely re-arms and the sidecar's
growth is bounded without a separate sweeper.

``now`` is always injected rather than read from the wall clock internally,
so a caller — and every TTL test — can drive elapsed scenarios deterministically
(the convention
``orchestrator/src/orchestrator/chronic_flake.py::FilingLedger`` established).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator, MutableMapping, MutableSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from shared import safe_io
from shared.safe_io import load_json_or_warn

logger = logging.getLogger(__name__)

# Sentinel for "no entry", so a stored ``None`` stays distinguishable from
# absence on the read path.
_MISSING: Any = object()


def guard_path(project_root: str | Path | None, filename: str) -> Path | None:
    """Where one guard's state lives, or ``None`` when there is no project root.

    ``None`` is not an error: it is the in-memory mode, which is what a
    bare-harness owner gets.  Returning it from HERE is what keeps the four
    call sites free of a ``state_path is None`` branch.

    The falsy/``'None'`` test is the guard
    ``orchestrator/src/orchestrator/scheduler.py::Scheduler._write_snapshot_best_effort``
    already applies for the same reason: pydantic types ``project_root`` as a
    ``Path`` and rejects ``None`` on construction AND assignment, so the only
    way an empty or literal ``'None'`` value arrives is a write that bypassed
    validation (``object.__setattr__``, as that module's guard tests do).
    Without the test, such a value would materialise a directory literally
    named ``./None/`` under the process CWD.
    """
    if not project_root or str(project_root) == 'None':
        return None
    return Path(project_root, 'data', 'orchestrator', 'guards', filename)


def _utc_now() -> datetime:
    """The default clock: an aware UTC timestamp.

    Aware, because the stored ``updated_at`` is compared against it and a
    naive/aware mix raises rather than mis-measuring.
    """
    return datetime.now(UTC)


@dataclass(frozen=True)
class _Entry:
    """One key's stored value and the moment it was recorded.

    ``updated_at`` is what the TTL measures against.  Frozen because an
    entry is replaced wholesale rather than edited — the store's two
    mutators differ precisely in whether they replace this timestamp.
    """

    value: Any
    updated_at: datetime


class _GuardStore:
    """The key->entry core shared by both facades.

    Private: callers hold a :class:`PersistentSet` or :class:`PersistentMap`,
    which is where the collection semantics live.  This class owns only the
    entries, the clock and the file.

    The cache is warmed once at construction and every mutation writes through
    one ``flush`` chokepoint (the ``LandedOutbox`` convention), so no code path
    can forget to persist.  A ``path`` of ``None`` makes both halves no-ops and
    the store purely in-memory.

    On disk the shape is ``{"<key>": {"value": <json>, "updated_at": "<iso>"}}``
    — structured, so a reader parses fields rather than a meaningful string,
    and human-readable, so an operator can see when a guard fired.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        ttl: timedelta,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = path
        self._ttl = ttl
        # Resolved here rather than as a default argument, so replacing the
        # module-level ``_utc_now`` moves the clock of every guard that did not
        # inject one — which is how the wiring tests drive expiry without
        # threading a test-only clock through four unrelated owners.
        self._now = now or _utc_now
        # Keys this instance removed, held until a write actually lands so a
        # failed flush does not silently forget the removal.  Re-storing a key
        # retracts its pending removal — otherwise a removal whose flush failed
        # would keep deleting the key from every later merge.
        self._removed: set[str] = set()
        self._entries: dict[str, _Entry] = self._load_raw()

    # --- reads ---

    def _is_live(self, entry: _Entry, now: datetime) -> bool:
        """Whether *entry* is still within the TTL at *now*.

        The TTL is an upper bound, not a tuning dial: a guard's state must not
        outlive the thing it was recorded about.  Everything reads through this
        predicate, so an expired entry is indistinguishable from an absent one
        and a guard that has aged out genuinely re-arms.
        """
        return now - entry.updated_at < self._ttl

    def live_keys(self) -> list[str]:
        """Every unexpired key, in insertion order."""
        now = self._now()
        return [k for k, e in self._entries.items() if self._is_live(e, now)]

    def has_live(self, key: str) -> bool:
        """Whether *key* is present and unexpired."""
        entry = self._entries.get(key)
        return entry is not None and self._is_live(entry, self._now())

    def live_get(self, key: str, default: Any = None) -> Any:
        """The value stored under *key*, or *default* if absent or expired."""
        entry = self._entries.get(key)
        if entry is None or not self._is_live(entry, self._now()):
            return default
        return entry.value

    # --- mutations (each flushes through the one chokepoint) ---

    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*, always refreshing the timestamp."""
        self._entries[key] = _Entry(value=value, updated_at=self._now())
        self._removed.discard(key)
        self.flush()

    def insert_if_absent(self, key: str, value: Any) -> bool:
        """Store *value* under *key* only if *key* is not already present.

        Returns whether anything changed.  A present, unexpired key is left
        with its ORIGINAL timestamp and triggers no write: the set facade's
        ``add`` is re-issued on every tick by some callers, and re-timestamping
        would make the TTL measure "last seen" rather than "first observed".
        An EXPIRED key is re-inserted with a fresh timestamp, which is how an
        aged-out guard re-arms.
        """
        if self.has_live(key):
            return False
        self._entries[key] = _Entry(value=value, updated_at=self._now())
        self._removed.discard(key)
        self.flush()
        return True

    def remove(self, key: str) -> bool:
        """Drop *key*.  Returns whether it was there to drop."""
        if self._entries.pop(key, None) is None:
            return False
        self._removed.add(key)
        self.flush()
        return True

    # --- disk ---

    def _load_raw(self) -> dict[str, _Entry]:
        """Read the file, returning ``{}`` on any failure (fail-open).

        A row that is valid JSON but schema-drifted is dropped INDIVIDUALLY
        with a WARNING rather than voiding the whole file — without that,
        fail-open would degrade to fail-empty for every other key in it
        (``orchestrator/src/orchestrator/landed_outbox.py::LandedOutbox._load_raw``).

        Losing this state is never a correctness bug, only a re-armed guard,
        which is why absent and corrupt both resolve to empty rather than
        raising into a steward loop or a scheduler tick.
        """
        if self._path is None:
            return {}
        data, ok = load_json_or_warn(self._path, default={}, on_corrupt='warn')
        if not ok or not isinstance(data, dict):
            return {}
        now = self._now()
        entries: dict[str, _Entry] = {}
        for key, row in data.items():
            entry = self._entry_from_row(key, row)
            if entry is not None and self._is_live(entry, now):
                entries[key] = entry
        return entries

    def _entry_from_row(self, key: str, row: Any) -> _Entry | None:
        """Rebuild one stored row, or ``None`` (with a WARNING) if it drifted."""
        if not isinstance(row, dict) or 'value' not in row or 'updated_at' not in row:
            logger.warning(
                'guard_state: dropping malformed row for key=%r in %s', key, self._path,
            )
            return None
        raw_updated_at = row['updated_at']
        try:
            updated_at = datetime.fromisoformat(raw_updated_at)
        except (TypeError, ValueError):
            logger.warning(
                'guard_state: dropping row with unparseable updated_at=%r for key=%r in %s',
                raw_updated_at, key, self._path,
            )
            return None
        if updated_at.tzinfo is None:
            # This store only ever writes aware timestamps; a naive one is a
            # hand-edit or a foreign writer.  Dropping it here is what keeps
            # the TTL comparison from raising on a naive/aware mix later —
            # the fail-open contract has to hold on the READ path too.
            logger.warning(
                'guard_state: dropping row with naive updated_at=%r for key=%r in %s',
                raw_updated_at, key, self._path,
            )
            return None
        return _Entry(value=row['value'], updated_at=updated_at)

    def flush(self) -> None:
        """Persist the entries: LOAD, MERGE, then write.

        Not a blind overwrite, because one file can have several owners in one
        process — multiple ``TaskSteward`` instances share one file per counter
        — and an overwrite would let one steward's flush erase another's
        entries, silently re-arming the guard this store exists to keep.
        Merge-by-key is sound because every key is globally unique (escalation
        ids, task ids).  Our own entries win and our own removals apply; the
        merged view is then adopted as this instance's cache, so a second
        owner's keys stop looking absent to us.

        Expired entries are pruned from the MERGED map, so another owner's dead
        rows are collected too and not just our own — this, plus the same prune
        on load, is the whole of what bounds the file's growth.  An empty result
        releases the file rather than writing ``{}``, so the guards directory
        cannot accumulate dead files.

        Fail-open, and that policy lives HERE rather than in the write helper
        (which always propagates): an ``OSError`` is warned and swallowed, and
        the in-memory entries stay authoritative for the rest of the process.
        A transient disk failure degrades this guard to exactly its old
        behaviour instead of crashing its caller.
        """
        if self._path is None:
            return
        merged = {**self._load_raw(), **self._entries}
        for key in self._removed:
            merged.pop(key, None)
        now = self._now()
        merged = {k: e for k, e in merged.items() if self._is_live(e, now)}
        try:
            if not merged:
                self._path.unlink(missing_ok=True)
            else:
                safe_io.atomic_write_text(
                    self._path,
                    json.dumps({k: self._row_from_entry(e) for k, e in merged.items()}),
                    encoding='utf-8',
                    mkdir=True,
                )
        except OSError as exc:
            logger.warning(
                'guard_state: failed to persist guard state at %s'
                ' (state is held in memory only): %s', self._path, exc,
            )
            return
        self._removed.clear()
        self._entries = merged

    @staticmethod
    def _row_from_entry(entry: _Entry) -> dict[str, Any]:
        return {'value': entry.value, 'updated_at': entry.updated_at.isoformat()}


class PersistentSet(MutableSet[str]):
    """A restart-durable set of keys.

    Backs the guards whose whole state is "this key happened": a capped
    escalation id, a derailed task id, a fingerprint whose blocker was already
    promoted, a task id already seen non-pending.

    ``add`` is idempotent in the strong sense — re-adding a key that is
    already present neither refreshes its TTL nor writes to disk — so a caller
    that re-asserts its whole set on every tick stays quiet.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        ttl: timedelta,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = _GuardStore(path, ttl=ttl, now=now)

    def add(self, value: str) -> None:
        self._store.insert_if_absent(value, True)

    def discard(self, value: str) -> None:
        self._store.remove(value)

    def update(self, values: Iterable[str]) -> None:
        """Add every element of *values*, writing once rather than per element.

        ``MutableSet`` has no ``update``; the derail registry marks a whole
        train in one call and the scheduler re-asserts a whole tick's
        observations, so the batched write is what keeps those quiet.
        """
        changed = False
        for value in values:
            changed |= self._store.insert_if_absent(value, True)
        if changed:
            self._store.flush()

    def __contains__(self, value: object) -> bool:
        return isinstance(value, str) and self._store.has_live(value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.live_keys())

    def __len__(self) -> int:
        return len(self._store.live_keys())

    def __repr__(self) -> str:
        """Render the live contents.

        Not cosmetic: existing assertions interpolate these attributes
        directly into their failure messages, and the ABCs supply no useful
        ``__repr__``, so without this a failure would report
        ``<PersistentSet object at 0x...>`` instead of the state it is about.
        """
        return f'{type(self).__name__}({set(self._store.live_keys())!r})'


class PersistentMap(MutableMapping[str, Any]):
    """A restart-durable map of key -> JSON-serialisable value.

    Backs the guards that count ("this escalation has burnt N attempts",
    "this fingerprint has advanced N times") or that remember an association
    ("this fingerprint's open fix task is that id").

    Unlike :class:`PersistentSet`'s ``add``, ``__setitem__`` ALWAYS refreshes
    the timestamp: a counter that is still being bumped has a subject that is
    demonstrably still alive, so its TTL should track the last bump.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        ttl: timedelta,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = _GuardStore(path, ttl=ttl, now=now)

    def __getitem__(self, key: str) -> Any:
        value = self._store.live_get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._store.put(key, value)

    def __delitem__(self, key: str) -> None:
        if not self._store.remove(key):
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.live_keys())

    def __len__(self) -> int:
        return len(self._store.live_keys())

    def __repr__(self) -> str:
        """Render the live contents — see :meth:`PersistentSet.__repr__`."""
        rendered = {k: self._store.live_get(k) for k in self._store.live_keys()}
        return f'{type(self).__name__}({rendered!r})'
