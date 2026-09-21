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

``now`` is always injected rather than read from the wall clock internally,
so a caller — and every TTL test — can drive elapsed scenarios deterministically
(the convention
``orchestrator/src/orchestrator/chronic_flake.py::FilingLedger`` established).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, MutableMapping, MutableSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for "no entry", so a stored ``None`` stays distinguishable from
# absence on the read path.
_MISSING: Any = object()


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
    entries, the clock and (from step-4) the file.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        ttl: timedelta,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._path = path
        self._ttl = ttl
        self._now = now
        self._entries: dict[str, _Entry] = {}

    # --- reads ---

    def live_keys(self) -> list[str]:
        """Every key currently visible, in insertion order."""
        return list(self._entries)

    def has_live(self, key: str) -> bool:
        """Whether *key* is currently visible."""
        return key in self._entries

    def live_get(self, key: str, default: Any = None) -> Any:
        """The value stored under *key*, or *default* if it is not present."""
        entry = self._entries.get(key)
        return default if entry is None else entry.value

    # --- mutations (each flushes through the one chokepoint) ---

    def put(self, key: str, value: Any) -> None:
        """Store *value* under *key*, always refreshing the timestamp."""
        self._entries[key] = _Entry(value=value, updated_at=self._now())
        self.flush()

    def insert_if_absent(self, key: str, value: Any) -> bool:
        """Store *value* under *key* only if *key* is not already present.

        Returns whether anything changed.  A present key is left with its
        ORIGINAL timestamp: the set facade's ``add`` is re-issued on every
        tick by some callers, and re-timestamping would make the TTL measure
        "last seen" rather than "first observed".
        """
        if key in self._entries:
            return False
        self._entries[key] = _Entry(value=value, updated_at=self._now())
        self.flush()
        return True

    def remove(self, key: str) -> bool:
        """Drop *key*.  Returns whether it was there to drop."""
        if self._entries.pop(key, None) is None:
            return False
        self.flush()
        return True

    def flush(self) -> None:
        """Persist the entries.  No-op until step-4 gives this store its disk half."""


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
        now: Callable[[], datetime] = _utc_now,
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
        now: Callable[[], datetime] = _utc_now,
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
