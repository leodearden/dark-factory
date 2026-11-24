"""Decide which audit reds are verify-cache leaks, and own every write of the cache's kill switch (I5)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from orchestrator.verify_cache.audit_run import AuditNode
from orchestrator.verify_cache.registry import MIN_DISABLING_LEAKS, Entry, PolicyState, Registry
from orchestrator.verify_evidence import Leg, module_leg_name
from orchestrator.verify_spec_shared import canonical_json

logger = logging.getLogger(__name__)

#: PRD decision 7: the streak counts attributable leaks inside this window.
LEAK_WINDOW = timedelta(days=7)
#: PRD decision 7 and § Audit: how long a tripped kill switch keeps the cache disabled.
DISABLE_FOR = timedelta(days=14)

LeakIdentity = tuple[str, str, int]

_RECORD_FIELDS = frozenset({
    'unit_id', 'key', 'entry_id', 'evidence_event_id', 'landing_sha', 'detected_at', 'nodes', 'counts',
})


@dataclass(frozen=True)
class LeakRecord:
    """One consumed green that an uncached audit run proved wrong: the shape stored in PolicyState.leaks."""

    unit_id: str
    key: str
    entry_id: int
    evidence_event_id: int
    landing_sha: str
    detected_at: datetime
    nodes: tuple[str, ...]
    counts: bool = True

    def __post_init__(self) -> None:
        if self.detected_at.utcoffset() is None:
            raise ValueError(f'leak record detected_at must be timezone-aware, got {self.detected_at}')

    @property
    def identity(self) -> LeakIdentity:
        return (self.unit_id, self.key, self.entry_id)

    def to_text(self) -> str:
        return canonical_json({
            'unit_id': self.unit_id, 'key': self.key, 'entry_id': self.entry_id,
            'evidence_event_id': self.evidence_event_id, 'landing_sha': self.landing_sha,
            'detected_at': self.detected_at.isoformat(), 'nodes': list(self.nodes), 'counts': self.counts,
        })

    @classmethod
    def parse(cls, text: str) -> LeakRecord:
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict) or set(payload) != _RECORD_FIELDS:
                raise ValueError(f'expected exactly the fields {sorted(_RECORD_FIELDS)}')
            return cls(
                unit_id=str(payload['unit_id']), key=str(payload['key']), entry_id=int(payload['entry_id']),
                evidence_event_id=int(payload['evidence_event_id']), landing_sha=str(payload['landing_sha']),
                detected_at=datetime.fromisoformat(payload['detected_at']),
                nodes=tuple(str(node) for node in payload['nodes']), counts=bool(payload['counts']),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f'malformed verify-cache leak record {text!r}: {exc}') from exc


@dataclass(frozen=True)
class Attribution:
    """The audit nodes that failed on a unit whose green *entry* the landing *landing_sha* replayed."""

    entry: Entry
    landing_sha: str
    nodes: tuple[str, ...]

    def record(self, detected_at: datetime) -> LeakRecord:
        return LeakRecord(
            unit_id=self.entry.unit_id, key=self.entry.key, entry_id=self.entry.entry_id,
            evidence_event_id=self.entry.evidence_event_id, landing_sha=self.landing_sha,
            detected_at=detected_at, nodes=self.nodes,
        )


@dataclass(frozen=True)
class AuditVerdict:
    attributions: tuple[Attribution, ...]
    unattributed: tuple[str, ...]


async def attribute(head: str, reproduced: Sequence[str], registry: Registry) -> AuditVerdict:
    """Which of the audit's *reproduced* nodes failed on a unit that *head*'s landing replayed from the cache."""
    if not reproduced:
        return AuditVerdict((), ())
    consumed: dict[str, list[Entry]] = {}
    for entry in await registry.consumed_for(head):
        consumed.setdefault(entry.unit_id, []).append(entry)
    nodes_by_entry: dict[int, list[str]] = {}
    entries: dict[int, Entry] = {}
    unattributed: list[str] = []
    for node in dict.fromkeys(reproduced):
        replayed = consumed.get(_unit_of(node) or '', [])
        if not replayed:
            unattributed.append(node)
        for entry in replayed:
            entries[entry.entry_id] = entry
            nodes_by_entry.setdefault(entry.entry_id, []).append(node)
    attributions = tuple(Attribution(entries[entry_id], head, tuple(nodes)) for entry_id, nodes in nodes_by_entry.items())
    return AuditVerdict(attributions, tuple(unattributed))


def _unit_of(node: str) -> str | None:
    parsed = AuditNode.parse(node)
    return None if parsed is None else module_leg_name(parsed.module, Leg.TEST)


@dataclass(frozen=True)
class LeakResponse:
    leaks: tuple[LeakRecord, ...]
    evicted: Mapping[str, int]
    counted: tuple[LeakIdentity, ...]
    streak_leaks: tuple[LeakRecord, ...]
    policy: PolicyState
    newly_disabled: bool
    l2_required: bool


async def apply(verdict: AuditVerdict, registry: Registry, now: datetime) -> LeakResponse:
    """Evict every leaked key, then record the leaks and trip or hold the kill switch; idempotent per leak."""
    if not verdict.attributions:
        current = await registry.policy_state()
        return LeakResponse((), {}, (), _counting(_records(current), now), current, False, current.is_disabled(now))
    leaks = tuple(attribution.record(now) for attribution in verdict.attributions)
    evicted = await _evict(registry, leaks)
    current = await registry.policy_state()
    replayable = await _replayable(registry, _records(current), now)
    step = _next_state(current, leaks, now, replayable)
    written = current
    if step.state != current:
        written = replace(step.state, updated_at=now)
        await registry.write_policy_state(written)
    return LeakResponse(
        leaks, evicted, step.counted, step.streak_leaks, written, step.newly_disabled, written.is_disabled(now),
    )


async def _evict(registry: Registry, leaks: Iterable[LeakRecord]) -> dict[str, int]:
    evicted: dict[str, int] = {}
    for leak in leaks:
        if leak.key not in evicted:
            evicted[leak.key] = await registry.evict_key(
                leak.key, f'audit leak: {leak.unit_id} entry {leak.entry_id} replayed at landing {leak.landing_sha}',
            )
    return evicted


async def _replayable(registry: Registry, stored: Sequence[LeakRecord], now: datetime) -> frozenset[LeakIdentity]:
    """The aged *stored* records whose landing still hands their entry back to attribute(), so could be redelivered."""
    aged = [record for record in stored if not _within(record, now)]
    replayable: set[LeakIdentity] = set()
    for landing_sha in dict.fromkeys(record.landing_sha for record in aged):
        still_consumed = {entry.entry_id for entry in await registry.consumed_for(landing_sha)}
        replayable.update(
            record.identity for record in aged
            if record.landing_sha == landing_sha and record.entry_id in still_consumed
        )
    return frozenset(replayable)


@dataclass(frozen=True)
class _Step:
    state: PolicyState
    counted: tuple[LeakIdentity, ...]
    streak_leaks: tuple[LeakRecord, ...]
    newly_disabled: bool


def _next_state(
    current: PolicyState, leaks: Iterable[LeakRecord], now: datetime, replayable: frozenset[LeakIdentity],
) -> _Step:
    """The policy row after *leaks*, with *current*'s updated_at so an unchanged row compares equal.

    A leak already stored, at any age, is never new. A record is dropped only once it is outside the
    window and its landing can no longer replay its entry, so no identity is forgotten while it can
    still be redelivered; a live disable keeps every record.
    """
    disabled = current.is_disabled(now)
    stored = _records(current)
    new = _new_leaks(stored, leaks)
    kept = stored if disabled else tuple(r for r in stored if _within(r, now) or r.identity in replayable)
    records = (*kept, *new)
    streak_leaks = _counting(records, now)
    trips = not disabled and bool(new) and len(streak_leaks) >= MIN_DISABLING_LEAKS
    state = PolicyState(
        disabled_until=_disabled_until(current, now, trips=trips),
        reason=_disable_reason(streak_leaks) if trips else _held_reason(current, now),
        streak=len(streak_leaks),
        leaks=tuple(record.to_text() for record in records),
        updated_at=current.updated_at,
    )
    return _Step(state, tuple(leak.identity for leak in new), streak_leaks, trips)


def _disabled_until(current: PolicyState, now: datetime, *, trips: bool) -> datetime | None:
    """A live disable holds unchanged, an expired one clears, and a trip starts a new one."""
    if current.is_disabled(now):
        return current.disabled_until
    return now + DISABLE_FOR if trips else None


def _held_reason(current: PolicyState, now: datetime) -> str | None:
    """*current*'s reason, unless the disable it explains has just expired."""
    if current.disabled_until is None or current.is_disabled(now):
        return current.reason
    return f'kill switch expired at {current.disabled_until.isoformat()}'


def _records(state: PolicyState) -> tuple[LeakRecord, ...]:
    return tuple(LeakRecord.parse(text) for text in state.leaks)


def _within(record: LeakRecord, now: datetime) -> bool:
    return now - record.detected_at <= LEAK_WINDOW


def _new_leaks(known: Iterable[LeakRecord], leaks: Iterable[LeakRecord]) -> tuple[LeakRecord, ...]:
    seen = {record.identity for record in known}
    new: list[LeakRecord] = []
    for leak in leaks:
        if leak.identity not in seen:
            seen.add(leak.identity)
            new.append(leak)
    return tuple(new)


def _counting(records: Iterable[LeakRecord], now: datetime) -> tuple[LeakRecord, ...]:
    """The records the streak counts: inside the window, and not recorded before an operator re-enable."""
    return tuple(record for record in records if _within(record, now) and record.counts)


def _disable_reason(counting: Sequence[LeakRecord]) -> str:
    leaks = '; '.join(
        f'{record.unit_id} entry {record.entry_id} at landing {record.landing_sha}' for record in counting
    )
    return f'{len(counting)} attributable verify-cache leaks within {LEAK_WINDOW.days} days: {leaks}'


async def enable(registry: Registry, reason: str, now: datetime) -> PolicyState:
    """The operator's re-enable: clear the kill switch, keeping every recorded leak so none counts again while retained."""
    if not reason or not reason.strip():
        raise ValueError(f'enable: reason must be a non-blank explanation, got {reason!r}')
    current = await registry.policy_state()
    state = PolicyState(
        disabled_until=None,
        reason=f'operator re-enabled: {reason.strip()}',
        streak=0,
        leaks=tuple(replace(record, counts=False).to_text() for record in _records(current)),
        updated_at=now,
    )
    await registry.write_policy_state(state)
    logger.warning(
        'verify_cache.enable: kill switch cleared (was disabled_until=%s) reason=%s',
        current.disabled_until, state.reason,
    )
    return state
