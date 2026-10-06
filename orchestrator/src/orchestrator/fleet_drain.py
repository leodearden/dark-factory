"""The restart-drain request: reader, honour decision and event tracker (task 5371).

``scripts/restart-all-orchestrators.sh --drain`` asks each orchestrator unit to
stop starting merge verifies by writing ``<fleet_dir>/<unit>.drain.json``; each
unit's merge-heartbeat pass reads its own request, decides whether to honour
it, halts or resumes merge admission accordingly, and acknowledges in its
heartbeat.  This module owns the reader's half of that contract: the request
path, its parse, the closed refusal vocabulary, the honour decision and the
one-``fleet_drain``-event-per-request bookkeeping.  The writer is the script;
the full contract is the 5371 spec's Contracts 1 and 2 and its Event section.

Like ``orchestrator.fleet_heartbeat``, kept dependency-free of ``Harness``:
every ambient fact (environment, clock, process liveness) is passed in.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from orchestrator.fleet_heartbeat import is_bare_unit_name

DRAIN_REQUEST_FIELDS: Mapping[str, type] = {
    'unit': str,
    'invocation_id': str,
    'sweep_pid': int,
    'requested_ts': int,
}

VerifyRecord = Mapping[str, Any]


class DrainRefusal(StrEnum):
    """Why a present drain request is not honoured; checked in this order."""

    MALFORMED = 'malformed'
    UNIT_MISMATCH = 'unit_mismatch'
    INVOCATION_MISMATCH = 'invocation_mismatch'
    SWEEP_DEAD = 'sweep_dead'
    EXPIRED = 'expired'


class DrainOutcome(StrEnum):
    """How an honoured drain request ended, as its ``fleet_drain`` event reports."""

    DRAINED = 'drained'
    VERIFIES_KILLED = 'verifies_killed'
    ABANDONED = 'abandoned'


@dataclass(frozen=True)
class DrainRequest:
    """A well-formed drain request, exactly as the sweep wrote it."""

    unit: str
    invocation_id: str
    sweep_pid: int
    requested_ts: int

    @property
    def key(self) -> tuple[int, int]:
        """What makes two reads the same request: one sweep, one request time."""
        return (self.requested_ts, self.sweep_pid)


@dataclass(frozen=True)
class MalformedDrainRequest:
    """A request file that exists but is not a well-formed request.

    *requested_ts* echoes the file's ``requested_ts`` when that one field is
    an int, so the sweep can still match the refusal to its own request.
    """

    requested_ts: int | None
    problem: str

    @property
    def refusal(self) -> DrainRefusal:
        return DrainRefusal.MALFORMED


@dataclass(frozen=True)
class DrainIdentity:
    """The incarnation a request must name to be honoured, and its age bound."""

    unit: str
    invocation_id: str
    max_age_secs: float

    @classmethod
    def from_environ(cls, env: Mapping[str, str], *, max_age_secs: float) -> DrainIdentity:
        """``ORCH_UNIT`` names the unit; systemd's ``INVOCATION_ID`` names this run of it."""
        return cls(
            unit=env.get('ORCH_UNIT', ''),
            invocation_id=env.get('INVOCATION_ID', ''),
            max_age_secs=max_age_secs,
        )


@dataclass(frozen=True)
class DrainVerdict:
    """The decision on a present request: honoured iff ``refused`` is None."""

    requested_ts: int | None
    refused: DrainRefusal | None
    request: DrainRequest | None

    def __post_init__(self) -> None:
        if self.refused is None and self.request is None:
            raise ValueError(
                'an honoured DrainVerdict must carry the request it honours '
                f'(requested_ts={self.requested_ts!r})'
            )

    @property
    def honoured(self) -> bool:
        return self.refused is None

    def heartbeat_block(self, *, admission_halted: bool) -> dict[str, Any]:
        """The heartbeat's ``drain`` acknowledgement for this verdict."""
        return {
            'requested_ts': self.requested_ts,
            'admission_halted': admission_halted,
            'refused': None if self.refused is None else str(self.refused),
        }


def drain_request_path(fleet_dir: Path, unit: str) -> Path:
    """``<fleet_dir>/<unit>.drain.json``; *unit* must be a bare file name."""
    if not is_bare_unit_name(unit):
        raise ValueError(
            f'refusing to name a drain request for unit {unit!r} in {fleet_dir}: '
            'the unit name is interpolated into <fleet_dir>/<unit>.drain.json, so '
            'it must be a non-blank single path component '
            '(e.g. "orchestrator-dark-factory.service").'
        )
    return Path(fleet_dir) / f'{unit}.drain.json'


def read_drain_request(path: Path) -> DrainRequest | MalformedDrainRequest | None:
    """Read the request at *path*: ``None`` when there is no file.

    Never raises: anything present that is not a well-formed request is a
    :class:`MalformedDrainRequest` naming what is wrong.
    """
    try:
        text = Path(path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        return MalformedDrainRequest(requested_ts=None, problem=f'unreadable: {exc}')
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return MalformedDrainRequest(requested_ts=None, problem=f'not JSON: {exc}')
    if not isinstance(raw, dict):
        return MalformedDrainRequest(
            requested_ts=None, problem=f'not a JSON object: {type(raw).__name__}',
        )
    problem = _shape_problem(raw)
    if problem is not None:
        return MalformedDrainRequest(requested_ts=_int_or_none(raw.get('requested_ts')), problem=problem)
    return DrainRequest(**raw)


def _shape_problem(raw: Mapping[str, Any]) -> str | None:
    if set(raw) != set(DRAIN_REQUEST_FIELDS):
        return f'keys {sorted(raw)} are not exactly {sorted(DRAIN_REQUEST_FIELDS)}'
    wrong = [
        f'{name}={raw[name]!r}' for name, kind in DRAIN_REQUEST_FIELDS.items()
        if type(raw[name]) is not kind
    ]
    if wrong:
        return f'wrong types (want str, str, int, int): {", ".join(wrong)}'
    if raw['sweep_pid'] <= 0:
        return f'sweep_pid={raw["sweep_pid"]} cannot name a sweep process'
    return None


def _int_or_none(value: object) -> int | None:
    return value if type(value) is int else None


def sweep_alive(pid: int) -> bool:
    """Whether the sweep process *pid* still exists; a process we may not signal counts."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, ValueError):
        return False
    return True


def evaluate_drain_request(
    read: DrainRequest | MalformedDrainRequest | None,
    identity: DrainIdentity,
    *,
    now: float,
    pid_alive: Callable[[int], bool] = sweep_alive,
) -> DrainVerdict | None:
    """Decide on what :func:`read_drain_request` returned; ``None`` when there is no request."""
    if read is None:
        return None
    if isinstance(read, MalformedDrainRequest):
        return DrainVerdict(requested_ts=read.requested_ts, refused=read.refusal, request=None)
    return DrainVerdict(
        requested_ts=read.requested_ts,
        refused=_refusal(read, identity, now=now, pid_alive=pid_alive),
        request=read,
    )


def _refusal(
    request: DrainRequest,
    identity: DrainIdentity,
    *,
    now: float,
    pid_alive: Callable[[int], bool],
) -> DrainRefusal | None:
    if request.unit != identity.unit:
        return DrainRefusal.UNIT_MISMATCH
    if not identity.invocation_id or request.invocation_id != identity.invocation_id:
        return DrainRefusal.INVOCATION_MISMATCH
    if not pid_alive(request.sweep_pid):
        return DrainRefusal.SWEEP_DEAD
    if now - request.requested_ts > identity.max_age_secs:
        return DrainRefusal.EXPIRED
    return None


@dataclass(frozen=True)
class FleetDrainEvent:
    """The one ``fleet_drain`` event an honoured request ends with."""

    request: DrainRequest
    outcome: DrainOutcome
    waited_secs: float
    awaited: tuple[VerifyRecord, ...]
    killed: tuple[VerifyRecord, ...] = ()
    refused: DrainRefusal | None = None

    @property
    def requested_ts(self) -> int:
        return self.request.requested_ts

    def as_payload(self) -> dict[str, Any]:
        return {
            'unit': self.request.unit,
            'requested_ts': self.request.requested_ts,
            'sweep_pid': self.request.sweep_pid,
            'waited_secs': self.waited_secs,
            'outcome': str(self.outcome),
            'refused': None if self.refused is None else str(self.refused),
            'merge_verifies_awaited': [dict(v) for v in self.awaited],
            'merge_verifies_killed': [dict(v) for v in self.killed],
        }


@dataclass
class _PendingDrain:
    request: DrainRequest
    awaited: dict[tuple[Any, ...], VerifyRecord] = field(default_factory=dict)

    def note(self, verifies: Iterable[VerifyRecord]) -> None:
        for verify in verifies:
            self.awaited.setdefault(_verify_identity(verify), verify)

    def end(
        self,
        outcome: DrainOutcome,
        *,
        now: float,
        killed: Sequence[VerifyRecord] = (),
        refused: DrainRefusal | None = None,
    ) -> FleetDrainEvent:
        return FleetDrainEvent(
            request=self.request,
            outcome=outcome,
            waited_secs=now - self.request.requested_ts,
            awaited=tuple(self.awaited.values()),
            killed=tuple(killed),
            refused=refused,
        )


def _verify_identity(verify: VerifyRecord) -> tuple[Any, ...]:
    return (verify.get('task_id'), verify.get('kind'), verify.get('started_ts'))


class DrainEventTracker:
    """Turns the per-pass drain verdicts into exactly one event per honoured request.

    The owner calls :meth:`observe` on every heartbeat pass and
    :meth:`shutdown` once, before the merge worker stops; each returns the
    events that pass ended (usually none).
    """

    def __init__(self) -> None:
        self._pending: _PendingDrain | None = None
        self._ended_key: tuple[int, int] | None = None

    def observe(
        self,
        verdict: DrainVerdict | None,
        verifies_in_flight: Sequence[VerifyRecord],
        *,
        now: float,
    ) -> tuple[FleetDrainEvent, ...]:
        honoured = verdict.request if verdict is not None and verdict.honoured else None
        events: list[FleetDrainEvent] = []
        pending = self._pending
        if pending is not None and (honoured is None or honoured.key != pending.request.key):
            refused = None if verdict is None else verdict.refused
            events.append(self._end(pending, DrainOutcome.ABANDONED, now=now, refused=refused))
            pending = None
        if honoured is not None and honoured.key != self._ended_key:
            pending = pending or _PendingDrain(honoured)
            self._pending = pending
            pending.note(verifies_in_flight)
            if not verifies_in_flight:
                events.append(self._end(pending, DrainOutcome.DRAINED, now=now))
        return tuple(events)

    def shutdown(
        self, verifies_in_flight: Sequence[VerifyRecord], *, now: float,
    ) -> tuple[FleetDrainEvent, ...]:
        pending = self._pending
        if pending is None:
            return ()
        pending.note(verifies_in_flight)
        if not verifies_in_flight:
            return (self._end(pending, DrainOutcome.DRAINED, now=now),)
        return (self._end(pending, DrainOutcome.VERIFIES_KILLED, now=now, killed=verifies_in_flight),)

    def _end(
        self,
        pending: _PendingDrain,
        outcome: DrainOutcome,
        *,
        now: float,
        killed: Sequence[VerifyRecord] = (),
        refused: DrainRefusal | None = None,
    ) -> FleetDrainEvent:
        self._ended_key = pending.request.key
        self._pending = None
        return pending.end(outcome, now=now, killed=killed, refused=refused)
