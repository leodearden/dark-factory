"""On-disk run records of the offline lane: schema, store, retention, read side."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from shared.safe_io import atomic_write_text

from orchestrator.verify_classify import is_external_kill_rc

logger = logging.getLogger(__name__)

RECORD_RETENTION = timedelta(days=14)
FIRST_LINE_MAX_CHARS = 300

# Underscored because command_key's alphabet cannot produce '_', so no slug is a
# legacy key; a generic pass named exactly a legacy name is refused by
# RunRecordStore.allocate_pass. Distinct names that slug alike share a directory
# until task 4078's name validator lands, and reads filter on the header command.
LEGACY_NUMERIC_COMMAND = 'legacy_numeric'
LEGACY_INFRA_COMMAND = 'legacy_infra'
LEGACY_COMMANDS: frozenset[str] = frozenset({LEGACY_NUMERIC_COMMAND, LEGACY_INFRA_COMMAND})

_FAILURE_KINDS = frozenset({'failed', 'error'})
_PROBE_RESULTS = frozenset({'up', 'outage', 'broken'})
_PASS_ID_TIMESTAMP_FORMAT = '%Y%m%dT%H%M%S'
_PASS_ID_TIMESTAMP_RE = re.compile(r'^\d{8}T\d{6}(?=-)')
_SHELL_SIGNAL_RANGE = range(129, 193)
_UNSET: Any = object()

_T = TypeVar('_T')


class Phase(StrEnum):
    DETECT = 'detect'
    CONFIRM = 'confirm'


class Leg(StrEnum):
    NUMERIC = 'numeric'
    INFRA = 'infra'
    GENERIC = 'generic'


class EndedBy(StrEnum):
    EXIT = 'exit'
    SIGNAL = 'signal'


def classify_exit(returncode: int) -> EndedBy:
    """SIGNAL for a negative code or the `sh -c` form 128 + n of an external kill."""
    if returncode < 0:
        return EndedBy.SIGNAL
    if returncode in _SHELL_SIGNAL_RANGE and is_external_kill_rc(-(returncode - 128)):
        return EndedBy.SIGNAL
    return EndedBy.EXIT


def command_key(name: str) -> str:
    """The directory and sentinel-id slug of a command name; legacy names pass through."""
    if name in LEGACY_COMMANDS:
        return name
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'command'


def records_root(project_root: str | Path | None) -> Path:
    if not project_root or str(project_root) == 'None':
        fallback = Path(tempfile.mkdtemp(prefix='offline-lane-records-'))
        logger.warning(
            'offline-lane: no usable project root (%r); writing run records under %s',
            project_root,
            fallback,
        )
        return fallback
    return Path(project_root, 'data', 'orchestrator', 'offline-lane')


def pass_id_for(started_at: datetime, head: str) -> str:
    return f'{started_at:{_PASS_ID_TIMESTAMP_FORMAT}}-{head[:6]}'


def _pass_started_at(pass_id: str) -> datetime | None:
    match = _PASS_ID_TIMESTAMP_RE.match(pass_id)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(), _PASS_ID_TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def _mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{what} must be an object, got {type(value).__name__}')
    return value


def _get(payload: Mapping[str, Any], key: str, expected: type[_T]) -> _T:
    if key not in payload:
        raise ValueError(f'missing {key!r}')
    value = payload[key]
    if not isinstance(value, expected) or (expected is not bool and isinstance(value, bool)):
        raise ValueError(f'{key!r} must be {expected.__name__}, got {type(value).__name__}')
    return value


def _get_optional(payload: Mapping[str, Any], key: str, expected: type[_T]) -> _T | None:
    if key in payload and payload[key] is None:
        return None
    return _get(payload, key, expected)


def _get_instant(payload: Mapping[str, Any], key: str) -> datetime:
    instant = datetime.fromisoformat(_get(payload, key, str))
    if instant.tzinfo is None:
        raise ValueError(f'{key!r} must carry a UTC offset')
    try:
        return instant.astimezone(UTC)
    except OverflowError as exc:
        raise ValueError(f'{key!r} is out of range in UTC: {exc}') from exc


def _get_member(payload: Mapping[str, Any], key: str, allowed: Collection[str]) -> str:
    value = _get(payload, key, str)
    if value not in allowed:
        raise ValueError(f'{key!r} must be one of {sorted(allowed)}, got {value!r}')
    return value


@dataclass(frozen=True)
class FailureRecord:
    node: str
    kind: str
    first_line: str

    @classmethod
    def from_json(cls, payload: object) -> FailureRecord:
        fields = _mapping(payload, 'failure')
        return cls(
            node=_get(fields, 'node', str),
            kind=_get_member(fields, 'kind', _FAILURE_KINDS),
            first_line=_get(fields, 'first_line', str),
        )


@dataclass(frozen=True)
class ProbeResult:
    result: str
    returncode: int | None
    failed_dependency: str | None
    first_line: str
    duration: float

    @classmethod
    def from_json(cls, payload: object) -> ProbeResult:
        fields = _mapping(payload, 'probe')
        duration = fields.get('duration')
        if isinstance(duration, bool) or not isinstance(duration, int | float):
            raise ValueError("'duration' must be a number")
        return cls(
            result=_get_member(fields, 'result', _PROBE_RESULTS),
            returncode=_get_optional(fields, 'returncode', int),
            failed_dependency=_get_optional(fields, 'failed_dependency', str),
            first_line=_get(fields, 'first_line', str),
            duration=float(duration),
        )


@dataclass(frozen=True)
class RunRecord:
    command: str
    leg: str
    head: str
    phase: str
    started_at: datetime
    ended_at: datetime
    returncode: int
    ended_by: str
    output_path: str
    junit_path: str | None
    failures: tuple[FailureRecord, ...]
    ids_available: bool
    outcomes: dict[str, str] | None
    session_complete: bool
    probe: ProbeResult | None

    def to_json(self) -> dict[str, Any]:
        return {
            'command': self.command,
            'leg': self.leg,
            'head': self.head,
            'phase': self.phase,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat(),
            'returncode': self.returncode,
            'ended_by': self.ended_by,
            'output_path': self.output_path,
            'junit_path': self.junit_path,
            'failures': [asdict(failure) for failure in self.failures],
            'ids_available': self.ids_available,
            'outcomes': self.outcomes,
            'session_complete': self.session_complete,
            'probe': None if self.probe is None else asdict(self.probe),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> RunRecord:
        fields = _mapping(payload, 'run record')
        outcomes = _get_optional(fields, 'outcomes', dict)
        if outcomes is not None and not all(
            isinstance(node, str) and isinstance(verdict, str) for node, verdict in outcomes.items()
        ):
            raise ValueError("'outcomes' must map strings to strings")
        probe = _get_optional(fields, 'probe', dict)
        return cls(
            command=_get(fields, 'command', str),
            leg=_get_member(fields, 'leg', {leg.value for leg in Leg}),
            head=_get(fields, 'head', str),
            phase=_get_member(fields, 'phase', {phase.value for phase in Phase}),
            started_at=_get_instant(fields, 'started_at'),
            ended_at=_get_instant(fields, 'ended_at'),
            returncode=_get(fields, 'returncode', int),
            ended_by=_get_member(fields, 'ended_by', {ended_by.value for ended_by in EndedBy}),
            output_path=_get(fields, 'output_path', str),
            junit_path=_get_optional(fields, 'junit_path', str),
            failures=tuple(FailureRecord.from_json(item) for item in _get(fields, 'failures', list)),
            ids_available=_get(fields, 'ids_available', bool),
            outcomes=outcomes,
            session_complete=_get(fields, 'session_complete', bool),
            probe=None if probe is None else ProbeResult.from_json(probe),
        )


@dataclass(frozen=True)
class LanePass:
    command: str
    leg: str
    head: str
    pass_id: str
    started_at: datetime
    directory: Path

    def output_path(self, phase: Phase | str) -> Path:
        return self.directory / f'{phase}.log'

    def junit_path(self, phase: Phase | str) -> Path:
        return self.directory / f'{phase}.junit.xml'

    @property
    def record_path(self) -> Path:
        return self.directory / 'record.json'


@dataclass(frozen=True)
class PassRecord:
    command: str
    leg: str
    head: str
    pass_id: str
    started_at: datetime
    detect: RunRecord | None
    confirm: RunRecord | None
    outcome: Any | None
    effect: Any | None


def _run_block(payload: Mapping[str, Any], phase: Phase) -> RunRecord | None:
    block = _get_optional(payload, phase.value, dict)
    return None if block is None else RunRecord.from_json(block)


def _pass_from_json(payload: object) -> PassRecord:
    fields = _mapping(payload, 'pass record')
    for key in ('outcome', 'effect'):
        if key not in fields:
            raise ValueError(f'missing {key!r}')
    return PassRecord(
        command=_get(fields, 'command', str),
        leg=_get_member(fields, 'leg', {leg.value for leg in Leg}),
        head=_get(fields, 'head', str),
        pass_id=_get(fields, 'pass_id', str),
        started_at=_get_instant(fields, 'started_at'),
        detect=_run_block(fields, Phase.DETECT),
        confirm=_run_block(fields, Phase.CONFIRM),
        outcome=fields['outcome'],
        effect=fields['effect'],
    )


def is_well_formed(payload: object) -> bool:
    try:
        _pass_from_json(payload)
    except ValueError:
        return False
    return True


def _subdirectories(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(entry for entry in directory.iterdir() if entry.is_dir())


def _listed_best_effort(directory: Path) -> list[Path]:
    try:
        return _subdirectories(directory)
    except OSError as exc:
        logger.warning('offline-lane: could not list run records %s: %s', directory, exc)
        return []


def _removed(pass_dir: Path) -> bool:
    try:
        shutil.rmtree(pass_dir)
    except OSError as exc:
        logger.warning('offline-lane: could not prune run records %s: %s', pass_dir, exc)
        return False
    return True


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise LookupError(f'no readable run record at {path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise LookupError(f'run record at {path} is not a JSON object')
    return payload


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + '\n', mkdir=True)


class RunRecordStore:
    def __init__(
        self,
        project_root: str | Path | None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = records_root(project_root)
        self._clock = clock or _utc_now
        self._current: dict[str, LanePass] = {}
        self._last: dict[tuple[str, str], RunRecord] = {}

    def now(self) -> datetime:
        return self._clock()

    def pass_directory(self, command: str, pass_id: str) -> Path:
        return self.root / command_key(command) / pass_id

    def allocate_pass(self, command: str, leg: Leg | str, head: str) -> LanePass:
        if Leg(leg) is Leg.GENERIC and command in LEGACY_COMMANDS:
            raise ValueError(f'{command!r} names a legacy leg, not a generic command')
        started_at = self.now()
        pass_id = pass_id_for(started_at, head)
        lane_pass = LanePass(
            command=command,
            leg=Leg(leg).value,
            head=head,
            pass_id=pass_id,
            started_at=started_at,
            directory=self.pass_directory(command, pass_id),
        )
        _write_payload(
            lane_pass.record_path,
            {
                'command': command,
                'leg': lane_pass.leg,
                'head': head,
                'pass_id': pass_id,
                'started_at': started_at.isoformat(),
                Phase.DETECT.value: None,
                Phase.CONFIRM.value: None,
                'outcome': None,
                'effect': None,
            },
        )
        self._current[command] = lane_pass
        for phase in Phase:
            self._last.pop((command, phase.value), None)
        return lane_pass

    def current_pass(self, command: str) -> LanePass | None:
        return self._current.get(command)

    def write_run(self, lane_pass: LanePass, record: RunRecord) -> None:
        payload = _read_payload(lane_pass.record_path)
        payload[Phase(record.phase).value] = record.to_json()
        _write_payload(lane_pass.record_path, payload)
        self._last[(lane_pass.command, record.phase)] = record

    def last_record(self, command: str, phase: Phase | str) -> RunRecord | None:
        return self._last.get((command, Phase(phase).value))

    def set_pass_blocks(
        self, lane_pass: LanePass, *, outcome: Any = _UNSET, effect: Any = _UNSET
    ) -> None:
        payload = _read_payload(lane_pass.record_path)
        if outcome is not _UNSET:
            payload['outcome'] = outcome
        if effect is not _UNSET:
            payload['effect'] = effect
        _write_payload(lane_pass.record_path, payload)

    def set_probe(self, lane_pass: LanePass, probe: ProbeResult) -> None:
        detect = _read_payload(lane_pass.record_path).get(Phase.DETECT.value)
        if detect is None:
            raise LookupError(f'no detect record yet at {lane_pass.record_path}')
        self.write_run(lane_pass, replace(RunRecord.from_json(detect), probe=probe))

    def load_pass(self, command: str, pass_id: str) -> PassRecord | None:
        path = self.pass_directory(command, pass_id) / 'record.json'
        if not path.is_file():
            return None
        try:
            record = _pass_from_json(_read_payload(path))
        except (LookupError, ValueError) as exc:
            logger.warning('offline-lane: skipping malformed run record %s: %s', path, exc)
            return None
        return record if record.command == command else None

    def passes_since(self, command: str, since: datetime) -> Iterator[PassRecord]:
        for pass_dir in _subdirectories(self.root / command_key(command)):
            started_at = _pass_started_at(pass_dir.name)
            if started_at is None or started_at < since:
                continue
            record = self.load_pass(command, pass_dir.name)
            if record is not None:
                yield record

    def prune(self, now: datetime, pinned_pass_ids: Collection[str] = ()) -> list[str]:
        cutoff = now - RECORD_RETENTION
        pruned: list[str] = []
        for command_dir in _subdirectories(self.root):
            for pass_dir in _listed_best_effort(command_dir):
                started_at = _pass_started_at(pass_dir.name)
                if started_at is None or started_at >= cutoff or pass_dir.name in pinned_pass_ids:
                    continue
                if _removed(pass_dir):
                    pruned.append(pass_dir.name)
        return pruned
