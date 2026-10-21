"""Program state for a /review-all Phase 6: the one library every program script imports.

The design-time statement is ``plans/review-all-program-tooling-prd.md`` §4
(H1–H3); this module is the machine home it names.
"""
from __future__ import annotations

import enum
import importlib
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKOUT_ROOT = Path(__file__).resolve().parents[4]
_SHARED_SRC = str(CHECKOUT_ROOT / 'shared' / 'src')


def _first_party(name: str) -> types.ModuleType:
    """Import a stdlib-only module from this checkout's ``shared/src``, whatever the cwd."""
    if _SHARED_SRC not in sys.path:
        sys.path.insert(0, _SHARED_SRC)
    return importlib.import_module(name)


_task_statuses = _first_party('shared.task_statuses')

RETURN_SCHEMA = 'review-all-prd-return/1'
PROGRAM_SCHEMA = 'review-all-program/1'
NON_TERMINAL_STATUSES: tuple[str, ...] = tuple(sorted(str(s) for s in _task_statuses.ACTIVE))
TERMINAL_STATUSES: tuple[str, ...] = tuple(sorted(str(s) for s in _task_statuses.TERMINAL))


@dataclass(frozen=True)
class SeamKeys:
    key: str
    fields: tuple[str, ...]


SEAM_WAIT_KEYS = SeamKeys('x_seam_wait', ('stream', 'contract', 'release_owner'))
SEAM_RELEASED_KEYS = SeamKeys('x_seam_released', ('by', 'at', 'onto', 'contract_as_delivered'))


class Pass(enum.StrEnum):
    AUTHOR = 'author'
    DECOMPOSE = 'decompose'
    INVESTIGATE = 'investigate'


class SessionKind(enum.StrEnum):
    AGENT = 'agent'
    SPAWN = 'spawn'


class QuestionSource(enum.StrEnum):
    AUTHOR = 'author'
    CRITIC = 'critic'


class LeadActionOp(enum.StrEnum):
    CANCEL = 'cancel'
    EDGE = 'edge'
    NOTE = 'note'
    STAMP = 'stamp'
    REPATH = 'repath'
    RELEASE = 'release'
    AMEND = 'amend'


class StreamMode(enum.StrEnum):
    AGENT = 'agent'
    SPAWN = 'spawn'


class StreamSize(enum.StrEnum):
    S = 'S'
    M = 'M'
    L = 'L'


class StreamRisk(enum.StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class StreamStatus(enum.StrEnum):
    BRIEFED = 'briefed'
    SPAWNED = 'spawned'
    AUTHORED = 'authored'
    COMMITTED = 'committed'
    DECOMPOSED = 'decomposed'
    RECORDED = 'recorded'


class StreamField(enum.StrEnum):
    """The stream-row fields an ``amend_stream`` ledger line may change."""

    KEYS = 'keys'
    DEPS = 'deps'
    DELETABLE = 'deletable'
    MODE = 'mode'
    REMEDY = 'remedy'


class AmendChange(enum.StrEnum):
    ADD = 'add'
    REMOVE = 'remove'
    SET = 'set'


class ProgramSection(enum.StrEnum):
    SEAMS_EXTRA = 'seams_extra'
    INVARIANTS = 'invariants'
    ROUTED = 'routed'
    CONVENTIONS_EXTRA = 'conventions_extra'
    TRIAGE = 'triage'


class ActionResultState(enum.StrEnum):
    APPLIED = 'applied'
    SKIPPED = 'skipped'
    REJECTED = 'rejected'
    KEPT_FOR_HUMAN = 'kept-for-human'


class RuledBy(enum.StrEnum):
    HUMAN = 'human'
    LEAD = 'lead'
    LEAD_YES = 'lead (--yes)'


class TriggerChainRole(enum.StrEnum):
    HUMAN_GATE = 'human_gate'
    COMPLETION_GATE = 'completion_gate'


class RejectionClass(enum.StrEnum):
    FORM = 'form'
    CONTRADICTION = 'contradiction'


class Priority(enum.StrEnum):
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


class LedgerOp(enum.StrEnum):
    STREAM_STATUS = 'stream_status'
    AMEND_STREAM = 'amend_stream'
    AMEND_PROGRAM = 'amend_program'
    XREF = 'xref'
    ACTION = 'action'
    ACTION_RESULT = 'action_result'
    SCANNED = 'scanned'
    NOTE = 'note'
    COMMIT = 'commit'


@dataclass(frozen=True)
class LedgerOpSpec:
    """The fields one ledger op carries beyond ``LEDGER_COMMON_FIELDS``, and their vocabularies."""

    required: frozenset[str]
    optional: frozenset[str]
    enums: Mapping[str, frozenset[str]]


def _values(vocabulary: type[enum.StrEnum]) -> frozenset[str]:
    return frozenset(member.value for member in vocabulary)


def _ledger_op(
    required: Sequence[str],
    optional: Sequence[str] = (),
    **enums: type[enum.StrEnum],
) -> LedgerOpSpec:
    return LedgerOpSpec(
        required=frozenset(required),
        optional=frozenset(optional),
        enums=types.MappingProxyType({field: _values(e) for field, e in enums.items()}),
    )


LEDGER_COMMON_FIELDS = frozenset({'seq', 'at', 'by', 'op'})
LEDGER_OPS: Mapping[LedgerOp, LedgerOpSpec] = types.MappingProxyType({
    LedgerOp.STREAM_STATUS: _ledger_op(
        ('stream', 'status'), ('prd_path', 'return'), status=StreamStatus,
    ),
    LedgerOp.AMEND_STREAM: _ledger_op(
        ('stream', 'field', 'change', 'value', 'ruling'),
        field=StreamField, change=AmendChange,
    ),
    LedgerOp.AMEND_PROGRAM: _ledger_op(('section', 'value'), ('ruling',), section=ProgramSection),
    LedgerOp.XREF: _ledger_op(('from', 'to', 'text', 'return')),
    LedgerOp.ACTION: _ledger_op(
        ('id', 'op', 'task', 'target', 'text', 'needs_human', 'return'), op=LeadActionOp,
    ),
    LedgerOp.ACTION_RESULT: _ledger_op(
        ('id', 'state', 'batch', 'detail'), state=ActionResultState,
    ),
    LedgerOp.SCANNED: _ledger_op(('return', 'repaths')),
    LedgerOp.NOTE: _ledger_op(('text',)),
    LedgerOp.COMMIT: _ledger_op(('sha', 'paths', 'measured_at')),
})


class StateError(Exception):
    """A program-state fact is malformed or contradicts another; the message names it."""


class ProgramInvalid(StateError):
    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        super().__init__('program.json is invalid: ' + '; '.join(self.violations))


class LedgerRefused(StateError):
    """An append-only JSONL line was refused; nothing was written."""


class StoreReaderError(StateError):
    """The task store could not be read; the message names the server and the failure."""


class StoreReaderUnsupported(StoreReaderError):
    """The server lacks a capability the reader requires (INV-11); there is no fallback."""

    def __init__(self, missing: str, detail: str) -> None:
        self.missing = missing
        super().__init__(f'the task store server does not support {missing!r}: {detail}')


class AmbiguousStream(StateError):
    """A task's finding keys belong to more than one stream."""


def _frozen(value: Any) -> Any:
    """A read-only deep copy of parsed JSON: objects become mapping proxies, arrays tuples."""
    if isinstance(value, Mapping):
        return types.MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class Stream:
    id: str
    name: str
    slug: str
    remedy: str
    deletable: bool
    keys: tuple[str, ...]
    mode: StreamMode
    size: StreamSize
    risk: StreamRisk
    deps: tuple[str, ...]
    seams: tuple[str, ...]


@dataclass(frozen=True)
class Program:
    run_id: str
    project_id: str
    project_root: str
    as_of_sha: str
    record_path: str
    doc_path: str
    plans_dir: str
    streams: tuple[Stream, ...]
    seams_extra: tuple[Any, ...]
    invariants: tuple[Any, ...]
    routed: tuple[Any, ...]
    conventions_extra: tuple[Any, ...]
    triage: tuple[Any, ...]

    @property
    def stream_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.streams)

    def stream(self, stream_id: str) -> Stream:
        for candidate in self.streams:
            if candidate.id == stream_id:
                return candidate
        raise StateError(f'no stream {stream_id!r} in program {self.run_id}')


_PROGRAM_TEXT_FIELDS = (
    'run_id', 'project_id', 'project_root', 'as_of_sha', 'record_path', 'doc_path', 'plans_dir',
)
_PROGRAM_SECTION_FIELDS = tuple(section.value for section in ProgramSection)
_STREAM_TEXT_FIELDS = ('id', 'name', 'slug', 'remedy')
_STREAM_LIST_FIELDS = ('keys', 'deps', 'seams')
_STREAM_VOCABULARIES: Mapping[str, type[enum.StrEnum]] = types.MappingProxyType({
    'mode': StreamMode, 'size': StreamSize, 'risk': StreamRisk,
})


def _typed(
    data: Mapping[str, Any], key: str, kind: type, prefix: str, violations: list[str],
) -> Any:
    """``data[key]`` when present and of *kind*; otherwise None, with a violation named by path."""
    if key not in data:
        violations.append(f'{prefix}{key}: missing')
        return None
    value = data[key]
    if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
        violations.append(f'{prefix}{key}: expected {kind.__name__}, got {value!r}')
        return None
    return value


def _text_list(
    data: Mapping[str, Any], key: str, prefix: str, violations: list[str],
) -> tuple[str, ...] | None:
    items = _typed(data, key, list, prefix, violations)
    if items is None:
        return None
    bad = [f'{prefix}{key}[{i}]: expected str, got {v!r}' for i, v in enumerate(items)
           if not isinstance(v, str)]
    violations.extend(bad)
    return None if bad else tuple(items)


def _vocabulary_value(
    data: Mapping[str, Any], key: str, vocabulary: type[enum.StrEnum], prefix: str,
    violations: list[str],
) -> enum.StrEnum | None:
    value = _typed(data, key, str, prefix, violations)
    if value is None:
        return None
    if value not in _values(vocabulary):
        violations.append(
            f'{prefix}{key}: {value!r} is not one of {sorted(_values(vocabulary))}'
        )
        return None
    return vocabulary(value)


def _parse_stream(raw: Any, index: int, violations: list[str]) -> Stream | None:
    prefix = f'streams[{index}].'
    if not isinstance(raw, Mapping):
        violations.append(f'streams[{index}]: expected object, got {raw!r}')
        return None
    found = len(violations)
    fields: dict[str, Any] = {k: _typed(raw, k, str, prefix, violations) for k in _STREAM_TEXT_FIELDS}
    fields['deletable'] = _typed(raw, 'deletable', bool, prefix, violations)
    fields.update({k: _text_list(raw, k, prefix, violations) for k in _STREAM_LIST_FIELDS})
    fields.update({
        k: _vocabulary_value(raw, k, vocabulary, prefix, violations)
        for k, vocabulary in _STREAM_VOCABULARIES.items()
    })
    return None if len(violations) > found else Stream(**fields)


def _duplicate_id_violations(streams: Sequence[Stream | None]) -> list[str]:
    seen: set[str] = set()
    violations = []
    for index, stream in enumerate(streams):
        if stream is None:
            continue
        if stream.id in seen:
            violations.append(f'streams[{index}].id: duplicate stream id {stream.id!r}')
        seen.add(stream.id)
    return violations


def _shared_key_violations(streams: Sequence[Stream]) -> list[str]:
    owners: dict[str, list[str]] = {}
    for stream in streams:
        for key in stream.keys:
            owners.setdefault(key, []).append(stream.id)
    return [
        f'key {key!r} is in more than one stream: {", ".join(ids)}'
        for key, ids in owners.items() if len(ids) > 1
    ]


def _unknown_dep_violations(streams: Sequence[Stream | None], known: set[str]) -> list[str]:
    return [
        f'streams[{index}].deps[{d}]: {dep!r} names no stream'
        for index, stream in enumerate(streams) if stream is not None
        for d, dep in enumerate(stream.deps) if dep not in known
    ]


def _cycle_members(pending: Mapping[str, Sequence[str]]) -> list[str]:
    """The pending streams that lie on a cycle: those still depended on once leaves are pruned."""
    members = set(pending)
    while True:
        depended_on = {dep for s in members for dep in pending[s] if dep in members}
        if depended_on == members:
            return sorted(members)
        members = depended_on


def _wave_numbers(deps_by_stream: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Each stream's wave, 1 + the max of its upstream waves; raises StateError naming a cycle.

    Every dependency must name a key of *deps_by_stream*.
    """
    waves: dict[str, int] = {}
    pending = dict(deps_by_stream)
    while pending:
        ready = {
            stream: 1 + max((waves[d] for d in deps), default=0)
            for stream, deps in pending.items() if all(d in waves for d in deps)
        }
        if not ready:
            raise StateError(
                f'dependency cycle among streams {", ".join(_cycle_members(pending))}'
            )
        waves.update(ready)
        for stream in ready:
            del pending[stream]
    return waves


def _cycle_violations(streams: Sequence[Stream]) -> list[str]:
    known = {s.id for s in streams}
    try:
        _wave_numbers({s.id: [d for d in s.deps if d in known] for s in streams})
    except StateError as cycle:
        return [str(cycle)]
    return []


def _stream_violations(raw_streams: Sequence[Any], streams: Sequence[Stream | None]) -> list[str]:
    """Cross-stream rules; a stream that failed its own parse still counts as a known id."""
    known = {raw.get('id') for raw in raw_streams if isinstance(raw, Mapping)}
    parsed = [s for s in streams if s is not None]
    return (
        _duplicate_id_violations(streams)
        + _shared_key_violations(parsed)
        + _unknown_dep_violations(streams, known)
        + _cycle_violations(parsed)
    )


def parse_program(data: Mapping[str, Any]) -> Program:
    """Validate program.json's parsed object into a frozen Program; raises ProgramInvalid.

    Every violation is collected and named by its path and value, not only the first.
    """
    if not isinstance(data, Mapping):
        raise ProgramInvalid([f'program.json must be a JSON object, got {data!r}'])
    violations: list[str] = []
    if data.get('schema') != PROGRAM_SCHEMA:
        violations.append(f'schema: expected {PROGRAM_SCHEMA!r}, got {data.get("schema")!r}')
    fields: dict[str, Any] = {k: _typed(data, k, str, '', violations) for k in _PROGRAM_TEXT_FIELDS}
    sections = {k: _typed(data, k, list, '', violations) for k in _PROGRAM_SECTION_FIELDS}
    raw_streams = _typed(data, 'streams', list, '', violations) or []
    streams = [_parse_stream(raw, i, violations) for i, raw in enumerate(raw_streams)]
    violations.extend(_stream_violations(raw_streams, streams))
    if violations:
        raise ProgramInvalid(violations)
    return Program(
        **fields,
        streams=tuple(s for s in streams if s is not None),
        **{k: _frozen(v) for k, v in sections.items()},
    )
