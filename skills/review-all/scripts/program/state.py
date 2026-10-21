"""Program state for a /review-all Phase 6: the one library every program script imports.

The design-time statement is ``plans/review-all-program-tooling-prd.md`` §4
(H1–H3); this module is the machine home it names.
"""
from __future__ import annotations

import enum
import fcntl
import json
import os
import re
import sys
import types
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple, Protocol, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'shared' / 'src'))

from shared import mcp_post, task_statuses

CHECKOUT_ROOT = Path(__file__).resolve().parents[4]

RETURN_SCHEMA = 'review-all-prd-return/1'
PROGRAM_SCHEMA = 'review-all-program/1'
NON_TERMINAL_STATUSES: tuple[str, ...] = tuple(sorted(str(s) for s in task_statuses.ACTIVE))
TERMINAL_STATUSES: tuple[str, ...] = tuple(sorted(str(s) for s in task_statuses.TERMINAL))


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
class FieldSpec:
    """The fields one kind of JSONL line carries (a ledger op, a ruling), and their vocabularies.

    A ledger op's spec excludes ``LEDGER_COMMON_FIELDS``; ``check_ledger_line`` adds them.
    """

    required: frozenset[str]
    optional: frozenset[str]
    enums: Mapping[str, type[enum.StrEnum]]


def _values(vocabulary: type[enum.StrEnum]) -> frozenset[str]:
    return frozenset(member.value for member in vocabulary)


def _field_spec(
    required: Sequence[str],
    optional: Sequence[str] = (),
    **enums: type[enum.StrEnum],
) -> FieldSpec:
    return FieldSpec(
        required=frozenset(required),
        optional=frozenset(optional),
        enums=types.MappingProxyType(enums),
    )


LEDGER_COMMON_FIELDS = frozenset({'seq', 'at', 'by', 'op'})
LEDGER_OPS: Mapping[LedgerOp, FieldSpec] = types.MappingProxyType({
    LedgerOp.STREAM_STATUS: _field_spec(
        ('stream', 'status'), ('prd_path', 'return'), status=StreamStatus,
    ),
    LedgerOp.AMEND_STREAM: _field_spec(
        ('stream', 'field', 'change', 'value', 'ruling'),
        field=StreamField, change=AmendChange,
    ),
    LedgerOp.AMEND_PROGRAM: _field_spec(('section', 'value'), ('ruling',), section=ProgramSection),
    LedgerOp.XREF: _field_spec(('from', 'to', 'text', 'return')),
    LedgerOp.ACTION: _field_spec(
        ('id', 'op', 'task', 'target', 'text', 'needs_human', 'return'), op=LeadActionOp,
    ),
    LedgerOp.ACTION_RESULT: _field_spec(
        ('id', 'state', 'batch', 'detail'), state=ActionResultState,
    ),
    LedgerOp.SCANNED: _field_spec(('return', 'repaths')),
    LedgerOp.NOTE: _field_spec(('text',)),
    LedgerOp.COMMIT: _field_spec(('sha', 'paths', 'measured_at')),
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
class Rejection:
    field: str
    kind: RejectionClass
    reason: str

    def __str__(self) -> str:
        return f'{self.kind}: {self.field}: {self.reason}'


class _Invalid:
    """The result of a field check that recorded a rejection."""


_INVALID = _Invalid()
_FieldCheck = Callable[[Any, str, list[Rejection]], Any]


def _reject(rejections: list[Rejection], field: str, reason: str) -> _Invalid:
    rejections.append(Rejection(field, RejectionClass.FORM, reason))
    return _INVALID


def _violations(rejections: Iterable[Rejection]) -> list[str]:
    """Rejections as ``<field>: <reason>`` text, for the errors that carry text."""
    return [f'{r.field}: {r.reason}' for r in rejections]


def _is_repo_relative(path: Any) -> bool:
    """A non-empty POSIX path that is not absolute and has no '..' segment."""
    if not isinstance(path, str) or not path or '\\' in path:
        return False
    posix = PurePosixPath(path)
    return not posix.is_absolute() and '..' not in posix.parts


def _of_type(kind: type, described: str) -> _FieldCheck:
    def check(value: Any, field: str, rejections: list[Rejection]) -> Any:
        if isinstance(value, kind) and (kind is bool or not isinstance(value, bool)):
            return value
        return _reject(rejections, field, f'expected {described}, got {value!r}')
    return check


_TEXT = _of_type(str, 'a string')
_BOOL = _of_type(bool, 'a bool')
_INT = _of_type(int, 'an integer')


def _vocabulary(vocabulary: type[enum.StrEnum]) -> _FieldCheck:
    def check(value: Any, field: str, rejections: list[Rejection]) -> Any:
        if isinstance(value, str) and value in _values(vocabulary):
            return vocabulary(value)
        return _reject(rejections, field, f'{value!r} is not one of {sorted(_values(vocabulary))}')
    return check


def _member(allowed: Iterable[Any], described: str) -> _FieldCheck:
    allowed = frozenset(allowed)

    def check(value: Any, field: str, rejections: list[Rejection]) -> Any:
        if isinstance(value, str | int) and not isinstance(value, bool) and value in allowed:
            return value
        return _reject(rejections, field, f'{value!r} is not {described}')
    return check


def _repo_path(value: Any, field: str, rejections: list[Rejection]) -> Any:
    if _is_repo_relative(value):
        return value
    return _reject(rejections, field, f'{value!r} is not a repo-relative POSIX path')


def _unchecked(value: Any, field: str, rejections: list[Rejection]) -> Any:
    return value


def _nullable(check: _FieldCheck) -> _FieldCheck:
    def nullable(value: Any, field: str, rejections: list[Rejection]) -> Any:
        return None if value is None else check(value, field, rejections)
    return nullable


def _list_of(check: _FieldCheck) -> _FieldCheck:
    def items(value: Any, field: str, rejections: list[Rejection]) -> Any:
        if not isinstance(value, list):
            return _reject(rejections, field, f'expected a list, got {value!r}')
        parsed = tuple(check(item, f'{field}[{i}]', rejections) for i, item in enumerate(value))
        return _INVALID if any(item is _INVALID for item in parsed) else parsed
    return items


_TEXTS = _list_of(_TEXT)


def _parsed_fields(
    value: Mapping[str, Any], field: str, required: Mapping[str, _FieldCheck],
    optional: Mapping[str, _FieldCheck], rejections: list[Rejection],
) -> dict[str, Any]:
    """Each present field checked at ``<field>.<name>`` (``<name>`` at the top level).

    A field outside *required* and *optional*, or a missing required one, is rejected.
    """
    def path(name: str) -> str:
        return f'{field}.{name}' if field else name

    for name in sorted(set(value) - set(required) - set(optional)):
        _reject(rejections, path(name), 'is not a field of this entry')
    for name in sorted(set(required) - set(value)):
        _reject(rejections, path(name), 'is required')
    checks = {**required, **optional}
    return {name: checks[name](value[name], path(name), rejections) for name in checks if name in value}


def _record(
    build: Callable[[dict[str, Any]], Any], required: Mapping[str, _FieldCheck],
    optional: Mapping[str, _FieldCheck] = types.MappingProxyType({}),
) -> _FieldCheck:
    """An object with exactly these fields, each checked at ``<field>.<name>``, then *build* it."""
    def record(value: Any, field: str, rejections: list[Rejection]) -> Any:
        if not isinstance(value, Mapping):
            return _reject(rejections, field, f'expected an object, got {value!r}')
        found = len(rejections)
        parsed = _parsed_fields(value, field, required, optional, rejections)
        return _INVALID if len(rejections) > found else build(parsed)
    return record


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


_STREAM = _record(lambda f: Stream(**f), {
    'id': _TEXT, 'name': _TEXT, 'slug': _TEXT, 'remedy': _TEXT, 'deletable': _BOOL,
    'keys': _TEXTS, 'mode': _vocabulary(StreamMode), 'size': _vocabulary(StreamSize),
    'risk': _vocabulary(StreamRisk), 'deps': _TEXTS, 'seams': _TEXTS,
})
_PROGRAM_FIELDS: Mapping[str, _FieldCheck] = types.MappingProxyType({
    'schema': _member({PROGRAM_SCHEMA}, repr(PROGRAM_SCHEMA)),
    **{name: _TEXT for name in (
        'run_id', 'project_id', 'project_root', 'as_of_sha', 'record_path', 'doc_path',
        'plans_dir',
    )},
    **{section.value: _list_of(_unchecked) for section in ProgramSection},
    'streams': _list_of(_unchecked),
})


class _StreamRow(NamedTuple):
    """The stream fields the cross-stream rules read, and where the row was read from."""

    where: str
    id: str
    keys: tuple[str, ...]
    deps: tuple[str, ...]


def _duplicate_id_violations(rows: Sequence[_StreamRow]) -> list[str]:
    seen: set[str] = set()
    violations = []
    for row in rows:
        if row.id in seen:
            violations.append(f'{row.where}.id: duplicate stream id {row.id!r}')
        seen.add(row.id)
    return violations


def _shared_key_violations(rows: Sequence[_StreamRow]) -> list[str]:
    owners: dict[str, list[str]] = {}
    for row in rows:
        for key in row.keys:
            owners.setdefault(key, []).append(row.id)
    return [
        f'key {key!r} is in more than one stream: {", ".join(ids)}'
        for key, ids in owners.items() if len(ids) > 1
    ]


def _unknown_dep_violations(rows: Sequence[_StreamRow], known: Collection[object]) -> list[str]:
    return [
        f'{row.where}.deps[{d}]: {dep!r} names no stream'
        for row in rows for d, dep in enumerate(row.deps) if dep not in known
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


def _cycle_violations(rows: Sequence[_StreamRow]) -> list[str]:
    ids = {row.id for row in rows}
    try:
        _wave_numbers({row.id: [d for d in row.deps if d in ids] for row in rows})
    except StateError as cycle:
        return [str(cycle)]
    return []


def _cross_stream_violations(rows: Sequence[_StreamRow], known: Collection[object]) -> list[str]:
    """The rules streams obey together, checked at parse and again after the ledger's amendments."""
    return (
        _duplicate_id_violations(rows)
        + _shared_key_violations(rows)
        + _unknown_dep_violations(rows, known)
        + _cycle_violations(rows)
    )


def parse_program(data: Any) -> Program:
    """Validate program.json's parsed object into a frozen Program; raises ProgramInvalid.

    Every violation is collected and named by its path and value, not only the first:
    each stream is checked on its own, and one that fails its own checks still counts
    as a known id for the others' deps.
    """
    if not isinstance(data, Mapping):
        raise ProgramInvalid([f'program.json must be a JSON object, got {data!r}'])
    rejections: list[Rejection] = []
    fields = _parsed_fields(data, '', _PROGRAM_FIELDS, {}, rejections)
    raw_streams = fields.pop('streams', ())
    raw_streams = () if raw_streams is _INVALID else raw_streams
    checked = [_STREAM(raw, f'streams[{i}]', rejections) for i, raw in enumerate(raw_streams)]
    streams = [(i, stream) for i, stream in enumerate(checked) if stream is not _INVALID]
    violations = _violations(rejections) + _cross_stream_violations(
        [_StreamRow(f'streams[{i}]', s.id, s.keys, s.deps) for i, s in streams],
        {raw.get('id') for raw in raw_streams if isinstance(raw, Mapping)},
    )
    if violations:
        raise ProgramInvalid(violations)
    del fields['schema']
    return Program(
        **{name: _frozen(value) for name, value in fields.items()},
        streams=tuple(stream for _, stream in streams),
    )


@dataclass(frozen=True)
class TaskRow:
    """One task as the store reports it; metadata is a read-only copy."""

    task_id: int
    title: str
    status: str
    priority: str
    updated_at: str
    metadata: Mapping[str, Any]
    dependencies: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'metadata', types.MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, 'dependencies', tuple(self.dependencies))


@dataclass(frozen=True)
class StreamView:
    """A stream row after the ledger's amendments, with its status and authored PRD path."""

    id: str
    keys: tuple[str, ...]
    deps: tuple[str, ...]
    deletable: bool
    mode: StreamMode
    remedy: str
    status: StreamStatus | None
    prd_path: str | None
    authored: bool


@dataclass(frozen=True)
class _AmendRule:
    changes: frozenset[AmendChange]
    convert: Callable[[Any], Any]


def _checked(kind: type) -> Callable[[Any], Any]:
    def convert(value: Any) -> Any:
        if not isinstance(value, kind):
            raise ValueError(f'expected {kind.__name__}, got {value!r}')
        return value
    return convert


_LIST_CHANGES = frozenset({AmendChange.ADD, AmendChange.REMOVE})
_SCALAR_CHANGES = frozenset({AmendChange.SET})
_AMEND_RULES: Mapping[str, _AmendRule] = types.MappingProxyType({
    StreamField.KEYS: _AmendRule(_LIST_CHANGES, _checked(str)),
    StreamField.DEPS: _AmendRule(_LIST_CHANGES, _checked(str)),
    StreamField.DELETABLE: _AmendRule(_SCALAR_CHANGES, _checked(bool)),
    StreamField.MODE: _AmendRule(_SCALAR_CHANGES, StreamMode),
    StreamField.REMEDY: _AmendRule(_SCALAR_CHANGES, _checked(str)),
})


def _amended(current: Any, change: str, value: Any) -> Any:
    if change == AmendChange.ADD:
        return current if value in current else (*current, value)
    if change == AmendChange.REMOVE:
        return tuple(item for item in current if item != value)
    return value


def _apply_amendment(view: StreamView, line: Mapping[str, Any]) -> StreamView:
    field, change = line.get('field'), line.get('change')
    rule = _AMEND_RULES.get(str(field))
    if rule is None or change not in rule.changes:
        allowed = sorted(rule.changes) if rule else sorted(_AMEND_RULES)
        raise StateError(
            f'ledger seq {line.get("seq")}: amend_stream field {field!r} does not take '
            f'change {change!r} (allowed: {", ".join(allowed)})'
        )
    try:
        value = rule.convert(line.get('value'))
    except ValueError as bad:
        raise StateError(f'ledger seq {line.get("seq")}: amend_stream {field} value: {bad}') from bad
    return replace(view, **{str(field): _amended(getattr(view, str(field)), str(change), value)})


def _apply_status(view: StreamView, line: Mapping[str, Any]) -> StreamView:
    try:
        status = StreamStatus(line.get('status'))
    except ValueError as bad:
        raise StateError(f'ledger seq {line.get("seq")}: stream_status {bad}') from bad
    return replace(
        view,
        status=status,
        prd_path=line.get('prd_path', view.prd_path),
        authored=view.authored or status is StreamStatus.AUTHORED,
    )


_STREAM_FOLDS: Mapping[str, Callable[[StreamView, Mapping[str, Any]], StreamView]] = (
    types.MappingProxyType({
        LedgerOp.STREAM_STATUS: _apply_status,
        LedgerOp.AMEND_STREAM: _apply_amendment,
    })
)


def _initial_view(stream: Stream) -> StreamView:
    return StreamView(
        id=stream.id, keys=stream.keys, deps=stream.deps, deletable=stream.deletable,
        mode=stream.mode, remedy=stream.remedy, status=None, prd_path=None, authored=False,
    )


def stream_views(program: Program, ledger: Iterable[Mapping[str, Any]]) -> Mapping[str, StreamView]:
    """Each stream's row after the ledger's ``stream_status`` and ``amend_stream`` lines, in seq order.

    The single home of a stream's keys and deps after amendments. The folded rows
    must still obey parse_program's cross-stream rules; StateError names each breach.
    """
    views = {stream.id: _initial_view(stream) for stream in program.streams}
    for line in sorted(ledger, key=lambda entry: entry['seq']):
        fold = _STREAM_FOLDS.get(line.get('op', ''))
        if fold is None:
            continue
        stream_id = line.get('stream')
        if stream_id not in views:
            raise StateError(f'ledger seq {line["seq"]}: {line["op"]} names no stream {stream_id!r}')
        views[stream_id] = fold(views[stream_id], line)
    violations = _cross_stream_violations(
        [_StreamRow(v.id, v.id, v.keys, v.deps) for v in views.values()], views,
    )
    if violations:
        raise StateError("the ledger's amendments break the program: " + '; '.join(violations))
    return types.MappingProxyType(views)


def waves(views: Mapping[str, StreamView]) -> Mapping[str, int]:
    """Each stream's wave, 1 + the max of its upstream streams' waves; never stored."""
    return types.MappingProxyType(_wave_numbers({v.id: v.deps for v in views.values()}))


UNSTREAMED = 'unstreamed'


def _finding_keys(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    keys = metadata.get('x_finding_key')
    if isinstance(keys, str):
        return (keys,)
    if isinstance(keys, list | tuple):
        return tuple(k for k in keys if isinstance(k, str))
    return ()


def stream_of(row: TaskRow, views: Mapping[str, StreamView]) -> str:
    """The stream a program task belongs to: its prd_path, else its finding keys, else UNSTREAMED."""
    prd_path = row.metadata.get('prd_path')
    for view in views.values():
        if prd_path is not None and view.prd_path == prd_path:
            return view.id
    keys = _finding_keys(row.metadata)
    hits = sorted({view.id for view in views.values() if set(keys) & set(view.keys)})
    if len(hits) > 1:
        raise AmbiguousStream(
            f'task {row.task_id}: finding keys {list(keys)} belong to streams {", ".join(hits)}'
        )
    return hits[0] if hits else UNSTREAMED


_PATH_SEGMENT = re.compile(r'[A-Za-z0-9._-]+')


def _segment(text: str, what: str) -> str:
    """*text* when it is one safe path segment; StateError naming it otherwise."""
    if not isinstance(text, str) or not _PATH_SEGMENT.fullmatch(text) or text in {'.', '..'}:
        raise StateError(f'{what} {text!r} is not one path segment of [A-Za-z0-9._-], nor . or ..')
    return text


def _pass_stem(stream: str, pass_: Pass | str, item: str | None) -> str:
    """The one naming rule for per-pass files: ``<S>.<pass>`` or ``<S>.investigate-<item>``."""
    pass_ = Pass(pass_)
    _segment(stream, 'stream')
    if pass_ is Pass.INVESTIGATE:
        if item is None:
            raise StateError(f'{stream}.investigate files are named by the item they investigate')
        return f'{stream}.investigate-{_segment(item, "item")}'
    if item is not None:
        raise StateError(f'{stream}.{pass_}: item {item!r} names only an investigate pass')
    return f'{stream}.{pass_}'


class BatchFiles(NamedTuple):
    actions: Path
    prompt: Path
    result: Path


@dataclass(frozen=True)
class Home:
    """The run's working state home and every file and directory of its layout."""

    root: Path

    @property
    def program_json(self) -> Path:
        return self.root / 'program.json'

    @property
    def rulings(self) -> Path:
        return self.root / 'rulings.jsonl'

    @property
    def ledger(self) -> Path:
        return self.root / 'ledger.jsonl'

    @property
    def sessions_json(self) -> Path:
        return self.root / 'sessions.json'

    @property
    def handoff(self) -> Path:
        return self.root / 'HANDOFF.md'

    @property
    def render_state_dir(self) -> Path:
        return self.root / 'render' / '.state'

    def render_doc(self, program: Program) -> Path:
        return self.root / 'render' / PurePosixPath(program.doc_path).name

    def prds_dir(self, stream: str) -> Path:
        return self.root / 'prds' / _segment(stream, 'stream')

    def inbox_file(self, stream: str, pass_: Pass | str, item: str | None = None) -> Path:
        return self.root / 'inbox' / f'{_pass_stem(stream, pass_, item)}.json'

    def returns_file(self, stream: str, pass_: Pass | str, item: str | None = None) -> Path:
        return self.root / 'returns' / f'{_pass_stem(stream, pass_, item)}.json'

    def brief_file(self, stream: str, pass_: Pass | str, item: str | None = None) -> Path:
        return self.root / 'briefs' / f'{_pass_stem(stream, pass_, item)}.md'

    def handback_file(
        self, stream: str, pass_: Pass | str, item: str | None = None,
        checkpoint: int | None = None,
    ) -> Path:
        suffix = '' if checkpoint is None else f'.checkpoint-{checkpoint}'
        return self.root / 'handbacks' / f'{_pass_stem(stream, pass_, item)}{suffix}.md'

    def batch_files(self, number: int) -> BatchFiles:
        actions = self.root / 'actions'
        return BatchFiles(
            actions / f'batch-{number}.json',
            actions / f'batch-{number}.prompt.md',
            actions / f'batch-{number}.result.json',
        )


def homes_root(environ: Mapping[str, str] = os.environ) -> Path:
    """``$REVIEW_ALL_HOME``, which must be absolute, else ``~/.claude/review-all``."""
    configured = environ.get('REVIEW_ALL_HOME')
    if configured is None:
        return Path.home() / '.claude' / 'review-all'
    if not Path(configured).is_absolute():
        raise StateError(f'REVIEW_ALL_HOME must be an absolute path, got {configured!r}')
    return Path(configured)


def run_home(run_id: str, environ: Mapping[str, str] = os.environ) -> Home:
    return Home(homes_root(environ) / _segment(run_id, 'run_id'))


def published_doc(program: Program) -> Path:
    return Path(program.project_root) / program.doc_path


def snapshot_dir(program: Program) -> Path:
    return published_doc(program).with_suffix('.state')


def fused_memory_url(
    environ: Mapping[str, str] = os.environ, checkout: Path = CHECKOUT_ROOT,
) -> str:
    """The fused-memory MCP endpoint: ``$FUSED_MEMORY_URL``, else the checkout's ``.mcp.json`` entry."""
    if environ.get('FUSED_MEMORY_URL'):
        return environ['FUSED_MEMORY_URL']
    mcp_json = checkout / '.mcp.json'
    try:
        url = json.loads(mcp_json.read_text())['mcpServers']['fused-memory']['url']
    except (OSError, ValueError, KeyError, TypeError):
        url = None
    if not isinstance(url, str) or not url:
        raise StateError(
            f'no fused-memory URL: FUSED_MEMORY_URL is unset and {mcp_json} carries no '
            "mcpServers['fused-memory'].url"
        )
    return url


def now_utc() -> str:
    """The one clock for ``at`` and ``measured_at``: ISO-8601 UTC, second precision."""
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


@dataclass(frozen=True)
class ItemRef:
    """A design question of one stream, written ``<stream>:<question id>``."""

    stream: str
    question: str

    @classmethod
    def parse(cls, text: str) -> ItemRef:
        parts = text.split(':') if isinstance(text, str) else []
        if len(parts) != 2 or not all(parts):
            raise StateError(f'item ref {text!r} is not <stream>:<question id>')
        return cls(parts[0], parts[1])

    def __str__(self) -> str:
        return f'{self.stream}:{self.question}'


_EntryCheck = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], None]


def _check_fields(spec: FieldSpec, line: Mapping[str, Any], what: str) -> None:
    """Refuse a line *spec* does not describe, naming every field at fault."""
    def checks(names: frozenset[str]) -> dict[str, _FieldCheck]:
        return {n: _vocabulary(spec.enums[n]) if n in spec.enums else _unchecked for n in names}

    rejections: list[Rejection] = []
    _parsed_fields(line, '', checks(spec.required), checks(spec.optional), rejections)
    if rejections:
        raise LedgerRefused(f'{what}: ' + '; '.join(_violations(rejections)))


def _parse_jsonl(text: str, path: Path) -> list[Mapping[str, Any]]:
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        try:
            line = json.loads(raw)
        except ValueError as bad:
            raise LedgerRefused(f'{path} line {number} is not JSON: {bad}') from bad
        if not isinstance(line, dict):
            raise LedgerRefused(f'{path} line {number} is not a JSON object: {raw!r}')
        lines.append(line)
    return lines


def _accept_each(
    prior: Sequence[Mapping[str, Any]], lines: Sequence[Mapping[str, Any]],
    check_entry: _EntryCheck, where: Callable[[int], str],
) -> None:
    """Check each line against everything before it; a refusal names the line by *where*."""
    accepted = list(prior)
    for number, line in enumerate(lines, 1):
        try:
            if not isinstance(line, Mapping):
                raise LedgerRefused(f'expected a JSON object, got {line!r}')
            check_entry(accepted, line)
        except LedgerRefused as refused:
            raise LedgerRefused(f'{where(number)}: {refused}') from None
        accepted.append(line)


def _read_checked(path: Path, check_entry: _EntryCheck) -> tuple[Mapping[str, Any], ...]:
    lines = _parse_jsonl(path.read_text(encoding='utf-8'), path) if path.exists() else []
    _accept_each((), lines, check_entry, lambda number: f'{path} line {number}')
    return tuple(_frozen(line) for line in lines)


def _locked_append(
    path: Path, lines: Sequence[Mapping[str, Any]], check_entry: _EntryCheck,
) -> None:
    """Append *lines* under an exclusive flock, all or nothing, after checking file and batch.

    ``a+`` gives O_APPEND, so an existing line can never be rewritten.
    """
    with path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = _parse_jsonl(handle.read(), path)
            _accept_each((), existing, check_entry, lambda number: f'{path} line {number}')
            _accept_each(existing, lines, check_entry, lambda number: f'appended line {number}')
            handle.write(''.join(json.dumps(line) + '\n' for line in lines))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def check_ledger_line(line: Mapping[str, Any]) -> None:
    """Refuse a ledger line its op does not describe (LEDGER_OPS plus the common fields)."""
    op = line.get('op')
    spec = LEDGER_OPS.get(op) if isinstance(op, str) else None
    if spec is None:
        raise LedgerRefused(f'unknown ledger op {op!r} (known: {", ".join(LEDGER_OPS)})')
    with_common = replace(spec, required=spec.required | LEDGER_COMMON_FIELDS)
    _check_fields(with_common, line, f'{op} line')
    seq = line['seq']
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise LedgerRefused(f'{op} line: seq must be an int, got {seq!r}')


def next_seq(existing: Sequence[Mapping[str, Any]]) -> int:
    return len(existing) + 1


def _check_ledger_entry(prior: Sequence[Mapping[str, Any]], line: Mapping[str, Any]) -> None:
    check_ledger_line(line)
    if line['seq'] != next_seq(prior):
        raise LedgerRefused(f'{line["op"]} line: expected seq {next_seq(prior)}, got {line["seq"]}')


def append_ledger(home: Home, lines: Sequence[Mapping[str, Any]]) -> None:
    """Append ledger lines whose seqs continue the file densely; refuses the whole batch otherwise."""
    _locked_append(home.ledger, lines, _check_ledger_entry)


def read_ledger(home: Home) -> tuple[Mapping[str, Any], ...]:
    return _read_checked(home.ledger, _check_ledger_entry)


RULING_FIELDS = _field_spec(
    ('id', 'ruled_by', 'date', 'stream', 'item', 'question', 'ruling', 'options',
     'consequence', 'keys', 'source', 'supersedes'),
    ('answers', 'reask'),
    ruled_by=RuledBy,
)


def next_ruling_id(rulings: Sequence[Mapping[str, Any]], ruled_by: RuledBy | str) -> str:
    """``D<n>`` for a human ruling, ``L<n>`` for a lead's; one counter across both prefixes."""
    prefix = 'D' if RuledBy(ruled_by) is RuledBy.HUMAN else 'L'
    return f'{prefix}{len(rulings) + 1}'


def _check_answers(answers: Any) -> None:
    if not isinstance(answers, list | tuple):
        raise LedgerRefused(f'answers must be a list of <stream>:<question id>, got {answers!r}')
    for entry in answers:
        try:
            ItemRef.parse(entry)
        except StateError as bad:
            raise LedgerRefused(f'answers entry: {bad}') from None


def check_ruling_line(line: Mapping[str, Any]) -> None:
    what = f'ruling {line.get("id")!r}'
    _check_fields(RULING_FIELDS, line, what)
    if line['ruled_by'] == RuledBy.HUMAN:
        for field in ('options', 'consequence'):
            if not line[field]:
                raise LedgerRefused(f'{what}: a human ruling needs a non-empty {field}')
    if 'answers' in line:
        _check_answers(line['answers'])
    if not isinstance(line.get('reask', False), bool):
        raise LedgerRefused(f'{what}: reask must be a bool, got {line["reask"]!r}')


def _check_ruling_entry(prior: Sequence[Mapping[str, Any]], line: Mapping[str, Any]) -> None:
    check_ruling_line(line)
    expected = next_ruling_id(prior, line['ruled_by'])
    if line['id'] != expected:
        raise LedgerRefused(f'expected id {expected!r}, got {line["id"]!r}')
    supersedes = line['supersedes']
    if supersedes is not None and supersedes not in {r['id'] for r in prior}:
        raise LedgerRefused(f'ruling {line["id"]!r} supersedes {supersedes!r}, which is no earlier ruling')


def append_rulings(home: Home, lines: Sequence[Mapping[str, Any]]) -> None:
    """Append rulings whose ids are next_ruling_id in turn; refuses the whole batch otherwise."""
    _locked_append(home.rulings, lines, _check_ruling_entry)


def read_rulings(home: Home) -> tuple[Mapping[str, Any], ...]:
    return _read_checked(home.rulings, _check_ruling_entry)


MCP_CLIENT_NAME = 'review-all-program'
_TASK_NOT_FOUND = 'TaskNotFoundError'
def _undeclared_argument(argument: str) -> re.Pattern[str]:
    """FastMCP's refusal of an argument its tool does not declare: pydantic's extra_forbidden."""
    return re.compile(rf'^{re.escape(argument)}\n\s+Extra inputs are not permitted', re.MULTILINE)


_CAPABILITY_REFUSALS: Mapping[str, re.Pattern[str]] = types.MappingProxyType({
    'include_rows': _undeclared_argument('include_rows'),
})
FIND_PAGE_LIMIT = 100


class _ToolRefused(StoreReaderError):
    """A tool answered with a failure; ``error_type`` is the payload's, when it carried one."""

    def __init__(self, tool: str, text: str, error_type: str | None) -> None:
        self.error_type = error_type
        super().__init__(f'{tool} failed: {text}')


def _decode_body(body: bytes, content_type: str, url: str) -> Mapping[str, Any]:
    """A JSON reply, or the first ``data:`` line of an SSE body (FusedMemoryClient's reading)."""
    text = body.decode('utf-8', errors='replace')
    if 'text/event-stream' in content_type:
        data = [line[len('data:'):] for line in text.splitlines() if line.startswith('data:')]
        text = data[0] if data else ''
    try:
        reply = json.loads(text)
    except ValueError:
        raise StoreReaderError(f'{url} answered an unparseable body: {text[:200]!r}') from None
    if not isinstance(reply, Mapping):
        raise StoreReaderError(f'{url} answered a non-object reply: {text[:200]!r}')
    return reply


class _McpSession:
    """``scripts/migrate_metadata_modules_to_files.py::FusedMemoryClient``'s wire shape over urllib.

    *url* is the endpoint form (``http://127.0.0.1:8002/mcp``). The handshake is
    lazy and the ``mcp-session-id`` is minted once.
    """

    def __init__(self, url: str, timeout: float) -> None:
        self._url = url
        self._timeout = timeout
        self._session_id = uuid.uuid4().hex
        self._initialized = False
        self._request_id = 1

    def _post(self, message: Mapping[str, Any]) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self._url, data=json.dumps(message).encode(), method='POST',
            headers={**mcp_post.MCP_POST_HEADERS, 'mcp-session-id': self._session_id},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                status, content_type = response.status, response.headers.get('Content-Type', '')
                body = response.read()
        except urllib.error.HTTPError as refused:
            raise StoreReaderError(f'{self._url} answered HTTP {refused.code}') from None
        except (urllib.error.URLError, OSError) as failed:
            raise StoreReaderError(f'cannot reach {self._url}: {failed}') from None
        if status == 202 or not body:
            return {}
        return _decode_body(body, content_type, self._url)

    def _handshake(self) -> None:
        self._post({
            'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'clientInfo': {'name': MCP_CLIENT_NAME, 'version': '1.0'},
                'capabilities': {},
            },
        })
        self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}})
        self._initialized = True

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """The tool's payload; any failure raises through ``_raise_for_failure``."""
        if not self._initialized:
            self._handshake()
        self._request_id += 1
        reply = self._post(
            mcp_post.mcp_tool_call_payload(name, dict(arguments), request_id=self._request_id)
        )
        _raise_for_failure(name, reply, arguments)
        return _reply_payload(name, reply)


def _first_text(result: Mapping[str, Any]) -> str | None:
    for entry in result.get('content') or ():
        if isinstance(entry, Mapping) and entry.get('type') == 'text':
            return str(entry.get('text'))
    return None


def _reply_payload(tool: str, reply: Mapping[str, Any]) -> Mapping[str, Any]:
    """structuredContent, else the first text content parsed as JSON."""
    result = reply.get('result')
    if not isinstance(result, Mapping):
        raise StoreReaderError(f'{tool}: the reply carries no result: {str(reply)[:200]}')
    structured = result.get('structuredContent')
    if isinstance(structured, Mapping):
        return structured
    text = _first_text(result)
    try:
        payload = json.loads(text) if text is not None else None
    except ValueError:
        payload = None
    if not isinstance(payload, Mapping):
        raise StoreReaderError(f'{tool}: the reply carries no readable payload: {str(result)[:200]}')
    return payload


def _failure(tool: str, reply: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """The failure text and error_type a reply carries, at any of its three layers; else None."""
    error = reply.get('error')
    if error is not None:
        return str(error.get('message', error) if isinstance(error, Mapping) else error), None
    result = reply.get('result')
    if isinstance(result, Mapping) and result.get('isError'):
        return _first_text(result) or str(result), None
    payload = _reply_payload(tool, reply)
    if 'error' in payload:
        return str(payload['error']), payload.get('error_type')
    return None


def _raise_for_failure(tool: str, reply: Mapping[str, Any], sent_args: Mapping[str, Any]) -> None:
    """The one boundary classifier: an undeclared-argument refusal of a capability we sent is INV-11's.

    Any other failure, including one whose text mentions the argument, is the tool's.
    """
    failure = _failure(tool, reply)
    if failure is None:
        return
    text, error_type = failure
    for argument, refusal in _CAPABILITY_REFUSALS.items():
        if argument in sent_args and refusal.search(text):
            raise StoreReaderUnsupported(argument, f'{tool} does not declare the argument: {text}')
    raise _ToolRefused(tool, text, error_type)


@dataclass(frozen=True)
class StoreRead:
    """One reader result: the rows, when the store was read, and the ids re-read in full."""

    rows: tuple[TaskRow, ...]
    measured_at: str
    reread: tuple[int, ...] = ()


def _strict_int(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise StoreReaderError(f'{what} is not an integer: {value!r}')
    return value


def _task_row(raw: Mapping[str, Any], id_key: str, updated_key: str) -> TaskRow:
    task_id = _strict_int(raw.get(id_key), f'task id {raw.get(id_key)!r}')
    try:
        return TaskRow(
            task_id=task_id, title=raw['title'], status=raw['status'],
            priority=raw['priority'], updated_at=raw[updated_key],
            metadata=raw['metadata'] if isinstance(raw['metadata'], Mapping) else {},
            dependencies=tuple(
                _strict_int(d, f'task {task_id} dependency') for d in raw['dependencies']
            ),
        )
    except KeyError as missing:
        raise StoreReaderError(f'task {task_id}: the store row lacks {missing}') from None


def _require_rows(page: Mapping[str, Any]) -> None:
    """INV-11: the server must echo include_rows and send metadata and dependencies on every row."""
    if page.get('include_rows') is not True:
        raise StoreReaderUnsupported(
            'include_rows',
            f'find_tasks_by_metadata sent no include_rows: true echo '
            f'(got {page.get("include_rows", "no echo")!r}); the server predates the widening',
        )
    for raw in page.get('results', ()):
        for key in ('metadata', 'dependencies'):
            if key not in raw:
                raise StoreReaderUnsupported(
                    key, f'find_tasks_by_metadata sent task {raw.get("task_id")} without {key!r}',
                )


class _Found(NamedTuple):
    rows: tuple[TaskRow, ...]
    reread: tuple[int, ...]


def _union(found: Iterable[_Found], measured_at: str) -> StoreRead:
    rows: dict[int, TaskRow] = {}
    reread: set[int] = set()
    for part in found:
        rows.update((row.task_id, row) for row in part.rows)
        reread.update(part.reread)
    return StoreRead(tuple(rows[i] for i in sorted(rows)), measured_at, tuple(sorted(reread)))


def _trigger_chain_of(row: TaskRow) -> tuple[Any, Any]:
    chain = row.metadata.get('trigger_chain')
    return (chain.get('run_id'), chain.get('role')) if isinstance(chain, Mapping) else (None, None)


class StoreReader:
    """The only code that reads the task store: fused-memory MCP tools, never sqlite, never a write."""

    def __init__(
        self, server_url: str, project_root: str, *,
        report: TextIO | None = None, timeout: float = 30.0,
    ) -> None:
        self._session = _McpSession(server_url, timeout)
        self._project_root = project_root
        self._report = report

    def _get_task(self, task_id: int) -> TaskRow | None:
        arguments = {'id': str(task_id), 'project_root': self._project_root}
        try:
            payload = self._session.call_tool('get_task', arguments)
        except _ToolRefused as refused:
            if refused.error_type == _TASK_NOT_FOUND:
                return None
            raise
        return _task_row(payload, 'id', 'updatedAt')

    def _reread(self, raw: Mapping[str, Any], cache: dict[int, TaskRow]) -> TaskRow:
        task_id = _strict_int(raw.get('task_id'), 'truncated row id')
        if task_id not in cache:
            row = self._get_task(task_id)
            if row is None:
                raise StoreReaderError(f'task {task_id} was listed truncated but get_task finds none')
            cache[task_id] = row
            report = self._report if self._report is not None else sys.stderr
            print(f'store reader: task {task_id} metadata is over the page budget; '
                  're-read in full with get_task', file=report)
        return cache[task_id]

    def _find(
        self, key: str, value: str | int | None = None, statuses: Sequence[str] | None = None,
        *, cache: dict[int, TaskRow],
    ) -> _Found:
        """Every row of one lookup, paged on has_more, with truncated rows re-read in full."""
        rows: list[TaskRow] = []
        reread: list[int] = []
        offset = 0
        while True:
            arguments: dict[str, Any] = {
                'project_root': self._project_root, 'key': key,
                'limit': FIND_PAGE_LIMIT, 'offset': offset, 'include_rows': True,
            }
            if value is not None:
                arguments['value'] = value
            if statuses is not None:
                arguments['statuses'] = list(statuses)
            page = self._session.call_tool('find_tasks_by_metadata', arguments)
            _require_rows(page)
            for raw in page['results']:
                if raw.get('truncated_row'):
                    rows.append(self._reread(raw, cache))
                    reread.append(rows[-1].task_id)
                else:
                    rows.append(_task_row(raw, 'task_id', 'updated_at'))
            pagination = page['pagination']
            if not pagination['has_more']:
                return _Found(tuple(rows), tuple(reread))
            if pagination['returned'] == 0:
                raise StoreReaderError(
                    f'find_tasks_by_metadata {key}={value!r} reported has_more with no rows at '
                    f'offset {offset}'
                )
            offset += pagination['returned']

    def program_tasks(self, run_id: str, prd_paths: Iterable[str]) -> StoreRead:
        """Tasks carrying x_finding_run = run_id, unioned with tasks whose prd_path is a stream's."""
        measured_at = now_utc()
        cache: dict[int, TaskRow] = {}
        found = [self._find('x_finding_run', run_id, cache=cache)]
        found += [self._find('prd_path', path, cache=cache) for path in prd_paths]
        return _union(found, measured_at)

    def held_leaves(self, run_id: str, prd_paths: Iterable[str]) -> StoreRead:
        program = self.program_tasks(run_id, prd_paths)
        held = tuple(
            row for row in program.rows
            if SEAM_WAIT_KEYS.key in row.metadata and SEAM_RELEASED_KEYS.key not in row.metadata
        )
        return replace(program, rows=held)

    def _chain_task(self, run_id: str, role: TriggerChainRole) -> StoreRead:
        measured_at = now_utc()
        found = self._find('trigger_chain', statuses=NON_TERMINAL_STATUSES, cache={})
        rows = tuple(r for r in found.rows if _trigger_chain_of(r) == (run_id, role))
        if len(rows) > 1:
            raise StoreReaderError(
                f'{len(rows)} {role} tasks carry trigger_chain.run_id = {run_id}: '
                f'{[r.task_id for r in rows]}'
            )
        return StoreRead(rows, measured_at, found.reread)

    def gate(self, run_id: str) -> StoreRead:
        return self._chain_task(run_id, TriggerChainRole.HUMAN_GATE)

    def completion_gate(self, run_id: str) -> StoreRead:
        return self._chain_task(run_id, TriggerChainRole.COMPLETION_GATE)

    def tasks_declaring_files(self, paths: Iterable[str]) -> StoreRead:
        measured_at = now_utc()
        cache: dict[int, TaskRow] = {}
        return _union(
            [self._find('files', path, NON_TERMINAL_STATUSES, cache=cache) for path in paths],
            measured_at,
        )

    def tasks_with_delivered_checks(self) -> StoreRead:
        measured_at = now_utc()
        return _union([self._find('delivered_checks', None, NON_TERMINAL_STATUSES, cache={})],
                      measured_at)

    def task(self, task_id: int) -> StoreRead:
        measured_at = now_utc()
        row = self._get_task(task_id)
        return StoreRead(() if row is None else (row,), measured_at)


class StoreView(Protocol):
    """The store reads validate_return needs; StoreReader provides them."""

    def task(self, task_id: int) -> StoreRead: ...

    def gate(self, run_id: str) -> StoreRead: ...


def open_reader(program: Program, environ: Mapping[str, str] = os.environ) -> StoreReader:
    return StoreReader(fused_memory_url(environ), program.project_root)


NOTES_CAP = 600


@dataclass(frozen=True)
class Session:
    kind: SessionKind
    title: str


@dataclass(frozen=True)
class OpenQuestion:
    id: str
    question: str
    options: tuple[str, ...]
    recommendation: str
    needs_human: bool
    source: QuestionSource
    lead_rulable: bool = False


@dataclass(frozen=True)
class SessionRuling:
    id: str
    question: str
    ruling: str
    options: tuple[str, ...]
    consequence: str


@dataclass(frozen=True)
class SeamDependency:
    stream: str
    contract: str


@dataclass(frozen=True)
class Xref:
    to: str | int
    text: str


@dataclass(frozen=True)
class LeadAction:
    op: LeadActionOp
    task: int
    text: str
    needs_human: bool
    target: Any = None


@dataclass(frozen=True)
class Filed:
    label: str
    task_id: int
    priority: Priority
    keys: tuple[str, ...]
    gate_dep: bool
    adopted: bool


@dataclass(frozen=True)
class Held:
    label: str
    task_id: int
    waits_on: str
    contract: str
    release_owner: str


@dataclass(frozen=True)
class Released:
    task_id: int
    onto: tuple[int, ...]
    contract_as_delivered: str


@dataclass(frozen=True)
class CommitPlanning:
    committed: tuple[int, ...]
    excluded: tuple[int, ...]
    stamping_errors: tuple[Any, ...]


@dataclass(frozen=True)
class OtherFiled:
    task_id: int
    why: str


@dataclass(frozen=True)
class Move:
    from_path: str
    to_path: str | None
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class PrdReturn:
    """One parsed return block; the keys its pass lacks are None or empty."""

    schema: str
    run_id: str
    stream: str
    pass_: Pass
    session: Session
    item: str | None = None
    prd_path: str | None = None
    manifest_path: str | None = None
    sidecar_path: str | None = None
    commits: tuple[str, ...] = ()
    open_questions: tuple[OpenQuestion, ...] = ()
    rulings_in_session: tuple[SessionRuling, ...] = ()
    seam_dependencies: tuple[SeamDependency, ...] = ()
    xrefs: tuple[Xref, ...] = ()
    lead_actions: tuple[LeadAction, ...] = ()
    notes: str | None = None
    touches: tuple[str, ...] = ()
    filed: tuple[Filed, ...] = ()
    held: tuple[Held, ...] = ()
    released: tuple[Released, ...] = ()
    commit_planning: CommitPlanning | None = None
    other_filed: tuple[OtherFiled, ...] = ()
    moves: tuple[Move, ...] = ()


@dataclass(frozen=True)
class PassKeys:
    required: frozenset[str]
    optional: frozenset[str]
    non_empty: frozenset[str] = frozenset()


_COMMON_RETURN_KEYS = frozenset({'schema', 'run_id', 'stream', 'pass', 'session'})
_AUTHOR_OPTIONAL_KEYS = frozenset({
    'manifest_path', 'sidecar_path', 'commits', 'open_questions', 'rulings_in_session',
    'seam_dependencies', 'xrefs', 'lead_actions', 'notes',
})
RETURN_KEYS: Mapping[Pass, PassKeys] = types.MappingProxyType({
    Pass.AUTHOR: PassKeys(_COMMON_RETURN_KEYS | {'prd_path', 'touches'}, _AUTHOR_OPTIONAL_KEYS),
    Pass.DECOMPOSE: PassKeys(
        _COMMON_RETURN_KEYS | {'prd_path', 'filed', 'moves'},
        _AUTHOR_OPTIONAL_KEYS | {'held', 'released', 'commit_planning', 'other_filed'},
    ),
    Pass.INVESTIGATE: PassKeys(
        _COMMON_RETURN_KEYS | {'item', 'rulings_in_session'}, frozenset({'notes'}),
        non_empty=frozenset({'rulings_in_session'}),
    ),
})


class TargetKind(enum.StrEnum):
    IDS = 'ids'
    KEYS = 'keys'
    PATH = 'path'
    NONE = 'none'


LEAD_ACTION_TARGETS: Mapping[LeadActionOp, TargetKind] = types.MappingProxyType({
    LeadActionOp.CANCEL: TargetKind.IDS,
    LeadActionOp.EDGE: TargetKind.IDS,
    LeadActionOp.RELEASE: TargetKind.IDS,
    LeadActionOp.STAMP: TargetKind.KEYS,
    LeadActionOp.REPATH: TargetKind.PATH,
    LeadActionOp.NOTE: TargetKind.NONE,
    LeadActionOp.AMEND: TargetKind.NONE,
})


@dataclass(frozen=True)
class ReturnCheck:
    rejections: tuple[Rejection, ...]
    value: PrdReturn | None
    measured_at: str | None

    @property
    def ok(self) -> bool:
        return not self.rejections


def _capped_notes(value: Any, field: str, rejections: list[Rejection]) -> Any:
    if _TEXT(value, field, rejections) is _INVALID:
        return _INVALID
    if len(value) > NOTES_CAP:
        return _reject(rejections, field, f'{len(value)} characters is over the {NOTES_CAP} cap')
    return value


def _no_commits(value: Any, field: str, rejections: list[Rejection]) -> Any:
    if value != []:
        return _reject(rejections, field, f'seats and sessions never commit; expected [], got {value!r}')
    return ()


def _ids_target(value: Any, field: str, rejections: list[Rejection]) -> Any:
    ids = [value] if isinstance(value, int) and not isinstance(value, bool) else value
    return _list_of(_INT)(ids, field, rejections)


def _no_target(value: Any, field: str, rejections: list[Rejection]) -> Any:
    return None if value is None else _reject(rejections, field, f'takes no target, got {value!r}')


_TARGET_CHECKS: Mapping[TargetKind, _FieldCheck] = types.MappingProxyType({
    TargetKind.IDS: _ids_target,
    TargetKind.KEYS: _TEXTS,
    TargetKind.PATH: _repo_path,
    TargetKind.NONE: _no_target,
})

_LEAD_ACTION_FIELDS = _record(
    lambda f: LeadAction(**f),
    {'op': _vocabulary(LeadActionOp), 'task': _INT, 'text': _TEXT, 'needs_human': _BOOL},
    {'target': _unchecked},
)


def _lead_action(value: Any, field: str, rejections: list[Rejection]) -> Any:
    action = _LEAD_ACTION_FIELDS(value, field, rejections)
    if action is _INVALID:
        return _INVALID
    target = _TARGET_CHECKS[LEAD_ACTION_TARGETS[action.op]](
        action.target, f'{field}.target', rejections,
    )
    return _INVALID if target is _INVALID else replace(action, target=target)


def _return_checks(program: Program) -> Mapping[str, _FieldCheck]:
    """The check for every top-level return key; RETURN_KEYS decides which a pass may carry."""
    stream_id = _member(program.stream_ids, f'a stream of {program.run_id}')
    return {
        'schema': _member({RETURN_SCHEMA}, f'{RETURN_SCHEMA!r}'),
        'run_id': _member({program.run_id}, f'this program\'s run_id {program.run_id!r}'),
        'stream': stream_id,
        'pass': _vocabulary(Pass),
        'session': _record(lambda f: Session(**f),
                           {'kind': _vocabulary(SessionKind), 'title': _TEXT}),
        'item': _TEXT,
        'prd_path': _repo_path,
        'manifest_path': _repo_path,
        'sidecar_path': _repo_path,
        'commits': _no_commits,
        'open_questions': _list_of(_record(
            lambda f: OpenQuestion(**f),
            {'id': _TEXT, 'question': _TEXT, 'options': _TEXTS, 'recommendation': _TEXT,
             'needs_human': _BOOL, 'source': _vocabulary(QuestionSource)},
            {'lead_rulable': _BOOL},
        )),
        'rulings_in_session': _list_of(_record(
            lambda f: SessionRuling(**f),
            {'id': _TEXT, 'question': _TEXT, 'ruling': _TEXT, 'options': _TEXTS,
             'consequence': _TEXT},
        )),
        'seam_dependencies': _list_of(_record(
            lambda f: SeamDependency(**f), {'stream': stream_id, 'contract': _TEXT},
        )),
        'xrefs': _list_of(_record(
            lambda f: Xref(**f),
            {'to': _xref_target(program), 'text': _TEXT},
        )),
        'lead_actions': _list_of(_lead_action),
        'notes': _capped_notes,
        'touches': _list_of(_repo_path),
        'filed': _list_of(_record(
            lambda f: Filed(**f),
            {'label': _TEXT, 'task_id': _INT, 'priority': _vocabulary(Priority),
             'keys': _TEXTS, 'gate_dep': _BOOL, 'adopted': _BOOL},
        )),
        'held': _list_of(_record(
            lambda f: Held(**f),
            {'label': _TEXT, 'task_id': _INT, 'waits_on': stream_id, 'contract': _TEXT,
             'release_owner': _TEXT},
        )),
        'released': _list_of(_record(
            lambda f: Released(**f),
            {'task_id': _INT, 'onto': _list_of(_INT), 'contract_as_delivered': _TEXT},
        )),
        'commit_planning': _record(
            lambda f: CommitPlanning(**f),
            {'committed': _list_of(_INT), 'excluded': _list_of(_INT),
             'stamping_errors': _list_of(_unchecked)},
        ),
        'other_filed': _list_of(_record(
            lambda f: OtherFiled(**f), {'task_id': _INT, 'why': _TEXT},
        )),
        'moves': _list_of(_record(
            lambda f: Move(from_path=f['from'], to_path=f['to'], symbols=f['symbols']),
            {'from': _repo_path, 'to': _nullable(_repo_path), 'symbols': _TEXTS},
        )),
    }


def _xref_target(program: Program) -> _FieldCheck:
    streams = _member({*program.stream_ids, 'program'}, 'a stream id, "program" or a task id')

    def check(value: Any, field: str, rejections: list[Rejection]) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return streams(value, field, rejections)
    return check


def _key_set_rejections(data: Mapping[str, Any], pass_: Pass) -> list[Rejection]:
    keys = RETURN_KEYS[pass_]
    rejections: list[Rejection] = []
    for key in sorted(set(data) - keys.required - keys.optional):
        _reject(rejections, key, f'is not a key of a {pass_} return, even when empty')
    for key in sorted(keys.required - set(data)):
        _reject(rejections, key, f'is required on a {pass_} return')
    for key in sorted(keys.non_empty & set(data)):
        if data[key] == []:
            _reject(rejections, key, f'must not be empty on a {pass_} return')
    return rejections


def _check_form(data: Any, program: Program) -> tuple[list[Rejection], PrdReturn | None]:
    """Every form rejection of one parsed return, else the typed PrdReturn."""
    if not isinstance(data, Mapping):
        return [Rejection('<file>', RejectionClass.FORM, 'is not a JSON object')], None
    checks = _return_checks(program)
    rejections: list[Rejection] = []
    pass_ = checks['pass'](data.get('pass'), 'pass', rejections)
    if pass_ is _INVALID:
        return rejections, None
    rejections += _key_set_rejections(data, pass_)
    parsed = {key: checks[key](data[key], key, rejections) for key in data if key in checks}
    if rejections:
        return rejections, None
    fields = {('pass_' if key == 'pass' else key): value for key, value in parsed.items()}
    return [], PrdReturn(**fields)


def validate_return(
    text: str, *, program: Program, ledger: Sequence[Mapping[str, Any]], store: StoreView,
) -> ReturnCheck:
    """Validate one return block: form first; a form-valid return is then checked against the store."""
    try:
        data = json.loads(text)
    except ValueError as bad:
        return ReturnCheck((Rejection('<file>', RejectionClass.FORM, f'does not parse: {bad}'),),
                           None, None)
    rejections, value = _check_form(data, program)
    if value is None:
        return ReturnCheck(tuple(rejections), None, None)
    if value.pass_ is not Pass.DECOMPOSE:
        return ReturnCheck((), value, None)
    contradictions, measured_at = _check_store(value, stream_views(program, ledger), store)
    return ReturnCheck(tuple(contradictions), None if contradictions else value, measured_at)


class _StoreReads:
    """One validation's store reads: each id is read once, and measured_at is the earliest read's."""

    def __init__(self, store: StoreView) -> None:
        self._store = store
        self._rows: dict[int, TaskRow | None] = {}
        self.measured_at: str | None = None

    def _noted(self, read: StoreRead) -> TaskRow | None:
        self.measured_at = min(filter(None, (self.measured_at, read.measured_at)))
        return read.rows[0] if read.rows else None

    def task(self, task_id: int) -> TaskRow | None:
        if task_id not in self._rows:
            self._rows[task_id] = self._noted(self._store.task(task_id))
        return self._rows[task_id]

    def gate(self, run_id: str) -> TaskRow | None:
        return self._noted(self._store.gate(run_id))


def _contradiction(field: str, reason: str) -> Rejection:
    return Rejection(field, RejectionClass.CONTRADICTION, reason)


def _authored_rejections(ret: PrdReturn, view: StreamView) -> list[Rejection]:
    if view.authored:
        return []
    return [_contradiction(
        'pass', f'{ret.stream} author return is not applied: the ledger has no '
        f'stream_status authored line for {ret.stream}',
    )]


def _filed_key_rejections(ret: PrdReturn, view: StreamView) -> list[Rejection]:
    return [
        _contradiction(f'filed[{i}].keys[{k}]',
                       f'{key!r} is not a key of {ret.stream} after amendments: {list(view.keys)}')
        for i, filed in enumerate(ret.filed) for k, key in enumerate(filed.keys)
        if key not in view.keys
    ]


def _filed_rejections(ret: PrdReturn, reads: _StoreReads) -> list[Rejection]:
    rejections = []
    for i, filed in enumerate(ret.filed):
        field, row = f'filed[{i}].task_id', reads.task(filed.task_id)
        if row is None:
            rejections.append(_contradiction(field, f'task {filed.task_id} is absent from the store'))
            continue
        run = row.metadata.get('x_finding_run')
        if run != ret.run_id:
            rejections.append(_contradiction(
                field, f'task {filed.task_id} carries x_finding_run {run!r}, not {ret.run_id!r}'))
        prd_path = row.metadata.get('prd_path')
        if not filed.adopted and prd_path != ret.prd_path:
            rejections.append(_contradiction(
                field, f'task {filed.task_id} carries prd_path {prd_path!r}, not {ret.prd_path!r}, '
                'and adopted is false'))
    return rejections


def _gate_dep_rejections(ret: PrdReturn, reads: _StoreReads) -> list[Rejection]:
    claimed = [(i, filed.task_id) for i, filed in enumerate(ret.filed) if filed.gate_dep]
    if not claimed:
        return []
    gate = reads.gate(ret.run_id)
    if gate is None:
        reason = f'no human_gate task carries trigger_chain.run_id = {ret.run_id}'
        return [_contradiction(f'filed[{i}].gate_dep', reason) for i, _ in claimed]
    return [
        _contradiction(f'filed[{i}].gate_dep', f'task {task_id} is not a dependency of gate '
                       f'{gate.task_id}: {list(gate.dependencies)}')
        for i, task_id in claimed if task_id not in gate.dependencies
    ]


def _held_rejections(ret: PrdReturn, reads: _StoreReads) -> list[Rejection]:
    rejections = []
    for i, held in enumerate(ret.held):
        field, row = f'held[{i}].task_id', reads.task(held.task_id)
        if row is None:
            rejections.append(_contradiction(field, f'task {held.task_id} is absent from the store'))
            continue
        if row.status != task_statuses.TaskStatus.DEFERRED:
            rejections.append(_contradiction(
                field, f'task {held.task_id} is {row.status}, not {task_statuses.TaskStatus.DEFERRED}'))
        wait = row.metadata.get(SEAM_WAIT_KEYS.key)
        if not isinstance(wait, Mapping):
            rejections.append(_contradiction(
                field, f'task {held.task_id} carries no {SEAM_WAIT_KEYS.key}'))
        elif wait.get('stream') != held.waits_on:
            rejections.append(_contradiction(
                f'held[{i}].waits_on', f'the store records {SEAM_WAIT_KEYS.key}.stream '
                f'{wait.get("stream")!r} for task {held.task_id}, not {held.waits_on!r}'))
    return rejections


def _released_rejections(ret: PrdReturn, reads: _StoreReads) -> list[Rejection]:
    rejections = []
    for i, released in enumerate(ret.released):
        field, row = f'released[{i}].task_id', reads.task(released.task_id)
        if row is None:
            rejections.append(_contradiction(field, f'task {released.task_id} is absent from the store'))
            continue
        if SEAM_RELEASED_KEYS.key not in row.metadata:
            rejections.append(_contradiction(
                field, f'task {released.task_id} carries no {SEAM_RELEASED_KEYS.key}'))
        rejections += [
            _contradiction(f'released[{i}].onto[{j}]', f'task {onto} is not a store dependency of '
                           f'task {released.task_id}: {list(row.dependencies)}')
            for j, onto in enumerate(released.onto) if onto not in row.dependencies
        ]
    return rejections


def _check_store(
    ret: PrdReturn, views: Mapping[str, StreamView], store: StoreView,
) -> tuple[list[Rejection], str | None]:
    """A decompose return's claims the store or the ledger contradicts, and when the store was read."""
    reads = _StoreReads(store)
    view = views[ret.stream]
    rejections = [
        *_authored_rejections(ret, view),
        *_filed_key_rejections(ret, view),
        *_filed_rejections(ret, reads),
        *_gate_dep_rejections(ret, reads),
        *_held_rejections(ret, reads),
        *_released_rejections(ret, reads),
    ]
    return rejections, reads.measured_at
