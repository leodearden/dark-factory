"""Program state for a /review-all Phase 6: the one library every program script imports.

The design-time statement is ``plans/review-all-program-tooling-prd.md`` §4
(H1–H3); this module is the machine home it names.
"""
from __future__ import annotations

import enum
import fcntl
import importlib
import json
import os
import re
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

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


def _check_view_deps(views: Mapping[str, StreamView]) -> None:
    for view in views.values():
        unknown = [dep for dep in view.deps if dep not in views]
        if unknown:
            raise StateError(f'stream {view.id} depends on {unknown}, which name no stream')


def stream_views(program: Program, ledger: Iterable[Mapping[str, Any]]) -> Mapping[str, StreamView]:
    """Each stream's row after the ledger's ``stream_status`` and ``amend_stream`` lines, in seq order.

    The single home of a stream's keys and deps after amendments.
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
    _check_view_deps(views)
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


def _check_fields(spec: LedgerOpSpec, line: Mapping[str, Any], what: str) -> None:
    missing = sorted(spec.required - set(line))
    if missing:
        raise LedgerRefused(f'{what} lacks {", ".join(missing)}')
    extra = sorted(set(line) - spec.required - spec.optional)
    if extra:
        raise LedgerRefused(f'{what} carries fields it does not take: {", ".join(extra)}')
    for field, allowed in spec.enums.items():
        value = line[field]
        if not isinstance(value, str) or value not in allowed:
            raise LedgerRefused(f'{what}: {field} {value!r} is not one of {sorted(allowed)}')


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


RULING_FIELDS = _ledger_op(
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
