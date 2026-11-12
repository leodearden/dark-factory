"""quality.db's single home: its schema, the LandingRow codec over ``landings``, value-or-gap
samples, and each project's stored marker derivation.

Design: plans/quality-dashboard-prd.md decision 14, §Contract "Per-landing row" and §Contract
"Markers".
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, NamedTuple

import aiosqlite
from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import (
    GitCrossCheck,
    Landing,
    LandingSource,
    LinesBase,
    parse_utc,
)
from shared.quality_series.rows import (
    CharterClass,
    CharterClassSource,
    Cost,
    LandingRow,
    Outcomes,
    ReviewRoundsSource,
    SourceEpochs,
)

from dashboard.data.quality_markers import (
    AxisLanding,
    DerivedState,
    Marker,
    MarkerKind,
    MarkerSource,
    PromptCoverage,
    RoutingFold,
    RoutingStep,
    RuleSpan,
    SourceLayer,
    StoredMarkerState,
)

__all__ = [
    'QUALITY_SCHEMA',
    'NewestSamples',
    'QualitySample',
    'SampleState',
    'ensure_marker_tables',
    'insert_sample',
    'landing_axis',
    'newest_samples',
    'read_landing_rows',
    'read_marker_state',
    'read_routing_fold',
    'read_samples',
    'record_marker_failure',
    'stored_cursor',
    'upsert_landing_rows',
    'write_marker_state',
]

_MARKERS_DERIVED_DDL = """\
CREATE TABLE IF NOT EXISTS markers_derived (
    project_id  TEXT    NOT NULL,
    marker_id   TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    at          TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    ref         TEXT,
    PRIMARY KEY (project_id, marker_id)
);
"""

QUALITY_SCHEMA = """\
CREATE TABLE IF NOT EXISTS landings (
    project_id              TEXT    NOT NULL,
    merge_sha               TEXT    NOT NULL,
    task_id                 TEXT    NOT NULL,
    landed_at               TEXT    NOT NULL,
    ordinal                 INTEGER NOT NULL,
    source                  TEXT    NOT NULL,
    unit_id                 TEXT    NOT NULL,
    unit_size               INTEGER NOT NULL,
    late_claim              INTEGER NOT NULL,
    git_first_parent        INTEGER NOT NULL,
    prev_landed_at          TEXT,
    lines_base              TEXT    NOT NULL,
    base_sha                TEXT,
    rejected                INTEGER,
    rejection_events        INTEGER,
    reviewed                INTEGER,
    review_blocking         INTEGER,
    flake_confirmed         INTEGER,
    flake_real_red          INTEGER,
    flake_rows              INTEGER,
    reverify_count          INTEGER,
    verify_failed_rows      INTEGER,
    all_role_usd            REAL,
    by_role                 TEXT    NOT NULL,
    implementer_invocations INTEGER,
    execute_iterations      INTEGER,
    turns                   INTEGER,
    turns_capped            INTEGER NOT NULL,
    code_lines              INTEGER,
    src_lines               INTEGER,
    test_lines              INTEGER,
    review_rounds           INTEGER,
    review_rounds_source    TEXT,
    charter_class           TEXT,
    charter_class_source    TEXT,
    modules                 TEXT    NOT NULL,
    rule_vintage            TEXT    NOT NULL,
    sampled_at              TEXT    NOT NULL,
    PRIMARY KEY (project_id, merge_sha, task_id)
);
CREATE INDEX IF NOT EXISTS idx_landings_project_ordinal ON landings(project_id, ordinal);

CREATE TABLE IF NOT EXISTS late_events (
    project_id  TEXT NOT NULL,
    merge_sha   TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    event_sha   TEXT NOT NULL,
    at          TEXT NOT NULL,
    PRIMARY KEY (project_id, merge_sha, task_id, kind, event_sha)
);

CREATE TABLE IF NOT EXISTS routing_steps (
    project_id      TEXT    NOT NULL,
    role            TEXT    NOT NULL,
    first_event_id  INTEGER NOT NULL,
    first_at        TEXT    NOT NULL,
    last_event_id   INTEGER NOT NULL,
    last_at         TEXT    NOT NULL,
    decisions       INTEGER NOT NULL,
    source_layer    TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    effort          TEXT    NOT NULL,
    budget_usd      REAL    NOT NULL,
    max_turns       INTEGER NOT NULL,
    prompt_digest   TEXT,
    PRIMARY KEY (project_id, first_event_id)
);

CREATE TABLE IF NOT EXISTS routing_rules (
    project_id      TEXT    NOT NULL,
    rule_id         TEXT    NOT NULL,
    first_event_id  INTEGER NOT NULL,
    first_at        TEXT    NOT NULL,
    last_at         TEXT    NOT NULL,
    decisions       INTEGER NOT NULL,
    PRIMARY KEY (project_id, rule_id)
);

CREATE TABLE IF NOT EXISTS marker_state (
    project_id          TEXT    PRIMARY KEY,
    routing_cursor      INTEGER,
    first_decision_at   TEXT,
    newest_decision_at  TEXT,
    digest_since        TEXT,
    fold_complete       INTEGER,
    skipped_routing     INTEGER,
    skipped_derived     INTEGER,
    roles_py_landings   INTEGER,
    roles_py_since      TEXT,
    roles_py_reason     TEXT,
    refreshed_at        TEXT,
    attempted_at        TEXT    NOT NULL,
    failure             TEXT
);

CREATE TABLE IF NOT EXISTS sample_state (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id              TEXT    NOT NULL,
    ts                      TEXT    NOT NULL,
    state                   TEXT    NOT NULL,
    reason                  TEXT,
    cursor_sha              TEXT,
    landings_sampled        INTEGER,
    landings_deferred       INTEGER,
    off_first_parent        INTEGER,
    merges_without_landing  INTEGER,
    late_claims             INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sample_state_project_ts ON sample_state(project_id, ts);
""" + _MARKERS_DERIVED_DDL


def _utc_text(at: datetime) -> str:
    """Fixed-width UTC ISO text, so a lexical comparison in SQL is a time comparison."""
    return at.astimezone(UTC).isoformat(timespec='microseconds')


class _Codec(NamedTuple):
    encode: Callable[[Any], object]
    decode: Callable[[Any], Any]


def _nullable(encode: Callable[[Any], object], decode: Callable[[Any], Any]) -> _Codec:
    return _Codec(
        lambda value: None if value is None else encode(value),
        lambda value: None if value is None else decode(value),
    )


def _enum(cls: type[StrEnum]) -> _Codec:
    return _nullable(lambda member: member.value, cls)


_PLAIN = _Codec(lambda value: value, lambda value: value)
_BOOL = _nullable(int, bool)
_UTC = _nullable(_utc_text, parse_utc)
_BY_ROLE = _Codec(
    lambda by_role: json.dumps([[role, usd] for role, usd in by_role]),
    lambda text: tuple((str(role), float(usd)) for role, usd in json.loads(text)),
)
_MODULES = _Codec(
    lambda modules: json.dumps([[m.module, m.src_lines] for m in modules]),
    lambda text: tuple(ModuleLines(str(module), int(lines)) for module, lines in json.loads(text)),
)
def _decode_epochs(text: str) -> SourceEpochs:
    """A stored rule vintage; a source added since the row was stored reads as not yet started."""
    stored: dict[str, Any] = json.loads(text)
    names = [f.name for f in fields(SourceEpochs)]
    unknown = sorted(set(stored) - set(names))
    if unknown:
        raise ValueError(f'stored rule_vintage names sources SourceEpochs lacks: {unknown}')
    return SourceEpochs(**{name: _UTC.decode(stored.get(name)) for name in names})


_EPOCHS = _Codec(
    lambda epochs: json.dumps(
        {f.name: _UTC.encode(getattr(epochs, f.name)) for f in fields(SourceEpochs)}
    ),
    _decode_epochs,
)


@dataclass(frozen=True)
class _Part:
    """One dataclass of a LandingRow, stored as one column per field named as the field."""

    attribute: str
    cls: Callable[..., Any]
    codecs: tuple[tuple[str, _Codec], ...]


_PARTS: tuple[_Part, ...] = (
    _Part('landing', Landing, (
        ('project_id', _PLAIN), ('merge_sha', _PLAIN), ('task_id', _PLAIN),
        ('landed_at', _UTC), ('ordinal', _PLAIN), ('source', _enum(LandingSource)),
        ('unit_id', _PLAIN), ('unit_size', _PLAIN), ('late_claim', _BOOL),
        ('git_first_parent', _BOOL), ('prev_landed_at', _UTC),
        ('lines_base', _enum(LinesBase)), ('base_sha', _PLAIN),
    )),
    _Part('outcomes', Outcomes, (
        ('rejected', _BOOL), ('rejection_events', _PLAIN), ('reviewed', _BOOL),
        ('review_blocking', _BOOL), ('flake_confirmed', _BOOL), ('flake_real_red', _BOOL),
        ('flake_rows', _PLAIN), ('reverify_count', _PLAIN), ('verify_failed_rows', _PLAIN),
    )),
    _Part('cost', Cost, (
        ('all_role_usd', _PLAIN), ('by_role', _BY_ROLE), ('implementer_invocations', _PLAIN),
        ('execute_iterations', _PLAIN), ('turns', _PLAIN), ('turns_capped', _BOOL),
        ('code_lines', _PLAIN), ('src_lines', _PLAIN), ('test_lines', _PLAIN),
        ('review_rounds', _PLAIN), ('review_rounds_source', _enum(ReviewRoundsSource)),
        ('charter_class', _enum(CharterClass)),
        ('charter_class_source', _enum(CharterClassSource)),
    )),
)

_WHOLE_FIELDS: tuple[tuple[str, _Codec], ...] = (('modules', _MODULES), ('rule_vintage', _EPOCHS))


def _require_every_field_coded(cls: Any, coded: Iterable[str]) -> None:
    """Refuse a codec that misses or invents a field of *cls*, so a new α field fails at import."""
    declared = {f.name for f in fields(cls)}
    named = set(coded)
    if named != declared:
        raise TypeError(
            f'the quality.db codec for {cls.__name__} is out of step with its fields: '
            f'uncoded {sorted(declared - named)}, not fields {sorted(named - declared)}'
        )


def _require_the_codec_covers_landing_row() -> None:
    for part in _PARTS:
        _require_every_field_coded(part.cls, (name for name, _ in part.codecs))
    _require_every_field_coded(
        LandingRow, (*(part.attribute for part in _PARTS), *(name for name, _ in _WHOLE_FIELDS)),
    )


_require_the_codec_covers_landing_row()

_KEY = ('project_id', 'merge_sha', 'task_id')

_ROW_COLUMNS: tuple[str, ...] = (
    *(name for part in _PARTS for name, _ in part.codecs),
    *(name for name, _ in _WHOLE_FIELDS),
)
"""Every column a LandingRow fills, in the order the INSERT and the SELECT both use."""


def _encode_row(row: LandingRow, sampled_at: str) -> tuple[object, ...]:
    values = [
        codec.encode(getattr(getattr(row, part.attribute), name))
        for part in _PARTS
        for name, codec in part.codecs
    ]
    values.extend(codec.encode(getattr(row, name)) for name, codec in _WHOLE_FIELDS)
    return (*values, sampled_at)


def _decode_row(record: Mapping[str, Any]) -> LandingRow:
    """A stored row as α's dataclasses, whose __post_init__ re-checks every invariant."""
    parts = {
        part.attribute: part.cls(**{
            name: codec.decode(record[name]) for name, codec in part.codecs
        })
        for part in _PARTS
    }
    whole = {name: codec.decode(record[name]) for name, codec in _WHOLE_FIELDS}
    return LandingRow(**parts, **whole)


_UPSERT_SQL = (
    f'INSERT INTO landings ({", ".join((*_ROW_COLUMNS, "sampled_at"))}) '
    f'VALUES ({", ".join("?" for _ in (*_ROW_COLUMNS, "sampled_at"))}) '
    f'ON CONFLICT({", ".join(_KEY)}) DO UPDATE SET '
    + ', '.join(
        f'{column} = excluded.{column}'
        for column in (*_ROW_COLUMNS, 'sampled_at')
        if column not in _KEY
    )
)

_READ_ROWS_SQL = (
    f'SELECT {", ".join(_ROW_COLUMNS)} FROM landings '
    'WHERE project_id = ? ORDER BY ordinal, task_id'
)


async def upsert_landing_rows(
    conn: aiosqlite.Connection, rows: Iterable[LandingRow], *, sampled_at: datetime,
) -> None:
    """Write *rows*, replacing any stored row with the same key; the caller commits."""
    stamp = _utc_text(sampled_at)
    await conn.executemany(_UPSERT_SQL, [_encode_row(row, stamp) for row in rows])


def _decode_rows(records: Iterable[Iterable[Any]]) -> list[LandingRow]:
    return [_decode_row(dict(zip(_ROW_COLUMNS, record, strict=True))) for record in records]


async def read_landing_rows(conn: aiosqlite.Connection, project_id: str) -> list[LandingRow]:
    """The project's stored rows in ordinal order, decoded off the event loop."""
    async with conn.execute(_READ_ROWS_SQL, (project_id,)) as cursor:
        records = await cursor.fetchall()
    return await asyncio.to_thread(_decode_rows, records)


async def stored_cursor(
    conn: aiosqlite.Connection, project_id: str, *, landed_by: datetime | None = None,
) -> str | None:
    """The merge_sha of the newest stored first claim on main, landed by *landed_by* if given."""
    sql = (
        'SELECT merge_sha FROM landings'
        ' WHERE project_id = ? AND git_first_parent = 1 AND late_claim = 0'
    )
    params: list[object] = [project_id]
    if landed_by is not None:
        sql += ' AND landed_at <= ?'
        params.append(_utc_text(landed_by))
    sql += ' ORDER BY ordinal DESC LIMIT 1'
    async with conn.execute(sql, params) as cursor:
        found = await cursor.fetchone()
    return None if found is None else found[0]


class SampleState(StrEnum):
    VALUE = 'value'
    GAP = 'gap'


def _invariant(condition: bool, invariant: str, **values: object) -> None:
    if not condition:
        shown = ', '.join(f'{name}={value!r}' for name, value in values.items())
        raise ValueError(f'{invariant} ({shown})')


def _is_count(value: int | None) -> bool:
    return value is not None and value >= 0


@dataclass(frozen=True)
class QualitySample:
    """One tick's outcome for one project: a measurement, or a gap saying why there is none.

    A gap carries no measurement at all, never zeros. A value's cursor_sha is None only while
    the project has no stored landing on main.
    """

    project_id: str
    ts: datetime
    state: SampleState
    reason: str | None
    cursor_sha: str | None
    landings_sampled: int | None
    landings_deferred: int | None
    git_check: GitCrossCheck | None

    def __post_init__(self) -> None:
        _invariant(self.ts.tzinfo is not None, 'a sample time carries its offset',
                   project_id=self.project_id, ts=self.ts)
        measurement = {
            'cursor_sha': self.cursor_sha,
            'landings_sampled': self.landings_sampled,
            'landings_deferred': self.landings_deferred,
            'git_check': self.git_check,
        }
        if self.state is SampleState.GAP:
            _invariant(bool(self.reason) and all(v is None for v in measurement.values()),
                       'a gap names a non-empty reason and carries no measurement',
                       project_id=self.project_id, reason=self.reason, **measurement)
        else:
            _invariant(self.state is SampleState.VALUE, 'a sample is a value or a gap',
                       project_id=self.project_id, state=self.state)
            _invariant(self.reason is None and _is_count(self.landings_sampled)
                       and _is_count(self.landings_deferred) and self.git_check is not None,
                       'a value carries no reason, non-negative counts and a git_check',
                       project_id=self.project_id, reason=self.reason, **measurement)

    @classmethod
    def value(
        cls, project_id: str, ts: datetime, *, cursor_sha: str | None, landings_sampled: int,
        landings_deferred: int, git_check: GitCrossCheck,
    ) -> QualitySample:
        return cls(project_id, ts, SampleState.VALUE, None, cursor_sha, landings_sampled,
                   landings_deferred, git_check)

    @classmethod
    def gap(cls, project_id: str, ts: datetime, reason: str) -> QualitySample:
        return cls(project_id, ts, SampleState.GAP, reason, None, None, None, None)


_GIT_CHECK_COLUMNS = ('off_first_parent', 'merges_without_landing', 'late_claims')
_SAMPLE_COLUMNS = (
    'project_id', 'ts', 'state', 'reason', 'cursor_sha', 'landings_sampled',
    'landings_deferred', *_GIT_CHECK_COLUMNS,
)


def _encode_git_check(check: GitCrossCheck | None) -> tuple[int | None, ...]:
    if check is None:
        return (None,) * len(_GIT_CHECK_COLUMNS)
    return tuple(getattr(check, name) for name in _GIT_CHECK_COLUMNS)


def _decode_git_check(
    off_first_parent: int | None, merges_without_landing: int | None, late_claims: int | None,
) -> GitCrossCheck | None:
    if off_first_parent is not None and merges_without_landing is not None and late_claims is not None:
        return GitCrossCheck(off_first_parent, merges_without_landing, late_claims)
    _invariant(off_first_parent is None and merges_without_landing is None and late_claims is None,
               'the git_check counts are stored together or not at all',
               off_first_parent=off_first_parent,
               merges_without_landing=merges_without_landing, late_claims=late_claims)
    return None


async def insert_sample(conn: aiosqlite.Connection, sample: QualitySample) -> None:
    """Append *sample*; the caller commits."""
    values = (
        sample.project_id, _utc_text(sample.ts), sample.state.value, sample.reason,
        sample.cursor_sha, sample.landings_sampled, sample.landings_deferred,
        *_encode_git_check(sample.git_check),
    )
    await conn.execute(
        f'INSERT INTO sample_state ({", ".join(_SAMPLE_COLUMNS)}) '
        f'VALUES ({", ".join("?" for _ in _SAMPLE_COLUMNS)})',
        values,
    )


def _decode_sample(record: Iterable[Any]) -> QualitySample:
    """A stored sample, re-checked by QualitySample's invariant."""
    (pid, ts, state, reason, cursor_sha, sampled, deferred,
     off_first_parent, merges_without_landing, late_claims) = record
    return QualitySample(
        project_id=pid, ts=parse_utc(ts), state=SampleState(state), reason=reason,
        cursor_sha=cursor_sha, landings_sampled=sampled, landings_deferred=deferred,
        git_check=_decode_git_check(off_first_parent, merges_without_landing, late_claims),
    )


_SELECT_SAMPLES = f'SELECT {", ".join(_SAMPLE_COLUMNS)} FROM sample_state WHERE project_id = ?'


async def read_samples(conn: aiosqlite.Connection, project_id: str) -> list[QualitySample]:
    """The project's samples in time order, each re-checked by QualitySample's invariant."""
    async with conn.execute(f'{_SELECT_SAMPLES} ORDER BY ts, id', (project_id,)) as cursor:
        records = await cursor.fetchall()
    return [_decode_sample(record) for record in records]


@dataclass(frozen=True)
class NewestSamples:
    """The project's newest sample of any state, and its newest value; None when it has none."""

    latest: QualitySample | None
    latest_value: QualitySample | None


_NEWEST_FIRST = 'ORDER BY ts DESC, id DESC'
_NEWEST_SAMPLES_SQL = (
    f'SELECT {", ".join(_SAMPLE_COLUMNS)} FROM sample_state WHERE id IN ('
    f'(SELECT id FROM sample_state WHERE project_id = :project {_NEWEST_FIRST} LIMIT 1), '
    f'(SELECT id FROM sample_state WHERE project_id = :project AND state = :value '
    f'{_NEWEST_FIRST} LIMIT 1)) {_NEWEST_FIRST}'
)


async def newest_samples(conn: aiosqlite.Connection, project_id: str) -> NewestSamples:
    """The newest sample and the newest value sample, in read_samples' order, without the rest.

    One statement reads both, so a tick committed between two reads cannot pair a superseded
    gap with a newer value.
    """
    params = {'project': project_id, 'value': SampleState.VALUE.value}
    async with conn.execute(_NEWEST_SAMPLES_SQL, params) as cursor:
        newest_first = [_decode_sample(record) for record in await cursor.fetchall()]
    return NewestSamples(
        latest=next(iter(newest_first), None),
        latest_value=next(
            (sample for sample in newest_first if sample.state is SampleState.VALUE), None,
        ),
    )


async def landing_axis(conn: aiosqlite.Connection, project_id: str) -> list[AxisLanding]:
    """The project's stored landings as (ordinal, landed_at), in read_landing_rows' order."""
    async with conn.execute(
        'SELECT ordinal, landed_at FROM landings WHERE project_id = ? ORDER BY ordinal, task_id',
        (project_id,),
    ) as cursor:
        return [AxisLanding(ordinal, parse_utc(landed_at)) for ordinal, landed_at in await cursor.fetchall()]


async def ensure_marker_tables(conn: aiosqlite.Connection) -> None:
    """Reshape a markers_derived from before the marker producer, which nothing ever wrote.

    Its rows are derived only and re-derived on the next marker tick, so the old table is
    dropped rather than migrated. The caller commits.
    """
    async with conn.execute(
        "SELECT 1 FROM pragma_table_info('markers_derived') WHERE name = 'marker_id'"
    ) as cursor:
        current = await cursor.fetchone() is not None
    if not current:
        await conn.execute('DROP TABLE IF EXISTS markers_derived')
        await conn.execute(_MARKERS_DERIVED_DDL)


@dataclass(frozen=True)
class _ProjectTable:
    """A frozen dataclass stored one row per value, one column per field, under a project_id."""

    name: str
    cls: Callable[..., Any]
    codecs: tuple[tuple[str, _Codec], ...]
    order_by: str

    def __post_init__(self) -> None:
        _require_every_field_coded(self.cls, (name for name, _ in self.codecs))

    @property
    def _columns(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.codecs)

    def insert_sql(self) -> str:
        columns = ('project_id', *self._columns)
        return (f'INSERT INTO {self.name} ({", ".join(columns)}) '
                f'VALUES ({", ".join("?" for _ in columns)})')

    def select_sql(self) -> str:
        return (f'SELECT {", ".join(self._columns)} FROM {self.name} '
                f'WHERE project_id = ? ORDER BY {self.order_by}')

    def encode(self, project_id: str, value: object) -> tuple[object, ...]:
        return (project_id, *(codec.encode(getattr(value, name)) for name, codec in self.codecs))

    def decode(self, record: Iterable[Any]) -> Any:
        return self.cls(**{
            name: codec.decode(stored)
            for (name, codec), stored in zip(self.codecs, record, strict=True)
        })


_STEPS = _ProjectTable('routing_steps', RoutingStep, (
    ('role', _PLAIN), ('first_event_id', _PLAIN), ('first_at', _UTC),
    ('last_event_id', _PLAIN), ('last_at', _UTC), ('decisions', _PLAIN),
    ('source_layer', _enum(SourceLayer)), ('model', _PLAIN), ('effort', _PLAIN), ('budget_usd', _PLAIN),
    ('max_turns', _PLAIN), ('prompt_digest', _PLAIN),
), order_by='first_event_id')

_RULES = _ProjectTable('routing_rules', RuleSpan, (
    ('rule_id', _PLAIN), ('first_event_id', _PLAIN), ('first_at', _UTC), ('last_at', _UTC),
    ('decisions', _PLAIN),
), order_by='first_event_id, rule_id')

_MARKER_COLUMNS = ('marker_id', 'kind', 'at', 'label', 'source', 'ref')


def _storable_marker(project_id: str, marker: Marker) -> bool:
    """A marker markers_derived holds whole: the project's own, unplaced, with no series or range."""
    return (marker.scope == project_id and marker.series is None and marker.until is None
            and not marker.tasks and marker.ordinal is None and marker.until_ordinal is None)


def _encode_marker(project_id: str, marker: Marker) -> tuple[object, ...]:
    _invariant(_storable_marker(project_id, marker),
               "a stored derived marker is its project's own, unplaced, with no series, "
               'until or tasks', project_id=project_id, marker_id=marker.marker_id,
               marker=marker)
    return (project_id, marker.marker_id, marker.kind.value, _utc_text(marker.at), marker.label,
            json.dumps(marker.source.to_wire(), sort_keys=True), marker.ref)


def _decode_marker(project_id: str, record: Iterable[Any]) -> Marker:
    marker_id, kind, at, label, source, ref = record
    return Marker(
        marker_id=marker_id, at=parse_utc(at), kind=MarkerKind(kind), scope=project_id,
        series=None, until=None, label=label, source=MarkerSource.from_wire(json.loads(source)),
        ref=ref,
    )


_INSERT_MARKER_SQL = (
    f'INSERT INTO markers_derived (project_id, {", ".join(_MARKER_COLUMNS)}) '
    f'VALUES ({", ".join("?" for _ in ("project_id", *_MARKER_COLUMNS))})'
)
_SELECT_MARKERS_SQL = (
    f'SELECT {", ".join(_MARKER_COLUMNS)} FROM markers_derived '
    'WHERE project_id = ? ORDER BY at, marker_id'
)

_STATE_COLUMNS = (
    'routing_cursor', 'first_decision_at', 'newest_decision_at', 'digest_since', 'fold_complete',
    'skipped_routing', 'skipped_derived', 'roles_py_landings', 'roles_py_since',
    'roles_py_reason', 'refreshed_at', 'attempted_at', 'failure',
)
_UPSERT_STATE_SQL = (
    f'INSERT INTO marker_state (project_id, {", ".join(_STATE_COLUMNS)}) '
    f'VALUES ({", ".join("?" for _ in ("project_id", *_STATE_COLUMNS))}) '
    'ON CONFLICT(project_id) DO UPDATE SET '
    + ', '.join(f'{column} = excluded.{column}' for column in _STATE_COLUMNS)
)
_SELECT_STATE_SQL = f'SELECT {", ".join(_STATE_COLUMNS)} FROM marker_state WHERE project_id = ?'


def _encode_state(state: DerivedState) -> tuple[object, ...]:
    fold, coverage = state.fold, state.prompt_coverage
    _invariant(coverage.digest_since == fold.digest_since,
               "the prompt coverage's digest_since is its fold's, stored once",
               project_id=fold.project_id, coverage_digest_since=coverage.digest_since,
               fold_digest_since=fold.digest_since)
    return (
        fold.project_id, fold.cursor, _UTC.encode(fold.first_at), _UTC.encode(fold.newest_at),
        _UTC.encode(fold.digest_since), _BOOL.encode(fold.complete), fold.skipped, state.skipped,
        coverage.roles_py_landings, _UTC.encode(coverage.counted_since), coverage.reason,
        _utc_text(state.refreshed_at), _utc_text(state.refreshed_at), None,
    )


_DERIVED_TABLES = ('routing_steps', 'routing_rules', 'markers_derived')


async def write_marker_state(conn: aiosqlite.Connection, state: DerivedState) -> None:
    """Replace the stored derivation of *state*'s project with *state*; the caller commits.

    Clears any recorded failure, since this derivation is the newest attempt.
    """
    project_id = state.fold.project_id
    state_row = _encode_state(state)
    markers = [_encode_marker(project_id, marker) for marker in state.markers]
    for table in _DERIVED_TABLES:
        await conn.execute(f'DELETE FROM {table} WHERE project_id = ?', (project_id,))
    await conn.executemany(_STEPS.insert_sql(), [_STEPS.encode(project_id, step) for step in state.fold.steps])
    await conn.executemany(_RULES.insert_sql(), [_RULES.encode(project_id, span) for span in state.fold.rules])
    await conn.executemany(_INSERT_MARKER_SQL, markers)
    await conn.execute(_UPSERT_STATE_SQL, state_row)


async def record_marker_failure(
    conn: aiosqlite.Connection, project_id: str, *, attempted_at: datetime, failure: str,
) -> None:
    """Record a failed marker refresh, keeping the newest good derivation; the caller commits."""
    await conn.execute(
        'INSERT INTO marker_state (project_id, attempted_at, failure) VALUES (?, ?, ?) '
        'ON CONFLICT(project_id) DO UPDATE SET '
        'attempted_at = excluded.attempted_at, failure = excluded.failure',
        (project_id, _utc_text(attempted_at), failure),
    )


@asynccontextmanager
async def _one_snapshot(conn: aiosqlite.Connection) -> AsyncIterator[None]:
    """Every read inside sees one database state: a commit on another connection cannot tear it,
    but a commit on this one while the snapshot is open ends the savepoint, so never snapshot-read
    on a connection another coroutine may commit on."""
    await conn.execute('SAVEPOINT read_marker_state')
    try:
        yield
    finally:
        await conn.execute('RELEASE read_marker_state')


async def _fetch(conn: aiosqlite.Connection, sql: str, project_id: str) -> list[Any]:
    async with conn.execute(sql, (project_id,)) as cursor:
        return list(await cursor.fetchall())


async def _fetch_state(conn: aiosqlite.Connection, project_id: str) -> dict[str, Any] | None:
    found = await _fetch(conn, _SELECT_STATE_SQL, project_id)
    return dict(zip(_STATE_COLUMNS, found[0], strict=True)) if found else None


async def _fetch_fold_rows(
    conn: aiosqlite.Connection, project_id: str,
) -> tuple[list[Any], list[Any]]:
    return (await _fetch(conn, _STEPS.select_sql(), project_id),
            await _fetch(conn, _RULES.select_sql(), project_id))


def _decode_fold(
    project_id: str, state: Mapping[str, Any], steps: Iterable[Any], rules: Iterable[Any],
) -> RoutingFold:
    return RoutingFold(
        project_id=project_id, cursor=state['routing_cursor'],
        steps=tuple(_STEPS.decode(record) for record in steps),
        rules=tuple(_RULES.decode(record) for record in rules),
        first_at=_UTC.decode(state['first_decision_at']),
        newest_at=_UTC.decode(state['newest_decision_at']),
        digest_since=_UTC.decode(state['digest_since']),
        skipped=state['skipped_routing'], complete=_BOOL.decode(state['fold_complete']),
    )


async def read_routing_fold(conn: aiosqlite.Connection, project_id: str) -> RoutingFold | None:
    """The routing fold of the project's newest good derivation, without its markers; None if it
    has never had one."""
    async with _one_snapshot(conn):
        state = await _fetch_state(conn, project_id)
        if state is None or state['refreshed_at'] is None:
            return None
        steps, rules = await _fetch_fold_rows(conn, project_id)
    return _decode_fold(project_id, state, steps, rules)


async def read_marker_state(
    conn: aiosqlite.Connection, project_id: str,
) -> StoredMarkerState | None:
    """The project's newest good derivation and newest attempt; None if it has never had one."""
    async with _one_snapshot(conn):
        state = await _fetch_state(conn, project_id)
        if state is None:
            return None
        attempted_at, failure = parse_utc(state['attempted_at']), state['failure']
        if state['refreshed_at'] is None:
            return StoredMarkerState(derived=None, attempted_at=attempted_at, failure=failure)
        steps, rules = await _fetch_fold_rows(conn, project_id)
        markers = await _fetch(conn, _SELECT_MARKERS_SQL, project_id)
    fold = _decode_fold(project_id, state, steps, rules)
    derived = DerivedState(
        fold=fold,
        markers=tuple(_decode_marker(project_id, record) for record in markers),
        skipped=state['skipped_derived'],
        prompt_coverage=PromptCoverage(
            digest_since=fold.digest_since, roles_py_landings=state['roles_py_landings'],
            counted_since=_UTC.decode(state['roles_py_since']), reason=state['roles_py_reason'],
        ),
        refreshed_at=parse_utc(state['refreshed_at']),
    )
    return StoredMarkerState(derived=derived, attempted_at=attempted_at, failure=failure)
