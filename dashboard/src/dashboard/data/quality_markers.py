"""The quality marker stream: the dated changes a project's quality series are read against.

A marker is one dated change that can explain a shift in a project's per-landing quality series:
a role's routing step (model, effort, budget, turns or prompt), a reviewer-policy or routing-rule
change, a deploy, a model-version change, an instrument change, or a hand-curated event. Derived
markers come from the project's own runs.db; curated ones from the committed
dashboard/regime-markers.yaml. A marker's kind alone decides its place on the landing axis: a
regime boundary, an exclusion range, or neither; its ordinal is the first landing at or after it,
resolved when it is served rather than stored. Design: plans/quality-dashboard-prd.md decisions
10-12 and §Contract "Markers".
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import yaml
from shared.project_id import fold_project_token
from shared.quality_series.landings import parse_utc
from shared.quality_series.stats import RegimeEvent

from dashboard.data.datum import Datum, DatumState, aged_at, unknown_datum

logger = logging.getLogger(__name__)

CURATED_MARKERS_PATH = Path(__file__).resolve().parents[3] / 'regime-markers.yaml'
"""The committed curated file, dashboard/regime-markers.yaml; read at call time."""

FLEET_SCOPE = 'fleet'
"""The wire scope of a marker that applies to every project."""


class MarkerKind(StrEnum):
    MODEL = 'model'
    EFFORT = 'effort'
    BUDGET = 'budget'
    TURNS = 'turns'
    PROMPT = 'prompt'
    REVIEWER_POLICY = 'reviewer_policy'
    ROUTING_RULE = 'routing_rule'
    CLI = 'cli'
    DEPLOY = 'deploy'
    ARTEFACT = 'artefact'
    REGIME = 'regime'
    INSTRUMENT = 'instrument'


class RegimeRole(StrEnum):
    """What a marker is to the landing-axis statistics."""

    BOUNDARY = 'boundary'
    EXCLUSION = 'exclusion'
    NONE = 'none'


REGIME_ROLE: Mapping[MarkerKind, RegimeRole] = MappingProxyType({
    MarkerKind.MODEL: RegimeRole.BOUNDARY,
    MarkerKind.EFFORT: RegimeRole.BOUNDARY,
    MarkerKind.BUDGET: RegimeRole.BOUNDARY,
    MarkerKind.TURNS: RegimeRole.BOUNDARY,
    MarkerKind.PROMPT: RegimeRole.BOUNDARY,
    MarkerKind.REVIEWER_POLICY: RegimeRole.BOUNDARY,
    MarkerKind.ROUTING_RULE: RegimeRole.BOUNDARY,
    MarkerKind.CLI: RegimeRole.BOUNDARY,
    MarkerKind.REGIME: RegimeRole.BOUNDARY,
    MarkerKind.ARTEFACT: RegimeRole.EXCLUSION,
    MarkerKind.DEPLOY: RegimeRole.NONE,
    MarkerKind.INSTRUMENT: RegimeRole.NONE,
})
"""Deploys land about daily and would cut every segment below the baseline floor; instrument
markers annotate the snapshot series, not landings."""


class MarkerInvariant(StrEnum):
    NON_EMPTY_ID = 'non_empty_id'
    PROMPT_COVERAGE = 'prompt_coverage'
    AWARE_INSTANT = 'aware_instant'
    ARTEFACT_RANGE = 'artefact_range'
    UNTIL_ONLY_ON_ARTEFACT = 'until_only_on_artefact'
    UNTIL_ORDER = 'until_order'
    NON_EMPTY_LABEL = 'non_empty_label'
    SERIES = 'series'
    SOURCE_FIELDS = 'source_fields'


class MarkerInvariantError(ValueError):
    """A marker value broke a declared invariant; ``invariant`` names which, ``values`` the inputs."""

    def __init__(self, invariant: MarkerInvariant, values: Mapping[str, object]) -> None:
        self.invariant = invariant
        self.values = dict(values)
        detail = ', '.join(f'{name}={value!r}' for name, value in self.values.items())
        super().__init__(f'{invariant.value}: {detail}')


def _require(condition: bool, invariant: MarkerInvariant, **values: object) -> None:
    if not condition:
        raise MarkerInvariantError(invariant, values)


def _require_aware(name: str, instant: datetime) -> None:
    _require(instant.tzinfo is not None and instant.utcoffset() is not None,
             MarkerInvariant.AWARE_INSTANT, **{name: instant})


def _iso_utc(instant: datetime) -> str:
    return instant.astimezone(UTC).isoformat()


class SourceType(StrEnum):
    ROUTING = 'routing'
    CONFIG_RELOAD = 'config_reload'
    POLICY_RULE = 'policy_rule'
    DEPLOY = 'deploy'
    MODEL_VERSION = 'model_version'
    INSTRUMENT = 'instrument'
    CURATED = 'curated'


class _SourceFields(NamedTuple):
    required: frozenset[str]
    optional: frozenset[str] = frozenset()


_SOURCE_DETAILS = ('role', 'path', 'rule', 'alias', 'file')

_SOURCE_FIELDS: Mapping[SourceType, _SourceFields] = MappingProxyType({
    SourceType.ROUTING: _SourceFields(frozenset({'role'}), frozenset({'path'})),
    SourceType.CONFIG_RELOAD: _SourceFields(frozenset({'path'})),
    SourceType.POLICY_RULE: _SourceFields(frozenset({'rule'})),
    SourceType.DEPLOY: _SourceFields(frozenset()),
    SourceType.MODEL_VERSION: _SourceFields(frozenset({'alias'})),
    SourceType.INSTRUMENT: _SourceFields(frozenset()),
    SourceType.CURATED: _SourceFields(frozenset({'file'})),
})
"""Which details each source type carries. A routing marker's optional ``path`` is the
config_reload path that dated it."""


@dataclass(frozen=True)
class MarkerSource:
    """Where a marker came from: its source type plus exactly the details that type carries."""

    type: SourceType
    role: str | None = None
    path: str | None = None
    rule: str | None = None
    alias: str | None = None
    file: str | None = None

    def __post_init__(self) -> None:
        fields = _SOURCE_FIELDS[self.type]
        given = {name: value for name, value in self._details() if value is not None}
        _require(
            fields.required <= set(given) <= fields.required | fields.optional
            and all(isinstance(value, str) and value for value in given.values()),
            MarkerInvariant.SOURCE_FIELDS,
            type=self.type, required=tuple(sorted(fields.required)),
            optional=tuple(sorted(fields.optional)), given=given,
        )

    def _details(self) -> tuple[tuple[str, str | None], ...]:
        return tuple((name, getattr(self, name)) for name in _SOURCE_DETAILS)

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {'type': self.type.value}
        wire.update((name, value) for name, value in self._details() if value is not None)
        return wire

    @classmethod
    def from_wire(cls, wire: Mapping[str, object]) -> MarkerSource:
        unknown = set(wire) - {'type', *_SOURCE_DETAILS}
        _require(not unknown, MarkerInvariant.SOURCE_FIELDS, unknown=tuple(sorted(unknown)))
        return cls(
            SourceType(str(wire.get('type'))),
            role=_wire_detail(wire, 'role'),
            path=_wire_detail(wire, 'path'),
            rule=_wire_detail(wire, 'rule'),
            alias=_wire_detail(wire, 'alias'),
            file=_wire_detail(wire, 'file'),
        )


def _wire_detail(wire: Mapping[str, object], name: str) -> str | None:
    value = wire.get(name)
    if value is None or isinstance(value, str):
        return value
    raise MarkerInvariantError(MarkerInvariant.SOURCE_FIELDS, {name: value})


@dataclass(frozen=True)
class Marker:
    """One dated change. ``scope`` None applies to every project; ``series`` None to every series.

    ``until`` is the inclusive end of an artefact range and exists only on an artefact.
    ``ordinal`` and ``until_ordinal`` are landing ordinals, set only when served.
    """

    marker_id: str
    at: datetime
    kind: MarkerKind
    scope: str | None
    series: tuple[str, ...] | None
    until: datetime | None
    label: str
    source: MarkerSource
    ref: str | None = None
    tasks: tuple[int | str, ...] = ()
    ordinal: int | None = None
    until_ordinal: int | None = None

    def __post_init__(self) -> None:
        _require(bool(self.marker_id), MarkerInvariant.NON_EMPTY_ID, marker_id=self.marker_id)
        _require_aware('at', self.at)
        if self.until is not None:
            _require_aware('until', self.until)
        is_artefact = self.kind is MarkerKind.ARTEFACT
        _require(not is_artefact or self.until is not None, MarkerInvariant.ARTEFACT_RANGE,
                 marker_id=self.marker_id, until=self.until)
        _require(is_artefact or self.until is None, MarkerInvariant.UNTIL_ONLY_ON_ARTEFACT,
                 marker_id=self.marker_id, kind=self.kind, until=self.until)
        _require(self.until is None or self.until >= self.at, MarkerInvariant.UNTIL_ORDER,
                 marker_id=self.marker_id, at=self.at, until=self.until)
        _require(bool(self.label.strip()), MarkerInvariant.NON_EMPTY_LABEL,
                 marker_id=self.marker_id, label=self.label)
        _require(self.series is None or (bool(self.series) and all(self.series)),
                 MarkerInvariant.SERIES, marker_id=self.marker_id, series=self.series)

    def to_wire(self) -> dict[str, object]:
        return {
            'id': self.marker_id,
            'at': _iso_utc(self.at),
            'ordinal': self.ordinal,
            'until_ordinal': self.until_ordinal,
            'kind': self.kind.value,
            'scope': FLEET_SCOPE if self.scope is None else self.scope,
            'series': None if self.series is None else list(self.series),
            'until': None if self.until is None else _iso_utc(self.until),
            'label': self.label,
            'source': self.source.to_wire(),
            'ref': self.ref,
            'tasks': list(self.tasks),
        }

    def regime_event(self) -> RegimeEvent | None:
        """This marker as the statistics read it, or None when it is not a landing-axis event."""
        role = REGIME_ROLE[self.kind]
        if role is RegimeRole.BOUNDARY:
            return RegimeEvent(event_id=self.marker_id, at=self.at, series=self.series)
        if role is RegimeRole.EXCLUSION:
            return RegimeEvent(event_id=self.marker_id, at=self.at, until=self.until,
                               series=self.series, artefact=True)
        return None


def regime_events(markers: Iterable[Marker]) -> tuple[RegimeEvent, ...]:
    """The markers the landing-axis statistics read: boundaries and exclusion ranges."""
    return tuple(event for marker in markers if (event := marker.regime_event()) is not None)


class CuratedMarkers(NamedTuple):
    markers: tuple[Marker, ...]
    parse_failures: int

    def for_project(self, project_id: str) -> tuple[Marker, ...]:
        """The curated markers in *project_id*'s scope, in file order."""
        return tuple(marker for marker in self.markers if in_scope(marker, project_id))


class _CuratedFieldError(ValueError):
    def __init__(self, field: str, value: object) -> None:
        super().__init__(f'{field}: unusable value {value!r}')


def load_curated_markers(path: Path | None = None) -> CuratedMarkers:
    """The curated markers in file order, and how many entries were rejected; never raises.

    A missing file is no markers and no failure. An unreadable file, or one whose top level is
    not a list, is one failure. Every rejected entry is one counted, logged failure, and its
    valid siblings still load (INV-4). Unknown keys are ignored.
    """
    source = CURATED_MARKERS_PATH if path is None else path
    if not source.exists():
        return CuratedMarkers((), 0)
    try:
        entries = yaml.safe_load(source.read_text())
    except (OSError, yaml.YAMLError) as exc:
        logger.warning('curated markers %s unreadable: %s', source, exc)
        return CuratedMarkers((), 1)
    if not isinstance(entries, list):
        logger.warning('curated markers %s: top level is a %s, not a list',
                       source, type(entries).__name__)
        return CuratedMarkers((), 1)
    markers: dict[str, Marker] = {}
    failures = 0
    for index, entry in enumerate(entries):
        try:
            marker = _curated_marker(entry, source.name)
            if marker.marker_id in markers:
                raise _CuratedFieldError('entry', 'a duplicate of an earlier entry')
        except (MarkerInvariantError, _CuratedFieldError) as exc:
            failures += 1
            logger.warning('curated marker #%d in %s rejected: %s', index, source, exc)
            continue
        markers[marker.marker_id] = marker
    return CuratedMarkers(tuple(markers.values()), failures)


def _curated_marker(entry: object, file_name: str) -> Marker:
    if not isinstance(entry, dict):
        raise _CuratedFieldError('entry', entry)
    at = _curated_instant(entry.get('date'), 'date', end_of_day=False)
    raw_until = entry.get('until')
    until = None if raw_until is None else _curated_instant(raw_until, 'until', end_of_day=True)
    kind = _curated_kind(entry.get('kind'))
    scope = _curated_project(entry.get('project'))
    series = _curated_series(entry.get('series'))
    label = _curated_label(entry.get('label'))
    return Marker(
        marker_id=_curated_id(at, until, kind, scope, series, label),
        at=at, kind=kind, scope=scope, series=series, until=until, label=label,
        source=MarkerSource(SourceType.CURATED, file=file_name),
        tasks=_curated_tasks(entry.get('tasks')),
    )


def _curated_instant(value: object, field: str, *, end_of_day: bool) -> datetime:
    """A date is that UTC day's first instant, or for an inclusive end its last; a naive time is refused."""
    if isinstance(value, str):
        value = _iso_value(value, field)
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise _CuratedFieldError(field, value)
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.max if end_of_day else time.min, tzinfo=UTC)
    raise _CuratedFieldError(field, value)


def _iso_value(text: str, field: str) -> date | datetime:
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise _CuratedFieldError(field, text) from None


def _curated_kind(value: object) -> MarkerKind:
    if value is None:
        return MarkerKind.REGIME
    if isinstance(value, str) and value in MarkerKind:
        return MarkerKind(value)
    raise _CuratedFieldError('kind', value)


def _curated_project(value: object) -> str | None:
    if value is None or (isinstance(value, str) and value):
        return value
    raise _CuratedFieldError('project', value)


def _curated_series(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(series, str) for series in value):
        return tuple(value)
    raise _CuratedFieldError('series', value)


def _curated_label(value: object) -> str:
    if isinstance(value, str):
        return value
    raise _CuratedFieldError('label', value)


def _curated_tasks(value: object) -> tuple[int | str, ...]:
    if value is None:
        return ()
    if isinstance(value, list) and all(
        isinstance(task, (int, str)) and not isinstance(task, bool) for task in value
    ):
        return tuple(value)
    raise _CuratedFieldError('tasks', value)


def _curated_id(
    at: datetime, until: datetime | None, kind: MarkerKind, scope: str | None,
    series: tuple[str, ...] | None, label: str,
) -> str:
    """A content hash, so an entry keeps its id when the file is reordered or another is inserted."""
    canonical = json.dumps({
        'date': _iso_utc(at),
        'until': None if until is None else _iso_utc(until),
        'kind': kind.value,
        'project': scope,
        'series': None if series is None else list(series),
        'label': label,
    }, sort_keys=True)
    return 'curated:' + hashlib.sha256(canonical.encode()).hexdigest()[:16]


class SourceLayer(StrEnum):
    """The routing layer that chose a decision's model
    (orchestrator/src/orchestrator/routing.py::resolve_route)."""

    ROLE_DEFAULT = 'role_default'
    CONFIG = 'config'
    POLICY_RULE = 'policy_rule'
    METADATA_OVERRIDE = 'metadata_override'


STEP_LAYERS = frozenset({SourceLayer.CONFIG, SourceLayer.ROLE_DEFAULT})
"""The routing layers whose decisions form a role's steps; policy rules and overrides do not."""


@dataclass(frozen=True)
class RoutingStep:
    """A run of one role's consecutive config/role_default decisions, matched by no policy rule,
    with one route.

    ``prompt_digest`` None means no decision in the run carried one yet.
    """

    role: str
    first_event_id: int
    first_at: datetime
    last_event_id: int
    last_at: datetime
    decisions: int
    source_layer: SourceLayer
    model: str
    effort: str
    budget_usd: float
    max_turns: int
    prompt_digest: str | None


@dataclass(frozen=True)
class RuleSpan:
    """When a routing policy rule matched, first and last, and how many decisions it matched."""

    rule_id: str
    first_event_id: int
    first_at: datetime
    last_at: datetime
    decisions: int


@dataclass(frozen=True)
class RoutingFold:
    """A project's routing_decision rows folded up to ``cursor``, the newest event id read.

    ``steps`` are in event order (per role, the last is that role's open step); ``first_at`` and
    ``newest_at`` bound the folded decisions; ``digest_since`` is the first one carrying a
    prompt digest. ``complete`` is False until a fold has read every row after its cursor.
    """

    project_id: str
    cursor: int
    steps: tuple[RoutingStep, ...]
    rules: tuple[RuleSpan, ...]
    first_at: datetime | None
    newest_at: datetime | None
    digest_since: datetime | None
    skipped: int
    complete: bool

    @classmethod
    def empty(cls, project_id: str) -> RoutingFold:
        return cls(project_id=project_id, cursor=0, steps=(), rules=(), first_at=None,
                   newest_at=None, digest_since=None, skipped=0, complete=False)


_ROUTING_SQL = """
SELECT id, timestamp, COALESCE(role, json_extract(doc, '$.role')),
       json_extract(doc, '$.source_layer'), json_extract(doc, '$.model'),
       json_extract(doc, '$.effort'), json_extract(doc, '$.budget_usd'),
       json_extract(doc, '$.max_turns'), json_extract(doc, '$.rule_id'),
       json_extract(doc, '$.prompt_digest')
FROM (
    SELECT id, timestamp, role,
           CASE WHEN json_valid(data) AND json_type(data) = 'object' THEN data END AS doc
    FROM events
    WHERE event_type = 'routing_decision' AND id > ?
    ORDER BY id
    LIMIT ?
)
ORDER BY id
"""


class _Decision(NamedTuple):
    event_id: int
    at: datetime
    role: str
    source_layer: SourceLayer
    model: str
    effort: str
    budget_usd: float
    max_turns: int
    rule_id: str | None
    prompt_digest: str | None


def _decision(row: tuple[object, ...]) -> _Decision | None:
    """One routing_decision row, or None when it cannot be read as one; an unknown layer cannot."""
    event_id, timestamp, role, layer, model, effort, budget, turns, rule_id, digest = row
    texts = (role, model, effort)
    if not (
        isinstance(event_id, int) and isinstance(timestamp, str)
        and all(isinstance(text, str) and text for text in texts)
        and isinstance(layer, str) and layer in SourceLayer
        and isinstance(budget, (int, float)) and isinstance(turns, int)
        and (rule_id is None or isinstance(rule_id, str))
        and (digest is None or isinstance(digest, str))
    ):
        return None
    try:
        at = parse_utc(timestamp)
    except ValueError:
        return None
    return _Decision(event_id, at, str(role), SourceLayer(layer), str(model), str(effort),
                     float(budget), turns, rule_id, digest)


def fold_routing_decisions(
    conn: sqlite3.Connection, fold: RoutingFold, *, batch: int | None,
) -> RoutingFold:
    """*fold* advanced over at most *batch* routing_decision rows after its cursor (None: all).

    Returns a new fold; the input is never mutated. Rows already folded are never re-read, so
    folding in batches until complete equals one full fold. An unreadable row is skipped and
    counted.
    """
    rows = conn.execute(_ROUTING_SQL, (fold.cursor, -1 if batch is None else batch)).fetchall()
    builder = _FoldBuilder(fold)
    for row in rows:
        builder.add(row[0], _decision(row))
    return builder.build(complete=batch is None or len(rows) < batch)


class _FoldBuilder:
    def __init__(self, fold: RoutingFold) -> None:
        self._fold = fold
        self._cursor = fold.cursor
        self._steps = list(fold.steps)
        self._open = {step.role: index for index, step in enumerate(self._steps)}
        self._rules = {span.rule_id: span for span in fold.rules}
        self._first_at = fold.first_at
        self._newest_at = fold.newest_at
        self._digest_since = fold.digest_since
        self._skipped = fold.skipped

    def add(self, event_id: int, decision: _Decision | None) -> None:
        self._cursor = max(self._cursor, event_id)
        if decision is None:
            self._skipped += 1
            return
        self._first_at = decision.at if self._first_at is None else self._first_at
        self._newest_at = decision.at if self._newest_at is None else max(self._newest_at, decision.at)
        if decision.prompt_digest is not None and self._digest_since is None:
            self._digest_since = decision.at
        if decision.rule_id is not None:
            self._add_rule(decision.rule_id, decision)
        elif decision.source_layer in STEP_LAYERS:
            self._add_step(decision)

    def _add_rule(self, rule_id: str, decision: _Decision) -> None:
        span = self._rules.get(rule_id)
        self._rules[rule_id] = (
            RuleSpan(rule_id, decision.event_id, decision.at, decision.at, 1) if span is None
            else replace(span, last_at=decision.at, decisions=span.decisions + 1)
        )

    def _add_step(self, decision: _Decision) -> None:
        index = self._open.get(decision.role)
        if index is not None and _continues(step := self._steps[index], decision):
            self._steps[index] = replace(
                step, last_event_id=decision.event_id, last_at=decision.at,
                decisions=step.decisions + 1,
                prompt_digest=step.prompt_digest or decision.prompt_digest,
            )
            return
        self._open[decision.role] = len(self._steps)
        self._steps.append(RoutingStep(
            role=decision.role, first_event_id=decision.event_id, first_at=decision.at,
            last_event_id=decision.event_id, last_at=decision.at, decisions=1,
            source_layer=decision.source_layer, model=decision.model, effort=decision.effort,
            budget_usd=decision.budget_usd, max_turns=decision.max_turns,
            prompt_digest=decision.prompt_digest,
        ))

    def build(self, *, complete: bool) -> RoutingFold:
        return RoutingFold(
            project_id=self._fold.project_id,
            cursor=self._cursor,
            steps=tuple(self._steps),
            rules=tuple(sorted(self._rules.values(), key=lambda span: span.first_event_id)),
            first_at=self._first_at,
            newest_at=self._newest_at,
            digest_since=self._digest_since,
            skipped=self._skipped,
            complete=complete,
        )


def _continues(step: RoutingStep, decision: _Decision) -> bool:
    """Same route, and no two different known prompt digests: an absent digest never splits."""
    same_route = (step.source_layer, step.model, step.effort, step.budget_usd, step.max_turns) == (
        decision.source_layer, decision.model, decision.effort, decision.budget_usd,
        decision.max_turns,
    )
    digests = (step.prompt_digest, decision.prompt_digest)
    return same_route and (None in digests or digests[0] == digests[1])


@dataclass(frozen=True)
class ConfigChange:
    """One path a config_reload reported changing; ``applied`` False means restart-required."""

    event_id: int
    at: datetime
    path: str
    old: object
    new: object
    applied: bool


class ConfigChanges(NamedTuple):
    changes: tuple[ConfigChange, ...]
    skipped: int


_CONFIG_RELOAD_SQL = """
SELECT id, timestamp, CASE WHEN json_valid(data) AND json_type(data) = 'object' THEN data END
FROM events
WHERE event_type = 'config_reload'
ORDER BY id
"""

_RELOAD_SECTIONS = (('applied', True), ('restart_required', False))


def read_config_changes(conn: sqlite3.Connection) -> ConfigChanges:
    """Every changed path of every config_reload row, in event order; unreadable parts are counted."""
    changes: list[ConfigChange] = []
    skipped = 0
    for event_id, timestamp, doc in conn.execute(_CONFIG_RELOAD_SQL):
        try:
            at = parse_utc(timestamp)
        except (TypeError, ValueError):
            at = None
        if at is None or doc is None:
            skipped += 1
            continue
        report = json.loads(doc)
        for section, applied in _RELOAD_SECTIONS:
            entries = report.get(section) or {}
            if not isinstance(entries, dict):
                skipped += 1
                continue
            for path, entry in entries.items():
                if isinstance(entry, dict) and {'old', 'new'} <= set(entry):
                    changes.append(ConfigChange(event_id, at, path, entry['old'], entry['new'],
                                                applied))
                else:
                    skipped += 1
    return ConfigChanges(tuple(changes), skipped)


MIN_PERSISTING_DECISIONS = 3
PERSISTENCE_SPAN = timedelta(hours=24)
COLLAPSE_WINDOW = timedelta(hours=24)
CAUSE_WINDOW = timedelta(hours=48)


class RouteField(StrEnum):
    """A routing step's value fields, in kind precedence order."""

    MODEL = 'model'
    EFFORT = 'effort'
    PROMPT = 'prompt'
    BUDGET = 'budget'
    TURNS = 'turns'


_ROUTE_VALUE: Mapping[RouteField, Callable[[RoutingStep], object]] = MappingProxyType({
    RouteField.MODEL: lambda step: step.model,
    RouteField.EFFORT: lambda step: step.effort,
    RouteField.PROMPT: lambda step: step.prompt_digest,
    RouteField.BUDGET: lambda step: step.budget_usd,
    RouteField.TURNS: lambda step: step.max_turns,
})

_ROUTE_KIND: Mapping[RouteField, MarkerKind] = MappingProxyType({
    RouteField.MODEL: MarkerKind.MODEL,
    RouteField.EFFORT: MarkerKind.EFFORT,
    RouteField.PROMPT: MarkerKind.PROMPT,
    RouteField.BUDGET: MarkerKind.BUDGET,
    RouteField.TURNS: MarkerKind.TURNS,
})

_CONFIG_SECTION: Mapping[RouteField, str] = MappingProxyType({
    RouteField.MODEL: 'models',
    RouteField.EFFORT: 'effort',
    RouteField.BUDGET: 'budgets',
    RouteField.TURNS: 'max_turns',
})
"""The config section that sets each field; a prompt has none."""

RouteDiff = dict[RouteField, tuple[object, object]]


def _route_diff(before: RoutingStep, after: RoutingStep) -> RouteDiff:
    """The changed fields in precedence order; an absent prompt digest on either side is no change."""
    diff: RouteDiff = {}
    for field, value in _ROUTE_VALUE.items():
        old, new = value(before), value(after)
        if field is RouteField.PROMPT and (old is None or new is None):
            continue
        if old != new:
            diff[field] = (old, new)
    return diff


def _config_key(role: str) -> str:
    """Mirrors orchestrator/src/orchestrator/routing.py::_config_key: reviewer* share 'reviewer'."""
    return 'reviewer' if role.startswith('reviewer') else role


@dataclass(frozen=True)
class AvailabilityFlag:
    """A role fell back from its configured route to the role default, then returned."""

    role: str
    from_at: datetime
    until_at: datetime
    decisions: int
    model: str

    def to_wire(self) -> dict[str, object]:
        return {'role': self.role, 'from_at': _iso_utc(self.from_at),
                'until_at': _iso_utc(self.until_at), 'decisions': self.decisions,
                'model': self.model}


def routing_markers(fold: RoutingFold, changes: Iterable[ConfigChange]) -> list[Marker]:
    """The de-noised routing changes of *fold*'s project, sorted by (at, id)."""
    reloads = tuple(changes)
    markers = [
        marker
        for steps, _flags in _denoised(fold)
        for marker in _role_markers(fold.project_id, steps, fold.newest_at, reloads)
    ]
    return sorted(markers, key=lambda marker: (marker.at, marker.marker_id))


def availability_flags(fold: RoutingFold) -> list[AvailabilityFlag]:
    """Every config -> role_default -> config fallback in *fold*, by start."""
    flags = [flag for _steps, flags in _denoised(fold) for flag in flags]
    return sorted(flags, key=lambda flag: (flag.from_at, flag.role))


def _denoised(fold: RoutingFold) -> list[tuple[list[RoutingStep], list[AvailabilityFlag]]]:
    by_role: dict[str, list[RoutingStep]] = {}
    for step in fold.steps:
        by_role.setdefault(step.role, []).append(step)
    return [_without_fallbacks(steps) for steps in by_role.values()]


def _without_fallbacks(
    steps: Sequence[RoutingStep],
) -> tuple[list[RoutingStep], list[AvailabilityFlag]]:
    """Remove each maximal role_default run bracketed by equal config steps, merging the brackets."""
    kept: list[RoutingStep] = []
    flags: list[AvailabilityFlag] = []
    index = 0
    while index < len(steps):
        end = _fallback_end(kept, steps, index)
        if end is None:
            kept.append(steps[index])
            index += 1
            continue
        run, back = steps[index:end], steps[end]
        flags.append(AvailabilityFlag(role=back.role, from_at=run[0].first_at,
                                      until_at=back.first_at,
                                      decisions=sum(step.decisions for step in run),
                                      model=run[0].model))
        kept[-1] = _merged(kept[-1], back)
        index = end + 1
    return kept, flags


def _fallback_end(kept: Sequence[RoutingStep], steps: Sequence[RoutingStep], index: int) -> int | None:
    """The index of the config step a role_default run starting at *index* returns to, else None."""
    if (not kept or kept[-1].source_layer is not SourceLayer.CONFIG
            or steps[index].source_layer is not SourceLayer.ROLE_DEFAULT):
        return None
    end = index
    while end < len(steps) and steps[end].source_layer is SourceLayer.ROLE_DEFAULT:
        end += 1
    if (end < len(steps) and steps[end].source_layer is SourceLayer.CONFIG
            and not _route_diff(kept[-1], steps[end])):
        return end
    return None


def _merged(before: RoutingStep, after: RoutingStep) -> RoutingStep:
    return replace(before, last_event_id=after.last_event_id, last_at=after.last_at,
                   decisions=before.decisions + after.decisions,
                   prompt_digest=before.prompt_digest or after.prompt_digest)


def _persists(steps: Sequence[RoutingStep], index: int, newest_at: datetime | None) -> bool:
    """A non-start step counts with >= 3 decisions and (a day's span or a next step changing another field)."""
    step = steps[index]
    if step.decisions < MIN_PERSISTING_DECISIONS:
        return False
    following = steps[index + 1] if index + 1 < len(steps) else None
    span_end = following.first_at if following is not None else (newest_at or step.last_at)
    if span_end - step.first_at >= PERSISTENCE_SPAN:
        return True
    if following is None:
        return False
    return bool(set(_route_diff(step, following)) - set(_route_diff(steps[index - 1], step)))


def _counted_groups(steps: Sequence[RoutingStep], newest_at: datetime | None) -> list[list[RoutingStep]]:
    """Counted steps, each group holding those within a day of its first."""
    groups: list[list[RoutingStep]] = []
    for index in range(1, len(steps)):
        if not _persists(steps, index, newest_at):
            continue
        step = steps[index]
        if groups and step.first_at - groups[-1][0].first_at < COLLAPSE_WINDOW:
            groups[-1].append(step)
        else:
            groups.append([step])
    return groups


def _role_markers(
    project_id: str, steps: Sequence[RoutingStep], newest_at: datetime | None,
    changes: Sequence[ConfigChange],
) -> list[Marker]:
    markers: list[Marker] = []
    if not steps:
        return markers
    state = steps[0]
    for group in _counted_groups(steps, newest_at):
        diff = _route_diff(state, group[-1])
        state = group[-1]
        if diff:
            markers.append(_routing_marker(project_id, group[0], diff, changes))
    return markers


def _routing_marker(
    project_id: str, first: RoutingStep, diff: RouteDiff, changes: Sequence[ConfigChange],
) -> Marker:
    cause = _cause(first, diff, changes)
    return Marker(
        marker_id=f'routing:{first.role}:{first.first_event_id}',
        at=first.first_at if cause is None else cause.at,
        kind=_ROUTE_KIND[next(iter(diff))],
        scope=project_id,
        series=None,
        until=None,
        label=f'{first.role}: ' + ', '.join(
            f'{field.value} {_shown(field, old)} -> {_shown(field, new)}'
            for field, (old, new) in diff.items()
        ),
        source=MarkerSource(SourceType.ROUTING, role=first.role,
                            path=None if cause is None else cause.path),
        ref=str(first.first_event_id),
    )


def _cause(first: RoutingStep, diff: RouteDiff, changes: Sequence[ConfigChange]) -> ConfigChange | None:
    """The earliest reload of a changed field's config path in the 48 h up to the first dispatch."""
    key = _config_key(first.role)
    paths = {f'{_CONFIG_SECTION[field]}.{key}' for field in diff if field in _CONFIG_SECTION}
    window_start = first.first_at - CAUSE_WINDOW
    matches = [change for change in changes
               if change.path in paths and window_start <= change.at <= first.first_at]
    return min(matches, key=lambda change: (change.at, change.event_id), default=None)


_DIGEST_SHOWN = 12


def _shown(field: RouteField, value: object) -> str:
    if field is RouteField.PROMPT and isinstance(value, str):
        return value[:_DIGEST_SHOWN]
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


REVIEWER_BRIEFING_PATH = 'review.reviewer_briefing'
"""The one config change routing_decision cannot see, so it is marked from the reload itself."""


def reviewer_policy_markers(project_id: str, changes: Iterable[ConfigChange]) -> list[Marker]:
    """An applied reviewer-briefing change, at its reload; restart-required ones never ran yet."""
    return [
        Marker(
            marker_id=f'config_reload:{change.event_id}:{change.path}', at=change.at,
            kind=MarkerKind.REVIEWER_POLICY, scope=project_id, series=None, until=None,
            label=f'reviewer briefing {change.old} -> {change.new}',
            source=MarkerSource(SourceType.CONFIG_RELOAD, path=change.path),
            ref=str(change.event_id),
        )
        for change in changes
        if change.applied and change.path == REVIEWER_BRIEFING_PATH
    ]


RULE_QUIET_DAYS = 7
"""A policy rule that has not matched for this long, at the fold's newest decision, is retired."""


class RuleMatch(StrEnum):
    """Which end of a policy rule's matches a routing_rule marker marks."""

    FIRST = 'first'
    LAST = 'last'


def rule_markers(fold: RoutingFold) -> list[Marker]:
    """Each policy rule's first match, and its last once it has been quiet for RULE_QUIET_DAYS."""
    markers: list[Marker] = []
    for span in fold.rules:
        markers.append(_rule_marker(fold.project_id, span, RuleMatch.FIRST, span.first_at,
                                    ref=str(span.first_event_id)))
        quiet = fold.newest_at is not None and (
            fold.newest_at - span.last_at >= timedelta(days=RULE_QUIET_DAYS)
        )
        if quiet:
            markers.append(_rule_marker(fold.project_id, span, RuleMatch.LAST, span.last_at,
                                        ref=None))
    return markers


def _rule_marker(
    project_id: str, span: RuleSpan, end: RuleMatch, at: datetime, *, ref: str | None,
) -> Marker:
    return Marker(
        marker_id=f'policy_rule:{span.rule_id}:{end.value}', at=at, kind=MarkerKind.ROUTING_RULE,
        scope=project_id, series=None, until=None,
        label=f'routing rule {span.rule_id}: {end.value} match',
        source=MarkerSource(SourceType.POLICY_RULE, rule=span.rule_id), ref=ref,
    )


class SourceMarkers(NamedTuple):
    markers: list[Marker]
    skipped: int


def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    """The table's column names; empty when the table does not exist."""
    return frozenset(row[1] for row in conn.execute(f'PRAGMA table_info({table})'))


def _dated_rows(rows: Iterable[tuple[str, object, object]]) -> tuple[list[tuple[datetime, str, str]], int]:
    """(at, key, value) for each row whose instant parses and whose key and value are text."""
    dated: list[tuple[datetime, str, str]] = []
    skipped = 0
    for key, timestamp, value in rows:
        try:
            at = parse_utc(timestamp) if isinstance(timestamp, str) else None
        except ValueError:
            at = None
        if at is None or not isinstance(value, str):
            skipped += 1
            continue
        dated.append((at, key, value))
    return sorted(dated), skipped


def deploy_markers(conn: sqlite3.Connection, project_id: str) -> SourceMarkers:
    """A marker at each run whose code_sha differs from the previous run's; the first is the baseline."""
    if not {'run_id', 'started_at', 'code_sha'} <= _columns(conn, 'runs'):
        return SourceMarkers([], 0)
    runs, skipped = _dated_rows(conn.execute(
        'SELECT run_id, started_at, code_sha FROM runs WHERE code_sha IS NOT NULL'))
    markers = [
        Marker(
            marker_id=f'deploy:{run_id}', at=at, kind=MarkerKind.DEPLOY, scope=project_id,
            series=None, until=None, label=f'deploy {sha[:12]}',
            source=MarkerSource(SourceType.DEPLOY), ref=sha,
        )
        for (_, _, previous), (at, run_id, sha) in pairwise(runs)
        if sha != previous
    ]
    return SourceMarkers(markers, skipped)


def model_version_markers(conn: sqlite3.Connection, project_id: str) -> SourceMarkers:
    """Per model alias, a marker at the first invocation of each model id after the alias's first."""
    if not {'model', 'model_id', 'started_at'} <= _columns(conn, 'invocations'):
        return SourceMarkers([], 0)
    first_seen, skipped = _dated_rows(conn.execute(
        'SELECT model, MIN(started_at), model_id FROM invocations '
        'WHERE model_id IS NOT NULL GROUP BY model, model_id'))
    by_alias: dict[str, list[tuple[datetime, str]]] = {}
    for at, alias, model_id in first_seen:
        by_alias.setdefault(alias, []).append((at, model_id))
    markers = [
        Marker(
            marker_id=f'model_version:{alias}:{model_id}', at=at, kind=MarkerKind.MODEL,
            scope=project_id, series=None, until=None, label=f'{alias} -> {model_id}',
            source=MarkerSource(SourceType.MODEL_VERSION, alias=alias), ref=model_id,
        )
        for alias, ids in by_alias.items()
        for at, model_id in ids[1:]
    ]
    return SourceMarkers(markers, skipped)


@dataclass(frozen=True)
class SnapshotIdentity:
    """Which instrument produced a quality-metrics snapshot: its sha, commit time and params."""

    as_of_sha: str
    committed_at: datetime
    params: Mapping[str, object]


def instrument_markers(project_id: str, snapshots: Iterable[SnapshotIdentity]) -> list[Marker]:
    """A marker at each snapshot whose params differ from the previous snapshot's, naming the keys."""
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.committed_at)
    markers: list[Marker] = []
    for before, after in pairwise(ordered):
        keys = sorted(key for key in before.params.keys() | after.params.keys()
                      if before.params.get(key) != after.params.get(key))
        if keys:
            markers.append(Marker(
                marker_id=f'instrument:{after.as_of_sha}', at=after.committed_at,
                kind=MarkerKind.INSTRUMENT, scope=project_id, series=None, until=None,
                label=f'instrument params changed: {", ".join(keys)}',
                source=MarkerSource(SourceType.INSTRUMENT), ref=after.as_of_sha,
            ))
    return markers


class DerivedMarkers(NamedTuple):
    markers: tuple[Marker, ...]
    skipped: int


def derive(
    fold: RoutingFold, conn: sqlite3.Connection, snapshots: Iterable[SnapshotIdentity],
) -> DerivedMarkers:
    """Every derived marker of *fold*'s project, re-read from the small sources; sorted by (at, id).

    ``skipped`` counts unreadable config_reload, runs and invocations records; the fold counts
    its own.
    """
    project_id = fold.project_id
    changes = read_config_changes(conn)
    deploys = deploy_markers(conn, project_id)
    models = model_version_markers(conn, project_id)
    markers = [
        *routing_markers(fold, changes.changes),
        *reviewer_policy_markers(project_id, changes.changes),
        *rule_markers(fold),
        *deploys.markers,
        *models.markers,
        *instrument_markers(project_id, snapshots),
    ]
    return DerivedMarkers(
        tuple(sorted(markers, key=lambda marker: (marker.at, marker.marker_id))),
        changes.skipped + deploys.skipped + models.skipped,
    )


def derived_markers(
    runs_db: sqlite3.Connection, project_id: str, snapshots: Iterable[SnapshotIdentity],
) -> list[Marker]:
    """The full derivation from nothing: one complete fold of *runs_db*, then :func:`derive`."""
    fold = fold_routing_decisions(runs_db, RoutingFold.empty(project_id), batch=None)
    return list(derive(fold, runs_db, snapshots).markers)


class AxisLanding(NamedTuple):
    ordinal: int
    landed_at: datetime


def resolve_ordinals(markers: Iterable[Marker], axis: Sequence[AxisLanding]) -> list[Marker]:
    """Each marker placed on *axis* (landings in ordinal order), as new markers.

    ``ordinal`` is the first landing at or after ``at`` (None past the newest); an artefact's
    ``until_ordinal`` is the last landing at or before ``until``, None when no landing falls in
    its range. The placement is shared/src/shared/quality_series/stats.py::regime_segments's,
    so a served range covers exactly the landings the statistics exclude.
    """
    landed_at = [landing.landed_at for landing in axis]
    placed = []
    for marker in markers:
        start = bisect_left(landed_at, marker.at)
        end = start if marker.until is None else bisect_right(landed_at, marker.until)
        placed.append(replace(
            marker,
            ordinal=axis[start].ordinal if start < len(axis) else None,
            until_ordinal=axis[end - 1].ordinal if end > start else None,
        ))
    return placed


def in_scope(marker: Marker, project_id: str) -> bool:
    """A fleet marker, or one whose project folds to *project_id* in any spelling."""
    return marker.scope is None or fold_project_token(marker.scope) == fold_project_token(project_id)


class CoverageState(StrEnum):
    WRITTEN = 'written'
    NOT_YET_WRITTEN = 'not_yet_written'


@dataclass(frozen=True)
class PromptCoverage:
    """Whether prompt changes are derivable yet (decision 10), and how stale the curated ones are.

    Written: ``digest_since`` is the first routing decision carrying a prompt digest. Not yet
    written: ``roles_py_landings`` is the count of roles.py landings since ``counted_since`` (the
    last curated prompt marker, else the first routing decision), or ``reason`` says why there
    is no count.
    """

    digest_since: datetime | None
    roles_py_landings: int | None
    counted_since: datetime | None
    reason: str | None

    def __post_init__(self) -> None:
        _require(self._consistent(), MarkerInvariant.PROMPT_COVERAGE,
                 digest_since=self.digest_since, roles_py_landings=self.roles_py_landings,
                 counted_since=self.counted_since, reason=self.reason)

    def _consistent(self) -> bool:
        counted = self.roles_py_landings is not None
        if self.digest_since is not None:
            return not counted and self.counted_since is None and self.reason is None
        if counted:
            return (self.roles_py_landings is not None and self.roles_py_landings >= 0
                    and self.counted_since is not None and self.reason is None)
        return self.counted_since is None and bool(self.reason)

    @property
    def state(self) -> CoverageState:
        return CoverageState.WRITTEN if self.digest_since is not None else CoverageState.NOT_YET_WRITTEN

    def to_wire(self) -> dict[str, object]:
        return {
            'state': self.state.value,
            'since': None if self.digest_since is None else _iso_utc(self.digest_since),
            'roles_py_landings': self.roles_py_landings,
            'counted_since': None if self.counted_since is None else _iso_utc(self.counted_since),
            'reason': self.reason,
        }


@dataclass(frozen=True)
class DerivedState:
    """One project's stored derivation: its routing fold and what was derived from it, and when."""

    fold: RoutingFold
    markers: tuple[Marker, ...]
    skipped: int
    prompt_coverage: PromptCoverage
    refreshed_at: datetime


@dataclass(frozen=True)
class StoredMarkerState:
    """The newest good derivation, if any, and the newest attempt; ``failure`` set when it failed."""

    derived: DerivedState | None
    attempted_at: datetime | None
    failure: str | None


@dataclass(frozen=True)
class MarkerStream:
    """What QUALITY_MARKERS serves for one project."""

    project_id: str
    markers: tuple[Marker, ...]
    availability: tuple[AvailabilityFlag, ...]
    prompt_coverage: PromptCoverage
    curated_parse_failures: int
    skipped_records: int

    def to_wire(self) -> dict[str, object]:
        return {
            'markers': [marker.to_wire() for marker in self.markers],
            'availability': [flag.to_wire() for flag in self.availability],
            'prompt_coverage': self.prompt_coverage.to_wire(),
            'curated_parse_failures': self.curated_parse_failures,
            'skipped_records': self.skipped_records,
        }

    def regime_events(self) -> tuple[RegimeEvent, ...]:
        return regime_events(self.markers)


def marker_stream(
    project_id: str, derived: DerivedState, curated: CuratedMarkers, axis: Sequence[AxisLanding],
) -> MarkerStream:
    """The project's derived markers plus the curated ones in its scope, placed on its axis."""
    chosen = [*derived.markers, *curated.for_project(project_id)]
    ordered = sorted(chosen, key=lambda marker: (marker.at, marker.marker_id))
    return MarkerStream(
        project_id=project_id,
        markers=tuple(resolve_ordinals(ordered, axis)),
        availability=tuple(availability_flags(derived.fold)),
        prompt_coverage=derived.prompt_coverage,
        curated_parse_failures=curated.parse_failures,
        skipped_records=derived.fold.skipped + derived.skipped,
    )


def marker_datum(
    project_id: str,
    stored: StoredMarkerState | None,
    curated: CuratedMarkers,
    axis: Sequence[AxisLanding],
    *,
    now: datetime,
    freshness_bound_seconds: int,
) -> Datum[MarkerStream]:
    """QUALITY_MARKERS for one project, as of its newest good derivation."""
    derived = None if stored is None else stored.derived
    if stored is None or derived is None:
        failure = None if stored is None else stored.failure
        reason = f'quality markers have not been derived for {project_id} yet'
        if failure is not None:
            reason += f'; the newest attempt failed: {failure}'
        return unknown_datum(reason, freshness_bound_seconds)
    stream = marker_stream(project_id, derived, curated, axis)
    stale_reason = _stale_reason(stored, derived)
    if stale_reason is not None:
        return Datum(stream, derived.refreshed_at, DatumState.STALE, stale_reason,
                     freshness_bound_seconds)
    return aged_at(Datum(stream, derived.refreshed_at, DatumState.FRESH, None,
                         freshness_bound_seconds), now)


def _stale_reason(stored: StoredMarkerState, derived: DerivedState) -> str | None:
    if stored.failure is not None:
        attempted = 'unknown' if stored.attempted_at is None else _iso_utc(stored.attempted_at)
        return (f'the newest marker refresh (at {attempted}) failed: {stored.failure}; serving '
                f'the derivation of {_iso_utc(derived.refreshed_at)}')
    if not derived.fold.complete:
        return (f'the routing fold is incomplete: folded through event {derived.fold.cursor}, '
                f'and later ticks continue it')
    return None
