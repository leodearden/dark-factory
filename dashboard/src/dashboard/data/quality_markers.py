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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import yaml
from shared.quality_series.landings import parse_utc
from shared.quality_series.stats import RegimeEvent

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


class CuratedMarkers(NamedTuple):
    markers: tuple[Marker, ...]
    parse_failures: int


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


STEP_LAYERS = frozenset({'config', 'role_default'})
"""The routing layers whose decisions form a role's steps; policy rules and overrides do not."""


@dataclass(frozen=True)
class RoutingStep:
    """A run of one role's consecutive config/role_default decisions with one route.

    ``prompt_digest`` None means no decision in the run carried one yet.
    """

    role: str
    first_event_id: int
    first_at: datetime
    last_event_id: int
    last_at: datetime
    decisions: int
    source_layer: str
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
    source_layer: str
    model: str
    effort: str
    budget_usd: float
    max_turns: int
    rule_id: str | None
    prompt_digest: str | None


def _decision(row: tuple[object, ...]) -> _Decision | None:
    """One routing_decision row, or None when it cannot be read as one."""
    event_id, timestamp, role, layer, model, effort, budget, turns, rule_id, digest = row
    texts = (role, layer, model, effort)
    if not (
        isinstance(event_id, int) and isinstance(timestamp, str)
        and all(isinstance(text, str) and text for text in texts)
        and isinstance(budget, (int, float)) and isinstance(turns, int)
        and (rule_id is None or isinstance(rule_id, str))
        and (digest is None or isinstance(digest, str))
    ):
        return None
    try:
        at = parse_utc(timestamp)
    except ValueError:
        return None
    return _Decision(event_id, at, str(role), str(layer), str(model), str(effort), float(budget),
                     turns, rule_id, digest)


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
        if decision.source_layer in STEP_LAYERS:
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
