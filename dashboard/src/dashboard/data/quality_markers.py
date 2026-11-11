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

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import NamedTuple

from shared.quality_series.stats import RegimeEvent

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
