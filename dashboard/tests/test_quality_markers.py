"""The quality marker stream: vocabulary, curated reader, routing fold, derivation and serving."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from shared.quality_series.stats import RegimeEvent

from dashboard.data.quality_markers import (
    Marker,
    MarkerInvariant,
    MarkerInvariantError,
    MarkerKind,
    MarkerSource,
    SourceType,
)

AT = datetime(2026, 11, 1, 8, 58, 42, tzinfo=UTC)

EXAMPLE_SOURCES = {
    SourceType.ROUTING: MarkerSource(SourceType.ROUTING, role='reviewer_comprehensive'),
    SourceType.CONFIG_RELOAD: MarkerSource(SourceType.CONFIG_RELOAD, path='review.reviewer_briefing'),
    SourceType.POLICY_RULE: MarkerSource(SourceType.POLICY_RULE, rule='large-plan-steps'),
    SourceType.DEPLOY: MarkerSource(SourceType.DEPLOY),
    SourceType.MODEL_VERSION: MarkerSource(SourceType.MODEL_VERSION, alias='opus'),
    SourceType.INSTRUMENT: MarkerSource(SourceType.INSTRUMENT),
    SourceType.CURATED: MarkerSource(SourceType.CURATED, file='regime-markers.yaml'),
}


def _marker(**overrides: object) -> Marker:
    fields: dict[str, object] = {
        'marker_id': 'routing:reviewer_comprehensive:42',
        'at': AT,
        'kind': MarkerKind.EFFORT,
        'scope': 'dark_factory',
        'series': None,
        'until': None,
        'label': 'reviewer_comprehensive effort high -> xhigh',
        'source': EXAMPLE_SOURCES[SourceType.ROUTING],
    }
    fields.update(overrides)
    return Marker(**fields)  # type: ignore[arg-type]


def _artefact(**overrides: object) -> Marker:
    fields: dict[str, object] = {
        'marker_id': 'curated:abc',
        'kind': MarkerKind.ARTEFACT,
        'until': AT + timedelta(hours=12),
        'series': ('flakes_confirmed',),
        'label': 'disk full',
        'source': EXAMPLE_SOURCES[SourceType.CURATED],
    }
    fields.update(overrides)
    return _marker(**fields)


class TestMarkerInvariants:
    def test_a_valid_marker_and_a_valid_artefact_construct(self) -> None:
        assert _marker().kind is MarkerKind.EFFORT
        assert _artefact().until == AT + timedelta(hours=12)

    @pytest.mark.parametrize(('build', 'invariant', 'offending'), [
        (lambda: _artefact(until=None), MarkerInvariant.ARTEFACT_RANGE, {'until': None}),
        (lambda: _artefact(until=AT - timedelta(seconds=1)), MarkerInvariant.UNTIL_ORDER,
         {'until': AT - timedelta(seconds=1)}),
        (lambda: _marker(until=AT + timedelta(hours=1)), MarkerInvariant.UNTIL_ONLY_ON_ARTEFACT,
         {'kind': MarkerKind.EFFORT}),
        (lambda: _marker(at=AT.replace(tzinfo=None)), MarkerInvariant.AWARE_INSTANT,
         {'at': AT.replace(tzinfo=None)}),
        (lambda: _artefact(until=(AT + timedelta(hours=1)).replace(tzinfo=None)),
         MarkerInvariant.AWARE_INSTANT, {'until': (AT + timedelta(hours=1)).replace(tzinfo=None)}),
        (lambda: _marker(label='  '), MarkerInvariant.NON_EMPTY_LABEL, {'label': '  '}),
        (lambda: _marker(series=()), MarkerInvariant.SERIES, {'series': ()}),
        (lambda: _marker(series=('flakes_confirmed', '')), MarkerInvariant.SERIES,
         {'series': ('flakes_confirmed', '')}),
        (lambda: _marker(marker_id=''), MarkerInvariant.NON_EMPTY_ID, {'marker_id': ''}),
    ])
    def test_construction_refuses_naming_the_invariant_and_the_values(
        self, build, invariant: MarkerInvariant, offending: dict[str, object],
    ) -> None:
        with pytest.raises(MarkerInvariantError) as err:
            build()

        assert err.value.invariant is invariant
        assert offending.items() <= err.value.values.items()
        assert invariant.value in str(err.value)

    def test_the_error_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            _marker(label='')


class TestMarkerSource:
    @pytest.mark.parametrize(('source_type', 'fields'), [
        (SourceType.CONFIG_RELOAD, {'path': 'effort.reviewer', 'role': 'reviewer'}),
        (SourceType.DEPLOY, {'alias': 'opus'}),
        (SourceType.ROUTING, {'role': 'merger', 'file': 'x.yaml'}),
        (SourceType.CURATED, {'file': 'regime-markers.yaml', 'rule': 'r'}),
    ])
    def test_refuses_a_field_its_type_does_not_carry(
        self, source_type: SourceType, fields: dict[str, str],
    ) -> None:
        with pytest.raises(MarkerInvariantError) as err:
            MarkerSource(source_type, **fields)

        assert err.value.invariant is MarkerInvariant.SOURCE_FIELDS
        assert err.value.values['type'] is source_type

    @pytest.mark.parametrize(('source_type', 'missing'), [
        (SourceType.ROUTING, 'role'),
        (SourceType.CONFIG_RELOAD, 'path'),
        (SourceType.POLICY_RULE, 'rule'),
        (SourceType.MODEL_VERSION, 'alias'),
        (SourceType.CURATED, 'file'),
    ])
    def test_refuses_a_missing_required_field(self, source_type: SourceType, missing: str) -> None:
        with pytest.raises(MarkerInvariantError) as err:
            MarkerSource(source_type)

        assert err.value.invariant is MarkerInvariant.SOURCE_FIELDS
        assert missing in str(err.value)

    def test_refuses_an_empty_required_field(self) -> None:
        with pytest.raises(MarkerInvariantError):
            MarkerSource(SourceType.ROUTING, role='')

    @pytest.mark.parametrize('source_type', list(SourceType))
    def test_round_trips_through_its_wire_form(self, source_type: SourceType) -> None:
        source = EXAMPLE_SOURCES[source_type]

        assert MarkerSource.from_wire(source.to_wire()) == source

    def test_the_wire_form_names_the_type_and_omits_absent_fields(self) -> None:
        assert EXAMPLE_SOURCES[SourceType.ROUTING].to_wire() == {
            'type': 'routing', 'role': 'reviewer_comprehensive',
        }
        assert EXAMPLE_SOURCES[SourceType.DEPLOY].to_wire() == {'type': 'deploy'}

    def test_a_routing_source_may_name_the_config_path_that_caused_it(self) -> None:
        source = MarkerSource(SourceType.ROUTING, role='reviewer', path='effort.reviewer')

        assert source.to_wire() == {'type': 'routing', 'role': 'reviewer', 'path': 'effort.reviewer'}


class TestMarkerWire:
    def test_wire_keys_are_exactly_the_contract(self) -> None:
        assert set(_marker().to_wire()) == {
            'id', 'at', 'ordinal', 'until_ordinal', 'kind', 'scope', 'series', 'until',
            'label', 'source', 'ref', 'tasks',
        }

    def test_wire_values(self) -> None:
        plus_two = timezone(timedelta(hours=2))
        marker = _artefact(
            at=AT.astimezone(plus_two), until=(AT + timedelta(hours=12)).astimezone(plus_two),
            scope=None, ref='abc123', tasks=(2631, '8460'), ordinal=7, until_ordinal=9,
        )

        assert marker.to_wire() == {
            'id': 'curated:abc',
            'at': '2026-11-01T08:58:42+00:00',
            'ordinal': 7,
            'until_ordinal': 9,
            'kind': 'artefact',
            'scope': 'fleet',
            'series': ['flakes_confirmed'],
            'until': '2026-11-01T20:58:42+00:00',
            'label': 'disk full',
            'source': {'type': 'curated', 'file': 'regime-markers.yaml'},
            'ref': 'abc123',
            'tasks': [2631, '8460'],
        }

    def test_an_every_series_point_marker_serves_nulls(self) -> None:
        wire = _marker().to_wire()

        assert wire['series'] is None
        assert wire['until'] is None
        assert wire['ordinal'] is None
        assert wire['until_ordinal'] is None
        assert wire['scope'] == 'dark_factory'
        assert wire['tasks'] == []


class TestRegimeEvent:
    @pytest.mark.parametrize('kind', [
        MarkerKind.MODEL, MarkerKind.EFFORT, MarkerKind.BUDGET, MarkerKind.TURNS,
        MarkerKind.PROMPT, MarkerKind.REVIEWER_POLICY, MarkerKind.ROUTING_RULE,
        MarkerKind.CLI, MarkerKind.REGIME,
    ])
    def test_boundary_kinds_are_regime_boundaries(self, kind: MarkerKind) -> None:
        marker = _marker(kind=kind, series=('rejected',))

        assert marker.regime_event() == RegimeEvent(
            event_id=marker.marker_id, at=AT, series=('rejected',),
        )

    def test_an_artefact_is_an_exclusion_range(self) -> None:
        marker = _artefact()

        assert marker.regime_event() == RegimeEvent(
            event_id='curated:abc', at=AT, until=AT + timedelta(hours=12),
            series=('flakes_confirmed',), artefact=True,
        )

    @pytest.mark.parametrize('kind', [MarkerKind.DEPLOY, MarkerKind.INSTRUMENT])
    def test_deploy_and_instrument_are_not_landing_axis_events(self, kind: MarkerKind) -> None:
        assert _marker(kind=kind).regime_event() is None
