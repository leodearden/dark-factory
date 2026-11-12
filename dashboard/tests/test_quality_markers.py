"""The quality marker stream: vocabulary, curated reader, routing fold, derivation and serving."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from _quality_fixture import (
    append_events,
    config_reload,
    routing_decision,
    write_invocations,
    write_runs_rows,
)
from _runs_db_fixture import EventRow, write_runs_db
from shared.quality_series.stats import RegimeEvent
from shared.tasks_db_readonly import connect_ro

from dashboard.data import quality_markers
from dashboard.data.datum import DatumState, validate_datum
from dashboard.data.quality_markers import (
    AvailabilityFlag,
    AxisLanding,
    ConfigChange,
    ConfigChanges,
    CuratedMarkers,
    DerivedState,
    Marker,
    MarkerInvariant,
    MarkerInvariantError,
    MarkerKind,
    MarkerSource,
    MarkerStream,
    PromptCoverage,
    RoutingFold,
    RoutingStep,
    RuleSpan,
    SnapshotIdentity,
    SourceLayer,
    SourceType,
    StoredMarkerState,
    availability_flags,
    derive,
    derived_markers,
    fold_routing_decisions,
    load_curated_markers,
    marker_datum,
    marker_stream,
    read_config_changes,
    resolve_ordinals,
    routing_markers,
)
from dashboard.data.quality_series import CATALOGUE

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


EFFORT_MARKER = Marker(
    marker_id='routing:reviewer_comprehensive:42',
    at=AT,
    kind=MarkerKind.EFFORT,
    scope='dark_factory',
    series=None,
    until=None,
    label='reviewer_comprehensive effort high -> xhigh',
    source=EXAMPLE_SOURCES[SourceType.ROUTING],
)


def _marker(**overrides: object) -> Marker:
    return replace(EFFORT_MARKER, **overrides)


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


END_OF_DAY = (23, 59, 59, 999999)

VALID_FIRST = """\
- date: 2026-07-15
  label: first
  tasks: [1]
"""
VALID_LAST = """\
- date: 2026-07-16
  label: last
"""


def _curated(tmp_path: Path, text: str, name: str = 'markers.yaml') -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


class TestCuratedReader:
    def test_a_missing_file_is_no_markers_and_no_failure(self, tmp_path: Path) -> None:
        assert load_curated_markers(tmp_path / 'absent.yaml') == CuratedMarkers((), 0)

    @pytest.mark.parametrize('text', ['date: [unclosed', 'date: 2026-07-15\nlabel: x\n', 'just text\n'])
    def test_an_unreadable_file_is_no_markers_and_one_failure(self, tmp_path: Path, text: str) -> None:
        assert load_curated_markers(_curated(tmp_path, text)) == CuratedMarkers((), 1)

    def test_an_old_shape_entry_is_a_fleet_regime_marker_for_every_series(self, tmp_path: Path) -> None:
        path = _curated(tmp_path, '- date: 2026-07-15\n  label: Verify-scope flip\n  tasks: [2593, "x1"]\n')

        (marker,), failures = load_curated_markers(path)

        assert failures == 0
        assert marker.kind is MarkerKind.REGIME
        assert marker.scope is None
        assert marker.series is None
        assert marker.until is None
        assert marker.at == datetime(2026, 7, 15, tzinfo=UTC)
        assert marker.label == 'Verify-scope flip'
        assert marker.tasks == (2593, 'x1')
        assert marker.source == MarkerSource(SourceType.CURATED, file='markers.yaml')
        assert marker.marker_id.startswith('curated:')

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        path = _curated(tmp_path, '- date: 2026-07-15\n  label: x\n  colour: red\n')

        markers, failures = load_curated_markers(path)

        assert failures == 0
        assert [marker.label for marker in markers] == ['x']

    def test_quoted_iso_strings_and_aware_yaml_timestamps_parse(self, tmp_path: Path) -> None:
        path = _curated(tmp_path, (
            '- {date: "2026-10-16", label: quoted date}\n'
            '- {date: "2026-10-16T12:00:00+02:00", label: quoted instant}\n'
            '- {date: 2026-10-16T12:00:00Z, label: yaml instant}\n'
            '- {date: 2026-10-16, until: "2026-10-17T06:00:00Z", kind: artefact, label: quoted until}\n'
        ))

        markers, failures = load_curated_markers(path)

        assert failures == 0
        assert [marker.at for marker in markers] == [
            datetime(2026, 10, 16, tzinfo=UTC),
            datetime(2026, 10, 16, 10, tzinfo=UTC),
            datetime(2026, 10, 16, 12, tzinfo=UTC),
            datetime(2026, 10, 16, tzinfo=UTC),
        ]
        assert markers[3].until == datetime(2026, 10, 17, 6, tzinfo=UTC)

    def test_a_date_only_until_ends_at_the_end_of_that_utc_day(self, tmp_path: Path) -> None:
        path = _curated(tmp_path, (
            '- date: 2026-10-16\n  until: 2026-10-16\n  kind: artefact\n'
            '  project: dark_factory\n  series: [flakes_confirmed]\n  label: disk full\n'
        ))

        (marker,), failures = load_curated_markers(path)

        assert failures == 0
        assert marker.until == datetime(2026, 10, 16, *END_OF_DAY, tzinfo=UTC)
        assert marker.scope == 'dark_factory'
        assert marker.series == ('flakes_confirmed',)
        assert marker.kind is MarkerKind.ARTEFACT

    def test_every_marker_kind_is_accepted(self, tmp_path: Path) -> None:
        entries = ''.join(
            f'- {{date: 2026-07-15, kind: {kind.value}, label: {kind.value}'
            + (', until: 2026-07-15' if kind is MarkerKind.ARTEFACT else '') + '}\n'
            for kind in MarkerKind
        )

        markers, failures = load_curated_markers(_curated(tmp_path, entries))

        assert failures == 0
        assert [marker.kind for marker in markers] == list(MarkerKind)

    @pytest.mark.parametrize('bad_entry', [
        '- just a string\n',
        '- date: 2026-07-15\n',
        '- {date: 2026-07-15, label: ""}\n',
        '- {date: 2026-07-15, label: 42}\n',
        '- {label: no date}\n',
        '- {date: 2026-07-15T12:00:00, label: naive yaml instant}\n',
        '- {date: "2026-07-15T12:00:00", label: naive quoted instant}\n',
        '- {date: not-a-date, label: unparseable date}\n',
        '- {date: 2026-07-15, kind: weather, label: unknown kind}\n',
        '- {date: 2026-07-15, kind: artefact, label: artefact without until}\n',
        '- {date: 2026-07-15, until: 2026-07-16, label: until on a regime entry}\n',
        '- {date: 2026-07-15, until: 2026-07-14, kind: artefact, label: until before date}\n',
        '- {date: 2026-07-15, series: flakes_confirmed, label: series not a list}\n',
        '- {date: 2026-07-15, series: [], label: empty series}\n',
        '- {date: 2026-07-15, series: [flakes_confirmed, ""], label: empty series id}\n',
        '- {date: 2026-07-15, series: [3], label: non-string series id}\n',
        '- {date: 2026-07-15, tasks: 2631, label: tasks not a list}\n',
        '- {date: 2026-07-15, project: [df], label: project not a string}\n',
        VALID_FIRST,
    ])
    def test_a_bad_entry_is_counted_once_and_logged_while_siblings_load(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, bad_entry: str,
    ) -> None:
        path = _curated(tmp_path, VALID_FIRST + bad_entry + VALID_LAST)

        with caplog.at_level(logging.WARNING, logger=quality_markers.__name__):
            markers, failures = load_curated_markers(path)

        assert failures == 1
        assert [marker.label for marker in markers] == ['first', 'last']
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_every_bad_entry_counts(self, tmp_path: Path) -> None:
        path = _curated(tmp_path, VALID_FIRST + '- 1\n- 2\n- {label: x}\n' + VALID_LAST)

        assert load_curated_markers(path).parse_failures == 3

    def test_ids_are_stable_across_a_reorder(self, tmp_path: Path) -> None:
        forward = load_curated_markers(_curated(tmp_path, VALID_FIRST + VALID_LAST, 'a.yaml'))
        backward = load_curated_markers(_curated(tmp_path, VALID_LAST + VALID_FIRST, 'b.yaml'))

        def ids(markers: tuple[Marker, ...]) -> dict[str, str]:
            return {marker.label: marker.marker_id for marker in markers}

        assert ids(forward.markers) == ids(backward.markers)
        assert len(set(ids(forward.markers).values())) == 2

    def test_no_path_reads_the_module_path_at_call_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(quality_markers, 'CURATED_MARKERS_PATH', _curated(tmp_path, VALID_LAST))

        assert [marker.label for marker in load_curated_markers().markers] == ['last']


class TestCommittedCuratedFile:
    def test_it_holds_the_legacy_regime_markers_and_the_disk_full_artefact(self) -> None:
        markers, failures = load_curated_markers()

        assert failures == 0
        regimes = [marker for marker in markers if marker.kind is MarkerKind.REGIME]
        assert [(marker.at, marker.label, marker.tasks) for marker in regimes] == [
            (datetime(2026, 7, 15, tzinfo=UTC), 'Starvation threshold raised to 72h', (2631,)),
            (datetime(2026, 7, 15, tzinfo=UTC), 'Auto-watcher benign closes', (2630,)),
            (datetime(2026, 7, 15, tzinfo=UTC), 'Verify-scope flip', (2593,)),
        ]
        (artefact,) = [marker for marker in markers if marker.kind is MarkerKind.ARTEFACT]
        assert artefact.scope == 'dark_factory'
        assert artefact.at == datetime(2026, 10, 16, tzinfo=UTC)
        assert artefact.until == datetime(2026, 10, 16, *END_OF_DAY, tzinfo=UTC)
        assert artefact.series == ('flakes_confirmed', 'flakes_real_red')
        assert len(markers) == 4

    def test_every_series_it_names_is_a_served_metric(self) -> None:
        metric_ids = {spec.metric_id for spec in CATALOGUE}
        entries = yaml.safe_load(quality_markers.CURATED_MARKERS_PATH.read_text())

        named = {series for entry in entries for series in entry.get('series') or ()}

        assert named
        assert named <= metric_ids


T0 = datetime(2026, 10, 27, 13, 0, tzinfo=UTC)


def _t(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _runs(tmp_path: Path, events: list[EventRow], name: str = 'project') -> Path:
    return write_runs_db(tmp_path / name, events)


def _fold(store: Path, fold: RoutingFold | None = None, *, batch: int | None = None) -> RoutingFold:
    with closing(connect_ro(store)) as conn:
        return fold_routing_decisions(conn, fold or RoutingFold.empty('dark_factory'), batch=batch)


def _step(
    role: str, first: float, last: float, decisions: int, *, first_id: int | None = None,
    last_id: int | None = None, layer: SourceLayer = SourceLayer.CONFIG, model: str = 'fable',
    effort: str = 'high',
    budget_usd: float = 8.0, max_turns: int = 100, digest: str | None = None,
) -> RoutingStep:
    """A step whose event ids equal its minutes unless given (one row per minute in the fixtures)."""
    return RoutingStep(
        role=role, first_event_id=int(first) if first_id is None else first_id, first_at=_t(first),
        last_event_id=int(last) if last_id is None else last_id, last_at=_t(last),
        decisions=decisions, source_layer=layer, model=model, effort=effort,
        budget_usd=budget_usd, max_turns=max_turns, prompt_digest=digest,
    )


def _decision(
    minutes: float, role: str, *, layer: str = 'config', model: str = 'fable',
    effort: str = 'high', budget_usd: float = 8.0, max_turns: int = 100,
    rule_id: str | None = None, digest: str | None = None,
) -> EventRow:
    return routing_decision(
        _t(minutes), role, model=model, effort=effort, budget_usd=budget_usd,
        max_turns=max_turns, source_layer=layer, rule_id=rule_id, prompt_digest=digest,
    )


class TestRoutingFold:
    def test_an_empty_fold_has_folded_nothing(self) -> None:
        assert RoutingFold.empty('reify') == RoutingFold(
            project_id='reify', cursor=0, steps=(), rules=(), first_at=None, newest_at=None,
            digest_since=None, skipped=0, complete=False,
        )

    def test_config_and_role_default_rows_run_length_encode_per_role(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [
            _decision(1, 'merger'),
            _decision(2, 'reviewer_comprehensive', model='opus', budget_usd=12.0, max_turns=60),
            _decision(3, 'merger'),
            _decision(4, 'merger', layer='role_default', model='opus'),
            _decision(5, 'merger'),
        ])

        fold = _fold(store)

        assert fold.steps == (
            _step('merger', 1, 3, 2),
            _step('reviewer_comprehensive', 2, 2, 1, model='opus', budget_usd=12.0, max_turns=60),
            _step('merger', 4, 4, 1, layer=SourceLayer.ROLE_DEFAULT, model='opus'),
            _step('merger', 5, 5, 1),
        )
        assert (fold.cursor, fold.first_at, fold.newest_at) == (5, _t(1), _t(5))
        assert fold.complete
        assert fold.project_id == 'dark_factory'

    def test_rule_rows_feed_rule_spans_and_only_unmatched_config_layers_feed_steps(
        self, tmp_path: Path,
    ) -> None:
        store = _runs(tmp_path, [
            _decision(1, 'implementer'),
            _decision(2, 'implementer', layer='policy_rule', rule_id='large-plan-steps',
                      model='opus'),
            _decision(3, 'implementer'),
            _decision(4, 'implementer', layer='metadata_override', model='sonnet'),
            _decision(5, 'implementer', rule_id='retry-tier-up'),
            _decision(6, 'implementer', layer='policy_rule', rule_id='large-plan-steps',
                      model='opus'),
        ])

        fold = _fold(store)

        assert fold.steps == (_step('implementer', 1, 3, 2),)
        assert fold.rules == (
            RuleSpan(rule_id='large-plan-steps', first_event_id=2, first_at=_t(2),
                     last_at=_t(6), decisions=2),
            RuleSpan(rule_id='retry-tier-up', first_event_id=5, first_at=_t(5),
                     last_at=_t(5), decisions=1),
        )
        assert fold.cursor == 6

    def test_a_rule_whose_model_is_refused_still_sets_fields_and_splits_no_step(
        self, tmp_path: Path,
    ) -> None:
        """resolve_route applies a rule's effort, budget and turns even when it refuses the rule's
        model, so such a row keeps layer config or role_default but carries the rule's values."""
        store = _runs(tmp_path, [
            _decision(1, 'implementer'),
            _decision(2, 'implementer', rule_id='large-plan-steps', max_turns=120),
            _decision(3, 'implementer'),
            _decision(4, 'implementer', layer='role_default', model='opus', max_turns=120,
                      rule_id='large-plan-steps'),
            _decision(5, 'implementer'),
        ])

        fold = _fold(store)

        assert fold.steps == (_step('implementer', 1, 5, 3),)
        assert fold.rules == (RuleSpan(rule_id='large-plan-steps', first_event_id=2,
                                       first_at=_t(2), last_at=_t(4), decisions=2),)
        assert (routing_markers(fold, ()), availability_flags(fold)) == ([], [])

    def test_a_row_naming_an_unknown_layer_is_skipped_and_counted(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [
            _decision(1, 'merger'),
            _decision(2, 'merger', layer='retired_layer', model='opus'),
            _decision(3, 'merger'),
        ])

        fold = _fold(store)

        assert fold.skipped == 1
        assert fold.steps == (_step('merger', 1, 3, 2),)
        assert fold.steps[0].source_layer is SourceLayer.CONFIG
        assert fold.newest_at == _t(3)

    def test_an_absent_digest_never_splits_a_step_and_a_new_one_does(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [
            _decision(1, 'architect'),
            _decision(2, 'architect', digest='a' * 64),
            _decision(3, 'architect'),
            _decision(4, 'architect', digest='b' * 64),
            _decision(5, 'architect'),
        ])

        fold = _fold(store)

        assert fold.steps == (
            _step('architect', 1, 3, 3, digest='a' * 64),
            _step('architect', 4, 5, 2, digest='b' * 64),
        )
        assert fold.digest_since == _t(2)

    def test_no_digest_anywhere_leaves_digest_since_unset(self, tmp_path: Path) -> None:
        assert _fold(_runs(tmp_path, [_decision(1, 'merger')])).digest_since is None

    def test_the_events_role_column_wins_over_the_payload(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [_decision(1, 'payload_role')])
        with closing(sqlite3.connect(store)) as conn:
            conn.execute("UPDATE events SET role = 'merger'")
            conn.commit()

        assert [step.role for step in _fold(store).steps] == ['merger']

    def test_unreadable_rows_are_skipped_and_counted(self, tmp_path: Path) -> None:
        no_model = {'role': 'merger', 'effort': 'high', 'budget_usd': 8.0, 'max_turns': 100,
                    'source_layer': 'config', 'rule_id': None}
        naive = routing_decision(_t(4).replace(tzinfo=None).isoformat(), 'merger', model='fable',
                                 effort='high', budget_usd=8.0, max_turns=100)
        store = _runs(tmp_path, [
            _decision(1, 'merger'),
            (_t(2), None, 'routing_decision', '{not json'),
            (_t(3), None, 'routing_decision', no_model),
            naive,
            _decision(5, 'merger'),
            (_t(6), None, 'routing_decision', '[1, 2]'),
        ])

        fold = _fold(store)

        assert fold.skipped == 4
        assert fold.steps == (_step('merger', 1, 5, 2),)
        assert (fold.cursor, fold.newest_at) == (6, _t(5))

    def test_other_event_types_are_not_read(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [_decision(1, 'merger'), (_t(2), '7', 'merge_finalized', {})])

        fold = _fold(store)

        assert fold.steps == (_step('merger', 1, 1, 1),)
        assert fold.skipped == 0

    def test_folding_in_batches_until_complete_equals_one_full_fold(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [
            _decision(1, 'merger'), _decision(2, 'merger', digest='a' * 64),
            _decision(3, 'architect'), (_t(4), None, 'routing_decision', '{not json'),
            _decision(5, 'merger', rule_id='r1'), _decision(6, 'merger', model='opus'),
            _decision(7, 'architect', layer='policy_rule', rule_id='r2'),
        ])

        fold = _fold(store, batch=3)
        assert not fold.complete
        partial = [fold]
        while not fold.complete:
            fold = _fold(store, fold, batch=3)
            partial.append(fold)

        assert fold == _fold(store)
        assert len(partial) == 3

    def test_a_fold_reads_only_rows_after_its_cursor(self, tmp_path: Path) -> None:
        first_rows = [_decision(1, 'merger'), _decision(2, 'merger')]
        later_rows = [_decision(3, 'merger'), _decision(4, 'merger', model='opus')]
        store = _runs(tmp_path, first_rows)
        before = _fold(store)
        with closing(sqlite3.connect(store)) as conn:
            conn.execute("UPDATE events SET data = json_set(data, '$.model', 'haiku') WHERE id = 1")
            conn.commit()

        assert _fold(store, before) == before

        append_events(store, later_rows)

        assert _fold(store, before) == _fold(_runs(tmp_path, first_rows + later_rows, 'fresh'))


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


class _Steps:
    """Hand-built steps for one role, numbered in order; each lasts until the next begins."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.steps: list[RoutingStep] = []

    def add(
        self, first_at: str, decisions: int, *, layer: SourceLayer = SourceLayer.CONFIG,
        model: str = 'opus',
        effort: str = 'high', budget_usd: float = 12.0, max_turns: int = 60,
        digest: str | None = None,
    ) -> _Steps:
        event_id = 1000 * (len(self.steps) + 1) + len(self.role)
        at = _at(first_at)
        self.steps.append(RoutingStep(
            role=self.role, first_event_id=event_id, first_at=at,
            last_event_id=event_id + decisions - 1, last_at=at + timedelta(minutes=decisions),
            decisions=decisions, source_layer=layer, model=model, effort=effort,
            budget_usd=budget_usd, max_turns=max_turns, prompt_digest=digest,
        ))
        return self


def _routing_fold(*roles: _Steps, newest_at: str, project_id: str = 'dark_factory') -> RoutingFold:
    steps = tuple(sorted((step for role in roles for step in role.steps),
                         key=lambda step: step.first_event_id))
    return RoutingFold(
        project_id=project_id, cursor=max(step.last_event_id for step in steps), steps=steps,
        rules=(), first_at=min(step.first_at for step in steps), newest_at=_at(newest_at),
        digest_since=None, skipped=0, complete=True,
    )


A = 'a' * 64
LATER = '2026-10-05T00:00:00'


def _reviewer_flip(first_xhigh: str) -> _Steps:
    return (_Steps('reviewer_comprehensive')
            .add('2026-08-17T13:24:00', 4588)
            .add(first_xhigh, 121, effort='xhigh', max_turns=90))


def _change(at: str, path: str, *, applied: bool = True, event_id: int = 1) -> ConfigChange:
    return ConfigChange(event_id=event_id, at=_at(at), path=path, old='high', new='xhigh',
                        applied=applied)


class TestRoutingMarkers:
    def test_row_7_the_reviewer_flip_is_one_effort_marker_at_its_first_dispatch(self) -> None:
        fold = _routing_fold(_reviewer_flip('2026-11-01T08:58:42'), newest_at='2026-11-03T00:00:00')

        (marker,) = routing_markers(fold, ())

        assert marker.kind is MarkerKind.EFFORT
        assert marker.at == _at('2026-11-01T08:58:42')
        assert marker.source == MarkerSource(SourceType.ROUTING, role='reviewer_comprehensive')
        assert 'effort high -> xhigh' in marker.label
        assert 'turns 60 -> 90' in marker.label
        assert marker.label.startswith('reviewer_comprehensive')
        assert marker.scope == 'dark_factory'
        assert marker.series is None
        assert marker.marker_id == f'routing:reviewer_comprehensive:{fold.steps[1].first_event_id}'
        assert marker.ref == str(fold.steps[1].first_event_id)

    def test_row_7_each_project_dates_its_own_flip(self) -> None:
        fold = _routing_fold(_reviewer_flip('2026-11-01T10:37:33'),
                             newest_at='2026-11-03T00:00:00', project_id='reify')

        assert [(m.at, m.scope) for m in routing_markers(fold, ())] == [
            (_at('2026-11-01T10:37:33'), 'reify'),
        ]

    def test_row_8_a_one_dispatch_fallback_is_an_availability_flag_not_a_change(self) -> None:
        merger = (_Steps('merger')
                  .add('2026-07-17T08:32:00', 31, effort='max', budget_usd=5.0, max_turns=50)
                  .add('2026-09-12T00:14:00', 1, effort='max', budget_usd=8.0, max_turns=100)
                  .add('2026-09-12T08:52:00', 15, model='fable', effort='max', budget_usd=8.0,
                       max_turns=100)
                  .add('2026-09-25T06:13:00', 37, model='fable', budget_usd=8.0, max_turns=100)
                  .add('2026-10-27T13:46:00', 1, layer=SourceLayer.ROLE_DEFAULT, budget_usd=8.0,
                       max_turns=100)
                  .add('2026-10-27T20:12:00', 6, model='fable', budget_usd=8.0, max_turns=100))
        fold = _routing_fold(merger, newest_at='2026-11-01T00:00:00')

        markers = routing_markers(fold, ())

        assert [(m.at, m.kind) for m in markers] == [
            (_at('2026-09-12T08:52:00'), MarkerKind.MODEL),
            (_at('2026-09-25T06:13:00'), MarkerKind.EFFORT),
        ]
        assert availability_flags(fold) == [AvailabilityFlag(
            role='merger', from_at=_at('2026-10-27T13:46:00'), until_at=_at('2026-10-27T20:12:00'),
            decisions=1, model='opus',
        )]

    def test_an_unreturned_role_default_run_is_no_availability_flag(self) -> None:
        merger = (_Steps('merger').add('2026-10-01T00:00:00', 30, model='fable')
                  .add('2026-10-27T13:46:00', 1, layer=SourceLayer.ROLE_DEFAULT))

        assert availability_flags(_routing_fold(merger, newest_at='2026-10-28T00:00:00')) == []

    def test_the_reify_architect_flap_is_one_marker_at_the_settling_step(self) -> None:
        architect = (_Steps('architect')
                     .add('2026-07-16T15:08:00', 1067, effort='max', budget_usd=15.0, max_turns=120)
                     .add('2026-08-19T11:03:00', 1465, effort='max', budget_usd=25.0, max_turns=180)
                     .add('2026-09-24T07:32:00', 1, effort='xhigh', budget_usd=25.0, max_turns=180)
                     .add('2026-09-24T07:44:00', 2, effort='max', budget_usd=25.0, max_turns=180)
                     .add('2026-09-24T08:34:00', 1095, effort='xhigh', budget_usd=25.0,
                          max_turns=180))
        fold = _routing_fold(architect, newest_at='2026-11-01T00:00:00')

        assert [(m.at, m.kind, m.label) for m in routing_markers(fold, ())] == [
            (_at('2026-08-19T11:03:00'), MarkerKind.BUDGET,
             'architect: budget 15 -> 25, turns 120 -> 180'),
            (_at('2026-09-24T08:34:00'), MarkerKind.EFFORT, 'architect: effort max -> xhigh'),
        ]

    def test_a_step_of_fewer_than_three_decisions_is_dropped(self) -> None:
        role = _Steps('merger').add('2026-10-01T00:00:00', 40).add('2026-10-02T00:00:00', 2,
                                                                     effort='xhigh')

        assert routing_markers(_routing_fold(role, newest_at='2026-10-10T00:00:00'), ()) == []

    def test_a_short_step_followed_by_a_change_of_the_same_field_is_dropped(self) -> None:
        role = (_Steps('debugger').add('2026-10-01T00:00:00', 40)
                .add('2026-10-02T00:00:00', 5, effort='xhigh')
                .add('2026-10-02T02:00:00', 50, effort='max'))

        assert [(m.at, m.label) for m in routing_markers(
            _routing_fold(role, newest_at='2026-10-10T00:00:00'), ())] == [
            (_at('2026-10-02T02:00:00'), 'debugger: effort high -> max'),
        ]

    def test_a_short_step_followed_by_another_fields_change_collapses_into_one_marker(self) -> None:
        role = (_Steps('debugger').add('2026-10-01T00:00:00', 40)
                .add('2026-10-02T00:00:00', 3, effort='xhigh')
                .add('2026-10-02T02:00:00', 50, effort='xhigh', max_turns=90))

        assert [(m.at, m.kind, m.label) for m in routing_markers(
            _routing_fold(role, newest_at='2026-10-10T00:00:00'), ())] == [
            (_at('2026-10-02T00:00:00'), MarkerKind.EFFORT,
             'debugger: effort high -> xhigh, turns 60 -> 90'),
        ]

    def test_counted_steps_within_a_day_of_the_first_collapse(self) -> None:
        role = (_Steps('implementer').add('2026-10-01T00:00:00', 40)
                .add('2026-10-02T00:00:00', 3, effort='xhigh')
                .add('2026-10-02T03:00:00', 3, effort='xhigh', max_turns=90)
                .add('2026-10-02T06:00:00', 30, effort='xhigh', max_turns=90, budget_usd=20.0))

        (marker,) = routing_markers(_routing_fold(role, newest_at='2026-10-10T00:00:00'), ())

        assert marker.at == _at('2026-10-02T00:00:00')
        assert marker.label == 'implementer: effort high -> xhigh, budget 12 -> 20, turns 60 -> 90'

    def test_an_empty_diff_against_the_previous_counted_state_is_no_marker(self) -> None:
        role = (_Steps('merger').add('2026-10-01T00:00:00', 40)
                .add('2026-10-05T00:00:00', 40, layer=SourceLayer.ROLE_DEFAULT))

        assert routing_markers(_routing_fold(role, newest_at='2026-10-10T00:00:00'), ()) == []

    @pytest.mark.parametrize(('second_step', 'kind'), [
        (lambda role: role.add(LATER, 40, digest=A, model='fable', effort='xhigh'), MarkerKind.MODEL),
        (lambda role: role.add(LATER, 40, digest='b' * 64, effort='xhigh'), MarkerKind.EFFORT),
        (lambda role: role.add(LATER, 40, digest='b' * 64, budget_usd=20.0), MarkerKind.PROMPT),
        (lambda role: role.add(LATER, 40, digest=A, budget_usd=20.0, max_turns=90),
         MarkerKind.BUDGET),
        (lambda role: role.add(LATER, 40, digest=A, max_turns=90), MarkerKind.TURNS),
    ])
    def test_kind_is_the_first_changed_field_by_precedence(
        self, second_step: Callable[[_Steps], _Steps], kind: MarkerKind,
    ) -> None:
        role = second_step(_Steps('architect').add('2026-10-01T00:00:00', 40, digest=A))

        (marker,) = routing_markers(_routing_fold(role, newest_at='2026-10-10T00:00:00'), ())

        assert marker.kind is kind

    def test_a_digest_first_written_is_no_prompt_marker(self) -> None:
        role = (_Steps('architect').add('2026-10-01T00:00:00', 40)
                .add('2026-10-05T00:00:00', 40, digest='a' * 64))

        assert routing_markers(_routing_fold(role, newest_at='2026-10-10T00:00:00'), ()) == []

    def test_a_digest_change_folded_from_rows_is_a_prompt_marker(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [
            *(_decision(minute, 'architect') for minute in range(1, 4)),
            *(_decision(minute, 'architect', digest='a' * 64) for minute in range(4, 7)),
            *(_decision(minute, 'architect', digest='b' * 64) for minute in range(7, 10)),
            _decision(60 * 30, 'merger'),
        ])

        (marker,) = routing_markers(_fold(store), ())

        assert (marker.kind, marker.at) == (MarkerKind.PROMPT, _t(7))

    def test_a_role_with_one_step_has_no_marker(self) -> None:
        fold = _routing_fold(_Steps('merger').add('2026-10-01T00:00:00', 400),
                             newest_at='2026-10-10T00:00:00')

        assert routing_markers(fold, ()) == []


class TestRoutingCause:
    FIRST = '2026-11-01T08:58:42'

    def _markers(self, *changes: ConfigChange) -> list[Marker]:
        fold = _routing_fold(_reviewer_flip(self.FIRST), newest_at='2026-11-03T00:00:00')
        return routing_markers(fold, changes)

    def test_a_reload_of_a_changed_fields_path_within_48h_before_dates_the_marker(self) -> None:
        (marker,) = self._markers(_change('2026-10-31T09:00:00', 'effort.reviewer'))

        assert marker.at == _at('2026-10-31T09:00:00')
        assert marker.source == MarkerSource(SourceType.ROUTING, role='reviewer_comprehensive',
                                             path='effort.reviewer')

    def test_the_earliest_matching_change_wins_and_restart_required_counts(self) -> None:
        (marker,) = self._markers(
            _change('2026-11-01T07:00:00', 'effort.reviewer', event_id=2),
            _change('2026-10-31T12:00:00', 'max_turns.reviewer', applied=False, event_id=1),
        )

        assert (marker.at, marker.source.path) == (_at('2026-10-31T12:00:00'), 'max_turns.reviewer')

    @pytest.mark.parametrize('change', [
        _change('2026-11-01T09:00:00', 'effort.reviewer'),
        _change('2026-10-30T08:58:00', 'effort.reviewer'),
        _change('2026-11-01T08:00:00', 'effort.architect'),
        _change('2026-11-01T08:00:00', 'effort.reviewer_comprehensive'),
        _change('2026-11-01T08:00:00', 'budgets.reviewer'),
    ])
    def test_a_late_early_or_unrelated_change_does_not_date_the_marker(
        self, change: ConfigChange,
    ) -> None:
        (marker,) = self._markers(change)

        assert marker.at == _at(self.FIRST)
        assert marker.source.path is None

    def test_a_non_reviewer_role_matches_its_own_config_key(self) -> None:
        role = (_Steps('architect').add('2026-10-01T00:00:00', 40)
                .add('2026-10-05T00:00:00', 40, model='fable'))
        fold = _routing_fold(role, newest_at='2026-10-10T00:00:00')

        (marker,) = routing_markers(fold, (_change('2026-10-04T23:00:00', 'models.architect'),))

        assert marker.at == _at('2026-10-04T23:00:00')


class TestConfigChanges:
    def test_applied_and_restart_required_entries_become_changes(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [
            config_reload(_t(1), applied={'effort.reviewer': {'old': 'high', 'new': 'xhigh'}},
                          restart_required={'max_concurrent_tasks': {'old': 4, 'new': 6}}),
            (_t(2), None, 'config_reload', '{not json'),
            config_reload(_t(3), applied={'review.reviewer_briefing': 'not a change'}),
            _decision(4, 'merger'),
        ])

        with closing(connect_ro(store)) as conn:
            changes = read_config_changes(conn)

        assert changes == ConfigChanges(changes=(
            ConfigChange(event_id=1, at=_t(1), path='effort.reviewer', old='high', new='xhigh',
                         applied=True),
            ConfigChange(event_id=1, at=_t(1), path='max_concurrent_tasks', old=4, new=6,
                         applied=False),
        ), skipped=2)


SHA_A = 'a' * 40
SHA_B = 'b' * 40


def _derived(
    store: Path, snapshots: tuple[SnapshotIdentity, ...] = (), project_id: str = 'dark_factory',
) -> list[Marker]:
    with closing(connect_ro(store)) as conn:
        return derived_markers(conn, project_id, snapshots)


class TestOtherDerivedSources:
    def test_an_applied_briefing_change_is_a_reviewer_policy_marker_at_the_reload(
        self, tmp_path: Path,
    ) -> None:
        store = _runs(tmp_path, [config_reload(_t(1), applied={
            'review.reviewer_briefing': {'old': 'diff-only', 'new': 'd2'},
        })])

        (marker,) = _derived(store)

        assert (marker.kind, marker.at) == (MarkerKind.REVIEWER_POLICY, _t(1))
        assert marker.source == MarkerSource(SourceType.CONFIG_RELOAD, path='review.reviewer_briefing')
        assert 'diff-only -> d2' in marker.label

    def test_a_restart_required_briefing_or_an_unmatched_routing_reload_is_no_marker(
        self, tmp_path: Path,
    ) -> None:
        store = _runs(tmp_path, [
            config_reload(_t(1), restart_required={
                'review.reviewer_briefing': {'old': 'diff-only', 'new': 'd2'},
            }),
            config_reload(_t(2), applied={'models.architect': {'old': 'opus', 'new': 'fable'}}),
        ])

        assert _derived(store) == []

    @pytest.mark.parametrize(('quiet_days', 'expected'), [
        (6.9, [(_t(1), 'first')]),
        (7, [(_t(1), 'first'), (_t(2), 'last')]),
    ])
    def test_a_policy_rule_marks_its_appearance_and_after_a_quiet_week_its_last(
        self, tmp_path: Path, quiet_days: float, expected: list[tuple[datetime, str]],
    ) -> None:
        store = _runs(tmp_path, [
            _decision(1, 'implementer', layer='policy_rule', rule_id='retry-tier-up'),
            _decision(2, 'implementer', layer='policy_rule', rule_id='retry-tier-up'),
            _decision(2 + quiet_days * 24 * 60, 'merger'),
        ])

        markers = _derived(store)

        assert all(marker.kind is MarkerKind.ROUTING_RULE for marker in markers)
        assert all(marker.source == MarkerSource(SourceType.POLICY_RULE, rule='retry-tier-up')
                   for marker in markers)
        assert [(marker.at, marker.marker_id.rsplit(':', 1)[-1]) for marker in markers] == expected

    def test_code_sha_change_points_are_deploy_markers(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [_decision(1, 'merger')])
        write_runs_rows(store, [
            ('r5', _t(50), SHA_A),
            ('r1', _t(10), SHA_A),
            ('r2', _t(20), None),
            ('r3', _t(30), SHA_B),
            ('r4', _t(40), SHA_B),
        ])

        markers = _derived(store)

        assert [(m.kind, m.at, m.ref) for m in markers] == [
            (MarkerKind.DEPLOY, _t(30), SHA_B), (MarkerKind.DEPLOY, _t(50), SHA_A),
        ]
        assert all(m.source == MarkerSource(SourceType.DEPLOY) for m in markers)

    def test_no_runs_table_or_no_code_sha_column_is_no_deploy_marker(self, tmp_path: Path) -> None:
        bare = _runs(tmp_path, [_decision(1, 'merger')], 'bare')
        old = _runs(tmp_path, [_decision(1, 'merger')], 'old')
        with closing(sqlite3.connect(old)) as conn:
            conn.execute('CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL)')
            conn.execute("INSERT INTO runs VALUES ('r1', '2026-10-21T00:00:00+00:00')")
            conn.commit()

        assert _derived(bare) == []
        assert _derived(old) == []

    def test_a_new_model_id_per_alias_after_the_first_is_a_model_marker(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [_decision(1, 'merger')])
        write_invocations(store, [
            ('opus', 'claude-opus-5-1', _t(10)),
            ('sonnet', 'claude-sonnet-4', _t(20)),
            ('opus', 'claude-opus-5-5', _t(30)),
            ('opus', 'claude-opus-5-1', _t(40)),
            ('opus', None, _t(50)),
        ])

        (marker,) = _derived(store)

        assert (marker.kind, marker.at) == (MarkerKind.MODEL, _t(30))
        assert marker.source == MarkerSource(SourceType.MODEL_VERSION, alias='opus')
        assert marker.label == 'opus -> claude-opus-5-5'

    def test_no_invocations_table_is_no_model_marker(self, tmp_path: Path) -> None:
        assert _derived(_runs(tmp_path, [_decision(1, 'merger')])) == []

    def test_a_params_change_between_snapshots_is_an_instrument_marker(self, tmp_path: Path) -> None:
        snapshots = (
            SnapshotIdentity(as_of_sha=SHA_A, committed_at=_t(10), params={'a': 1, 'b': 2}),
            SnapshotIdentity(as_of_sha='c' * 40, committed_at=_t(20), params={'a': 1, 'b': 2}),
            SnapshotIdentity(as_of_sha=SHA_B, committed_at=_t(30), params={'a': 1, 'b': 3, 'c': 1}),
        )

        (marker,) = _derived(_runs(tmp_path, [_decision(1, 'merger')]), snapshots)

        assert (marker.kind, marker.at, marker.ref) == (MarkerKind.INSTRUMENT, _t(30), SHA_B)
        assert 'b' in marker.label and 'c' in marker.label and 'a' not in marker.label.split(':')[-1]

    def test_unreadable_source_rows_are_skipped_and_counted(self, tmp_path: Path) -> None:
        store = _runs(tmp_path, [_decision(1, 'merger'), (_t(2), None, 'config_reload', '{bad')])
        write_runs_rows(store, [('r1', '2026-10-21T00:00:00', SHA_A), ('r2', _t(5), SHA_B)])

        with closing(connect_ro(store)) as conn:
            derived = derive(_fold(store), conn, ())

        assert derived.skipped == 2
        assert derived.markers == ()


def _every_source(tmp_path: Path) -> Path:
    store = _runs(tmp_path, [
        *(_decision(minute, 'reviewer_comprehensive') for minute in range(1, 5)),
        config_reload(_t(5), applied={'effort.reviewer': {'old': 'high', 'new': 'xhigh'},
                                      'review.reviewer_briefing': {'old': 'a', 'new': 'b'}}),
        *(_decision(minute, 'reviewer_comprehensive', effort='xhigh') for minute in range(6, 10)),
        _decision(10, 'implementer', layer='policy_rule', rule_id='large-plan-steps'),
        _decision(60 * 24 * 9, 'merger'),
    ])
    write_runs_rows(store, [('r1', _t(1), SHA_A), ('r2', _t(7), SHA_B)])
    write_invocations(store, [('opus', 'claude-opus-5-1', _t(1)), ('opus', 'claude-opus-5-5', _t(8))])
    return store


class TestDerivedMarkers:
    SNAPSHOTS = (
        SnapshotIdentity(as_of_sha=SHA_A, committed_at=_t(1), params={'v': 1}),
        SnapshotIdentity(as_of_sha=SHA_B, committed_at=_t(9), params={'v': 2}),
    )

    def test_every_derived_marker_is_scoped_to_the_project_for_every_series(
        self, tmp_path: Path,
    ) -> None:
        markers = _derived(_every_source(tmp_path), self.SNAPSHOTS, project_id='reify')

        assert {marker.kind for marker in markers} == {
            MarkerKind.EFFORT, MarkerKind.REVIEWER_POLICY, MarkerKind.ROUTING_RULE,
            MarkerKind.DEPLOY, MarkerKind.MODEL, MarkerKind.INSTRUMENT,
        }
        assert {marker.scope for marker in markers} == {'reify'}
        assert {marker.series for marker in markers} == {None}
        assert markers == sorted(markers, key=lambda marker: (marker.at, marker.marker_id))

    def test_it_is_idempotent_and_equals_the_incremental_path(self, tmp_path: Path) -> None:
        store = _every_source(tmp_path)
        fold = RoutingFold.empty('dark_factory')
        with closing(connect_ro(store)) as conn:
            while not fold.complete:
                fold = fold_routing_decisions(conn, fold, batch=2)
            incremental = derive(fold, conn, self.SNAPSHOTS)

        full = _derived(store, self.SNAPSHOTS)

        assert full == _derived(store, self.SNAPSHOTS)
        assert list(incremental.markers) == full


AXIS = (
    AxisLanding(ordinal=11, landed_at=_t(10)),
    AxisLanding(ordinal=12, landed_at=_t(20)),
    AxisLanding(ordinal=13, landed_at=_t(30)),
)


class TestResolveOrdinals:
    @pytest.mark.parametrize(('minutes', 'ordinal'), [(5, 11), (15, 12), (20, 12), (30, 13), (31, None)])
    def test_a_marker_sits_at_the_first_landing_at_or_after_it(
        self, minutes: float, ordinal: int | None,
    ) -> None:
        marker = _marker(at=_t(minutes))

        (resolved,) = resolve_ordinals([marker], AXIS)

        assert (resolved.ordinal, resolved.until_ordinal) == (ordinal, None)
        assert marker.ordinal is None

    @pytest.mark.parametrize(('start', 'end', 'ordinals'), [
        (15, 30, (12, 13)),
        (10, 10, (11, 11)),
        (21, 29, (13, None)),
        (31, 40, (None, None)),
        (0, 5, (11, None)),
    ])
    def test_an_artefact_spans_the_landings_inside_its_range(
        self, start: float, end: float, ordinals: tuple[int | None, int | None],
    ) -> None:
        (resolved,) = resolve_ordinals([_artefact(at=_t(start), until=_t(end))], AXIS)

        assert (resolved.ordinal, resolved.until_ordinal) == ordinals

    def test_no_landings_resolve_nothing(self) -> None:
        (resolved,) = resolve_ordinals([_marker()], ())

        assert resolved.ordinal is None


NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
COUNTED = PromptCoverage(digest_since=None, roles_py_landings=4, counted_since=_t(0), reason=None)


def _deploy(minutes: float) -> Marker:
    return Marker(marker_id=f'deploy:r{minutes}', at=_t(minutes), kind=MarkerKind.DEPLOY,
                  scope='dark_factory', series=None, until=None, label='deploy abc',
                  source=MarkerSource(SourceType.DEPLOY), ref='abc')


def _state(
    *, complete: bool = True, refreshed_at: datetime = NOW - timedelta(minutes=5),
) -> DerivedState:
    merger = (_Steps('merger').add('2026-10-01T00:00:00', 30, model='fable')
              .add('2026-10-27T13:46:00', 1, layer=SourceLayer.ROLE_DEFAULT)
              .add('2026-10-27T20:12:00', 6, model='fable'))
    fold = replace(_routing_fold(merger, newest_at='2026-11-01T00:00:00'), complete=complete,
                   skipped=2)
    return DerivedState(
        fold=fold, markers=(_deploy(25), replace(EFFORT_MARKER, at=_t(12))), skipped=3,
        prompt_coverage=COUNTED, refreshed_at=refreshed_at,
    )


def _curated_set() -> CuratedMarkers:
    fleet = Marker(marker_id='curated:fleet', at=_t(1), kind=MarkerKind.REGIME, scope=None,
                   series=None, until=None, label='fleet regime',
                   source=MarkerSource(SourceType.CURATED, file='regime-markers.yaml'))
    return CuratedMarkers(markers=(
        fleet,
        _artefact(marker_id='curated:ours', scope='Dark-Factory', at=_t(15), until=_t(20)),
        replace(fleet, marker_id='curated:theirs', scope='reify'),
    ), parse_failures=2)


class TestMarkerStream:
    def test_it_serves_derived_and_in_scope_curated_markers_with_ordinals(self) -> None:
        stream = marker_stream('dark_factory', _state(), _curated_set(), AXIS)

        assert [(m.marker_id, m.ordinal, m.until_ordinal) for m in stream.markers] == [
            ('curated:fleet', 11, None),
            ('routing:reviewer_comprehensive:42', 12, None),
            ('curated:ours', 12, 12),
            ('deploy:r25', 13, None),
        ]
        assert stream.availability == (AvailabilityFlag(
            role='merger', from_at=_at('2026-10-27T13:46:00'), until_at=_at('2026-10-27T20:12:00'),
            decisions=1, model='opus',
        ),)
        assert (stream.curated_parse_failures, stream.skipped_records) == (2, 5)
        assert stream.prompt_coverage == COUNTED

    def test_its_regime_events_are_the_boundaries_and_exclusions_only(self) -> None:
        stream = marker_stream('dark_factory', _state(), _curated_set(), AXIS)

        assert [event.event_id for event in stream.regime_events()] == [
            'curated:fleet', 'routing:reviewer_comprehensive:42', 'curated:ours',
        ]
        assert [event.artefact for event in stream.regime_events()] == [False, False, True]

    def test_its_wire_shape(self) -> None:
        wire = marker_stream('dark_factory', _state(), _curated_set(), AXIS).to_wire()

        assert set(wire) == {'markers', 'availability', 'prompt_coverage',
                             'curated_parse_failures', 'skipped_records'}
        assert wire['availability'] == [{
            'role': 'merger', 'from_at': '2026-10-27T13:46:00+00:00',
            'until_at': '2026-10-27T20:12:00+00:00', 'decisions': 1, 'model': 'opus',
        }]
        assert wire['prompt_coverage'] == {
            'state': 'not_yet_written', 'since': None, 'roles_py_landings': 4,
            'counted_since': _t(0).isoformat(), 'reason': None,
        }
        assert isinstance(wire['markers'], list) and len(wire['markers']) == 4


class TestPromptCoverage:
    def test_a_written_digest_serves_its_first_appearance_and_no_count(self) -> None:
        coverage = PromptCoverage(digest_since=_t(5), roles_py_landings=None, counted_since=None,
                                  reason=None)

        assert coverage.to_wire() == {'state': 'written', 'since': _t(5).isoformat(),
                                      'roles_py_landings': None, 'counted_since': None,
                                      'reason': None}

    def test_an_unwritten_digest_without_a_count_serves_why(self) -> None:
        coverage = PromptCoverage(digest_since=None, roles_py_landings=None, counted_since=None,
                                  reason='factory repo unreadable')

        assert coverage.to_wire()['state'] == 'not_yet_written'
        assert coverage.to_wire()['reason'] == 'factory repo unreadable'

    @pytest.mark.parametrize(('digest_since', 'landings', 'since', 'reason'), [
        (_t(5), 3, _t(0), None),
        (_t(5), None, None, 'why'),
        (None, None, None, None),
        (None, 3, None, None),
        (None, 3, _t(0), 'both'),
        (None, -1, _t(0), None),
    ])
    def test_it_refuses_an_inconsistent_coverage(
        self, digest_since: datetime | None, landings: int | None, since: datetime | None,
        reason: str | None,
    ) -> None:
        with pytest.raises(MarkerInvariantError) as err:
            PromptCoverage(digest_since=digest_since, roles_py_landings=landings,
                           counted_since=since, reason=reason)

        assert err.value.invariant is MarkerInvariant.PROMPT_COVERAGE


class TestMarkerDatum:
    def _datum(self, stored: StoredMarkerState | None):
        datum = marker_datum('dark_factory', stored, _curated_set(), AXIS, now=NOW,
                             freshness_bound_seconds=1200)
        validate_datum(datum, NOW)
        return datum

    def test_nothing_derived_yet_is_unknown_naming_the_project(self) -> None:
        datum = self._datum(None)

        assert (datum.state, datum.value) == (DatumState.UNKNOWN, None)
        assert 'dark_factory' in (datum.reason or '')
        assert 'not been derived' in (datum.reason or '')

    def test_nothing_derived_and_a_failure_is_unknown_naming_the_failure(self) -> None:
        datum = self._datum(StoredMarkerState(derived=None, attempted_at=NOW,
                                              failure='runs.db is absent'))

        assert datum.state is DatumState.UNKNOWN
        assert 'runs.db is absent' in (datum.reason or '')

    def test_a_failed_newest_attempt_serves_the_previous_state_stale(self) -> None:
        state = _state()
        datum = self._datum(StoredMarkerState(derived=state, attempted_at=NOW,
                                              failure='over its 30 s budget'))

        assert datum.state is DatumState.STALE
        assert 'over its 30 s budget' in (datum.reason or '')
        assert datum.as_of == state.refreshed_at
        assert isinstance(datum.value, MarkerStream)

    def test_an_incomplete_fold_is_stale(self) -> None:
        state = _state(complete=False)
        datum = self._datum(StoredMarkerState(derived=state, attempted_at=state.refreshed_at,
                                              failure=None))

        assert datum.state is DatumState.STALE
        assert 'incomplete' in (datum.reason or '')

    def test_a_state_past_the_bound_is_stale(self) -> None:
        state = _state(refreshed_at=NOW - timedelta(seconds=1201))
        datum = self._datum(StoredMarkerState(derived=state, attempted_at=state.refreshed_at,
                                              failure=None))

        assert datum.state is DatumState.STALE
        assert 'freshness bound' in (datum.reason or '')

    def test_a_complete_recent_state_is_fresh(self) -> None:
        state = _state()
        datum = self._datum(StoredMarkerState(derived=state, attempted_at=state.refreshed_at,
                                              failure=None))

        assert (datum.state, datum.as_of, datum.reason) == (DatumState.FRESH, state.refreshed_at, None)
        assert datum.freshness_bound_seconds == 1200
