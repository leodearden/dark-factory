"""The quality marker stream: vocabulary, curated reader, routing fold, derivation and serving."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from _quality_fixture import append_events, routing_decision
from _runs_db_fixture import EventRow, write_runs_db
from shared.quality_series.stats import RegimeEvent
from shared.tasks_db_readonly import connect_ro

from dashboard.data import quality_markers
from dashboard.data.quality_markers import (
    CuratedMarkers,
    Marker,
    MarkerInvariant,
    MarkerInvariantError,
    MarkerKind,
    MarkerSource,
    RoutingFold,
    RoutingStep,
    RuleSpan,
    SourceType,
    fold_routing_decisions,
    load_curated_markers,
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
    last_id: int | None = None, layer: str = 'config', model: str = 'fable', effort: str = 'high',
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
            _step('merger', 4, 4, 1, layer='role_default', model='opus'),
            _step('merger', 5, 5, 1),
        )
        assert (fold.cursor, fold.first_at, fold.newest_at) == (5, _t(1), _t(5))
        assert fold.complete
        assert fold.project_id == 'dark_factory'

    def test_rule_rows_feed_rule_spans_and_only_config_layers_feed_steps(self, tmp_path: Path) -> None:
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

        assert fold.steps == (_step('implementer', 1, 5, 3),)
        assert fold.rules == (
            RuleSpan(rule_id='large-plan-steps', first_event_id=2, first_at=_t(2),
                     last_at=_t(6), decisions=2),
            RuleSpan(rule_id='retry-tier-up', first_event_id=5, first_at=_t(5),
                     last_at=_t(5), decisions=1),
        )
        assert fold.cursor == 6

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
