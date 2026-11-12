"""/api/v2/dashboard/quality: its landing-window vocabulary, the served echo, and the route."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _dt_helpers import fixed_clock, installed_clock
from _quality_fixture import GitRepo, config_reload, routing_decision
from _quality_rows import (
    NO_OUTCOMES,
    history,
    write_landing_events,
    write_marker_state_db,
    write_quality_db,
)
from _runs_db_fixture import EventRow
from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import GitCrossCheck
from shared.quality_series.rows import LandingRow, Outcomes

from dashboard.api.window import (
    ServedLandingWindow,
    ServedWindow,
    _parse_window,
    parse_landing_window,
    with_window,
)
from dashboard.config import DashboardConfig
from dashboard.data import quality_markers
from dashboard.data.quality_marker_sampler import refresh_markers
from dashboard.data.quality_store import QualitySample
from dashboard.loops import _QualityStore

LANDINGS: dict[str, int | None] = {'100': 100, '300': 300, 'all': None}
"""A landing vocabulary for the parser alone; the route's own is driven through the route."""


@pytest.mark.parametrize(
    ('params', 'expected'),
    [
        ({'window': '100'}, ServedLandingWindow(requested='100', served='100', landings=100)),
        ({'window': '300'}, ServedLandingWindow(requested='300', served='300', landings=300)),
        ({'window': 'all'}, ServedLandingWindow(requested='all', served='all', landings=None)),
        ({}, ServedLandingWindow(requested='100', served='100', landings=100)),
        ({'window': ''}, ServedLandingWindow(requested='100', served='100', landings=100)),
        ({'window': '7d'}, ServedLandingWindow(requested='7d', served='100', landings=100)),
        ({'window': 'weird'}, ServedLandingWindow(requested='weird', served='100', landings=100)),
    ],
)
def test_the_landing_window_serves_its_vocabulary_and_declines_the_rest(
    params: dict[str, str], expected: ServedLandingWindow,
) -> None:
    assert parse_landing_window(params, vocabulary=LANDINGS, default='100') == expected


def test_the_landing_window_refuses_a_default_its_vocabulary_cannot_serve() -> None:
    with pytest.raises(ValueError, match="'7d'"):
        parse_landing_window({'window': '100'}, vocabulary=LANDINGS, default='7d')


def test_a_served_landing_window_names_landings_not_days() -> None:
    window = parse_landing_window({'window': 'all'}, vocabulary=LANDINGS, default='100')

    assert window.to_wire() == {'requested': 'all', 'served': 'all', 'landings': None}
    assert with_window({'QUALITY': {}}, window) == {'QUALITY': {}, 'WINDOW': window.to_wire()}


def test_the_day_vocabulary_is_untouched() -> None:
    assert _parse_window({'window': '7d'}) == ServedWindow(requested='7d', served='7d', days=7)


# --- the route ------------------------------------------------------------------------------

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
URL = '/api/v2/dashboard/quality'
DATUM_KEYS = {'value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'}
STORED = 120


STORED_START = NOW - timedelta(days=1) - timedelta(hours=2 * (STORED - 1))


@contextmanager
def _seeded(
    client: Any, tmp_path: Path, extra_events: Iterable[EventRow] = (),
    outcome: Callable[[int], Outcomes] = lambda ordinal: NO_OUTCOMES,
    modules: Callable[[int], tuple[ModuleLines, ...]] = lambda ordinal: (),
    units: Sequence[int] | None = None,
) -> Iterator[Any]:
    """The app over a root named 'dark_factory' holding STORED sampled landings, beside an
    unsampled 'other-proj', serving at NOW; runs.db also holds *extra_events* verbatim, and
    *outcome* and *modules* give each landing's outcomes and modules by ordinal, *units* its
    verify units as _quality_rows.history takes them."""
    root, other = tmp_path / 'dark_factory', tmp_path / 'other-proj'
    root.mkdir()
    other.mkdir()
    config = DashboardConfig(project_root=root, known_project_roots=[other])
    rows = history(STORED, start=STORED_START, outcome=outcome, modules=modules, units=units)
    write_landing_events(root, rows, extra_events)
    sample = QualitySample.value(
        'dark_factory', NOW - timedelta(minutes=5), cursor_sha=rows[-1].landing.merge_sha,
        landings_sampled=STORED, landings_deferred=0, git_check=GitCrossCheck(0, 0, 0),
    )
    asyncio.run(write_quality_db(config.quality_db, rows, [sample]))
    client.app.state.config = config
    with installed_clock(client.app, fixed_clock(NOW)):
        yield client


@pytest.fixture
def quality_client(client, tmp_path: Path) -> Iterator[Any]:
    with _seeded(client, tmp_path) as seeded:
        yield seeded


def _get(client: Any, query: str) -> dict[str, Any]:
    response = client.get(f'{URL}?{query}')
    assert response.status_code == 200, response.text
    return response.json()


def test_the_route_serves_the_quality_series_with_its_window(quality_client) -> None:
    body = _get(quality_client, 'project=dark_factory&window=100')

    assert set(body) == {'QUALITY', 'QUALITY_MARKERS', 'QUALITY_SNAPSHOTS', 'WINDOW', 'served_at'}
    assert body['served_at'] == NOW.isoformat()
    assert set(body['QUALITY']) == DATUM_KEYS
    assert body['QUALITY']['state'] == 'fresh'
    assert len(body['QUALITY']['value']['axis']) == 100
    assert body['WINDOW'] == {'requested': '100', 'served': '100', 'landings': 100}


@pytest.mark.parametrize(('label', 'landings'), [('300', 300), ('all', None)])
def test_the_route_serves_every_window_its_chip_offers(
    quality_client, label: str, landings: int | None,
) -> None:
    body = _get(quality_client, f'project=dark_factory&window={label}')

    assert body['WINDOW'] == {'requested': label, 'served': label, 'landings': landings}
    assert len(body['QUALITY']['value']['axis']) == STORED


def test_snapshots_are_unknown_until_their_producer_lands(quality_client) -> None:
    snapshots = _get(quality_client, 'project=dark_factory&window=100')['QUALITY_SNAPSHOTS']

    assert set(snapshots) == DATUM_KEYS
    assert (snapshots['state'], snapshots['value'], snapshots['as_of']) == ('unknown', None, None)
    assert snapshots['reason'].startswith('no producer has written here')
    assert 'snapshot' in snapshots['reason']


def test_markers_never_derived_for_a_project_are_unknown_saying_so(quality_client) -> None:
    markers = _get(quality_client, 'project=dark_factory&window=100')['QUALITY_MARKERS']

    assert set(markers) == DATUM_KEYS
    assert (markers['state'], markers['value'], markers['as_of']) == ('unknown', None, None)
    assert 'markers have not been derived for dark_factory' in markers['reason']


def test_markers_for_a_token_naming_no_root_are_unknown_naming_it(quality_client) -> None:
    markers = _get(quality_client, 'project=nowhere')['QUALITY_MARKERS']

    assert (markers['state'], markers['value']) == ('unknown', None)
    assert 'nowhere' in markers['reason']


# --- the marker stream through the route (PRD sketch rows 6-8) ------------------------------

FLIP_START = datetime(2026, 10, 31, 0, 0, tzinfo=UTC)
DF_RELOAD = datetime(2026, 11, 1, 8, 50, tzinfo=UTC)
DF_FLIP = datetime(2026, 11, 1, 8, 58, 42, tzinfo=UTC)
REIFY_FLIP = datetime(2026, 11, 1, 10, 37, 33, tzinfo=UTC)
FALLBACK = datetime(2026, 10, 27, 13, 46, tzinfo=UTC)
RETURNED = datetime(2026, 10, 27, 20, 12, tzinfo=UTC)


def _reviewer(at: datetime, *, effort: str, max_turns: int) -> EventRow:
    return routing_decision(at, 'reviewer_comprehensive', model='opus', effort=effort,
                            budget_usd=12.0, max_turns=max_turns)


def _merger(at: datetime, *, layer: str = 'config', model: str = 'fable') -> EventRow:
    return routing_decision(at, 'merger', model=model, effort='high', budget_usd=8.0,
                            max_turns=100, source_layer=layer)


def _reviewer_flip(first_xhigh: datetime) -> list[EventRow]:
    """Sketch row 7: high/60 dispatches, then a persisting xhigh/90 step from *first_xhigh*."""
    before = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
    return [
        *(_reviewer(before + timedelta(hours=h), effort='high', max_turns=60) for h in range(4)),
        *(_reviewer(first_xhigh + timedelta(hours=h), effort='xhigh', max_turns=90)
          for h in (0, 1, 2, 40)),
    ]


MERGER_FALLBACK: list[EventRow] = [
    *(_merger(datetime(2026, 10, 26, 10, tzinfo=UTC) + timedelta(hours=h)) for h in range(5)),
    _merger(FALLBACK, layer='role_default', model='opus'),
    *(_merger(RETURNED + timedelta(hours=h)) for h in range(6)),
]
"""Sketch row 8: one role_default dispatch between two equal config steps."""

DF_EVENTS: list[EventRow] = [
    *MERGER_FALLBACK,
    *_reviewer_flip(DF_FLIP)[:4],
    config_reload(DF_RELOAD, applied={'effort.reviewer': {'old': 'high', 'new': 'xhigh'}}),
    *_reviewer_flip(DF_FLIP)[4:],
]
REIFY_EVENTS: list[EventRow] = _reviewer_flip(REIFY_FLIP)


FLIP_LANDINGS = 48


def _every_third_flaky(ordinal: int) -> Outcomes:
    return replace(NO_OUTCOMES, flake_confirmed=ordinal % 3 == 0, flake_real_red=False,
                   flake_rows=1)


def _project(root: Path, project_id: str, events: list[EventRow]) -> list[LandingRow]:
    """A git checkout whose runs.db records 48 landings across 2026-11-01 and *events*; every
    landing has a flake verdict, every third one confirmed."""
    GitRepo(root).init()
    rows = history(FLIP_LANDINGS, start=FLIP_START, project_id=project_id,
                   outcome=_every_third_flaky)
    write_landing_events(root, rows, events)
    return rows


async def _refresh(config: DashboardConfig, at: datetime, factory: Path, curated: Path) -> None:
    async with _QualityStore(config.quality_db) as store:
        await refresh_markers(store.connection, config.all_project_roots, now=at,
                              curated_path=curated, factory_repo=factory)


@pytest.fixture
def derived_client(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """df and reify, each with stored landings and its own runs.db, after one marker tick; df's
    landings also have a value sample."""
    no_curated = tmp_path / 'no-curated-markers.yaml'
    monkeypatch.setattr(quality_markers, 'CURATED_MARKERS_PATH', no_curated)
    df = _project(tmp_path / 'dark_factory', 'dark_factory', DF_EVENTS)
    reify = _project(tmp_path / 'reify', 'reify', REIFY_EVENTS)
    config = DashboardConfig(project_root=tmp_path / 'dark_factory',
                             known_project_roots=[tmp_path / 'reify'])
    df_sample = QualitySample.value(
        'dark_factory', NOW - timedelta(minutes=5), cursor_sha=df[-1].landing.merge_sha,
        landings_sampled=len(df), landings_deferred=0, git_check=GitCrossCheck(0, 0, 0),
    )
    asyncio.run(write_quality_db(config.quality_db, [*df, *reify], [df_sample]))
    asyncio.run(_refresh(config, NOW - timedelta(minutes=5), tmp_path / 'not-a-repo', no_curated))
    client.app.state.config = config
    with installed_clock(client.app, fixed_clock(NOW)):
        yield client


def _first_landing_at_or_after(at: datetime) -> int:
    landed_at = [row.landing.landed_at for row in history(FLIP_LANDINGS, start=FLIP_START)]
    return bisect_left(landed_at, at)


def _routing(markers: dict[str, Any]) -> list[dict[str, Any]]:
    return [marker for marker in markers['value']['markers']
            if marker['source']['type'] == 'routing']


def test_row_7_each_project_serves_its_own_flip_dated_from_its_own_runs_db(
    derived_client,
) -> None:
    df = _get(derived_client, 'project=dark_factory')['QUALITY_MARKERS']
    reify = _get(derived_client, 'project=reify')['QUALITY_MARKERS']

    assert (df['state'], reify['state']) == ('fresh', 'fresh')
    [df_flip] = _routing(df)
    [reify_flip] = _routing(reify)
    assert (df_flip['kind'], df_flip['at'], df_flip['scope']) == (
        'effort', DF_RELOAD.isoformat(), 'dark_factory')
    assert df_flip['source'] == {'type': 'routing', 'role': 'reviewer_comprehensive',
                                 'path': 'effort.reviewer'}
    assert df_flip['ordinal'] == _first_landing_at_or_after(DF_RELOAD)
    assert (reify_flip['kind'], reify_flip['at'], reify_flip['scope']) == (
        'effort', REIFY_FLIP.isoformat(), 'reify')
    assert reify_flip['source'] == {'type': 'routing', 'role': 'reviewer_comprehensive'}
    assert reify_flip['ordinal'] == _first_landing_at_or_after(REIFY_FLIP)


def test_a_derived_routing_marker_splits_the_quality_series_at_its_landing(
    derived_client,
) -> None:
    body = _get(derived_client, 'project=dark_factory&window=all')

    [flip] = _routing(body['QUALITY_MARKERS'])
    boundaries = [marker['id'] for marker in body['QUALITY_MARKERS']['value']['markers']
                  if marker['kind'] not in {'deploy', 'instrument', 'artefact'}]
    assert boundaries == [flip['id']]
    assert flip['id'].startswith('routing:reviewer_comprehensive:')
    [comparison] = [comparison for comparison in body['QUALITY']['value']['before_after']
                    if comparison['metric'] == 'flakes_confirmed']
    assert comparison['marker_id'] == flip['id']
    assert (comparison['n_before'], comparison['n_after']) == (
        flip['ordinal'], FLIP_LANDINGS - flip['ordinal'])


def test_row_8_a_one_dispatch_fallback_is_availability_not_a_marker(derived_client) -> None:
    value = _get(derived_client, 'project=dark_factory')['QUALITY_MARKERS']['value']

    assert [m for m in value['markers'] if m['source'].get('role') == 'merger'] == []
    assert value['availability'] == [{
        'role': 'merger', 'from_at': FALLBACK.isoformat(), 'until_at': RETURNED.isoformat(),
        'decisions': 1, 'model': 'opus',
    }]


def test_the_marker_value_carries_prompt_coverage_and_the_curated_failures(
    derived_client,
) -> None:
    value = _get(derived_client, 'project=dark_factory')['QUALITY_MARKERS']['value']

    assert set(value) == {'markers', 'availability', 'prompt_coverage',
                          'curated_parse_failures', 'skipped_records'}
    assert value['prompt_coverage']['state'] == 'not_yet_written'
    assert value['prompt_coverage']['reason'].startswith('GitUnavailable: ')
    assert (value['curated_parse_failures'], value['skipped_records']) == (0, 0)


ARTEFACT_DAY = (STORED_START + timedelta(days=5)).date()
ARTEFACT_ORDINALS = frozenset(
    ordinal for ordinal in range(STORED)
    if (STORED_START + timedelta(hours=2 * ordinal)).date() == ARTEFACT_DAY
)
CURATED_ARTEFACT = f"""\
- date: {ARTEFACT_DAY.isoformat()}
  until: {ARTEFACT_DAY.isoformat()}
  kind: artefact
  project: dark_factory
  series: [flakes_confirmed]
  label: "fixture: a disk-full day of unconfirmable flakes"
"""


def _flaky(ordinal: int) -> Outcomes:
    confirmed = ordinal in ARTEFACT_ORDINALS or ordinal % 10 == 0
    return replace(NO_OUTCOMES, flake_confirmed=confirmed, flake_real_red=False, flake_rows=1)


def _rate_outside_the_artefact() -> float:
    outside = [row.outcomes.flake_confirmed is True
               for row in history(STORED, start=STORED_START, outcome=_flaky)
               if row.landing.ordinal not in ARTEFACT_ORDINALS]
    return sum(outside) / len(outside)


@pytest.fixture
def curated_artefact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    curated = tmp_path / 'regime-markers.yaml'
    curated.write_text(CURATED_ARTEFACT)
    monkeypatch.setattr(quality_markers, 'CURATED_MARKERS_PATH', curated)


def _store_an_empty_derivation(config: DashboardConfig) -> None:
    asyncio.run(write_marker_state_db(config.quality_db, 'dark_factory', (),
                                      refreshed_at=NOW - timedelta(minutes=5)))


def test_row_6_a_curated_artefact_is_excluded_from_the_baseline_and_served_as_a_range(
    client, tmp_path: Path, curated_artefact: None,
) -> None:
    with _seeded(client, tmp_path, outcome=_flaky) as seeded:
        _store_an_empty_derivation(seeded.app.state.config)
        body = _get(seeded, 'project=dark_factory&window=all')

    assert len(ARTEFACT_ORDINALS) == 12
    baseline = body['QUALITY']['value']['metrics']['flakes_confirmed']['baseline']
    assert baseline['p0'] == pytest.approx(_rate_outside_the_artefact())
    [artefact] = body['QUALITY_MARKERS']['value']['markers']
    assert artefact['kind'] == 'artefact'
    assert (artefact['ordinal'], artefact['until_ordinal']) == (
        min(ARTEFACT_ORDINALS), max(ARTEFACT_ORDINALS))


def test_before_the_first_derivation_a_curated_artefact_is_still_excluded(
    client, tmp_path: Path, curated_artefact: None,
) -> None:
    with _seeded(client, tmp_path, outcome=_flaky) as seeded:
        body = _get(seeded, 'project=dark_factory&window=all')

    assert body['QUALITY_MARKERS']['state'] == 'unknown'
    baseline = body['QUALITY']['value']['metrics']['flakes_confirmed']['baseline']
    assert baseline['p0'] == pytest.approx(_rate_outside_the_artefact())


def test_a_label_and_a_project_id_spelling_resolve_to_the_same_root(quality_client) -> None:
    by_id = _get(quality_client, 'project=dark_factory')['QUALITY']
    dashed = _get(quality_client, 'project=dark-factory')['QUALITY']
    other_label = _get(quality_client, 'project=other-proj')['QUALITY']
    other_id = _get(quality_client, 'project=other_proj')['QUALITY']

    assert by_id == dashed
    assert by_id['state'] == 'fresh'
    assert other_label == other_id
    assert other_label['state'] == 'unknown'
    assert 'other_proj' in other_label['reason']


def test_a_project_id_resolves_whatever_separator_runs_the_root_basename_holds(
    client, tmp_path: Path,
) -> None:
    root = tmp_path / 'twin--dash-'
    root.mkdir()
    client.app.state.config = DashboardConfig(project_root=root)
    with installed_clock(client.app, fixed_clock(NOW)):
        by_label = _get(client, 'project=twin--dash-')['QUALITY']
        by_id = _get(client, 'project=twin_dash')['QUALITY']

    assert by_id == by_label
    assert 'no configured project root' not in by_id['reason']


def test_an_unconfigured_project_is_unknown_naming_the_token(quality_client) -> None:
    quality = _get(quality_client, 'project=nowhere')['QUALITY']

    assert (quality['state'], quality['value']) == ('unknown', None)
    assert 'nowhere' in quality['reason']


def test_a_missing_project_is_unknown_saying_none_was_named(quality_client) -> None:
    quality = _get(quality_client, 'window=100')['QUALITY']

    assert (quality['state'], quality['value']) == ('unknown', None)
    assert 'no project' in quality['reason']


def test_a_declined_window_echoes_the_request_and_serves_the_default(quality_client) -> None:
    body = _get(quality_client, 'project=dark_factory&window=7d')

    assert body['WINDOW'] == {'requested': '7d', 'served': '100', 'landings': 100}
    assert len(body['QUALITY']['value']['axis']) == 100


PENDING_TRAIN: EventRow = (NOW - timedelta(hours=3), '9001', 'train_merged', {'verdict': 'pending'})
"""Neither a coalesced nor an attributed train: the landing rule refuses it."""


def test_a_train_event_the_landing_rule_refuses_is_served_stale(client, tmp_path: Path) -> None:
    with _seeded(client, tmp_path, [PENDING_TRAIN]) as seeded:
        quality = _get(seeded, 'project=dark_factory&window=100')['QUALITY']

    assert quality['state'] == 'stale'
    assert 'UnrecognisedTrainEvent' in quality['reason']
    assert len(quality['value']['axis']) == 100


def test_an_unexpected_read_error_is_served_unknown_and_logged_as_a_bug(
    client, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with _seeded(client, tmp_path) as seeded:
        with closing(sqlite3.connect(seeded.app.state.config.quality_db)) as conn:
            conn.execute("UPDATE landings SET source = 'bogus' "
                         "WHERE rowid = (SELECT MIN(rowid) FROM landings)")
            conn.commit()
        with caplog.at_level(logging.WARNING, logger='dashboard.api.quality'):
            body = _get(seeded, 'project=dark_factory&window=100')

    assert set(body) == {'QUALITY', 'QUALITY_MARKERS', 'QUALITY_SNAPSHOTS', 'WINDOW', 'served_at'}
    quality = body['QUALITY']
    assert set(quality) == DATUM_KEYS
    assert (quality['state'], quality['value']) == ('unknown', None)
    assert 'unexpected error' in quality['reason']
    assert 'ValueError' in quality['reason']
    [record] = [r for r in caplog.records if r.name == 'dashboard.api.quality']
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None
    assert 'BUG, not an outage' in record.getMessage()


def test_a_quality_db_that_is_not_a_database_is_served_unknown_naming_the_error(
    client, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with _seeded(client, tmp_path) as seeded:
        seeded.app.state.config.quality_db.write_bytes(b'not a sqlite database' * 64)
        with caplog.at_level(logging.WARNING, logger='dashboard.api.quality'):
            body = _get(seeded, 'project=dark_factory&window=100')

    for key in ('QUALITY', 'QUALITY_MARKERS'):
        assert (body[key]['state'], body[key]['value']) == ('unknown', None)
        assert body[key]['reason'].startswith('quality.db unreadable: DatabaseError')
    records = [r for r in caplog.records if r.name == 'dashboard.api.quality']
    assert len(records) == 2, 'one WARNING per datum that read quality.db'
    assert all(record.exc_info is not None for record in records)


# --- the module drill-down: ?module= keeps the landings touching one module ------------------

DRILL_TRAIN = range(61, 64)
"""The ordinals of one coalesced train whose members all touch the dashboard module."""
DRILL_UNITS = [1] * DRILL_TRAIN.start + [len(DRILL_TRAIN)] + [1] * (STORED - DRILL_TRAIN.stop)


def _touches_dashboard(ordinal: int) -> bool:
    return ordinal % 3 == 0 or ordinal in DRILL_TRAIN


def _drill_modules(ordinal: int) -> tuple[ModuleLines, ...]:
    """Every third landing and the train touch dashboard; every other landing, and every fifth
    one besides, touches orchestrator."""
    dashboard = (ModuleLines('dashboard', 40),) if _touches_dashboard(ordinal) else ()
    elsewhere = not _touches_dashboard(ordinal) or ordinal % 5 == 0
    return (*dashboard, *((ModuleLines('orchestrator', 25),) if elsewhere else ()))


def _rejected_in_dashboard(ordinal: int) -> Outcomes:
    rejected = _touches_dashboard(ordinal)
    return replace(NO_OUTCOMES, rejected=rejected, rejection_events=int(rejected))


DASHBOARD_ORDINALS = [ordinal for ordinal in range(STORED) if _touches_dashboard(ordinal)]
UNSAMPLED: EventRow = (NOW - timedelta(hours=1), '9100', 'merge_finalized',
                       {'state': 'done', 'merge_sha': 'sha-not-yet-sampled'})
"""A landing runs.db records after the sampled cursor that quality.db does not hold."""


@pytest.fixture
def drill_client(client, tmp_path: Path) -> Iterator[Any]:
    with _seeded(client, tmp_path, [UNSAMPLED], outcome=_rejected_in_dashboard,
                 modules=_drill_modules, units=DRILL_UNITS) as seeded:
        yield seeded


def _metric_arrays(metric: dict[str, Any]) -> dict[str, list[Any]]:
    arrays = {'points': metric['points'],
              **{f'rolling.{name}': values for name, values in metric['rolling'].items()}}
    if metric['ewma'] is not None:
        arrays.update({f'ewma.{name}': values for name, values in metric['ewma'].items()})
    return arrays


def test_a_module_serves_the_axis_of_the_landings_touching_it(drill_client) -> None:
    value = _get(drill_client, 'project=dark_factory&module=dashboard&window=all')['QUALITY']['value']

    ordinals = [point['ordinal'] for point in value['axis']]
    assert ordinals == DASHBOARD_ORDINALS
    assert ordinals != list(range(ordinals[0], ordinals[0] + len(ordinals))), 'a gapped subset'
    assert set(DRILL_TRAIN) <= set(ordinals), 'a train is served whole'
    for metric_id, metric in value['metrics'].items():
        for name, array in _metric_arrays(metric).items():
            assert len(array) == len(ordinals), f'{metric_id}.{name}'


def test_a_module_headline_is_computed_over_its_landings(drill_client) -> None:
    whole = _get(drill_client, 'project=dark_factory&window=all')['QUALITY']['value']
    drilled = _get(drill_client, 'project=dark_factory&module=dashboard&window=all')

    headline = drilled['QUALITY']['value']['metrics']['rejected']['headline']
    assert headline['state'] == whole['metrics']['rejected']['headline']['state']
    assert (headline['value']['n'], headline['value']['k']) == (
        len(DASHBOARD_ORDINALS), len(DASHBOARD_ORDINALS))
    assert whole['metrics']['rejected']['headline']['value']['k'] < 100


def test_a_module_reads_the_freshness_of_the_whole_project(drill_client) -> None:
    whole = _get(drill_client, 'project=dark_factory&window=all')['QUALITY']
    drilled = _get(drill_client, 'project=dark_factory&module=dashboard&window=all')['QUALITY']

    assert (drilled['state'], drilled['reason']) == ('stale', '1 landing not yet sampled')
    assert (whole['state'], whole['reason']) == (drilled['state'], drilled['reason'])


def test_a_module_no_landing_touches_is_an_empty_axis_with_unknown_headlines(drill_client) -> None:
    quality = _get(drill_client, 'project=dark_factory&module=nowhere&window=all')['QUALITY']

    assert quality['state'] == 'stale'
    assert quality['value']['axis'] == []
    for metric_id, metric in quality['value']['metrics'].items():
        assert (metric['headline']['state'], metric['headline']['value']) == ('unknown', None), (
            metric_id)
        assert all(array == [] for array in _metric_arrays(metric).values()), metric_id


def test_a_blank_module_serves_what_no_module_serves(drill_client) -> None:
    whole = _get(drill_client, 'project=dark_factory&window=all')
    blank = _get(drill_client, 'project=dark_factory&module=&window=all')

    assert blank == whole
    assert len(whole['QUALITY']['value']['axis']) == STORED
