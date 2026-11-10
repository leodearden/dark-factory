"""/api/v2/dashboard/quality: its landing-window vocabulary, the served echo, and the route."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _dt_helpers import fixed_clock, installed_clock
from _quality_rows import history, write_landing_events, write_quality_db
from shared.quality_series.landings import GitCrossCheck

from dashboard.api.quality import _QUALITY_WINDOWS
from dashboard.api.window import (
    ServedLandingWindow,
    ServedWindow,
    _parse_landing_window,
    _parse_window,
    with_window,
)
from dashboard.config import DashboardConfig
from dashboard.data.quality_store import QualitySample


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
    assert _parse_landing_window(params, vocabulary=_QUALITY_WINDOWS, default='100') == expected


def test_the_landing_window_refuses_a_default_its_vocabulary_cannot_serve() -> None:
    with pytest.raises(ValueError, match="'7d'"):
        _parse_landing_window({'window': '100'}, vocabulary=_QUALITY_WINDOWS, default='7d')


def test_a_served_landing_window_names_landings_not_days() -> None:
    window = _parse_landing_window({'window': 'all'}, vocabulary=_QUALITY_WINDOWS, default='100')

    assert window.to_wire() == {'requested': 'all', 'served': 'all', 'landings': None}
    assert with_window({'QUALITY': {}}, window) == {'QUALITY': {}, 'WINDOW': window.to_wire()}


def test_the_day_vocabulary_is_untouched() -> None:
    assert _parse_window({'window': '7d'}) == ServedWindow(requested='7d', served='7d', days=7)


# --- the route ------------------------------------------------------------------------------

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
URL = '/api/v2/dashboard/quality'
DATUM_KEYS = {'value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'}
STORED = 120


@pytest.fixture
def quality_client(client, tmp_path: Path) -> Iterator[Any]:
    """The app over a root named 'dark_factory' holding STORED sampled landings, beside an
    unsampled 'other-proj', serving at NOW."""
    root, other = tmp_path / 'dark_factory', tmp_path / 'other-proj'
    root.mkdir()
    other.mkdir()
    config = DashboardConfig(project_root=root, known_project_roots=[other])
    rows = history(STORED, start=NOW - timedelta(days=1) - timedelta(hours=2 * (STORED - 1)))
    write_landing_events(root, rows)
    sample = QualitySample.value(
        'dark_factory', NOW - timedelta(minutes=5), cursor_sha=rows[-1].landing.merge_sha,
        landings_sampled=STORED, landings_deferred=0, git_check=GitCrossCheck(0, 0, 0),
    )
    asyncio.run(write_quality_db(config.quality_db, rows, [sample]))
    client.app.state.config = config
    with installed_clock(client.app, fixed_clock(NOW)):
        yield client


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


def test_markers_and_snapshots_are_unknown_until_their_producers_land(quality_client) -> None:
    body = _get(quality_client, 'project=dark_factory&window=100')

    markers, snapshots = body['QUALITY_MARKERS'], body['QUALITY_SNAPSHOTS']
    for datum in (markers, snapshots):
        assert set(datum) == DATUM_KEYS
        assert (datum['state'], datum['value'], datum['as_of']) == ('unknown', None, None)
        assert datum['reason'].startswith('no producer has written here')
    assert 'marker' in markers['reason']
    assert 'snapshot' in snapshots['reason']


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
