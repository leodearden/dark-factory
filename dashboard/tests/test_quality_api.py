"""/api/v2/dashboard/quality: its landing-window vocabulary and the served echo."""

from __future__ import annotations

import pytest

from dashboard.api.quality import _QUALITY_WINDOWS
from dashboard.api.window import (
    ServedLandingWindow,
    ServedWindow,
    _parse_landing_window,
    _parse_window,
    with_window,
)


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
