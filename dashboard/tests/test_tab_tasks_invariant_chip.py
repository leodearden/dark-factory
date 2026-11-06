"""Asset-wiring tests for the Tasks tab's claimant-invariant chip.

The chip DECISION is tested as a pure function in
``dashboard/tests/js/invariant_chip.test.mjs``. That the Tasks tab actually
reads ``DF_DATA.INVARIANTS`` and polls its endpoint is derived, not grepped, by
``test_app_poll_scope.py``'s tasks floor. This module covers what neither can
see: that the module is reachable from the browser, tagged in ``index.html``
exactly once with its siblings' cache-buster, and evaluated before the JSX that
calls its global. Like ``test_tab_tasks_offline_banner.py`` it asserts only
structural facts about served HTML, never JSX source text (esc-3857-9).
"""

from __future__ import annotations

from _cache_buster_helpers import redux_cache_buster_versions
from _dashboard_helpers import ScriptTagCollector, assert_script_loads_before

_CHIP_SRC = '/static/redux/invariant_chip.js'


def _redux_script_srcs(body: str) -> list[str]:
    collector = ScriptTagCollector()
    collector.feed(body)
    srcs = (attrs.get('src') or '' for attrs in collector.script_attrs)
    return [src for src in srcs if src.startswith('/static/redux/')]


def test_chip_module_is_served(_client):
    resp = _client.get(_CHIP_SRC)

    assert resp.status_code == 200
    assert 'invariantChipFor' in resp.text


def test_index_html_tags_the_chip_once_with_the_sibling_cache_buster(index_html_body):
    srcs = _redux_script_srcs(index_html_body)
    chip = [src for src in srcs if src.startswith(_CHIP_SRC)]
    assert len(chip) == 1, f'expected exactly one {_CHIP_SRC} tag, got {chip}'

    sibling_versions = redux_cache_buster_versions(
        ' '.join(src for src in srcs if not src.startswith(_CHIP_SRC))
    )
    assert len(sibling_versions) == 1, f'the redux siblings disagree on ?v=: {sibling_versions}'
    assert redux_cache_buster_versions(chip[0]) == sibling_versions, (
        f'{chip[0]} does not carry its siblings\' ?v={sibling_versions}'
    )


def test_chip_module_loads_before_tab_tasks_jsx(index_html_body):
    assert_script_loads_before(
        index_html_body,
        _CHIP_SRC,
        '/static/redux/tab_tasks.jsx',
        'invariant_chip.js',
        'tab_tasks.jsx',
        consumer_note='tab_tasks.jsx reads window.DF_INVARIANT_CHIP at module scope.',
    )
