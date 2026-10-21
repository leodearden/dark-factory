"""Asset wiring for the Tasks tab's deferral holds.

The cell DECISION is tested as a pure function in
``dashboard/tests/js/deferral_cell.test.mjs``. This module covers what that
suite cannot see: the module is reachable from the browser, tagged in
``index.html`` exactly once with its siblings' cache-buster, and evaluated
before the JSX that reads its global. It asserts structural facts about served
HTML, never JSX source text.
"""

from __future__ import annotations

from _cache_buster_helpers import redux_cache_buster_versions
from _dashboard_helpers import ScriptTagCollector, assert_script_loads_before

_CELL_SRC = '/static/redux/deferral_cell.js'


def _redux_script_srcs(body: str) -> list[str]:
    collector = ScriptTagCollector()
    collector.feed(body)
    srcs = (attrs.get('src') or '' for attrs in collector.script_attrs)
    return [src for src in srcs if src.startswith('/static/redux/')]


def test_cell_module_is_served(_client):
    resp = _client.get(_CELL_SRC)

    assert resp.status_code == 200
    assert 'deferralCellState' in resp.text


def test_index_html_tags_the_cell_once_with_the_sibling_cache_buster(index_html_body):
    srcs = _redux_script_srcs(index_html_body)
    cell = [src for src in srcs if src.startswith(_CELL_SRC)]
    assert len(cell) == 1, f'expected exactly one {_CELL_SRC} tag, got {cell}'

    sibling_versions = redux_cache_buster_versions(
        ' '.join(src for src in srcs if not src.startswith(_CELL_SRC))
    )
    assert len(sibling_versions) == 1, f'the redux siblings disagree on ?v=: {sibling_versions}'
    assert redux_cache_buster_versions(cell[0]) == sibling_versions, (
        f'{cell[0]} does not carry its siblings\' ?v={sibling_versions}'
    )


def test_cell_module_loads_before_tab_tasks_jsx(index_html_body):
    assert_script_loads_before(
        index_html_body,
        _CELL_SRC,
        '/static/redux/tab_tasks.jsx',
        'deferral_cell.js',
        'tab_tasks.jsx',
        consumer_note='tab_tasks.jsx reads window.DF_DEFERRAL_CELL at module scope.',
    )
