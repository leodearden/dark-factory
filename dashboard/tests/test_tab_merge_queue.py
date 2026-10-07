"""MergeTab's wiring probes; its behaviour is tested where it can execute.

tabs.jsx and app.jsx are ``type="text/babel"`` behind CDN Babel with no JSX
render harness, so nothing here renders MergeTab. What the merge-queue tab
DECIDES lives in plain-JS modules that node runs:

  * "In queue now" — the tile's headline and spark, the per-project "queued"
    pip and the rail badge — reads the one served ``in_queue`` Datum through
    ``merge_queue.js`` (``projectInQueue``, ``inQueueOver``, ``inQueueHistory``),
    and the latency panel's caption is ``merge_queue.js::latencyCaption``; both
    are covered by dashboard/tests/js/merge_queue.test.mjs.
  * The Recent-merges caption ("showing N of M in <window>") is
    ``window_chip.js::recentMergesCaption``, covered by
    dashboard/tests/js/window_chip.test.mjs.

What is left for this file is the WIRING that source text can settle and
behaviour cannot: that MergeTab and the rail stopped reading two wire keys the
server no longer emits (task 5595). A stale read of either renders
``undefined`` silently rather than failing, so its absence is pinned here —
the wiring-not-behaviour probe ``test_datum_components.py``'s docstring
sanctions. Positive source pins are deliberately NOT added: they verify no
behaviour and break on harmless refactors.

Every probe runs over comment-stripped code, so prose that names a retired key
is free to.
"""

from __future__ import annotations

import pathlib

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments

_REDUX_DIR = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'static' / 'redux'

_RETIRED_WIRE_KEYS = ('active_approximate', 'latency.count')
"""Keys /merge-queue stopped emitting: the approx badge's flag, and the
with-duration subset the "Merges (window)" tile used to sum as if it were every
attempt (sketch #9's 167-vs-246 contradiction)."""


def _merge_tab_code() -> str:
    source = strip_js_comments((_REDUX_DIR / 'tabs.jsx').read_text())
    return extract_function_body(source, 'MergeTab')


def _app_code() -> str:
    return strip_js_comments((_REDUX_DIR / 'app.jsx').read_text())


class TestMergeTabAssetServed:
    """Smoke-tests that the tabs.jsx static asset is served correctly."""

    def test_tabs_jsx_served(self, _client):
        resp = _client.get('/static/redux/tabs.jsx')
        assert resp.status_code == 200


@pytest.mark.parametrize('surface, code', [
    ('tabs.jsx::MergeTab', _merge_tab_code),
    ('app.jsx', _app_code),
], ids=['merge-tab', 'app-rail'])
@pytest.mark.parametrize('retired_key', _RETIRED_WIRE_KEYS)
def test_no_surface_reads_a_retired_merge_queue_key(surface, code, retired_key):
    assert retired_key not in code(), (
        f'{surface} still reads `{retired_key}`, which /merge-queue no longer '
        'emits — it renders undefined. Read the queue through merge_queue.js and '
        'the attempt count from the outcomes total.'
    )


def _feed_code() -> str:
    source = strip_js_comments((_REDUX_DIR / 'shell.jsx').read_text())
    return extract_function_body(source, 'buildFeedEntries')


def test_the_live_feed_dates_a_queued_row_through_queued_since():
    """The live probe never emits a row ``timestamp``: reading one drops every queued merge."""
    assert 'a.timestamp' not in _feed_code(), (
        'shell.jsx::buildFeedEntries still reads `a.timestamp` on a queued row, a key '
        'the live get_merge_queue probe never emits, so every queued merge is skipped. '
        'Read the enqueue instant through merge_queue.js::queuedSince.'
    )
