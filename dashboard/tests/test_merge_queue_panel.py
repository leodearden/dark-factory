"""Tests for GET /partials/merge-queue route and template rendering."""

from __future__ import annotations

import re
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_DEPTH = {
    'labels': ['2026-04-10T10:00:00+00:00', '2026-04-10T10:15:00+00:00'],
    'values': [3, 1],
}

MOCK_OUTCOMES = {
    'labels': ['done', 'conflict', 'blocked', 'already_merged'],
    'values': [10, 3, 1, 2],
}

MOCK_LATENCY = {
    'p50': 1500,
    'p95': 4200,
    'p99': 8800,
    'count': 16,
    'mean_ms': 2100.0,
}

MOCK_RECENT = [
    {
        'task_id': 'task-001',
        'run_id': 'run-001',
        'outcome': 'done',
        'duration_ms': 1200,
        'timestamp': '2026-04-10T10:30:00+00:00',
    },
    {
        'task_id': 'task-002',
        'run_id': 'run-002',
        'outcome': 'conflict',
        'duration_ms': None,
        'timestamp': '2026-04-10T10:15:00+00:00',
    },
]

MOCK_SPEC = {
    'hit_count': 42,
    'discard_count': 8,
    'total': 50,
    'hit_rate': 0.84,
}

_UNSET = object()


def _extract_inline_script(html: str) -> str:
    """Extract the body of the first inline <script> block from rendered HTML.

    Used by TestMergeQueueListenerLifecycle to scope assertions against the
    partial's own <script> block rather than the full response text.  This
    prevents spurious failures if the route ever wraps the partial in a
    base layout that itself contains document.addEventListener calls (e.g.
    the alpine:init listener in base.html / burndown.html / costs.html).
    """
    match = re.search(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
    assert match is not None, 'No inline <script> block found in response HTML'
    return match.group(1)


def _patch_merge_queue_data(
    depth=_UNSET,
    outcomes=_UNSET,
    latency=_UNSET,
    recent=_UNSET,
    spec=_UNSET,
):
    """Return an ExitStack patching all 5 merge queue aggregate functions."""
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.aggregate_queue_depth_timeseries',
        new_callable=AsyncMock,
        return_value=depth if depth is not _UNSET else MOCK_DEPTH,
    ))
    stack.enter_context(patch(
        'dashboard.app.aggregate_outcome_distribution',
        new_callable=AsyncMock,
        return_value=outcomes if outcomes is not _UNSET else MOCK_OUTCOMES,
    ))
    stack.enter_context(patch(
        'dashboard.app.aggregate_latency_stats',
        new_callable=AsyncMock,
        return_value=latency if latency is not _UNSET else MOCK_LATENCY,
    ))
    stack.enter_context(patch(
        'dashboard.app.aggregate_recent_merges',
        new_callable=AsyncMock,
        return_value=recent if recent is not _UNSET else MOCK_RECENT,
    ))
    stack.enter_context(patch(
        'dashboard.app.aggregate_speculative_stats',
        new_callable=AsyncMock,
        return_value=spec if spec is not _UNSET else MOCK_SPEC,
    ))
    return stack


# ---------------------------------------------------------------------------
# TestMergeQueueRoute
# ---------------------------------------------------------------------------

class TestMergeQueueRoute:
    def test_returns_200(self, client):
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200

    def test_content_type_html(self, client):
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert 'text/html' in resp.headers['content-type']

    def test_partial_heading(self, client):
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert 'Merge Queue' in resp.text

    def test_canvas_ids_present(self, client):
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert 'mergeQueueDepthChart' in resp.text
        assert 'mergeOutcomeChart' in resp.text

    def test_latency_values_rendered(self, client):
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        # p50=1500ms → "2s", p95=4200ms → "4s", p99=8800ms → "9s"
        assert '2s' in resp.text   # p50
        assert '4s' in resp.text   # p95
        assert '9s' in resp.text   # p99

    def test_speculative_values_rendered(self, client):
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert '42' in resp.text    # hit_count
        assert '8' in resp.text     # discard_count
        assert '84' in resp.text    # hit_rate 84%

    def test_recent_merges_table_rendered(self, client):
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert 'task-001' in resp.text
        assert 'task-002' in resp.text


# ---------------------------------------------------------------------------
# TestEmptyState
# ---------------------------------------------------------------------------

class TestEmptyState:
    def test_empty_state_message(self, client):
        """All empty mocks → empty-state message shown."""
        with _patch_merge_queue_data(
            depth={'labels': [], 'values': []},
            outcomes={'labels': [], 'values': []},
            latency={'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0},
            recent=[],
            spec={'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0},
        ):
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200
        assert 'No merge activity' in resp.text


# ---------------------------------------------------------------------------
# TestMergeQueueTimeWindow
# ---------------------------------------------------------------------------

class TestMergeQueueTimeWindow:
    def _call_args_hours(self, mock_fn):
        """Extract the hours kwarg from the most recent call to mock_fn."""
        call = mock_fn.call_args
        return call.kwargs.get('hours') or call.args[1]

    def test_window_24h_forwards_hours_24(self, client):
        with _patch_merge_queue_data() as _, \
             patch('dashboard.app.aggregate_queue_depth_timeseries',
                   new_callable=AsyncMock,
                   return_value=MOCK_DEPTH) as mock_depth:
            client.get('/partials/merge-queue?window=24h')
            hours = mock_depth.call_args.kwargs.get('hours')
            assert hours == 24

    def test_window_7d_forwards_hours_168(self, client):
        with patch('dashboard.app.aggregate_queue_depth_timeseries',
                   new_callable=AsyncMock,
                   return_value=MOCK_DEPTH) as mock_depth, \
             patch('dashboard.app.aggregate_outcome_distribution',
                   new_callable=AsyncMock, return_value=MOCK_OUTCOMES), \
             patch('dashboard.app.aggregate_latency_stats',
                   new_callable=AsyncMock, return_value=MOCK_LATENCY), \
             patch('dashboard.app.aggregate_recent_merges',
                   new_callable=AsyncMock, return_value=MOCK_RECENT), \
             patch('dashboard.app.aggregate_speculative_stats',
                   new_callable=AsyncMock, return_value=MOCK_SPEC):
            client.get('/partials/merge-queue?window=7d')
            hours = mock_depth.call_args.kwargs.get('hours')
            assert hours == 168

    def test_window_30d_forwards_hours_720(self, client):
        with patch('dashboard.app.aggregate_queue_depth_timeseries',
                   new_callable=AsyncMock,
                   return_value=MOCK_DEPTH) as mock_depth, \
             patch('dashboard.app.aggregate_outcome_distribution',
                   new_callable=AsyncMock, return_value=MOCK_OUTCOMES), \
             patch('dashboard.app.aggregate_latency_stats',
                   new_callable=AsyncMock, return_value=MOCK_LATENCY), \
             patch('dashboard.app.aggregate_recent_merges',
                   new_callable=AsyncMock, return_value=MOCK_RECENT), \
             patch('dashboard.app.aggregate_speculative_stats',
                   new_callable=AsyncMock, return_value=MOCK_SPEC):
            client.get('/partials/merge-queue?window=30d')
            hours = mock_depth.call_args.kwargs.get('hours')
            assert hours == 720

    def test_default_window_forwards_hours_168(self, client):
        """No window param → default 7d → 168 hours."""
        with patch('dashboard.app.aggregate_queue_depth_timeseries',
                   new_callable=AsyncMock,
                   return_value=MOCK_DEPTH) as mock_depth, \
             patch('dashboard.app.aggregate_outcome_distribution',
                   new_callable=AsyncMock, return_value=MOCK_OUTCOMES), \
             patch('dashboard.app.aggregate_latency_stats',
                   new_callable=AsyncMock, return_value=MOCK_LATENCY), \
             patch('dashboard.app.aggregate_recent_merges',
                   new_callable=AsyncMock, return_value=MOCK_RECENT), \
             patch('dashboard.app.aggregate_speculative_stats',
                   new_callable=AsyncMock, return_value=MOCK_SPEC):
            client.get('/partials/merge-queue')
            hours = mock_depth.call_args.kwargs.get('hours')
            assert hours == 168


# ---------------------------------------------------------------------------
# TestIndexWiring
# ---------------------------------------------------------------------------

class TestIndexWiring:
    def test_index_contains_merge_queue_section(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert 'data-section="merge-queue"' in resp.text

    def test_index_contains_hx_get(self, client):
        resp = client.get('/')
        assert 'hx-get="/partials/merge-queue"' in resp.text

    def test_index_contains_poll_base(self, client):
        resp = client.get('/')
        assert 'data-poll-base' in resp.text


# ---------------------------------------------------------------------------
# TestMergeQueueListenerLifecycle
# ---------------------------------------------------------------------------

class TestMergeQueueListenerLifecycle:
    def test_inline_script_has_no_persistent_htmx_after_settle_listener(self, client):
        """The inline script must have exactly one document.addEventListener call,
        and it must be the DOMContentLoaded one.

        The original zombie-listener bug added a persistent
        document.addEventListener('htmx:afterSettle', ...) block that accumulated
        on every htmx re-swap (every 15s polling cycle) with no cleanup.  A
        narrow 'htmx:afterSettle' not-in check would miss a future reintroduction
        under a different event name (htmx:load, htmx:afterSwap, …).  The
        count-based invariant guards against ANY extra document-level listener
        regardless of event name, and the DOMContentLoaded presence check ensures
        the one permitted listener is the expected one.

        The count assertion is scoped to the partial's own inline <script> block
        via _extract_inline_script so that future base-layout wrapping (e.g.
        the alpine:init document.addEventListener call in base.html, burndown.html,
        or costs.html) does not cause spurious failures when the zombie-listener
        bug is NOT reintroduced.
        """
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200
        script_body = _extract_inline_script(resp.text)
        # Exactly one document-level listener is allowed (the DOMContentLoaded one).
        # Scoped to the partial's own <script> block, not the full response text.
        assert script_body.count('document.addEventListener(') == 1
        assert 'DOMContentLoaded' in script_body

    def test_render_all_invoked_directly_in_iife(self, client):
        """renderAll() must be called directly within the IIFE so charts render
        after an htmx swap (when DOMContentLoaded has already fired).

        The old assertion ``assert 'renderAll()' in resp.text`` was tautological:
        the substring 'renderAll()' also appears inside the function definition
        ``function renderAll() {``, so deleting the direct-invocation line would
        not have caused it to fail.

        The trailing-semicolon form ``renderAll();`` is the disambiguator: the
        definition line ends with `` {``, not ``;``, so only the actual call site
        contains the semicoloned form.  ``count == 1`` is strictly tighter than
        ``>= 2``: it catches the original regression (no direct invocation →
        count becomes 0) AND prevents accidental double-invocation, AND is immune
        to drift from HTML comments or documentation text that happens to mention
        ``renderAll()``.

        Both assertions are scoped to the partial's own inline <script> block via
        _extract_inline_script to avoid false failures from any outer layout.
        """
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200
        script_body = _extract_inline_script(resp.text)
        # Exactly one direct invocation of renderAll() must appear in the script.
        # Trailing semicolon uniquely identifies the call site vs. the definition.
        assert script_body.count('renderAll();') == 1
