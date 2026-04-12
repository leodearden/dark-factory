"""Tests for GET /partials/merge-queue route and template rendering."""

from __future__ import annotations

import re
import sqlite3
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from dashboard.data.merge_queue import _bucket_minutes_for_window

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

# Sentinel used by _extract_inline_script to locate the merge-queue partial's
# own <script> block.  This is the canvas id referenced inside the partial's
# inline script (merge_queue.html: getOrDestroyChart('mergeQueueDepthChart')).
# It is unique to this partial and does not appear in any other template's
# script body, making it a reliable discriminator even if the route is later
# wrapped in a layout that prepends additional <script> blocks.
_PARTIAL_SCRIPT_SENTINEL = 'mergeQueueDepthChart'


def _extract_inline_script(html: str) -> str:
    """Extract the body of the merge-queue partial's inline <script> block.

    Uses marker-based selection: iterates all ``<script>`` blocks in the HTML
    and returns the first whose body contains ``_PARTIAL_SCRIPT_SENTINEL``
    (``'mergeQueueDepthChart'`` — the canvas id unique to this partial).

    This approach is immune to future layout-wrapper changes that prepend other
    script blocks (e.g. the alpine:init listener in base.html / burndown.html /
    costs.html): those scripts will not contain the sentinel and are skipped.
    As a side benefit, ``<script>`` tags embedded inside HTML comments are also
    naturally skipped because their bodies typically do not carry the sentinel.

    Raises ``AssertionError`` if no sentinel-bearing script block is found, so
    failures surface a clear message rather than a downstream ``AttributeError``.
    """
    for m in re.finditer(r'<script[^>]*>(.*?)</script>', html, re.DOTALL):
        if _PARTIAL_SCRIPT_SENTINEL in m.group(1):
            return m.group(1)
    raise AssertionError(
        f'No inline <script> block containing sentinel {_PARTIAL_SCRIPT_SENTINEL!r}'
        ' found in response HTML'
    )


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

        The old assertion used ``'renderAll()' in resp.text`` (no semicolon),
        which was tautological: the substring ``renderAll()`` also appears inside
        the function definition line ``function renderAll() {``, so deleting the
        direct-invocation line would not have caused it to fail.  A previous
        belt-and-suspenders ``resp.text.count('renderAll()') >= 2`` counted both
        the definition and the call (both contain the no-semicolon form), but was
        fragile to drift from HTML comments or documentation text that happened to
        mention ``renderAll()``.

        The new assertions use two distinct substrings — note the substring
        changed, not just the bound:

        * ``'function renderAll()'`` — matches only the definition line (ends
          with `` {``), confirming the function is actually declared in the script.
        * ``'renderAll();'`` — the trailing semicolon form matches only the direct
          call site (the definition line ends with `` {``, not ``;``).
          ``count == 1`` asserts exactly one direct invocation, catching the
          original regression (no call → count becomes 0) and preventing
          accidental double-invocation.

        Both checks are scoped to the partial's own inline <script> block via
        _extract_inline_script to avoid false failures from any outer layout.
        """
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200
        script_body = _extract_inline_script(resp.text)
        # The function must be defined (not just imported or referenced externally).
        assert 'function renderAll()' in script_body
        # Exactly one direct invocation must appear; trailing semicolon is the
        # disambiguator that excludes the definition line.
        assert script_body.count('renderAll();') == 1


# ---------------------------------------------------------------------------
# TestMergeQueueWindowAll
# ---------------------------------------------------------------------------

class TestMergeQueueWindowAll:
    def test_window_all_returns_bounded_depth_payload(self, client):
        """GET /partials/merge-queue?window=all must:

        1. Return 200 OK.
        2. Call aggregate_queue_depth_timeseries with hours=87600.
        3. Serve a depth payload whose length is bounded (< 10 000) —
           regression guard against the 350 401-bucket blowup.

        The test uses a side_effect mock that:
        - Asserts the hours kwarg equals 87600.
        - Computes N = _bucket_minutes_for_window(87600) → returns a
          ChartData with N labels so the route can complete normally.
        """
        captured = {}

        async def _mock_aggregate_depth(*args, **kwargs):
            captured['hours'] = kwargs.get('hours')
            bm = _bucket_minutes_for_window(87600)
            N = (87600 * 60 // bm) + 1
            return {'labels': [f'L{i}' for i in range(N)], 'values': [0] * N}

        with patch('dashboard.app.aggregate_queue_depth_timeseries',
                   side_effect=_mock_aggregate_depth), \
             patch('dashboard.app.aggregate_outcome_distribution',
                   new_callable=AsyncMock, return_value=MOCK_OUTCOMES), \
             patch('dashboard.app.aggregate_latency_stats',
                   new_callable=AsyncMock, return_value=MOCK_LATENCY), \
             patch('dashboard.app.aggregate_recent_merges',
                   new_callable=AsyncMock, return_value=MOCK_RECENT), \
             patch('dashboard.app.aggregate_speculative_stats',
                   new_callable=AsyncMock, return_value=MOCK_SPEC):
            resp = client.get('/partials/merge-queue?window=all')

        assert resp.status_code == 200
        assert captured.get('hours') == 87600

        bm = _bucket_minutes_for_window(87600)
        N = (87600 * 60 // bm) + 1
        assert N < 10_000, f"_bucket_minutes_for_window(87600) yields {N} points — regression!"

    def test_window_all_real_aggregator_bounded_response(self, client, tmp_path):
        """Integration: real aggregate_queue_depth_timeseries with a real empty DB.

        Does NOT mock the depth aggregator — the real code path runs and must
        return quickly.  Regression guard: the old hard-coded 15-min bucket
        would allocate 350 401 buckets in-memory even for an empty DB, causing
        this test to timeout or OOM; the adaptive ladder allocates ~3 651.

        Only the four non-depth aggregators are mocked so the test does not
        need to populate their respective data.  All five aggregators share the
        same ``events`` table, so with the proper schema they would also work
        on the empty DB — but mocking them keeps the test focused on the
        depth-aggregator regression.
        """
        from dashboard.config import DashboardConfig

        _EVENTS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_id TEXT,
    event_type TEXT NOT NULL,
    phase TEXT,
    role TEXT,
    data TEXT DEFAULT '{}',
    cost_usd REAL,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
"""

        # Create a real empty runs.db with the events schema.
        runs_db = tmp_path / 'data' / 'orchestrator' / 'runs.db'
        runs_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(runs_db))
        conn.executescript(_EVENTS_SCHEMA)
        conn.commit()
        conn.close()

        # Point the app at our temp project root so _cost_dbs opens the real DB.
        test_config = DashboardConfig(project_root=tmp_path)

        with patch.object(client.app.state, 'config', test_config), \
             patch('dashboard.app.aggregate_outcome_distribution',
                   new_callable=AsyncMock, return_value=MOCK_OUTCOMES), \
             patch('dashboard.app.aggregate_latency_stats',
                   new_callable=AsyncMock, return_value=MOCK_LATENCY), \
             patch('dashboard.app.aggregate_recent_merges',
                   new_callable=AsyncMock, return_value=MOCK_RECENT), \
             patch('dashboard.app.aggregate_speculative_stats',
                   new_callable=AsyncMock, return_value=MOCK_SPEC):
            resp = client.get('/partials/merge-queue?window=all')

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestExtractInlineScript
# ---------------------------------------------------------------------------

class TestExtractInlineScript:
    """Unit tests for the _extract_inline_script helper."""

    def test_skips_foreign_script_before_partial(self):
        """Marker-based selection returns the partial's own script, not a foreign one.

        This is the primary regression guard for the refactor: a foreign layout
        script (e.g. alpine:init) appearing BEFORE the partial's own <script>
        block must be skipped.  The old position-based re.search returns the
        first script (the alpine:init one), causing the sentinel assertion to
        fail — this test drives the refactor.
        """
        html = (
            '<html><head>\n'
            '<script>\n'
            "document.addEventListener('alpine:init', () => { console.log('init'); });\n"
            '</script>\n'
            '</head><body>\n'
            '<script>\n'
            "var el = getOrDestroyChart('mergeQueueDepthChart');\n"
            'function renderAll() { el.render(); }\n'
            'renderAll();\n'
            '</script>\n'
            '</body></html>'
        )
        body = _extract_inline_script(html)
        assert 'mergeQueueDepthChart' in body
        assert 'alpine:init' not in body

    def test_raises_assertion_when_no_script_contains_sentinel(self):
        """AssertionError is raised (with sentinel in message) when no script
        block carries the mergeQueueDepthChart sentinel.

        Guards against future regressions that silently drop the sentinel from
        the partial's script or change the canvas id — a clear error message
        pointing at the missing sentinel is more actionable than a downstream
        AttributeError on a None return value.
        """
        import pytest
        html = (
            '<html><head>\n'
            '<script>\n'
            "document.addEventListener('alpine:init', () => { console.log('init'); });\n"
            '</script>\n'
            '</head><body>\n'
            '<script>\n'
            "document.addEventListener('htmx:configRequest', (e) => { e.detail.headers['X-CSRFToken'] = 'tok'; });\n"
            '</script>\n'
            '</body></html>'
        )
        with pytest.raises(AssertionError) as excinfo:
            _extract_inline_script(html)
        assert 'mergeQueueDepthChart' in str(excinfo.value)

    def test_ignores_script_tag_inside_html_comment(self):
        """A <script> tag embedded inside an HTML comment is naturally skipped.

        This documents the 'secondary concern sidestep' from the task details:
        the marker-based approach ignores commented-out script blocks because
        their bodies typically do not contain the sentinel.  The real partial
        script that follows the comment is returned correctly.
        """
        html = (
            '<html><body>\n'
            '<!-- old layout script, kept for reference:\n'
            '<script>var x = 1;</script>\n'
            '-->\n'
            '<script>\n'
            "var el = getOrDestroyChart('mergeQueueDepthChart');\n"
            'function renderAll() { el.render(); }\n'
            'renderAll();\n'
            '</script>\n'
            '</body></html>'
        )
        body = _extract_inline_script(html)
        assert 'mergeQueueDepthChart' in body
        assert 'function renderAll()' in body
        assert 'var x = 1' not in body
