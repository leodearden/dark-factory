"""Tests for GET /partials/merge-queue route and template rendering."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from dashboard.app import unique_css_ids
from dashboard.config import DashboardConfig as _DashboardConfig
from dashboard.data.merge_queue import _bucket_minutes_for_window
from tests._dt_helpers import make_fixed_datetime_cls

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
    typically skipped because their bodies do not carry the sentinel.

    Note: the regex does not parse HTML comments; skipping relies entirely on
    the sentinel not appearing in the commented block.  If a commented-out
    ``<script>`` block happens to contain the sentinel, it will be returned
    instead of the real partial script.

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


# Default project PID derived from the app's resolved config.project_root so that the
# patch key always matches runtime state regardless of machine/worktree layout.
_DEFAULT_PID = str(_DashboardConfig().project_root)


def _patch_merge_queue_data(
    depth=_UNSET,
    outcomes=_UNSET,
    latency=_UNSET,
    recent=_UNSET,
    spec=_UNSET,
):
    """Return an ExitStack patching merge queue functions for a single default project.

    After the per-project route rewrite (step-12), patches ``build_per_project_merge_queue``
    with a single-project dict keyed by the app's default project root.  ``load_task_titles``
    is patched to return ``{}`` so no filesystem reads happen during tests.
    """
    d = depth if depth is not _UNSET else MOCK_DEPTH
    o = outcomes if outcomes is not _UNSET else MOCK_OUTCOMES
    l_ = latency if latency is not _UNSET else MOCK_LATENCY
    r = recent if recent is not _UNSET else MOCK_RECENT
    s = spec if spec is not _UNSET else MOCK_SPEC

    projects = {
        _DEFAULT_PID: {
            'depth_timeseries': d,
            'outcomes': o,
            'latency': l_,
            'recent': r,
            'speculative': s,
        }
    }

    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.build_per_project_merge_queue',
        new_callable=AsyncMock,
        return_value=projects,
    ))
    stack.enter_context(patch(
        'dashboard.app.load_task_titles',
        return_value={},
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
    """Verify that the hours window parameter is forwarded to build_per_project_merge_queue."""

    def _bppq_hours(self, mock_bppq):
        """Extract the hours kwarg from the most recent call to build_per_project_merge_queue."""
        return mock_bppq.call_args.kwargs.get('hours')

    def test_window_24h_forwards_hours_24(self, client):
        with patch('dashboard.app.build_per_project_merge_queue',
                   new_callable=AsyncMock, return_value={}) as mock_bppq, \
             patch('dashboard.app.load_task_titles', return_value={}):
            client.get('/partials/merge-queue?window=24h')
        assert self._bppq_hours(mock_bppq) == 24

    def test_window_7d_forwards_hours_168(self, client):
        with patch('dashboard.app.build_per_project_merge_queue',
                   new_callable=AsyncMock, return_value={}) as mock_bppq, \
             patch('dashboard.app.load_task_titles', return_value={}):
            client.get('/partials/merge-queue?window=7d')
        assert self._bppq_hours(mock_bppq) == 168

    def test_window_30d_forwards_hours_720(self, client):
        with patch('dashboard.app.build_per_project_merge_queue',
                   new_callable=AsyncMock, return_value={}) as mock_bppq, \
             patch('dashboard.app.load_task_titles', return_value={}):
            client.get('/partials/merge-queue?window=30d')
        assert self._bppq_hours(mock_bppq) == 720

    def test_default_window_forwards_hours_720(self, client):
        """No window param → default 30d → 720 hours (changed 7d→30d, task 841 UX fix)."""
        with patch('dashboard.app.build_per_project_merge_queue',
                   new_callable=AsyncMock, return_value={}) as mock_bppq, \
             patch('dashboard.app.load_task_titles', return_value={}):
            client.get('/partials/merge-queue')
        assert self._bppq_hours(mock_bppq) == 720


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

    def test_merge_queue_section_uses_innerhtml_swap(self, client):
        """The merge-queue polling section must use hx-swap="innerHTML" (not morph:innerHTML).

        Chart-bearing sections must use plain innerHTML so the inline chart-init
        <script> re-executes after each htmx swap (morph mode preserves the DOM
        and does not re-run inline scripts, leaving stale/destroyed <canvas>
        elements blank on the second poll).  Convention established in commit
        2c2ced98d5 for the orchestrators/performance/memory-graphs sections.
        """
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text

        # The merge-queue section must use hx-swap="innerHTML" (not morph:innerHTML)
        assert re.search(
            r'data-section="merge-queue"[^<]*hx-swap="innerHTML"', html
        ), 'merge-queue section must have hx-swap="innerHTML"'

        # Must NOT use the default morph:innerHTML
        assert not re.search(
            r'data-section="merge-queue"[^<]*hx-swap="morph:innerHTML"', html
        ), 'merge-queue section must NOT use hx-swap="morph:innerHTML"'


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
        2. Call build_per_project_merge_queue with hours=87600.
        3. N = _bucket_minutes_for_window(87600) must be bounded (< 10 000) —
           regression guard against the 350 401-bucket blowup.
        """
        captured = {}

        async def _mock_bppq(project_dbs, *, hours, now, recent_window_minutes):
            captured['hours'] = hours
            bm = _bucket_minutes_for_window(hours)
            N = (hours * 60 // bm) + 1
            return {
                _DEFAULT_PID: {
                    'depth_timeseries': {'labels': [f'L{i}' for i in range(N)], 'values': [0] * N},
                    'outcomes': {'labels': [], 'values': []},
                    'latency': {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0, 'mean_ms': 0.0},
                    'recent': [],
                    'speculative': {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': 0.0},
                }
            }

        with patch('dashboard.app.build_per_project_merge_queue', side_effect=_mock_bppq), \
             patch('dashboard.app.load_task_titles', return_value={}):
            resp = client.get('/partials/merge-queue?window=all')

        assert resp.status_code == 200
        assert captured.get('hours') == 87600

        bm = _bucket_minutes_for_window(87600)
        N = (87600 * 60 // bm) + 1
        assert N < 10_000, f"_bucket_minutes_for_window(87600) yields {N} points — regression!"

    def test_window_all_real_aggregator_bounded_response(self, client, tmp_path):
        """Integration: real queue_depth_timeseries (via build_per_project_merge_queue) on
        an empty DB.

        Does NOT mock build_per_project_merge_queue — the real code path runs and must
        return quickly.  Regression guard: the old hard-coded 15-min bucket
        would allocate 350 401 buckets in-memory even for an empty DB, causing
        this test to timeout or OOM; the adaptive ladder allocates ~3 651.
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

        # Point the app at our temp project root so _project_scoped_dbs_labeled opens the real DB.
        test_config = DashboardConfig(project_root=tmp_path)

        with patch.object(client.app.state, 'config', test_config):
            resp = client.get('/partials/merge-queue?window=all')

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestPartialsMergeQueueSharedNow (step-15)
# ---------------------------------------------------------------------------

class TestPartialsMergeQueueSharedNow:
    def test_partials_merge_queue_passes_shared_now_to_build_per_project(self, client):
        """partials_merge_queue captures now once and forwards it to build_per_project_merge_queue.

        After the per-project route rewrite (step-12), the shared ``now`` is passed as a kwarg
        to ``build_per_project_merge_queue``.  That function threads it to all 5 per-DB queries
        internally.  The test verifies the outer routing layer captures and forwards the fixed
        clock value correctly.
        """
        FIXED_NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)

        _FixedDT = make_fixed_datetime_cls(FIXED_NOW)

        mock_bppq = AsyncMock(return_value={})

        with patch('dashboard.app.datetime', _FixedDT), \
             patch('dashboard.app.build_per_project_merge_queue', mock_bppq), \
             patch('dashboard.app.load_task_titles', return_value={}):
            resp = client.get('/partials/merge-queue')

        assert resp.status_code == 200

        now_kwarg = mock_bppq.call_args.kwargs.get('now')

        assert now_kwarg is not None, "build_per_project_merge_queue was not passed `now`"
        assert isinstance(now_kwarg, datetime), (
            f"Expected `now` to be a datetime, got {type(now_kwarg)!r}"
        )
        assert now_kwarg == FIXED_NOW, (
            f"`now` value does not match fixed clock: got {now_kwarg!r}"
        )


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
        assert _PARTIAL_SCRIPT_SENTINEL in body
        assert 'alpine:init' not in body

    def test_raises_assertion_when_no_script_contains_sentinel(self):
        """AssertionError is raised (with sentinel in message) when no script
        block carries the mergeQueueDepthChart sentinel.

        Guards against future regressions that silently drop the sentinel from
        the partial's script or change the canvas id — a clear error message
        pointing at the missing sentinel is more actionable than a downstream
        AttributeError on a None return value.
        """
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
        assert _PARTIAL_SCRIPT_SENTINEL in str(excinfo.value)

    def test_sentinel_in_commented_script_is_still_matched(self):
        """A commented-out <script> containing the sentinel IS returned first.

        This documents the known limitation of the marker-based approach: the
        regex does not parse HTML comments.  If a commented-out ``<script>``
        block happens to contain the sentinel, it will be matched and returned
        before the real partial script.  The 'skipping' seen in
        ``test_ignores_script_tag_inside_html_comment`` relies entirely on the
        commented block not containing the sentinel — not on structural parsing
        of HTML comments.
        """
        html = (
            '<html><body>\n'
            '<!-- old partial, kept for reference:\n'
            '<script>\n'
            "var el = getOrDestroyChart('mergeQueueDepthChart');  // legacy\n"
            '</script>\n'
            '-->\n'
            '<script>\n'
            "var el = getOrDestroyChart('mergeQueueDepthChart');\n"
            'function renderAll() { el.render(); }\n'
            'renderAll();\n'
            '</script>\n'
            '</body></html>'
        )
        # The commented-out block contains the sentinel and is matched first.
        body = _extract_inline_script(html)
        assert _PARTIAL_SCRIPT_SENTINEL in body
        # 'function renderAll()' only appears in the real (second) script —
        # its absence confirms the commented block was returned, not the real one.
        assert 'function renderAll()' not in body

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
        assert _PARTIAL_SCRIPT_SENTINEL in body
        assert 'function renderAll()' in body
        assert 'var x = 1' not in body

    def test_raises_on_no_script_blocks(self):
        """AssertionError is raised when the HTML has no <script> blocks at all.

        Documents the degenerate edge case: the for-loop in _extract_inline_script
        does not iterate on empty HTML, falling through to the AssertionError raise.
        """
        with pytest.raises(AssertionError):
            _extract_inline_script('<html><body></body></html>')


# ---------------------------------------------------------------------------
# TestDepthTrimLeadingZeros
# ---------------------------------------------------------------------------


def _extract_depth_data(html: str) -> dict:
    """Extract the depth_timeseries for the first project from the allProjects JSON object.

    The template embeds ``var allProjects = {{ projects | tojson }};``.  This helper
    locates that assignment, raw-decodes the JSON object, and returns the
    ``depth_timeseries`` sub-dict for the first (and typically only) project.

    Uses :class:`json.JSONDecoder` ``raw_decode`` rather than regex so that complex
    nested JSON does not trip up a greedy pattern.
    """
    script = _extract_inline_script(html)
    marker = 'var allProjects = '
    idx = script.find(marker)
    assert idx >= 0, f"Could not find '{marker}' in script block"
    start = idx + len(marker)
    all_projects, _ = json.JSONDecoder().raw_decode(script, start)
    assert all_projects, "allProjects dict is empty"
    first_pid = next(iter(all_projects))
    return all_projects[first_pid]['depth_timeseries']


class TestDepthTrimLeadingZeros:
    """Verify that /partials/merge-queue strips leading zero buckets from the depth series."""

    def test_leading_zeros_stripped_from_rendered_depth(self, client):
        """When aggregate_queue_depth_timeseries returns a long zero-prefix, the
        rendered depthData must have the leading zeros removed.

        Setup: mock returns 4 buckets — [0, 0, 0, 5] — only the last is non-zero.
        After trim_leading_zero_buckets the rendered payload must have exactly
        1 label/value pair with value 5.
        """
        raw_depth = {
            'labels': ['2026-04-10T10:00', '2026-04-10T10:15', '2026-04-10T10:30', '2026-04-10T10:45'],
            'values': [0, 0, 0, 5],
        }
        with _patch_merge_queue_data(depth=raw_depth):
            resp = client.get('/partials/merge-queue')

        assert resp.status_code == 200
        depth_data = _extract_depth_data(resp.text)

        # The rendered payload must be shorter than the raw aggregate.
        assert len(depth_data['labels']) < len(raw_depth['labels']), (
            "Rendered depth has same length as raw aggregate — leading zeros were NOT trimmed"
        )
        # The first rendered value must be non-zero.
        assert depth_data['values'][0] != 0, (
            f"First rendered depth value is {depth_data['values'][0]!r}, expected non-zero"
        )
        # Exactly the non-zero tail should be present.
        assert depth_data['labels'] == ['2026-04-10T10:45']
        assert depth_data['values'] == [5]

    def test_depth_without_leading_zeros_unchanged(self, client):
        """When the depth series has no leading zeros, the rendered payload is unchanged."""
        raw_depth = {
            'labels': ['2026-04-10T10:00', '2026-04-10T10:15', '2026-04-10T10:30'],
            'values': [3, 0, 2],
        }
        with _patch_merge_queue_data(depth=raw_depth):
            resp = client.get('/partials/merge-queue')

        assert resp.status_code == 200
        depth_data = _extract_depth_data(resp.text)

        # Interior zero is preserved; no trimming happens.
        assert depth_data['labels'] == raw_depth['labels']
        assert depth_data['values'] == raw_depth['values']


# ---------------------------------------------------------------------------
# TestMergeQueuePerProject — per-project card layout (step-11/12)
# ---------------------------------------------------------------------------

_PID_A = '/tmp/dark-factory'
_PID_B = '/tmp/other'

# Expected css_id values for canvas id attributes:
#   css_id('/tmp/dark-factory') → 'tmp_dark_factory'
#   css_id('/tmp/other')        → 'tmp_other'
_SAFE_A = 'tmp_dark_factory'
_SAFE_B = 'tmp_other'

_MOCK_PROJECT_DATA = {
    _PID_A: {
        'depth_timeseries': MOCK_DEPTH,
        'outcomes': MOCK_OUTCOMES,
        'latency': MOCK_LATENCY,
        'recent': MOCK_RECENT,
        'speculative': MOCK_SPEC,
    },
    _PID_B: {
        'depth_timeseries': MOCK_DEPTH,
        'outcomes': MOCK_OUTCOMES,
        'latency': MOCK_LATENCY,
        'recent': MOCK_RECENT,
        'speculative': MOCK_SPEC,
    },
}


def _patch_per_project_merge_data(projects=_UNSET, titles=_UNSET):
    """Patch the new per-project route helpers.

    Uses ``create=True`` so the test remains usable before the imports are wired
    into app.py (the test will still FAIL on the html assertions at that point,
    which is the desired TDD state).
    """
    stack = ExitStack()
    stack.enter_context(patch(
        'dashboard.app.build_per_project_merge_queue',
        new_callable=AsyncMock,
        create=True,
        return_value=projects if projects is not _UNSET else _MOCK_PROJECT_DATA,
    ))
    stack.enter_context(patch(
        'dashboard.app.load_task_titles',
        create=True,
        return_value=titles if titles is not _UNSET else {},
    ))
    return stack


class TestMergeQueuePerProject:
    """The /partials/merge-queue partial renders one card block per project."""

    def test_renders_per_project_cards(self, client):
        """Two-project mock produces depth/outcome canvas IDs for both projects."""
        with _patch_per_project_merge_data():
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200
        html = resp.text

        # (a) depth canvas for project A
        assert f'id="mergeQueueDepthChart-{_SAFE_A}"' in html, (
            f'Expected mergeQueueDepthChart-{_SAFE_A} canvas id in HTML'
        )
        # (b) depth canvas for project B
        assert f'id="mergeQueueDepthChart-{_SAFE_B}"' in html, (
            f'Expected mergeQueueDepthChart-{_SAFE_B} canvas id in HTML'
        )
        # (c) outcome canvas IDs
        assert f'id="mergeOutcomeChart-{_SAFE_A}"' in html
        assert f'id="mergeOutcomeChart-{_SAFE_B}"' in html

        # (d) project display names appear as h3 headers (project_name filter: basename)
        assert re.search(r'<h3[^>]*>\s*dark-factory\s*<', html), (
            "Expected 'dark-factory' as h3 header in HTML"
        )
        assert re.search(r'<h3[^>]*>\s*other\s*<', html), (
            "Expected 'other' as h3 header in HTML"
        )


# ---------------------------------------------------------------------------
# TestRecentMergesTitles — Title column in recent-merges table (step-13/14)
# ---------------------------------------------------------------------------

_NOW_ISO = '2026-04-22T12:00:00+00:00'


class TestRecentMergesTitles:
    """The recent-merges table has a Title column populated from task-title enrichment."""

    def test_title_column_rendered(self, client):
        """Task title from load_task_titles appears in the recent-merges row."""
        recent_row = {
            'task_id': '7',
            'run_id': 'r1',
            'outcome': 'done',
            'duration_ms': 1000,
            'timestamp': _NOW_ISO,
            'title': 'Fix chart regression',  # pre-enriched (as route does it)
        }
        projects = {
            _DEFAULT_PID: {
                'depth_timeseries': MOCK_DEPTH,
                'outcomes': MOCK_OUTCOMES,
                'latency': MOCK_LATENCY,
                'recent': [recent_row],
                'speculative': MOCK_SPEC,
            }
        }
        with patch('dashboard.app.build_per_project_merge_queue',
                   new_callable=AsyncMock, return_value=projects), \
             patch('dashboard.app.load_task_titles', return_value={'7': 'Fix chart regression'}):
            resp = client.get('/partials/merge-queue')

        assert resp.status_code == 200
        html = resp.text

        # (a) title text appears in the HTML next to task-id 7
        assert 'Fix chart regression' in html, (
            "Expected task title 'Fix chart regression' in HTML"
        )

        # (b) the table header has a 'Title' column
        assert '<th' in html and 'Title' in html, (
            "Expected 'Title' column header in recent-merges table"
        )

    def test_missing_title_renders_gracefully(self, client):
        """A row with no title shows '—' or empty cell (not an error)."""
        recent_row = {
            'task_id': '99',
            'run_id': 'r2',
            'outcome': 'conflict',
            'duration_ms': None,
            'timestamp': _NOW_ISO,
            'title': '',  # no match
        }
        projects = {
            _DEFAULT_PID: {
                'depth_timeseries': MOCK_DEPTH,
                'outcomes': MOCK_OUTCOMES,
                'latency': MOCK_LATENCY,
                'recent': [recent_row],
                'speculative': MOCK_SPEC,
            }
        }
        with patch('dashboard.app.build_per_project_merge_queue',
                   new_callable=AsyncMock, return_value=projects), \
             patch('dashboard.app.load_task_titles', return_value={}):
            resp = client.get('/partials/merge-queue')

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestRecentMergesWindow — 15-minute sliding window (step-15/16)
# ---------------------------------------------------------------------------

_WINDOW_EVENTS_SCHEMA = """\
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


class TestRecentMergesWindow:
    """The 15-minute sliding window is applied to recent merges in the route."""

    def test_only_last_15_minutes_shown(self, client, tmp_path):
        """Only merge_attempt rows within the last 15 minutes appear in the partial.

        Uses a real runs.db (no mock for build_per_project_merge_queue) with 4 events:
        - task-inside-1 (5m ago): should appear
        - task-inside-2 (14m ago): should appear (just within 15m window)
        - task-outside-3 (16m ago): should NOT appear (just outside 15m window)
        - task-outside-4 (30m ago): should NOT appear
        """
        from dashboard.config import DashboardConfig

        FIXED_NOW = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        _FixedDT = make_fixed_datetime_cls(FIXED_NOW)

        # Create the DB under tmp_path to match _project_scoped_dbs_labeled's lookup.
        runs_db = tmp_path / 'data' / 'orchestrator' / 'runs.db'
        runs_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(runs_db))
        conn.executescript(_WINDOW_EVENTS_SCHEMA)
        events = [
            # Inside 15m window — should appear in recent-merges table
            (FIXED_NOW - timedelta(minutes=5), 'run-1', 'task-inside-1'),
            (FIXED_NOW - timedelta(minutes=14), 'run-2', 'task-inside-2'),
            # Outside 15m window — should be filtered out
            (FIXED_NOW - timedelta(minutes=16), 'run-3', 'task-outside-3'),
            (FIXED_NOW - timedelta(minutes=30), 'run-4', 'task-outside-4'),
        ]
        for ts, run_id, task_id in events:
            conn.execute(
                'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
                'VALUES (?, ?, ?, ?, ?)',
                (ts.isoformat(), run_id, task_id, 'merge_attempt',
                 json.dumps({'outcome': 'done'})),
            )
        conn.commit()
        conn.close()

        test_config = DashboardConfig(project_root=tmp_path)
        with patch('dashboard.app.datetime', _FixedDT), \
             patch.object(client.app.state, 'config', test_config):
            resp = client.get('/partials/merge-queue')

        assert resp.status_code == 200
        html = resp.text

        # task-inside-1 and task-inside-2 are within 15 minutes — must appear
        assert 'task-inside-1' in html, (
            'task-inside-1 (5m ago) should appear in recent merges'
        )
        assert 'task-inside-2' in html, (
            'task-inside-2 (14m ago) should appear in recent merges'
        )
        # task-outside-3 and task-outside-4 are outside 15 minutes — must NOT appear
        assert 'task-outside-3' not in html, (
            'task-outside-3 (16m ago) should be filtered out by the 15-minute window'
        )
        assert 'task-outside-4' not in html, (
            'task-outside-4 (30m ago) should be filtered out by the 15-minute window'
        )


# ---------------------------------------------------------------------------
# TestMergeQueueCanvasIdCollision — server-generated canvas_id (step-3/4)
# ---------------------------------------------------------------------------

_PID_COLLISION_A = '/tmp/dark-factory'
_PID_COLLISION_B = '/tmp/dark_factory'

_MOCK_COLLIDING_PROJECT_DATA = {
    _PID_COLLISION_A: {
        'depth_timeseries': MOCK_DEPTH,
        'outcomes': MOCK_OUTCOMES,
        'latency': MOCK_LATENCY,
        'recent': MOCK_RECENT,
        'speculative': MOCK_SPEC,
    },
    _PID_COLLISION_B: {
        'depth_timeseries': MOCK_DEPTH,
        'outcomes': MOCK_OUTCOMES,
        'latency': MOCK_LATENCY,
        'recent': MOCK_RECENT,
        'speculative': MOCK_SPEC,
    },
}


class TestMergeQueueCanvasIdCollision:
    """Two pids that css_id-collide must render distinct canvas id attributes."""

    def test_colliding_pids_render_distinct_canvases(self, client):
        """html contains four distinct canvas ids: unsuffixed + _1 variants for each chart type."""
        with _patch_per_project_merge_data(projects=_MOCK_COLLIDING_PROJECT_DATA):
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200
        html = resp.text

        # First project: unsuffixed
        assert 'id="mergeQueueDepthChart-tmp_dark_factory"' in html, (
            'Expected unsuffixed mergeQueueDepthChart canvas id for first project'
        )
        assert 'id="mergeOutcomeChart-tmp_dark_factory"' in html, (
            'Expected unsuffixed mergeOutcomeChart canvas id for first project'
        )
        # Second project: suffixed with _1
        assert 'id="mergeQueueDepthChart-tmp_dark_factory_1"' in html, (
            'Expected _1-suffixed mergeQueueDepthChart canvas id for second project'
        )
        assert 'id="mergeOutcomeChart-tmp_dark_factory_1"' in html, (
            'Expected _1-suffixed mergeOutcomeChart canvas id for second project'
        )
        # canvas_id key must appear in the serialized allProjects JSON in the inline script
        script_body = _extract_inline_script(html)
        assert '"canvas_id"' in script_body, (
            'canvas_id must be present in the serialized allProjects payload in the inline script'
        )


# ---------------------------------------------------------------------------
# TestUniqueCssIds — unit tests for unique_css_ids helper (step-1/2)
# ---------------------------------------------------------------------------


class TestUniqueCssIds:
    """Unit tests for the unique_css_ids(values) helper in dashboard.app.

    Each test drives a documented case from the plan.  All must fail with
    ImportError or AttributeError until step-2 implements the helper.
    """

    def test_non_colliding_passthrough(self):
        """Non-colliding inputs are returned unchanged."""
        assert unique_css_ids(['a', 'b']) == ['a', 'b']

    def test_two_way_collision(self):
        """/tmp/dark-factory and /tmp/dark_factory both normalize to tmp_dark_factory."""
        result = unique_css_ids(['/tmp/dark-factory', '/tmp/dark_factory'])
        assert result == ['tmp_dark_factory', 'tmp_dark_factory_1']

    def test_three_way_collision(self):
        """Three inputs with the same css_id get _0 (no suffix), _1, _2."""
        result = unique_css_ids(['a-b', 'a.b', 'a b'])
        assert result == ['a_b', 'a_b_1', 'a_b_2']

    def test_counter_collision_edge_case(self):
        """Third 'foo' must skip already-taken foo_1 and land on foo_2."""
        result = unique_css_ids(['foo', 'foo_1', 'foo'])
        assert result == ['foo', 'foo_1', 'foo_2']

    def test_empty_input(self):
        """Empty sequence returns empty list."""
        assert unique_css_ids([]) == []

    def test_order_preserved_mixed(self):
        """Non-colliding and colliding entries are interleaved correctly."""
        result = unique_css_ids(['x', '/tmp/a', 'y', '/tmp/a'])
        assert result == ['x', 'tmp_a', 'y', 'tmp_a_1']


# ---------------------------------------------------------------------------
# TestMergeQueueJsUsesCanvasId — structural guard that JS reads server value
# ---------------------------------------------------------------------------


class TestMergeQueueJsUsesCanvasId:
    """The inline JS must read allProjects[pid].canvas_id, not recompute it.

    This is a structural regression guard: if someone reintroduces the client-side
    css_id regex (e.g. 'pid.replace(...)'), this test will catch it immediately.
    """

    def test_js_reads_canvas_id_from_server(self, client):
        """JS renderAll() uses allProjects[pid].canvas_id, not a local pid.replace regex."""
        with _patch_merge_queue_data():
            resp = client.get('/partials/merge-queue')
        assert resp.status_code == 200

        script_body = _extract_inline_script(resp.text)

        # (a) The JS must read the server-generated canvas_id
        assert 'allProjects[pid].canvas_id' in script_body, (
            "Expected 'allProjects[pid].canvas_id' in the inline script body — "
            "JS should read the server-generated value, not recompute it"
        )

        # (b) The client-side css_id regex must be gone
        assert "pid.replace(/[^a-zA-Z0-9_]/g, '_')" not in script_body, (
            "The client-side pid.replace regex must be removed — "
            "safeId should come from allProjects[pid].canvas_id"
        )
