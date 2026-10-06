"""Wiring tests for the escalation-analytics endpoint + data.js registration.

Follows the source-assertion idiom established in test_tab_escalations.py:
static text checks against data.js (no JS runtime in this project) plus a
TestClient-driven route test against the real FastAPI app.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _dashboard_helpers import (
    assert_script_loads_before,
    extract_df_data_block,
    extract_function_body,
    find_script_position,
    strip_js_comments,
)

# ---------------------------------------------------------------------------
# Helper: extract a named JS/JSX function body (brace-aware).
# Imported from `_dashboard_helpers` — scopes token-presence checks to a
# specific function body rather than searching the entire file (which would
# give false confidence when a token appears in an unrelated context).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# step-17: data.js registers the escalation-analytics endpoint
# ---------------------------------------------------------------------------


def test_data_js_registers_escalation_analytics_endpoint(data_js_body: str) -> None:
    """data.js must register /api/v2/dashboard/escalation-analytics -> ['ESCALATION_ANALYTICS'].

    The entry must be present in the static (unwindowed) section of endpointsFor,
    and the empty-defaults block must initialise ESCALATION_ANALYTICS with the
    Seam-2 contract shape: generated_at/parse_failures/regime_markers/per_project
    (so applyKey has a target before the first fetch resolves).
    """
    assert '/api/v2/dashboard/escalation-analytics' in data_js_body, (
        "data.js does not contain the literal URL '/api/v2/dashboard/escalation-analytics' — "
        'add it to the unwindowed entries in endpointsFor.'
    )
    assert (
        "'ESCALATION_ANALYTICS'" in data_js_body or '"ESCALATION_ANALYTICS"' in data_js_body
    ), (
        "data.js does not reference 'ESCALATION_ANALYTICS' — add it as the mapped key "
        "for '/api/v2/dashboard/escalation-analytics' in endpointsFor."
    )
    assert '/api/v2/dashboard/escalation-analytics?window=' not in data_js_body, (
        'escalation-analytics must stay unwindowed (no ?window= query param) — the '
        'frontend windows client-side over samples/flow_daily (per the PRD).'
    )
    seed_block = extract_df_data_block(data_js_body, 'ESCALATION_ANALYTICS')
    assert seed_block, (
        'data.js does not contain an `ESCALATION_ANALYTICS: { ... }` seed block — '
        'add the initializer to the window.DF_DATA assignment so applyKey has '
        'something to replace on each poll.'
    )
    for field_name in ('generated_at', 'parse_failures', 'regime_markers', 'per_project'):
        assert re.search(rf'\b{field_name}\s*:', seed_block), (
            f"ESCALATION_ANALYTICS seed missing key '{field_name}:' — "
            'add it to the window.DF_DATA ESCALATION_ANALYTICS initializer in data.js.'
        )


# ---------------------------------------------------------------------------
# GET /api/v2/dashboard/escalation-analytics — real config + tmp archive.
# Uses the module-level `client` fixture from conftest.py (function-scoped —
# a fresh TestClient/lifespan per test) so each test can point app.state.config
# at its own tmp project root, mirroring test_api_curator.py's _override_client
# idiom without needing a local helper.
#
# The route lives in dashboard.api.escalations beside /escalations, and both
# read the one escalation corpus. No autouse fixture clears its cache or the
# analytics memo: every route test here requests ``analytics_caches``.
# ---------------------------------------------------------------------------


@pytest.fixture()
def analytics_caches():
    """Clear the corpus cache and the analytics memo before and after the test."""
    from dashboard.api import escalations as escalation_routes
    from dashboard.data import escalation_corpus

    def _clear() -> None:
        escalation_corpus._corpus_cache_clear()
        escalation_routes._analytics_memo_clear()

    _clear()
    yield
    _clear()


def _write_esc(esc_dir: Path, esc: dict) -> None:
    """Write a minimal escalation dict as esc-*.json at the queue root.

    iter_all_escalation_paths only globs 'esc-*.json' (not '*.json') at the
    root tier (escalation.queue), so esc['id'] must start with 'esc-'.
    """
    esc_dir.mkdir(parents=True, exist_ok=True)
    (esc_dir / f"{esc['id']}.json").write_text(json.dumps(esc))


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _make_config(tmp_path: Path, *, known_project_roots: list[Path] | None = None):
    from dashboard.config import DashboardConfig

    return DashboardConfig(project_root=tmp_path, known_project_roots=known_project_roots or [])


def _analytics(client) -> dict:
    resp = client.get('/api/v2/dashboard/escalation-analytics')
    assert resp.status_code == 200
    return resp.json()['ESCALATION_ANALYTICS']


class TestEscalationAnalyticsRoute:
    """GET /api/v2/dashboard/escalation-analytics against the real app + a tmp archive."""

    def test_returns_wrapped_contract_keys(self, client, tmp_path, analytics_caches):
        """200 + ESCALATION_ANALYTICS wrapping the Seam-2 contract keys, and served_at."""
        now = datetime(2026, 7, 16, 18, 0, 0, tzinfo=UTC)
        esc_dir = tmp_path / 'data' / 'escalations'
        _write_esc(esc_dir, {
            'id': 'esc-t1-1', 'task_id': 't1', 'agent_role': 'implementer',
            'severity': 'blocking', 'category': 'cleanup_needed', 'summary': 'x',
            'timestamp': _iso(now - timedelta(days=1)),
            'status': 'resolved', 'level': 0,
            'resolved_at': _iso(now - timedelta(hours=1)),
            'resolved_by': 'interactive',
        })

        client.app.state.config = _make_config(tmp_path)

        resp = client.get('/api/v2/dashboard/escalation-analytics')

        assert resp.status_code == 200
        body = resp.json()
        assert 'ESCALATION_ANALYTICS' in body, (
            'route response missing the ESCALATION_ANALYTICS wrapper key — data.js '
            "applyKey('ESCALATION_ANALYTICS', body['ESCALATION_ANALYTICS']) needs it, "
            'matching the ESCALATIONS/CURATOR_STATE/SCHEDULER envelope precedent.'
        )
        assert isinstance(body['served_at'], str)
        analytics = body['ESCALATION_ANALYTICS']
        assert set(analytics) == {
            'generated_at', 'parse_failures', 'regime_markers', 'per_project',
            # The two archive-reach signals, and the only fields that can tell
            # an absent archive from an empty one: ``archives_present`` (all)
            # is the completeness diagnostic, ``archives_reached`` (any) is
            # the corpus cache's own cacheability fact.
            'archives_present', 'archives_reached',
            # The corpus' named views across every orchestrator queue.
            'views',
        }
        assert isinstance(analytics['generated_at'], str) and analytics['generated_at']
        assert analytics['archives_present'] is True
        assert analytics['archives_reached'] is True
        assert analytics['parse_failures'] == 0
        assert isinstance(analytics['regime_markers'], list)
        assert len(analytics['per_project']) == 1
        entry = analytics['per_project'][0]
        assert entry['project'] == tmp_path.name
        for views in (analytics['views'], entry['views']):
            assert set(views) == {'queue_pending', 'open_in_history'}
            assert all(
                datetime.fromisoformat(wired['as_of'])
                == datetime.fromisoformat(analytics['generated_at'])
                for wired in views.values()
            )

    def test_primary_root_ordered_first(self, client, tmp_path, analytics_caches):
        """per_project lists the primary project_root before known_project_roots."""
        primary = tmp_path / 'primary'
        secondary = tmp_path / 'secondary'
        (primary / 'data' / 'escalations').mkdir(parents=True)
        (secondary / 'data' / 'escalations').mkdir(parents=True)

        client.app.state.config = _make_config(primary, known_project_roots=[secondary])

        per_project = _analytics(client)['per_project']
        assert [p['project'] for p in per_project] == ['primary', 'secondary']

    def test_malformed_regime_markers_never_500s(self, client, tmp_path, monkeypatch,
                                                 analytics_caches):
        """Row 9 (endpoint level): a broken regime-markers file degrades loudly.

        Mirrors load_regime_markers' own fail-soft contract (step-1) but pins it
        through the real route: a corrupt markers file must never surface as a
        500 — only a counted parse_failures increment.
        """
        import dashboard.data.escalation_analytics as escalation_analytics_module
        from dashboard.api.escalations import _analytics_memo_clear

        (tmp_path / 'data' / 'escalations').mkdir(parents=True)
        client.app.state.config = _make_config(tmp_path)

        assert _analytics(client)['parse_failures'] == 0

        bad_markers = tmp_path / 'bad-regime-markers.yaml'
        bad_markers.write_text('date: [unclosed')
        monkeypatch.setattr(
            escalation_analytics_module, '_DEFAULT_REGIME_MARKERS_PATH', bad_markers,
        )
        _analytics_memo_clear()

        assert _analytics(client)['parse_failures'] >= 1


class TestEscalationAnalyticsCacheability:
    """The corpus cache is the one freshness authority; analytics derives from one generation.

    The route memoises its derivation per corpus GENERATION (queues + the
    walk's instant), so the ~10k-record aggregation and the pins fan-out are
    paid once per walk — and can never be served over a different walk than
    the one /escalations reads. Whether a walk is cached at all is the corpus
    cache's rule, asserted here through the route:

    * a walk that reached NO queue directory is served but not cached: it is
      O(1) to repeat (one negative stat per queue), while caching it would
      report an empty archive for a whole TTL after the volume mounts;
    * a PARTIAL walk IS cached. Measured 2026-08-01 against the unit's own
      ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (9 roots), 2 roots have no
      ``data/escalations`` dir while the other 7 hold ~9.1k records, so a
      rule keyed on "every root reached" would cache nothing, ever;
    * unreadable files never gate caching: a corrupt record is permanent.

    The builder is spied on with ``wraps=``, so the real payload is served
    and the spy only counts derivations.
    """

    @staticmethod
    def _spy(monkeypatch) -> MagicMock:
        import dashboard.api.escalations as escalation_routes

        # SYNC, not AsyncMock: the route calls the builder through
        # asyncio.to_thread, which hands it a plain callable.
        build = MagicMock(wraps=escalation_routes.build_escalation_analytics)
        monkeypatch.setattr(escalation_routes, 'build_escalation_analytics', build)
        return build

    def test_a_reached_archive_is_served_from_cache(self, client, monkeypatch, tmp_path,
                                                    analytics_caches):
        """The ordinary walk is derived once per TTL."""
        (tmp_path / 'data' / 'escalations').mkdir(parents=True)
        client.app.state.config = _make_config(tmp_path)
        build = self._spy(monkeypatch)

        _analytics(client)
        _analytics(client)

        assert build.call_count == 1, f'expected 1 build call, got {build.call_count}'

    def test_a_partial_scan_is_still_cached(self, client, monkeypatch, tmp_path,
                                            analytics_caches):
        """THE regression test: one archive-less root must not defeat the cache.

        A real two-root config in the production shape — the primary root has
        ``data/escalations`` on disk, the secondary has never escalated and so
        has no such dir: the installed 9-root config in miniature.
        """
        primary = tmp_path / 'primary'
        secondary = tmp_path / 'secondary'
        (primary / 'data' / 'escalations').mkdir(parents=True)
        secondary.mkdir()

        client.app.state.config = _make_config(primary, known_project_roots=[secondary])
        build = self._spy(monkeypatch)

        _analytics(client)
        second = _analytics(client)

        assert second['archives_present'] is False
        assert build.call_count == 1, f'expected 1 build call, got {build.call_count}'

    def test_an_unreached_archive_is_not_pinned_for_the_ttl(self, client, monkeypatch,
                                                            tmp_path, analytics_caches):
        """A walk that reached nothing is re-walked every poll, so a new archive shows at once."""
        client.app.state.config = _make_config(tmp_path)
        build = self._spy(monkeypatch)

        first = _analytics(client)
        _write_esc(tmp_path / 'data' / 'escalations', {
            'id': 'esc-t1-1', 'task_id': 't1', 'agent_role': 'implementer',
            'severity': 'blocking', 'category': 'cleanup_needed', 'summary': 'x',
            'timestamp': '2026-07-15T18:00:00+00:00', 'status': 'pending', 'level': 0,
        })
        second = _analytics(client)

        assert first['archives_reached'] is False
        assert second['archives_reached'] is True
        assert second['per_project'][0]['views']['open_in_history']['value'] == 1
        assert build.call_count == 2, f'expected 2 build calls, got {build.call_count}'

    def test_a_new_corpus_generation_is_derived_afresh(self, client, monkeypatch, tmp_path,
                                                       analytics_caches):
        """The memo answers for one walk only: a new walk inside its TTL is re-derived."""
        from dashboard.data import escalation_corpus

        (tmp_path / 'data' / 'escalations').mkdir(parents=True)
        client.app.state.config = _make_config(tmp_path)
        build = self._spy(monkeypatch)

        first = _analytics(client)
        escalation_corpus._corpus_cache_clear()
        second = _analytics(client)

        assert second['generated_at'] != first['generated_at']
        assert build.call_count == 2, f'expected 2 build calls, got {build.call_count}'

    def test_parse_failures_alone_never_defeats_the_cache(self, client, monkeypatch,
                                                          tmp_path, analytics_caches):
        """A corrupt record is permanent; gating on it would re-walk on every poll, forever."""
        esc_dir = tmp_path / 'data' / 'escalations'
        esc_dir.mkdir(parents=True)
        (esc_dir / 'esc-bad-1.json').write_text('{not json')
        client.app.state.config = _make_config(tmp_path)
        build = self._spy(monkeypatch)

        _analytics(client)
        second = _analytics(client)

        assert second['parse_failures'] == 1
        assert build.call_count == 1, f'expected 1 build call, got {build.call_count}'

    def test_real_builder_partial_scan_is_cached_end_to_end(self, client, tmp_path,
                                                            analytics_caches):
        """The binding test: NO spy anywhere, real builder through the real route.

        ``generated_at`` is the corpus walk's instant, stamped from the live
        clock at microsecond resolution, so two GETs that return the SAME
        ``generated_at`` were served from ONE walk — a re-walk would carry a
        later stamp.
        """
        primary = tmp_path / 'primary'
        secondary = tmp_path / 'secondary'
        (primary / 'data' / 'escalations').mkdir(parents=True)
        secondary.mkdir()

        client.app.state.config = _make_config(primary, known_project_roots=[secondary])

        first = _analytics(client)
        second = _analytics(client)

        # The producer really does distinguish the two questions over a real tree.
        assert first['archives_present'] is False
        assert first['archives_reached'] is True
        # ...and the partial walk really is cached.
        assert second['generated_at'] == first['generated_at'], (
            'second GET walked again — a partial scan defeated the corpus cache'
        )


# ---------------------------------------------------------------------------
# task 2659 (delta) — tab_escalation_analytics.jsx UI wiring
# ---------------------------------------------------------------------------
#
# The tests below consume the served-asset fixtures (tab_analytics_jsx_body,
# app_jsx_body, shell_jsx_body, index_html_body) that now live in conftest.py
# and `extract_function_body` from _dashboard_helpers (task 3549), plus the
# load-order helpers `find_script_position` and `assert_script_loads_before`,
# which are ALSO imported from _dashboard_helpers rather than defined here:
# task 4881 retired the byte-identical copy this module and four others each
# carried, together with the `ScriptTagCollector` those two parse with (which
# is why it is not in this module's import list — nothing here calls it
# directly).  Their contract lives in
# test_jsx_source_helpers.py::TestScriptOrderHelpers.


# ---------------------------------------------------------------------------
# step-1 test: tab_escalation_analytics.jsx is served and exports the component
# ---------------------------------------------------------------------------


def test_tab_analytics_jsx_served_and_exports(_client) -> None:
    """GET /static/redux/tab_escalation_analytics.jsx returns 200 with the expected wiring.

    Asserts:
    (a) 200 HTTP status.
    (b) function EscalationAnalyticsTab( is declared.
    (c) exports ADDITIVELY via window.DF_TABS.EscalationAnalyticsTab = (not
        window.DF_TABS = {).
    (d) reads window.DF_DATA.ESCALATION_ANALYTICS (or an aliased DF.ESCALATION_ANALYTICS).
    (e) renders <ProjectGroup (subsection-per-project).
    (f) fold state is persisted via useOpenSet( referencing 'df.open.escanalytics'.

    (d)-(f) are scoped to the extracted `EscalationAnalyticsTab` function body
    via `extract_function_body`, and (c)'s export check requires the actual
    `=` assignment syntax rather than a bare dotted-path substring — this
    file's own header comment mentions "window.DF_TABS.EscalationAnalyticsTab"
    in prose, so an unscoped raw substring check would still pass even if the
    real wiring were deleted from the component.
    """
    resp = _client.get('/static/redux/tab_escalation_analytics.jsx')
    assert resp.status_code == 200, (
        f'Expected 200 for /static/redux/tab_escalation_analytics.jsx, got {resp.status_code}'
    )
    body = resp.text
    assert 'function EscalationAnalyticsTab(' in body, (
        'tab_escalation_analytics.jsx does not define `function EscalationAnalyticsTab(` — '
        'the component must be declared as a named function for the export to work.'
    )
    tab_body = extract_function_body(body, 'EscalationAnalyticsTab')
    # Additive export — must NOT clobber window.DF_TABS = {...} and must assign
    # EscalationAnalyticsTab. Requires the assignment's `=` (not just the
    # dotted path) so a prose mention in a comment cannot satisfy the check.
    assert re.search(
        r'window\.DF_TABS\.EscalationAnalyticsTab\s*=\s*EscalationAnalyticsTab\b', body
    ), (
        'tab_escalation_analytics.jsx does not set window.DF_TABS.EscalationAnalyticsTab — '
        'add `window.DF_TABS.EscalationAnalyticsTab = EscalationAnalyticsTab;` at the bottom '
        'of the file to export additively without clobbering the existing window.DF_TABS object.'
    )
    assert 'window.DF_TABS = {' not in body, (
        'tab_escalation_analytics.jsx clobbers window.DF_TABS = {...} — tabs.jsx already '
        'creates that object; this file must mutate it additively instead.'
    )
    # Reads ESCALATION_ANALYTICS data — scoped to the component body so a
    # token surviving only in a comment cannot produce a false pass.
    assert 'ESCALATION_ANALYTICS' in tab_body, (
        'EscalationAnalyticsTab does not reference ESCALATION_ANALYTICS — it should '
        'read window.DF_DATA.ESCALATION_ANALYTICS (or an alias like DF.ESCALATION_ANALYTICS) '
        'for its data source.'
    )
    # Renders ProjectGroup for subsection-per-project folding
    assert '<ProjectGroup' in tab_body, (
        'EscalationAnalyticsTab does not render <ProjectGroup — each project must be '
        'wrapped in a ProjectGroup from window.DF_SHELL for foldable sections.'
    )
    # Fold state persisted with the correct key
    assert 'useOpenSet(' in tab_body, (
        'EscalationAnalyticsTab does not call useOpenSet( — add the local copy of '
        "useOpenSet from tab_escalations.jsx and call it with project ids and 'df.open.escanalytics'."
    )
    assert "'df.open.escanalytics'" in tab_body, (
        "EscalationAnalyticsTab does not reference the localStorage key "
        "'df.open.escanalytics' — pass it as the storageKey argument to useOpenSet so fold "
        'state is persisted.'
    )


# ---------------------------------------------------------------------------
# step-3 test: index.html registers tab_escalation_analytics.jsx in correct
# load order
# ---------------------------------------------------------------------------


def test_index_html_registers_tab_analytics_load_order(index_html_body: str) -> None:
    """index.html must include tab_escalation_analytics.jsx, loaded AFTER
    data.js, shell.jsx, and tabs.jsx and BEFORE app.jsx; must be a classic
    synchronous script (no defer/async/type=module); and every
    /static/redux/*?v= cache-buster must be at or past this tab's floor of 30.

    Check (g) is an ANTI-REVERT PIN, not a live bump check: index.html is far
    past 30 today, so it fails only if someone rolls the cache-busters back
    below what this tab needed. Whether the versions are UNIFORM, and whether
    the newest bump landed, are both asserted in test_index_html.py.

    Checks:
    (a) tab_escalation_analytics.jsx script tag exists.
    (b) Loads after data.js.
    (c) Loads after shell.jsx.
    (d) Loads after tabs.jsx.
    (e) Loads before app.jsx.
    (f) Not deferred/async/module.
    (g) Every /static/redux/ v= cache-buster is >= 30.
    """
    _TAB_ANALYTICS_PREFIX = '/static/redux/tab_escalation_analytics.jsx'

    # (a) tab_escalation_analytics.jsx script tag must exist
    result = find_script_position(index_html_body, _TAB_ANALYTICS_PREFIX)
    assert result is not None, (
        f'No <script src="{_TAB_ANALYTICS_PREFIX}..."> tag found in index.html — '
        'add it after tab_escalations.jsx and before app.jsx.'
    )
    _, analytics_attrs = result

    # (f) Must be a classic synchronous script
    assert 'defer' not in analytics_attrs, (
        'tab_escalation_analytics.jsx script tag has defer= — remove it; classic '
        'synchronous scripts are required for Babel-standalone transpilation.'
    )
    assert 'async' not in analytics_attrs, (
        'tab_escalation_analytics.jsx script tag has async= — remove it.'
    )
    assert (analytics_attrs.get('type') or '').lower() in ('text/babel', ''), (
        'tab_escalation_analytics.jsx script must have type="text/babel" (or no type) — '
        f'got {analytics_attrs.get("type")!r}.'
    )

    # (b) Loads after data.js
    assert_script_loads_before(
        index_html_body,
        '/static/redux/data.js',
        _TAB_ANALYTICS_PREFIX,
        'data.js',
        'tab_escalation_analytics.jsx',
        'data.js must load before tab_escalation_analytics.jsx so window.DF_DATA is seeded.',
    )

    # (c) Loads after shell.jsx
    assert_script_loads_before(
        index_html_body,
        '/static/redux/shell.jsx',
        _TAB_ANALYTICS_PREFIX,
        'shell.jsx',
        'tab_escalation_analytics.jsx',
        'shell.jsx must load before tab_escalation_analytics.jsx so window.DF_SHELL is available.',
    )

    # (d) Loads after tabs.jsx
    assert_script_loads_before(
        index_html_body,
        '/static/redux/tabs.jsx',
        _TAB_ANALYTICS_PREFIX,
        'tabs.jsx',
        'tab_escalation_analytics.jsx',
        'tabs.jsx must load before tab_escalation_analytics.jsx so window.DF_TABS is available to mutate.',
    )

    # (e) Loads before app.jsx
    assert_script_loads_before(
        index_html_body,
        _TAB_ANALYTICS_PREFIX,
        '/static/redux/app.jsx',
        'tab_escalation_analytics.jsx',
        'app.jsx',
        'tab_escalation_analytics.jsx must load before app.jsx so EscalationAnalyticsTab is set on window.DF_TABS.',
    )

    # (g) floor only — uniformity lives in test_index_html.py (see docstring).
    versions = {int(v) for v in re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body)}
    assert versions, (
        'index.html carries no /static/redux/*?v=<n> asset tags at all — the '
        'cache-buster convention has been dropped or the URLs were rewritten.'
    )
    assert min(versions) >= 30, (
        f'the oldest index.html cache-buster version is {min(versions)}, '
        'expected >= 30 (the floor tab_escalation_analytics.jsx landed at).'
    )


# ---------------------------------------------------------------------------
# step-5 test: app.jsx wires the esc-analytics tab
# ---------------------------------------------------------------------------


def test_app_jsx_wires_analytics_tab(app_jsx_body: str) -> None:
    """app.jsx must destructure EscalationAnalyticsTab from window.DF_TABS,
    add an 'esc-analytics' tab entry, handle it in renderTab, and give it a
    toolbarConfig entry.

    Asserts four structural wiring contracts:
    (a) EscalationAnalyticsTab is destructured from window.DF_TABS.
    (b) tabs[] contains an entry with id 'esc-analytics' and label 'Analytics'.
    (c) renderTab switch has a `case 'esc-analytics':` branch returning
        <EscalationAnalyticsTab projectFilter={projects} />.
    (d) toolbarConfig has an 'esc-analytics' entry.

    That the tab carries no global window chip (it owns its own client-side
    7d/28d toggle) is no longer a source grep here: it is executed by
    dashboard/tests/js/window_chip.test.mjs (b), which asserts esc-analytics
    has no window_chip.js::TAB_WINDOWS entry and windowForTab leaves the
    window alone on it.
    """
    # (a) EscalationAnalyticsTab destructured from window.DF_TABS
    assert re.search(
        r'const\s*\{[^}]*EscalationAnalyticsTab[^}]*\}\s*=\s*window\.DF_TABS', app_jsx_body
    ), (
        'app.jsx does not destructure EscalationAnalyticsTab from window.DF_TABS — add '
        '`EscalationAnalyticsTab` to the `const { ... } = window.DF_TABS;` destructure.'
    )
    # (b) tabs[] entry
    assert "id: 'esc-analytics'" in app_jsx_body, (
        "app.jsx tabs array does not contain an entry with `id: 'esc-analytics'` — "
        "add `{ id: 'esc-analytics', label: 'Analytics' }` to the tabs array."
    )
    assert re.search(r"id:\s*'esc-analytics'[^}]*label:\s*'Analytics'", app_jsx_body), (
        "app.jsx tabs array entry for 'esc-analytics' does not have `label: 'Analytics'` — "
        "add `{ id: 'esc-analytics', label: 'Analytics' }` to the tabs array."
    )
    # (c) renderTab switch case
    assert "case 'esc-analytics':" in app_jsx_body, (
        "app.jsx renderTab switch does not have `case 'esc-analytics':` — add the case "
        'branch to render <EscalationAnalyticsTab projectFilter={projects} />.'
    )
    assert re.search(
        r"case 'esc-analytics':\s*return\s*<EscalationAnalyticsTab\s+projectFilter=\{projects\}",
        app_jsx_body,
    ), (
        "app.jsx renderTab `case 'esc-analytics':` does not return "
        '<EscalationAnalyticsTab projectFilter={projects} /> — add it.'
    )
    # (d) toolbarConfig esc-analytics entry
    parts = app_jsx_body.split('toolbarConfig')
    assert len(parts) > 1 and "'esc-analytics'" in parts[1], (
        "app.jsx toolbarConfig does not have an 'esc-analytics' entry — add "
        "`'esc-analytics': { showAgents: false, search: false }` to toolbarConfig."
    )


# ---------------------------------------------------------------------------
# step-7 test: shell.jsx registers the analytics rail entry + glyph
# ---------------------------------------------------------------------------


def test_shell_jsx_registers_analytics_rail_and_glyph(shell_jsx_body: str) -> None:
    """shell.jsx must include a Rail item with id 'esc-analytics' and a Glyph
    case 'esc-analytics'.

    Asserts only the routing-relevant id and glyph-key wiring.  Cosmetic
    fields (label, SVG path data) are intentionally omitted — they can be
    renamed without breaking the routing.
    """
    assert "id: 'esc-analytics'" in shell_jsx_body, (
        "shell.jsx Rail items array does not contain an entry with "
        "`id: 'esc-analytics'` — add the analytics item to the Rail items array."
    )
    assert "case 'esc-analytics':" in shell_jsx_body, (
        "shell.jsx Glyph switch does not have a `case 'esc-analytics':` branch — "
        'add an esc-analytics case returning a simple stroke SVG.'
    )


# ---------------------------------------------------------------------------
# step-9 test: cross-cutting foundation — window toggle, slice helper,
# parse_failures chip, regime-marker overlay
# ---------------------------------------------------------------------------


def test_tab_analytics_window_toggle_and_crosscutting(tab_analytics_jsx_body: str) -> None:
    """tab_escalation_analytics.jsx must have the cross-cutting foundation every
    panel (Origin/Lifespan/Workflow) builds on:

    (a) EscalationAnalyticsTab renders a `<Segmented` window toggle with
        '7d'/'28d'/'all' options, default '28d', persisted via
        `usePersistedState('df.escanalytics.window', '28d')`.
    (b) A `windowCutoffDate`/`sliceDailyByWindow` helper pair anchors window
        slicing to the payload's `generated_at` — NOT the browser clock
        (`Date.now`).
    (c) A `parse_failures` warning chip, rendered only when `> 0`.
    (d) A reusable `RegimeMarkers` (or `ChartMarkers`) overlay component fed
        the payload's real `regime_markers` data (not just the
        empty-defaults literal), plus a `TimeChart` wrapper that composes it
        with a chart primitive.
    """
    body = tab_analytics_jsx_body

    # (a) Window toggle, scoped to EscalationAnalyticsTab's own body.
    tab_body = extract_function_body(body, 'EscalationAnalyticsTab')
    assert re.search(
        r"usePersistedState\(\s*['\"]df\.escanalytics\.window['\"]\s*,\s*['\"]28d['\"]\s*\)",
        tab_body,
    ), (
        "EscalationAnalyticsTab does not call "
        "usePersistedState('df.escanalytics.window', '28d') — the window toggle "
        'must default to 28d and persist under that key.'
    )
    assert '<Segmented' in tab_body, (
        'EscalationAnalyticsTab does not render a <Segmented toggle for the '
        '7d/28d/all window control.'
    )
    for option in ("'7d'", "'28d'", "'all'"):
        assert option in tab_body, (
            f'EscalationAnalyticsTab window Segmented is missing the {option} option.'
        )

    # (c) parse_failures warning chip, guarded by > 0, also in the tab body.
    assert re.search(r'analytics\.parse_failures\s*>\s*0', tab_body), (
        'EscalationAnalyticsTab does not guard a parse_failures warning chip with '
        '`analytics.parse_failures > 0` — add one so a corrupt-archive count is '
        'surfaced loudly (INV-4) instead of silently.'
    )

    # (b) Client-side window-slice helpers, anchored to generated_at.
    assert 'function windowCutoffDate(' in body, (
        'tab_escalation_analytics.jsx does not define `function windowCutoffDate(` — '
        'add the helper that computes the window cutoff relative to generated_at.'
    )
    cutoff_body = extract_function_body(body, 'windowCutoffDate')
    assert 'generatedAt' in cutoff_body, (
        'windowCutoffDate does not reference its generatedAt parameter — the window '
        'cutoff must be anchored to the payload clock, not the browser clock.'
    )
    assert 'Date.now(' not in cutoff_body, (
        'windowCutoffDate calls Date.now() — window slicing must be anchored to the '
        "payload's generated_at, immune to browser-clock skew, not the live clock."
    )
    assert 'function sliceDailyByWindow(' in body, (
        'tab_escalation_analytics.jsx does not define `function sliceDailyByWindow(` — '
        'add the helper that filters date-keyed daily buckets down to the active window.'
    )

    # (d) Reusable regime-marker overlay, fed the real regime_markers field,
    # plus the TimeChart wrapper every panel chart will render through.
    assert (
        'function RegimeMarkers(' in body or 'function ChartMarkers(' in body
    ), (
        'tab_escalation_analytics.jsx does not define a RegimeMarkers/ChartMarkers '
        'overlay component — add one so time charts can render vertical regime '
        'markers without modifying charts.jsx.'
    )
    assert re.search(r'\.regime_markers\b', body), (
        'tab_escalation_analytics.jsx never dot-references `.regime_markers` off the '
        'analytics payload — the marker overlay must be fed the real '
        'analytics.regime_markers data, not just the empty-defaults literal.'
    )
    assert 'function TimeChart(' in body, (
        'tab_escalation_analytics.jsx does not define `function TimeChart(` — add the '
        'wrapper that composes a chart primitive with the RegimeMarkers overlay so '
        'every panel chart gets regime markers consistently.'
    )


# ---------------------------------------------------------------------------
# step-11 test: Origin panel
# ---------------------------------------------------------------------------


def test_tab_analytics_origin_panel(tab_analytics_jsx_body: str) -> None:
    """The Origin panel (top-N-by-source filings chart + benign-rate table).

    Asserts, scoped to the Origin panel's own function body:
    (a) A `StackedAreaChart` fed `daily_by_source`, with the long tail folded
        into an `'other'` bucket.
    (b) A benign-rate table over `origin.sources` referencing `benign_rate`
        and `stamped_share` (stamped-vs-inferred split), whose segmented bar
        draws one segment per served resolution class —
        escalation_views.js::resolutionSegments over `s.classes` — rather than
        a fixed benign/actionable pair that hides every other class.
    (c) A `predictably_benign` badge.
    (d) A per-source `Sparkline` fed by `daily_spark`.
    (e) Rows sorted by benign COUNT — a `.sort(` referencing `.benign`.
    """
    body = tab_analytics_jsx_body

    assert 'function OriginPanel(' in body, (
        'tab_escalation_analytics.jsx does not define `function OriginPanel(` — '
        'add the Origin panel component.'
    )
    origin_body = extract_function_body(body, 'OriginPanel')

    # (a) StackedAreaChart over daily_by_source, long tail folded into 'other'.
    assert 'daily_by_source' in origin_body, (
        'OriginPanel does not reference `daily_by_source` — the filings-by-source '
        'chart must be built from origin.daily_by_source.'
    )
    assert '<C.StackedAreaChart' in origin_body, (
        'OriginPanel does not render <C.StackedAreaChart — the top-N-by-source '
        'filings/day chart must use the StackedAreaChart primitive from '
        'window.DF_CHARTS.'
    )
    assert "'other'" in origin_body, (
        "OriginPanel does not reference an `'other'` bucket — sources beyond the "
        'top-N must be folded into an "other" stack rather than dropped.'
    )

    # (b) Benign-rate table over sources[], stamped-vs-inferred split, actionable.
    assert 'sources' in origin_body, (
        'OriginPanel does not reference `sources` — the benign-rate table must be '
        'built from origin.sources.'
    )
    assert 'benign_rate' in origin_body, (
        'OriginPanel does not reference `benign_rate` in the per-source table.'
    )
    assert 'stamped_share' in origin_body, (
        'OriginPanel does not reference `stamped_share` — render the stamped-vs-'
        'inferred split.'
    )
    assert re.search(r'resolutionSegments\(\s*s\.classes\s*\)\.map\(', origin_body), (
        'OriginPanel does not draw its bar from `resolutionSegments(s.classes).map(` — '
        'every served resolution class gets a segment, so the parts add up to the '
        'whole the benign rate is a share of.'
    )
    origin_code = strip_js_comments(origin_body)
    assert not re.search(r'\bs\.(benign|actionable)\b', origin_code), (
        'OriginPanel still reads `s.benign`/`s.actionable` — the payload serves the '
        'split as `s.classes`, keyed by every resolution class.'
    )

    # (c) predictably_benign badge.
    assert 'predictably_benign' in origin_body, (
        'OriginPanel does not reference `predictably_benign` — render a badge when '
        'a source is predictably benign.'
    )

    # (d) Per-source Sparkline fed by daily_spark.
    assert '<C.Sparkline' in origin_body, (
        'OriginPanel does not render <C.Sparkline — add the per-source daily_spark '
        'column.'
    )
    assert 'daily_spark' in origin_body, (
        'OriginPanel does not reference `daily_spark` — feed it to the per-source '
        'Sparkline column.'
    )

    # (e) Rows sorted by benign COUNT: SOME .sort( call references .benign nearby
    # (the panel may also .sort() dates/rankings for the chart — scan every
    # occurrence rather than assuming the first one is the row sort).
    sort_positions = [m.start() for m in re.finditer(r'\.sort\(', origin_body)]
    assert sort_positions, (
        'OriginPanel does not call `.sort(` — the benign-rate table rows must be '
        'sorted DESC by benign count.'
    )
    assert any('.benign' in origin_body[i : i + 120] for i in sort_positions), (
        'No `.sort(` call in OriginPanel references `.benign` nearby — rows must be '
        'sorted by benign COUNT (volume × rate), not alphabetically or by filings.'
    )


# ---------------------------------------------------------------------------
# step-13 test: Lifespan panel
# ---------------------------------------------------------------------------


def test_tab_analytics_lifespan_panel(tab_analytics_jsx_body: str) -> None:
    """The Lifespan panel (percentile tiles + tier-overlaid ECDF + open items).

    Asserts, scoped to the Lifespan panel's own function body:
    (a) `StatTile` percentiles keyed by level, from `percentiles_by_level`.
    (b) A client-side ECDF built from `lifespan.samples`, fed to a
        `LineChart` with one series per resolver tier.
    (c) A log-x treatment (`Math.log`) with a vertical 6h (`21600`) freshness
        marker.
    (d) An open-items list sorted DESC by `age_secs`, with `breach_6h`
        highlighting.
    (e) A render-when-present `triage_segments` block, guarded by
        `lifespan.triage_segments &&`.
    """
    body = tab_analytics_jsx_body

    assert 'function LifespanPanel(' in body, (
        'tab_escalation_analytics.jsx does not define `function LifespanPanel(` — '
        'add the Lifespan panel component.'
    )
    lifespan_body = extract_function_body(body, 'LifespanPanel')

    # (a) StatTile percentiles keyed by level from percentiles_by_level.
    assert 'percentiles_by_level' in lifespan_body, (
        'LifespanPanel does not reference `percentiles_by_level` — the percentile '
        'StatTiles must be built from lifespan.percentiles_by_level.'
    )
    assert '<C.StatTile' in lifespan_body, (
        'LifespanPanel does not render <C.StatTile — add percentile tiles per level.'
    )

    # (b) ECDF from samples, overlaid by resolver tier, on a LineChart.
    assert 'samples' in lifespan_body, (
        'LifespanPanel does not reference `samples` — the ECDF must be built from '
        'lifespan.samples.'
    )
    assert '<C.LineChart' in lifespan_body, (
        'LifespanPanel does not render <C.LineChart — the ECDF must use the '
        'LineChart primitive.'
    )
    assert 'tier' in lifespan_body, (
        'LifespanPanel does not reference `tier` — the ECDF must be overlaid with '
        'one series per resolver tier.'
    )

    # (c) log-x treatment + vertical 6h freshness marker.
    assert 'Math.log' in lifespan_body, (
        'LifespanPanel does not reference `Math.log` — the ECDF threshold grid must '
        'be log-spaced (log-x axis).'
    )
    assert '21600' in lifespan_body, (
        'LifespanPanel does not reference `21600` (6h in seconds) — add a vertical '
        '6h freshness marker over the ECDF.'
    )

    # (d) Open-items list sorted DESC by age_secs, breach_6h highlighting.
    assert 'open_items' in lifespan_body, (
        'LifespanPanel does not reference `open_items` — add the open-items list.'
    )
    sort_positions = [m.start() for m in re.finditer(r'\.sort\(', lifespan_body)]
    assert sort_positions, (
        'LifespanPanel does not call `.sort(` — open_items must be sorted DESC by '
        'age_secs.'
    )
    assert any('age_secs' in lifespan_body[i : i + 120] for i in sort_positions), (
        'No `.sort(` call in LifespanPanel references `age_secs` nearby — open '
        'items must be ranked by pending age.'
    )
    assert 'breach_6h' in lifespan_body, (
        'LifespanPanel does not reference `breach_6h` — highlight open items that '
        'have breached the 6h freshness threshold.'
    )

    # (e) render-when-present triage_segments block (2555 forward-compat).
    assert re.search(r'lifespan\.triage_segments\s*&&', lifespan_body), (
        'LifespanPanel does not guard a block with `lifespan.triage_segments &&` — '
        'the filed→triaged→resolved segment block must render only when present, '
        'not be zero-filled.'
    )


# ---------------------------------------------------------------------------
# step-15 test: Workflow panel
# ---------------------------------------------------------------------------


def test_tab_analytics_workflow_panel(tab_analytics_jsx_body: str) -> None:
    """The Workflow panel (tier-absorption chart + action mix + churn/throughput
    + a reserved ζ flow-diagram mount seam).

    Asserts, scoped to the Workflow panel's own function body:
    (a) A 100%-normalized `StackedAreaChart` of tier absorption from
        `tier_weekly`, with a per-week normalization (division by a week
        total).
    (b) A total-volume `Sparkline` above the tier-absorption chart.
    (c) An action-mix `Donut` from `action_mix`.
    (d) A churn `LineChart` from `churn_daily`.
    (e) An esc-per-done `LineChart` from `esc_per_done_daily`, plotting
        `ratio`.
    (f) A reserved `esc-flow-slot` mount seam fed the windowed `flow_daily`.
    """
    body = tab_analytics_jsx_body

    assert 'function WorkflowPanel(' in body, (
        'tab_escalation_analytics.jsx does not define `function WorkflowPanel(` — '
        'add the Workflow panel component.'
    )
    workflow_body = extract_function_body(body, 'WorkflowPanel')

    # (a) 100%-normalized StackedAreaChart of tier absorption from tier_weekly.
    assert 'tier_weekly' in workflow_body, (
        'WorkflowPanel does not reference `tier_weekly` — the tier-absorption '
        'chart must be built from workflow.tier_weekly.'
    )
    assert '<C.StackedAreaChart' in workflow_body, (
        'WorkflowPanel does not render <C.StackedAreaChart — the 100%-normalized '
        'tier-absorption chart must use the StackedAreaChart primitive.'
    )
    assert re.search(r'/\s*\w*[Tt]otal', workflow_body), (
        'WorkflowPanel does not divide by a week-total-like variable — the '
        'tier-absorption chart must be 100%-normalized (each tier count ÷ week '
        'total), not raw counts.'
    )

    # (b) total-volume Sparkline above the tier-absorption chart.
    assert '<C.Sparkline' in workflow_body, (
        'WorkflowPanel does not render <C.Sparkline — add a total-volume '
        'sparkline above the tier-absorption chart.'
    )

    # (c) action-mix Donut.
    assert 'action_mix' in workflow_body, (
        'WorkflowPanel does not reference `action_mix` — add the action-mix donut.'
    )
    assert '<C.Donut' in workflow_body, (
        'WorkflowPanel does not render <C.Donut — the action-mix chart must use '
        'the Donut primitive.'
    )
    # The donut states the population it divides: the project's `terminal`
    # count, the one whole that origin's classes and action_mix both sum to.
    assert re.search(r'of \{terminal\} terminal', workflow_body), (
        'WorkflowPanel does not caption the action-mix donut `of {terminal} terminal` '
        '— the donut must say which population its shares are of.'
    )
    tab_body = extract_function_body(body, 'EscalationAnalyticsTab')
    assert re.search(r'<WorkflowPanel\b[^>]*\bterminal=\{p\.terminal\}', tab_body), (
        'EscalationAnalyticsTab does not pass `terminal={p.terminal}` to WorkflowPanel.'
    )

    # (d)/(e) churn + esc-per-done LineCharts (two distinct charts).
    assert 'churn_daily' in workflow_body, (
        'WorkflowPanel does not reference `churn_daily` — add the churn LineChart.'
    )
    assert 'esc_per_done_daily' in workflow_body, (
        'WorkflowPanel does not reference `esc_per_done_daily` — add the '
        'esc-per-done LineChart.'
    )
    line_chart_positions = [m.start() for m in re.finditer(r'<C\.LineChart', workflow_body)]
    assert len(line_chart_positions) >= 2, (
        'WorkflowPanel must render at least two <C.LineChart charts — one for '
        'churn_daily, one for esc_per_done_daily.'
    )
    assert any(
        'ratio' in workflow_body[i : i + 300] for i in line_chart_positions
    ), (
        'No <C.LineChart in WorkflowPanel is fed a `ratio`-derived series nearby — '
        'the esc-per-done chart must plot esc_per_done_daily[].ratio.'
    )

    # (f) reserved esc-flow-slot mount seam, fed the windowed flow_daily.
    assert 'esc-flow-slot' in workflow_body, (
        'WorkflowPanel does not render an `esc-flow-slot` placeholder — reserve a '
        'stable mount seam for the ζ lifecycle-flow-diagram component (dep on δ).'
    )
    assert 'flow_daily' in workflow_body, (
        'WorkflowPanel does not reference `flow_daily` — the esc-flow-slot must be '
        'fed the windowed flow_daily data.'
    )


def test_esc_per_done_chart_does_not_compact_its_series(tab_analytics_jsx_body: str) -> None:
    """A null-ratio day keeps its x-axis slot instead of being dropped (task 3489).

    ``esc_per_done_daily[].ratio`` is null on days where done == 0 — the ONLY
    consumer in the dashboard that originates a real hole.  This panel used to
    drop those rows entirely::

        const epdRows = escPerDoneDaily.filter(row => row.ratio != null);
        const epdDates = epdRows.map(row => row.date);

    which is worse than it looks: the filter removes the day from the LABEL row
    as well as from the values, so the series is COMPACTED and every surviving
    sample is silently redated — a Tuesday reading slides into Monday's slot.
    That is the precise hazard spark_path.js's header names, and it was only
    ever a workaround for LineChart having no gap support.  LineChart now
    breaks its line across a hole (task 3489), so the workaround is obsolete
    AND actively wrong.

    Asserted structurally rather than by comment wording: the dates and the
    values must come from the SAME row list, and that list must be the windowed
    rows themselves, not a filtered copy.
    """
    workflow_body = extract_function_body(tab_analytics_jsx_body, 'WorkflowPanel')

    filter_on_ratio = re.search(r'\.filter\([^)]*\bratio\b[^)]*\bnull\b', workflow_body)
    assert filter_on_ratio is None, (
        'WorkflowPanel still filters rows on a null `ratio`: '
        f'`{filter_on_ratio.group(0) if filter_on_ratio else ""}`. Dropping the '
        'row removes its date from the label row too, compacting the x-axis and '
        'redating every surviving sample. Pass the null through instead — '
        'LineChart draws it as a gap.'
    )

    date_sources = set(
        re.findall(r'(\w+)\s*\.map\(\s*\(?\s*\w+\s*\)?\s*=>\s*\w+\.date\b', workflow_body)
    )
    ratio_sources = set(
        re.findall(r'(\w+)\s*\.map\(\s*\(?\s*\w+\s*\)?\s*=>\s*\w+\.ratio\b', workflow_body)
    )
    assert date_sources, (
        'WorkflowPanel derives no `.date` label row via a `.map(row => row.date)` '
        '— the esc-per-done chart needs one label per row.'
    )
    assert ratio_sources, (
        'WorkflowPanel derives no `.ratio` series via a `.map(row => row.ratio)` '
        '— the esc-per-done chart must plot the ratios.'
    )
    assert date_sources == ratio_sources, (
        f'the esc-per-done labels come from {sorted(date_sources)} but the values '
        f'from {sorted(ratio_sources)}. Both must be derived from the SAME row '
        f'list, or a dropped/added row shifts the labels out of step with the '
        f'samples and silently redates the series.'
    )

    source = next(iter(ratio_sources))
    assignment = re.search(rf'\b(?:const|let|var)\s+{re.escape(source)}\s*=\s*([^;]+);', workflow_body)
    assert assignment is not None, (
        f'could not find where `{source}` (the row list feeding the esc-per-done '
        f'chart) is assigned in WorkflowPanel.'
    )
    assert '.filter(' not in assignment.group(1), (
        f'`{source}` is assigned from a filtered list: `{assignment.group(1).strip()}`. '
        f'The esc-per-done chart must be fed the windowed rows themselves, so a '
        f'day with no measurement keeps its slot and renders as a gap.'
    )


# ---------------------------------------------------------------------------
# amendment: pin charts.jsx's chart padding to the value RegimeMarkers assumes
# ---------------------------------------------------------------------------


def test_charts_jsx_padding_matches_analytics_marker_overlay(charts_jsx_body: str) -> None:
    """Pin charts.jsx's LineChart/StackedAreaChart padL/padR to the values
    tab_escalation_analytics.jsx's RegimeMarkers overlay hardcodes as
    `_CHART_PAD_L`/`_CHART_PAD_R` (38/12).

    charts.jsx is intentionally NOT modified by the escalation-analytics tab
    (see task 2659 design decisions: markers/ECDF are composed locally from
    existing primitives instead of extending the shared chart primitives, to
    avoid a broader-scope edit and contention with sibling tasks also
    building on charts.jsx). RegimeMarkers therefore duplicates padL/padR as
    private constants rather than importing them from charts.jsx, so that
    duplication is invisible to any test scoped to tab_escalation_analytics.jsx
    alone — if charts.jsx's padding ever changes, the overlay would silently
    drift out of alignment with the chart it decorates. This test is the
    tripwire: it fails loudly the moment the two constants diverge, without
    requiring charts.jsx to export anything.
    """
    for fn_name in ('LineChart', 'StackedAreaChart'):
        fn_body = extract_function_body(charts_jsx_body, fn_name)
        assert re.search(r'padL\s*=\s*38\b', fn_body), (
            f'charts.jsx {fn_name} no longer declares padL = 38 — '
            'tab_escalation_analytics.jsx hardcodes _CHART_PAD_L = 38 for its '
            'RegimeMarkers overlay (see that file) and must be updated to match, '
            'or charts.jsx should export the constant instead.'
        )
        assert re.search(r'padR\s*=\s*12\b', fn_body), (
            f'charts.jsx {fn_name} no longer declares padR = 12 — '
            'tab_escalation_analytics.jsx hardcodes _CHART_PAD_R = 12 for its '
            'RegimeMarkers overlay (see that file) and must be updated to match, '
            'or charts.jsx should export the constant instead.'
        )


# ---------------------------------------------------------------------------
# task 5596 (PRD leaf eta): the analytics tab reads the corpus through
# escalation_views.js and states the age of the walk it shows
# ---------------------------------------------------------------------------


def test_tab_analytics_reads_escalation_views_at_module_scope(tab_analytics_jsx_body: str) -> None:
    """DF_ESCALATION_VIEWS is destructured at module scope, with no fallback."""
    code = strip_js_comments(tab_analytics_jsx_body)
    m = re.search(r'const\s*\{([^}]*)\}\s*=\s*window\.DF_ESCALATION_VIEWS\s*;', code)
    assert m is not None, (
        'tab_escalation_analytics.jsx does not destructure `window.DF_ESCALATION_VIEWS` '
        'at module scope (`const { … } = window.DF_ESCALATION_VIEWS;`, no `|| {}`).'
    )
    names = {n.split(':')[-1].strip() for n in m.group(1).split(',') if n.strip()}
    for name in ('resolutionSegments', 'corpusAgeCaption', 'openInHistoryOver'):
        assert name in names, (
            f'tab_escalation_analytics.jsx does not take `{name}` from DF_ESCALATION_VIEWS.'
        )


def test_tab_analytics_states_the_corpus_age(tab_analytics_jsx_body: str) -> None:
    """The tab renders how old the corpus walk behind every panel is."""
    tab_body = extract_function_body(strip_js_comments(tab_analytics_jsx_body), 'EscalationAnalyticsTab')
    assert re.search(r'\{\s*corpusAgeCaption\(', tab_body), (
        'EscalationAnalyticsTab renders no `{corpusAgeCaption(…)}` — the payload is '
        'derived from a cached walk and must say when that walk was.'
    )
