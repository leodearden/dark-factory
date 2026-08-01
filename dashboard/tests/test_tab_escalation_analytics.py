"""Wiring tests for the escalation-analytics endpoint + data.js registration.

Follows the source-assertion idiom established in test_tab_escalations.py:
static text checks against data.js (no JS runtime in this project) plus a
TestClient-driven route test against the real FastAPI app.
"""

from __future__ import annotations

import html.parser
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Module-scoped fixtures (static data.js content) — mirrors test_tab_escalations.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def data_js_body(_client):
    return _client.get('/static/redux/data.js').text


@pytest.fixture(scope='module')
def tab_analytics_jsx_body(_client):
    return _client.get('/static/redux/tab_escalation_analytics.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


@pytest.fixture(scope='module')
def shell_jsx_body(_client):
    return _client.get('/static/redux/shell.jsx').text


@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text


@pytest.fixture(scope='module')
def charts_jsx_body(_client):
    return _client.get('/static/redux/charts.jsx').text


# ---------------------------------------------------------------------------
# Helper: extract a named seed block from window.DF_DATA (brace-aware).
# Copied from test_tab_escalations.py — see that module for the full rationale
# (brace-depth walk; does not skip braces inside JS string literals).
# ---------------------------------------------------------------------------


def _extract_df_data_block(src: str, key: str) -> str:
    """Return the body of the ``<key>: { ... }`` seed object, braces included."""
    m = re.search(rf'{re.escape(key)}\s*:\s*\{{', src)
    if m is None:
        return ''
    start = m.end() - 1  # index of the opening `{`
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return ''


# ---------------------------------------------------------------------------
# Helper: extract a named JS/JSX function body (brace-aware).
# Copied from test_tab_escalations.py — scopes token-presence checks to a
# specific function body rather than searching the entire file (which would
# give false confidence when a token appears in an unrelated context).
# ---------------------------------------------------------------------------


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the body block of a ``function <fn_name>(`` declaration, braces included.

    Uses the same brace-depth walk as ``_extract_df_data_block``.  Only matches
    named ``function`` declarations — not arrow functions or class methods.
    Returns the empty string if the function is not found.

    Paren-depth walks past the parameter list before looking for the body's
    opening ``{`` — a destructured parameter (``function Foo({ a, b }) {``)
    contains its own ``{``/``}`` pair *inside* the parameter list, so naively
    taking the first ``{`` after the opening ``(`` would return just the
    destructuring pattern (e.g. ``{ a, b }``) instead of the function body.
    """
    m = re.search(rf'\bfunction\s+{re.escape(fn_name)}\s*\(', src)
    if m is None:
        return ''
    paren_depth = 1
    i = m.end()
    while i < len(src) and paren_depth > 0:
        if src[i] == '(':
            paren_depth += 1
        elif src[i] == ')':
            paren_depth -= 1
        i += 1
    if paren_depth != 0:
        return ''
    start = src.find('{', i)
    if start == -1:
        return ''
    depth = 0
    for j in range(start, len(src)):
        c = src[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    return ''


# ---------------------------------------------------------------------------
# Load-order helpers (copied from test_tab_escalations.py / test_index_html.py)
# ---------------------------------------------------------------------------


class _ScriptTagCollector(html.parser.HTMLParser):
    """Collects the attribute dicts for every <script> start-tag encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.script_attrs: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == 'script':
            self.script_attrs.append(dict(attrs))


def _find_script_position(
    body: str, src_prefix: str
) -> tuple[int, dict[str, str | None]] | None:
    """Return ``(index, attrs)`` for the first <script> tag whose ``src``
    starts with ``src_prefix``, or ``None`` if no such tag exists.
    """
    collector = _ScriptTagCollector()
    collector.feed(body)
    for i, attrs in enumerate(collector.script_attrs):
        if (attrs.get('src') or '').startswith(src_prefix):
            return i, attrs
    return None


def _assert_script_loads_before(
    body: str,
    before_src_prefix: str,
    after_src_prefix: str,
    before_label: str,
    after_label: str,
    consumer_note: str = '',
) -> None:
    """Assert that the script for ``before_src_prefix`` loads BEFORE the
    script for ``after_src_prefix`` in ``body``.  Combines a
    defer/async/type=module false-pass guard with the document-order
    position comparison.
    """
    before_result = _find_script_position(body, before_src_prefix)
    assert before_result is not None, (
        f'No <script src="{before_src_prefix}..."> tag found in index.html. '
        f'{consumer_note}'
    )
    before_pos, before_attrs = before_result
    before_src = before_attrs.get('src')

    after_result = _find_script_position(body, after_src_prefix)
    assert after_result is not None, (
        f'<script src="{after_src_prefix}..."> not found in index.html — '
        f'cannot verify load-order invariant for {before_label}.'
    )
    after_pos, after_attrs = after_result

    # Both tags must be classic synchronous scripts — otherwise document order
    # diverges from execution order and the position comparison below is moot.
    for _label, _attrs in [
        (before_label, before_attrs),
        (after_label, after_attrs),
    ]:
        assert 'defer' not in _attrs, (
            f'{_label} has a defer attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert 'async' not in _attrs, (
            f'{_label} has an async attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert (_attrs.get('type') or '').lower() != 'module', (
            f'{_label} has type="module"; ES modules are deferred by default, '
            f'so document order no longer implies execution order.'
        )

    assert before_pos < after_pos, (
        f'{before_label} (position {before_pos}, src={before_src!r}) must load '
        f'BEFORE {after_label} (position {after_pos}). '
        f'{consumer_note}'
    )


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
    seed_block = _extract_df_data_block(data_js_body, 'ESCALATION_ANALYTICS')
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
# step-19: GET /api/v2/dashboard/escalation-analytics — real config + tmp archive.
# Uses the module-level `client` fixture from conftest.py (function-scoped —
# a fresh TestClient/lifespan per test) so each test can point app.state.config
# at its own tmp project root, mirroring test_api_curator.py's _override_client
# idiom without needing a local helper.
# ---------------------------------------------------------------------------


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


class TestEscalationAnalyticsRoute:
    """GET /api/v2/dashboard/escalation-analytics against the real app + a tmp archive."""

    def test_returns_wrapped_contract_keys(self, client, tmp_path):
        """200 + ESCALATION_ANALYTICS wrapping the four Seam-2 contract keys."""
        from dashboard.app import _analytics_cache_clear

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
        _analytics_cache_clear()

        resp = client.get('/api/v2/dashboard/escalation-analytics')

        assert resp.status_code == 200
        body = resp.json()
        assert 'ESCALATION_ANALYTICS' in body, (
            'route response missing the ESCALATION_ANALYTICS wrapper key — data.js '
            "applyKey('ESCALATION_ANALYTICS', body['ESCALATION_ANALYTICS']) needs it, "
            'matching the ESCALATIONS/CURATOR_STATE/SCHEDULER envelope precedent.'
        )
        analytics = body['ESCALATION_ANALYTICS']
        assert set(analytics) == {
            'generated_at', 'parse_failures', 'regime_markers', 'per_project',
            # Whether the scan reached every configured archive at all — the
            # route's cacheability signal, and the only field that can tell an
            # absent archive from an empty one.
            'archives_present',
        }
        assert isinstance(analytics['generated_at'], str) and analytics['generated_at']
        assert analytics['archives_present'] is True
        assert analytics['parse_failures'] == 0
        assert isinstance(analytics['regime_markers'], list)
        assert len(analytics['per_project']) == 1
        assert analytics['per_project'][0]['project'] == tmp_path.name

    def test_primary_root_ordered_first(self, client, tmp_path):
        """per_project lists the primary project_root before known_project_roots."""
        from dashboard.app import _analytics_cache_clear

        primary = tmp_path / 'primary'
        secondary = tmp_path / 'secondary'
        (primary / 'data' / 'escalations').mkdir(parents=True)
        (secondary / 'data' / 'escalations').mkdir(parents=True)

        client.app.state.config = _make_config(primary, known_project_roots=[secondary])
        _analytics_cache_clear()

        resp = client.get('/api/v2/dashboard/escalation-analytics')

        assert resp.status_code == 200
        per_project = resp.json()['ESCALATION_ANALYTICS']['per_project']
        assert [p['project'] for p in per_project] == ['primary', 'secondary']

    def test_malformed_regime_markers_never_500s(self, client, tmp_path, monkeypatch):
        """Row 9 (endpoint level): a broken regime-markers file degrades loudly.

        Mirrors load_regime_markers' own fail-soft contract (step-1) but pins it
        through the real route: a corrupt markers file must never surface as a
        500 — only a counted parse_failures increment.
        """
        import dashboard.data.escalation_analytics as escalation_analytics_module
        from dashboard.app import _analytics_cache_clear

        (tmp_path / 'data' / 'escalations').mkdir(parents=True)
        client.app.state.config = _make_config(tmp_path)
        _analytics_cache_clear()

        resp1 = client.get('/api/v2/dashboard/escalation-analytics')
        assert resp1.status_code == 200
        assert resp1.json()['ESCALATION_ANALYTICS']['parse_failures'] == 0

        bad_markers = tmp_path / 'bad-regime-markers.yaml'
        bad_markers.write_text('date: [unclosed')
        monkeypatch.setattr(
            escalation_analytics_module, '_DEFAULT_REGIME_MARKERS_PATH', bad_markers,
        )
        _analytics_cache_clear()

        resp2 = client.get('/api/v2/dashboard/escalation-analytics')
        assert resp2.status_code == 200
        assert resp2.json()['ESCALATION_ANALYTICS']['parse_failures'] >= 1


def _analytics_payload(**overrides):
    """A builder-shaped analytics payload, overridable one key at a time.

    Shaped exactly as ``build_escalation_analytics`` returns it, so a stub
    standing in for the builder cannot hand the route something the real
    builder never would.
    """
    payload = {
        'generated_at': '2026-07-16T18:00:00+00:00',
        'parse_failures': 0,
        'regime_markers': [],
        'per_project': [],
        'archives_present': True,
    }
    payload.update(overrides)
    return payload


class TestEscalationAnalyticsCacheability:
    """A scan that never reached the archive must not be pinned for the TTL.

    Same rule the memory-evals route follows, applied here because the two
    routes are written as one idiom and a fix to only one of them would leave
    them silently divergent.  The signal differs — memory-evals already
    carried ``root_present``, analytics needed ``archives_present`` added —
    but the reasoning is identical: an absent archive is O(1) to re-check
    (one ``is_dir`` stat per project), so re-checking every poll costs
    nothing, while caching it keeps the tab reporting an empty archive for a
    full TTL window after the volume mounts.

    Accepted, LOUD trade-off: a multi-project config where one root
    legitimately has no ``data/escalations`` dir makes this payload
    permanently uncacheable.  ``TTLCache``'s per-key lock serializes cold
    callers, so that degrades to one walk at a time rather than N in
    parallel — and ``archives_present: false`` sits in the payload, so the
    misconfiguration is visible and fixable instead of being a silent perf
    cliff nobody can see.
    """

    def test_a_reached_archive_is_served_from_cache(self, client, monkeypatch, tmp_path):
        """Gating the cache must not disable it — the ordinary payload still caches."""
        import dashboard.app as app_module
        from dashboard.app import _analytics_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _analytics_cache_clear()
        # SYNC, not AsyncMock: the route calls the builder through
        # asyncio.to_thread, which hands it a plain callable.
        build = MagicMock(return_value=_analytics_payload())
        monkeypatch.setattr(app_module, 'build_escalation_analytics', build)

        assert client.get('/api/v2/dashboard/escalation-analytics').status_code == 200
        assert client.get('/api/v2/dashboard/escalation-analytics').status_code == 200

        assert build.call_count == 1, f'expected 1 build call, got {build.call_count}'

    def test_an_unreached_archive_is_not_pinned_for_the_ttl(self, client, monkeypatch, tmp_path):
        """``archives_present: False`` is re-checked every poll."""
        import dashboard.app as app_module
        from dashboard.app import _analytics_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _analytics_cache_clear()
        build = MagicMock(return_value=_analytics_payload(archives_present=False))
        monkeypatch.setattr(app_module, 'build_escalation_analytics', build)

        r1 = client.get('/api/v2/dashboard/escalation-analytics')
        r2 = client.get('/api/v2/dashboard/escalation-analytics')

        assert r1.status_code == r2.status_code == 200
        assert r2.json()['ESCALATION_ANALYTICS']['archives_present'] is False
        assert build.call_count == 2, f'expected 2 build calls, got {build.call_count}'

    def test_parse_failures_alone_never_defeats_the_cache(self, client, monkeypatch, tmp_path):
        """The whole reason the predicate is not keyed on ``parse_failures``.

        A corrupt record is permanent, so gating on it would re-run a ~10k
        record walk on every poll, forever, to rediscover the same bad file.
        The archive WAS reached; that is the question the cache asks.
        """
        import dashboard.app as app_module
        from dashboard.app import _analytics_cache_clear

        client.app.state.config = _make_config(tmp_path)
        _analytics_cache_clear()
        build = MagicMock(return_value=_analytics_payload(parse_failures=7))
        monkeypatch.setattr(app_module, 'build_escalation_analytics', build)

        assert client.get('/api/v2/dashboard/escalation-analytics').status_code == 200
        r2 = client.get('/api/v2/dashboard/escalation-analytics')

        assert r2.json()['ESCALATION_ANALYTICS']['parse_failures'] == 7
        assert build.call_count == 1, f'expected 1 build call, got {build.call_count}'


# ---------------------------------------------------------------------------
# task 2659 (delta) — tab_escalation_analytics.jsx UI wiring
# ---------------------------------------------------------------------------
#
# The fixtures/helpers above (tab_analytics_jsx_body, app_jsx_body,
# shell_jsx_body, index_html_body, _extract_function_body,
# _ScriptTagCollector, _find_script_position, _assert_script_loads_before)
# were scaffolded in prereq-1; the tests below consume them.


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
    via `_extract_function_body`, and (c)'s export check requires the actual
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
    tab_body = _extract_function_body(body, 'EscalationAnalyticsTab')
    assert tab_body, (
        'Could not locate the `function EscalationAnalyticsTab(` body in '
        'tab_escalation_analytics.jsx.'
    )
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
    synchronous script (no defer/async/type=module); and all
    /static/redux/*?v= cache-busters must share a single version >= 30.

    Checks:
    (a) tab_escalation_analytics.jsx script tag exists.
    (b) Loads after data.js.
    (c) Loads after shell.jsx.
    (d) Loads after tabs.jsx.
    (e) Loads before app.jsx.
    (f) Not deferred/async/module.
    (g) All /static/redux/ v= cache-busters share one version >= 30.
    """
    _TAB_ANALYTICS_PREFIX = '/static/redux/tab_escalation_analytics.jsx'

    # (a) tab_escalation_analytics.jsx script tag must exist
    result = _find_script_position(index_html_body, _TAB_ANALYTICS_PREFIX)
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
    _assert_script_loads_before(
        index_html_body,
        '/static/redux/data.js',
        _TAB_ANALYTICS_PREFIX,
        'data.js',
        'tab_escalation_analytics.jsx',
        'data.js must load before tab_escalation_analytics.jsx so window.DF_DATA is seeded.',
    )

    # (c) Loads after shell.jsx
    _assert_script_loads_before(
        index_html_body,
        '/static/redux/shell.jsx',
        _TAB_ANALYTICS_PREFIX,
        'shell.jsx',
        'tab_escalation_analytics.jsx',
        'shell.jsx must load before tab_escalation_analytics.jsx so window.DF_SHELL is available.',
    )

    # (d) Loads after tabs.jsx
    _assert_script_loads_before(
        index_html_body,
        '/static/redux/tabs.jsx',
        _TAB_ANALYTICS_PREFIX,
        'tabs.jsx',
        'tab_escalation_analytics.jsx',
        'tabs.jsx must load before tab_escalation_analytics.jsx so window.DF_TABS is available to mutate.',
    )

    # (e) Loads before app.jsx
    _assert_script_loads_before(
        index_html_body,
        _TAB_ANALYTICS_PREFIX,
        '/static/redux/app.jsx',
        'tab_escalation_analytics.jsx',
        'app.jsx',
        'tab_escalation_analytics.jsx must load before app.jsx so EscalationAnalyticsTab is set on window.DF_TABS.',
    )

    # (g) All /static/redux/ v= cache-busters share one version >= 30
    versions = set(re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body))
    assert len(versions) == 1, (
        f'index.html has mixed /static/redux/?v= cache-buster versions: {sorted(versions)} — '
        'bump all of them uniformly to the same value.'
    )
    v = int(next(iter(versions)))
    assert v >= 30, (
        f'index.html cache-buster version is {v}, expected >= 30.'
    )


# ---------------------------------------------------------------------------
# step-5 test: app.jsx wires the esc-analytics tab
# ---------------------------------------------------------------------------


def test_app_jsx_wires_analytics_tab(app_jsx_body: str) -> None:
    """app.jsx must destructure EscalationAnalyticsTab from window.DF_TABS,
    add an 'esc-analytics' tab entry, handle it in renderTab, and configure
    toolbarConfig with showWindow: false (the tab owns its own client-side
    window toggle rather than the global server-side window chip).

    Asserts four structural wiring contracts:
    (a) EscalationAnalyticsTab is destructured from window.DF_TABS.
    (b) tabs[] contains an entry with id 'esc-analytics' and label 'Analytics'.
    (c) renderTab switch has a `case 'esc-analytics':` branch returning
        <EscalationAnalyticsTab projectFilter={projects} />.
    (d) toolbarConfig has an 'esc-analytics' entry with showWindow: false.
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
    # (d) toolbarConfig esc-analytics entry with showWindow: false
    parts = app_jsx_body.split('toolbarConfig')
    assert len(parts) > 1 and "'esc-analytics'" in parts[1], (
        "app.jsx toolbarConfig does not have an 'esc-analytics' entry — add "
        "`'esc-analytics': { showWindow: false, showAgents: false, search: false }` "
        'to toolbarConfig.'
    )
    esc_analytics_entry = parts[1].split("'esc-analytics'", 1)[1][:200]
    assert re.search(r'showWindow\s*:\s*false', esc_analytics_entry), (
        "app.jsx toolbarConfig 'esc-analytics' entry does not set `showWindow: false` — "
        'the tab owns its own client-side window toggle, not the global chip.'
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
    tab_body = _extract_function_body(body, 'EscalationAnalyticsTab')
    assert tab_body, (
        'Could not locate the `function EscalationAnalyticsTab(` body in '
        'tab_escalation_analytics.jsx.'
    )
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
    cutoff_body = _extract_function_body(body, 'windowCutoffDate')
    assert cutoff_body, 'Could not locate the windowCutoffDate( function body.'
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
    (b) A benign-rate table over `origin.sources` referencing `benign_rate`,
        `stamped_share` (stamped-vs-inferred split), and `actionable`
        (benign/actionable segmented bar).
    (c) A `predictably_benign` badge.
    (d) A per-source `Sparkline` fed by `daily_spark`.
    (e) Rows sorted by benign COUNT — a `.sort(` referencing `.benign`.
    """
    body = tab_analytics_jsx_body

    assert 'function OriginPanel(' in body, (
        'tab_escalation_analytics.jsx does not define `function OriginPanel(` — '
        'add the Origin panel component.'
    )
    origin_body = _extract_function_body(body, 'OriginPanel')
    assert origin_body, 'Could not locate the OriginPanel( function body.'

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
    assert 'actionable' in origin_body, (
        'OriginPanel does not reference `actionable` — render the benign/actionable '
        'segmented bar.'
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
    lifespan_body = _extract_function_body(body, 'LifespanPanel')
    assert lifespan_body, 'Could not locate the LifespanPanel( function body.'

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
    workflow_body = _extract_function_body(body, 'WorkflowPanel')
    assert workflow_body, 'Could not locate the WorkflowPanel( function body.'

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
        fn_body = _extract_function_body(charts_jsx_body, fn_name)
        assert fn_body, f'Could not locate the {fn_name}( function body in charts.jsx.'
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
