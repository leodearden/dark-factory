"""Static source-contract tests for the Burndown tab wiring.

Tests fetch JSX source via TestClient and assert structural contracts as text.
Follows the idiom in test_tab_curator.py / test_tab_overview.py.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def tabs_jsx_body(_client):
    return _client.get('/static/redux/tabs.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


@pytest.fixture(scope='module')
def charts_jsx_body(_client):
    return _client.get('/static/redux/charts.jsx').text


# ---------------------------------------------------------------------------
# Chart labels/values pairing probe
# ---------------------------------------------------------------------------
#
# The defect class this guards against is "the labels prop and the values come
# from DIFFERENT series objects".  A chart's x-axis row and its y-value arrays
# are co-indexed by construction only when they are read off the same block, so
# pairing one block's labels with another's values both overruns the values
# (undefined past their length) and index-shifts every earlier point onto the
# wrong label.
#
# Matching is by EXPRESSION CONTENT, never by line number: these shift whenever
# an unrelated tab earlier in the file is edited.
#
# The scan is INVERTED — it starts from every `labels={...}` prop in the file and
# derives the component that prop sits on, rather than sweeping a hardcoded list
# of chart tags.  So a chart added later with a component this file does not use
# today is covered automatically, and a labels prop whose enclosing tag is not a
# window.DF_CHARTS component fails the vacuity check rather than silently
# dropping out of the sweep.

_LABELS_PROP_RE = re.compile(r'\blabels=\{')
_LABELS_RE = re.compile(r'labels=\{([^{}]+)\}')
_VALUES_RE = re.compile(r'values[:=]\s*\{?([^,}\n]+)')
_TAG_START_RE = re.compile(r'<([A-Za-z_$][\w$]*)')
# Any tag start or end inside an element's attribute list means the element is
# not flat self-closing, so a regex chunk cannot be attributed to it safely.
_TAG_BOUNDARY_RE = re.compile(r'</?[A-Za-z_$]')
# A trailing call suffix such as `.map(String)` is presentation, not series identity.
_CALL_SUFFIX_RE = re.compile(r'\.\w+\([^()]*\)$')
_DF_CHARTS_DESTRUCTURE_RE = re.compile(r'const\s*\{([^{}]*)\}\s*=\s*window\.DF_CHARTS')
_DF_CHARTS_EXPORT_RE = re.compile(r'window\.DF_CHARTS\s*=\s*\{([^{}]*)\}')


def _series_root(expr):
    """Normalize a chart expression to the series object it reads from.

    `b.labels` -> `b`, `pb.done` -> `pb`, `p.hist_outer.values` -> `p.hist_outer`,
    `d.depth.labels.map(String)` -> `d.depth`.
    """
    expr = expr.strip()
    prev = None
    while prev != expr:
        prev = expr
        expr = _CALL_SUFFIX_RE.sub('', expr).strip()
    return expr.rsplit('.', 1)[0] if '.' in expr else expr


def _chart_component_aliases(src):
    """Map local JSX alias -> canonical name for everything pulled off DF_CHARTS.

    Parsed out of tabs.jsx's own `const { StackedAreaChart: SA, ... } =
    window.DF_CHARTS` line, so the known-component list is never a hardcoded
    second copy that can drift from what the file actually renders.
    """
    m = _DF_CHARTS_DESTRUCTURE_RE.search(src)
    if not m:
        return {}
    aliases = {}
    for part in m.group(1).split(','):
        part = part.strip()
        if not part:
            continue
        canonical, _, alias = part.partition(':')
        canonical = canonical.strip()
        aliases[alias.strip() or canonical] = canonical
    return aliases


def _df_charts_exports(src):
    """Names exported by charts.jsx's `window.DF_CHARTS = { ... }` line."""
    m = _DF_CHARTS_EXPORT_RE.search(src)
    if not m:
        return set()
    return {
        part.split(':', 1)[0].strip()
        for part in m.group(1).split(',')
        if part.strip()
    }


def _element_at(src, pos):
    """Return (tag, start) of the JSX element whose attribute list contains `pos`."""
    cursor = pos
    while True:
        lt = src.rfind('<', 0, cursor)
        if lt == -1:
            raise AssertionError(
                f'no enclosing JSX tag found for the prop at offset {pos} '
                f'(line {src.count(chr(10), 0, pos) + 1})'
            )
        m = _TAG_START_RE.match(src, lt)
        if m:
            return m.group(1), lt
        cursor = lt


def _chart_sites(src):
    """Return one record per chart element — every element with a `labels` prop.

    Each record is a dict with `tag`, `labels_expr`, `labels_root`,
    `values_exprs` and `values_roots`.

    An element runs from its tag start to the first `/>` after it.  That is only
    a sound bound for a FLAT SELF-CLOSING element, so it is asserted rather than
    assumed: if a chart is ever rewritten as `<LC ...>...</LC>` (say to nest a
    legend), or a `<` otherwise appears in its attribute list, the chunk would
    run past the element and silently attribute a DIFFERENT element's props to
    it — a spurious failure, or worse a spurious pass.  Failing loudly on an
    element this probe cannot parse is the safe direction.
    """
    sites = []
    for lm in _LABELS_PROP_RE.finditer(src):
        tag, start = _element_at(src, lm.start())
        end = src.find('/>', start)
        boundary = _TAG_BOUNDARY_RE.search(src, start + 1)
        limit = boundary.start() if boundary else len(src)
        assert end != -1 and end < limit, (
            f'<{tag}> element at line {src.count(chr(10), 0, start) + 1} is not a '
            'flat self-closing element, so this probe cannot attribute its props '
            '(a following element\'s labels/values would be read as its own). '
            'Teach _chart_sites to parse it rather than deleting the assertion.'
        )
        chunk = src[start:end + 2]
        labels_m = _LABELS_RE.search(chunk)
        labels_expr = labels_m.group(1).strip() if labels_m else None
        values_exprs = [v.strip() for v in _VALUES_RE.findall(chunk)]
        sites.append({
            'tag': tag,
            'labels_expr': labels_expr,
            'labels_root': _series_root(labels_expr) if labels_expr else None,
            'values_exprs': values_exprs,
            'values_roots': [_series_root(v) for v in values_exprs],
        })
    return sites


class TestChartLabelValuePairing:
    def test_per_project_status_mix_uses_own_label_row(self, tabs_jsx_body):
        """The per-project status-mix chart must use that project's OWN label row.

        `b.labels` (DF.BURNDOWN) is the sorted UNION of every project's snapshot
        timestamps (redux_api.py:878), while `pb.*` are one project's own series.
        Pairing them overruns the values and index-shifts them.
        """
        sites = [
            s for s in _chart_sites(tabs_jsx_body)
            if s['tag'] == 'SA' and s['values_roots'] and set(s['values_roots']) == {'pb'}
        ]
        assert len(sites) == 1, (
            f'expected exactly one per-project <SA> site, found {len(sites)}'
        )
        site = sites[0]
        assert site['labels_expr'] != 'b.labels', (
            'per-project status-mix chart pairs the aggregate union label row '
            '`b.labels` with per-project values — must be `pb.labels`'
        )
        assert site['labels_root'] == 'pb', (
            f"expected labels root 'pb', got {site['labels_root']!r} "
            f"(labels={site['labels_expr']!r})"
        )

    def test_aggregate_status_mix_uses_aggregate_label_row(self, tabs_jsx_body):
        """The aggregate status-mix chart must keep the union label row.

        Present so the per-project fix cannot be "achieved" by mutating this
        site instead — the aggregate series ARE densified onto `b.labels`.
        """
        sites = [
            s for s in _chart_sites(tabs_jsx_body)
            if s['tag'] == 'SA' and s['values_roots'] and set(s['values_roots']) == {'b'}
        ]
        assert len(sites) == 1, (
            f'expected exactly one aggregate <SA> site, found {len(sites)}'
        )
        assert sites[0]['labels_root'] == 'b'

    def test_every_chart_pairs_labels_and_values_from_one_series(self, tabs_jsx_body):
        """EVERY chart must read its labels and its values off the same object.

        This pins the defect CLASS, so the same mistake authored at a chart
        added later to this file also fails — including one rendered with a
        chart component this file does not use today, because the sweep starts
        from labels props rather than from a fixed set of tags.
        """
        mismatched = [
            (s['tag'], s['labels_expr'], s['values_exprs'])
            for s in _chart_sites(tabs_jsx_body)
            if s['labels_root'] is not None
            and any(r != s['labels_root'] for r in s['values_roots'])
        ]
        assert not mismatched, (
            'chart sites pair labels and values from different series objects: '
            f'{mismatched}'
        )

    def test_chart_pairing_probe_is_not_vacuous(self, tabs_jsx_body):
        """The probe must actually match the charts it claims to guard.

        A structural probe that silently matches nothing reports green forever,
        which is strictly worse than no test at all.
        """
        sites = _chart_sites(tabs_jsx_body)
        assert len(sites) >= 8, f'expected >=8 chart sites, found {len(sites)}'
        # Only <SA> is asserted present — it is the chart type this module is
        # about.  Pinning the exact tag SET would turn an unrelated, legitimate
        # edit (dropping the last <BC> histogram, say) into a failure here that
        # reads as a probe malfunction.
        assert any(s['tag'] == 'SA' for s in sites), (
            f'no <SA> site found — tags seen: {sorted({s["tag"] for s in sites})}'
        )
        for s in sites:
            assert s['labels_expr'], f'no labels expression parsed for <{s["tag"]}> site'
            assert s['values_exprs'], f'no values expressions parsed for <{s["tag"]}> site'

    def test_every_labels_prop_sits_on_a_chart_component(
        self, tabs_jsx_body, charts_jsx_body
    ):
        """Every `labels={...}` prop must sit on a component from window.DF_CHARTS.

        The pairing sweep above is only a class guarantee if the tag it derives
        for each labels prop is really the chart component rendering it.  This
        checks that derivation against tabs.jsx's own DF_CHARTS destructure,
        cross-checked against what charts.jsx actually exports — so a labels
        prop attributed to a `<div>` (the backwards walk mis-parsed) or to a
        component that is not a chart fails here instead of passing quietly.
        """
        aliases = _chart_component_aliases(tabs_jsx_body)
        assert aliases, (
            'could not parse the `const { ... } = window.DF_CHARTS` destructure '
            'in tabs.jsx — the probe has no known-component list to check against'
        )
        exported = _df_charts_exports(charts_jsx_body)
        assert exported, 'could not parse `window.DF_CHARTS = { ... }` in charts.jsx'

        unknown = sorted({
            s['tag'] for s in _chart_sites(tabs_jsx_body)
            if aliases.get(s['tag']) not in exported
        })
        assert not unknown, (
            f'labels props found on non-chart components {unknown}; DF_CHARTS '
            f'aliases in tabs.jsx: {sorted(aliases)}'
        )


# ---------------------------------------------------------------------------
# app.jsx — passes displayWindow (not `window`) into BurnTab
# ---------------------------------------------------------------------------

class TestAppDisplayWindowProp:
    def test_burntab_receives_display_window(self, app_jsx_body):
        """app.jsx must forward the active display-window key into BurnTab."""
        assert re.search(r'<BurnTab[^>]*displayWindow=\{win\}', app_jsx_body)

    def test_burntab_prop_not_named_window(self, app_jsx_body):
        """The prop must not shadow the browser global `window` inside BurnTab."""
        assert not re.search(r'<BurnTab[^>]*\bwindow=', app_jsx_body)


# ---------------------------------------------------------------------------
# tabs.jsx BurnTab — signature and smoothing state
# ---------------------------------------------------------------------------

class TestBurnTabSignature:
    def test_destructures_display_window(self, tabs_jsx_body):
        """BurnTab must accept displayWindow in its parameter destructure."""
        assert re.search(r'function BurnTab\s*\(\s*\{[^}]*displayWindow', tabs_jsx_body)

    def test_does_not_destructure_window_param(self, tabs_jsx_body):
        """BurnTab must NOT have a param named `window` (would shadow the global)."""
        assert not re.search(r'function BurnTab\s*\(\s*\{[^}]*\bwindow\b', tabs_jsx_body)


class TestBurnTabSmoothingChip:
    def test_default_smoothing_for_window_referenced(self, tabs_jsx_body):
        """BurnTab must call defaultSmoothingForWindow to derive the default."""
        assert 'defaultSmoothingForWindow' in tabs_jsx_body

    def test_smoothing_options_referenced(self, tabs_jsx_body):
        """BurnTab must reference SMOOTHING_OPTIONS for the chip's option list."""
        assert 'SMOOTHING_OPTIONS' in tabs_jsx_body

    def test_chip_group_rendered_for_smoothing(self, tabs_jsx_body):
        """A ChipGroup (or equivalent) must appear in BurnTab for smoothing."""
        # Accept either ChipGroup or Segmented — both are reusable controls.
        assert 'ChipGroup' in tabs_jsx_body or re.search(
            r'Segmented[^;]*SMOOTHING_OPTIONS', tabs_jsx_body
        )

    def test_charts_destructure_includes_smoothing_exports(self, tabs_jsx_body):
        """tabs.jsx must destructure the smoothing helpers from window.DF_CHARTS."""
        assert 'deriveVelocitySeries' in tabs_jsx_body
        assert 'defaultSmoothingForWindow' in tabs_jsx_body
        assert 'SMOOTHING_OPTIONS' in tabs_jsx_body


# ---------------------------------------------------------------------------
# velocity sparks — added in step-7 (these fail until step-8 GREEN)
# ---------------------------------------------------------------------------

class TestVelocitySparkWiring:
    def test_net_velocity_tile_uses_derive(self, tabs_jsx_body):
        """Net velocity StatTile spark must use deriveVelocitySeries, not raw b.done.

        The regex ties the tile's label attribute to its spark attribute within the
        same element, so the test fails if the Net velocity tile reverts to spark={b.done}.
        """
        assert re.search(
            r'label=["\']Net velocity["\'].*?spark=\{deriveVelocitySeries\(',
            tabs_jsx_body,
            re.DOTALL,
        )

    def test_completion_trend_spark_uses_derive(self, tabs_jsx_body):
        """Per-project Velocity panel 'Completion trend' must use deriveVelocitySeries."""
        assert re.search(r'Completion trend', tabs_jsx_body)
        assert re.search(r'deriveVelocitySeries\(pb\.done', tabs_jsx_body)

    def test_backlog_trend_spark_uses_derive(self, tabs_jsx_body):
        """Per-project Velocity panel 'Backlog change rate' must use deriveVelocitySeries.

        The label was changed from 'Backlog trend' to 'Backlog change rate' to
        clarify that this spark shows the rate of change of backlog (derivative),
        not the backlog level over time.
        """
        assert re.search(r'Backlog change rate', tabs_jsx_body)
        assert re.search(r'deriveVelocitySeries\(pb\.pending', tabs_jsx_body)

    def test_status_mix_area_stays_cumulative(self, tabs_jsx_body):
        """Status-mix StackedArea charts must still use raw values (not derived)."""
        # The SA stacks for the aggregate view use b.done directly.
        assert re.search(r"values:\s*b\.done", tabs_jsx_body)

    def test_completed_window_tile_stays_cumulative(self, tabs_jsx_body):
        """'Completed (window)' tile spark must remain on raw b.done.

        Ties the label and spark attributes within the same element so that
        a regression swapping this tile to deriveVelocitySeries is caught.
        """
        assert re.search(
            r'label=["\']Completed \(window\)["\'].*?spark=\{b\.done\}',
            tabs_jsx_body,
            re.DOTALL,
        )
