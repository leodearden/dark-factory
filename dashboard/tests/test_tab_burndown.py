"""Static source-contract tests for the Burndown tab wiring.

Tests fetch JSX source via TestClient and assert structural contracts as text.
Follows the idiom in test_tab_curator.py / test_tab_overview.py.
"""

from __future__ import annotations

import re

from _dashboard_helpers import (
    DF_CHARTS_DESTRUCTURE_RE,
    DF_CHARTS_EXPORT_RE,
    destructure_bindings,
    extract_function_body,
    strip_js_comments,
    walk_balanced,
)

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
# A stacks prop built by a helper — `stacks={burndownStacks(pb, CP)}` — names
# its series object as the builder's FIRST argument rather than in inline
# `values:` entries.  Task 4361 extracted the two status-mix stack literals into
# burndown_bands.js (so that per-band wiring could be covered behaviourally by
# node --test instead of by JSX source-text greps), which would otherwise have
# left both <SA> sites with NO parsed values expressions — silently dropping
# them out of the pairing sweep below while every test here stayed green. That
# is coverage loss wearing the costume of success, so the probe learns the new
# spelling instead. The CONTRACT is unchanged: labels and values must still
# come off the same series object. Only where the values root is written moved.
_STACKS_BUILDER_RE = re.compile(r'stacks=\{\s*\w+\(\s*([A-Za-z_$][\w$.]*)\s*[,)]')
_TAG_START_RE = re.compile(r'<([A-Za-z_$][\w$]*)')
# Any tag start or end inside an element's attribute list means the element is
# not flat self-closing, so a regex chunk cannot be attributed to it safely.
_TAG_BOUNDARY_RE = re.compile(r'</?[A-Za-z_$]')
# A trailing call suffix such as `.map(String)` is presentation, not series identity.
_CALL_SUFFIX_RE = re.compile(r'\.\w+\([^()]*\)$')


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
    m = DF_CHARTS_DESTRUCTURE_RE.search(src)
    if not m:
        return {}
    return {local: canonical for canonical, local in destructure_bindings(m.group(1))}


def _df_charts_exports(src):
    """Names exported by charts.jsx's `window.DF_CHARTS = { ... }` line."""
    m = DF_CHARTS_EXPORT_RE.search(src)
    if not m:
        return set()
    return {canonical for canonical, _local in destructure_bindings(m.group(1))}


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
        # Plus any helper-built stacks prop, whose series object is the
        # builder's first argument. Additive, so a site mixing inline `values:`
        # entries with a builder call is still checked on both.
        values_exprs += [v.strip() for v in _STACKS_BUILDER_RE.findall(chunk)]
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
        """Net velocity StatTile history must use deriveVelocitySeries, not raw b.done.

        The regex ties the tile's label attribute to its history attribute within the
        same element, so the test fails if the Net velocity tile reverts to
        history={b.done}.  The prop was named `spark` until task 5588 renamed it
        `history` — the series is the tile's PAST, and `spark` named the drawing
        rather than the data beside a `datum` that carries the present value.
        """
        assert re.search(
            r'label=["\']Net velocity["\'].*?history=\{deriveVelocitySeries\(',
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
        """Status-mix StackedArea charts must still use raw values (not derived).

        Was `re.search(r"values:\\s*b\\.done", tabs_jsx_body)` — a source-text
        grep that task 4361 made unsatisfiable by extracting the stack literals
        into burndown_bands.js. Repointed rather than deleted, because the
        CONTRACT is still real: the status-mix bands plot the raw cumulative
        series, unlike the velocity charts this class otherwise covers, which
        legitimately derive theirs.

        The contract is now split across two homes, and both halves are
        covered:

        * WHICH series each band plots — that `done` carries `block.done` and
          not some neighbour or derivative — is asserted BY IDENTITY for all
          five bands in dashboard/tests/js/burndown_bands.test.mjs
          ("each band is sourced from its OWN same-named block field"). That is
          strictly stronger than the grep it replaces, which only proved the
          characters `values: b.done` appeared somewhere in the file.

        * WHICH object is handed to the builder at the call site is what
          remains checkable here, and is what this test asserts: the raw
          burndown block, not a derived series.
        """
        sites = [
            s for s in _chart_sites(tabs_jsx_body)
            if s['tag'] == 'SA' and set(s['values_roots']) == {'b'}
        ]
        assert len(sites) == 1, (
            f'expected exactly one aggregate <SA> site, found {len(sites)}'
        )
        for expr in sites[0]['values_exprs']:
            assert 'derive' not in expr and 'velocity' not in expr.lower(), (
                'the aggregate status-mix chart is plotting a DERIVED series '
                f'({expr!r}); its bands must stay raw and cumulative.'
            )

    def test_completed_window_tile_stays_cumulative(self, tabs_jsx_body):
        """'Completed (window)' tile history must remain on raw b.done.

        Ties the label and history attributes within the same element so that
        a regression swapping this tile to deriveVelocitySeries is caught.
        (`spark` -> `history`: see the Net velocity test above.)
        """
        assert re.search(
            r'label=["\']Completed \(window\)["\'].*?history=\{b\.done\}',
            tabs_jsx_body,
            re.DOTALL,
        )


# ---------------------------------------------------------------------------
# tabs.jsx BurnTab — every tile, pip and cell reads a SERVED burndown Datum
# ---------------------------------------------------------------------------
#
# The burndown payload carries its own staleness: each block (the aggregate and
# every project) serves `latest` and `forecast` Datums whose state says whether
# a project was carried forward, is missing from the window, or is fresh.
# plainDatum/derivedDatum wrap a bare number in the ENDPOINT's receipt, which
# knows only when the payload arrived — so a tile built that way would read a
# carried project's hours-old count as fresh. BurnTab therefore builds no
# Datum of its own; burndown_bands.js::burndownDatum stamps the served one.

_BURN_TILE_LABELS = {'Net velocity', 'Completed (window)', 'Pending', 'Forecast clear'}


def _burn_tab_body(tabs_jsx_body):
    return strip_js_comments(extract_function_body(tabs_jsx_body, 'BurnTab'))


def _self_closing_elements(src, tag):
    """Every flat ``<tag ... />`` element's attribute text, braces respected.

    Walks brace depth so an arrow's ``=>`` or a nested JSX value inside a
    ``{...}`` prop cannot end the element early; only a ``/>`` at depth 0 does.
    """
    elements = []
    for m in re.finditer(rf'<{tag}\b', src):
        depth = 0
        for i in range(m.end(), len(src)):
            c = src[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            elif depth == 0 and src.startswith('/>', i):
                elements.append(src[m.end():i])
                break
        else:
            raise AssertionError(f'<{tag} at offset {m.start()} is never closed')
    return elements


def _prop_expr(attrs, name):
    """The expression inside ``name={...}``, or None when the prop is absent."""
    m = re.search(rf'\b{name}=\{{', attrs)
    if not m:
        return None
    return walk_balanced(attrs, m.end() - 1)[1:-1].strip()


def _prop_label(attrs):
    m = re.search(r'\blabel=["\']([^"\']*)["\']', attrs)
    return m.group(1) if m else None


def _burn_tiles(body):
    return {_prop_label(a): a for a in _self_closing_elements(body, 'ST')}


class TestBurnTabReadsServedDatums:
    def test_burntab_builds_no_endpoint_granular_datum(self, tabs_jsx_body):
        """No plainDatum/derivedDatum: every reading is a served, stamped Datum."""
        body = _burn_tab_body(tabs_jsx_body)
        assert 'plainDatum(' not in body
        assert 'derivedDatum(' not in body

    def test_burntab_reads_the_latest_and_forecast_datums(self, tabs_jsx_body):
        body = _burn_tab_body(tabs_jsx_body)
        for field in ('latest', 'forecast'):
            assert re.search(rf'burndownDatum\([^()]*["\']{field}["\']\s*\)', body), (
                f'BurnTab never reads the served {field!r} Datum through burndownDatum'
            )

    def test_tabs_jsx_binds_the_datum_reader_and_forecast_formatter(self, tabs_jsx_body):
        m = re.search(r'const\s*\{([^{}]*)\}\s*=\s*window\.DF_BURNDOWN_BANDS', tabs_jsx_body)
        assert m, 'tabs.jsx no longer destructures window.DF_BURNDOWN_BANDS'
        locals_bound = {local for _, local in destructure_bindings(m.group(1))}
        assert {'burndownDatum', 'forecastText'} <= locals_bound

    def test_every_aggregate_tile_renders_a_served_datum(self, tabs_jsx_body):
        """Each tile's datum is a burndownDatum result, never a wrapped bare number."""
        body = _burn_tab_body(tabs_jsx_body)
        tiles = _burn_tiles(body)
        assert set(tiles) == _BURN_TILE_LABELS
        for label, attrs in tiles.items():
            expr = _prop_expr(attrs, 'datum')
            assert expr, f'the {label!r} tile passes no datum'
            if not expr.startswith('burndownDatum('):
                assert re.fullmatch(r'[A-Za-z_$][\w$]*', expr), (
                    f'the {label!r} tile datum {expr!r} is neither a burndownDatum '
                    'call nor a name bound to one'
                )
                assert re.search(rf'\bconst\s+{re.escape(expr)}\s*=\s*burndownDatum\(', body), (
                    f'the {label!r} tile datum {expr!r} is not bound to a burndownDatum result'
                )

    def test_backlog_tile_label_is_retired(self, tabs_jsx_body):
        """The tile shows the pending MEMBER; the backlog VIEW (pending + deferred) is OrchTab's."""
        assert not re.search(r'label=["\']Backlog["\']', _burn_tab_body(tabs_jsx_body))

    def test_active_label_is_retired_for_running(self, tabs_jsx_body):
        body = _burn_tab_body(tabs_jsx_body)
        assert not re.search(r'label=["\']active["\']', body)
        assert not re.search(r'>\s*Active\s*<', body)
        assert re.search(r'label=["\']running["\']', body)
        assert re.search(r'>\s*Running\s*<', body)

    def test_forecast_tile_formats_the_served_range(self, tabs_jsx_body):
        """No client point estimate: the server refuses to synthesise one on sparse history."""
        body = _burn_tab_body(tabs_jsx_body)
        assert _prop_expr(_burn_tiles(body)['Forecast clear'], 'format') == 'forecastText'

    def test_endpoint_table_no_longer_names_burndown(self, tabs_jsx_body):
        src = strip_js_comments(tabs_jsx_body)
        m = re.search(r'const\s+EP\s*=\s*Object\.freeze\(\{', src)
        assert m, 'tabs.jsx no longer declares its EP endpoint table'
        assert not re.search(r'\bburndown\s*:', walk_balanced(src, m.end() - 1))
        assert 'EP.burndown' not in src


# ---------------------------------------------------------------------------
# OrchTab "Completed / day" — the server's per-day series, never re-derived
# ---------------------------------------------------------------------------
#
# The spark plots the project's served `completed_per_day`
# (burndown.py::compute_window_completion): one entry per ISO day, the same
# N-day window its velocity divides by, so the spark and the velocity beside it
# agree about the same window.


class TestCompletedPerDayIsServerSeries:
    def test_orchtab_spark_plots_the_served_completed_per_day(self, tabs_jsx_body):
        body = strip_js_comments(extract_function_body(tabs_jsx_body, 'OrchTab'))
        label_at = body.find('Completed / day')
        assert label_at != -1, "OrchTab no longer labels a 'Completed / day' spark"
        sparks = _self_closing_elements(body[label_at:], 'SP')
        assert sparks, "no <SP> follows OrchTab's 'Completed / day' label"
        values = _prop_expr(sparks[0], 'values')
        assert values and 'BURNDOWN_BY_PROJECT' in values and 'completed_per_day' in values, (
            f"the 'Completed / day' spark plots {values!r}, not the project's served "
            'completed_per_day series'
        )
