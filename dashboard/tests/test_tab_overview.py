"""Wiring tests for the Overview tab UI in tab_overview.jsx.

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_curator.py / test_tab_orchestrators.py /
test_index_html.py.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments, walk_balanced


@pytest.fixture(scope='module')
def tab_overview_jsx_body(_client):
    return _client.get('/static/redux/tab_overview.jsx').text


class TestOverviewTabCurrentTaskRemoved:
    """The 'Current task' column must not appear in the Overview tab."""

    def test_tab_overview_jsx_served(self, _client):
        resp = _client.get('/static/redux/tab_overview.jsx')
        assert resp.status_code == 200

    def test_overview_tab_positive_anchor_function(self, tab_overview_jsx_body):
        """File must still export OverviewTab — guards against a renamed/empty file."""
        assert 'function OverviewTab(' in tab_overview_jsx_body

    def test_current_task_td_render_ref_removed(self, tab_overview_jsx_body):
        """The JSX expression {o.current_task} must NOT appear in tab_overview.jsx."""
        assert not re.search(r'\{\s*o\.current_task\s*\}', tab_overview_jsx_body)

    def test_current_task_th_removed(self, tab_overview_jsx_body):
        """The <th>Current task</th> column header must NOT appear in tab_overview.jsx."""
        assert '<th>Current task</th>' not in tab_overview_jsx_body


class TestHostLoadCardStaleness:
    """HostLoadCard must surface a stale/offline badge when /api/load fails.

    step-19 RED: The test below fails today because the panel-head renders no
    badge.  After step-20 GREEN it passes.
    """

    def test_host_load_card_stale_badge_rendered(self, tab_overview_jsx_body):
        """HostLoadCard panel-head must render a stale/offline badge element.

        Accepts either a className containing 'stale' (e.g. className="badge stale")
        or an element with visible text '>stale<' / '>offline<'.
        This is the key behavioral contract: a failed /api/load must surface a
        visible stale/offline marker rather than frozen/blank values.
        """
        jsx = tab_overview_jsx_body
        has_stale_class = bool(re.search(r'className=["\'][^"\']*stale', jsx))
        has_stale_text = bool(re.search(r'>stale<|>offline<', jsx))
        assert has_stale_class or has_stale_text, (
            'HostLoadCard panel-head must render a stale/offline badge '
            '(e.g. <span className="badge stale">stale</span> or equivalent); '
            'currently no stale badge is rendered in tab_overview.jsx'
        )


@pytest.fixture(scope='module')
def tab_overview_jsx_code(tab_overview_jsx_body):
    """`tab_overview.jsx` with every comment stripped.

    Same shape as `test_tab_memory_evals.py`'s `tab_memory_evals_jsx_code`
    fixture, for the same reason it was created there: a substring assertion
    over the raw body is satisfied by a MENTION in a comment just as well as by
    a render site.  That false-pass mode is not hypothetical — the memory-evals
    file's `alarmed_open` / `clear` assertions once passed while matching only
    explanatory prose.

    It matters here because the phantom branch below is necessarily accompanied
    by a comment explaining what a phantom IS, and that comment names both
    `is_phantom` and `unreviewed` — so the branch-scoping regex in
    `_phantom_branch` would otherwise anchor on the comment's first mention of
    `is_phantom` rather than on the `if (v?.is_phantom)` render site.

    The stripping itself is delegated to `_dashboard_helpers.strip_js_comments`,
    which is quote-aware (it will not eat a `//` inside a string literal) and
    whose contract is pinned by `TestStripJsComments` in
    test_jsx_source_helpers.py.
    """
    return strip_js_comments(tab_overview_jsx_body)


class TestReconciliationHealthRowPhantom:
    """The Reconciliation System-health row must not paint a PHANTOM verdict red.

    A phantom is the placeholder the reconciliation judge fabricates when it
    could not parse its own model output — the run went UNREVIEWED.  It is
    stored with `severity='serious'`, so the row's `ok: sev !== 'serious'`
    renders a red `bad` dot reading `verdict: serious · halt`: a serious
    finding no judge ever made.  `get_latest_verdict` now ships `is_phantom`
    (task 3287 step-8); this is the renderer that has to consume it, or the
    user-visible defect stays in place and the payload field is dead weight.

    Measured caveat, recorded honestly: the newest verdict on the live DB
    today is an ordinary `ok`, so this is a LATENT mis-render.  It was visible
    across the 2026-07-20..29 stretch and returns the next time a phantom is
    the newest row.
    """

    @staticmethod
    def _phantom_branch(code: str) -> str:
        """The `is_phantom` branch body, isolated from the rest of the file.

        Every assertion below is scoped to this slice rather than grepping the
        whole 400-line file.  A file-wide `'unreviewed' in code` /
        `'is_phantom' in code` grep keeps passing if the branch is edited back
        to ``sub: `verdict: ${sev}` `` as long as the word survives ANYWHERE
        else — in a different health row, an unrelated label, or (before the
        comment-stripping fixture) a comment.  Those greps pinned wording, not
        behaviour, so they are gone; this is the one scope with teeth.

        The scope terminator is `};` — the end of the branch's `return {...};`
        — NOT a bare `}`.  A bare `}` stops at the first `${...}` template
        interpolation in the `sub` string, truncating the match before the
        `ok:` / `warn:` keys these assertions exist to inspect, which would
        leave them scanning text that can never contain them.
        """
        match = re.search(r'is_phantom[\s\S]{0,400}?\};', code)
        assert match, 'no is_phantom branch found in tab_overview.jsx render code'
        return match.group(0)

    def test_ordinary_severity_path_survives(self, tab_overview_jsx_code):
        """(a) Positive anchor: the non-phantom branch is unchanged.

        A genuine `severity=serious` verdict must still paint red — suppressing
        that would be strictly worse than the over-report being fixed.
        """
        assert "sev !== 'serious'" in tab_overview_jsx_code

    def test_phantom_branch_does_not_paint_red(self, tab_overview_jsx_code):
        """(b) Negative guard: the phantom branch must not set `ok: false`.

        A phantom is `warn` (yellow): the run genuinely went unreviewed, which
        is degraded, but no judge found anything serious.
        """
        branch = self._phantom_branch(tab_overview_jsx_code)
        assert not re.search(r'\bok:\s*false\b', branch), (
            'the is_phantom branch must not render a red `bad` row — a phantom '
            f'is a fabricated placeholder, not a serious finding; got: {branch!r}'
        )
        assert re.search(r'\bwarn:\s*true\b', branch), (
            'the is_phantom branch must render `warn: true` (yellow) — the run '
            f'genuinely went unreviewed, which is degraded; got: {branch!r}'
        )

    def test_phantom_branch_labels_the_run_unreviewed(self, tab_overview_jsx_code):
        """(c) The operator-facing text in THIS branch must say `unreviewed`.

        `verdict: serious` is the exact lie being fixed, so the substitute
        label has to name what actually happened — and the `verdict: ${sev}`
        template the ordinary path uses must NOT survive inside the phantom
        branch, which is what an edit reverting the fix would leave behind.
        """
        branch = self._phantom_branch(tab_overview_jsx_code)
        assert 'unreviewed' in branch, (
            'the is_phantom branch must label the row unreviewed rather than '
            f'reporting a verdict no judge made; got: {branch!r}'
        )
        assert 'verdict: ${sev}' not in branch, (
            'the is_phantom branch must not fall back to the ordinary '
            f'`verdict: ${{sev}}` label; got: {branch!r}'
        )


@pytest.fixture(scope='module')
def overview_code(tab_overview_jsx_code):
    """OverviewTab's body with comments stripped, so prose satisfies no probe."""
    return extract_function_body(tab_overview_jsx_code, 'OverviewTab')


def _const_bound_to(code, expression, what):
    match = re.search(rf'\bconst\s+(\w+)\s*=\s*{expression}', code)
    assert match, f'OverviewTab does not bind {what} to a const'
    return match.group(1)


def _fleet_census(code):
    return _const_bound_to(code, r'censusOver\(\s*D\s*,\s*null\s*\)', 'censusOver(D, null)')


def _running_tile_entry(code):
    return _const_bound_to(
        code,
        r"CENSUS_TILES\.find\(\s*(?P<tile>\w+)\s*=>\s*(?P=tile)\.key\s*===\s*'running'\s*\)",
        "CENSUS_TILES.find(t => t.key === 'running')",
    )


def _panel(code, title):
    """The grid cell whose panel head reads *title*, up to the next grid cell."""
    start = code.find(f'>{title}<')
    assert start != -1, f'OverviewTab renders no panel titled {title!r}'
    end = code.find('className="col-span-', start)
    return code[start:] if end == -1 else code[start:end]


def _the_views_map(panel):
    """The single ``CENSUS_VIEWS.map(v => ...)`` in *panel*: (parameter, call text)."""
    maps = list(re.finditer(r'\bCENSUS_VIEWS\.map\(\s*(\w+)\s*=>', panel))
    assert len(maps) == 1, f'expected one CENSUS_VIEWS.map in the panel, found {len(maps)}'
    paren = panel.index('(', maps[0].start())
    return maps[0].group(1), walk_balanced(panel, paren, '(', ')')


class TestOverviewReadsTheCensus:
    """The Overview renders every task count from the served census (leaf γ2, task 5589).

    Before: the "Active tasks" tile, the "Task pipeline" card and the
    Orchestrators table's "Done" column summed per-orchestrator summaries that
    /orchestrators stopped measuring (task 5587), so they read zeros or an
    em-dash with no census behind them. Now one fleet census —
    ``censusOver(D, null)`` — feeds the running tile and the pipeline, and each
    orchestrator row reads its own project's census, all through StatTile or the
    shared DatumReading with a named reading from task_snapshot.js. The readings
    themselves are executed over the boundary sketches in
    dashboard/tests/js/task_snapshot.test.mjs; these pins prove the wiring
    reaches them.
    """

    @pytest.mark.parametrize(
        'retired',
        [
            'DF_ORCH_SUMMARY',
            'orchSummary',
            'hasOrchSummary',
            'ORCH_SUMMARY_ABSENT_REASON',
            'tasksTotal',
            'taskShare',
            'ORCHESTRATORS.reduce',
            'Active tasks',
        ],
    )
    def test_the_orchestrator_summary_readers_are_gone(self, tab_overview_jsx_code, retired):
        assert retired not in tab_overview_jsx_code, (
            f'tab_overview.jsx still contains `{retired}`. The Overview reads the '
            'census now; a per-orchestrator summary is a count /orchestrators no '
            'longer measures.'
        )

    def test_destructures_the_census_reader_without_fallback(self, tab_overview_jsx_code):
        destructure = re.search(r'const\s*\{([^}]*)\}\s*=\s*window\.DF_TASK_SNAPSHOT\s*;', tab_overview_jsx_code)
        assert destructure, 'tab_overview.jsx does not destructure window.DF_TASK_SNAPSHOT at module scope.'
        assert not re.search(r'window\.DF_TASK_SNAPSHOT\s*(\|\||&&|\?\?)', tab_overview_jsx_code), (
            'a fallback turns a load-order regression into a silently blank Overview.'
        )
        bound = set(re.findall(r'\w+', destructure.group(1)))
        used = {
            'projectCensus', 'censusOver', 'censusSegments', 'censusHistory',
            'censusTotal', 'terminalOfTotal', 'viewShareText', 'CENSUS_VIEWS', 'CENSUS_TILES',
        }
        assert used <= bound, f'tab_overview.jsx reads {sorted(used - bound)} without binding them'

    def test_datum_reading_comes_from_the_shared_shell(self, tab_overview_jsx_code):
        assert re.search(r'const\s*\{[^}]*\bDatumReading\b[^}]*\}\s*=\s*window\.DF_SHELL\s*;', tab_overview_jsx_code), (
            'tab_overview.jsx renders <DatumReading> without binding it from window.DF_SHELL.'
        )

    def test_binds_one_fleet_census(self, overview_code):
        calls = re.findall(r'censusOver\(\s*D\s*,\s*null\s*\)', overview_code)
        assert len(calls) == 1, (
            f'OverviewTab calls censusOver(D, null) {len(calls)} time(s); the tile and '
            'the pipeline must read one Datum object.'
        )
        _fleet_census(overview_code)

    def test_the_running_tile_reads_the_fleet_census(self, overview_code):
        fleet = _fleet_census(overview_code)
        running = _running_tile_entry(overview_code)
        tiles = [t for t in re.findall(r'<StatTile\b[\s\S]*?/>', overview_code) if f'datum={{{fleet}}}' in t]
        assert len(tiles) == 1, f'expected one StatTile reading {fleet}, found {len(tiles)}'
        tile = tiles[0]
        assert f'label={{{running}.label}}' in tile
        assert f'format={{{running}.reading}}' in tile
        assert re.search(rf'history=\{{\s*censusHistory\(\s*D\s*,\s*null\s*,\s*{running}\s*\)\s*\}}', tile), (
            "the tile's spark is not the burndown series of the member its headline shows."
        )
        for prop in ('unit', 'hint'):
            assert not re.search(rf'\b{prop}=', tile), (
                f'the running tile carries a `{prop}=`: a number beside the Datum has no '
                'state, age or reason of its own.'
            )

    def test_the_pipeline_meta_is_the_census_total(self, overview_code):
        fleet = _fleet_census(overview_code)
        panel = _panel(overview_code, 'Task pipeline')
        assert re.search(rf'<DatumReading\s+datum=\{{{fleet}\}}\s+format=\{{censusTotal\}}', panel)

    def test_the_pipeline_bar_maps_the_census_segments(self, overview_code):
        fleet = _fleet_census(overview_code)
        panel = _panel(overview_code, 'Task pipeline')
        assert re.search(rf'censusSegments\(\s*{fleet}\s*\)\.map\(', panel)

    def test_the_pipeline_rows_are_the_views_read_off_the_census(self, overview_code):
        fleet = _fleet_census(overview_code)
        view, call = _the_views_map(_panel(overview_code, 'Task pipeline'))
        assert f'{{{view}.label}}' in call
        assert f'P[{view}.tone]' in call
        assert re.search(rf'<DatumReading\s+datum=\{{{fleet}\}}\s+format=\{{{view}\.count\}}', call)
        assert re.search(
            rf'<DatumReading\s+datum=\{{{fleet}\}}\s+format=\{{viewShareText\(\s*{view}\.key\s*\)\}}', call
        ), "each row's share must be a reading of the census, so a hole renders the placeholder."

    def test_no_hand_division_of_task_counts_remains(self, overview_code):
        assert not re.search(r'/\s*tasks[A-Z]', overview_code), 'a `/ tasks…` division remains'
        assert not re.search(r'/\s*(?:\w+\.)*total\b', overview_code), (
            'a `v / total` division remains; at total 0 it renders "NaN%".'
        )

    def test_the_orchestrators_table_reads_each_project_census(self, overview_code):
        panel = _panel(overview_code, 'Orchestrators · current work')
        assert '>Terminal<' in panel
        assert '>Done<' not in panel, 'the column is the terminal view, not the done member'
        assert re.search(
            r'<DatumReading\s+datum=\{\s*projectCensus\(\s*D\s*,\s*o\.project\s*\)\s*\}\s+format=\{\s*terminalOfTotal\s*\}',
            panel,
        )
