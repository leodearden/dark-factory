"""Wiring tests for the Overview tab UI in tab_overview.jsx.

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_curator.py / test_tab_orchestrators.py /
test_index_html.py.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments, walk_balanced


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

    The stripping itself is delegated to `_dashboard_helpers.strip_js_comments`,
    which is quote-aware (it will not eat a `//` inside a string literal) and
    whose contract is pinned by `TestStripJsComments` in
    test_jsx_source_helpers.py.
    """
    return strip_js_comments(tab_overview_jsx_body)


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
        r"TASK_CENSUS_TILES\.find\(\s*(?P<tile>\w+)\s*=>\s*(?P=tile)\.key\s*===\s*'running'\s*\)",
        "TASK_CENSUS_TILES.find(t => t.key === 'running')",
    )


def _panel(code, title):
    """The grid cell whose panel head reads *title*, up to the next grid cell."""
    start = code.find(f'>{title}<')
    assert start != -1, f'OverviewTab renders no panel titled {title!r}'
    end = code.find('className="col-span-', start)
    return code[start:] if end == -1 else code[start:end]


def _the_views_map(panel):
    """The single ``TASK_CENSUS_VIEWS.map(v => ...)`` in *panel*: (parameter, call text)."""
    maps = list(re.finditer(r'\bTASK_CENSUS_VIEWS\.map\(\s*(\w+)\s*=>', panel))
    assert len(maps) == 1, f'expected one TASK_CENSUS_VIEWS.map in the panel, found {len(maps)}'
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
            'censusTotal', 'terminalOfTotal', 'viewShareText', 'TASK_CENSUS_VIEWS', 'TASK_CENSUS_TILES',
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


def _health_row(code, label):
    """The ``{ l: '<label>', ... }`` entry of the System health list, braces included."""
    match = re.search(r"\{\s*l:\s*'" + re.escape(label) + "'", code)
    assert match, f'OverviewTab has no System health row labelled {label!r}'
    row = walk_balanced(code, match.start())
    assert row, f'the {label!r} health row is never closed'
    return row


def _stat_tile(code, label):
    match = re.search(r'<StatTile\s+label="' + re.escape(label) + r'"(.*?)/>', code, re.DOTALL)
    assert match, f'OverviewTab renders no <StatTile label="{label}" ... /> tile'
    return match.group(0)


class TestOverviewReadsTheMemoryReadings:
    """The Overview reads the write queue and the memory ops through memory_readings.js.

    Before: the System health "Write queue" row read raw queue counts that an
    offline probe left as confident zeros, and the Activity timeline summed the
    reads/writes series client-side while the Memory tab's donut reduced a
    different query. Now both read one served value each through one reader;
    the readings themselves are executed in
    dashboard/tests/js/memory_readings.test.mjs.
    """

    def test_destructures_the_memory_readers_without_fallback(self, tab_overview_jsx_code):
        destructure = re.search(
            r'const\s*\{([^}]*)\}\s*=\s*window\.DF_MEMORY_READINGS\s*;', tab_overview_jsx_code,
        )
        assert destructure, 'tab_overview.jsx does not destructure window.DF_MEMORY_READINGS at module scope.'
        assert not re.search(r'window\.DF_MEMORY_READINGS\s*(\|\||&&|\?\?)', tab_overview_jsx_code)
        bound = set(re.findall(r'\w+', destructure.group(1)))
        used = {'writeQueue', 'queueCountsText', 'queueHealth', 'newestHourOps', 'opsTotals', 'opsCaption'}
        assert used <= bound, f'tab_overview.jsx reads {sorted(used - bound)} without binding them'

    @pytest.mark.parametrize('retired', ['queue.counts', 'MEMORY_TIMESERIES'])
    def test_the_retired_reads_are_gone(self, overview_code, retired):
        assert retired not in overview_code, (
            f'OverviewTab still reads `{retired}`, which the server no longer serves.'
        )

    def test_no_client_queue_arithmetic_remains(self, overview_code):
        assert not re.search(r'queue\.pending\s*\+\s*queue\.retry', overview_code), (
            'OverviewTab still sums queue counts by hand.'
        )

    def test_the_write_queue_row_reads_one_queue_datum(self, overview_code):
        queue = _const_bound_to(overview_code, r'writeQueue\(\s*D\s*\)', 'writeQueue(D)')
        row = _health_row(overview_code, 'Write queue')
        assert re.search(
            r'sub:\s*<DatumReading\s+datum=\{\s*' + queue + r'\s*\}\s+format=\{\s*queueCountsText\s*\}',
            row,
        ), f'the Write queue row does not render queueCountsText over {queue}:\n{row}'
        assert re.search(r'\.\.\.\s*queueHealth\(\s*' + queue + r'\s*\)', row), (
            f'the Write queue row does not spread queueHealth({queue}):\n{row}'
        )

    def test_the_memory_ops_tile_reads_the_newest_served_hour(self, overview_code):
        tile = _stat_tile(overview_code, 'Memory ops / min')
        assert re.search(r'datum=\{\s*newestHourOps\(\s*D\s*\)', tile), (
            f'the Memory ops tile is not handed newestHourOps(D):\n{tile}'
        )
        assert re.search(r'history=\{[^}]*MEMORY_OPS\.total\b', tile), (
            f'the Memory ops spark does not read MEMORY_OPS.total:\n{tile}'
        )

    def test_the_activity_timeline_meta_states_the_served_totals(self, overview_code):
        panel = _panel(overview_code, 'Activity timeline')
        assert re.search(
            r'<DatumReading\s+datum=\{\s*opsTotals\(\s*D\s*\)\s*\}\s+format=\{\s*opsCaption\s*\}',
            panel,
        ), f'the Activity timeline meta does not render opsCaption over opsTotals(D):\n{panel}'



class TestOverviewReadsTheSpendReading:
    """The Spend (today) tile reads today's spend through spend_readings.js, as the topbar does.

    One reader for both surfaces, so they cannot disagree; the hole before
    /costs delivers is executed in dashboard/tests/js/spend_readings.test.mjs.
    """

    def test_destructures_the_spend_reader_without_fallback(self, tab_overview_jsx_code):
        destructure = re.search(
            r'const\s*\{([^}]*)\}\s*=\s*window\.DF_SPEND_READINGS\s*;', tab_overview_jsx_code,
        )
        assert destructure, 'tab_overview.jsx does not destructure window.DF_SPEND_READINGS at module scope.'
        assert not re.search(r'window\.DF_SPEND_READINGS\s*(\|\||&&|\?\?)', tab_overview_jsx_code)
        bound = set(re.findall(r'\w+', destructure.group(1)))
        assert {'todaySpend', 'spendText'} <= bound

    def test_the_spend_tile_reads_todays_spend(self, overview_code):
        tile = _stat_tile(overview_code, 'Spend (today)')
        assert re.search(r'datum=\{\s*todaySpend\(\s*D\s*\)\s*\}\s+format=\{\s*spendText\s*\}', tile), (
            f'the Spend (today) tile is not todaySpend(D) formatted by spendText:\n{tile}'
        )

def _health_rows_const(code):
    """The one const array holding the System health rows: (name, array text)."""
    bound = []
    for match in re.finditer(r'\bconst\s+(\w+)\s*=\s*\[', code):
        array = walk_balanced(code, match.end() - 1, '[', ']')
        if re.search(r"\bl:\s*'Graphiti'", array):
            bound.append((match.group(1), array))
    assert len(bound) == 1, f'expected the health rows bound to ONE const array, found {len(bound)}'
    return bound[0]


def _health_rows_map(panel, rows):
    """The ``<rows>.map(<param> => ...)`` call in the System health panel: (param, call text)."""
    match = re.search(rf'\b{rows}\.map\(\s*(\w+)\s*=>', panel)
    assert match, f'the System health panel does not draw its rows with {rows}.map(...)'
    return match.group(1), walk_balanced(panel, panel.index('(', match.start()), '(', ')')


class TestSystemHealthIsDerived:
    """Every System health row and the header are derived, never hardcoded (task 6309).

    Before: the header read a literal 'all ok'; the Graphiti and Mem0 rows were
    `ok: true` over counts get_status never serves; the Taskmaster row was the
    literal 'mcp v0.18 · responsive'; the fused-memory row was green before
    /memory had ever arrived; the Reconciliation row was green with no verdict
    served. The decisions now live in system_health.js and are executed in
    dashboard/tests/js/system_health.test.mjs, phantom verdicts included; these
    pins cover only the wiring.
    """

    _HELPERS = {
        'graphitiHealth', 'mem0Health', 'taskStoreHealth', 'fusedMemoryHealth', 'reconHealth', 'walHealth',
        'healthTone', 'healthSummary',
    }

    def test_destructures_the_health_decisions_without_fallback(self, tab_overview_jsx_code):
        destructure = re.search(
            r'const\s*\{([^}]*)\}\s*=\s*window\.DF_SYSTEM_HEALTH\s*;', tab_overview_jsx_code,
        )
        assert destructure, 'tab_overview.jsx does not destructure window.DF_SYSTEM_HEALTH at module scope.'
        assert not re.search(r'window\.DF_SYSTEM_HEALTH\s*(\|\||&&|\?\?)', tab_overview_jsx_code)
        bound = set(re.findall(r'\w+', destructure.group(1)))
        unbound = self._HELPERS - bound
        assert not unbound, f'tab_overview.jsx reads {sorted(unbound)} without binding them'

    def test_the_header_summarises_exactly_the_rows_drawn(self, overview_code):
        rows, _ = _health_rows_const(overview_code)
        panel = _panel(overview_code, 'System health')
        assert re.search(rf'<span className="meta">\s*\{{\s*healthSummary\(\s*{rows}\s*\)\s*\}}\s*</span>', panel), (
            f'the System health header is not healthSummary({rows}):\n{panel}'
        )
        _health_rows_map(panel, rows)

    def test_no_hardcoded_verdict_remains(self, overview_code):
        assert not re.search(r'\ball ok\b', overview_code), "OverviewTab still hardcodes 'all ok'."
        assert 'v0.18' not in overview_code, 'OverviewTab still hardcodes the Taskmaster version.'

    @pytest.mark.parametrize(
        'label, helper, retired',
        [
            ('Graphiti', 'graphitiHealth', 'node_count'),
            ('Mem0', 'mem0Health', 'memory_count'),
            ('Taskmaster', 'taskStoreHealth', 'mcp v'),
            ('fused-memory', 'fusedMemoryHealth', None),
            ('Reconciliation', 'reconHealth', 'RECON_STATE'),
            ('SQLite WAL', 'walHealth', 'MEMORY_STATUS'),
        ],
    )
    def test_the_row_spreads_its_derived_health(self, overview_code, label, helper, retired):
        _, rows_array = _health_rows_const(overview_code)
        row = _health_row(rows_array, label)
        assert re.search(rf'\.\.\.\s*{helper}\(\s*D\s*\)', row), f'the {label} row does not spread {helper}(D):\n{row}'
        assert not re.search(r'\bok\s*:', row), f'the {label} row still sets its own `ok:`:\n{row}'
        if retired:
            assert retired not in row, f'the {label} row still reads {retired!r}:\n{row}'

    def test_the_dot_and_badge_read_one_tone(self, overview_code):
        rows, _ = _health_rows_const(overview_code)
        panel = _panel(overview_code, 'System health')
        param, call = _health_rows_map(panel, rows)
        tone = rf'healthTone\(\s*{param}\s*\)'
        assert re.search(rf'className=\{{`dot \$\{{{tone}\}}`\}}', call), f'the row dot is not healthTone({param}):\n{call}'
        assert re.search(rf'className=\{{`badge \$\{{{tone}\}}`\}}>\{{{tone}\}}</span>', call), (
            f'the row badge class and text are not healthTone({param}):\n{call}'
        )
        assert not re.search(r'\.ok\s*\?', panel), f'the System health panel still inlines a tone ternary:\n{panel}'

    def test_the_row_renders_its_title(self, overview_code):
        rows, _ = _health_rows_const(overview_code)
        param, call = _health_rows_map(_panel(overview_code, 'System health'), rows)
        assert re.search(rf'\btitle=\{{\s*{param}\.title\b', call), (
            f'the health row never renders {param}.title, so a hole reason never reaches the operator:\n{call}'
        )
