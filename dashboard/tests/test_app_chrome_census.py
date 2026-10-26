"""Contract: the topbar pill and the rail badge read ONE census binding.

The reported defect was two surfaces counting two populations: the topbar's
"tasks active" summed per-orchestrator summaries (all zeros once /orchestrators
stopped measuring counts) while the rail's Tasks badge counted in-progress,
blocked and pending rows of ACTIVE_TASKS. Neither was the census.

The fix is structural, so the test is too. App computes ``censusOver(DD, null)``
exactly once, and both surfaces hand that same binding to the shared
``DatumReading`` with a named reading from task_snapshot.js — the topbar
``runningOfInFlight`` ("25 running of 43 in-flight"), the rail
``inFlightCount`` ("43"). Two readings of one Datum object cannot disagree
about in-flight; the node suite (dashboard/tests/js/task_snapshot.test.mjs)
executes both readings over that one datum.

The .jsx files cannot run under any harness here (test_datum_components.py's
docstring), so every probe below reads the SERVED source with comments
stripped: a presence probe satisfied by prose is a vacuous pass.
"""

from __future__ import annotations

import re

from _dashboard_helpers import extract_function_body, strip_js_comments, walk_balanced


def _app_code(app_jsx_body: str) -> str:
    return strip_js_comments(app_jsx_body)


def _app_body(app_jsx_body: str) -> str:
    return extract_function_body(_app_code(app_jsx_body), 'App')


def _const_object(body: str, name: str) -> str:
    """The ``const <name> = { ... }`` literal in *body*, braces included."""
    match = re.search(rf'\bconst\s+{name}\s*=\s*\{{', body)
    assert match, f'App no longer declares `const {name} = {{ ... }}`'
    literal = walk_balanced(body, match.end() - 1)
    assert literal, f'the `{name}` object literal is never closed'
    return literal


def _census_binding(app_jsx_body: str) -> str:
    body = _app_body(app_jsx_body)
    calls = re.findall(r'censusOver\(\s*DD\s*,\s*null\s*\)', body)
    assert len(calls) == 1, (
        f'App calls censusOver(DD, null) {len(calls)} time(s); it must call it '
        'exactly once, so the topbar and the rail read one Datum object.'
    )
    bound = re.search(r'\bconst\s+(\w+)\s*=\s*censusOver\(\s*DD\s*,\s*null\s*\)', body)
    assert bound, 'App does not bind censusOver(DD, null) to a const'
    return bound.group(1)


def _task_reading(literal: str, where: str) -> tuple[str, str]:
    """The (datum, format) of the ``tasks:`` entry's DatumReading in *literal*."""
    match = re.search(
        r'\btasks\s*:\s*<DatumReading\s+datum=\{\s*(\w+)\s*\}\s+format=\{\s*(\w+)\s*\}\s*/>',
        literal,
    )
    assert match, (
        f'{where} has no `tasks: <DatumReading datum={{…}} format={{…}} />` entry — '
        'the tasks count must render through the shared DatumReading.'
    )
    return match.group(1), match.group(2)


def test_app_destructures_the_census_reader_without_fallback(app_jsx_body: str) -> None:
    code = _app_code(app_jsx_body)
    assert re.search(r'=\s*window\.DF_TASK_SNAPSHOT\s*;', code), (
        'app.jsx does not destructure window.DF_TASK_SNAPSHOT at module scope.'
    )
    assert not re.search(r'window\.DF_TASK_SNAPSHOT\s*(\|\||&&|\?\?)', code), (
        'app.jsx guards its window.DF_TASK_SNAPSHOT read with a fallback, which '
        'turns a load-order regression into a silently blank topbar.'
    )


def test_topbar_and_rail_read_the_same_census_binding(app_jsx_body: str) -> None:
    census = _census_binding(app_jsx_body)
    body = _app_body(app_jsx_body)

    topbar = _task_reading(_const_object(body, 'summary'), 'the topbar `summary`')
    rail = _task_reading(_const_object(body, 'railCounts'), '`railCounts`')

    assert topbar == (census, 'runningOfInFlight'), (
        f'the topbar tasks pill renders {topbar}; expected ({census!r}, '
        "'runningOfInFlight') — the running sub-view with its in-flight superset."
    )
    assert rail == (census, 'inFlightCountReading'), (
        f'the rail Tasks badge renders {rail}; expected ({census!r}, '
        "'inFlightCountReading') — the same in-flight number the topbar shows."
    )
    sites = re.findall(rf'<DatumReading\s+datum=\{{\s*{census}\s*\}}', body)
    assert len(sites) == 2, (
        f'{census} feeds {len(sites)} DatumReading site(s) in App; exactly the '
        'topbar pill and the rail badge read it.'
    )


def test_the_two_old_task_reductions_are_gone(app_jsx_body: str) -> None:
    code = _app_code(app_jsx_body)
    for retired in ('orchSummary', 'DF_ORCH_SUMMARY', 'tasksActive', 'ORCHESTRATORS.reduce', 'ACTIVE_TASKS'):
        assert retired not in code, (
            f'app.jsx still references `{retired}`. The topbar and rail read the '
            'census now; the per-orchestrator reduction and the ACTIVE_TASKS row '
            'count were two different populations.'
        )


def test_the_escalations_rail_count_is_untouched(app_jsx_body: str) -> None:
    rail = _const_object(_app_body(app_jsx_body), 'railCounts')
    assert re.search(r'\besc\s*:\s*DD\.ESCALATIONS\?\.summary\?\.by_status\?\.pending\s*\?\?\s*0', rail), (
        'the railCounts `esc:` entry changed; this leaf rewires only the tasks count.'
    )


def test_stat_strip_renders_the_tasks_node(shell_jsx_body: str) -> None:
    body = extract_function_body(strip_js_comments(shell_jsx_body), 'StatStrip')
    assert 'tasksActive' not in body, 'StatStrip still reads summary.tasksActive'
    assert 'tasks active' not in body, "StatStrip still labels the pill 'tasks active'"
    assert re.search(r'\{\s*summary\.tasks\s*\}', body), (
        'StatStrip does not render {summary.tasks}, the DatumReading node App builds.'
    )


def test_the_topbar_queue_reads_the_write_queue_datum(app_jsx_body: str) -> None:
    """The topbar queue count is the served write-queue Datum, not a bare number.

    A bare ``queue.counts.pending`` rendered a confident 0 whenever the queue
    probe failed; through ``writeQueue`` an unmeasured queue is an em-dash
    with the probe's reason (memory_readings.test.mjs executes the reader).
    """
    summary = _const_object(_app_body(app_jsx_body), 'summary')
    assert re.search(r'\bqueue\s*:\s*<DatumReading\s+datum=\{\s*writeQueue\(\s*DD\s*\)\s*\}', summary), (
        f'summary.queue is not a <DatumReading> over writeQueue(DD):\n{summary}'
    )
    assert 'queue.counts' not in _app_body(app_jsx_body)


def test_the_topbar_spend_reads_the_shared_spend_reading(app_jsx_body: str) -> None:
    """The topbar spend is a Datum reading, never a seeded $0.00.

    ``DD.COSTS?.summary?.today ?? 0`` rendered a confident $0.00 before /costs
    had ever delivered (data.js seeds today = 0). spend_readings.js::todaySpend
    reads it through the /costs receipt, for the topbar and the Overview tile
    alike; spend_readings.test.mjs executes that reading, and this pins only
    the wiring.
    """
    code = _app_code(app_jsx_body)
    destructure = re.search(r'const\s*\{([^}]*)\}\s*=\s*window\.DF_SPEND_READINGS\s*;', code)
    assert destructure, 'app.jsx does not destructure window.DF_SPEND_READINGS at module scope.'
    assert {'todaySpend', 'spendText'} <= set(re.findall(r'\w+', destructure.group(1)))
    assert not re.search(r'window\.DF_SPEND_READINGS\s*(\|\||&&|\?\?)', code)
    summary = _const_object(_app_body(app_jsx_body), 'summary')
    assert re.search(
        r'\bspend24h\s*:\s*<DatumReading\s+datum=\{\s*todaySpend\(\s*DD\s*\)\s*\}\s+format=\{\s*spendText\s*\}\s*/>',
        summary,
    ), f'summary.spend24h is not a <DatumReading> over todaySpend(DD) formatted by spendText:\n{summary}'
    assert '?? 0' not in summary, f'the topbar summary still zero-fills a reading:\n{summary}'


def test_stat_strip_renders_the_spend_node(shell_jsx_body: str) -> None:
    """A spend hole reaches the operator as '—', not as a TypeError on ``.toFixed``."""
    body = extract_function_body(strip_js_comments(shell_jsx_body), 'StatStrip')
    assert re.search(r'\{\s*summary\.spend24h\s*\}', body), (
        'StatStrip does not render {summary.spend24h}, the DatumReading node App builds.'
    )
    assert 'spend24h.toFixed' not in body, 'StatStrip still formats the spend as a bare number.'
