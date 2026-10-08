"""MemoryTab's wiring probes; its behaviour is tested where it can execute.

tabs.jsx is ``type="text/babel"`` behind CDN Babel with no JSX render harness,
so nothing here renders MemoryTab. What the Memory tab DECIDES lives in
``memory_readings.js`` and is executed by
dashboard/tests/js/memory_readings.test.mjs: whether the write queue is a hole
or a measurement and what its hint says (``writeQueue``, ``queueHint``), and
the memory-ops caption, donut total and newest hour (``opsTotals``,
``opsCaption``, ``opsTotalText``, ``newestHourOps``) — the reconciliation of
PRD sketch #11 at the client.

What is left for this file is the WIRING that source text can settle and
behaviour cannot: that MemoryTab reaches those readers, and kept no second
answer beside them (a client re-count of the breakdown, a raw read of the
retired queue counts or of the two retired wire keys). Every probe runs over
comment-stripped code, so prose that names a retired key is free to.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments


@pytest.fixture(scope='module')
def tabs_jsx_code(tabs_jsx_body):
    return strip_js_comments(tabs_jsx_body)


@pytest.fixture(scope='module')
def memory_tab_code(tabs_jsx_code):
    return extract_function_body(tabs_jsx_code, 'MemoryTab')


def _stat_tile(code: str, label: str) -> str:
    """The self-closing ``<ST label="<label>" ... />`` element; raises on a miss."""
    match = re.search(r'<ST\s+label="' + re.escape(label) + r'"(.*?)/>', code, re.DOTALL)
    assert match is not None, (
        f'MemoryTab renders no self-closing <ST label="{label}" ... /> tile, and an '
        'assertion over an empty slice would pass vacuously.'
    )
    return match.group(0)


def _panel(code: str, title: str) -> str:
    """The grid cell whose panel head reads *title*, up to the next grid cell."""
    start = code.find(f'>{title}<')
    assert start != -1, f'MemoryTab renders no panel titled {title!r}'
    end = code.find('className="col-span-', start)
    return code[start:] if end == -1 else code[start:end]


def test_tabs_destructures_the_memory_readers_without_fallback(tabs_jsx_code):
    destructure = re.search(
        r'const\s*\{([^}]*)\}\s*=\s*window\.DF_MEMORY_READINGS\s*;', tabs_jsx_code,
    )
    assert destructure, 'tabs.jsx does not destructure window.DF_MEMORY_READINGS at module scope.'
    assert not re.search(r'window\.DF_MEMORY_READINGS\s*(\|\||&&|\?\?)', tabs_jsx_code), (
        'a fallback turns a load-order regression into a silently blank Memory tab.'
    )
    bound = set(re.findall(r'\w+', destructure.group(1)))
    used = {'writeQueue', 'queueHint', 'newestHourOps', 'opsTotals', 'opsCaption', 'opsTotalText'}
    assert used <= bound, f'tabs.jsx reads {sorted(used - bound)} without binding them'


def test_the_write_queue_tile_reads_one_queue_datum(memory_tab_code):
    """The tile's reading and its hint come from ONE writeQueue(DF) Datum."""
    binding = re.search(r'\bconst\s+(\w+)\s*=\s*writeQueue\(\s*DF\s*\)', memory_tab_code)
    assert binding, 'MemoryTab does not bind writeQueue(DF) to a const'
    queue = binding.group(1)
    assert len(re.findall(r'\bwriteQueue\(', memory_tab_code)) == 1, (
        'MemoryTab builds the write-queue Datum more than once'
    )
    tile = _stat_tile(memory_tab_code, 'Write queue')
    assert re.search(rf'datum=\{{\s*{queue}\s*\}}', tile), (
        f'the Write queue tile is not handed the bound `{queue}` Datum:\n{tile}'
    )
    assert re.search(rf'hint=\{{\s*queueHint\(\s*{queue}\s*\)\s*\}}', tile), (
        f'the Write queue tile hint is not queueHint({queue}):\n{tile}'
    )


def test_the_ops_tile_reads_the_newest_served_hour(memory_tab_code):
    tile = _stat_tile(memory_tab_code, 'Ops / hr')
    assert re.search(r'datum=\{\s*newestHourOps\(', tile), (
        f'the Ops / hr tile is not handed newestHourOps(...):\n{tile}'
    )
    assert re.search(r'history=\{[^}]*MEMORY_OPS\.total\b', tile), (
        f'the Ops / hr spark does not read the served hourly MEMORY_OPS.total:\n{tile}'
    )


def test_the_donut_centre_is_the_served_total(memory_tab_code):
    assert re.search(r'centerValue=\{\s*opsTotalText\(', memory_tab_code), (
        "the Operations donut's centre is not opsTotalText(...)"
    )
    assert '.reduce(' not in memory_tab_code, (
        'MemoryTab re-counts a served series with .reduce( — the server derives '
        'MEMORY_OPS.totals so the caption and the donut cannot disagree.'
    )


def test_the_reads_vs_writes_panel_states_the_window_totals(memory_tab_code):
    panel = _panel(memory_tab_code, 'Reads vs writes · last 24h')
    assert re.search(
        r'<DatumReading\s+datum=\{\s*opsTotals\([^)]*\)\s*\}\s+format=\{\s*opsCaption\s*\}',
        panel,
    ), f'the Reads vs writes panel does not render opsCaption over opsTotals:\n{panel}'


@pytest.mark.parametrize('retired', ['queue.counts', 'MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN'])
def test_the_retired_reads_are_gone(memory_tab_code, retired):
    assert retired not in memory_tab_code, (
        f'MemoryTab still reads `{retired}`, which /memory and /memory-graphs no '
        'longer serve — it would render undefined silently.'
    )
