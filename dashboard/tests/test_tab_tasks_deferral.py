"""The Tasks tab's deferral holds: asset wiring, and the round trip from the collector to the DOM.

The cell DECISION is tested as a pure function in
``dashboard/tests/js/deferral_cell.test.mjs``. This module covers what that
suite cannot see. First, the module is reachable from the browser, tagged in
``index.html`` exactly once with its siblings' cache-buster, and evaluated
before the JSX that reads its global; those tests assert structural facts about
served HTML, never JSX source text. Second, rows the REAL collector shaped from
``_deferral_fixture.py`` render in the real Tasks tab, and each project's
header count equals the count ``shared.task_deferral.needs_human`` gives over
the same rows. ``js/tasks_deferral_render.mjs`` is the node half. Its name is
off the ``*.test.mjs`` glob because only this module builds its payload.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _cache_buster_helpers import redux_cache_buster_versions
from _canned_mcp import CannedMCP
from _dashboard_helpers import ScriptTagCollector, assert_script_loads_before, node_path
from _deferral_fixture import (
    NOW,
    FixtureRow,
    deferral_fixture,
    expected_needs_human,
    raw_rows,
    second_project_fixture,
    status_map,
)
from shared.process_identity import capture

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import collect_tasks_with_counts

_CELL_SRC = '/static/redux/deferral_cell.js'
_JS_DIR = Path(__file__).parent / 'js'
_RENDER_SUITE = 'tasks_deferral_render.mjs'
_TAP_TEST_COUNT_RE = re.compile(r'^# tests (\d+)$', re.MULTILINE)


def _redux_script_srcs(body: str) -> list[str]:
    collector = ScriptTagCollector()
    collector.feed(body)
    srcs = (attrs.get('src') or '' for attrs in collector.script_attrs)
    return [src for src in srcs if src.startswith('/static/redux/')]


def test_cell_module_is_served(_client):
    resp = _client.get(_CELL_SRC)

    assert resp.status_code == 200
    assert 'deferralCellState' in resp.text


def test_index_html_tags_the_cell_once_with_the_sibling_cache_buster(index_html_body):
    srcs = _redux_script_srcs(index_html_body)
    cell = [src for src in srcs if src.startswith(_CELL_SRC)]
    assert len(cell) == 1, f'expected exactly one {_CELL_SRC} tag, got {cell}'

    sibling_versions = redux_cache_buster_versions(
        ' '.join(src for src in srcs if not src.startswith(_CELL_SRC))
    )
    assert len(sibling_versions) == 1, f'the redux siblings disagree on ?v=: {sibling_versions}'
    assert redux_cache_buster_versions(cell[0]) == sibling_versions, (
        f'{cell[0]} does not carry its siblings\' ?v={sibling_versions}'
    )


def test_cell_module_loads_before_tab_tasks_jsx(index_html_body):
    assert_script_loads_before(
        index_html_body,
        _CELL_SRC,
        '/static/redux/tab_tasks.jsx',
        'deferral_cell.js',
        'tab_tasks.jsx',
        consumer_note='tab_tasks.jsx reads window.DF_DEFERRAL_CELL at module scope.',
    )


async def _no_runtime(client, escalation_urls):
    return {}


async def _collected_entry(
    fixture: tuple[FixtureRow, ...], root: Path, client
) -> tuple[str, dict[str, Any]]:
    """*fixture* served to the real collector for one root: its label and wire entry."""
    root.mkdir(parents=True, exist_ok=True)
    canned = CannedMCP(rows=raw_rows(fixture), status_map=status_map(fixture))
    with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
        _active, snapshots = await collect_tasks_with_counts(
            client, DashboardConfig(project_root=root), now=NOW,
        )
    ((label, snapshot),) = snapshots.items()
    return label, snapshot.to_wire()


def _needs_human_count(fixture: tuple[FixtureRow, ...]) -> int:
    verdicts = expected_needs_human(fixture, now=NOW).values()
    return sum(1 for reason in verdicts if reason is not None)


def _without_deferral(entry: dict[str, Any]) -> dict[str, Any]:
    """*entry* as a backend that predates the ``deferral`` key would serve it."""
    old = copy.deepcopy(entry)
    for row in old['rows']['value']:
        row.pop('deferral', None)
    return old


@pytest.fixture()
async def render_payload(tmp_path, dummy_client, monkeypatch) -> dict[str, Any]:
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', _no_runtime)
    live_holder = capture(os.getpid())
    assert live_holder is not None
    first = deferral_fixture(now=NOW, live_holder=live_holder)
    second = second_project_fixture(now=NOW)
    first_label, first_entry = await _collected_entry(first, tmp_path / 'alpha', dummy_client)
    second_label, second_entry = await _collected_entry(second, tmp_path / 'beta', dummy_client)
    return {
        'served_at': NOW.isoformat(),
        'projects': {first_label: first_entry, second_label: second_entry},
        'expected_needs_human': {
            first_label: _needs_human_count(first),
            second_label: _needs_human_count(second),
        },
        'old_backend': {first_label: _without_deferral(first_entry)},
    }


def test_collected_holds_render_in_the_tasks_tab_with_the_shared_count(render_payload, tmp_path):
    counts = render_payload['expected_needs_human']
    assert sorted(counts.values()) == [1, 10], (
        f'the fixture must give the two projects distinct, non-zero counts: {counts}'
    )
    payload_path = tmp_path / 'deferral_render_payload.json'
    payload_path.write_text(json.dumps(render_payload))

    result = subprocess.run(
        [node_path(), '--test', '--test-reporter=tap', _RENDER_SUITE],
        capture_output=True,
        text=True,
        cwd=_JS_DIR,
        env={**os.environ, 'DF_DEFERRAL_RENDER_PAYLOAD': str(payload_path)},
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f'{_RENDER_SUITE} exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    ran = _TAP_TEST_COUNT_RE.search(result.stdout)
    assert ran is not None and int(ran.group(1)) > 0, (
        f'{_RENDER_SUITE} ran no tests, which proves nothing\n{result.stdout}'
    )
