"""The public seams of the task-snapshot collectors."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from _canned_mcp import CannedMCP, _raw_row

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import TASKS_BUDGET, collect_tasks_with_counts
from dashboard.data.datum import DatumState

pytestmark = pytest.mark.usefixtures('fresh_task_snapshot_state')

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)

_PAIRS = ((1, 'in-progress'), (2, 'pending'), (3, 'done'))


@pytest.mark.parametrize(
    ('field', 'bad'),
    [('per_project', 0), ('total', -1.0), ('root_concurrency', 0)],
)
def test_tasks_budget_refuses_a_bound_that_cannot_serve_a_root(field, bad):
    with pytest.raises(ValueError, match=field) as refused:
        replace(TASKS_BUDGET, **{field: bad})

    assert str(bad) in str(refused.value), (
        f'the refusal must name the offending value {bad!r}, got {refused.value}'
    )


@pytest.mark.parametrize('bad', [0, -3])
async def test_a_terminal_window_that_reads_no_row_is_refused_before_any_read(
    bad, dummy_client, dashboard_config, monkeypatch,
):
    from dashboard.data.task_snapshot import acquire_terminal_window

    canned = CannedMCP()
    monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', canned)

    with pytest.raises(ValueError, match='window') as refused:
        await acquire_terminal_window(
            dummy_client, dashboard_config, dashboard_config.project_root,
            now=NOW, terminal_total=10, window=bad,
        )

    assert str(bad) in str(refused.value), (
        f'the refusal must name the offending value {bad!r}, got {refused.value}'
    )
    assert canned.calls == [], (
        f'a window that cannot hold a row must not reach the substrate: {canned.calls}'
    )


async def test_reset_forgets_every_unit_the_last_good_and_the_rotation(
    tmp_path, dummy_client, monkeypatch,
):
    from dashboard.data.active_tasks import reset_task_snapshot_state

    roots = [tmp_path / name for name in ('alpha', 'beta', 'gamma')]
    for root in roots:
        root.mkdir()
    config = DashboardConfig(project_root=roots[0], known_project_roots=roots[1:])
    monkeypatch.setattr(
        'dashboard.data.active_tasks.fetch_task_runtime', AsyncMock(return_value={}),
    )
    canned = CannedMCP(
        rows=[_raw_row(task_id, status) for task_id, status in _PAIRS],
        status_map=dict(_PAIRS),
        status_page_size=2000,
    )
    monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', canned)
    width_one = replace(TASKS_BUDGET, root_concurrency=1)

    await collect_tasks_with_counts(dummy_client, config, now=NOW, budget=width_one)
    reset_task_snapshot_state()
    canned.calls.clear()
    canned.fail_when = lambda call: True
    _rows, snapshots = await collect_tasks_with_counts(
        dummy_client, config, now=NOW, budget=width_one,
    )

    read_roots = {call['args']['project_root'] for call in canned.calls}
    assert read_roots == {str(root) for root in roots}, (
        f'after the reset the render read only {sorted(read_roots)}; a root '
        'that issued no call was served the unit the first render cached'
    )
    census_states = {label: unit.census.state for label, unit in snapshots.items()}
    assert census_states == dict.fromkeys(('alpha', 'beta', 'gamma'), DatumState.UNKNOWN), (
        f'census states {census_states}: a STALE census is the first render\'s '
        'last good surviving the reset'
    )
    assert canned.calls[0]['args']['project_root'] == str(roots[0]), (
        f'the first read after the reset was for {canned.calls[0]["args"]["project_root"]}; '
        'the rotation must restart at the primary root, not continue at beta'
    )
