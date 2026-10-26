"""The scheduler payload states whether each lock's holder is still a live task.

``holder_live`` is how the Scheduler heatmap tells a DEPARTED holder (a lock no
live task can release, with a waiter queued behind it) from an ordinary held
module. It is three-state: ``None`` means unknown, never departed, so a caller
that supplies no live set cannot make every held module look abandoned.

The contract is asserted through the public collector, ``collect_scheduler_state``.
Two states that collector never produces are reached through the helpers that
compute them: unknown liveness, and the ``(project, path)`` merge of two module
entries — every configured root carries its own label
(``dashboard/project_labels.py::ProjectLabels``) and emits each path once, so no
two of the collector's entries share a key.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from dashboard.config import DashboardConfig
from dashboard.data.scheduler import (
    _merge_module_lists,
    _module_contention_counts,
    collect_scheduler_state,
)

LABEL = 'proj'


def _active_task(task_id: str, files: list[str]) -> dict:
    """One ``collect_tasks_with_counts`` row in project LABEL."""
    return {
        'id': f'{LABEL}/T-{task_id}',
        'project': LABEL,
        'title': f'task {task_id}',
        'priority': 'medium',
        'status': 'pending',
        'started': 0,
        'meta_files': files,
    }


def _config(tmp_path: Path, side: str) -> DashboardConfig:
    """The one root ``tmp_path/<side>/proj``, labelled LABEL."""
    root = tmp_path / side / LABEL
    root.mkdir(parents=True)
    return DashboardConfig(project_root=root)


async def _collect_modules(
    client, config: DashboardConfig, holders_by_side: dict[str, dict], active: list[dict],
) -> dict[str, dict]:
    """Run the collector, each root's snapshot holding ``holders_by_side[<side>]``."""

    async def mcp_tool_call(_client, _url, tool, args):
        if tool != 'get_scheduler_state':
            return []
        return {
            'skip_counts': {},
            'parks': {},
            'effective_priorities': {},
            'pin_queue': [],
            'overrides': {},
            'current_holders': holders_by_side[Path(args['project_root']).parent.name],
            'snapshot_at': '2026-01-01T00:00:00+00:00',
        }

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', side_effect=mcp_tool_call),
        patch(
            'dashboard.data.scheduler.collect_tasks_with_counts',
            AsyncMock(return_value=(active, {})),
        ),
    ):
        _rows, modules, *_ = await collect_scheduler_state(client, config)
    return {m['path']: m for m in modules}


async def test_holder_in_the_live_set_is_live(dummy_client, tmp_path):
    modules = await _collect_modules(
        dummy_client, _config(tmp_path, 'solo'),
        {'solo': {'src/a.py': '1'}},
        [_active_task('1', ['src/a.py'])],
    )
    assert modules['src/a.py']['holder_live'] is True


async def test_holder_absent_from_the_live_set_has_departed_and_only_the_waiter_counts(
    dummy_client, tmp_path,
):
    modules = await _collect_modules(
        dummy_client, _config(tmp_path, 'solo'),
        {'solo': {'src/a.py': '9'}},
        [_active_task('2', ['src/a.py'])],
    )
    entry = modules['src/a.py']
    assert entry['holder'] == '9'
    assert entry['contention'] == 1
    assert entry['holder_live'] is False


async def test_module_without_a_holder_has_no_holder_liveness(dummy_client, tmp_path):
    modules = await _collect_modules(
        dummy_client, _config(tmp_path, 'solo'),
        {'solo': {}},
        [_active_task('1', ['src/a.py'])],
    )
    assert modules['src/a.py']['holder_live'] is None


def _module_entry(holders: dict[str, str], waiter: str) -> dict:
    """One project's ``src/a.py`` entry: *waiter* contends it, *holders* may hold it."""
    [entry] = _module_contention_counts(
        [{'task_id': waiter, 'lock_set': ['src/a.py']}],
        holders,
        project=LABEL,
        live_task_ids={waiter},
    )
    return entry


def test_merge_carries_holder_live_with_the_holder_it_describes():
    [entry] = _merge_module_lists(
        [_module_entry({}, waiter='2')],
        [_module_entry({'src/a.py': '9'}, waiter='2')],
    )
    assert entry['contention'] == 2, 'both lists contributed to one merged entry'
    assert entry['holder'] == '9'
    assert entry['holder_live'] is False


def test_merge_keeps_holder_live_when_a_later_list_has_no_holder():
    [entry] = _merge_module_lists(
        [_module_entry({'src/a.py': '9'}, waiter='2')],
        [_module_entry({}, waiter='2')],
    )
    assert entry['contention'] == 2, 'both lists contributed to one merged entry'
    assert entry['holder'] == '9'
    assert entry['holder_live'] is False


def test_unknown_liveness_never_reads_as_departed():
    modules = _module_contention_counts(
        [{'task_id': '1', 'lock_set': ['m']}],
        {'m': '9'},
        project='p',
    )
    assert modules[0]['holder_live'] is None
