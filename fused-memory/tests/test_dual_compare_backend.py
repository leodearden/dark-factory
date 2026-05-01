"""Divergence-detection tests for :class:`DualCompareBackend`.

Pairs a real :class:`SqliteTaskBackend` with a stub backend whose
return values can be perturbed, then asserts that the comparator logs
the divergence and still surfaces the primary's outcome unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.backends.dual_compare_backend import DualCompareBackend
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.taskmaster_types import TaskmasterError
from fused_memory.config.schema import TaskmasterConfig


def _stub_backend():
    s = AsyncMock()
    s.connected = True
    s.restart_count = 1
    s.start = AsyncMock()
    s.close = AsyncMock()
    s.ensure_connected = AsyncMock()
    s.is_alive = AsyncMock(return_value=(True, None))
    return s


@pytest_asyncio.fixture
async def primary(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path), backend_mode='sqlite')
    b = SqliteTaskBackend(cfg)
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def project_root(tmp_path):
    return str(tmp_path / 'proj')


@pytest.mark.asyncio
async def test_zero_divergence_when_secondary_agrees(primary, project_root):
    secondary = _stub_backend()
    # Have the secondary mirror a successful add_task wire shape.
    secondary.add_task = AsyncMock(return_value={'id': '1', 'message': 'ok'})
    wrapper = DualCompareBackend(primary, secondary)

    out = await wrapper.add_task(project_root=project_root, title='a')

    assert out['id'] == '1'
    assert wrapper.divergence_count == 0


@pytest.mark.asyncio
async def test_divergence_counted_when_secondary_returns_failure(
    primary, project_root, caplog,
):
    secondary = _stub_backend()
    secondary.add_task = AsyncMock(side_effect=TaskmasterError(
        'TASKMASTER_TOOL_ERROR', 'simulated mismatch',
    ))
    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        # Primary succeeds; secondary throws — outcome differs.
        await wrapper.add_task(project_root=project_root, title='a')

    assert wrapper.divergence_count == 1
    assert any('dual_compare.divergence' in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_get_tasks_compares_normalised_tree(
    primary, project_root, caplog,
):
    await primary.add_task(project_root=project_root, title='a')
    secondary = _stub_backend()
    # Wrong title — divergence after normalisation.
    secondary.get_tasks = AsyncMock(return_value={'tasks': [
        {'id': '1', 'title': 'WRONG', 'status': 'pending', 'priority': 'medium',
         'description': '', 'details': '', 'dependencies': [], 'subtasks': []},
    ]})
    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        out = await wrapper.get_tasks(project_root=project_root)

    assert out['tasks'][0]['title'] == 'a'  # primary wins
    assert wrapper.divergence_count == 1


@pytest.mark.asyncio
async def test_volatile_fields_never_trigger_divergence(
    primary, project_root, caplog,
):
    """``updatedAt`` differs by definition between Taskmaster and SQLite —
    the comparator must strip it before comparing."""
    await primary.add_task(project_root=project_root, title='a')
    primary_listing = await primary.get_tasks(project_root=project_root)

    skew = []
    for task in primary_listing['tasks']:
        copy = dict(task)
        copy['updatedAt'] = '1970-01-01T00:00:00.000Z'
        skew.append(copy)
    secondary = _stub_backend()
    secondary.get_tasks = AsyncMock(return_value={'tasks': skew})
    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        await wrapper.get_tasks(project_root=project_root)

    assert wrapper.divergence_count == 0


@pytest.mark.asyncio
async def test_primary_exception_propagates_after_logging_divergence(
    primary, project_root, caplog,
):
    secondary = _stub_backend()
    # Secondary "succeeds" — primary will raise (no such task).
    secondary.get_task = AsyncMock(return_value={'id': 99, 'title': 'phantom'})
    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        with pytest.raises(TaskmasterError):
            await wrapper.get_task('99', project_root=project_root)

    assert wrapper.divergence_count == 1


@pytest.mark.asyncio
async def test_primary_label_routes_via_dual_compare_primary(tmp_path):
    """`dual_compare_primary` controls which side the wrapper serves."""
    cfg = TaskmasterConfig(project_root=str(tmp_path), backend_mode='dual_compare')
    project_root = str(tmp_path / 'proj')
    primary = SqliteTaskBackend(cfg)
    await primary.start()
    secondary = _stub_backend()
    secondary.add_task = AsyncMock(return_value={'id': '999', 'message': 'ok'})

    wrapper = DualCompareBackend(
        primary=secondary, secondary=primary,
        primary_label='taskmaster', secondary_label='sqlite',
    )
    out = await wrapper.add_task(project_root=project_root, title='a')
    # When the secondary is sqlite, the wire id from the primary (the stub)
    # is what callers see.
    assert out['id'] == '999'
    await primary.close()


@pytest.mark.asyncio
async def test_config_property_proxies_to_primary(primary, project_root):
    """``wrapper.config`` proxies primary's config so callers that still
    reach into ``backend.config`` (rather than carrying their own) keep
    working — closes the AttributeError silent-disable hole that kept the
    curator off the entire soak."""
    secondary = _stub_backend()
    wrapper = DualCompareBackend(primary, secondary)
    assert wrapper.config is primary.config
