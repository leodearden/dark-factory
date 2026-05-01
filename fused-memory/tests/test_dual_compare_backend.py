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


@pytest.mark.asyncio
async def test_ensure_connected_pokes_both_sides(primary, caplog):
    """``ensure_connected`` must drive *both* backends so a silently-broken
    secondary connection surfaces as a counter increment + warning log
    rather than going unobserved."""
    secondary = _stub_backend()
    secondary.ensure_connected = AsyncMock(
        side_effect=RuntimeError('secondary down'),
    )
    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        # Primary still succeeds → no exception propagates.
        await wrapper.ensure_connected()

    secondary.ensure_connected.assert_awaited_once()
    assert wrapper.secondary_health_failures == 1
    assert any(
        'secondary.ensure_connected failed' in m for m in caplog.messages
    )


@pytest.mark.asyncio
async def test_ensure_connected_propagates_primary_failure(caplog):
    """A primary failure must still surface — secondary's outcome is
    incidental on the failure path."""
    primary = _stub_backend()
    primary.ensure_connected = AsyncMock(
        side_effect=RuntimeError('primary down'),
    )
    secondary = _stub_backend()
    wrapper = DualCompareBackend(primary, secondary)

    with pytest.raises(RuntimeError, match='primary down'):
        await wrapper.ensure_connected()
    assert wrapper.secondary_health_failures == 0


@pytest.mark.asyncio
async def test_post_write_verify_logs_state_drift(primary, project_root, caplog):
    """Read-back verify catches the silent-write-drop case the input-echo
    comparator was structurally blind to.

    Setup: secondary's ``set_task_status`` returns the same wire shape as
    primary (so ``_normalize_set_status`` shows zero divergence) but the
    secondary's ``get_task`` continues to report the OLD status — the
    write was effectively dropped on the secondary side.

    Assert: the post-write verify task fires, finds the divergence,
    increments ``verifies_diverged``, and logs ``verify_divergence``.
    """
    await primary.add_task(project_root=project_root, title='a')

    secondary = _stub_backend()
    secondary.set_task_status = AsyncMock(return_value={'tasks': [
        {'taskId': '1', 'newStatus': 'done'},
    ]})
    # Stale read — drop on the secondary side.
    secondary.get_task = AsyncMock(return_value={
        'id': 1, 'title': 'a', 'status': 'pending', 'priority': 'medium',
        'description': '', 'details': '',
        'dependencies': [], 'subtasks': [],
    })

    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        await wrapper.set_task_status('1', 'done', project_root)
        # Drain in-flight verify tasks.
        for t in list(wrapper._inflight_verifies):
            await t

    assert wrapper.verifies_total == 1
    assert wrapper.verifies_diverged == 1
    assert any(
        'dual_compare.verify_divergence' in m for m in caplog.messages
    ), caplog.messages
    # Per-method counters should also have ticked.
    by_method = wrapper.verifies_by_method.get('set_task_status', {})
    assert by_method == {'total': 1, 'diverged': 1}


@pytest.mark.asyncio
async def test_post_write_verify_no_divergence_when_state_matches(
    primary, project_root, caplog,
):
    """Happy path: when both backends agree on post-write state, the
    verify completes silently and ``verifies_diverged`` stays at zero."""
    await primary.add_task(project_root=project_root, title='a')

    secondary = _stub_backend()
    secondary.set_task_status = AsyncMock(return_value={'tasks': [
        {'taskId': '1', 'newStatus': 'done'},
    ]})
    # Apply the write on the primary first so we can hand the secondary a
    # honest read-back of whatever shape the primary's get_task emits.
    await primary.set_task_status('1', 'done', project_root)
    primary_view = await primary.get_task('1', project_root)
    secondary.get_task = AsyncMock(return_value=dict(primary_view))

    wrapper = DualCompareBackend(primary, secondary)
    await wrapper.set_task_status('1', 'done', project_root)
    for t in list(wrapper._inflight_verifies):
        await t

    assert wrapper.verifies_total == 1
    assert wrapper.verifies_diverged == 0


@pytest.mark.asyncio
async def test_get_task_divergence_logs_diff_only(
    primary, project_root, caplog,
):
    """``get_task`` divergence emits ``task=<id> fields={...}`` listing only
    the differing fields — not the whole 240-char clipped blob."""
    await primary.add_task(project_root=project_root, title='a')
    secondary = _stub_backend()
    # Same id and shape, only ``status`` differs.
    secondary.get_task = AsyncMock(return_value={
        'id': 1, 'title': 'a', 'status': 'in-progress', 'priority': 'medium',
        'description': '', 'details': '', 'testStrategy': '',
        'dependencies': [], 'subtasks': [],
    })

    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        await wrapper.get_task('1', project_root=project_root)

    msgs = [m for m in caplog.messages if 'dual_compare.divergence' in m]
    assert len(msgs) == 1, msgs
    msg = msgs[0]
    assert 'task=1' in msg
    assert 'fields={' in msg
    assert 'status:' in msg
    # Only ``status`` differed — not ``title``/``priority``/etc.
    assert 'title:' not in msg
    assert 'priority:' not in msg


@pytest.mark.asyncio
async def test_get_tasks_divergence_emits_one_line_per_task(
    primary, project_root, caplog,
):
    """``get_tasks`` walks both arrays keyed by id and emits one log line
    per task that differs — not one giant unreadable blob."""
    await primary.add_task(project_root=project_root, title='alpha')
    await primary.add_task(project_root=project_root, title='beta')
    primary_listing = await primary.get_tasks(project_root=project_root)

    # Skew task 2's status, leave task 1 alone.
    skew = []
    for t in primary_listing['tasks']:
        copy = dict(t)
        if str(copy.get('id')) == '2':
            copy['status'] = 'done'
        skew.append(copy)
    secondary = _stub_backend()
    secondary.get_tasks = AsyncMock(return_value={'tasks': skew})

    wrapper = DualCompareBackend(primary, secondary)

    with caplog.at_level('WARNING'):
        await wrapper.get_tasks(project_root=project_root)

    msgs = [m for m in caplog.messages if 'dual_compare.divergence' in m]
    # Exactly one line — only task 2 differed.
    assert len(msgs) == 1, msgs
    assert 'task=2' in msgs[0]
    assert 'task=1' not in msgs[0]
    assert 'status:' in msgs[0]


@pytest.mark.asyncio
async def test_post_write_verify_remove_tasks_expects_not_found(
    primary, project_root, caplog,
):
    """``remove_tasks`` inverts the contract: both sides should now raise
    ``TaskmasterError`` on the read-back. If secondary still returns
    the task, that's a divergence."""
    await primary.add_task(project_root=project_root, title='a')

    secondary = _stub_backend()
    secondary.remove_tasks = AsyncMock(return_value={'successful': 1, 'failed': 0})
    # Stale: secondary still has the task after remove.
    secondary.get_task = AsyncMock(return_value={
        'id': 1, 'title': 'a', 'status': 'pending', 'priority': 'medium',
        'description': '', 'details': '', 'dependencies': [], 'subtasks': [],
    })

    wrapper = DualCompareBackend(primary, secondary)
    with caplog.at_level('WARNING'):
        await wrapper.remove_tasks(['1'], project_root)
        for t in list(wrapper._inflight_verifies):
            await t

    assert wrapper.verifies_total == 1
    assert wrapper.verifies_diverged == 1


@pytest.mark.asyncio
async def test_remove_tasks_multi_id_zero_divergence_when_secondary_agrees(
    primary, project_root,
):
    """Multi-id parity: secondary agrees on the count → no divergence."""
    await primary.add_task(project_root=project_root, title='a')
    await primary.add_task(project_root=project_root, title='b')

    secondary = _stub_backend()
    # Wire shape primary returns: successful=2, failed=0.
    secondary.remove_tasks = AsyncMock(return_value={
        'successful': 2, 'failed': 0,
    })

    wrapper = DualCompareBackend(primary, secondary)
    out = await wrapper.remove_tasks(['1', '2'], project_root)

    assert out['successful'] == 2
    assert wrapper.divergence_count == 0
    # Verify the secondary received the list, not a comma-string.
    sent_args, _ = secondary.remove_tasks.call_args
    assert sent_args[0] == ['1', '2']


@pytest.mark.asyncio
async def test_remove_tasks_multi_id_divergence_on_count_mismatch(
    primary, project_root, caplog,
):
    """Multi-id divergence: secondary disagrees on the successful count."""
    await primary.add_task(project_root=project_root, title='a')
    await primary.add_task(project_root=project_root, title='b')

    secondary = _stub_backend()
    secondary.remove_tasks = AsyncMock(return_value={
        'successful': 1, 'failed': 1,
    })

    wrapper = DualCompareBackend(primary, secondary)
    with caplog.at_level('WARNING'):
        await wrapper.remove_tasks(['1', '2'], project_root)

    assert wrapper.divergence_count == 1
    assert any('dual_compare.divergence' in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_dual_compare_writes_per_side_backend_ops(primary, project_root, tmp_path):
    """When a write_journal is wired, ``_dispatch_pair`` emits one
    ``backend_op`` per physical backend, both linked to the parent
    write_op_id propagated through the contextvar by the interceptor.

    Simulates the interceptor's contextvar publication so the wrapper
    can be tested in isolation.
    """
    from fused_memory.backends.dual_compare_backend import _current_write_op_id
    from fused_memory.services.write_journal import WriteJournal

    secondary = _stub_backend()
    secondary.set_task_status = AsyncMock(return_value={'tasks': [
        {'taskId': '1', 'newStatus': 'done'},
    ]})

    journal = WriteJournal(tmp_path / 'wj_dual')
    await journal.initialize()
    try:
        await primary.add_task(project_root=project_root, title='a')
        wrapper = DualCompareBackend(primary, secondary)
        wrapper.set_write_journal(journal)

        wo_id = 'test-write-op-id'
        token = _current_write_op_id.set(wo_id)
        try:
            await wrapper.set_task_status('1', 'done', project_root)
        finally:
            _current_write_op_id.reset(token)
        # Drain post-write verify so close() isn't racing it.
        for t in list(wrapper._inflight_verifies):
            await t

        async with journal._db.execute(
            'SELECT backend FROM backend_ops WHERE write_op_id = ?',
            (wo_id,),
        ) as cur:
            backends = sorted(row[0] for row in await cur.fetchall())

        # Two rows — one per physical backend.
        assert 'sqlite_task_backend' in backends
        # Secondary stub's class is AsyncMock — labelled by lowercased class.
        assert 'asyncmock' in backends or 'taskmaster' in backends
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_dispatch_cancellation_drains_and_logs_divergence(caplog):
    """Outer cancellation must NOT silently swallow divergence: the inner
    gather is shielded, the caller still sees ``CancelledError``, and a
    background drain runs ``_compare`` once the inner settles.

    Pre-fix this was the headline soak bug — sqlite's local commit fired
    before cancel propagated, tm's stdio + tasks.json rewrite often did
    not, and ``_compare`` never ran. The result: zero ``set_task_status``
    divergences logged in the storm window despite confirmed state drift.
    """
    import asyncio

    primary = _stub_backend()
    secondary = _stub_backend()

    # Secondary returns immediately; primary is slow so cancellation can
    # land mid-flight.
    secondary.set_task_status = AsyncMock(return_value={'tasks': [
        {'taskId': '1', 'newStatus': 'done'},
    ]})

    async def slow_primary(*a, **k):
        await asyncio.sleep(0.05)
        return {'tasks': [{'taskId': '1', 'newStatus': 'in-progress'}]}

    primary.set_task_status = AsyncMock(side_effect=slow_primary)

    wrapper = DualCompareBackend(primary, secondary)

    async def caller():
        await wrapper.set_task_status('1', 'done', '/proj')

    task = asyncio.create_task(caller())
    # Let both inner subtasks start, then cancel the caller.
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert wrapper.cancelled_dispatch_count == 1
    # Drain should have spawned and still be in flight (or just settled).
    drains = list(wrapper._inflight_drains)
    assert drains  # registered

    with caplog.at_level('WARNING'):
        # Wait for inner gather + drain to settle.
        if drains:
            await asyncio.gather(*drains, return_exceptions=True)

    # Once settled, the comparator must have run and logged the
    # in-progress vs done divergence.
    assert wrapper.divergence_count == 1
    assert any('dual_compare.divergence' in m for m in caplog.messages), caplog.messages
