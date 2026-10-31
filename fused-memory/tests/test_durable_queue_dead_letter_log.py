"""The durable write queue's dead-letter WARNING.

Once a dead-lettered row is deleted, the WARNING is the triager's only record
of the abandoned write, so it must name the operation, the group_id, the
attempt count, the error and the retry-budget classification.

pytest runs with ``asyncio_mode='strict'``, so every async test needs an
explicit ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import poll_until

from fused_memory.services.durable_queue import DurableWriteQueue


async def _poll_until_dead(
    q: DurableWriteQueue,
    *,
    group_id: str | None = None,
    expected_dead: int,
    timeout: float = 20.0,
) -> None:
    """Poll queue stats until at least *expected_dead* items reach status='dead'.

    Mirrors ``test_durable_queue.py::_poll_until_dead``. Fixed sleeps are unsafe here:
    the suite runs under ``-n auto --dist loadgroup`` with a 60s per-test
    timeout that kills the whole worker, and the generous 20s budget means a
    regression fails on the observed COUNT rather than by timing out.
    """
    last_counts: dict = {}

    async def _dead_count_reached():
        stats = await q.get_stats(group_id=group_id)
        last_counts.update(stats['counts'])
        return stats['counts'].get('dead', 0) >= expected_dead

    try:
        await poll_until(_dead_count_reached, timeout=timeout, interval=0.05)
    except AssertionError as exc:
        raise AssertionError(
            f'Timed out waiting for {expected_dead} dead item(s) '
            f'(group_id={group_id!r}); last counts={last_counts}'
        ) from exc


def _queue(tmp_path, execute, **overrides) -> DurableWriteQueue:
    """Plain budget 5, extended transient budget 12, sub-second backoff."""
    # Annotated: without it pyright joins the heterogeneous literal values into
    # a single narrow type and rejects the ** unpacking below.
    kwargs: dict[str, Any] = dict(
        data_dir=tmp_path / 'queue',
        execute_write=execute,
        workers_per_group=1,
        semaphore_limit=5,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_delay_seconds=0.05,
        write_timeout_seconds=2.0,
        transient_max_attempts=12,
    )
    kwargs.update(overrides)
    return DurableWriteQueue(**kwargs)


@pytest.mark.asyncio
async def test_dead_letter_warning_names_operation_and_group(tmp_path, caplog):
    execute = AsyncMock(side_effect=RuntimeError('boom'))
    q = _queue(tmp_path, execute)
    await q.initialize()
    try:
        with caplog.at_level(
            logging.WARNING, logger='fused_memory.services.durable_queue'
        ):
            await q.enqueue(
                group_id='proj-alpha', operation='add_episode',
                payload={'content': 'c', 'group_id': 'proj-alpha', 'name': 'ep'},
            )
            await _poll_until_dead(
                q, group_id='proj-alpha', expected_dead=1, timeout=20.0
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'add_episode' in r.getMessage() and 'proj-alpha' in r.getMessage()
            for r in warnings
        ), (
            'expected the dead-letter WARNING to name both the operation and '
            'the group_id; got: '
            f'{[r.getMessage() for r in warnings]}'
        )
        assert any(
            '5 attempts' in r.getMessage() and 'RuntimeError' in r.getMessage()
            for r in warnings
        ), (
            'expected the dead-letter WARNING to carry the attempt count and '
            f'the error text; got: {[r.getMessage() for r in warnings]}'
        )
    finally:
        await q.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('exc', 'label', 'budget'),
    [
        (ConnectionResetError('reset'), 'transient', 3),
        (RuntimeError('boom'), 'normal', 5),
    ],
    ids=['transient', 'normal'],
)
async def test_dead_letter_warning_labels_the_budget_it_exhausted(
    tmp_path, caplog, exc, label, budget
):
    """One WARNING pairs the classification label with the attempt count of
    the budget the item actually used up, so the label cannot name one budget
    while the item died on the other."""
    q = _queue(tmp_path, AsyncMock(side_effect=exc), transient_max_attempts=3)
    await q.initialize()
    try:
        with caplog.at_level(
            logging.WARNING, logger='fused_memory.services.durable_queue'
        ):
            await q.enqueue(
                group_id='proj-alpha', operation='add_episode',
                payload={'content': 'c', 'group_id': 'proj-alpha', 'name': 'ep'},
            )
            await _poll_until_dead(
                q, group_id='proj-alpha', expected_dead=1, timeout=20.0
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            f'{budget} attempts' in r.getMessage()
            and f'[{label}]' in r.getMessage()
            for r in warnings
        ), (
            f'expected one dead-letter WARNING carrying both "{budget} attempts" '
            f'and [{label}]; got: {[r.getMessage() for r in warnings]}'
        )
    finally:
        await q.close()
