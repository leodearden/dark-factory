"""The durable write queue's dead-letter WARNING.

Once a dead-lettered row is deleted, the WARNING is the triager's only record
of the abandoned write, so it must name the operation, the group_id, the
attempt count, the error and the retry-budget classification.

pytest runs with ``asyncio_mode='strict'``, so every async test needs an
explicit ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import make_retry_budget_queue, poll_until_dead


@pytest.mark.asyncio
async def test_dead_letter_warning_names_operation_and_group(tmp_path, caplog):
    execute = AsyncMock(side_effect=RuntimeError('boom'))
    q = make_retry_budget_queue(tmp_path, execute)
    await q.initialize()
    try:
        with caplog.at_level(
            logging.WARNING, logger='fused_memory.services.durable_queue'
        ):
            await q.enqueue(
                group_id='proj-alpha', operation='add_episode',
                payload={'content': 'c', 'group_id': 'proj-alpha', 'name': 'ep'},
            )
            await poll_until_dead(q, group_id='proj-alpha', expected_dead=1)

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
    q = make_retry_budget_queue(
        tmp_path, AsyncMock(side_effect=exc), transient_max_attempts=3
    )
    await q.initialize()
    try:
        with caplog.at_level(
            logging.WARNING, logger='fused_memory.services.durable_queue'
        ):
            await q.enqueue(
                group_id='proj-alpha', operation='add_episode',
                payload={'content': 'c', 'group_id': 'proj-alpha', 'name': 'ep'},
            )
            await poll_until_dead(q, group_id='proj-alpha', expected_dead=1)

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
