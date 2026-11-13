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
from _fm_helpers import make_retry_budget_queue
from _fm_polling import poll_until_dead
from _queued_write_fixtures import queued


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('exc', 'label', 'budget'),
    [
        (ConnectionResetError('reset'), 'transient', 3),
        (RuntimeError('boom'), 'normal', 5),
    ],
    ids=['transient', 'normal'],
)
async def test_one_dead_letter_warning_carries_every_triage_field(
    tmp_path, caplog, exc, label, budget
):
    """All five fields sit on ONE record, so a triager never has to stitch
    lines together, and the label names the budget the item actually used up."""
    q = make_retry_budget_queue(
        tmp_path, AsyncMock(side_effect=exc), transient_max_attempts=3
    )
    await q.initialize()
    try:
        with caplog.at_level(
            logging.WARNING, logger='fused_memory.services.durable_queue'
        ):
            await q.enqueue(queued('add_episode', {'content': 'c', 'group_id': 'proj-alpha', 'name': 'ep'}, group_id='proj-alpha'))
            await poll_until_dead(q, group_id='proj-alpha', expected_dead=1)

        dead_letter_lines = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and 'dead-lettered' in r.getMessage()
        ]
        assert len(dead_letter_lines) == 1, dead_letter_lines
        (line,) = dead_letter_lines
        for field in (
            'add_episode',
            'proj-alpha',
            f'{budget} attempts',
            type(exc).__name__,
            f'[{label}]',
        ):
            assert field in line, f'{field!r} missing from {line!r}'
    finally:
        await q.close()
