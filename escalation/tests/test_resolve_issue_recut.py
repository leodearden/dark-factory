"""``recut=True`` qualifies a restart resolution: re-cut the task fresh off main (task 5516).

The flag is legal only with ``action='restart'``. Both chokepoints enforce
that pairing: the ``resolve_issue`` tool refuses with code
``recut_requires_restart`` before any read or gate, and
``EscalationQueue.resolve`` raises ``ValueError`` before its lock. Either way
nothing is persisted. On the applied path the queue stamps ``recut`` inside
its lock, beside ``resolution_action``, so the archived record and the
resolve-callback argument both carry it.

Driven through the public tool surface (``tool.fn``) and the queue's public
``resolve``; observed through ``queue.get`` and the resolve callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _escalation_seed import seed_escalation
from _resolve_issue_call import call_resolve_issue

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

NON_RESTART_ACTIONS = ['resume', 'park', 'abandon', 'close_only']


def _seed(queue: EscalationQueue, *, level: int = 0, task_id: str = '5516') -> Escalation:
    return seed_escalation(
        queue, level=level, task_id=task_id,
        summary=f'recut test escalation (level={level})',
    )


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def server(queue: EscalationQueue):
    return create_server(queue, startup_sweep=False)


@pytest.fixture
def fired(queue: EscalationQueue) -> list[Escalation]:
    calls: list[Escalation] = []
    queue.set_resolve_callback(calls.append)
    return calls


def _stored(queue: EscalationQueue, esc_id: str) -> Escalation:
    stored = queue.get(esc_id)
    assert stored is not None
    return stored


def _assert_restart_recut_applied(
    result: dict[str, Any], queue: EscalationQueue, fired: list[Escalation], esc_id: str,
) -> None:
    assert 'error' not in result, result
    assert result['recut'] is True
    stored = _stored(queue, esc_id)
    assert stored.status == 'resolved'
    assert stored.recut is True
    assert stored.resolution_action == 'restart'
    assert [(e.id, e.recut, e.resolution_action) for e in fired] == [
        (esc_id, True, 'restart'),
    ], 'the harness reads recut off the resolve-callback argument'


@pytest.mark.asyncio
class TestResolveIssueRecut:
    async def test_restart_with_recut_stamps_the_flag(self, queue, server, fired):
        rec = _seed(queue)

        result = await call_resolve_issue(
            server, escalation_id=rec.id, resolution='r', action='restart', recut=True,
        )

        _assert_restart_recut_applied(result, queue, fired, rec.id)

    async def test_plain_restart_does_not_recut(self, queue, server, fired):
        rec = _seed(queue)

        result = await call_resolve_issue(
            server, escalation_id=rec.id, resolution='r', action='restart',
        )

        assert 'error' not in result, result
        assert _stored(queue, rec.id).recut is False
        assert [e.recut for e in fired] == [False]

    @pytest.mark.parametrize('action', NON_RESTART_ACTIONS)
    async def test_recut_with_any_other_action_is_refused_and_persists_nothing(
        self, queue, server, fired, action: str,
    ):
        rec = _seed(queue, level=2 if action == 'park' else 0)

        result = await call_resolve_issue(
            server, escalation_id=rec.id, resolution='r', action=action, recut=True,
        )

        assert result.get('code') == 'recut_requires_restart', result
        assert action in result['error']
        assert 'restart' in result['error']
        stored = _stored(queue, rec.id)
        assert stored.status == 'pending'
        assert stored.resolution_action is None
        assert stored.recut is False
        assert fired == []

    async def test_recut_is_not_level_gated(self, queue, server, fired):
        rec = _seed(queue, level=1)

        result = await call_resolve_issue(
            server, escalation_id=rec.id, resolution='r', action='restart', recut=True,
        )

        _assert_restart_recut_applied(result, queue, fired, rec.id)


class TestLegacyRecord:
    def test_a_record_written_without_recut_reads_back_false(self):
        data = Escalation(
            id='esc-5516-1', task_id='5516', agent_role='implementer',
            severity='blocking', category='scope_violation', summary='legacy',
        ).to_dict()
        data.pop('recut', None)

        assert Escalation.from_dict(data).recut is False


class TestQueueResolveRecut:
    def test_restart_with_recut_stamps_archive_and_callback(self, queue, fired):
        rec = _seed(queue)

        esc = queue.resolve(rec.id, 'r', resolution_action='restart', recut=True)

        assert esc is not None and esc.recut is True
        assert _stored(queue, rec.id).recut is True
        assert not (queue.queue_dir / f'{rec.id}.json').exists()
        assert [(e.id, e.recut) for e in fired] == [(rec.id, True)]

    def test_recut_with_a_non_restart_action_raises_before_anything_persists(
        self, queue, fired,
    ):
        rec = _seed(queue)

        with pytest.raises(ValueError, match='restart'):
            queue.resolve(rec.id, 'r', resolution_action='resume', recut=True)

        stored = _stored(queue, rec.id)
        assert stored.status == 'pending'
        assert stored.recut is False
        assert (queue.queue_dir / f'{rec.id}.json').exists()
        assert fired == []

    def test_already_terminal_resolve_does_not_stamp_recut(self, queue, fired):
        rec = _seed(queue)
        queue.resolve(rec.id, 'first', resolution_action='resume')

        queue.resolve(rec.id, 'second', resolution_action='restart', recut=True)

        stored = _stored(queue, rec.id)
        assert stored.recut is False
        assert stored.resolution_action == 'resume'
        assert len(fired) == 1
