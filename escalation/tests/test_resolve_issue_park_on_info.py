"""resolve_issue refuses ``action='park'`` on an info-severity target (task 6578).

An info record never pins its task (``escalation/src/escalation/pins.py::classify_pins``
link 1), so a park cannot hold the task it blocks. The tool refuses with code
``park_on_info`` and persists nothing.

Driven only through the public tool surface: in-process via ``tool.fn`` (the
header-less path) and, for the capability precedence, over real HTTP. Observed
through ``queue.get`` and the resolve callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _escalation_http import escalation_http_call
from _escalation_seed import seed_escalation

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

pytestmark = pytest.mark.asyncio


async def _resolve_issue(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('resolve_issue')
    return tool.fn(**kwargs)


def _seed(queue: EscalationQueue, *, level: int, task_id: str, **kw: Any) -> Escalation:
    return seed_escalation(
        queue, level=level, task_id=task_id,
        summary=f'park-on-info test escalation (level={level})', **kw,
    )


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def server(queue: EscalationQueue):
    return create_server(queue)


@pytest.fixture
def fired(queue: EscalationQueue) -> list[Escalation]:
    calls: list[Escalation] = []
    queue.set_resolve_callback(calls.append)
    return calls


def _assert_park_on_info_refusal(result: dict[str, Any], rec: Escalation) -> None:
    assert result.get('code') == 'park_on_info', result
    assert 'error' in result
    assert 'hint' in result
    assert result['escalation_id'] == rec.id
    assert result['task_id'] == rec.task_id
    assert rec.id in result['error']
    assert rec.task_id in result['error']


class TestParkOnInfoRefused:
    async def test_info_l2_park_is_refused_with_structured_ids(self, queue, server, fired):
        rec = _seed(queue, level=2, task_id='6578-a', severity='info')

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='hold it', action='park',
        )

        _assert_park_on_info_refusal(result, rec)
        assert 'deferred' in result['hint']
        assert 'close_only' in result['hint']

    @pytest.mark.parametrize('severity', ['INFO', ' Info '])
    async def test_unnormalised_info_severity_is_refused(
        self, queue, server, fired, severity,
    ):
        rec = _seed(queue, level=2, task_id='6578-b', severity=severity)

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='hold it', action='park',
        )

        _assert_park_on_info_refusal(result, rec)

    async def test_info_l1_park_is_refused_and_not_promoted(self, queue, server, fired):
        rec = _seed(queue, level=1, task_id='6578-c', severity='info')

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='hold it', action='park',
        )

        _assert_park_on_info_refusal(result, rec)
        after = queue.get(rec.id)
        assert after is not None
        assert after.level == 1

    async def test_refusal_persists_nothing(self, queue, server, fired):
        rec = _seed(queue, level=2, task_id='6578-d', severity='info')
        before = queue.get(rec.id)
        assert before is not None
        snapshot = before.to_dict()

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='hold it', action='park',
        )

        assert result.get('code') == 'park_on_info'
        after = queue.get(rec.id)
        assert after is not None
        assert after.to_dict() == snapshot
        assert after.status == 'pending'
        assert after.level == 2
        assert after.resolution_action is None
        assert after.resolution is None
        assert after.resolved_by is None
        assert after.resolved_at is None
        assert (queue.queue_dir / f'{rec.id}.json').exists()

    async def test_refusal_fires_no_resolve_callback(self, queue, server, fired):
        member = _seed(queue, level=1, task_id='6578-e-member', severity='info')
        head = _seed(
            queue, level=2, task_id='6578-e', severity='info', members=[member.id],
        )

        result = await _resolve_issue(
            server, escalation_id=head.id, resolution='hold it', action='park',
        )

        assert result.get('code') == 'park_on_info'
        assert fired == []

    async def test_already_parked_info_specimen_is_refused(self, queue, server, fired):
        rec = _seed(queue, level=1, task_id='6578-f', severity='info')
        queue.park(rec.id, 'held', resolved_by='watcher')
        parked = queue.get(rec.id)
        assert parked is not None
        assert parked.status == 'pending'
        assert parked.level == 2
        assert parked.resolution_action == 'park'
        snapshot = parked.to_dict()
        fired.clear()

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='hold it again', action='park',
        )

        _assert_park_on_info_refusal(result, rec)
        after = queue.get(rec.id)
        assert after is not None
        assert after.to_dict() == snapshot
        assert fired == []


class TestGateDoesNotOverReach:
    @pytest.mark.parametrize(
        'severity', ['blocking', 'critical', 'urgent', 'not-a-severity'],
    )
    async def test_non_info_l2_still_parks(self, queue, server, fired, severity):
        """'not-a-severity' pins the predicate as ``== 'info'``: pins.py link 2
        fails an unknown severity safe to pinning, so that park does hold."""
        rec = _seed(queue, level=2, task_id='6578-g', severity=severity)

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='hold it', action='park',
        )

        assert 'code' not in result, result
        assert result['status'] == 'pending'
        assert result['resolution_action'] == 'park'
        assert result['level'] == 2
        assert [e.id for e in fired] == [rec.id]

    async def test_blocking_l1_park_still_promotes(self, queue, server, fired):
        rec = _seed(queue, level=1, task_id='6578-h', severity='blocking')

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='hold it', action='park',
        )

        assert 'code' not in result, result
        after = queue.get(rec.id)
        assert after is not None
        assert after.level == 2
        assert after.resolution_action == 'park'

    async def test_non_info_cluster_with_info_member_still_parks(
        self, queue, server, fired,
    ):
        """Clusters are not refused: the gate reads only the target's severity.

        The edge this leaves open (an info member on another task is re-pended
        after the park) is follow-up ticket tkt_0RW39B98R5X6JW9QJJ6F5FRCW1.
        """
        blocking_member = _seed(queue, level=1, task_id='6578-i-a', severity='blocking')
        info_member = _seed(queue, level=1, task_id='6578-i-b', severity='info')
        head = _seed(
            queue, level=2, task_id='6578-i-a', severity='blocking',
            members=[blocking_member.id, info_member.id],
        )

        result = await _resolve_issue(
            server, escalation_id=head.id, resolution='hold it', action='park',
        )

        assert 'code' not in result, result
        assert result['status'] == 'pending'
        assert result['resolution_action'] == 'park'
        fired_ids = {e.id for e in fired}
        assert {head.id, blocking_member.id, info_member.id} <= fired_ids

    async def test_info_l2_close_only_still_dismisses(self, queue, server, fired):
        rec = _seed(queue, level=2, task_id='6578-j1', severity='info')

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='closing', action='close_only',
        )

        assert 'code' not in result, result
        assert result['status'] == 'dismissed'
        assert not (queue.queue_dir / f'{rec.id}.json').exists()
        assert [e.id for e in fired] == [rec.id]

    async def test_info_l2_resume_still_resolves(self, queue, server, fired):
        rec = _seed(queue, level=2, task_id='6578-j2', severity='info')

        result = await _resolve_issue(
            server, escalation_id=rec.id, resolution='go on', action='resume',
        )

        assert 'code' not in result, result
        assert result['status'] == 'resolved'
        assert [e.id for e in fired] == [rec.id]


class TestCapabilityGatePrecedence:
    @pytest.fixture
    def served_info_l2(self, tmp_path, serve_escalation_mcp):
        """A running server holding one pending info L2.

        A sync fixture because the server factory runs its own event loop.
        """
        base_url, _port, queue = serve_escalation_mcp(tmp_path / 'esc')
        rec = _seed(queue, level=2, task_id='6578-k', severity='info')
        return base_url, queue, rec

    async def test_level_forbidden_is_reported_before_park_on_info(
        self, served_info_l2,
    ):
        base_url, queue, rec = served_info_l2
        before = queue.get(rec.id)
        assert before is not None
        snapshot = before.to_dict()

        result = await escalation_http_call(
            base_url, 'resolve_issue', levels='0,1',
            escalation_id=rec.id, resolution='r', action='park',
        )

        assert result.get('code') == 'level_forbidden', result
        after = queue.get(rec.id)
        assert after is not None
        assert after.to_dict() == snapshot

    async def test_permitted_connection_gets_park_on_info(self, served_info_l2):
        base_url, queue, rec = served_info_l2
        before = queue.get(rec.id)
        assert before is not None
        snapshot = before.to_dict()

        result = await escalation_http_call(
            base_url, 'resolve_issue', levels='0,1,2',
            escalation_id=rec.id, resolution='r', action='park',
        )

        _assert_park_on_info_refusal(result, rec)
        after = queue.get(rec.id)
        assert after is not None
        assert after.to_dict() == snapshot
