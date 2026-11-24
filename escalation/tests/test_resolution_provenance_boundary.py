"""Boundary rows 16-19 of ``plans/instruments-honesty-prd.md``, over real HTTP.

Task 8562, PRD decisions 8 and 9 and §Contract "Resolution provenance". Each
test owns a fresh queue directory served by conftest's
``serve_escalation_mcp``. Identities are sent through
``_escalation_http.escalation_http_call``, and every read-back goes through
``queue.get``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _escalation_http import escalation_http_call
from _escalation_seed import seed_escalation

from escalation.archive import archive_dir_for_date
from escalation.authority import COORDINATOR_IDENTITY, EXECUTOR_IDENTITY
from escalation.classify import resolver_tier
from escalation.models import Escalation
from escalation.queue import EscalationQueue

pytestmark = pytest.mark.asyncio

_TASK = '8562'
# The auto-watcher's wire identity, spelled as the client sends it
# (escalation/tests/test_authority.py pins it the same way).
_AUTO_WATCHER = 'orchestrator-escalation-watcher-auto'


@pytest.fixture
def served(tmp_path: Path, serve_escalation_mcp) -> tuple[str, EscalationQueue]:
    base_url, _port, queue = serve_escalation_mcp(tmp_path / 'escalations')
    return base_url, queue


async def _resolve(
    base_url: str, escalation_id: str, *, identity: str | None = None, **kw: Any,
) -> dict[str, Any]:
    kw.setdefault('resolution', 'closing it')
    return await escalation_http_call(
        base_url, 'resolve_issue', identity=identity, escalation_id=escalation_id, **kw,
    )


def _record(queue: EscalationQueue, escalation_id: str) -> Escalation:
    record = queue.get(escalation_id)
    assert record is not None
    return record


def _snapshot(queue: EscalationQueue, escalation_id: str) -> dict[str, Any]:
    return _record(queue, escalation_id).to_dict()


class TestRow16UnclassifiedLevel2Close:
    """A level-2 close without a resolution_class is refused and changes nothing."""

    @pytest.mark.parametrize('action', ['close_only', 'resume'])
    async def test_the_refusal_is_typed_and_writes_nothing(self, served, action: str):
        base_url, queue = served
        l2 = seed_escalation(queue, level=2, task_id=_TASK)
        before = _snapshot(queue, l2.id)

        result = await _resolve(
            base_url, l2.id, action=action, resolved_by='escalation-watcher',
        )

        assert result['code'] == 'resolution_class_required'
        assert l2.id in result['error']
        assert _snapshot(queue, l2.id) == before

    async def test_a_level_1_record_still_closes(self, served):
        base_url, queue = served
        l1 = seed_escalation(queue, level=1, task_id=_TASK)

        result = await _resolve(
            base_url, l1.id, action='close_only', resolved_by='escalation-watcher',
        )

        assert 'error' not in result
        assert _record(queue, l1.id).status != 'pending'


class TestRow17HumanRulingOnACluster:
    """A human ruling is stamped on the head and inherited by every member."""

    @pytest.mark.parametrize(
        'identity', [None, COORDINATOR_IDENTITY], ids=['header-less', 'coordinator'],
    )
    async def test_head_and_members_carry_the_ruling(self, served, identity: str | None):
        base_url, queue = served
        members = [seed_escalation(queue, level=1, task_id=_TASK) for _ in range(2)]
        head = seed_escalation(
            queue, level=2, task_id=_TASK, members=[m.id for m in members],
        )

        result = await _resolve(
            base_url, head.id, identity=identity, action='close_only',
            resolution_class='actionable', ruled_by_human=True,
            resolved_by='escalation-watcher',
        )

        assert 'error' not in result
        assert result['ruled_by_human'] is True
        closed = _record(queue, head.id)
        assert closed.ruled_by_human is True
        assert closed.resolution_class == 'actionable'
        assert resolver_tier(closed) == 'human-ruled'
        for member in members:
            record = _record(queue, member.id)
            assert record.resolution_class == closed.resolution_class
            assert record.ruled_by_human is True


class TestRow18RuledByHumanForbidden:
    """The executor and the auto-watcher may not assert a human ruling."""

    async def test_the_executor_asserting_a_ruling_on_a_level_2_is_refused_first(self, served):
        base_url, queue = served
        l2 = seed_escalation(queue, level=2, task_id=_TASK)
        before = _snapshot(queue, l2.id)

        result = await _resolve(
            base_url, l2.id, identity=EXECUTOR_IDENTITY, action='close_only',
            resolution_class='actionable', ruled_by_human=True,
        )

        assert result['code'] == 'ruled_by_human_forbidden'
        assert EXECUTOR_IDENTITY in result['error']
        assert _snapshot(queue, l2.id) == before

    async def test_the_auto_watcher_asserting_a_ruling_on_a_level_1_is_refused(self, served):
        base_url, queue = served
        l1 = seed_escalation(queue, level=1, task_id=_TASK)
        before = _snapshot(queue, l1.id)

        result = await _resolve(
            base_url, l1.id, identity=_AUTO_WATCHER, action='close_only',
            resolution_class='benign', ruled_by_human=True,
        )

        assert result['code'] == 'ruled_by_human_forbidden'
        assert _AUTO_WATCHER in result['error']
        assert _snapshot(queue, l1.id) == before

    @pytest.mark.parametrize('flag', [None, False])
    @pytest.mark.parametrize(
        ('identity', 'level'), [(EXECUTOR_IDENTITY, 2), (_AUTO_WATCHER, 1)],
        ids=['executor', 'auto-watcher'],
    )
    async def test_no_ruling_asserted_is_not_refused_by_this_gate(
        self, served, identity: str, level: int, flag: bool | None,
    ):
        base_url, queue = served
        record = seed_escalation(queue, level=level, task_id=_TASK)

        result = await _resolve(
            base_url, record.id, identity=identity, action='close_only',
            resolution_class='benign', ruled_by_human=flag,
        )

        assert result.get('code') != 'ruled_by_human_forbidden'


class TestRow19LegacyRecord:
    """A pre-field archived record reads as not asserted, an attended-seat close."""

    async def test_a_legacy_archived_record_reads_none_and_attended_seat(self, served):
        _base_url, queue = served
        legacy = Escalation(
            id=queue.make_id(_TASK), task_id=_TASK, agent_role='implementer',
            severity='blocking', category='scope_violation', summary='legacy close',
            level=2,
        )
        legacy.status = 'resolved'
        legacy.resolution = 'closed before ruled_by_human existed'
        legacy.resolved_by = 'escalation-watcher'
        legacy.resolved_at = '2026-07-01T00:00:00+00:00'
        payload = legacy.to_dict()
        del payload['ruled_by_human']
        archive = archive_dir_for_date(queue.queue_dir, legacy.resolved_at)
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f'{legacy.id}.json').write_text(json.dumps(payload))

        record = _record(queue, legacy.id)

        assert record.ruled_by_human is None
        assert resolver_tier(record) == 'attended-seat'
