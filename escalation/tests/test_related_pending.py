"""The sideways census a resolve reports: ``related_pending`` (task 4886).

PRD ``docs/prds/truth-propagation-record-mechanics.md`` leaf beta.  Resolving
one record says nothing about its twins: another pending record on the same
task, or a pending L2 clustering a shared member, can carry the same
now-answered question.  The census LISTS them for the resolver to dispose of;
it never closes anything, because a pin is indistinguishable from an answered
question on member evidence alone.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from _pending_tool_fixtures import _file

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.related_pending import related_pending
from escalation.server import _AMENDMENT_TRUNCATION_ANCHOR_TASK_ID, create_server

_ENTRY_KEYS = {'id', 'category', 'severity', 'level', 'same_task', 'shared_member'}


def _esc(
    id: str,  # noqa: A002 — mirrors the Escalation attribute name
    *,
    task_id: str = 'T1',
    level: int = 1,
    members: list[str] | None = None,
    status: str = 'pending',
    **kw: Any,
) -> Escalation:
    """A real ``escalation.models.Escalation``."""
    return Escalation(
        id=id,
        task_id=task_id,
        agent_role='implementer',
        severity=kw.pop('severity', 'blocking'),
        category=kw.pop('category', 'design_concern'),
        summary='s',
        level=level,
        members=list(members or []),
        status=status,
        **kw,
    )


def _by_id(entries: list[Any]) -> dict[str, Any]:
    return {e['id']: e for e in entries}


class TestRelatedPending:
    def test_same_task_record_at_any_level_is_listed(self):
        resolved = _esc('esc-T1-1', level=2, members=['m1'])
        twin_l1 = _esc('esc-T1-2', level=1)
        twin_l0 = _esc('esc-T1-3', level=0)

        census = _by_id(related_pending([twin_l1, twin_l0], resolved=resolved))

        assert set(census) == {'esc-T1-2', 'esc-T1-3'}
        for entry in census.values():
            assert entry['same_task'] is True
            assert entry['shared_member'] is None

    def test_other_task_l2_sharing_a_member_names_the_smallest_match(self):
        resolved = _esc('esc-T1-1', level=2, members=['m3', 'm2', 'm1'])
        sibling = _esc('esc-T2-1', task_id='T2', level=2, members=['m9', 'm3', 'm2'])

        [entry] = related_pending([sibling], resolved=resolved)

        assert entry['id'] == 'esc-T2-1'
        assert entry['same_task'] is False
        assert entry['shared_member'] == 'm2'

    def test_l2_clustering_the_resolved_record_itself_is_listed(self):
        resolved = _esc('esc-T1-5', level=1)
        clusterer = _esc('esc-T9-1', task_id='T9', level=2, members=['esc-T1-5', 'mx'])

        [entry] = related_pending([clusterer], resolved=resolved)

        assert entry['id'] == 'esc-T9-1'
        assert entry['same_task'] is False
        assert entry['shared_member'] == 'esc-T1-5'

    def test_same_task_and_member_sharing_appears_once_with_both_markers(self):
        resolved = _esc('esc-T1-1', level=2, members=['m1'])
        both = _esc('esc-T1-9', level=2, members=['m1'])

        census = related_pending([both, both], resolved=resolved)

        assert census == [{
            'id': 'esc-T1-9', 'category': 'design_concern', 'severity': 'blocking',
            'level': 2, 'same_task': True, 'shared_member': 'm1',
        }]

    def test_records_under_one_synthetic_anchor_are_all_same_task_twins(self):
        """Documented breadth: a synthetic anchor is one task_id for every instance."""
        anchor = _AMENDMENT_TRUNCATION_ANCHOR_TASK_ID
        resolved = _esc('esc-a-1', task_id=anchor, severity='info', category='infra_issue')
        others = [
            _esc(f'esc-a-{i}', task_id=anchor, severity='info', category='infra_issue')
            for i in (2, 3)
        ]

        census = _by_id(related_pending(others, resolved=resolved))

        assert set(census) == {'esc-a-2', 'esc-a-3'}
        for entry in census.values():
            assert entry['same_task'] is True
            assert entry['shared_member'] is None

    def test_the_resolved_record_itself_is_never_listed(self):
        resolved = _esc('esc-T1-1', level=2, members=['m1'])

        assert related_pending([resolved], resolved=resolved) == []

    def test_non_qualifying_records_are_not_listed(self):
        resolved = _esc('esc-T1-1', level=2, members=['m1'])
        closed_twin = _esc('esc-T1-2', status='resolved')
        other_task_l1 = _esc('esc-T2-1', task_id='T2', level=1, members=['m1'])
        unrelated = _esc('esc-T3-1', task_id='T3', level=2, members=['m7'])

        census = related_pending(
            [closed_twin, other_task_l1, unrelated], resolved=resolved,
        )

        assert census == []

    def test_entry_keys_are_exactly_the_documented_shape(self):
        resolved = _esc('esc-T1-1', level=2, members=['m1'])
        pending = [
            _esc('esc-T1-2'),
            _esc('esc-T2-1', task_id='T2', level=2, members=['m1']),
        ]

        census = related_pending(pending, resolved=resolved)

        assert len(census) == 2
        for entry in census:
            assert set(entry) == _ENTRY_KEYS

    def test_output_is_sorted_by_id(self):
        resolved = _esc('esc-T1-1', level=2, members=['m1'])
        pending = [
            _esc('esc-T1-9'),
            _esc('esc-T2-1', task_id='T2', level=2, members=['m1']),
            _esc('esc-T1-3'),
        ]

        census = related_pending(pending, resolved=resolved)

        assert [e['id'] for e in census] == ['esc-T1-3', 'esc-T1-9', 'esc-T2-1']

    def test_no_twins_is_empty(self):
        resolved = _esc('esc-T1-1', level=2, members=['m1'])

        assert related_pending([], resolved=resolved) == []
        assert related_pending(
            [_esc('esc-T5-1', task_id='T5')], resolved=resolved,
        ) == []

    def test_inputs_are_not_mutated(self):
        resolved = _esc('esc-T1-1', level=2, members=['m2', 'm1'])
        pending = [
            _esc('esc-T1-2'),
            _esc('esc-T2-1', task_id='T2', level=2, members=['m1', 'm2']),
            resolved,
        ]
        before = [e.to_dict() for e in pending]
        resolved_before = resolved.to_dict()
        order_before = [e.id for e in pending]

        related_pending(pending, resolved=resolved)

        assert [e.id for e in pending] == order_before
        assert [e.to_dict() for e in pending] == before
        assert resolved.to_dict() == resolved_before


async def _resolve(server, **kw: Any) -> dict[str, Any]:
    """``resolve_issue`` is a SYNC tool, so ``tool.fn`` returns directly."""
    tool = await server.get_tool('resolve_issue')
    return tool.fn(**kw)


class _CensusFailingQueue(EscalationQueue):
    """A REAL queue whose ``get_pending`` raises once ``fail_census`` is armed.

    The ``test_pins_recovery_annotation.py::_CountingQueue`` shape: a subclass
    delegating through ``super()``, not a mock.  Armed only after seeding, so
    everything before the census runs against the real scan.
    """

    fail_census = False

    def get_pending(self):
        if self.fail_census:
            raise OSError('census scan failed')
        return super().get_pending()


def _record_bytes(queue: EscalationQueue, esc_id: str) -> bytes:
    return (queue.queue_dir / f'{esc_id}.json').read_bytes()


@pytest.mark.asyncio
class TestResolveIssueRelatedPending:
    async def test_resolve_lists_pending_twins_after_the_cascade(self, tmp_path):
        """PRD beta acceptance: twins listed, cascade-closed member not, nothing touched."""
        queue = EscalationQueue(tmp_path / 'esc')
        m1 = _file(queue, 'T1', level=1, summary='member of L2-A')
        twin = _file(queue, 'T1', level=1, summary='same question, not clustered')
        l2_a = _file(queue, 'T1', level=2, members=[m1.id, 'm2'], category='design_concern')
        l2_b = _file(queue, 'T2', level=2, members=['m2', 'm9'], category='infra_issue')
        loner = _file(queue, 'T5', level=1, summary='no twins anywhere')
        server = create_server(queue, startup_sweep=False)
        twin_bytes = _record_bytes(queue, twin.id)
        l2_b_bytes = _record_bytes(queue, l2_b.id)

        response = await _resolve(
            server, escalation_id=l2_a.id, resolution='ruled', action='resume',
        )

        assert response['status'] == 'resolved'
        assert response['related_pending'] == [
            {
                'id': twin.id, 'category': twin.category, 'severity': twin.severity,
                'level': 1, 'same_task': True, 'shared_member': None,
            },
            {
                'id': l2_b.id, 'category': 'infra_issue', 'severity': l2_b.severity,
                'level': 2, 'same_task': False, 'shared_member': 'm2',
            },
        ]
        assert m1.id not in {e['id'] for e in response['related_pending']}, (
            'the census is taken after the cascade: a closed member is not actionable'
        )
        assert _record_bytes(queue, twin.id) == twin_bytes, 'report-only'
        assert _record_bytes(queue, l2_b.id) == l2_b_bytes, 'report-only'

        lonely = await _resolve(
            server, escalation_id=loner.id, resolution='ruled', action='resume',
        )
        assert lonely['related_pending'] == []

    async def test_park_carries_the_census_without_listing_itself(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        target = _file(queue, 'T1', level=1)
        twin = _file(queue, 'T1', level=1)
        server = create_server(queue, startup_sweep=False)

        response = await _resolve(
            server, escalation_id=target.id, resolution='parked', action='park',
        )

        assert response['status'] == 'pending'
        assert [e['id'] for e in response['related_pending']] == [twin.id]

    async def test_close_only_carries_the_census(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        target = _file(queue, 'T1', level=1)
        twin = _file(queue, 'T1', level=1)
        server = create_server(queue, startup_sweep=False)

        response = await _resolve(
            server, escalation_id=target.id, resolution='closed', action='close_only',
        )

        assert response['status'] == 'dismissed'
        assert [e['id'] for e in response['related_pending']] == [twin.id]

    async def test_census_failure_omits_the_key_and_keeps_the_resolve(
        self, tmp_path, caplog: pytest.LogCaptureFixture,
    ):
        """Absent means UNKNOWN: a false [] would read as "no twins"."""
        queue = _CensusFailingQueue(tmp_path / 'esc')
        target = _file(queue, 'T1', level=1)
        _file(queue, 'T1', level=1)
        server = create_server(queue, startup_sweep=False)
        queue.fail_census = True

        with caplog.at_level(logging.ERROR, logger='escalation.server'):
            response = await _resolve(
                server, escalation_id=target.id, resolution='ruled', action='resume',
            )

        assert 'error' not in response
        assert response['status'] == 'resolved'
        assert not (queue.queue_dir / f'{target.id}.json').exists(), 'still archived'
        assert 'related_pending' not in response
        assert any(
            r.levelno >= logging.ERROR and target.id in r.getMessage()
            for r in caplog.records
        ), f'the census failure must be logged: {[r.getMessage() for r in caplog.records]}'

    async def test_error_return_carries_no_census(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, 'T1', level=1)
        server = create_server(queue, startup_sweep=False)

        response = await _resolve(
            server, escalation_id='esc-nope-1', resolution='x', action='resume',
        )

        assert 'error' in response
        assert 'related_pending' not in response
