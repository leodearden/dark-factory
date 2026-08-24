"""The close-time refusal at the ``set_task_status`` chokepoint (task 3112).

Defect 2's user-observable half. ``consolidate_memories`` reports closure at
*op* time, but nothing stood between a curator's closure CLAIM and the gate
task actually going ``done``. ``TaskInterceptor._apply_status_transition`` is
the sole seam — ``SqliteTaskBackend`` raises ``StatusWriteAuthorityError`` if
status is written any other way — so the refusal lives there.

Fixture conventions follow ``test_task_interceptor.py``: an ``AsyncMock``
taskmaster whose ``get_task`` returns the ``before`` snapshot, a real
``EventBuffer`` on ``tmp_path``, and ``TaskInterceptor(taskmaster, reconciler,
event_buffer)``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.reconciliation.event_buffer import EventBuffer

_TOPIC = 'seam-demo-topic'
_PROJECT_ROOT = '/tmp/seam-demo'


def _uuid(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def _gate_metadata(**overrides):
    meta = {
        'execution_class': 'operational',
        'operational_mode': 'gate',
        'task_kind': 'deterministic',
        'always_escalates': True,
        GATE_METADATA_KEY: {'topic': _TOPIC},
    }
    meta.update(overrides)
    return meta


def _member(mid, *, canonical=None):
    meta = {'topic': _TOPIC}
    if canonical is not None:
        meta['canonical'] = canonical
    return {'id': mid, 'created_at': '2026-08-24T00:00:00+00:00', 'metadata': meta}


_WELL_FORMED = [_member(_uuid(1), canonical=True), _member(_uuid(2))]
_MALFORMED = [_member(_uuid(1)), _member(_uuid(2))]  # no canonical


def _scroll(members, *, total=None, raises=None):
    """A stand-in for the injected memory scroll."""
    calls = []

    async def scroll(filters, *, limit):
        calls.append({'filters': filters, 'limit': limit})
        if raises is not None:
            raise raises
        return list(members)

    async def count(filters):
        calls.append({'count': filters})
        if raises is not None:
            raise raises
        return len(members) if total is None else total

    scroll.calls = calls
    scroll.count = count
    return scroll


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(
        return_value={
            'id': '9001',
            'status': 'pending',
            'title': 'Consolidation gate',
            'metadata': _gate_metadata(),
        }
    )
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.set_status_and_stamp_audit = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': []})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'seam_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


async def _set_done(interceptor, **kwargs):
    return await interceptor.set_task_status(
        '9001', 'done', project_root=_PROJECT_ROOT, **kwargs
    )


class TestSeamRefusal:
    @pytest.mark.asyncio
    async def test_refuses_to_close_over_a_malformed_cluster(
        self, interceptor, taskmaster
    ):
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'
        assert result['topic'] == _TOPIC
        assert [r['code'] for r in result['reasons']] == ['no_canonical']

    @pytest.mark.asyncio
    async def test_a_refusal_must_carry_an_error_key(self, interceptor):
        """The CSV branch computes all_ok from ``result.get('error') is None``,
        so a refusal lacking 'error' would be REPORTED AS SUCCESS."""
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result.get('error') is not None

    @pytest.mark.asyncio
    async def test_a_refusal_mutates_nothing(
        self, interceptor, taskmaster, reconciler, event_buffer
    ):
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        await _set_done(interceptor)
        taskmaster.set_task_status.assert_not_called()
        taskmaster.set_status_and_stamp_audit.assert_not_called()
        reconciler.reconcile_task.assert_not_called()
        stats = await event_buffer.get_stats()
        assert stats['size'] == 0

    @pytest.mark.asyncio
    async def test_a_well_formed_cluster_proceeds(self, interceptor, taskmaster):
        interceptor.set_consolidation_scroll(_scroll(_WELL_FORMED))
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1


class TestSeamFailsClosed:
    @pytest.mark.asyncio
    async def test_a_scroll_that_raises_refuses_rather_than_passing(
        self, interceptor, taskmaster
    ):
        """``get_memories_by_metadata`` propagates a read TimeoutError rather
        than returning [], so this is reachable. A gate whose job is refuting a
        false closure claim must not pass when it cannot see (INV-3)."""
        interceptor.set_consolidation_scroll(
            _scroll(_WELL_FORMED, raises=TimeoutError('qdrant read timeout'))
        )
        result = await _set_done(interceptor)
        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'
        taskmaster.set_task_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unexpected_exception_also_refuses(self, interceptor):
        interceptor.set_consolidation_scroll(
            _scroll(_WELL_FORMED, raises=RuntimeError('boom'))
        )
        result = await _set_done(interceptor)
        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'


class TestSeamDormancy:
    """Nothing on the current corpus can regress: dormancy is STRUCTURAL."""

    @pytest.mark.asyncio
    async def test_dormant_when_no_scroll_is_wired(self, interceptor, taskmaster):
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1

    @pytest.mark.asyncio
    async def test_dormant_without_the_gate_key(self, interceptor, taskmaster):
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': {'execution_class': 'operational', 'operational_mode': 'gate'},
        }
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert scroll.calls == []

    @pytest.mark.asyncio
    async def test_dormant_when_not_a_gate(self, interceptor, taskmaster):
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _gate_metadata(operational_mode='llm'),
        }
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert scroll.calls == []

    @pytest.mark.asyncio
    async def test_dormant_for_a_non_done_transition(self, interceptor, taskmaster):
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await interceptor.set_task_status(
            '9001', 'in-progress', project_root=_PROJECT_ROOT
        )
        assert result.get('error') is None
        assert scroll.calls == []


class TestSeamShapeAndPrecedence:
    @pytest.mark.asyncio
    async def test_metadata_arriving_as_a_json_string_is_still_gated(
        self, interceptor, taskmaster
    ):
        """``before['metadata']`` may be a dict OR a JSON string."""
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': json.dumps(_gate_metadata()),
        }
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result['error'] == 'consolidation_not_closed'

    @pytest.mark.asyncio
    async def test_the_terminal_exit_gate_still_runs_first(
        self, interceptor, taskmaster
    ):
        """Gate precedence is unchanged: a terminal task is rejected by the
        earlier gate and the scroll is never consulted."""
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'cancelled',
            'metadata': _gate_metadata(),
        }
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result['error'] == 'terminal_exit_rejected'
        assert scroll.calls == []

    @pytest.mark.asyncio
    async def test_the_scroll_is_asked_for_the_gates_topic(self, interceptor):
        scroll = _scroll(_WELL_FORMED)
        interceptor.set_consolidation_scroll(scroll)
        await _set_done(interceptor)
        assert any(
            call.get('filters', call.get('count')) == {'topic': _TOPIC}
            for call in scroll.calls
        )
