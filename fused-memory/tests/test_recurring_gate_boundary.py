"""submit_task boundary tests for the recurring-human-gate guard (task 3588).

The unit-level invariant matrix for ``recurring_gate_guard`` lives in
tests/test_recurring_gate_guard.py; this file asserts the guard is WIRED
into the ``submit_task`` tool boundary, and realizes the task's literal
user-observable signal end to end.

Naming follows the existing dedicated-boundary-file convention
(test_candidate_key_boundary.py, test_operational_routing_boundary_matrix.py)
rather than appending to the large shared test_tools_validation.py — the
`files` list drives this repo's lock charter, and claiming that heavily
shared file would block unrelated submit_task work for this task's whole
lifetime for no functional gain.

Part A drives the lightweight mocked-interceptor harness
(test_tools_validation.py::TestSubmitTaskPremiseLintGuard) to prove
rejection happens BEFORE persistence and that exempt callers pay zero
added I/O. Part B drives a REAL stack — SqliteTaskBackend + EventBuffer +
TicketStore + real TaskInterceptor + create_mcp_server — so the
bounded-queue invariant is asserted against real persisted rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.server.tools import create_mcp_server


def _parse_tool_result(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    Mirrors test_task_tools.py's fixture of the same name: these project
    roots are synthetic, not real git worktrees, so the real resolver would
    reject them.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


def _carrier(
    task_id: str = '5902',
    status: str = 'pending',
    subject_key: str = 'gate_subject',
    subject: str = '5879',
) -> dict[str, Any]:
    """A persisted human-gate carrier row in the `_row_to_task` wire shape."""
    return {
        'id': task_id,
        'title': 'GATE: stranded task 5879 needs a human ruling',
        'status': status,
        'metadata': {
            'execution_class': 'operational',
            'operational_mode': 'gate',
            subject_key: subject,
        },
    }


def _submission(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'project_root': '/project',
        'prompt': 'Escalate stranded task 5879 for a human ruling',
        'title': 'GATE: stranded task 5879 needs a human ruling',
        'description': 'Task 5879 has been stranded; a human must rule on it.',
        'agent_id': 'recon-stage-task_knowledge_sync',
        'metadata': {
            'execution_class': 'operational',
            'operational_mode': 'gate',
            'gate_subject': '5879',
        },
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Part A — lightweight wiring (mocked interceptor)
# ---------------------------------------------------------------------------


class TestSubmitTaskRecurringGateWiring:
    """The guard runs inside the submit_task tool, before persistence."""

    @staticmethod
    def _server(*carriers: dict[str, Any]):
        mock_ti = AsyncMock()
        mock_ti.submit_task.return_value = {'ticket': 'tkt_x'}
        mock_ti.get_tasks = AsyncMock(return_value={'tasks': list(carriers)})
        return create_mcp_server(AsyncMock(), task_interceptor=mock_ti), mock_ti

    @pytest.mark.asyncio
    async def test_duplicate_open_gate_is_rejected_before_persistence(self):
        server, mock_ti = self._server(_carrier())

        result = await server._tool_manager.call_tool('submit_task', _submission())

        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'RecurringGateViolation', f'got {parsed!r}'
        assert '5902' in parsed.get('error', ''), f'got {parsed!r}'
        assert parsed.get('existing_gate_task_id') == '5902', f'got {parsed!r}'
        # Rejection happens BEFORE the interceptor is reached.
        mock_ti.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stored_alias_carrier_is_matched_at_the_boundary(self):
        # The already-filed 5902/5916/5929 population keys the subject via
        # stranded_task_id, so a read-side alias must work end to end.
        server, mock_ti = self._server(
            _carrier(task_id='5916', subject_key='stranded_task_id')
        )

        result = await server._tool_manager.call_tool('submit_task', _submission())

        parsed = _parse_tool_result(result)
        assert parsed.get('existing_gate_task_id') == '5916', f'got {parsed!r}'
        mock_ti.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_incoming_see_also_related_task_id_is_not_rejected(self):
        """A submission whose `related_task_id` is a see-also must get through.

        `related_task_id` is a generic cross-reference in the live corpus
        (3042 -> 2885, 3046 -> 3045, both code_tdd) as well as a gate subject
        (3240/3361/3463), so the incoming side cannot treat it as a subject.
        Rejecting here would keep a genuinely novel human decision from ever
        reaching a human — the one failure direction this otherwise fail-open
        guard must not take.
        """
        server, mock_ti = self._server(_carrier())

        result = await server._tool_manager.call_tool(
            'submit_task',
            _submission(
                metadata={
                    'execution_class': 'operational',
                    'operational_mode': 'gate',
                    'related_task_id': '5879',
                }
            ),
        )

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'got {parsed!r}'
        mock_ti.submit_task.assert_awaited_once()
        # No subject resolved => the corpus read is never issued either.
        mock_ti.get_tasks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stored_related_task_id_carrier_is_matched_at_the_boundary(self):
        # dark-factory's own gates 3240/3361/3463 key their subject via
        # related_task_id, so the STORED side must keep reading it.
        server, mock_ti = self._server(
            _carrier(task_id='3463', subject_key='related_task_id')
        )

        result = await server._tool_manager.call_tool('submit_task', _submission())

        parsed = _parse_tool_result(result)
        assert parsed.get('existing_gate_task_id') == '3463', f'got {parsed!r}'
        mock_ti.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_carrier_does_not_block(self):
        server, mock_ti = self._server(_carrier(status='cancelled'))

        result = await server._tool_manager.call_tool('submit_task', _submission())

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'got {parsed!r}'
        mock_ti.submit_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_recon_caller_is_exempt_and_pays_no_added_io(self):
        server, mock_ti = self._server(_carrier())

        result = await server._tool_manager.call_tool(
            'submit_task', _submission(agent_id='claude-interactive')
        )

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'got {parsed!r}'
        mock_ti.submit_task.assert_awaited_once()
        # The corpus read is never issued for an exempt caller.
        mock_ti.get_tasks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_code_tdd_submission_with_incidental_subject_passes_through(self):
        server, mock_ti = self._server(_carrier())

        result = await server._tool_manager.call_tool(
            'submit_task',
            _submission(
                metadata={'execution_class': 'code_tdd', 'gate_subject': '5879'}
            ),
        )

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'got {parsed!r}'
        mock_ti.submit_task.assert_awaited_once()
        mock_ti.get_tasks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gate_without_a_subject_passes_through(self):
        server, mock_ti = self._server(_carrier())

        result = await server._tool_manager.call_tool(
            'submit_task',
            _submission(
                metadata={'execution_class': 'operational', 'operational_mode': 'gate'}
            ),
        )

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'got {parsed!r}'
        mock_ti.submit_task.assert_awaited_once()
        mock_ti.get_tasks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_corpus_read_failure_fails_open(self):
        server, mock_ti = self._server(_carrier())
        mock_ti.get_tasks = AsyncMock(side_effect=RuntimeError('backend down'))

        result = await server._tool_manager.call_tool('submit_task', _submission())

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'got {parsed!r}'
        mock_ti.submit_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_corpus_read_is_narrowed_to_non_terminal_statuses(self):
        from shared.task_statuses import TERMINAL

        server, mock_ti = self._server()

        await server._tool_manager.call_tool('submit_task', _submission())

        mock_ti.get_tasks.assert_awaited_once()
        statuses = mock_ti.get_tasks.await_args.kwargs['statuses']
        # Pushed into the backend's WHERE status IN (...) rather than
        # filtered in Python — this runs on a write path.
        assert statuses, 'statuses filter must be supplied'
        assert not (set(map(str, statuses)) & set(map(str, TERMINAL)))


# ---------------------------------------------------------------------------
# Part B — end-to-end on a real stack
# ---------------------------------------------------------------------------


async def _build_stack(tmp_path):
    """Real SqliteTaskBackend + EventBuffer + TicketStore + TaskInterceptor
    + create_mcp_server. Mirrors test_operational_routing_boundary_matrix.py's
    `_build_stack` (and test_task_tools.py's `real_task_stack`).

    `config.taskmaster` is cleared so `_get_curator()`'s one-shot
    corpus-backfill background task never fires — keeps the stack hermetic
    regardless of the ambient fused-memory/config/config.yaml.
    """
    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
    from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.middleware.ticket_store import TicketStore
    from fused_memory.reconciliation.event_buffer import EventBuffer

    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'gate_eb.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    ticket_store = TicketStore(tmp_path / 'gate_tickets.db')
    await ticket_store.initialize()

    config = FusedMemoryConfig()
    config.taskmaster = None

    interceptor = TaskInterceptor(
        backend, None, event_buffer, config, ticket_store=ticket_store
    )
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    return server, interceptor, backend, event_buffer, ticket_store


async def _teardown_stack(interceptor, ticket_store, event_buffer, backend):
    """Mirrors real_task_stack's teardown: close ticket_store, cancel any live
    curator-worker tasks, then close event_buffer and backend."""
    await ticket_store.close()
    for _wt in list(interceptor._worker_tasks.values()):
        if not _wt.done():
            _wt.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await _wt
    await event_buffer.close()
    await backend.close()


@pytest_asyncio.fixture
async def real_stack(tmp_path):
    server, interceptor, backend, event_buffer, ticket_store = await _build_stack(
        tmp_path
    )
    try:
        yield server, str(tmp_path)
    finally:
        await _teardown_stack(interceptor, ticket_store, event_buffer, backend)


class TestRecurringGateEndToEnd:
    """The task's literal user-observable signal, on real persisted rows.

    `planning_mode=True` is used so submit_task returns a `task_id`
    SYNCHRONOUSLY in `deferred` status — no ticket/resolve_ticket round-trip
    — and `deferred` is non-terminal, so the guard is proven to block on it.
    """

    @staticmethod
    async def _file_gate(server, root: str, cycle: int = 1):
        """File one cycle's gate for subject 5879.

        The per-cycle text is VARIED deliberately. planning_mode already
        collapses byte-identical submissions via its own combine path
        (measured: three identical filings return the same task_id with
        ``combined: True``), which would make every assertion below vacuous.
        Varying the text reproduces the actual measured failure mode — the
        5902/5916/5929 carriers each carried their own cycle's prose, and
        `gate_subject` is the ONLY thing linking them.
        """
        return _parse_tool_result(
            await server._tool_manager.call_tool(
                'submit_task',
                _submission(
                    project_root=root,
                    planning_mode=True,
                    title=(
                        f'GATE cycle {cycle}: stranded task 5879 needs a '
                        f'human ruling'
                    ),
                    description=(
                        f'Cycle {cycle} observed task 5879 stranded again; '
                        f'a human must rule on it.'
                    ),
                    prompt=(
                        f'Cycle {cycle}: escalate stranded task 5879 for a '
                        f'human ruling'
                    ),
                ),
            )
        )

    @pytest.mark.asyncio
    async def test_second_gate_for_the_same_subject_is_refused(self, real_stack):
        server, root = real_stack

        first = await self._file_gate(server, root, cycle=1)
        assert first.get('task_id'), f'got {first!r}'
        assert first.get('status') == 'deferred', f'got {first!r}'

        second = await self._file_gate(server, root, cycle=2)
        assert second.get('error_type') == 'RecurringGateViolation', f'got {second!r}'
        assert second.get('existing_gate_task_id') == first['task_id'], (
            f'got {second!r}, first={first!r}'
        )
        assert first['task_id'] in second.get('error', ''), f'got {second!r}'
        assert 'task_id' not in second, f'no second carrier may be minted: {second!r}'

    @pytest.mark.asyncio
    async def test_exactly_one_carrier_exists_for_the_subject(self, real_stack):
        server, root = real_stack

        await self._file_gate(server, root, cycle=1)
        await self._file_gate(server, root, cycle=2)
        await self._file_gate(server, root, cycle=3)

        listing = _parse_tool_result(
            await server._tool_manager.call_tool(
                'get_tasks', {'project_root': root},
            )
        )
        carriers = [
            t
            for t in listing['tasks']
            if isinstance(t.get('metadata'), dict)
            and t['metadata'].get('gate_subject') == '5879'
        ]
        # The bounded-queue invariant: three distinct-text filings for the
        # same subject, one carrier. Unguarded this lands three (measured) —
        # which is exactly the 5902 -> 5916 -> 5929 chain.
        assert len(carriers) == 1, f'got {len(carriers)} carriers: {carriers!r}'

    @pytest.mark.asyncio
    async def test_recurrence_after_closure_is_not_blocked(self, real_stack):
        server, root = real_stack

        first = await self._file_gate(server, root, cycle=1)
        assert first.get('task_id'), f'got {first!r}'

        cancelled = _parse_tool_result(
            await server._tool_manager.call_tool(
                'set_task_status',
                {'id': first['task_id'], 'status': 'cancelled', 'project_root': root},
            )
        )
        assert 'error' not in cancelled, f'got {cancelled!r}'

        third = await self._file_gate(server, root, cycle=2)
        # The condition genuinely recurred after closure — a fresh gate is
        # exactly what should happen.
        assert third.get('task_id'), f'got {third!r}'
        assert third['task_id'] != first['task_id'], f'got {third!r}'
