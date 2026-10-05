"""The completion-claim gate wired into the `add_memory` tool (task 4715).

`test_completion_claim_gate_ingestion.py` pins the same gate on `add_episode`;
this pins that `add_memory` runs it too: a contradicted or unconfirmable claim
naming concrete work is INGESTED and TAGGED (never rejected), the tag reaches
the service at both of the tool's call sites, the flag is echoed on the
response, and one operator escalation is filed per ref, only once the service
has accepted the write.

The emitter is NOT monkeypatched. Every server registers `dark_factory` at
`tmp_path`, so the real `emit_unverified_claim_escalation` files into
`tmp_path/data/escalations` and each assertion about escalations reads the
queue through its own API.

The content says "has been applied": `applied` is outside
`task_filter.PRESENT_TENSE_COMPLETION_RE`, so the recon-only 2824 gate cannot
fire on it and a recon-stage agent_id observes THIS gate in isolation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

from fused_memory.memory_metadata import MemoryMetadataValidationError, MetadataViolation
from fused_memory.server.tools import create_mcp_server

# The write-triage harness, imported rather than re-derived (the established
# pattern — see test_entities_gate_ingestion.py): reaching add_memory's C1
# fallback call site needs triage enabled with a calibrated candidate.
from server.test_add_memory_write_triage_gate import _candidate, _configure_config

_PROJECT_ID = 'dark_factory'
_CLAIM_CONTENT = "task 5422's de-flake fix has been applied"
_NO_CLAIM_CONTENT = 'renamed the helper for clarity'
_IN_PROGRESS = {'5422': 'in-progress'}


def _mock_service(**config_overrides) -> AsyncMock:
    mock_service = AsyncMock()
    mem_result = MagicMock()
    mem_result.model_dump.return_value = {'id': 'mem-1'}
    mock_service.add_memory.return_value = mem_result
    _configure_config(
        mock_service,
        **{'enabled': False, 'near_dup_guard_enabled': False, **config_overrides},
    )
    return mock_service


def _server(mock_service, root: Path, *, statuses: dict | None = None):
    task_interceptor = MagicMock()
    task_interceptor.get_statuses = AsyncMock(return_value=statuses or {})
    task_interceptor.get_ticket_row = AsyncMock(return_value=None)
    server = create_mcp_server(
        mock_service,
        task_interceptor=task_interceptor,
        known_projects={_PROJECT_ID: str(root)},
    )
    return server, task_interceptor


async def _call(server, **overrides) -> dict:
    args = {
        'content': _CLAIM_CONTENT,
        'category': 'decisions_and_rationale',
        'agent_id': 'claude-interactive',
        'project_id': _PROJECT_ID,
    }
    args.update(overrides)
    return await server._tool_manager.call_tool('add_memory', args)


def _pending_escalations(root: Path) -> list:
    queue_dir = root / 'data' / 'escalations'
    if not queue_dir.exists():
        return []
    return EscalationQueue(queue_dir).get_pending()


def _gate_warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith('completion_claim_gate.unverified')
    ]


class TestContradictedClaimsAreTagged:

    @pytest.mark.asyncio
    async def test_contradicted_claim_on_a_graph_write_is_ingested_and_tagged(self, tmp_path):
        mock_service = _mock_service()
        server, _ = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        result = await _call(server)

        assert 'error' not in result, f'the gate must tag, never reject: {result!r}'
        mock_service.add_memory.assert_awaited_once()
        assert mock_service.add_memory.call_args.kwargs.get('unverified_claim') is True
        flag = result.get('unverified_claim')
        assert isinstance(flag, dict), f'no structured flag on the response: {result!r}'
        assert flag.get('tag') == 'unverified_claim'
        claims = flag.get('claims')
        assert isinstance(claims, list) and len(claims) == 1, f'{flag!r}'
        entry = claims[0]
        assert entry.get('ref') == '5422', f'{entry!r}'
        assert entry.get('subject') == 'task', f'{entry!r}'
        assert entry.get('status') == 'mismatch', f'{entry!r}'
        assert entry.get('observed') == 'in-progress', f'{entry!r}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'category', ['decisions_and_rationale', 'procedural_knowledge', None],
    )
    async def test_gate_is_category_independent(self, tmp_path, category):
        mock_service = _mock_service()
        server, _ = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        result = await _call(server, category=category)

        assert 'error' not in result, f'category={category!r}: {result!r}'
        assert mock_service.add_memory.call_args.kwargs.get('unverified_claim') is True, (
            f'category={category!r} was not tagged'
        )
        assert result.get('unverified_claim', {}).get('claims'), (
            f'category={category!r}: no flag on the response: {result!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'agent_id',
        [
            'recon-stage-task_knowledge_sync',
            'claude-task-5422-implementer',
            'claude-interactive',
            None,
        ],
    )
    async def test_gate_is_not_recon_stage_scoped(self, tmp_path, agent_id):
        mock_service = _mock_service()
        server, _ = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        result = await _call(server, agent_id=agent_id)

        assert 'error' not in result, f'agent_id={agent_id!r} was rejected: {result!r}'
        mock_service.add_memory.assert_awaited_once()
        assert mock_service.add_memory.call_args.kwargs.get('unverified_claim') is True, (
            f'agent_id={agent_id!r} was not tagged'
        )
        assert result.get('unverified_claim', {}).get('claims'), (
            f'agent_id={agent_id!r}: no flag on the response: {result!r}'
        )


class TestInertPaths:

    @pytest.mark.asyncio
    async def test_no_claim_write_is_untouched(self, tmp_path):
        mock_service = _mock_service()
        server, task_interceptor = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        result = await _call(server, content=_NO_CLAIM_CONTENT)

        task_interceptor.get_statuses.assert_not_awaited()
        task_interceptor.get_ticket_row.assert_not_awaited()
        assert 'unverified_claim' not in mock_service.add_memory.call_args.kwargs
        assert 'unverified_claim' not in result
        assert _pending_escalations(tmp_path) == []

    @pytest.mark.asyncio
    async def test_verified_claim_is_inert(self, tmp_path):
        mock_service = _mock_service()
        server, _ = _server(mock_service, tmp_path, statuses={'5422': 'done'})

        result = await _call(server)

        assert 'unverified_claim' not in mock_service.add_memory.call_args.kwargs
        assert 'unverified_claim' not in result
        assert _pending_escalations(tmp_path) == []


class TestOperatorEscalation:

    @pytest.mark.asyncio
    async def test_tagged_write_files_one_operator_escalation(self, tmp_path):
        mock_service = _mock_service()
        server, _ = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        result = await _call(server)

        pending = _pending_escalations(tmp_path)
        assert len(pending) == 1, f'{pending!r}'
        esc = pending[0]
        assert esc.category == 'unverified_completion_claim'
        assert esc.task_id == 'unverified-claim-5422'
        assert result['unverified_claim'].get('escalation_id') == esc.id

        await _call(server)

        assert [e.id for e in _pending_escalations(tmp_path)] == [esc.id], (
            'a repeated claim about the same ref must fold onto the open escalation'
        )


class TestGateOrdering:

    @pytest.mark.asyncio
    async def test_a_rejected_write_consults_no_authority_and_files_nothing(self, tmp_path):
        mock_service = _mock_service(near_dup_guard_enabled=True)
        mock_service.search.return_value = [_candidate('m1', 0.97, content=_CLAIM_CONTENT)]
        server, task_interceptor = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        result = await _call(server, category='procedural_knowledge')

        assert result.get('error_type') == 'ProceduralKnowledgeNearDuplicateWriteRejected', (
            f'{result!r}'
        )
        mock_service.add_memory.assert_not_awaited()
        task_interceptor.get_statuses.assert_not_awaited()
        assert _pending_escalations(tmp_path) == []

    @pytest.mark.asyncio
    async def test_a_service_level_metadata_reject_reports_nothing(self, tmp_path, caplog):
        # What MemoryService.add_memory raises under memory_metadata.enforce
        # when a violation is fatal; the claim was checked, the write never landed.
        mock_service = _mock_service()
        mock_service.add_memory.side_effect = MemoryMetadataValidationError([
            MetadataViolation(
                key='topic', code='invalid_topic_slug',
                message='topic is malformed', fatal=True,
            ),
        ])
        server, _ = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        with caplog.at_level(logging.DEBUG):
            result = await _call(server)

        assert result.get('error_type') == 'MemoryMetadataValidationError', f'{result!r}'
        assert mock_service.add_memory.call_args.kwargs.get('unverified_claim') is True
        assert _pending_escalations(tmp_path) == [], (
            'an escalation saying the write was ingested was filed for a write '
            'the service rejected'
        )
        assert _gate_warnings(caplog) == [], (
            f'the INGESTED-and-tagged line was logged for a rejected write: '
            f'{_gate_warnings(caplog)!r}'
        )


class TestBothServiceCallSitesCarryTheTag:

    @pytest.mark.asyncio
    async def test_write_triage_fallback_retry_carries_the_tag(self, tmp_path):
        mock_service = _mock_service(enabled=True)
        mem_result = MagicMock()
        mem_result.model_dump.return_value = {'id': 'fallback-id'}
        mock_service.add_memory.side_effect = [
            RuntimeError('parent_id rejected by the write seam'),
            mem_result,
        ]
        mock_service.search.return_value = [_candidate('m1', 0.97, content=_CLAIM_CONTENT)]
        server, _ = _server(mock_service, tmp_path, statuses=_IN_PROGRESS)

        result = await _call(server, category='procedural_knowledge')

        assert 'error' not in result, f'the write was blocked: {result!r}'
        assert mock_service.add_memory.await_count == 2, (
            f'expected attach then fallback: {mock_service.add_memory.await_args_list!r}'
        )
        for label, call in zip(
            ('attach', 'fallback'), mock_service.add_memory.await_args_list, strict=True,
        ):
            assert call.kwargs.get('unverified_claim') is True, (
                f'the {label} call dropped the tag: {call.kwargs!r}'
            )
        assert result.get('unverified_claim', {}).get('claims'), f'{result!r}'
