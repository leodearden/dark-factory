"""Doubles shared by test_normalize_topic_slugs.py and the test_topic_slug_*.py suites."""
from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock

from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY


def rec(memory_id: str, topic: object = None, **meta) -> dict:
    """One scroll-shaped record, the shape ``scroll_all_by_metadata`` yields.

    ``topic=None`` means "no ``topic`` key at all", which is a genuinely
    different case from ``topic`` present-and-``None`` — the caller spells the
    latter explicitly via ``metadata``.
    """
    metadata: dict = dict(meta)
    if topic is not None:
        metadata['topic'] = topic
    return {'id': memory_id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': metadata}


def by_reason(skips: list[dict], reason: str) -> list[dict]:
    return [s for s in skips if s.get('reason') == reason]


def gate_task(
    task_id: str,
    topic: str,
    *,
    project_id: str = 'dark_factory',
    project_root: str = '/repo',
    **block_extras,
) -> dict:
    """One gate task as the MCP ``get_task`` read hands it over.

    ``block_extras`` seed the SIBLING keys inside the block (``provenance``,
    ``considered_and_kept``, ...) — the ones the patch must leave untouched.
    """
    block = {'topic': topic}
    block.update(block_extras)
    return {
        'id': task_id,
        'project_id': project_id,
        'project_root': project_root,
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': True,
            GATE_METADATA_KEY: block,
        },
    }


def task_listing(project_id: str, project_root: str, *tasks: dict) -> dict:
    """One ``get_tasks`` response in the measured live shape."""
    return {'tasks': list(tasks), 'project_id': project_id, 'project_root': project_root}


def census_client(answers: dict[str, object]) -> MagicMock:
    """A client whose ``get_tasks`` answers per root: a response, or an exception.

    Answers are deep-copied, as a JSON transport would hand them over, so the
    census can never alias the double's own state.
    """
    async def _call_tool(name, payload):
        assert name == 'get_tasks', name
        answer = answers[payload['project_root']]
        if isinstance(answer, BaseException):
            raise answer
        return copy.deepcopy(answer)

    client = MagicMock()
    client.call_tool = AsyncMock(side_effect=_call_tool)
    return client


_UNSET = object()


def write_boundary_service(
    *,
    record: object = _UNSET,
    update_response: object = _UNSET,
) -> AsyncMock:
    """A stateless ``MemoryService`` double for the write boundary.

    Children are configured through ``.return_value`` rather than reassigned
    to fresh ``AsyncMock``s, because a reassigned child stops propagating into
    the parent's ``mock_calls`` — and the ORDER of the live re-read relative to
    the write is one of the things this suite has to pin.
    """
    service = AsyncMock()
    service.get_memory_by_id.return_value = (
        {'id': 'm1', 'metadata': {'topic': 'legacy_topic'}}
        if record is _UNSET else record
    )
    if update_response is _UNSET:
        service.update_memory.side_effect = lambda **kwargs: {
            'status': 'updated',
            'store': 'mem0',
            'id': kwargs['memory_id'],
            'content_amended': False,
            'metadata_patched': True,
        }
    else:
        service.update_memory.return_value = update_response
    return service


def mcp_client(*, result: object = None, error: Exception | None = None) -> MagicMock:
    """A ``FusedMemoryClient`` double.  Injected so no test opens a socket."""
    client = MagicMock()
    if error is not None:
        client.call_tool = AsyncMock(side_effect=error)
    else:
        client.call_tool = AsyncMock(
            return_value=result if result is not None
            else {'id': '4220', 'updated': True, 'message': 'ok'}
        )
    return client


def result_row(outcome: str, **overrides) -> dict:
    row = {
        'project_id': 'dark_factory',
        'memory_id': 'm1',
        'old_topic': 'legacy_topic',
        'new_topic': 'legacy-topic',
        'outcome': outcome,
    }
    row.update(overrides)
    return row


def refuse_writes_to(service: MagicMock, *memory_ids: str) -> None:
    """Make *service* refuse every write to *memory_ids*, forwarding the rest."""
    forward = service.update_memory.side_effect
    refused = set(memory_ids)

    async def _update(**kwargs):
        if kwargs['memory_id'] in refused:
            return {'error_type': 'write_refused', 'error': 'refused by test'}
        return await forward(**kwargs)

    service.update_memory = AsyncMock(side_effect=_update)


def fail_writes_moving_back_to(service: MagicMock, memory_id: str, legacy_topic: str) -> None:
    """Make *service* raise on the write that would put *memory_id* back on *legacy_topic*.

    That write is the rollback of a rename, so this is how a failed undo is
    staged; every other write is forwarded.
    """
    forward = service.update_memory.side_effect

    async def _update(**kwargs):
        if (kwargs['memory_id'], kwargs['metadata_patch'].get('topic')) == (
                memory_id, legacy_topic):
            raise RuntimeError('undo failed')
        return await forward(**kwargs)

    service.update_memory = AsyncMock(side_effect=_update)
