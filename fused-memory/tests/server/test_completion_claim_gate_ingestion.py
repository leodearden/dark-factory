"""Integration tests for the completion-claim verification gate on episode
ingestion (task 3142, PRD leaf pi / contract C4).

The gate is the code-level enforcement of the "Terminal-State Pre-Check
Discipline" that until now existed only as Stage-1 prompt prose. Reify
escalation ``esc-5603-1`` is the motivating incident: an episode asserting a
fix "has been applied" for a still-in-progress task was fanned out by
Graphiti's extraction pipeline into FIVE false edges. ``esc-3085-1`` extended
the scope to filing/dispatch claims naming a ticket that does not exist, and
across projects.

CONTRACT, and the two ways it differs from its closest sibling
(``_premature_completion_block``, task 2824):

* It TAGS, never rejects. The episode is always ingested; a non-verified claim
  only adds ``unverified_claim=True`` to the service call (which rides through
  to the Graphiti ``source_description`` prefix and every derived Mem0 fact's
  metadata) plus a structured flag on the tool response.
* It applies to ALL writers, not only ``recon-stage-`` agents.

The content used throughout says "has been applied" deliberately: `applied` is
NOT in ``task_filter.PRESENT_TENSE_COMPLETION_RE``'s vocabulary, so the 2824
gate cannot fire on it and these tests observe THIS gate in isolation even for
a recon-stage agent id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

# An applied-work completion claim naming task 5422 — the esc-5603-1 shape.
_APPLIED_CONTENT = "task 5422's de-flake fix has been applied"
_PROJECT_ID = 'dark_factory'
_KNOWN_PROJECTS = {'dark_factory': '/root'}


def _episode_service():
    """An AsyncMock memory_service whose add_episode returns a dict-dumpable
    result (so the tool's `return result.model_dump()` yields a real dict).
    """
    mock_service = AsyncMock()
    _ep_result = MagicMock()
    _ep_result.model_dump.return_value = {'id': 'ep'}
    mock_service.add_episode.return_value = _ep_result
    return mock_service


def _server(
    mock_service,
    *,
    statuses: dict | None = None,
    known_projects: dict | None = None,
):
    """Build a hermetic server whose task_interceptor answers `statuses`.

    `statuses` is the string-keyed {id: status} map real get_statuses returns.
    get_ticket_row is stubbed to the "no such ticket" answer; tests that care
    about tickets override it.
    """
    task_interceptor = MagicMock()
    task_interceptor.get_statuses = AsyncMock(return_value=statuses or {})
    task_interceptor.get_ticket_row = AsyncMock(return_value=None)
    return create_mcp_server(
        mock_service,
        task_interceptor=task_interceptor,
        known_projects=_KNOWN_PROJECTS if known_projects is None else known_projects,
    )


def _service_kwargs(mock_service) -> dict:
    """The kwargs memory_service.add_episode was called with."""
    return mock_service.add_episode.call_args.kwargs


class TestAddEpisodeUnverifiedClaimTagging:
    """PRIMARY SIGNAL (first half): a completion claim naming a task whose LIVE
    status is non-terminal is INGESTED and TAGGED — never rejected.
    """

    @pytest.mark.asyncio
    async def test_tags_applied_work_claim_for_in_progress_task(self):
        """The acceptance case. Episode claims task 5422's fix "has been
        applied"; task 5422 is live in-progress. The episode must still be
        ingested (add_episode awaited exactly once, no error dict), the service
        call must carry unverified_claim=True, and the tool response must carry
        a structured flag naming the claim text, the ref, and the OBSERVED live
        status.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, (
            f'The gate must TAG, never reject — got an error dict: {result!r}'
        )
        mock_service.add_episode.assert_awaited_once()
        assert _service_kwargs(mock_service).get('unverified_claim') is True, (
            'memory_service.add_episode must be called with unverified_claim=True '
            f'so the tag reaches the Graphiti/Mem0 artefacts; got kwargs: '
            f'{_service_kwargs(mock_service)!r}'
        )

        flag = result.get('unverified_claim')
        assert isinstance(flag, dict), (
            f"Expected a structured 'unverified_claim' flag on the response, got: {result!r}"
        )
        assert flag.get('tag') == 'unverified_claim', f'Unexpected flag shape: {flag!r}'
        claims = flag.get('claims')
        assert isinstance(claims, list) and len(claims) == 1, (
            f'Expected exactly one flagged claim, got: {flag!r}'
        )
        entry = claims[0]
        assert entry.get('ref') == '5422', f'Flag must name the ref; got: {entry!r}'
        assert entry.get('subject') == 'task', f'Expected subject=task; got: {entry!r}'
        assert entry.get('kind') == 'applied_work', f'Expected kind=applied_work; got: {entry!r}'
        assert entry.get('project_id') == _PROJECT_ID, (
            f"Flag must name the claim's resolved project; got: {entry!r}"
        )
        assert entry.get('status') == 'mismatch', (
            f'A live non-terminal status CONTRADICTS the claim; got: {entry!r}'
        )
        assert entry.get('observed') == 'in-progress', (
            f'Flag must record the OBSERVED live status verbatim (INV-2); got: {entry!r}'
        )
        assert 'has been applied' in entry.get('text', ''), (
            f'Flag must quote the claiming clause; got: {entry!r}'
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
    async def test_gate_is_not_recon_stage_scoped(self, agent_id):
        """Unlike the 2824 premature-completion gate, this one is NOT wrapped in
        an `agent_id.startswith('recon-stage-')` guard: a false completion claim
        does the same corpus damage whoever writes it. Every writer — recon
        stage, task implementer, interactive, or an unset agent_id — gets the
        same tag.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'in-progress'})

        args = {'content': _APPLIED_CONTENT, 'project_id': _PROJECT_ID}
        if agent_id is not None:
            args['agent_id'] = agent_id
        result = await server._tool_manager.call_tool('add_episode', args)

        assert 'error' not in result, (
            f'The gate must never reject (agent_id={agent_id!r}); got: {result!r}'
        )
        mock_service.add_episode.assert_awaited_once()
        assert _service_kwargs(mock_service).get('unverified_claim') is True, (
            f'Expected the tag for agent_id={agent_id!r}; got kwargs: '
            f'{_service_kwargs(mock_service)!r}'
        )
        assert result.get('unverified_claim', {}).get('claims'), (
            f'Expected a flag on the response for agent_id={agent_id!r}; got: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_status_read_is_scoped_to_the_claimed_project_root(self):
        """The status read must go to the claimed project's root from the
        known_projects registry, batched over the claimed ids — the same read
        shape _premature_completion_block uses.
        """
        mock_service = _episode_service()
        task_interceptor = MagicMock()
        task_interceptor.get_statuses = AsyncMock(return_value={'5422': 'in-progress'})
        task_interceptor.get_ticket_row = AsyncMock(return_value=None)
        server = create_mcp_server(
            mock_service,
            task_interceptor=task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        task_interceptor.get_statuses.assert_awaited_once()
        kwargs = task_interceptor.get_statuses.call_args.kwargs
        assert kwargs.get('project_root') == _KNOWN_PROJECTS[_PROJECT_ID], (
            f'Status read must target the claimed project root; got: {kwargs!r}'
        )
        assert list(kwargs.get('ids') or []) == ['5422'], (
            f'Status read must be batched over the claimed ids; got: {kwargs!r}'
        )
