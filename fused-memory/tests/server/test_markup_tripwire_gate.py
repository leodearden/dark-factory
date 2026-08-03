"""Boundary tests for the MCP-markup write tripwire (task 3141).

Wiring tests for the four fused-memory write tools — add_memory, add_episode,
submit_task, update_task — asserting that a write carrying raw MCP envelope
markup is REJECTED before it reaches the service/interceptor. The pure helpers
and the storm counter are unit-tested in the sibling ``test_markup_tripwire.py``.

Harness: an AsyncMock service (plus AsyncMock task_interceptor for the task
tools) wired through create_mcp_server and invoked via
``server._tool_manager.call_tool(...)`` — the same shape as
``test_add_memory_near_duplicate_gate.py`` and
``test_tools_validation.py::TestSubmitTaskPremiseLintGuard``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.markup_tripwire import (
    _MARKUP_STORM_THRESHOLD,
    _MARKUP_STORM_WINDOW_SECONDS,
)
from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'

# One specimen per literal in MCP_MARKUP_PATTERNS. The '</content>\n</invoke>'
# tail is the shape of the real corpus specimens (DF 3083 vector 1); the
# '<parameter name=' fragment is vector 2, the one that mis-parsed task 3210's
# priority silently.
_LEAKED_CONTENT = 'a real memory sentence\n</content>\n</invoke>'
_LEAKED_INVOKE = 'a real memory sentence </invoke>'
_LEAKED_PARAMETER = 'a real memory sentence <parameter name="priority">high</parameter>'
_CLEAN_CONTENT = 'a perfectly ordinary memory about the merge lane'


def _pass_through(mock_service: AsyncMock, method: str) -> None:
    """Give *method*'s return value a real ``model_dump``.

    An unspecced AsyncMock chains AsyncMock all the way down, so
    ``result.model_dump()`` would be an unawaited coroutine unless the return
    value is an explicit MagicMock (mirrors
    test_add_memory_near_duplicate_gate.py::_configure_pass_through_add_memory).
    """
    result = MagicMock()
    result.model_dump.return_value = {'id': 'ok'}
    getattr(mock_service, method).return_value = result


def _parse(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict.

    Mirrors test_tools_validation.py::_parse_tool_result — the task tools return
    through a different FastMCP path than the memory tools.
    """
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest.fixture
def task_server(monkeypatch):
    """A server whose task tools accept a synthetic project_root.

    '/project' is not a real git working tree, so resolve_main_checkout is
    stubbed to pass it through — the same passthrough
    TestSubmitTaskPremiseLintGuard uses.
    """
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))
    interceptor = AsyncMock()
    interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
    interceptor.update_task.return_value = {'ok': True}
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    return server, interceptor


@pytest.fixture
def mixed_server(monkeypatch):
    """One server exposing BOTH the memory tools and the task tools.

    Needed for the storm tests specifically: the counter is meant to be shared
    across all four write boundaries, which can only be observed if rejections
    from different tools land on the same server instance.
    """
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))
    mock_service = AsyncMock()
    _pass_through(mock_service, 'add_memory')
    _pass_through(mock_service, 'add_episode')
    interceptor = AsyncMock()
    interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
    interceptor.update_task.return_value = {'ok': True}
    server = create_mcp_server(mock_service, task_interceptor=interceptor)
    return server, mock_service, interceptor


async def _reject_add_memory(server, agent_id: str = 'a') -> dict:
    """Drive one add_memory rejection and return the parsed block dict."""
    return _parse(await server._tool_manager.call_tool(
        'add_memory',
        {
            'content': _LEAKED_INVOKE,
            'category': 'observations_and_summaries',
            'agent_id': agent_id,
            'project_id': _PROJECT_ID,
        },
    ))


async def _reject_submit_task(
    server, agent_id: str = 'a', project_root: str = '/project'
) -> dict:
    """Drive one submit_task rejection and return the parsed block dict."""
    return _parse(await server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': project_root,
            'prompt': 'a clean prompt',
            'description': f'dirty {_LEAKED_INVOKE}',
            'agent_id': agent_id,
        },
    ))


def _assert_markup_block(result: object, *, field: str, pattern: str, agent_id: str) -> None:
    """Assert *result* is the structured markup rejection (INV-1)."""
    assert isinstance(result, dict), f'Expected a dict, got {type(result)}: {result!r}'
    assert result.get('error') == 'mcp_markup_write_blocked', f'got: {result!r}'
    assert result.get('error_type') == 'McpEnvelopeMarkupWriteRejected', f'got: {result!r}'
    assert result.get('field') == field, f'expected field={field!r}, got: {result!r}'
    assert result.get('matched_pattern') == pattern, (
        f'expected matched_pattern={pattern!r}, got: {result!r}'
    )
    assert result.get('agent_id') == agent_id, f'expected agent_id echoed, got: {result!r}'
    # Only that a hint reached the caller — the WORDING (that it names DF 3083
    # and the override key) is pinned once, in
    # test_markup_tripwire.py::test_hint_names_df_3083_and_the_override_key.
    # Repeating the prose assertion through every gate test would red the whole
    # suite for a legitimate rewrite once 3083 lands and the pointer changes.
    assert result.get('hint'), f'expected a non-empty hint, got: {result!r}'


class TestAddMemoryMarkupGate:
    """add_memory rejects leaked envelope markup in content."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern'),
        [
            (_LEAKED_CONTENT, '</content>'),
            (_LEAKED_INVOKE, '</invoke>'),
            (_LEAKED_PARAMETER, '<parameter name='),
        ],
    )
    async def test_rejects_each_pattern_without_calling_the_service(self, content, pattern):
        """Every literal is rejected, and the write never reaches the store.

        assert_not_called is the load-bearing assertion: a block dict returned
        AFTER the memory landed would leave the specimen in the corpus anyway.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': content,
                'category': 'observations_and_summaries',
                'agent_id': 'claude-task-3141',
                'project_id': _PROJECT_ID,
            },
        )

        _assert_markup_block(
            result, field='content', pattern=pattern, agent_id='claude-task-3141'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_excerpt_echoes_the_rejected_content(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
            },
        )
        assert result.get('content_excerpt') == _LEAKED_INVOKE[:200]

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CLEAN_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
            },
        )
        assert result == {'id': 'ok'}, f'expected the write to pass through, got: {result!r}'
        mock_service.add_memory.assert_called_once()
        assert mock_service.add_memory.call_args.kwargs['content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self):
        """metadata={'allow_mcp_markup': True} is the documented escape hatch.

        Required for correctness, not convenience: DF 3083's own description
        quotes all three literals in prose, so without this the sibling task this
        containment leaf exists to feed could never be written about.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': True, 'source': 'doc'},
            },
        )
        assert result == {'id': 'ok'}, f'expected the override to allow the write, got: {result!r}'
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_flag_is_stripped_before_persistence(self):
        """The flag is a write-time control, not corpus metadata.

        Mirrors how add_memory already pops 'allow_near_duplicate' — if it were
        persisted it would enter the stored metadata vocabulary and, worse, ride
        along on future reads of a memory that only ever needed it once.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': True, 'source': 'doc'},
            },
        )
        forwarded = mock_service.add_memory.call_args.kwargs['metadata']
        assert 'allow_mcp_markup' not in (forwarded or {}), (
            f'the override flag must not be persisted, got: {forwarded!r}'
        )
        assert (forwarded or {}).get('source') == 'doc', (
            f'other metadata keys must survive the strip, got: {forwarded!r}'
        )

    @pytest.mark.asyncio
    async def test_a_non_true_override_value_does_not_bypass_the_gate(self):
        """Fail-closed at the boundary too, not just in the helper."""
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': 'yes'},
            },
        )
        _assert_markup_block(result, field='content', pattern='</invoke>', agent_id='a')
        mock_service.add_memory.assert_not_called()


class TestAddEpisodeMarkupGate:
    """add_episode rejects leaked envelope markup in content."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern'),
        [
            (_LEAKED_CONTENT, '</content>'),
            (_LEAKED_INVOKE, '</invoke>'),
            (_LEAKED_PARAMETER, '<parameter name='),
        ],
    )
    async def test_rejects_each_pattern_without_calling_the_service(self, content, pattern):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': content,
                'agent_id': 'claude-task-3141',
                'project_id': _PROJECT_ID,
            },
        )

        _assert_markup_block(
            result, field='content', pattern=pattern, agent_id='claude-task-3141'
        )
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'content': _CLEAN_CONTENT, 'agent_id': 'a', 'project_id': _PROJECT_ID},
        )
        assert result == {'id': 'ok'}, f'expected the write to pass through, got: {result!r}'
        mock_service.add_episode.assert_called_once()
        assert mock_service.add_episode.call_args.kwargs['content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _LEAKED_INVOKE,
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': True},
            },
        )
        assert result == {'id': 'ok'}, f'expected the override to allow the write, got: {result!r}'
        mock_service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_markup_is_rejected_before_the_recon_stage_content_guards(self):
        """Ordering: the tripwire fires FIRST, so markup never reaches those guards.

        A recon-stage agent writing markup-bearing content must get the markup
        rejection, not a mixed-temporal-framing/snapshot verdict derived from
        text that is partly serialized envelope. Those heuristics read prose;
        feeding them a leaked tail would let the leak masquerade as a different
        (and misleading) violation.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        # Content that trips BOTH the markup tripwire and mixed-temporal framing.
        content = (
            'As of now there are 5 pending tasks; historically there were 9.\n'
            '</content>\n</invoke>'
        )
        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': content,
                'agent_id': 'recon-stage-1',
                'project_id': _PROJECT_ID,
            },
        )
        assert result.get('error') == 'mcp_markup_write_blocked', (
            f'expected the markup tripwire to win the ordering, got: {result!r}'
        )
        mock_service.add_episode.assert_not_called()


class TestSubmitTaskMarkupGate:
    """submit_task rejects leaked envelope markup in any of its four text fields."""

    @pytest.mark.asyncio
    async def test_rejects_markup_in_description_before_the_interceptor(self, task_server):
        """The DF 3083 vector-2 case, previously a SILENT mis-parse.

        A '<parameter name="priority">' fragment in a description reached the
        interceptor's description parser, which derived the wrong value from it
        without complaint (reify task 3210 was filed priority=high and stored as
        medium). Loud rejection ahead of that parser is the whole point.
        """
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Reconcile task 7',
                'description': f'Reconcile task 7.\n{_LEAKED_PARAMETER}',
                'agent_id': 'claude-task-3141',
            },
        ))

        _assert_markup_block(
            result, field='description', pattern='<parameter name=', agent_id='claude-task-3141'
        )
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    async def test_rejects_markup_in_each_scanned_field(self, task_server, field):
        """All four text fields are scanned, so no channel is left unchecked.

        premise_lint_guard already lints exactly these four at this boundary for
        the same reason: they all flow into the same description parser, so
        checking description alone would leave three ways in.
        """
        server, interceptor = task_server
        args = {
            'project_root': '/project',
            'prompt': 'a clean prompt',
            'description': 'a clean description',
            'agent_id': 'a',
        }
        args[field] = f'dirty {_LEAKED_INVOKE}'

        result = _parse(await server._tool_manager.call_tool('submit_task', args))

        _assert_markup_block(result, field=field, pattern='</invoke>', agent_id='a')
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_submit_task_reaches_the_interceptor(self, task_server):
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Reconcile task 7',
                'description': 'Reconcile task 7 status against the knowledge graph.',
                'agent_id': 'a',
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.submit_task.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True])
    async def test_override_lets_deliberate_markup_through(self, task_server, as_json):
        """Both metadata shapes honour the override.

        submit_task accepts metadata as an object OR a JSON string; an author
        documenting the leak (as DF 3083's own description does) must get through
        whichever shape they used.
        """
        server, interceptor = task_server
        metadata = {'allow_mcp_markup': True, 'execution_class': 'code_tdd'}

        result = _parse(await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Document the leak',
                'description': f'The leak looks like {_LEAKED_INVOKE}',
                'agent_id': 'a',
                'metadata': json.dumps(metadata) if as_json else metadata,
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.submit_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_flag_is_not_persisted_into_task_metadata(self, task_server):
        """'allow_mcp_markup' is a write-time control, not task metadata.

        It is deliberately absent from the task metadata vocabulary
        (docs/task-authoring.md), so letting it through would mint an
        unrecognised key on every task written with the override.
        """
        server, interceptor = task_server

        await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Document the leak',
                'description': f'The leak looks like {_LEAKED_INVOKE}',
                'agent_id': 'a',
                'metadata': {'allow_mcp_markup': True, 'execution_class': 'code_tdd'},
            },
        )
        forwarded = interceptor.submit_task.call_args.kwargs.get('metadata')
        assert 'allow_mcp_markup' not in json.dumps(forwarded), (
            f'the override flag must not reach the interceptor, got: {forwarded!r}'
        )
        assert 'code_tdd' in json.dumps(forwarded), (
            f'other metadata must survive the strip, got: {forwarded!r}'
        )


class TestUpdateTaskMarkupGate:
    """update_task rejects leaked envelope markup in any of its scanned fields."""

    @pytest.mark.asyncio
    async def test_rejects_markup_in_description_before_the_interceptor(self, task_server):
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '3141',
                'project_root': '/project',
                'description': f'updated.\n{_LEAKED_CONTENT}',
                'agent_id': 'claude-task-3141',
            },
        ))

        _assert_markup_block(
            result, field='description', pattern='</content>', agent_id='claude-task-3141'
        )
        interceptor.update_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    async def test_rejects_markup_in_each_scanned_field(self, task_server, field):
        server, interceptor = task_server
        args = {'id': '3141', 'project_root': '/project', 'agent_id': 'a'}
        args[field] = f'dirty {_LEAKED_INVOKE}'

        result = _parse(await server._tool_manager.call_tool('update_task', args))

        _assert_markup_block(result, field=field, pattern='</invoke>', agent_id='a')
        interceptor.update_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_update_task_reaches_the_interceptor(self, task_server):
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '3141',
                'project_root': '/project',
                'description': 'a clean updated description',
                'agent_id': 'a',
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.update_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self, task_server):
        """The case that makes the hatch load-bearing: updating DF 3083 itself.

        3083's description quotes all three literals, so without this an
        update_task on the very sibling this leaf feeds would be permanently
        blocked.
        """
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '3083',
                'project_root': '/project',
                'description': f'The leak emits {_LEAKED_CONTENT}',
                'agent_id': 'a',
                'metadata': {'allow_mcp_markup': True},
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.update_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_flag_is_not_persisted_into_task_metadata(self, task_server):
        """The mirror of the submit_task assertion — stickier at THIS boundary.

        update_task's metadata goes through a shallow last-write-wins merge into
        the task's EXISTING blob (or an additive merge under
        metadata_mode='additive'), so an unstripped flag would not merely be
        minted on a fresh task: it would be grafted permanently onto a live one,
        outliving the single write that set it.
        """
        server, interceptor = task_server

        await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '3083',
                'project_root': '/project',
                'description': f'The leak emits {_LEAKED_CONTENT}',
                'agent_id': 'a',
                'metadata': {'allow_mcp_markup': True, 'execution_class': 'code_tdd'},
            },
        )
        forwarded = interceptor.update_task.call_args.kwargs.get('metadata')
        assert 'allow_mcp_markup' not in json.dumps(forwarded), (
            f'the override flag must not reach the interceptor, got: {forwarded!r}'
        )
        assert 'code_tdd' in json.dumps(forwarded), (
            f'other metadata must survive the strip, got: {forwarded!r}'
        )


class TestMarkupStormIntegration:
    """A BURST of rejections escalates instead of being absorbed silently (INV-4).

    One bounced write is routine — an agent retries and moves on. What must not
    be swallowed is a *stream* of them, because that means the upstream harness
    serialization leak is running right now and every write it touches is a
    would-be corpus specimen. These tests pin the wiring: rejections from any of
    the four boundaries feed ONE per-server counter, the crossing response
    carries the machine-readable storm block, and the best-effort escalation is
    attempted exactly once without ever changing the rejection outcome.
    """

    @pytest.mark.asyncio
    async def test_counter_is_shared_across_the_write_boundaries(self, mixed_server):
        """Two add_memory rejections plus one submit_task rejection trip the threshold.

        The mixed tool sequence is the point: a leak does not politely confine
        itself to one tool, so a per-tool counter would need `threshold` hits on
        the *same* boundary before anyone heard about it.
        """
        server, _mock_service, _interceptor = mixed_server
        assert _MARKUP_STORM_THRESHOLD == 3, (
            'this test drives exactly 3 rejections; update it alongside the constant'
        )

        first = await _reject_add_memory(server)
        second = await _reject_add_memory(server)
        third = await _reject_submit_task(server)

        assert 'storm' not in first, f'below threshold must carry no storm key: {first!r}'
        assert 'storm' not in second, f'below threshold must carry no storm key: {second!r}'

        storm = third.get('storm')
        assert storm is not None, f'threshold-crossing rejection must report a storm: {third!r}'
        assert storm.get('count') == _MARKUP_STORM_THRESHOLD, f'got: {storm!r}'
        assert storm.get('threshold') == _MARKUP_STORM_THRESHOLD, f'got: {storm!r}'
        assert storm.get('window_seconds') == _MARKUP_STORM_WINDOW_SECONDS, f'got: {storm!r}'
        # Presence only; the storm hint's wording is pinned once, in
        # test_markup_tripwire.py::test_storm_hint_names_df_3083.
        assert storm.get('hint'), f'the storm needs a non-empty hint: {storm!r}'

    @pytest.mark.asyncio
    async def test_below_threshold_rejections_are_still_full_blocks(self, mixed_server):
        """The write is bounced either way — the storm is additive, not a gate.

        A first-offence rejection that returned a thinner dict would make the
        caller's remediation depend on how many *other* agents happened to leak
        recently.
        """
        server, mock_service, _interceptor = mixed_server

        result = await _reject_add_memory(server, agent_id='claude-task-3141')

        _assert_markup_block(
            result, field='content', pattern='</invoke>', agent_id='claude-task-3141'
        )
        assert 'storm' not in result, f'got: {result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_each_server_starts_with_a_fresh_counter(self, monkeypatch):
        """Per-server state, so no counter bleeds between servers (or tests).

        Asserting the SECOND server also fires on its own third rejection is the
        discriminator: with a module-global counter the first server's fire would
        still be inside the rate-limit window, so the second server's burst would
        be suppressed entirely.
        """
        monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))

        def _build():
            interceptor = AsyncMock()
            interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
            return create_mcp_server(AsyncMock(), task_interceptor=interceptor)

        first_server = _build()
        for _ in range(_MARKUP_STORM_THRESHOLD - 1):
            assert 'storm' not in await _reject_submit_task(first_server)
        assert 'storm' in await _reject_submit_task(first_server), 'server A should have fired'

        second_server = _build()
        for i in range(_MARKUP_STORM_THRESHOLD - 1):
            below = await _reject_submit_task(second_server)
            assert 'storm' not in below, f'server B rejection {i + 1} fired early: {below!r}'
        crossing = await _reject_submit_task(second_server)
        assert 'storm' in crossing, (
            f'server B must fire on its OWN third rejection, not inherit A: {crossing!r}'
        )

    @pytest.mark.asyncio
    async def test_escalation_is_emitted_once_when_the_storm_fires(
        self, mixed_server, monkeypatch
    ):
        """The queue write happens on the burst only, and its id is echoed back.

        Folding the escalation id into the storm block is what lets the rejected
        caller (or a reviewer reading the response) find the filed escalation
        without grepping logs.
        """
        server, _mock_service, _interceptor = mixed_server
        calls: list[tuple] = []

        def _fake_emit(project_root, storm):
            calls.append((project_root, storm))
            return 'esc-markup-tripwire-1'

        monkeypatch.setattr(
            'fused_memory.server.tools.emit_markup_storm_escalation', _fake_emit
        )

        for _ in range(_MARKUP_STORM_THRESHOLD - 1):
            await _reject_add_memory(server)
        assert calls == [], f'no escalation may be filed below threshold: {calls!r}'

        crossing = await _reject_submit_task(server)

        assert len(calls) == 1, f'expected exactly one escalation attempt, got: {calls!r}'
        project_root, storm_arg = calls[0]
        assert project_root == '/project', (
            f'the storm must be escalated against the project_root, got: {project_root!r}'
        )
        assert storm_arg.get('count') == _MARKUP_STORM_THRESHOLD, f'got: {storm_arg!r}'
        assert crossing.get('storm', {}).get('escalation_id') == 'esc-markup-tripwire-1', (
            f'the filed escalation id must be echoed in the storm block: {crossing!r}'
        )

    @pytest.mark.asyncio
    async def test_every_project_in_the_window_is_escalated_not_only_the_crossing_one(
        self, mixed_server, monkeypatch
    ):
        """Two rejections from project A plus one from B must escalate BOTH.

        One counter serves every project on the server, so escalating only the
        write that crossed the threshold would file the whole burst into B's
        queue — naming B as the leaker on the strength of A's rejections — and
        then, because the rate limit is shared too, A's next rejections inside
        the same window would fire nothing at all. The actively-leaking project
        would never get an escalation.
        """
        server, _mock_service, _interceptor = mixed_server
        assert _MARKUP_STORM_THRESHOLD == 3, (
            'this test drives exactly 3 rejections; update it alongside the constant'
        )
        calls: list[tuple] = []

        def _fake_emit(project_root, storm):
            calls.append((project_root, storm))
            return f'esc-markup-tripwire-{len(calls)}'

        monkeypatch.setattr(
            'fused_memory.server.tools.emit_markup_storm_escalation', _fake_emit
        )

        await _reject_submit_task(server, project_root='/project-a')
        await _reject_submit_task(server, project_root='/project-a')
        crossing = await _reject_submit_task(server, project_root='/project-b')

        assert [c[0] for c in calls] == ['/project-a', '/project-b'], (
            f'each distinct project in the window gets its own escalation: {calls!r}'
        )
        storm = crossing.get('storm')
        assert storm is not None, f'the crossing rejection must report a storm: {crossing!r}'
        assert storm.get('projects') == ['/project-a', '/project-b'], (
            f'the response must name every project the burst spanned: {storm!r}'
        )
        assert storm.get('escalation_id') == 'esc-markup-tripwire-2', (
            f"the echoed id is the one filed for THIS caller's project: {storm!r}"
        )

    @pytest.mark.asyncio
    async def test_escalation_failure_does_not_break_the_rejection(
        self, mixed_server, monkeypatch
    ):
        """Escalation is purely additive: the write was already refused.

        emit_markup_storm_escalation is documented never to raise, but the call
        site must not depend on that promise — a queue failure that turned a
        rejection into a 500 would trade containment for an outage.
        """
        server, _mock_service, _interceptor = mixed_server

        def _boom(project_root, storm):
            raise RuntimeError('escalation queue is on fire')

        monkeypatch.setattr('fused_memory.server.tools.emit_markup_storm_escalation', _boom)

        for _ in range(_MARKUP_STORM_THRESHOLD - 1):
            await _reject_add_memory(server)
        crossing = await _reject_submit_task(server, agent_id='claude-task-3141')

        _assert_markup_block(
            crossing, field='description', pattern='</invoke>', agent_id='claude-task-3141'
        )
        assert 'storm' in crossing, (
            f'the burst is still reported even when escalation fails: {crossing!r}'
        )
        assert 'escalation_id' not in crossing['storm'], (
            f'no id to echo when filing failed: {crossing["storm"]!r}'
        )

    @pytest.mark.asyncio
    async def test_storm_is_logged_at_error_with_a_greppable_prefix(
        self, mixed_server, monkeypatch, caplog
    ):
        """The burst must be visible to an operator who never reads MCP responses.

        The MCP response goes to the leaking agent, which is precisely the party
        least likely to act on it, so the ERROR log line is the operator-facing
        half of INV-4 — and the escalation detail tells a reader to grep for
        exactly this prefix.
        """
        server, _mock_service, _interceptor = mixed_server
        monkeypatch.setattr(
            'fused_memory.server.tools.emit_markup_storm_escalation', lambda *_: None
        )

        with caplog.at_level('ERROR'):
            for _ in range(_MARKUP_STORM_THRESHOLD - 1):
                await _reject_add_memory(server)
            errors_below = [r for r in caplog.records if r.levelname == 'ERROR']
            assert errors_below == [], f'no ERROR below threshold, got: {errors_below!r}'

            await _reject_submit_task(server)

        storm_errors = [
            r for r in caplog.records
            if r.levelname == 'ERROR' and 'markup_tripwire_storm' in r.getMessage()
        ]
        assert len(storm_errors) == 1, (
            f'expected one greppable markup_tripwire_storm ERROR line, got: {caplog.text!r}'
        )
