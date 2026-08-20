"""The MCP-markup write boundary for fused-memory's four task/memory write tools.

Originally (task 3141) this file pinned an IN-LINE gate: each of add_memory,
add_episode, submit_task and update_task called a ``_markup_gate`` closure in
its own body and RETURNED a ``build_markup_block`` dict when the write carried
raw MCP envelope markup.

Task 4458 replaced that with ONE mechanism at the dispatch boundary
(:mod:`fused_memory.server.markup_guard`, wrapping ``ToolManager.call_tool``),
which covers every tool and every string argument instead of four hand-listed
call sites, and hands the caller a ``repaired_call`` to resubmit. Two write
tools — add_system_record and update_memory — never had an in-line gate at all;
they are covered now.

So this file is in two halves, and neither is a deletion of the other:

* The first half is the ORIGINAL suite, re-pointed. Every specimen and every
  parametrisation is kept, but each now asserts the post-retirement fact: with
  NO guard installed the write reaches the mock service, because there is
  exactly ONE mechanism and it does not live in the tool body (INV-5, D1) —
  while the same specimen against a GUARDED server is still refused.
* The second half is the parity demonstration the retirement was predicated on:
  the same specimens driven over the retired gate's entire scanned surface, at
  the boundary.

Harness: an AsyncMock service (plus AsyncMock task_interceptor for the task
tools) wired through create_mcp_server and invoked via
``server._tool_manager.call_tool(...)`` — the same shape as
``test_add_memory_near_duplicate_gate.py`` and
``test_tools_validation.py::TestSubmitTaskPremiseLintGuard``. The guard is
installed by ``main.py``, NOT by ``create_mcp_server``, so a server built here
is UNGUARDED unless a test installs it — which is what makes the first half's
assertions meaningful.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import RepairPolicy
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

from fused_memory.server.markup_guard import install_markup_guard
from fused_memory.server.markup_tripwire import (
    _MARKUP_STORM_THRESHOLD,
    _MARKUP_STORM_WINDOW_SECONDS,
)
from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'

# One specimen per literal in the retired gate's pattern list. The
# '</content>\n</invoke>' tail is the shape of the real corpus specimens (DF
# 3083 vector 1); the '<parameter name=' fragment is vector 2, the one that
# mis-parsed task 3210's priority silently. Held VERBATIM through the
# retirement so the boundary is compared against the same corpus the gate was.
_LEAKED_CONTENT = 'a real memory sentence\n</content>\n</invoke>'
_LEAKED_INVOKE = 'a real memory sentence </invoke>'
_LEAKED_PARAMETER = 'a real memory sentence <parameter name="priority">high</parameter>'
_CLEAN_CONTENT = 'a perfectly ordinary memory about the merge lane'

#: The in-line gate's rejection vocabulary. Nothing may return it any more:
#: that spelling existing at all would mean a second mechanism came back.
_RETIRED_ERROR_TYPE = 'McpEnvelopeMarkupWriteRejected'
_RETIRED_ERROR = 'mcp_markup_write_blocked'


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


def _assert_no_in_line_rejection(result: object) -> dict:
    """Assert nothing in the TOOL BODY refused this write for markup.

    The load-bearing assertion of the whole first half. It is deliberately
    phrased as 'no in-line markup rejection' rather than 'no error at all':
    other guards in these tool bodies (premise lint, recon-stage content
    guards, the near-duplicate gate) legitimately still run and may have
    opinions about a given payload. What must be gone is THIS mechanism.
    """
    parsed = _parse(result)
    if isinstance(parsed, dict):
        assert parsed.get('error') != _RETIRED_ERROR, (
            f'a second markup mechanism is back in the tool body: {parsed!r}'
        )
        assert parsed.get('error_type') != _RETIRED_ERROR_TYPE, (
            f'a second markup mechanism is back in the tool body: {parsed!r}'
        )
    return parsed if isinstance(parsed, dict) else {}


def _boundary_payload(exc_info) -> dict:
    """The guard's structured refusal, parsed out of the raised ToolError."""
    return json.loads(str(exc_info.value))


def _assert_boundary_refused(exc_info, *, field: str, pattern: str) -> dict:
    """Assert the surviving mechanism refused, in ITS vocabulary.

    Two error_types are contracted — ``mcp_markup_detected`` when the residue
    parses and a repair is offered, ``mcp_markup_unrepairable`` when it does not
    and the residue is escalated instead. Both are refusals; neither is the
    retired spelling. Which one a given specimen earns is pinned per specimen in
    the second half of this file.
    """
    payload = _boundary_payload(exc_info)
    assert payload['error_type'] in ('mcp_markup_detected', 'mcp_markup_unrepairable'), (
        f'got: {payload!r}'
    )
    assert payload['error_type'] != _RETIRED_ERROR_TYPE, f'got: {payload!r}'
    assert payload['field'] == field, f'got: {payload!r}'
    assert payload['matched_pattern'] == pattern, f'got: {payload!r}'
    return payload


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

    Needed for the storm tests specifically: one counter serves every write
    boundary on a server, which can only be observed if rejections from
    different tools land on the same server instance.
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


@pytest.fixture
def memory_server():
    """The memory-tool server the original per-test bodies built inline."""
    mock_service = AsyncMock()
    _pass_through(mock_service, 'add_memory')
    _pass_through(mock_service, 'add_episode')
    return create_mcp_server(mock_service), mock_service


class TestAddMemoryOnlyTheBoundaryRejects:
    """add_memory has no markup mechanism of its own left.

    Every assertion below was previously 'the tool body returns a markup block'.
    It is now the pair that replaces it: unguarded, the write reaches the store,
    because the only mechanism lives at the dispatch boundary; guarded, the same
    specimen is refused before the tool body is entered at all.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern'),
        [
            (_LEAKED_CONTENT, '</content>'),
            (_LEAKED_INVOKE, '</invoke>'),
            (_LEAKED_PARAMETER, '<parameter name='),
        ],
    )
    async def test_only_the_boundary_rejects_each_pattern(
        self, memory_server, content, pattern
    ):
        server, mock_service = memory_server
        arguments = {
            'content': content,
            'category': 'observations_and_summaries',
            'agent_id': 'claude-task-3141',
            'project_id': _PROJECT_ID,
        }

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('add_memory', arguments)
        )
        mock_service.add_memory.assert_called_once()

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        mock_service.add_memory.reset_mock()
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('add_memory', arguments)

        _assert_boundary_refused(exc_info, field='content', pattern=pattern)
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_refusal_locates_the_damage_for_the_caller(self, memory_server):
        """Re-pointed from the gate's ``content_excerpt`` echo.

        The gate could only hand back the first 200 characters of the offending
        string. The boundary names the tool and the field AND, when the residue
        parses, hands back the CLEANED value inside repaired_call — strictly
        more than the excerpt, and resubmittable rather than merely diagnostic.
        """
        server, _mock_service = memory_server
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {
                    'content': _LEAKED_CONTENT,
                    'category': 'observations_and_summaries',
                    'agent_id': 'a',
                    'project_id': _PROJECT_ID,
                },
            )

        payload = _assert_boundary_refused(
            exc_info, field='content', pattern='</content>'
        )
        assert payload['tool'] == 'add_memory', f'got: {payload!r}'
        assert payload['repaired_call']['content'] == 'a real memory sentence\n'

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self, memory_server):
        server, mock_service = memory_server

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
    async def test_override_lets_deliberate_markup_through(self, memory_server):
        """metadata={'allow_mcp_markup': True} is the documented escape hatch.

        Required for correctness, not convenience: DF 3083's own description
        quotes all three literals in prose, so without this the sibling task this
        containment leaf exists to feed could never be written about. Now
        exercised against the GUARDED server, which is the party that reads it.
        """
        server, mock_service = memory_server
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )

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
    async def test_override_flag_is_stripped_before_persistence(self, memory_server):
        """The flag is a write-time control, not corpus metadata.

        Mirrors how add_memory already pops 'allow_near_duplicate' — if it were
        persisted it would enter the stored metadata vocabulary and, worse, ride
        along on future reads of a memory that only ever needed it once.

        The TOOL BODY remains the party that strips it, which is why
        ``strip_markup_override`` outlived the gate: the guard forwards the flag
        UNCHANGED to a tool that declares a ``metadata`` parameter, so deleting
        the strip would persist a control flag into the corpus.
        """
        server, mock_service = memory_server
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )

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
    async def test_a_non_true_override_value_does_not_bypass_the_boundary(
        self, memory_server
    ):
        """Fail-closed, re-pointed: it is the BOUNDARY that must not be fooled.

        Unguarded there is nothing left to fool — which is itself the retirement
        being asserted — so the discriminating half runs against the guard.
        """
        server, mock_service = memory_server
        arguments = {
            'content': _LEAKED_INVOKE,
            'category': 'observations_and_summaries',
            'agent_id': 'a',
            'project_id': _PROJECT_ID,
            'metadata': {'allow_mcp_markup': 'yes'},
        }

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('add_memory', arguments)
        )

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        mock_service.add_memory.reset_mock()
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('add_memory', arguments)

        _assert_boundary_refused(exc_info, field='content', pattern='</invoke>')
        mock_service.add_memory.assert_not_called()


class TestAddEpisodeOnlyTheBoundaryRejects:
    """add_episode, the same pair."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern'),
        [
            (_LEAKED_CONTENT, '</content>'),
            (_LEAKED_INVOKE, '</invoke>'),
            (_LEAKED_PARAMETER, '<parameter name='),
        ],
    )
    async def test_only_the_boundary_rejects_each_pattern(
        self, memory_server, content, pattern
    ):
        server, mock_service = memory_server
        arguments = {
            'content': content,
            'agent_id': 'claude-task-3141',
            'project_id': _PROJECT_ID,
        }

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('add_episode', arguments)
        )
        mock_service.add_episode.assert_called_once()

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        mock_service.add_episode.reset_mock()
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('add_episode', arguments)

        _assert_boundary_refused(exc_info, field='content', pattern=pattern)
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self, memory_server):
        server, mock_service = memory_server

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'content': _CLEAN_CONTENT, 'agent_id': 'a', 'project_id': _PROJECT_ID},
        )
        assert result == {'id': 'ok'}, f'expected the write to pass through, got: {result!r}'
        mock_service.add_episode.assert_called_once()
        assert mock_service.add_episode.call_args.kwargs['content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self, memory_server):
        server, mock_service = memory_server
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )

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
    async def test_markup_is_refused_before_the_recon_stage_content_guards(
        self, memory_server
    ):
        """Ordering, now guaranteed by CONSTRUCTION rather than by statement order.

        A recon-stage agent writing markup-bearing content must get the markup
        refusal, not a mixed-temporal-framing/snapshot verdict derived from text
        that is partly serialized envelope. Those heuristics read prose; feeding
        them a leaked tail would let the leak masquerade as a different (and
        misleading) violation.

        The gate won this race by being the first statement in the tool body —
        a property any later edit could reorder away. The boundary wins it
        because the tool body is never entered, which no edit inside the body
        can undo. Unguarded, the same content now reaches those guards, which is
        the retirement itself.
        """
        server, _mock_service = memory_server
        # Content that trips BOTH the markup boundary and mixed-temporal framing.
        content = (
            'As of now there are 5 pending tasks; historically there were 9.\n'
            '</content>\n</invoke>'
        )
        arguments = {
            'content': content,
            'agent_id': 'recon-stage-1',
            'project_id': _PROJECT_ID,
        }

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('add_episode', arguments)
        )

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('add_episode', arguments)

        _assert_boundary_refused(exc_info, field='content', pattern='</content>')


class TestSubmitTaskOnlyTheBoundaryRejects:
    """submit_task, across the four text fields the retired gate hand-listed."""

    @pytest.mark.asyncio
    async def test_only_the_boundary_rejects_the_vector_2_description(self, task_server):
        """The DF 3083 vector-2 case, previously a SILENT mis-parse.

        A '<parameter name="priority">' fragment in a description reached the
        interceptor's description parser, which derived the wrong value from it
        without complaint (reify task 3210 was filed priority=high and stored as
        medium). Loud refusal ahead of that parser is the whole point, and it is
        now the boundary that delivers it.
        """
        server, interceptor = task_server
        arguments = {
            'project_root': '/project',
            'prompt': 'Reconcile task 7',
            'description': f'Reconcile task 7.\n{_LEAKED_PARAMETER}',
            'agent_id': 'claude-task-3141',
        }

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('submit_task', arguments)
        )
        interceptor.submit_task.assert_called_once()

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        interceptor.submit_task.reset_mock()
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('submit_task', arguments)

        _assert_boundary_refused(
            exc_info, field='description', pattern='<parameter name='
        )
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    async def test_only_the_boundary_rejects_each_scanned_field(self, task_server, field):
        """All four text fields, kept parametrised.

        premise_lint_guard already lints exactly these four at this boundary for
        the same reason: they all flow into the same description parser, so
        checking description alone would leave three ways in. The boundary is
        strictly wider still — it scans EVERY string argument — so these four
        are a lower bound on its coverage rather than a description of it.
        """
        server, interceptor = task_server
        arguments = {
            'project_root': '/project',
            'prompt': 'a clean prompt',
            'description': 'a clean description',
            'agent_id': 'a',
        }
        arguments[field] = f'dirty {_LEAKED_INVOKE}'

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('submit_task', arguments)
        )
        interceptor.submit_task.assert_called_once()

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        interceptor.submit_task.reset_mock()
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('submit_task', arguments)

        _assert_boundary_refused(exc_info, field=field, pattern='</invoke>')
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
        assert result.get('error') != _RETIRED_ERROR, f'got: {result!r}'
        interceptor.submit_task.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True])
    async def test_override_lets_deliberate_markup_through(self, task_server, as_json):
        """Both metadata shapes honour the override.

        submit_task accepts metadata as an object OR a JSON string; an author
        documenting the leak (as DF 3083's own description does) must get through
        whichever shape they used. The boundary reads the flag off the live
        argument before any tool body has coerced it, so it has to understand
        the JSON-string spelling an agent actually sends over the wire too.
        """
        server, interceptor = task_server
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
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
        assert result.get('error') != _RETIRED_ERROR, f'got: {result!r}'
        interceptor.submit_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_flag_is_not_persisted_into_task_metadata(self, task_server):
        """'allow_mcp_markup' is a write-time control, not task metadata.

        It is deliberately absent from the task metadata vocabulary
        (docs/task-authoring.md), so letting it through would mint an
        unrecognised key on every task written with the override. The tool body
        still owns the strip — the guard forwards the flag unchanged to a tool
        that declares ``metadata``.
        """
        server, interceptor = task_server
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )

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


class TestUpdateTaskOnlyTheBoundaryRejects:
    """update_task, the same four fields."""

    @pytest.mark.asyncio
    async def test_only_the_boundary_rejects_the_description(self, task_server):
        server, interceptor = task_server
        arguments = {
            'id': '3141',
            'project_root': '/project',
            'description': f'updated.\n{_LEAKED_CONTENT}',
            'agent_id': 'claude-task-3141',
        }

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('update_task', arguments)
        )
        interceptor.update_task.assert_called_once()

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        interceptor.update_task.reset_mock()
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('update_task', arguments)

        _assert_boundary_refused(exc_info, field='description', pattern='</content>')
        interceptor.update_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    async def test_only_the_boundary_rejects_each_scanned_field(self, task_server, field):
        server, interceptor = task_server
        arguments = {'id': '3141', 'project_root': '/project', 'agent_id': 'a'}
        arguments[field] = f'dirty {_LEAKED_INVOKE}'

        _assert_no_in_line_rejection(
            await server._tool_manager.call_tool('update_task', arguments)
        )
        interceptor.update_task.assert_called_once()

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )
        interceptor.update_task.reset_mock()
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('update_task', arguments)

        _assert_boundary_refused(exc_info, field=field, pattern='</invoke>')
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
        assert result.get('error') != _RETIRED_ERROR, f'got: {result!r}'
        interceptor.update_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self, task_server):
        """The case that makes the hatch load-bearing: updating DF 3083 itself.

        3083's description quotes all three literals, so without this an
        update_task on the very sibling this leaf feeds would be permanently
        blocked.
        """
        server, interceptor = task_server
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )

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
        assert result.get('error') != _RETIRED_ERROR, f'got: {result!r}'
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
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
        )

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


# ---------------------------------------------------------------------------
# task 4458 step-9: the SAME specimens, against the BOUNDARY guard.
#
# THE DEMONSTRATION the retirement of the four in-line gates above is
# predicated on. Both mechanisms are live at this point — the boundary simply
# wins, because it runs at ToolManager.call_tool, strictly before any tool body.
# The suite above is left untouched and green: it builds servers through
# create_mcp_server alone, and the guard is installed by main.py, so those
# servers still exercise the in-line gate.
#
# Everything below reuses the specimen constants and fixtures above verbatim,
# so this is a specimen-for-specimen comparison rather than a fresh, friendlier
# corpus.
# ---------------------------------------------------------------------------

#: The boundary's two contracted rejection vocabularies. Neither is the in-line
#: gate's ``McpEnvelopeMarkupWriteRejected``, which is the point: after
#: retirement the caller sees these, so the parity claim has to be made against
#: them and not against the old spelling.
_DETECTED = 'mcp_markup_detected'
_UNREPAIRABLE = 'mcp_markup_unrepairable'

#: MEASURED outcome per specimen — (specimen, matched_pattern, on a tool whose
#: leaked parameter is named ``content``, on a task tool whose leaked parameter
#: is title/description/details/prompt).
#:
#: Every one of the three is REJECTED on every one of the four tools; that is
#: the parity claim, and it is asserted for all of them. The two vocabularies
#: differ only in whether a REPAIR is offered alongside the rejection, and that
#: difference is a property of shared.toolcall_markup.repair's refusal to guess
#: (its NO SILENT PARTIAL REPAIR contract), not a hole in the containment:
#:
#: * ``</content>`` mis-closes a parameter genuinely named ``content``, so on
#:   add_memory/add_episode it is a candidate mis-close position and the value
#:   is repairable. On submit_task/update_task there IS no ``content``
#:   parameter, so the same literal names nothing the repairer can anchor on.
#: * ``</invoke>`` closes the ENVELOPE, not a parameter. Nothing downstream of
#:   it can be attributed to any argument.
#: * ``<parameter name="priority">high</parameter>`` arrives with no preceding
#:   mis-close, so the opener is embedded mid-text rather than following a
#:   closed parameter — there is no anchor, and inventing one would be exactly
#:   the guess the repairer refuses to make.
#:
#: Refusing without a repair is still a REFUSAL: nothing is written, and the
#: residue is escalated instead of dropped. The repairable case is covered
#: separately by TestBoundaryRecoversAbsorbedParameters below, on corpus-shaped
#: specimens.
_BOUNDARY_SPECIMENS = [
    (_LEAKED_CONTENT, '</content>', _DETECTED, _UNREPAIRABLE),
    (_LEAKED_INVOKE, '</invoke>', _UNREPAIRABLE, _UNREPAIRABLE),
    (_LEAKED_PARAMETER, '<parameter name=', _UNREPAIRABLE, _UNREPAIRABLE),
]


def _install_boundary_guard(server) -> None:
    """Install the guard the way main.py does, minus the safe wrapper.

    ``known_projects=None`` on purpose: a burst then resolves to no
    project_root and the sink files NOTHING, so this suite cannot write an
    escalation into '/project' or anywhere else. Every test below stays under
    the storm threshold anyway (each builds its own server, and so its own
    counter), but the belt is free.
    """
    install_markup_guard(
        server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=None
    )


def _absorbed(clean: str, param: str, value: str) -> str:
    """A corpus-shaped specimen: *clean* mis-closed, then absorbing *param*.

    Assembled from shared.toolcall_markup's own constants so no raw envelope
    sentinel is authored here (same helper, same reason, as
    tests/test_markup_guard_fused_memory.py::_leaked).
    """
    return (
        clean
        + closer_for('parameter')
        + CANONICAL_OPENER_PREFIX
        + f'"{param}">'
        + value
        + closer_for('parameter')
    )


def _assert_boundary_rejection(exc_info, *, field: str, pattern: str, error_type: str) -> dict:
    """Assert the raised ToolError carries the boundary's structured payload.

    The boundary RAISES where the in-line gate RETURNED a dict. That is not a
    stylistic difference: a returned dict is destroyed by the output validation
    of any tool annotated ``-> str``, so raising is the only shape that reaches
    every caller intact.
    """
    payload = json.loads(str(exc_info.value))
    assert payload['error_type'] == error_type, f'got: {payload!r}'
    assert payload['error_type'] != 'McpEnvelopeMarkupWriteRejected'
    assert payload['field'] == field, f'got: {payload!r}'
    # Specimen-for-specimen: the boundary names the SAME literal the in-line
    # gate named for this specimen.
    assert payload['matched_pattern'] == pattern, f'got: {payload!r}'
    return payload


@pytest.fixture
def guarded_memory_server(memory_server):
    """The suite's own ``memory_server`` fixture, with the boundary installed."""
    server, mock_service = memory_server
    _install_boundary_guard(server)
    return server, mock_service


@pytest.fixture
def guarded_task_server(task_server):
    """The suite's own ``task_server`` fixture, with the boundary installed."""
    server, interceptor = task_server
    _install_boundary_guard(server)
    return server, interceptor


class TestBoundaryRejectsTheSameSpecimens:
    """Every specimen the in-line gate rejects, the boundary rejects too.

    Four tools x three specimens, plus the four scanned text fields on each
    task tool. Nothing reaches the service or the interceptor in any of them.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern', 'error_type', '_task_error_type'), _BOUNDARY_SPECIMENS
    )
    async def test_add_memory(
        self, guarded_memory_server, content, pattern, error_type, _task_error_type
    ):
        server, mock_service = guarded_memory_server

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {
                    'content': content,
                    'category': 'observations_and_summaries',
                    'agent_id': 'claude-task-3141',
                    'project_id': _PROJECT_ID,
                },
            )

        _assert_boundary_rejection(
            exc_info, field='content', pattern=pattern, error_type=error_type
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern', 'error_type', '_task_error_type'), _BOUNDARY_SPECIMENS
    )
    async def test_add_episode(
        self, guarded_memory_server, content, pattern, error_type, _task_error_type
    ):
        server, mock_service = guarded_memory_server

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_episode',
                {
                    'content': content,
                    'agent_id': 'claude-task-3141',
                    'project_id': _PROJECT_ID,
                },
            )

        _assert_boundary_rejection(
            exc_info, field='content', pattern=pattern, error_type=error_type
        )
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    @pytest.mark.parametrize(
        ('content', 'pattern', '_memory_error_type', 'error_type'), _BOUNDARY_SPECIMENS
    )
    async def test_submit_task_in_each_scanned_field(
        self, guarded_task_server, field, content, pattern, _memory_error_type, error_type
    ):
        """All four text fields, as the in-line gate scans them.

        The boundary is strictly WIDER — it scans every string argument, not an
        explicit four-field dict handed to it at the call site — so this is a
        lower bound on its coverage, not a description of it.
        """
        server, interceptor = guarded_task_server
        args = {
            'project_root': '/project',
            'prompt': 'a clean prompt',
            'description': 'a clean description',
            'agent_id': 'a',
        }
        args[field] = f'dirty {content}'

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('submit_task', args)

        _assert_boundary_rejection(
            exc_info, field=field, pattern=pattern, error_type=error_type
        )
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    @pytest.mark.parametrize(
        ('content', 'pattern', '_memory_error_type', 'error_type'), _BOUNDARY_SPECIMENS
    )
    async def test_update_task_in_each_scanned_field(
        self, guarded_task_server, field, content, pattern, _memory_error_type, error_type
    ):
        server, interceptor = guarded_task_server
        args = {'id': '3141', 'project_root': '/project', 'agent_id': 'a'}
        args[field] = f'dirty {content}'

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('update_task', args)

        _assert_boundary_rejection(
            exc_info, field=field, pattern=pattern, error_type=error_type
        )
        interceptor.update_task.assert_not_called()


class TestBoundaryRecoversAbsorbedParameters:
    """The diagnostic the single-literal in-line gate could never produce.

    The gate above matched a literal and echoed a 200-char excerpt back. It had
    no way to say WHICH sibling argument the leak swallowed, because it never
    parsed the residue — so a caller whose priority or agent_id had been eaten
    learned only that its text was dirty. ``repaired_call`` is the COMPLETE
    argument map, resubmittable verbatim, with the absorbed parameter restored.

    The specimens here are corpus-shaped (PRD section 2.1): a mis-close FOLLOWED
    by the next parameter's opener. The three constants above are held verbatim
    for the parity comparison and are deliberately not reshaped to fit.
    """

    @pytest.mark.asyncio
    async def test_add_memory_recovers_the_absorbed_agent_id(self, guarded_memory_server):
        server, mock_service = guarded_memory_server

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {
                    'content': _absorbed(_CLEAN_CONTENT, 'agent_id', 'claude-task-4458'),
                    'category': 'observations_and_summaries',
                    'project_id': _PROJECT_ID,
                },
            )

        payload = _assert_boundary_rejection(
            exc_info, field='content', pattern='</parameter>', error_type=_DETECTED
        )
        assert payload['recovered_params'] == ['agent_id']
        assert payload['repaired_call'] == {
            'content': _CLEAN_CONTENT,
            'category': 'observations_and_summaries',
            'project_id': _PROJECT_ID,
            'agent_id': 'claude-task-4458',
        }
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_episode_recovers_the_absorbed_agent_id(self, guarded_memory_server):
        server, mock_service = guarded_memory_server

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_episode',
                {
                    'content': _absorbed(_CLEAN_CONTENT, 'agent_id', 'claude-task-4458'),
                    'project_id': _PROJECT_ID,
                },
            )

        payload = _assert_boundary_rejection(
            exc_info, field='content', pattern='</parameter>', error_type=_DETECTED
        )
        assert payload['repaired_call']['agent_id'] == 'claude-task-4458'
        assert payload['repaired_call']['content'] == _CLEAN_CONTENT
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_task_recovers_the_absorbed_priority(self, guarded_task_server):
        """DF 3083 vector 2, now with the swallowed value HANDED BACK.

        Reify task 3210 was filed priority=high and stored as medium because the
        description parser derived a value from the fragment without complaint.
        The in-line gate stopped the silent mis-parse; the boundary additionally
        tells the caller what the fragment ate.
        """
        server, interceptor = guarded_task_server

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'submit_task',
                {
                    'project_root': '/project',
                    'prompt': 'Reconcile task 7',
                    'description': _absorbed('Reconcile task 7.', 'priority', 'high'),
                    'agent_id': 'a',
                },
            )

        payload = _assert_boundary_rejection(
            exc_info, field='description', pattern='</parameter>', error_type=_DETECTED
        )
        assert payload['recovered_params'] == ['priority']
        assert payload['repaired_call'] == {
            'project_root': '/project',
            'prompt': 'Reconcile task 7',
            'description': 'Reconcile task 7.',
            'agent_id': 'a',
            'priority': 'high',
        }
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_task_recovers_the_absorbed_priority(self, guarded_task_server):
        server, interceptor = guarded_task_server

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'update_task',
                {
                    'id': '3141',
                    'project_root': '/project',
                    'description': _absorbed('updated.', 'priority', 'high'),
                    'agent_id': 'a',
                },
            )

        payload = _assert_boundary_rejection(
            exc_info, field='description', pattern='</parameter>', error_type=_DETECTED
        )
        assert payload['recovered_params'] == ['priority']
        assert payload['repaired_call']['priority'] == 'high'
        assert payload['repaired_call']['description'] == 'updated.'
        interceptor.update_task.assert_not_called()


class TestBoundaryPortsTheNonRejectionBehaviours:
    """The two ways a call must NOT be rejected, carried onto the boundary.

    An override that stops working is a containment failure of its own: DF
    3083's own description quotes all three literals in prose, so a task about
    the leak could not be filed without it.
    """

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_reach_the_service(
        self, guarded_memory_server
    ):
        server, mock_service = guarded_memory_server

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': True},
            },
        )

        assert result == {'id': 'ok'}, f'expected pass-through, got: {result!r}'
        mock_service.add_memory.assert_called_once()
        assert mock_service.add_memory.call_args.kwargs['content'] == _LEAKED_INVOKE

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True])
    async def test_override_on_submit_task_as_dict_and_as_json_string(
        self, guarded_task_server, as_json
    ):
        """submit_task's metadata is string-or-dict, and both spellings work.

        The boundary reads the override off the live argument before any tool
        body has coerced it, so the JSON-string spelling — the one an agent
        actually sends over the wire — has to be understood there too.
        """
        server, interceptor = guarded_task_server
        metadata = {'allow_mcp_markup': True}

        result = _parse(await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'a clean prompt',
                'description': f'dirty {_LEAKED_INVOKE}',
                'agent_id': 'a',
                'metadata': json.dumps(metadata) if as_json else metadata,
            },
        ))

        assert result.get('error_type') != _DETECTED, f'got: {result!r}'
        interceptor.submit_task.assert_called_once()
        assert (
            interceptor.submit_task.call_args.kwargs['description']
            == f'dirty {_LEAKED_INVOKE}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('value', ['yes', 1, 'True'])
    async def test_a_non_literal_true_override_does_not_unlock_the_boundary(
        self, guarded_memory_server, value
    ):
        """The truthiness case the in-line gate guards, ported.

        A truthy-but-not-True value is far more likely to be a caller mistake
        than a deliberate quotation, and a guard that accepted 'yes' would be
        unlocked by any dictionary that happened to carry that key.
        """
        server, mock_service = guarded_memory_server

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {
                    'content': _LEAKED_INVOKE,
                    'category': 'observations_and_summaries',
                    'agent_id': 'a',
                    'project_id': _PROJECT_ID,
                    'metadata': {'allow_mcp_markup': value},
                },
            )

        _assert_boundary_rejection(
            exc_info, field='content', pattern='</invoke>', error_type=_UNREPAIRABLE
        )
        mock_service.add_memory.assert_not_called()


# ---------------------------------------------------------------------------
# task 4458 step-11: the storm escape (INV-4), re-pointed onto the boundary.
#
# The retired gate carried its own MarkupStormCounter in tools.py. The boundary
# carries one too, inside the shared middleware, and this class re-points the
# original class's assertions onto it. Three deltas, all measured, none a loss
# of containment — they are stated here rather than silently absorbed:
#
# 1. KEY. The retired counter keyed by project_root alone (the tool bodies had
#    already translated project_id through known_projects). The middleware keys
#    by (project, outcome): 'outcome' because a burst of silent REPAIRS is as
#    urgent as a burst of refusals and pooling them would name an outcome that
#    never burst, and 'project' UNTRANSLATED because translation needs a
#    registry the shared layer does not have. Consequence, asserted below: a
#    memory-tool burst counts under the bare project_id while a task-tool burst
#    counts under the project_root, so they no longer pool into one bucket.
#    Filed as follow-up rather than fixed here — the keying lives in beta's
#    shared module, which two sibling leaves also edit.
# 2. ECHO. The retired block echoed the filed escalation_id back to the caller.
#    The middleware hands the burst to the sink without waiting for an id, so
#    the response reports the burst but not where it was filed. The operator
#    -facing ERROR line and the filed record carry that.
# 3. ANCHOR. The filing anchor is 'markup-guard', not the 'markup-tripwire'
#    anchor the L1 escalation watcher shares and squats. Pinned in
#    tests/test_markup_guard_fused_memory.py::TestStormEscape.
# ---------------------------------------------------------------------------


async def _refuse_add_memory(server, project_id: str = _PROJECT_ID) -> dict:
    """Drive one add_memory refusal and return the parsed boundary payload."""
    with pytest.raises(ToolError) as exc_info:
        await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': project_id,
            },
        )
    return _boundary_payload(exc_info)


async def _refuse_submit_task(server, project_root: str = '/project') -> dict:
    """Drive one submit_task refusal and return the parsed boundary payload."""
    with pytest.raises(ToolError) as exc_info:
        await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': project_root,
                'prompt': 'a clean prompt',
                'description': f'dirty {_LEAKED_INVOKE}',
                'agent_id': 'a',
            },
        )
    return _boundary_payload(exc_info)


class TestMarkupStormAtTheBoundary:
    """A BURST of refusals escalates instead of being absorbed silently (INV-4).

    One bounced write is routine — an agent retries and moves on. What must not
    be swallowed is a *stream* of them, because that means the upstream harness
    serialization leak is running right now and every write it touches is a
    would-be corpus specimen.
    """

    @pytest.fixture
    def guarded_mixed_server(self, mixed_server):
        server, mock_service, interceptor = mixed_server
        _install_boundary_guard(server)
        return server, mock_service, interceptor

    @pytest.mark.asyncio
    async def test_a_burst_reports_the_storm_to_the_crossing_caller(
        self, guarded_mixed_server
    ):
        server, _mock_service, _interceptor = guarded_mixed_server
        assert _MARKUP_STORM_THRESHOLD == 3, (
            'this test drives exactly 3 refusals; update it alongside the constant'
        )

        first = await _refuse_add_memory(server)
        second = await _refuse_add_memory(server)
        third = await _refuse_add_memory(server)

        assert 'storm' not in first, f'below threshold must carry no storm key: {first!r}'
        assert 'storm' not in second, f'below threshold must carry no storm key: {second!r}'

        storm = third.get('storm')
        assert storm is not None, f'threshold-crossing refusal must report a storm: {third!r}'
        assert storm['count'] == _MARKUP_STORM_THRESHOLD, f'got: {storm!r}'
        assert storm['threshold'] == _MARKUP_STORM_THRESHOLD, f'got: {storm!r}'
        assert storm['window_seconds'] == _MARKUP_STORM_WINDOW_SECONDS, f'got: {storm!r}'
        assert storm['project'] == _PROJECT_ID, f'got: {storm!r}'

    @pytest.mark.asyncio
    async def test_the_counter_is_keyed_by_project_and_outcome(
        self, guarded_mixed_server
    ):
        """Delta 1, asserted rather than assumed.

        Two memory refusals plus one task refusal used to pool into one bucket
        and fire. They no longer do: the memory tools carry a project_id and the
        task tools a project_root, so the middleware — which does not translate
        — sees two projects. Containment is unaffected (all three writes are
        still refused); what changes is how quickly a MIXED-tool leak trips the
        alarm. A real burst reaches the threshold within a bucket regardless:
        the measured incident was 41 refusals.
        """
        server, _mock_service, _interceptor = guarded_mixed_server

        await _refuse_add_memory(server)
        await _refuse_add_memory(server)
        crossing = await _refuse_submit_task(server)

        assert 'storm' not in crossing, (
            f'memory-tool and task-tool refusals are separate buckets now: {crossing!r}'
        )

    @pytest.mark.asyncio
    async def test_below_threshold_refusals_are_still_full_refusals(
        self, guarded_mixed_server
    ):
        """The write is bounced either way — the storm is additive, not a gate.

        A first-offence refusal that carried less would make the caller's
        remediation depend on how many *other* agents happened to leak recently.
        """
        server, mock_service, _interceptor = guarded_mixed_server

        payload = await _refuse_add_memory(server)

        assert payload['error_type'] == 'mcp_markup_unrepairable', f'got: {payload!r}'
        assert payload['field'] == 'content', f'got: {payload!r}'
        assert 'storm' not in payload, f'got: {payload!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_each_server_starts_with_a_fresh_counter(self, mixed_server, monkeypatch):
        """Per-server state, so no counter bleeds between servers (or tests).

        Asserting the SECOND server also fires on its own third refusal is the
        discriminator: with a module-global counter the first server's fire would
        still be inside the rate-limit window, so the second server's burst would
        be suppressed entirely.
        """
        monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))

        def _build():
            interceptor = AsyncMock()
            interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
            server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
            _install_boundary_guard(server)
            return server

        first_server = _build()
        for _ in range(_MARKUP_STORM_THRESHOLD - 1):
            assert 'storm' not in await _refuse_submit_task(first_server)
        assert 'storm' in await _refuse_submit_task(first_server), 'server A should have fired'

        second_server = _build()
        for i in range(_MARKUP_STORM_THRESHOLD - 1):
            below = await _refuse_submit_task(second_server)
            assert 'storm' not in below, f'server B refusal {i + 1} fired early: {below!r}'
        crossing = await _refuse_submit_task(second_server)
        assert 'storm' in crossing, (
            f'server B must fire on its OWN third refusal, not inherit A: {crossing!r}'
        )

    @pytest.mark.asyncio
    async def test_escalation_is_filed_once_against_the_resolved_project_root(
        self, mixed_server, monkeypatch, tmp_path
    ):
        """The queue write happens on the burst only, against a real root.

        The memory tools carry only a project_id, so the sink resolves it
        through known_projects before filing — otherwise the escalation would
        open a queue at a relative directory named after the project.
        """
        server, _mock_service, _interceptor = mixed_server
        install_markup_guard(
            server,
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects={_PROJECT_ID: str(tmp_path)},
        )
        calls: list[tuple] = []

        def _fake_emit(project_root, storm, *, anchor_task_id):
            calls.append((project_root, storm, anchor_task_id))
            return 'esc-markup-guard-1'

        monkeypatch.setattr(
            'fused_memory.server.markup_guard.emit_markup_storm_escalation', _fake_emit
        )

        for _ in range(_MARKUP_STORM_THRESHOLD - 1):
            await _refuse_add_memory(server)
        assert calls == [], f'no escalation may be filed below threshold: {calls!r}'

        crossing = await _refuse_add_memory(server)

        assert len(calls) == 1, f'expected exactly one escalation attempt, got: {calls!r}'
        project_root, storm_arg, anchor = calls[0]
        assert project_root == str(tmp_path), (
            f'the bare project_id must be resolved before filing, got: {project_root!r}'
        )
        assert storm_arg['count'] == _MARKUP_STORM_THRESHOLD, f'got: {storm_arg!r}'
        assert anchor == 'markup-guard', (
            f'must not file under the anchor the L1 watcher squats, got: {anchor!r}'
        )
        assert crossing.get('storm', {}).get('count') == _MARKUP_STORM_THRESHOLD

    @pytest.mark.asyncio
    async def test_escalation_failure_does_not_break_the_refusal(
        self, guarded_mixed_server, monkeypatch
    ):
        """Escalation is purely additive: the write was already refused.

        emit_markup_storm_escalation is documented never to raise, but the call
        site must not depend on that promise — a queue failure that turned a
        refusal into a 500 would trade containment for an outage.
        """
        server, mock_service, _interceptor = guarded_mixed_server

        def _boom(*args, **kwargs):
            raise RuntimeError('escalation queue is on fire')

        monkeypatch.setattr(
            'fused_memory.server.markup_guard.emit_markup_storm_escalation', _boom
        )

        for _ in range(_MARKUP_STORM_THRESHOLD - 1):
            await _refuse_add_memory(server)
        crossing = await _refuse_add_memory(server)

        assert crossing['error_type'] == 'mcp_markup_unrepairable', f'got: {crossing!r}'
        assert crossing['field'] == 'content', f'got: {crossing!r}'
        assert 'storm' in crossing, (
            f'the burst is still reported even when escalation fails: {crossing!r}'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_storm_is_logged_at_error_with_a_greppable_prefix(
        self, guarded_mixed_server, caplog
    ):
        """The burst must be visible to an operator who never reads MCP responses.

        The MCP response goes to the leaking agent, which is precisely the party
        least likely to act on it, so the ERROR log line is the operator-facing
        half of INV-4.
        """
        server, _mock_service, _interceptor = guarded_mixed_server

        with caplog.at_level('ERROR'):
            for _ in range(_MARKUP_STORM_THRESHOLD - 1):
                await _refuse_add_memory(server)
            below = [
                r for r in caplog.records
                if r.levelname == 'ERROR' and 'markup_guard_storm' in r.getMessage()
            ]
            assert below == [], f'no storm ERROR below threshold, got: {below!r}'

            await _refuse_add_memory(server)

        storm_errors = [
            r for r in caplog.records
            if r.levelname == 'ERROR' and 'markup_guard_storm' in r.getMessage()
        ]
        assert storm_errors, (
            f'expected a greppable markup_guard_storm ERROR line, got: {caplog.text!r}'
        )
