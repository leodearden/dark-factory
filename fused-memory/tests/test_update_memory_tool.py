"""Argument (arm) validation for the ``update_memory`` MCP tool — task 3088.

Step 25: RED tests, failing before step 26 adds §5(a) guard steps 5-7.

The authorization gate has its own file (``tests/server/test_update_memory_authz_gate.py``);
everything here runs as an ALREADY-AUTHORIZED caller, so a failure can only be
about the arguments. Harness copied from ``test_update_edge_tool.py``:
``create_mcp_server(mock_service)`` + ``_tool_manager.call_tool`` + a local
``_parse_tool_result``.

The posture under test is ``update_edge``'s: a contradictory or under-specified
argument set is rejected LOUD, with a message naming the offending argument —
never silently dropped, coerced, or half-applied. A caller must never be able to
believe it wrote something it did not.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from fused_memory.backends.mem0_client import MEM0_MANAGED_METADATA_KEYS
from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.models.enums import MemoryCategory

_PROJECT_ID = 'dark_factory'
_MEMORY_ID = '77a3f6bc-0000-0000-0000-000000000000'

# On the default allowlist for BOTH arms, so no case here can fail for an
# authorization reason and be mistaken for a validation result.
_AGENT = 'recon-stage-memory_consolidator'


def _parse_tool_result(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


def _mock_service():
    """A mock MemoryService whose mem0_update leaves are REAL config values.

    A bare AsyncMock makes every leaf a Mock, which the fail-closed authz
    resolver rejects — every case would then pass for the wrong reason
    (Mem0UpdateNotAuthorized rather than ValidationError).
    """
    mock_service = AsyncMock()
    mock_service.config.mem0_update = Mem0UpdateConfig()
    mock_service.update_memory = AsyncMock(
        return_value={
            'status': 'updated',
            'store': 'mem0',
            'id': _MEMORY_ID,
            'content_amended': False,
            'metadata_patched': True,
        }
    )
    return mock_service


async def _call_tool(mock_service, **args):
    from fused_memory.server.tools import create_mcp_server

    server = create_mcp_server(mock_service)
    result = await server._tool_manager.call_tool('update_memory', {
        'memory_id': _MEMORY_ID,
        'store': 'mem0',
        'project_id': _PROJECT_ID,
        'agent_id': _AGENT,
        **args,
    })
    return _parse_tool_result(result)


def _assert_rejected(result, mock_service, *named):
    """Every rejection is a structured ValidationError that dispatched nothing."""
    assert result.get('error_type') == 'ValidationError', result
    message = result.get('error')
    assert isinstance(message, str) and message, result
    for token in named:
        assert token in message, (
            f'the rejection must name {token!r} so the caller can fix it; '
            f'got {message!r}'
        )
    mock_service.update_memory.assert_not_called()


class TestNoArmSupplied:
    """(a) A call that asks for nothing is a caller bug, not a no-op success."""

    @pytest.mark.asyncio
    async def test_no_arm_supplied_is_rejected(self):
        mock_service = _mock_service()
        result = await _call_tool(mock_service)
        _assert_rejected(
            result, mock_service, 'content', 'metadata_patch', 'metadata_delete_keys',
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('args', [
        {'metadata_patch': {}},
        {'metadata_delete_keys': []},
        {'metadata_patch': {}, 'metadata_delete_keys': []},
    ], ids=['empty_patch', 'empty_delete_keys', 'both_empty'])
    async def test_empty_arms_count_as_no_arm(self, args):
        """An empty dict/list is an absent arm, not a present one — otherwise a
        caller whose patch computed to {} gets a success envelope for a write
        that never happened."""
        mock_service = _mock_service()
        result = await _call_tool(mock_service, **args)
        _assert_rejected(result, mock_service)


class TestContentArm:
    """(b)/(c) The content arm's own bars."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('content', ['', '   ', '\n\t '], ids=['empty', 'spaces', 'ws'])
    async def test_empty_or_whitespace_content_is_rejected(self, content):
        """Amending a record to whitespace destroys it as surely as deleting it,
        and does so while reporting success."""
        mock_service = _mock_service()
        result = await _call_tool(mock_service, content=content, reason='consolidation')
        _assert_rejected(result, mock_service, 'content')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('reason', [None, '', '   '], ids=['missing', 'empty', 'spaces'])
    async def test_content_without_reason_is_rejected(self, reason):
        """A silent rewrite is invisible to every downstream reader; the reason
        is the only durable record of WHY the text changed."""
        mock_service = _mock_service()
        kwargs = {} if reason is None else {'reason': reason}
        result = await _call_tool(mock_service, content='rewritten text', **kwargs)
        _assert_rejected(result, mock_service, 'reason', 'content')

    @pytest.mark.asyncio
    async def test_metadata_only_call_needs_no_reason(self):
        """The concrete differential bar: a tag is cheap to notice and cheap to
        correct, so it does not carry the content arm's justification cost."""
        mock_service = _mock_service()
        result = await _call_tool(mock_service, metadata_patch={'topic': 'docs-prd-landing'})
        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()
        assert mock_service.update_memory.call_args[1].get('reason') is None

    @pytest.mark.asyncio
    async def test_content_with_reason_is_accepted(self):
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, content='rewritten text', reason='consolidation',
        )
        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()


class TestReservedKeys:
    """(d) mem0-owned keys are rejected at the boundary, never silently dropped."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('key', sorted(MEM0_MANAGED_METADATA_KEYS))
    async def test_reserved_key_in_patch_is_rejected_naming_it(self, key):
        mock_service = _mock_service()
        result = await _call_tool(mock_service, metadata_patch={key: 'x', 'topic': 'ok'})
        _assert_rejected(result, mock_service, key)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('key', sorted(MEM0_MANAGED_METADATA_KEYS))
    async def test_reserved_key_in_delete_keys_is_rejected_naming_it(self, key):
        """Deleting a mem0-owned key is the more destructive direction: mem0's
        own get/search read those keys, so a successful deletion would make the
        point unreadable by the store that owns it."""
        mock_service = _mock_service()
        result = await _call_tool(mock_service, metadata_delete_keys=[key])
        _assert_rejected(result, mock_service, key)

    @pytest.mark.asyncio
    async def test_rejection_is_not_a_silent_drop(self):
        """The whole point of failing loud here: a caller must never come away
        believing it wrote `created_at`."""
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, metadata_patch={'created_at': '1999-01-01', 'topic': 'ok'},
        )
        _assert_rejected(result, mock_service, 'created_at')
        assert result.get('status') != 'updated', result


class TestCategoryIsProtectedAtTheBoundary:
    """`category` cannot be deleted, and cannot be patched to a bogus value.

    The service layer already stops `category` being lost SILENTLY: a
    `metadata_mode='replace'` that never named the key carries it through
    (`_apply_metadata_delta`). This closes the EXPLICIT half of the same hole —
    a caller-supplied delta that names the key destructively — which that
    carry-through deliberately leaves reachable.

    Both rules exist for the same reason the carry-through does:
    ``Mem0Backend.search`` pushes `category` down to Qdrant as a payload
    filter, so a record with the key missing — or holding a string no
    ``MemoryCategory`` filter can ever match — is permanently unreachable by
    every category-scoped search. A value that is wrong is exactly as invisible
    as a value that is absent, so validating one without the other would leave
    the hole open.

    Same posture as the mem0-owned reserved-key rejections, but a DIFFERENT
    rule, and the difference is the point: mem0-owned keys are refused from
    BOTH arms because mem0 recomputes them, whereas `category` is refused only
    from the delete arm and stays freely patchable. There is no coherent caller
    intent behind removing it — a caller who wants a different category patches
    it — so the delete arm is where the line goes.
    """

    @pytest.mark.asyncio
    async def test_deleting_category_is_rejected_naming_it(self):
        mock_service = _mock_service()
        result = await _call_tool(mock_service, metadata_delete_keys=['category'])
        _assert_rejected(result, mock_service, 'category')

    @pytest.mark.asyncio
    async def test_deleting_category_alongside_an_ordinary_key_is_still_rejected(self):
        """The rule is per-key, not "the list is entirely ordinary keys"."""
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, metadata_delete_keys=['topic', 'category'],
        )
        _assert_rejected(result, mock_service, 'category')

    @pytest.mark.asyncio
    async def test_bogus_category_value_is_rejected_naming_value_and_valid_ones(self):
        """A category no filter can match leaves the record as unreachable as a
        missing one, so the boundary that already validates reserved KEYS must
        not wave through an unmatchable VALUE."""
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, metadata_patch={'category': 'not_a_real_category'},
        )
        _assert_rejected(
            result, mock_service, 'not_a_real_category', 'observations_and_summaries',
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bad', [
        '', 'OBSERVATIONS_AND_SUMMARIES', 'observations and summaries', 12,
    ], ids=['empty', 'wrong_case', 'spaces', 'non_string'])
    async def test_every_unmatchable_category_value_is_rejected(self, bad):
        """Including the near-misses: Qdrant payload equality is exact, so a
        case- or separator-variant is just as invisible as a typo."""
        mock_service = _mock_service()
        result = await _call_tool(mock_service, metadata_patch={'category': bad})
        assert result.get('error_type') == 'ValidationError', result
        mock_service.update_memory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('value', [c.value for c in MemoryCategory])
    async def test_every_valid_category_is_accepted(self, value):
        """Protected, not frozen — a deliberate re-categorization must reach the
        service, validated against the same enum add_memory resolves through."""
        mock_service = _mock_service()
        result = await _call_tool(mock_service, metadata_patch={'category': value})
        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_deleting_an_ordinary_custom_key_still_succeeds(self):
        """The delete arm is narrowed by exactly one key, not closed."""
        mock_service = _mock_service()
        result = await _call_tool(mock_service, metadata_delete_keys=['topic'])
        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_replace_that_omits_category_is_not_an_error(self):
        """The explicit record of which of the review's two fixes shipped.

        Rejecting any replace whose patch omits `category` would force every
        replace caller to restate the key and turn a routine tag edit into an
        error. What protects the key here is the service-layer carry-through,
        not a boundary rejection — so this call must sail straight through.
        """
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, metadata_patch={'topic': 'cgl-eta'}, metadata_mode='replace',
        )
        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()
        kwargs = mock_service.update_memory.call_args.kwargs
        assert 'category' not in (kwargs.get('metadata_patch') or {}), (
            'the boundary must not silently inject a category the caller never '
            'named — carrying it through is the service layer\'s job, where the '
            'record\'s ACTUAL existing value is known'
        )


class TestContradictoryArms:
    """(e)/(f)/(g) Argument combinations that state two incompatible intents."""

    @pytest.mark.asyncio
    async def test_key_in_both_patch_and_delete_keys_is_rejected(self):
        """Write it and remove it cannot both be honoured; picking one silently
        would make the outcome depend on implementation order."""
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service,
            metadata_patch={'topic': 'x', 'kind': 'note'},
            metadata_delete_keys=['topic'],
        )
        _assert_rejected(result, mock_service, 'topic')

    @pytest.mark.asyncio
    async def test_replace_mode_with_delete_keys_is_rejected(self):
        """Replace already decides the whole custom subset, so a delete list is
        either redundant or contradictory — never meaningful."""
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service,
            metadata_patch={'topic': 'x'},
            metadata_delete_keys=['kind'],
            metadata_mode='replace',
        )
        _assert_rejected(result, mock_service, 'metadata_mode', 'metadata_delete_keys')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('args', [
        {'metadata_mode': 'replace'},
        {'metadata_mode': 'replace', 'metadata_patch': {}},
        {'metadata_mode': 'replace', 'content': 'new text', 'reason': 'consolidation'},
    ], ids=['bare', 'empty_patch', 'with_content'])
    async def test_replace_mode_without_patch_is_rejected(self, args):
        """An empty REPLACE means "delete every custom key" — never what a caller
        meant, and unrecoverable once it lands.

        The third case is the reachable one: `content` satisfies the at-least-one-arm
        bar, so `metadata_mode='replace'` slips through every other gate and the
        record's provenance is wiped as a side effect of an ordinary amend.
        """
        mock_service = _mock_service()
        result = await _call_tool(mock_service, **args)
        _assert_rejected(result, mock_service, 'metadata_mode', 'metadata_patch')

    @pytest.mark.asyncio
    async def test_rejected_replace_leaves_the_record_untouched(self):
        """The task-2180 regression, asserted where a caller would observe it.

        `update_memory.assert_not_called()` only proves the TOOL did not dispatch.
        It says nothing about a design that dispatches first and validates inside
        the service — by which point the custom subset is already gone. This fake
        applies the real replace semantics, so the record itself is the witness.
        """
        record = {'kind': 'observation', 'src_project': 'dark_factory', 'topic': 'reify'}

        async def _apply(**kwargs):
            if kwargs.get('metadata_mode') == 'replace':
                record.clear()
                record.update(kwargs.get('metadata_patch') or {})
            return {'status': 'updated', 'store': 'mem0', 'id': _MEMORY_ID}

        mock_service = _mock_service()
        mock_service.update_memory = AsyncMock(side_effect=_apply)

        result = await _call_tool(
            mock_service, content='new text', metadata_mode='replace', reason='consolidation',
        )

        assert result.get('error_type') == 'ValidationError', result
        assert record == {
            'kind': 'observation', 'src_project': 'dark_factory', 'topic': 'reify',
        }, 'a rejected call must not have wiped the record before rejecting it'

    @pytest.mark.asyncio
    async def test_unknown_metadata_mode_is_rejected(self):
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, metadata_patch={'topic': 'x'}, metadata_mode='clobber',
        )
        _assert_rejected(result, mock_service, 'metadata_mode')


class TestMergeIsForgiving:
    """(h) Only `replace` turns emptiness into destruction."""

    @pytest.mark.asyncio
    async def test_merge_mode_with_no_patch_is_not_an_error(self):
        """An empty merge is a well-defined no-op on the metadata, so it must not
        block the arm the caller actually asked for."""
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, content='rewritten', reason='consolidation', metadata_mode='merge',
        )
        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_mode_with_delete_keys_only_is_not_an_error(self):
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, metadata_delete_keys=['stale_tag'], metadata_mode='merge',
        )
        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()


class TestEnvelopeMetadataStaysDistinct:
    """(i) `metadata` (causation envelope) and `metadata_patch` (record payload)
    are two different things that must never leak into each other."""

    @pytest.mark.asyncio
    async def test_envelope_causation_is_consumed_not_written(self):
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service,
            metadata_patch={'topic': 'docs-prd-landing'},
            metadata={'_causation_id': 'evt-1'},
        )
        assert result.get('error_type') is None, result
        kwargs = mock_service.update_memory.call_args[1]
        assert kwargs.get('causation_id') == 'evt-1'
        assert kwargs.get('metadata_patch') == {'topic': 'docs-prd-landing'}, (
            'the envelope must never be merged into the record payload'
        )

    @pytest.mark.asyncio
    async def test_metadata_patch_is_never_read_for_causation(self):
        mock_service = _mock_service()
        result = await _call_tool(
            mock_service, metadata_patch={'topic': 'x', '_causation_id': 'planted'},
        )
        assert result.get('error_type') is None, result
        kwargs = mock_service.update_memory.call_args[1]
        assert kwargs.get('causation_id') != 'planted', (
            'metadata_patch is record payload; reading causation out of it would '
            'let a record key silently steer the event graph'
        )
        assert kwargs.get('metadata_patch') == {'topic': 'x', '_causation_id': 'planted'}
