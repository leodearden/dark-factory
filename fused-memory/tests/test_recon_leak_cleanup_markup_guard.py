"""Recon's leaked-fragment cleanup must survive the boundary markup guard.

Task 9317, esc-markup-residue-9: the memory_consolidator stage amended a Mem0
record whose text carried a leaked tool-call tail, with clean ``content`` but a
``reason`` that quoted the removed fragment verbatim. The guard scans every
string argument, so the call was refused as unrepairable and nothing was
written. The stage is taught a structural reason instead, and these tests drive
that taught text through the real guarded server.

AUTHORING RULE: this file never spells a raw envelope literal. Every specimen is
built from ``shared.toolcall_markup`` constants, as
tests/test_markup_guard_fused_memory.py::_leaked does.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fastmcp_tool_result import tool_result_dict
from _recon_prompt_write_scope import recon_system_prompts, split_by_write_grant
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import RepairPolicy
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for, detect

from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.reconciliation import prompts
from fused_memory.server.markup_guard import install_markup_guard
from fused_memory.server.tools import create_mcp_server
from fused_memory.utils.toolcall_xml_leak import find_toolcall_xml_leak, has_toolcall_xml_leak

_PROJECT_ID = 'dark_factory'
_MEMORY_ID = 'mem-leaked-1'
_RECON_AGENT_ID = 'recon-stage-memory_consolidator'

_CLAIM = 'Run the fused-memory reconciliation suite with -n 8 after moving a stage callee.'
_FRAGMENT = (
    closer_for('content') + '\n' + CANONICAL_OPENER_PREFIX + '"category">procedural_knowledge'
)
_LEAKED_RECORD = _CLAIM + _FRAGMENT
_CLEAN = _LEAKED_RECORD[: find_toolcall_xml_leak(_LEAKED_RECORD)[0].index].rstrip()

_QUOTING_REASON = (
    "Remove a leaked serialized tool-call fragment ('"
    + closer_for('content')
    + "' + '"
    + CANONICAL_OPENER_PREFIX
    + '"category">procedural_knowledge'
    + "') appended to the stored text by the original writer; the substantive claim is unchanged."
)

_PROMPTS = recon_system_prompts()
_WRITING, _NOT_WRITING = split_by_write_grant(_PROMPTS)


def _guarded_server(tmp_path) -> tuple[Any, AsyncMock]:
    """The real server and production guard tier, over a fake service.

    The ``mem0_update`` config is real: a bare Mock leaf makes the fail-closed
    authz resolver deny, and the accepted-call test would then pass for the
    wrong reason.
    """
    service = AsyncMock()
    service.config.mem0_update = Mem0UpdateConfig()
    service.update_memory = AsyncMock(
        return_value={'status': 'updated', 'store': 'mem0', 'id': _MEMORY_ID}
    )
    server = create_mcp_server(service)
    install_markup_guard(
        server,
        policy=RepairPolicy.REJECT_WITH_REPAIR,
        known_projects={_PROJECT_ID: str(tmp_path)},
    )
    return server, service


async def _amend_leaked_record(server: Any, reason: str) -> Any:
    return await server._tool_manager.call_tool(
        'update_memory',
        {
            'memory_id': _MEMORY_ID,
            'store': 'mem0',
            'project_id': _PROJECT_ID,
            'content': _CLEAN,
            'reason': reason,
            'agent_id': _RECON_AGENT_ID,
        },
    )


def test_the_specimen_is_a_leak_by_the_corpus_detector():
    assert has_toolcall_xml_leak(_LEAKED_RECORD)
    assert not has_toolcall_xml_leak(_CLEAN)
    assert _CLEAN == _CLAIM


@pytest.mark.asyncio
async def test_a_reason_quoting_the_fragment_is_refused_and_writes_nothing(tmp_path):
    server, service = _guarded_server(tmp_path)

    with pytest.raises(ToolError) as exc_info:
        await _amend_leaked_record(server, _QUOTING_REASON)

    payload = json.loads(str(exc_info.value))
    assert payload['error_type'] == 'mcp_markup_unrepairable'
    assert payload['field'] == 'reason'
    service.update_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_taught_reason_is_accepted_and_forwarded_with_the_clean_content(tmp_path):
    server, service = _guarded_server(tmp_path)

    result = tool_result_dict(
        await _amend_leaked_record(server, prompts.LEAKED_FRAGMENT_CLEANUP_REASON)
    )

    assert result['status'] == 'updated'
    service.update_memory.assert_awaited_once()
    forwarded = service.update_memory.await_args.kwargs
    assert forwarded['reason'] == prompts.LEAKED_FRAGMENT_CLEANUP_REASON
    assert forwarded['content'] == _CLEAN


def test_the_write_scope_split_is_not_vacuous():
    assert _WRITING and _NOT_WRITING


@pytest.mark.parametrize('name', _WRITING)
def test_every_memory_writing_prompt_teaches_the_reason_once(name):
    assert _PROMPTS[name].count(prompts.LEAKED_FRAGMENT_CLEANUP_REASON) == 1


@pytest.mark.parametrize('name', _NOT_WRITING)
def test_no_read_only_prompt_teaches_the_reason(name):
    assert prompts.LEAKED_FRAGMENT_CLEANUP_REASON not in _PROMPTS[name]


@pytest.mark.parametrize('name', sorted(_PROMPTS))
def test_no_recon_prompt_spells_envelope_markup_or_offers_the_override(name):
    prompt = _PROMPTS[name]
    assert detect(prompt) is None, (
        f'{name} spells an envelope literal; an agent copying it hands the guard '
        'the very text it refuses. Describe the markup in words.'
    )
    assert 'allow_mcp_markup' not in prompt
