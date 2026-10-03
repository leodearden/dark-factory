"""Drift pin: `_MEMORY_INSTRUCTIONS` must name the search tool's caller_* params.

The prose in `orchestrator/agents/roles.py::_MEMORY_INSTRUCTIONS` is
hand-transcribed and must agree with the real `search` signature; nothing
mechanically enforces that, and the survey's "prompt text drifted twice in one
file" shape is exactly this (INV-5).

SCOPE DISCIPLINE — this file pins SIGNATURE-DERIVED NAMES ONLY.  No bullet or
em-dash shape, no verbatim prose substrings, no wording pins, no splice-order
`endswith` check.  So a rewording of the instructions never produces a false
failure, while a RENAMED parameter still produces a true one.

It lives in the fused-memory suite because `fused-memory/pyproject.toml` sets
``pythonpath = ["src", "../orchestrator/src"]``, making both packages hard
imports here.  The orchestrator suite cannot import `fused_memory` at all, so
the same assertion there would degrade to a `pytest.importorskip` — a drift
guard that silently evaporates, the worst possible shape for one.

Task 3202's `test_metadata_vocabulary_prompt_pinning.py` is an unmerged SIBLING
covering the metadata-vocabulary registry.  If both land, reconcile them into
one file rather than duplicating the harness.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

from orchestrator.agents.roles import _MEMORY_INSTRUCTIONS, ROLES

from fused_memory.server.tools import create_mcp_server


def _caller_param_names() -> list[str]:
    """Read the caller_* parameter names off the LIVE search tool signature.

    `search` is a closure inside `create_mcp_server`, so there is no
    module-level symbol to import — `_tool_manager.get_tool(...).fn` is the
    established accessor spelling in this suite.
    """
    tool = create_mcp_server(AsyncMock())._tool_manager.get_tool('search')
    assert tool is not None, 'the search tool is not registered on the MCP server'
    return [n for n in inspect.signature(tool.fn).parameters if n.startswith('caller_')]


def test_memory_instructions_name_every_caller_param():
    names = _caller_param_names()

    assert names, (
        'The search tool declares NO caller_* parameter, so this pin would pass '
        'vacuously — which is the one way a drift guard can be worse than absent. '
        'RED: task 3212 item (2) has not landed.'
    )
    missing = [n for n in names if n not in _MEMORY_INSTRUCTIONS]
    assert not missing, (
        f'_MEMORY_INSTRUCTIONS does not name {missing} — an agent cannot pass a '
        'parameter the instructions never mention, so the journal stays '
        'unattributed while the tool looks instrumented. RED: prose not updated.'
    )


def test_memory_instructions_are_spliced_into_at_least_one_role():
    spliced = sorted(n for n, r in ROLES.items() if _MEMORY_INSTRUCTIONS in r.system_prompt)

    assert spliced, (
        'No role splices _MEMORY_INSTRUCTIONS into its system_prompt, so the block '
        'above is pinned against prose no agent ever reads. RED: splice lost.'
    )
