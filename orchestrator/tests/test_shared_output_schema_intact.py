"""I-SHARED-INTACT regression guard (PRD ``plans/mcp-verdict-servers-prd.md``
task η / task 2487).

Task η removed the ``--json-schema`` / ``result.structured_output`` scaffold
from the four *orchestrator verdict roles* (merger/reviewer/triage/judge),
which now read their verdicts from on-disk artifacts. It deliberately made
NO change to the SHARED ``shared.cli_invoke`` machinery — the
``--json-schema`` render, the CLI-2.1.168 ``'*'`` deny-list expansion (which
omits the synthetic ``StructuredOutput`` tool), and the
``structured_output`` parse — because fused-memory recon / curator /
adjudicator still ride it (``reconciliation/stages/base.py`` →
``run_stage_via_cli(output_schema=get_report_schema())``;
``task_curator.py``).

This standing guard drives ``shared.cli_invoke.invoke_claude_agent`` with a
recon/curator-like config (a non-trivial object ``output_schema`` +
``disallowed_tools=['*']``) and asserts the shared path still:
  (a) renders ``--json-schema`` with the schema JSON;
  (b) expands the ``'*'`` wildcard to ``_REAL_BUILTIN_TOOLS_DENYLIST`` (real
      builtins denied, ``StructuredOutput`` NOT denied); and
  (c) parses a ``StructuredOutput``-bearing subprocess result back into
      ``AgentResult.structured_output``.

It is green-by-construction today (η touched none of this) and has no impl
pair — its value is future regression detection: it fails loudly only if a
later edit breaks the recon/curator output_schema path.

ADDITIVE to — and does NOT modify — the canonical shared-package coverage
(``shared/tests/test_build_claude_argv.py``,
``shared/tests/test_cli_invoke.py``); it re-asserts the same invariant from
the orchestrator's test tree per task η's stated module scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from shared.cli_invoke import (
    _REAL_BUILTIN_TOOLS_DENYLIST,
    _SCHEMA_OUTPUT_TOOL,
    AgentResult,
    _SubprocessResult,
    invoke_claude_agent,
)

# A non-trivial object schema, standing in for a recon/curator report schema
# (get_report_schema()) — the literal shape whose output_schema flows through
# build_claude_argv on the shared path.
_RECON_LIKE_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'summary': {'type': 'string'},
        'findings': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['summary', 'findings'],
}

# The payload a successful StructuredOutput tool call reports in the CLI's
# JSON stdout — what must survive back into AgentResult.structured_output.
_STRUCTURED_PAYLOAD: dict[str, Any] = {'summary': 'ok', 'findings': ['a', 'b']}


def _make_subprocess_result(**overrides: Any) -> _SubprocessResult:
    """A canned _SubprocessResult whose stdout carries a StructuredOutput payload."""
    stdout = json.dumps({
        'result': 'ok',
        'is_error': False,
        'subtype': 'success',
        'cost_usd': 0.01,
        'duration_ms': 100,
        'num_turns': 1,
        'session_id': 'test-session',
        'structured_output': _STRUCTURED_PAYLOAD,
    })
    defaults: dict[str, Any] = {
        'stdout': stdout,
        'stderr': '',
        'returncode': 0,
        'duration_ms': 100,
        'timed_out': False,
    }
    defaults.update(overrides)
    return _SubprocessResult(**defaults)


async def _drive_recon_like(cwd: Path) -> tuple[list[str], AgentResult]:
    """Invoke the shared non-sandbox path with a recon/curator-like config,
    returning the argv handed to _run_subprocess and the parsed AgentResult.
    """
    captured: list[list[str]] = []

    async def fake_run_subprocess(cmd: list[str], *args: Any, **kwargs: Any) -> _SubprocessResult:
        captured.append(list(cmd))
        return _make_subprocess_result()

    with patch('shared.cli_invoke._run_subprocess', side_effect=fake_run_subprocess):
        result = await invoke_claude_agent(
            prompt='hello',
            system_prompt='sys prompt',
            cwd=cwd,
            model='opus',
            max_turns=20,
            max_budget_usd=3.0,
            allowed_tools=['Read'],
            disallowed_tools=['*'],
            mcp_config=None,
            output_schema=_RECON_LIKE_SCHEMA,
            permission_mode='bypassPermissions',
        )

    assert len(captured) == 1, f'expected exactly one subprocess invocation; got {captured!r}'
    return captured[0], result


@pytest.mark.asyncio
async def test_json_schema_rendered_with_schema_json(tmp_path: Path) -> None:
    """(a) the shared path renders ``--json-schema`` immediately followed by
    ``json.dumps(output_schema)`` — the recon/curator schema contract.
    """
    argv, _ = await _drive_recon_like(tmp_path)

    assert '--json-schema' in argv
    idx = argv.index('--json-schema')
    assert argv[idx + 1] == json.dumps(_RECON_LIKE_SCHEMA)


@pytest.mark.asyncio
async def test_wildcard_deny_expands_and_spares_structured_output_tool(tmp_path: Path) -> None:
    """(b) ``'*'`` + output_schema expands to _REAL_BUILTIN_TOOLS_DENYLIST
    (real builtins e.g. Bash/Read denied) while the synthetic
    ``StructuredOutput`` tool is NOT denied anywhere in argv.
    """
    argv, _ = await _drive_recon_like(tmp_path)

    d_idx = argv.index('--disallowed-tools')
    js_idx = argv.index('--json-schema', d_idx)
    deny_values = argv[d_idx + 1:js_idx]

    assert deny_values == _REAL_BUILTIN_TOOLS_DENYLIST, f'got {deny_values!r}'
    assert '*' not in deny_values
    # The schema tool must survive the expansion — denying it would
    # permission-block every structured-output call (the CLI-2.1.168 bug).
    assert _SCHEMA_OUTPUT_TOOL not in argv


@pytest.mark.asyncio
async def test_structured_output_still_parses_from_subprocess(tmp_path: Path) -> None:
    """(c) a StructuredOutput-bearing subprocess result still parses back into
    ``AgentResult.structured_output`` — the end-to-end "still produces
    structured_output" signal (boundary test #14).
    """
    _, result = await _drive_recon_like(tmp_path)

    assert isinstance(result, AgentResult)
    assert result.structured_output == _STRUCTURED_PAYLOAD
