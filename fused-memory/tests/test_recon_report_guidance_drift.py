"""Drift-guard tests for _RECON_REPORT_TOOL_GUIDANCE (task-2559).

_RECON_REPORT_TOOL_GUIDANCE (reconciliation/prompts/__init__.py) is generated
from live recon_report FastMCP tool signatures via
render_recon_report_tool_guidance() rather than hand-transcribed, because
hand-maintenance drifted twice — most recently, every example omitted the
required `run_id` kwarg, undetected across a full reviewer round.

This module asserts:
- every agent-called report tool's rendered call shape in the shipped
  guidance carries EVERY one of its live signature's parameters (so a
  future signature change cannot silently drop a required kwarg again);
- render_recon_report_tool_guidance() is idempotent with the shipped
  constant (no unrendered manual edits can slip in);
- every report-tool call example in the fully ASSEMBLED stage prompts
  (bare inline examples included, not just the generated guidance block)
  shows `run_id` — this is the only way to catch hand-written inline
  examples like stage2.py's cross-project routing snippets or stage3.py's
  verification pseudocode, which render_recon_report_tool_guidance() never
  touches.

start_report is harness-called (agents never call it themselves) and is
intentionally excluded — see reconciliation/prompts/__init__.py.
"""
from __future__ import annotations

import inspect
import re

import pytest

from fused_memory.reconciliation.prompts import (
    _RECON_REPORT_TOOL_GUIDANCE,
    render_recon_report_tool_guidance,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.server.recon_report import ReconReportState, create_recon_report_server

# Agent-called report tools (excludes start_report — harness-called).
_AGENT_CALLED_REPORT_TOOLS = (
    'add_finding',
    'set_stat',
    'inc_stat',
    'complete',
    'cite_entity',
    'cite_edge',
    'cite_task',
    'cite_memory',
)


def _live_tools():
    """Build a throwaway recon_report server and return its tool registry."""
    state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0)
    mcp = create_recon_report_server(state)
    return mcp._tool_manager._tools


def _extract_call_args(text: str, call_opener: str) -> str:
    """Return the balanced-paren argument substring following *call_opener* in *text*.

    *call_opener* must include the tool name and the opening paren, e.g.
    ``"mcp__recon-report__add_finding("``.
    """
    start = text.index(call_opener)
    paren_idx = start + len(call_opener) - 1
    assert text[paren_idx] == '('
    depth = 0
    for i in range(paren_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[paren_idx + 1 : i]
    raise AssertionError(f'Unbalanced parens scanning for {call_opener!r}')


def _iter_call_openers(text: str, tool_name: str):
    """Yield the index of the opening '(' for every *tool_name* call in *text*.

    Matches both the bare form (``tool_name(``) and the
    ``mcp__recon-report__``-prefixed form (``mcp__recon-report__tool_name(``),
    as long as the call opener is not itself embedded inside a longer
    identifier — i.e. the character immediately before the match (before
    the prefix, if present, otherwise before the bare name) must not be an
    identifier character.
    """
    pattern = re.compile(r'(?<![A-Za-z0-9_])(?:mcp__recon-report__)?' + re.escape(tool_name) + r'\(')
    for m in pattern.finditer(text):
        yield m.end() - 1


def _extract_call_args_at(text: str, paren_idx: int) -> str:
    """Return the balanced-paren argument substring starting at *paren_idx* ('(')."""
    assert text[paren_idx] == '('
    depth = 0
    for i in range(paren_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[paren_idx + 1 : i]
    raise AssertionError(f'Unbalanced parens scanning from index {paren_idx}')


class TestReconReportGuidanceDrift:
    """_RECON_REPORT_TOOL_GUIDANCE call shapes match live tool signatures."""

    @pytest.mark.parametrize('tool_name', _AGENT_CALLED_REPORT_TOOLS)
    def test_call_shape_includes_every_signature_param(self, tool_name):
        """Every param of the live signature appears as `param=` in the rendered call."""
        tools = _live_tools()
        sig = inspect.signature(tools[tool_name].fn)
        call_opener = f'mcp__recon-report__{tool_name}('
        args_substr = _extract_call_args(_RECON_REPORT_TOOL_GUIDANCE, call_opener)
        for param_name in sig.parameters:
            assert f'{param_name}=' in args_substr, (
                f'{tool_name} call shape missing `{param_name}=` — '
                f'got: {call_opener}{args_substr})'
            )

    @pytest.mark.parametrize('tool_name', _AGENT_CALLED_REPORT_TOOLS)
    def test_call_shape_shows_run_id_placeholder(self, tool_name):
        """Every report-tool example shows the canonical run_id placeholder."""
        call_opener = f'mcp__recon-report__{tool_name}('
        args_substr = _extract_call_args(_RECON_REPORT_TOOL_GUIDANCE, call_opener)
        assert 'run_id=<from Reconciliation Context>' in args_substr

    def test_render_function_is_idempotent_with_shipped_constant(self):
        """render_recon_report_tool_guidance() reproduces the shipped constant exactly."""
        assert render_recon_report_tool_guidance() == _RECON_REPORT_TOOL_GUIDANCE


# Fully assembled stage-prompt texts to scan for inline report-tool call examples.
# build_stage2_system_prompt('autopilot_video') is included because it injects
# AUTOPILOT_VIDEO_CONTAMINATION_GUARDRAIL, which carries its own cross-project
# add_finding example only present under that project_id — the plain
# STAGE2_SYSTEM_PROMPT / build_stage2_system_prompt('dark_factory') texts never
# surface it.
_ASSEMBLED_PROMPT_CASES = (
    ('STAGE1_SYSTEM_PROMPT', STAGE1_SYSTEM_PROMPT),
    ('STAGE2_SYSTEM_PROMPT', STAGE2_SYSTEM_PROMPT),
    ('STAGE3_SYSTEM_PROMPT', STAGE3_SYSTEM_PROMPT),
    ("build_stage2_system_prompt('dark_factory')", build_stage2_system_prompt('dark_factory')),
    ("build_stage2_system_prompt('autopilot_video')", build_stage2_system_prompt('autopilot_video')),
)


class TestReconReportRunIdGuardOverAssembledPrompts:
    """Every report-tool call example in the ASSEMBLED stage prompts shows run_id.

    render_recon_report_tool_guidance() (see TestReconReportGuidanceDrift above)
    only generates the shared guidance block — it cannot reach hand-written
    inline examples elsewhere in a stage prompt. This class scans the fully
    assembled prompt TEXT (not source files) for every report-tool call opener,
    bare or `mcp__recon-report__`-prefixed, and asserts each one's argument
    list carries `run_id`. This covers stage2.py's cross-project routing
    examples, stage3.py's verification pseudocode, and the autopilot_video.py
    contamination-guardrail example.

    Scanning assembled prompt text (rather than source modules) means
    recon_self_model.MCP_CALL_SIGNATURES — verified never rendered into any
    stage prompt, and task 2231's (W5-ξ) domain — cannot be matched here.

    start_report is harness-called (agents never call it themselves) and is
    intentionally excluded, as in TestReconReportGuidanceDrift.
    """

    @pytest.mark.parametrize(
        'prompt_label,prompt_text',
        _ASSEMBLED_PROMPT_CASES,
        ids=[label for label, _ in _ASSEMBLED_PROMPT_CASES],
    )
    def test_every_report_tool_call_example_shows_run_id(self, prompt_label, prompt_text):
        for tool_name in _AGENT_CALLED_REPORT_TOOLS:
            for paren_idx in _iter_call_openers(prompt_text, tool_name):
                args_substr = _extract_call_args_at(prompt_text, paren_idx)
                assert 'run_id' in args_substr, (
                    f'{prompt_label}: a `{tool_name}(...)` call example is missing '
                    f'`run_id` — got: {tool_name}({args_substr})'
                )
