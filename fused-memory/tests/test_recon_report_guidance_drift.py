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
  constant (no unrendered manual edits can slip in).

start_report is harness-called (agents never call it themselves) and is
intentionally excluded — see reconciliation/prompts/__init__.py.
"""
from __future__ import annotations

import inspect

import pytest

from fused_memory.reconciliation.prompts import (
    _RECON_REPORT_TOOL_GUIDANCE,
    render_recon_report_tool_guidance,
)
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
