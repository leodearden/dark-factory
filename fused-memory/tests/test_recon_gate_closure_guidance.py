"""Composition-contract tests for _GATE_CLOSURE_ARCHIVE_GUIDANCE (task 3023).

Stage 1 emits the finding/flag and Stage 2 files the remediation task, so the
"a pending-only escalation probe is not proof of absence" suppression policy is
only sound if BOTH stages hold the identical rule.  These tests pin the WIRING
(one shared constant, embedded verbatim in both assembled prompts and surviving
both branches of ``build_stage2_system_prompt``) rather than the prose — the
same shape as the existing ``_STAGE1_/_STAGE2_GRAPHITI_QUEUED_GUIDANCE``
constants.  Prose may be reworded freely; the wiring may not silently break.

Background: a recon auditor probing a ``done`` ``task_kind='deterministic'``
gate task with ``get_pending_escalations(task_id=...)`` gets ``[]`` once a human
has resolved the born-at-L2 record (it moves to ``data/escalations/archive/``),
and repeatedly concluded the record was never written — filing remediation tasks
across 16 recorded instances.
"""

from __future__ import annotations

import re

import pytest

from fused_memory.reconciliation.prompts import _GATE_CLOSURE_ARCHIVE_GUIDANCE
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)

# Mirrors _AGENT_CALLED_REPORT_TOOLS in test_recon_report_guidance_drift.py —
# the tools whose call examples that suite requires to carry `run_id=`.
_AGENT_CALLED_REPORT_TOOLS = (
    'add_finding',
    'set_stat',
    'inc_stat',
    'complete',
    'cite_entity',
    'cite_edge',
    'cite_task',
    'cite_memory',
    'cite_run',
)


class TestGateClosureArchiveGuidance:
    """_GATE_CLOSURE_ARCHIVE_GUIDANCE is a single shared constant wired into both stages."""

    # -- (a) the constant exists ------------------------------------------

    def test_constant_is_a_non_empty_string(self):
        """The shared constant imports and carries real content."""
        assert isinstance(_GATE_CLOSURE_ARCHIVE_GUIDANCE, str)
        assert _GATE_CLOSURE_ARCHIVE_GUIDANCE.strip(), (
            '_GATE_CLOSURE_ARCHIVE_GUIDANCE must not be empty — an empty '
            'constant would satisfy the embedding assertions vacuously.'
        )

    # -- (b) it names the disconfirming tool -------------------------------

    def test_names_the_disconfirming_tool(self):
        """It must name `get_task_escalations` — the guidance is inert without it.

        This is also the token the shipped recon_code_fix_premise_registry.yaml
        entry asserts on, so retiring the tool self-corrects the guard.
        """
        assert 'get_task_escalations' in _GATE_CLOSURE_ARCHIVE_GUIDANCE, (
            'The guidance must name the archive-inclusive lookup the auditor '
            'is required to call; without it the rule cannot be followed.'
        )

    # -- (c)/(d) embedded verbatim in both stage prompts -------------------

    def test_embedded_verbatim_in_stage1_prompt(self):
        """Stage 1 — the stage that emits the flag — carries the guidance."""
        assert _GATE_CLOSURE_ARCHIVE_GUIDANCE in STAGE1_SYSTEM_PROMPT, (
            'STAGE1_SYSTEM_PROMPT must interpolate _GATE_CLOSURE_ARCHIVE_GUIDANCE '
            'verbatim (no re-wording, no partial copy).'
        )

    def test_embedded_verbatim_in_stage2_prompt(self):
        """Stage 2 — the stage that files the remediation task — carries the guidance."""
        assert _GATE_CLOSURE_ARCHIVE_GUIDANCE in STAGE2_SYSTEM_PROMPT, (
            'STAGE2_SYSTEM_PROMPT must interpolate _GATE_CLOSURE_ARCHIVE_GUIDANCE '
            'verbatim (no re-wording, no partial copy).'
        )

    # -- (e) survives both runtime builder branches ------------------------

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        """Both branches of the runtime builder keep the guidance.

        `autopilot_video` injects an extra guardrail section before
        `## Available Tools`; that injection must not displace this guidance.
        """
        built = build_stage2_system_prompt(project_id)

        assert _GATE_CLOSURE_ARCHIVE_GUIDANCE in built, (
            f'build_stage2_system_prompt({project_id!r}) dropped '
            '_GATE_CLOSURE_ARCHIVE_GUIDANCE.'
        )

    # -- (f) drift trap: no bare report-tool call examples ------------------

    def test_no_report_tool_call_example_without_run_id(self):
        """Keep test_recon_report_guidance_drift.py green.

        That suite scans the ASSEMBLED stage prompts and requires every
        recon-report tool call example to carry `run_id=`.  Since this constant
        is interpolated into both stage prompts, a bare `add_finding(...)`
        example written here would fail that suite from a surprising location.
        """
        for tool_name in _AGENT_CALLED_REPORT_TOOLS:
            pattern = re.compile(
                r'(?<![A-Za-z0-9_])(?:mcp__recon-report__)?'
                + re.escape(tool_name)
                + r'\(([^)]*)\)'
            )
            for m in pattern.finditer(_GATE_CLOSURE_ARCHIVE_GUIDANCE):
                assert 'run_id' in m.group(1), (
                    f'_GATE_CLOSURE_ARCHIVE_GUIDANCE contains a '
                    f'`{tool_name}(...)` call example missing `run_id` — got: '
                    f'{tool_name}({m.group(1)}). Describe the behaviour in prose '
                    'instead, or add run_id= to the example.'
                )
