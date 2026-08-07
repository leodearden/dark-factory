"""Tests pinning that the recon Stage 1/2 prompts advertise the memory-amend
and episode-mutation MCP tools they can already call (esc-3391-1 ruling).

The recon stage prompts ARE the documentation for the only consumer of these
tools — the sleep-mode CLI agents that run each reconciliation stage have no
other source of guidance. `--disallowed-tools` OMITS a denied tool from the
agent's tool listing rather than rejecting an attempted call
(cli_stage_runner.py:96-104), so the inverse failure mode is just as real: a
tool the agent genuinely holds but that the curated "## Available Tools"
block never mentions is a tool the agent simply never learns exists. That is
a legibility defect, not a cosmetic one (esc-3391-1):
`mcp__fused-memory__update_memory`, `mcp__fused-memory__redact_episode_content`,
and `mcp__fused-memory__delete_episode` are all callable from Stage 1 and
Stage 2 today (neither STAGE1_DISALLOWED nor STAGE2_DISALLOWED folds
DISALLOW_MEMORY_WRITES — see cli_stage_runner.py), yet none of the three is
named anywhere in either stage's system prompt.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

# The three tools this task advertises. Parameterizing over this tuple keeps
# the assertion set below in lockstep with the shared constant added to
# prompts/__init__.py (step-2) and with the disallow-list guards added in
# step-3/step-4 — a tool added or removed here should make every relevant
# test class in this file notice.
_ADVERTISED_TOOLS = (
    'mcp__fused-memory__update_memory',
    'mcp__fused-memory__redact_episode_content',
    'mcp__fused-memory__delete_episode',
)


class TestStagePromptsAdvertiseAmendAndEpisodeTools:
    """Stage 1 and Stage 2 prompts must name all three tools; Stage 3 must not."""

    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_advertised_in_stage1_prompt(self, tool_name: str) -> None:
        assert tool_name in STAGE1_SYSTEM_PROMPT, (
            f'{tool_name} must be advertised in STAGE1_SYSTEM_PROMPT — Stage 1 '
            'can already call it (not in STAGE1_DISALLOWED) but the prompt '
            'never names it.'
        )

    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_advertised_in_stage2_prompt(self, tool_name: str) -> None:
        assert tool_name in STAGE2_SYSTEM_PROMPT, (
            f'{tool_name} must be advertised in STAGE2_SYSTEM_PROMPT — Stage 2 '
            'can already call it (not in STAGE2_DISALLOWED) but the prompt '
            'never names it.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_advertised_in_both_build_stage2_system_prompt_branches(
        self, tool_name: str, project_id: str
    ) -> None:
        """Pinned across BOTH conditional branches of the runtime builder — the
        same discipline test_recon_gate_closure_guidance.py applies to
        _GATE_CLOSURE_ARCHIVE_GUIDANCE."""
        built = build_stage2_system_prompt(project_id)
        assert tool_name in built, (
            f'build_stage2_system_prompt({project_id!r}) must still advertise '
            f'{tool_name}.'
        )

    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_absent_from_stage3_prompt(self, tool_name: str) -> None:
        """Negative guard: a future edit must not advertise a write tool into
        the read-only stage."""
        assert tool_name not in STAGE3_SYSTEM_PROMPT, (
            f'{tool_name} must NOT appear in STAGE3_SYSTEM_PROMPT — Stage 3 is '
            'read-only and must never be told about a write tool.'
        )

    def test_build_stage2_system_prompt_autopilot_video_does_not_raise(self) -> None:
        """The '## Available Tools' sentinel must still occur exactly once in
        STAGE2_SYSTEM_PROMPT after later steps splice text into that block —
        build_stage2_system_prompt raises RuntimeError otherwise."""
        built = build_stage2_system_prompt('autopilot_video')
        assert isinstance(built, str) and built
