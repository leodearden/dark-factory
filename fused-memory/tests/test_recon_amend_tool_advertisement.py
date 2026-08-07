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

from pathlib import Path

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


class TestDisallowListForAmendAndEpisodeTools:
    """Pins the grant the prompt now advertises, so a future disallow-list edit
    cannot silently make the prompt lie.

    Two parts. PART A is the cheap structural guard (set membership, mirroring
    the established shape at test_merge_entities.py:948-959 and
    test_redact_episode_content.py:491-506). PART B is the by-MEASUREMENT pin
    the 2026-08-06 operator amendment (esc-3623-3) requires: "assert on the
    RENDERED Stage 3 tool surface, not merely on set membership. A test that
    only checks 'update_memory' in STAGE3_DISALLOWED can pass while the tool
    remains live if the constant is composed differently than expected." Part B
    exists because `--disallowed-tools` OMITS a denied tool from the agent's
    tool listing rather than rejecting an attempted call
    (cli_stage_runner.py:96-104) — so a denial is only real once it reaches the
    rendered CLI argv, not merely a Python list.

    Measured fact this guard exists for: `redact_episode_content` and
    `delete_episode` already satisfy every assertion below, but `update_memory`
    shipped with task 3088 in NO disallow list at all — Stage 3 (the read-only
    integrity check) held a live Mem0 silent-rewrite primitive. Its agent_id
    `recon-stage-integrity_check` matches `Mem0UpdateConfig`'s default
    `content_amend_allowed_agent_prefixes=['recon-stage-']`, and the
    `mem0_update` block in `config/config.yaml` is fully commented out, so that
    default applies — Stage 3 both passed the authz gate and held the tool.
    """

    # -- PART A: set membership (cheap structural guard) --------------------

    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_not_in_stage1_disallowed(self, tool_name: str) -> None:
        """Stage 1 may call it — what esc-3391-1's gate premise got wrong."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert tool_name not in STAGE1_DISALLOWED

    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_not_in_stage2_disallowed(self, tool_name: str) -> None:
        """Stage 2 holds the full memory write surface."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE2_DISALLOWED
        assert tool_name not in STAGE2_DISALLOWED

    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_in_stage3_disallowed(self, tool_name: str) -> None:
        """Stage 3 is read-only."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert tool_name in STAGE3_DISALLOWED

    @pytest.mark.parametrize('tool_name', _ADVERTISED_TOOLS)
    def test_in_disallow_memory_writes(self, tool_name: str) -> None:
        """The single list through which the Stage 3 denial above is achieved
        (folded into STAGE3_DISALLOWED only) — pinned separately so a future
        edit satisfying the assertion above by some other route is still
        visible."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert tool_name in DISALLOW_MEMORY_WRITES

    # -- PART B: the RENDERED Stage 3 tool surface (by-measurement pin) -----

    def test_stage3_rendered_argv_denies_the_amend_and_episode_tools(self) -> None:
        """Measure the denial at the end of the real pipeline instead of
        asserting on a constant.

        Path: IntegrityCheck.get_disallowed_tools() -> run_stage_via_cli
        (disallowed_tools=..., cli_stage_runner.py:463) -> invoke_with_cap_retry
        (disallowed_tools=...) -> shared.cli_invoke.build_claude_argv, which
        renders the actual `--disallowed-tools` CLI flag (cli_invoke.py:1849).
        This runs through both the genuine stage hook (so a stage overriding
        get_disallowed_tools() would be caught) and the argv renderer (so a
        differently-composed disallow constant would be caught) — the failure
        mode a pure set-membership assertion (Part A) cannot see.

        output_schema={'type': 'object'} is passed to match the realistic
        recon call site and exercise (not dodge) cli_invoke.py:1845-1848's
        wildcard-expansion branch. That branch only rewrites disallowed_tools
        when '*' is present; DISALLOW_BUILTIN (['Bash', 'Edit', 'Write',
        'NotebookEdit']) carries no '*', so STAGE3_DISALLOWED renders as a
        straight passthrough here — confirmed against today's code.
        """
        from shared.cli_invoke import build_claude_argv
        from test_stages import _mock_stage_deps

        from fused_memory.models.reconciliation import StageId
        from fused_memory.reconciliation.stages.task_knowledge_sync import IntegrityCheck

        stage = IntegrityCheck(StageId.integrity_check, **_mock_stage_deps())
        disallowed_tools = stage.get_disallowed_tools()

        cmd, temp_files = build_claude_argv(
            model='opus',
            max_budget_usd=5.0,
            system_prompt='irrelevant to this test — only --disallowed-tools is inspected',
            max_turns=50,
            permission_mode='bypassPermissions',
            allowed_tools=None,
            disallowed_tools=disallowed_tools,
            mcp_config=None,
            output_schema={'type': 'object'},
            effort=None,
            resume_session_id=None,
            session_id=None,
        )
        try:
            flag_idx = cmd.index('--disallowed-tools')
            rendered = []
            for token in cmd[flag_idx + 1:]:
                if token.startswith('--'):
                    break
                rendered.append(token)

            assert 'mcp__fused-memory__update_memory' in rendered, (
                'Stage 3 must not be able to invoke update_memory — it must appear in '
                'the RENDERED --disallowed-tools argv, not merely in STAGE3_DISALLOWED.'
            )
            # Passing controls — prove the harness itself is measuring correctly.
            assert 'mcp__fused-memory__redact_episode_content' in rendered
            assert 'mcp__fused-memory__delete_episode' in rendered
        finally:
            for path in temp_files:
                Path(path).unlink(missing_ok=True)


class TestStagePromptsCarryTheAnnotationNorm:
    """Pins the WIRING of the shared stale/wrong-knowledge annotation norm —
    constant identity, verbatim embedding, exactly-once, survives both
    build_stage2_system_prompt branches, absent from Stage 3 — never its
    prose. Same discipline test_recon_gate_closure_guidance.py's module
    docstring states explicitly: "these tests pin the WIRING ... rather than
    the prose ... Prose may be reworded freely; the wiring may not silently
    break."

    STALE_KNOWLEDGE_ANNOTATION_NORM does not exist yet. Each test below
    imports it locally (rather than at module scope) so the ImportError is
    isolated to this class — the other classes in this file, and their
    already-passing assertions, stay unaffected and collectible.
    """

    def test_is_nonempty_str(self) -> None:
        from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
        assert isinstance(STALE_KNOWLEDGE_ANNOTATION_NORM, str)
        assert STALE_KNOWLEDGE_ANNOTATION_NORM

    def test_embedded_exactly_once_in_stage1_prompt(self) -> None:
        from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
        assert STAGE1_SYSTEM_PROMPT.count(STALE_KNOWLEDGE_ANNOTATION_NORM) == 1, (
            'STAGE1_SYSTEM_PROMPT must interpolate STALE_KNOWLEDGE_ANNOTATION_NORM '
            'verbatim, exactly once (INV-5: stated once, interpolated, never re-pasted).'
        )

    def test_embedded_exactly_once_in_stage2_prompt(self) -> None:
        from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
        assert STAGE2_SYSTEM_PROMPT.count(STALE_KNOWLEDGE_ANNOTATION_NORM) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate STALE_KNOWLEDGE_ANNOTATION_NORM '
            'verbatim, exactly once (INV-5: stated once, interpolated, never re-pasted).'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_both_build_stage2_system_prompt_branches(self, project_id: str) -> None:
        from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
        built = build_stage2_system_prompt(project_id)
        assert STALE_KNOWLEDGE_ANNOTATION_NORM in built, (
            f'build_stage2_system_prompt({project_id!r}) dropped '
            'STALE_KNOWLEDGE_ANNOTATION_NORM.'
        )

    def test_absent_from_stage3_prompt(self) -> None:
        """Stage 3 holds none of update_memory / redact_episode_content /
        delete_episode after step-4 — it must not be told how to use them.
        The "never tell a stage about an action it is not sanctioned to
        take" rule that render_escalation_boundary_note's docstring states."""
        from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
        assert STALE_KNOWLEDGE_ANNOTATION_NORM not in STAGE3_SYSTEM_PROMPT

    def test_does_not_contain_the_available_tools_sentinel(self) -> None:
        """build_stage2_system_prompt raises RuntimeError unless
        '## Available Tools' appears exactly once in STAGE2_SYSTEM_PROMPT —
        the constant must not introduce a second occurrence."""
        from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
        assert '## Available Tools' not in STALE_KNOWLEDGE_ANNOTATION_NORM

    def test_does_not_contain_a_bare_recon_report_example(self) -> None:
        """test_recon_report_guidance_drift.py scans the assembled prompts and
        requires every recon-report tool-call example to carry run_id=; the
        norm must introduce none."""
        from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
        assert 'mcp__recon-report__' not in STALE_KNOWLEDGE_ANNOTATION_NORM
