"""The per-(tool, stage) advertisement contract for the STAGE-GATED recon-report tools.

`--disallowed-tools` OMITS a denied tool from the agent's tool listing rather
than rejecting an attempted call (see the `--disallowed-tools` handling in
`build_claude_argv`, shared/cli_invoke.py). That makes the two halves of the
contract equally invisible failures, and this module asserts BOTH:

- DENIES => the stage's prompt must NOT name the tool. Telling a stage about an
  action it cannot take produces silence, not a refusal the agent could act on.
  Already asserted, per (tool, stage), by
  `test_recon_report_guidance_drift.py::test_stage_gated_tools_absent_from_denying_stage_prompts`
  — restated here so this file reads as the WHOLE contract rather than half of
  one, which is the discipline that module's own docstring states ("one contract
  rather than two unrelated guards a future edit can quietly split").
- HOLDS => the stage's prompt must NAME the tool. This is the converse the drift
  guard deliberately declines to assert, naming task 4395 as its owner: a tool
  the agent genuinely holds but that no prompt ever mentions is a tool the agent
  simply never learns exists (the held-but-unadvertised defect esc-3391-1 /
  test_recon_amend_tool_advertisement.py exists to catch, one namespace over).

The stage-agnostic server-level listing (`RECON_REPORT_INSTRUCTIONS`) is NOT a
substitute channel for the HOLDS arm. The claude CLI truncates FastMCP server
`instructions` at 2048 characters; `RECON_REPORT_INSTRUCTIONS` is ~7560, so both
stage-gated tool NAMES survive (they sit in the `Tools:` roster near the top)
while every numbered entry carrying their call shape, gating and error codes
falls past the cut and reaches no agent. The stage prompts are passed as the
system prompt and are subject to no such cap, so they are the only channel that
can carry the contract.

There is deliberately NO hand-maintained tool-name list here. The (tool, stage)
matrix is DERIVED from `STAGE_GATED_REPORT_TOOLS` (which lives next to the
@mcp.tool() registrations) crossed with the live `STAGE{1,2,3}_DISALLOWED` lists
that actually reach `--disallowed-tools`. A newly stage-gated tool is held to
this contract automatically, with no edit to this file. That is the same drift
`test_recon_report_guidance_drift.py` deleted its own hand-written tool tuple to
close ("it lived in a test file the tool author never opens"), and it is the
drift that produced this task: `repair_memory_citation` gained a THIRD holding
(Stage 1) after the original absence guard was written, and nobody noticed.
"""

from __future__ import annotations

import re

import pytest

from fused_memory.reconciliation import cli_stage_runner
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.server.recon_report import STAGE_GATED_REPORT_TOOLS

# (prompt label, assembled prompt, the cli_stage_runner attribute that gates it).
# The disallow-list ATTRIBUTE NAME travels with each case rather than a
# pre-computed denied set, so a failure message can name the list a reader has
# to open to act on it.
_STAGE_CASES = (
    ('STAGE1_SYSTEM_PROMPT', STAGE1_SYSTEM_PROMPT, 'STAGE1_DISALLOWED'),
    ('STAGE2_SYSTEM_PROMPT', STAGE2_SYSTEM_PROMPT, 'STAGE2_DISALLOWED'),
    ('STAGE3_SYSTEM_PROMPT', STAGE3_SYSTEM_PROMPT, 'STAGE3_DISALLOWED'),
)

# Explicit ids: a prompt is a ~50KB string, and pytest's default id for a string
# param is the string itself, which makes every test id here unreadable and
# unselectable on the command line. Named by the stage each case gates.
_STAGE_CASE_IDS = [label for label, _text, _attr in _STAGE_CASES]

# Both branches of the runtime Stage 2 builder — the same discipline
# test_recon_amend_tool_advertisement.py applies to its own advertisement block.
# 'dark_factory' currently returns STAGE2_SYSTEM_PROMPT unchanged;
# 'autopilot_video' splices extra text in, so the two are not interchangeable.
_STAGE2_PROJECT_IDS = ('dark_factory', 'autopilot_video')


def _names_tool(text: str, tool_name: str) -> bool:
    """Is *tool_name* named in *text* as a whole identifier, not as a substring?

    Same shape as `test_recon_report_guidance_drift.py::_names_tool` — copied
    rather than imported across test modules, as that module documents. A bare
    ``tool_name in text`` is not a coverage check: it passes on any longer word
    that merely CONTAINS the name. That hole is LIVE here, not hypothetical —
    `render_entity_standing_decision_schema_section()` puts the string
    `entity_standing_decision` into BOTH stage prompts, so a substring check for
    `write_entity_standing_decision` in the wrong direction would be satisfied by
    a section that never mentions the tool.

    The optional `mcp__recon-report__` prefix is the one deliberate difference
    from the drift module's helper, and mirrors that module's
    `_iter_call_openers`: stage prompts advertise a tool by its FULLY-QUALIFIED
    call name (as `AMEND_AND_EPISODE_TOOLS_BLOCK` does for the fused-memory
    namespace), where the character before the bare name is `_` and a
    prefix-blind identifier boundary would never match.
    """
    pattern = (
        r'(?<![A-Za-z0-9_])(?:mcp__recon-report__)?'
        + re.escape(tool_name)
        + r'(?![A-Za-z0-9_])'
    )
    return re.search(pattern, text) is not None


def _iter_call_openers(text: str, tool_name: str):
    """Yield the index of the opening '(' for every *tool_name* call in *text*.

    Same shape as `test_recon_report_guidance_drift.py::_iter_call_openers` —
    matches the bare and the `mcp__recon-report__`-prefixed form, and refuses a
    match whose opener is embedded inside a longer identifier.
    """
    pattern = re.compile(
        r'(?<![A-Za-z0-9_])(?:mcp__recon-report__)?' + re.escape(tool_name) + r'\('
    )
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


def _stage_denies(tool_name: str, disallowed_attr: str) -> bool:
    """Does the stage gated by *disallowed_attr* deny *tool_name*?

    Reads the live disallow list and strips the `mcp__recon-report__` prefix
    exactly as `test_recon_report_guidance_drift.py` does, so both guards answer
    "may this stage call this tool?" from the same single source of truth — the
    lists that actually render into `--disallowed-tools`.
    """
    denied = {
        name.removeprefix('mcp__recon-report__')
        for name in getattr(cli_stage_runner, disallowed_attr)
    }
    return tool_name in denied


class TestStageGatedToolAdvertisementMatchesTheDisallowLists:
    """Every stage that HOLDS a stage-gated tool names it; every stage that DENIES it does not."""

    @pytest.mark.parametrize(
        'prompt_label,prompt_text,disallowed_attr', _STAGE_CASES, ids=_STAGE_CASE_IDS
    )
    @pytest.mark.parametrize('tool_name', sorted(STAGE_GATED_REPORT_TOOLS))
    def test_held_stage_gated_tool_is_named_in_that_stages_prompt(
        self, tool_name, prompt_label, prompt_text, disallowed_attr
    ):
        if _stage_denies(tool_name, disallowed_attr):
            pytest.skip(
                f'{tool_name} is denied by {disallowed_attr} — the absence arm below owns this pair'
            )
        assert _names_tool(prompt_text, tool_name), (
            f'{prompt_label} never names `{tool_name}`, but that stage HOLDS it — '
            f'it is absent from {disallowed_attr} (cli_stage_runner.py), so the agent '
            'can call it. `--disallowed-tools` OMITS denied tools rather than rejecting '
            'calls, so an unadvertised held tool is indistinguishable from one the stage '
            'does not have: the agent never learns it exists. Advertise it in the stage '
            'prompt — the server-level RECON_REPORT_INSTRUCTIONS listing is truncated by '
            'the CLI at 2048 chars and does not reach the agent with the call shape.'
        )

    @pytest.mark.parametrize(
        'prompt_label,prompt_text,disallowed_attr', _STAGE_CASES, ids=_STAGE_CASE_IDS
    )
    @pytest.mark.parametrize('tool_name', sorted(STAGE_GATED_REPORT_TOOLS))
    def test_denied_stage_gated_tool_is_absent_from_that_stages_prompt(
        self, tool_name, prompt_label, prompt_text, disallowed_attr
    ):
        if not _stage_denies(tool_name, disallowed_attr):
            pytest.skip(
                f'{tool_name} is absent from {disallowed_attr}, so that stage HOLDS it — '
                'the naming arm above owns this pair'
            )
        assert not _names_tool(prompt_text, tool_name), (
            f'{prompt_label} names `{tool_name}`, but that stage DENIES it '
            f'({disallowed_attr} in cli_stage_runner.py). Naming a denied tool in a stage '
            'prompt tells the agent about an action it cannot take, and the denial '
            'surfaces as a silently missing tool rather than an explanation.'
        )

    @pytest.mark.parametrize('project_id', _STAGE2_PROJECT_IDS)
    @pytest.mark.parametrize('tool_name', sorted(STAGE_GATED_REPORT_TOOLS))
    def test_stage2_contract_holds_in_both_build_stage2_system_prompt_branches(
        self, tool_name, project_id
    ):
        """The RUNTIME Stage 2 prompt, not just the module constant.

        Pinned across BOTH conditional branches of the builder — the same
        discipline test_recon_amend_tool_advertisement.py applies to
        AMEND_AND_EPISODE_TOOLS_BLOCK. The 'autopilot_video' branch splices
        additional text into the prompt, so a block that survives the
        'dark_factory' branch is not thereby proven to survive this one.
        """
        built = build_stage2_system_prompt(project_id)
        if _stage_denies(tool_name, 'STAGE2_DISALLOWED'):
            assert not _names_tool(built, tool_name), (
                f'build_stage2_system_prompt({project_id!r}) names `{tool_name}`, but '
                'Stage 2 DENIES it (STAGE2_DISALLOWED in cli_stage_runner.py).'
            )
        else:
            assert _names_tool(built, tool_name), (
                f'build_stage2_system_prompt({project_id!r}) must still name `{tool_name}` '
                '— Stage 2 HOLDS it (absent from STAGE2_DISALLOWED in cli_stage_runner.py) '
                'and a held-but-unadvertised tool is invisible to the agent.'
            )


class TestStageGatedToolBlocksAreWiredNotRePasted:
    """Pins the WIRING of the two advertisement blocks — constant identity,
    verbatim, exactly once, per stage — never their prose.

    This is the INV-5 guard the membership assertions above cannot be. Every
    per-name assertion in the class above passes equally if the block text were
    re-pasted inline in stage1.py / stage2.py instead of interpolating the
    shared constant, which is exactly the drift these constants exist to
    prevent. Same discipline test_recon_gate_closure_guidance.py's docstring
    states: prose may be reworded freely; the wiring may not silently break.

    Each test imports its constant LOCALLY rather than at module scope, the
    pattern TestStagePromptsCarryTheAnnotationNorm documents in
    test_recon_amend_tool_advertisement.py: while a constant does not exist
    yet, its ImportError is isolated to the test that needs it and the
    already-green assertions elsewhere in this file stay collectible.
    """

    def test_citation_repair_block_embedded_exactly_once_in_stage1_prompt(self):
        from fused_memory.reconciliation.prompts import CITATION_REPAIR_TOOL_BLOCK
        assert STAGE1_SYSTEM_PROMPT.count(CITATION_REPAIR_TOOL_BLOCK) == 1, (
            'STAGE1_SYSTEM_PROMPT must interpolate CITATION_REPAIR_TOOL_BLOCK verbatim, '
            'exactly once (INV-5: stated once, interpolated, never re-pasted).'
        )

    def test_citation_repair_block_embedded_exactly_once_in_stage2_prompt(self):
        from fused_memory.reconciliation.prompts import CITATION_REPAIR_TOOL_BLOCK
        assert STAGE2_SYSTEM_PROMPT.count(CITATION_REPAIR_TOOL_BLOCK) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate CITATION_REPAIR_TOOL_BLOCK verbatim, '
            'exactly once (INV-5: stated once, interpolated, never re-pasted).'
        )

    def test_citation_repair_block_absent_from_stage3_prompt(self):
        """Stage 3 is denied repair_memory_citation via
        DISALLOW_RECON_REPORT_JOURNAL_WRITES — it is the stage that DETECTS
        dangling citations, and detect and repair must not be the same actor."""
        from fused_memory.reconciliation.prompts import CITATION_REPAIR_TOOL_BLOCK
        assert CITATION_REPAIR_TOOL_BLOCK not in STAGE3_SYSTEM_PROMPT

    def test_standing_decision_write_block_embedded_exactly_once_in_stage2_prompt(self):
        from fused_memory.reconciliation.prompts import ENTITY_STANDING_DECISION_WRITE_BLOCK
        assert STAGE2_SYSTEM_PROMPT.count(ENTITY_STANDING_DECISION_WRITE_BLOCK) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate ENTITY_STANDING_DECISION_WRITE_BLOCK '
            'verbatim, exactly once (INV-5: stated once, interpolated, never re-pasted).'
        )

    def test_standing_decision_write_block_absent_from_stage1_prompt(self):
        """Stage 2 is the ONLY stage holding write_entity_standing_decision —
        Stage 1 and Stage 3 are denied it via DISALLOW_RECON_REPORT_LEDGER_WRITES.
        A block that leaked into Stage 1 would also trip the absence arm above,
        but this pins the CONSTANT rather than the tool name, so a reworded
        block still cannot drift into the wrong stage."""
        from fused_memory.reconciliation.prompts import ENTITY_STANDING_DECISION_WRITE_BLOCK
        assert ENTITY_STANDING_DECISION_WRITE_BLOCK not in STAGE1_SYSTEM_PROMPT

    def test_standing_decision_write_block_absent_from_stage3_prompt(self):
        from fused_memory.reconciliation.prompts import ENTITY_STANDING_DECISION_WRITE_BLOCK
        assert ENTITY_STANDING_DECISION_WRITE_BLOCK not in STAGE3_SYSTEM_PROMPT

    @pytest.mark.parametrize('project_id', _STAGE2_PROJECT_IDS)
    def test_citation_repair_block_survives_both_build_stage2_branches(self, project_id):
        from fused_memory.reconciliation.prompts import CITATION_REPAIR_TOOL_BLOCK
        built = build_stage2_system_prompt(project_id)
        assert built.count(CITATION_REPAIR_TOOL_BLOCK) == 1, (
            f'build_stage2_system_prompt({project_id!r}) dropped or duplicated '
            'CITATION_REPAIR_TOOL_BLOCK.'
        )

    @pytest.mark.parametrize('project_id', _STAGE2_PROJECT_IDS)
    def test_standing_decision_write_block_survives_both_build_stage2_branches(
        self, project_id
    ):
        from fused_memory.reconciliation.prompts import ENTITY_STANDING_DECISION_WRITE_BLOCK
        built = build_stage2_system_prompt(project_id)
        assert built.count(ENTITY_STANDING_DECISION_WRITE_BLOCK) == 1, (
            f'build_stage2_system_prompt({project_id!r}) dropped or duplicated '
            'ENTITY_STANDING_DECISION_WRITE_BLOCK.'
        )

    @pytest.mark.parametrize(
        'heading',
        ['## Entity Standing Decisions', '## Investigation Outcome Records'],
    )
    def test_headings_the_standing_decision_block_cross_references_exist_in_stage2(
        self, heading
    ):
        """ENTITY_STANDING_DECISION_WRITE_BLOCK deliberately POINTS AT these two
        sections instead of re-pasting them — the grounds enum lives in the
        first (rendered by render_entity_standing_decision_schema_section()) and
        the arm-2 evidence pool in the second (render_investigation_outcome_section()),
        and each is single-sourced there.

        Cross-referencing a heading that does not exist in the target prompt is
        a recorded hazard: STALE_KNOWLEDGE_ANNOTATION_NORM's comment carries two
        such MUST-NOTs, both for headings that exist in one stage prompt and not
        another. That block avoids the hazard by naming no heading at all; this
        one accepts it (it is Stage-2-only, so both headings are reachable) and
        pays for it with this guard. If a renderer is reworded or dropped, the
        pointer must be updated in the same change.
        """
        assert heading in STAGE2_SYSTEM_PROMPT, (
            f'ENTITY_STANDING_DECISION_WRITE_BLOCK points the agent at {heading!r}, but '
            'no such heading is in STAGE2_SYSTEM_PROMPT. Either the renderer that emits '
            'it was reworded/dropped, or the block now points at nothing.'
        )


class TestHazardsTheNewBlocksAreExposedTo:
    """Regression pins for the two prompt-assembly hazards these blocks newly touch.

    Neither is hypothetical: both are recorded in `prompts/__init__.py` as
    MUST-NOTs on every constant interpolated into a stage f-string, and both
    fail in a way that points somewhere other than the block that caused them.
    """

    @pytest.mark.parametrize(
        'block_name',
        ['CITATION_REPAIR_TOOL_BLOCK', 'ENTITY_STANDING_DECISION_WRITE_BLOCK'],
    )
    def test_block_does_not_contain_the_available_tools_sentinel(self, block_name):
        """`build_stage2_system_prompt` raises RuntimeError unless '## Available
        Tools' occurs EXACTLY ONCE in STAGE2_SYSTEM_PROMPT, and both blocks land
        inside it. A block carrying a second copy breaks Stage 2 at build time
        with an error naming the sentinel, not the block."""
        from fused_memory.reconciliation import prompts as prompts_module
        block = getattr(prompts_module, block_name)
        assert '## Available Tools' not in block, (
            f'{block_name} contains the literal "## Available Tools". That sentinel must '
            'occur exactly once in STAGE2_SYSTEM_PROMPT or build_stage2_system_prompt '
            'raises RuntimeError. Reword the heading.'
        )

    def test_build_stage2_system_prompt_autopilot_video_does_not_raise(self):
        """The end-to-end version of the assertion above, mirroring
        test_recon_amend_tool_advertisement.py's guard of the same name: measure
        the sentinel at the builder that actually enforces it."""
        built = build_stage2_system_prompt('autopilot_video')
        assert isinstance(built, str) and built

    @pytest.mark.parametrize(
        'prompt_label,prompt_text,disallowed_attr', _STAGE_CASES, ids=_STAGE_CASE_IDS
    )
    def test_repair_memory_citation_examples_carry_both_run_ids(
        self, prompt_label, prompt_text, disallowed_attr
    ):
        """Every repair_memory_citation call example in an ASSEMBLED prompt shows
        BOTH `run_id` and `target_run_id`.

        `test_recon_report_guidance_drift.py::TestReconReportRunIdGuardOverAssembledPrompts`
        scans `_SHARED_GUIDANCE_REPORT_TOOLS`, which EXCLUDES the stage-gated
        tools — so the one stage-gated tool that IS run-scoped is covered by
        nothing. This is that carve-out's equivalent, and it guards the specific
        trap the block's prose calls out: the two ids are not interchangeable,
        and passing the target's id as `run_id` fails `run_id_unknown` rather
        than doing the intended thing. An example missing either one teaches the
        conflation the prose exists to prevent.

        Runs over all three stages, so a future example spliced into a stage
        that does not hold the tool is caught here too (its absence there is
        asserted separately; this asserts the shape of anything that appears).
        """
        examples = 0
        for paren_idx in _iter_call_openers(prompt_text, 'repair_memory_citation'):
            examples += 1
            args = _extract_call_args_at(prompt_text, paren_idx)
            for required in ('run_id', 'target_run_id'):
                # Identifier boundary, not a substring: `'run_id=' in args` is
                # satisfied by `target_run_id=` alone, which is the very
                # conflation this test exists to catch.
                found = re.search(
                    rf'(?<![A-Za-z0-9_]){re.escape(required)}\s*=', args
                )
                assert found, (
                    f'{prompt_label} contains a repair_memory_citation(...) example '
                    f'missing `{required}=`: {args!r}. `run_id` is the CALLER\'s current '
                    'run and `target_run_id` is the run that OWNS the finding; an example '
                    'showing one without the other invites exactly the conflation that '
                    f'fails `run_id_unknown`. (Stage gating for this prompt: '
                    f'{disallowed_attr}.)'
                )

        # Non-vacuity: a scan that finds nothing is a guard that looks like
        # coverage but isn't — the failure mode test_recon_report_guidance_drift.py
        # names. A stage that HOLDS the tool must show at least one call example,
        # since the call shape is the part the truncated server listing never
        # delivers. A denying stage must show none, which the absence arm asserts.
        if not _stage_denies('repair_memory_citation', disallowed_attr):
            assert examples, (
                f'{prompt_label} shows no repair_memory_citation(...) call example, but '
                f'that stage HOLDS the tool (absent from {disallowed_attr}). Naming a '
                'tool without its call shape is not advertisement: the call shape is '
                'exactly what the CLI-truncated server listing never delivers.'
            )
