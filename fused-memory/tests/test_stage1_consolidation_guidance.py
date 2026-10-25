"""Wiring contracts for Stage 1's consolidation guidance (tasks 3134, 5239).

Stage 1 PROPOSES a near-duplicate cluster; code decides whether it is
consolidated and executes it (PRD plans/memory-auto-consolidation-prd.md D12).

These tests pin the WIRING — every tool a stage's tool block advertises is one
the server registers and the stage holds, the proposal call example matches
the registered schema, a text shared by both stages names no tool either is
denied, a shared constant renders exactly once — and NOT the prose, which may
be reworded freely. Same convention as
``tests/test_recon_gate_closure_guidance.py``.

Pinning a prompt IS behaviour testing here: a reconciliation stage is an LLM
agent whose only runtime artifact is its system prompt, and
``--disallowed-tools`` OMITS a denied tool rather than rejecting the call, so a
stage told to use a tool it does not hold cannot tell why it is missing.

Division of labour with task 3112 (``## Consolidation Gate``, rendered from
``fused_memory.reconciliation.consolidation_gate``): 3112 owns the TARGET END
STATE; Stage 1's proposal section owns how a cluster is put forward. The end
state is cross-referenced, never restated (INV-5).
"""

from __future__ import annotations

import re

import pytest
from _fused_memory_tool_surface import (
    FUSED_MEMORY_TOOL_PREFIX,
    denied_fused_memory_tool_names,
    fused_memory_tool_names_mentioned,
    registered_fused_memory_tools,
)

from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED, STAGE2_DISALLOWED
from fused_memory.reconciliation.consolidation_proposal import PROPOSER_AGENT_ID
from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
from fused_memory.reconciliation.prompts.stage1 import (
    PROPOSING_A_CONSOLIDATION_HEADING,
    STAGE1_SYSTEM_PROMPT,
)
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt

_PROPOSE_TOOL_ID = f'{FUSED_MEMORY_TOOL_PREFIX}propose_consolidation'
_ADVERTISED_ID_RE = re.compile(re.escape(FUSED_MEMORY_TOOL_PREFIX) + r'[a-z_]+')
_STAGE2_SYSTEM_PROMPT = build_stage2_system_prompt('dark_factory')


def _slice(prompt: str, heading: str) -> str:
    """*heading*'s slice of *prompt*, up to the next top-level heading.

    Keeps each assertion scoped to the section under test, so a token that
    happens to appear elsewhere in a 60k-char prompt cannot make a
    section-level claim pass vacuously.
    """
    start = prompt.index(heading)
    end = prompt.find('\n## ', start + len(heading))
    return prompt[start:] if end == -1 else prompt[start:end]


def _section(heading: str) -> str:
    return _slice(STAGE1_SYSTEM_PROMPT, heading)


def _advertised_tool_ids(prompt: str) -> set[str]:
    return set(_ADVERTISED_ID_RE.findall(_slice(prompt, '## Available Tools')))


def _call_text(text: str, callee: str) -> str:
    """The argument text of the first ``callee(...)`` in *text*, brackets balanced."""
    start = text.index(f'{callee}(') + len(callee) + 1
    depth = 1
    for i, char in enumerate(text[start:], start):
        depth += {'(': 1, ')': -1}.get(char, 0)
        if depth == 0:
            return text[start:i]
    raise AssertionError(f'unbalanced call to {callee}')


class TestEveryAdvertisedToolIsHeld:
    """A tool block may advertise only tools the server registers and the stage holds."""

    @pytest.mark.parametrize(
        ('prompt', 'disallowed'),
        [
            pytest.param(STAGE1_SYSTEM_PROMPT, STAGE1_DISALLOWED, id='stage1'),
            pytest.param(_STAGE2_SYSTEM_PROMPT, STAGE2_DISALLOWED, id='stage2'),
        ],
    )
    def test_every_advertised_tool_is_registered_and_not_denied(self, prompt, disallowed):
        advertised = _advertised_tool_ids(prompt)

        assert advertised, 'the tool block advertises no fused-memory tool at all'
        assert not advertised - registered_fused_memory_tools().keys()
        assert not advertised & set(disallowed)


def test_stage1_advertises_the_proposal_tool():
    assert _PROPOSE_TOOL_ID in _advertised_tool_ids(STAGE1_SYSTEM_PROMPT)


class TestTheProposalCallMatchesTheToolSchema:
    """The section's call example must be a call the registered tool accepts."""

    def _kwargs(self) -> set[str]:
        call = _call_text(_section(PROPOSING_A_CONSOLIDATION_HEADING), _PROPOSE_TOOL_ID)
        return set(re.findall(r'(?:^|,)\s*([a-z_]+)=', call))

    def _schema(self) -> dict:
        return registered_fused_memory_tools()[_PROPOSE_TOOL_ID].inputSchema

    def test_every_example_kwarg_is_a_real_parameter(self):
        assert self._kwargs() <= set(self._schema()['properties'])

    def test_the_example_names_every_required_parameter(self):
        assert set(self._schema()['required']) <= self._kwargs()

    def test_the_example_agent_id_is_the_one_the_tool_admits(self):
        call = _call_text(_section(PROPOSING_A_CONSOLIDATION_HEADING), _PROPOSE_TOOL_ID)

        assert re.findall(r"agent_id='([^']*)'", call) == [PROPOSER_AGENT_ID]

    def test_the_section_is_not_duplicated(self):
        # A second copy would make every slice above scope silently to the first.
        assert STAGE1_SYSTEM_PROMPT.count(PROPOSING_A_CONSOLIDATION_HEADING) == 1

    def test_no_doubled_brace_survives_into_the_rendered_prompt(self):
        # `STAGE1_SYSTEM_PROMPT` is an f-string, so literal braces in its
        # source must be doubled — and a doubled brace SURVIVING into the
        # rendered text means the source over-escaped, shipping a malformed
        # payload example to the agent silently.  (A single-brace slip fails
        # loudly at import instead, so only the quiet direction needs a pin.)
        assert '{{' not in STAGE1_SYSTEM_PROMPT
        assert '}}' not in STAGE1_SYSTEM_PROMPT


class TestStage1CitationGateAlignment:
    """`## UUID Resolution Discipline` must teach the POST-3624 citation gate.

    Exactly two wire-typed things are pinned: the payload literal an agent must
    type verbatim for the server to honour the one sanctioned bypass
    (``metadata={'allow_dangling_citations': True}``, the spelling
    `server/tools.py::_CITATION_REPOINT_REQUIRED_HINT` advertises), and
    ``replacement_memory_id``. The section's prose is deliberately unpinned.
    """

    def test_the_sanctioned_bypass_is_named_as_the_agent_must_type_it(self) -> None:
        section = _section('## UUID Resolution Discipline')
        assert "metadata={'allow_dangling_citations': True}" in section

    def test_the_bypass_does_not_displace_the_survivor_naming_rule(self) -> None:
        # Anti-over-correction: `replacement_memory_id` remains the correct
        # answer for a delete that has a survivor.
        assert 'replacement_memory_id' in _section('## UUID Resolution Discipline')


class TestStage1DoesNotCiteTheRetiredExemption:
    """No recon-stage-scoped story may survive in the delete-discipline section.

    The one remaining `recon-stage-*` mention in stage1.py belongs to the
    `ReconMixedFramingWriteRejected` write gate — a still-live, unrelated
    mechanism — and sits in a different section, outside the slice asserted here.
    """

    def test_the_delete_discipline_section_names_no_recon_stage_exemption(self) -> None:
        assert 'recon-stage' not in _section('## UUID Resolution Discipline')


class TestTheSharedNorm:
    """``STALE_KNOWLEDGE_ANNOTATION_NORM`` renders verbatim into BOTH stage prompts.

    So it may name only tools BOTH stages hold: a shared text that prescribes
    a tool one rendering stage is denied instructs that stage to do something
    it cannot, with no way to tell why.
    """

    def test_the_shared_norm_names_no_tool_either_stage_is_denied(self) -> None:
        named = fused_memory_tool_names_mentioned(STALE_KNOWLEDGE_ANNOTATION_NORM)
        denied = denied_fused_memory_tool_names(STAGE1_DISALLOWED, STAGE2_DISALLOWED)

        assert not named & denied

    def test_it_still_renders_exactly_once_into_both_stage_prompts(self) -> None:
        # Same exactly-once shape as
        # tests/test_stages.py::test_boundary_note_rendered_verbatim_into_all_three_stage_prompts.
        # A shared constant that renders twice states its rule twice.
        assert STAGE1_SYSTEM_PROMPT.count(STALE_KNOWLEDGE_ANNOTATION_NORM) == 1
        assert _STAGE2_SYSTEM_PROMPT.count(STALE_KNOWLEDGE_ANNOTATION_NORM) == 1

    def test_it_keeps_the_must_nots_its_own_comment_block_demands(self) -> None:
        # Live wiring constraints, not style: build_stage2_system_prompt RAISES
        # unless '## Available Tools' appears exactly once in
        # STAGE2_SYSTEM_PROMPT, and test_recon_report_guidance_drift.py requires
        # every `mcp__recon-report__` example in an assembled prompt to carry
        # `run_id=`.
        assert '## Available Tools' not in STALE_KNOWLEDGE_ANNOTATION_NORM
        assert 'mcp__recon-report__' not in STALE_KNOWLEDGE_ANNOTATION_NORM

    def test_it_does_not_cross_reference_a_stage1_only_heading(self) -> None:
        # The proposal section exists in stage1.py only; stage2.py has none.
        assert PROPOSING_A_CONSOLIDATION_HEADING not in STALE_KNOWLEDGE_ANNOTATION_NORM

    def test_it_keeps_the_survivor_naming_rule(self) -> None:
        # Anti-over-correction: `replacement_memory_id` stays right for
        # superseding a SINGLE record, which the norm still covers.
        assert 'replacement_memory_id' in STALE_KNOWLEDGE_ANNOTATION_NORM


class TestStage1IsToldTheGuardNowAppliesToIt:
    """Stage 1 must be told what a near-duplicate soft-block means for it.

    A soft-block advertises `allow_near_duplicate` as the way through, so left
    unexplained the reflex is to set the override and carry on — re-creating
    the +1-per-pass ratchet. The pins are on the DIRECTIVE as much as the
    mechanism: naming the flag is not enough unless the same section routes a
    soft-block to the proposal instead of to the override.
    """

    def test_the_override_flag_is_named_in_the_proposal_section(self) -> None:
        assert 'allow_near_duplicate' in _section(PROPOSING_A_CONSOLIDATION_HEADING)

    def test_the_soft_block_is_named_by_its_wire_identity(self) -> None:
        # `server/near_duplicate_guard.py` emits this exact `error_type`.
        assert (
            'ProceduralKnowledgeNearDuplicateWriteRejected'
            in _section(PROPOSING_A_CONSOLIDATION_HEADING)
        )

    def test_a_soft_block_is_routed_to_the_proposal_not_to_the_override(self) -> None:
        # Both tokens in the SAME slice, so the prompt cannot name the escape
        # hatch in one place and the sanctioned path in another.
        section = _section(PROPOSING_A_CONSOLIDATION_HEADING)
        assert 'allow_near_duplicate' in section
        assert 'propose_consolidation' in section

    def test_the_override_is_never_advertised_outside_this_section(self) -> None:
        # The flag may be shown at most once, and only inside the section that
        # also states when NOT to use it.
        literal = "metadata={'allow_near_duplicate': True}"
        assert STAGE1_SYSTEM_PROMPT.count(literal) <= 1
        if literal in STAGE1_SYSTEM_PROMPT:
            assert literal in _section(PROPOSING_A_CONSOLIDATION_HEADING)
