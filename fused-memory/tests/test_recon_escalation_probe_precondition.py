"""Prompt-contract tests for the Stage-1 escalation-probe precondition (task 3052).

Deliverable (3) of task 3052.  Stage 1 has repeatedly filed
"no escalation was filed for X" / channel-dead / zero-yield findings against
conditions it is *structurally* unable to observe: all four escalation READ
tools are denied to it via ``cli_stage_runner.STAGE1_DISALLOWED`` ->
``DISALLOW_ESCALATION_READS``.  The precondition section states the prohibition,
records the facts that make the historical findings wrong even for a reader who
*does* hold the tools, and keeps the one clause Stage 1 can actually execute
(the ``metadata.gate_escalated_at`` age check).

Following the prompt-contract convention of ``test_recon_gate_closure_guidance``
/ ``test_standing_decision_prompt_drift``: pin STRUCTURE, never prose.  Three
structural facts are asserted and nothing else —

(a) the section heading is present (one exact string, pinned as a module
    constant so the pin is one line rather than a prose dump);
(b) THE INVARIANT THAT MATTERS — the prompt never *licenses* a tool Stage 1 is
    denied.  Derived from the live ``DISALLOW_ESCALATION_READS`` constant so a
    fifth denied read tool added later cannot silently slip past, and checked in
    BOTH the fully-qualified (``mcp__escalation__get_escalation``) and bare
    (``get_escalation``) spellings, since prose would naturally use the bare
    one;
(c) the executable clause survives — the section names ``gate_escalated_at`` and
    the stall threshold, with the threshold DERIVED from
    ``stage1_stall_detector.STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS`` rather
    than hardcoded, so a future threshold change breaks this test instead of
    leaving stale prompt text.

Deliberately NOT tested: the section's prose wording beyond those three facts.

DEVIATION FROM THE PLAN'S LITERAL TEXT, recorded here so a reader diffing
plan.json against this file does not read it as an oversight.  Plan step-11
specifies for (b) that *any* occurrence of a denied name "must be within the
precondition section AND on a line that also carries the denial marker".  The
containment half is unsatisfiable against the assembled prompt and always was:
``prompts/__init__.py::_GATE_CLOSURE_ARCHIVE_GUIDANCE`` already names
``get_task_escalations`` — correctly, as a stated reason, immediately followed
by "which is not part of your tool surface" — and that block is byte-pinned by
``tests/test_recon_gate_closure_guidance.py``, so it can neither move into this
section nor be reworded.  The step's own stated rationale ("the names can appear
as the stated REASON but never as an imperative") is fully preserved by the
denial-marker half applied GLOBALLY, which is strictly stronger than applying it
to one section: every occurrence anywhere in the prompt must sit on a denying
line.  The section-containment fact is kept in the form that is true and still
load-bearing — this section names all four denied tools, each on a denying line
(``TestPreconditionSectionNamesTheDeniedTools``).  Filed as an escalate_info on
task 3052.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_ESCALATION_READS,
    STAGE1_DISALLOWED,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_stall_detector import (
    STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
)

#: The exact heading of the new section.  One line, so the pin is a pin and not
#: a copy of the prose underneath it.
PRECONDITION_HEADING = '## Escalation-Probe Precondition (task 3052)'

#: Phrases that mark a line as DENYING the tool it names rather than inviting a
#: call to it.  Every spelling the assembled Stage-1 prompt actually uses is
#: enumerated here; a line naming a denied tool must carry at least one.
DENIAL_MARKERS = (
    'DENIED',
    'denied',
    'not available to you',
    'not part of your tool surface',
)

#: MCP prefix stripped to obtain the bare spelling prose would naturally use.
_MCP_ESCALATION_PREFIX = 'mcp__escalation__'


def _denied_tool_spellings() -> list[str]:
    """Both spellings of every live member of ``DISALLOW_ESCALATION_READS``.

    Derived from the constant, never enumerated by hand: a fifth denied read
    tool added later is picked up automatically instead of slipping past.
    """
    spellings: list[str] = []
    for qualified in DISALLOW_ESCALATION_READS:
        spellings.append(qualified)
        spellings.append(qualified.removeprefix(_MCP_ESCALATION_PREFIX))
    return spellings


def _section(prompt: str, heading: str) -> str:
    """Text of ``heading``'s section, terminated by the next ``## `` heading."""
    marker = f'\n{heading}\n'
    assert marker in prompt, f'section heading not found: {heading!r}'
    rest = prompt.split(marker, 1)[1]
    body: list[str] = []
    for line in rest.split('\n'):
        if line.startswith('## '):
            break
        body.append(line)
    return '\n'.join(body)


def _lines_naming(prompt: str, needle: str) -> list[str]:
    return [line for line in prompt.split('\n') if needle in line]


class TestPreconditionSectionPresent:
    """(a) The section exists, under its exact heading."""

    def test_heading_present(self):
        assert PRECONDITION_HEADING in STAGE1_SYSTEM_PROMPT

    def test_section_is_non_empty(self):
        assert _section(STAGE1_SYSTEM_PROMPT, PRECONDITION_HEADING).strip()


class TestPromptNeverLicensesADeniedTool:
    """(b) The invariant that matters, derived from the live disallow list.

    A denied tool's name may appear as the stated REASON Stage 1 cannot answer a
    question; it may never appear as an imperative.  Operationalised as: every
    line naming one — in either spelling, anywhere in the assembled prompt —
    also carries a denial marker.
    """

    def test_disallow_list_is_folded_into_stage1(self):
        # The precondition's premise: these really are denied to Stage 1.
        for qualified in DISALLOW_ESCALATION_READS:
            assert qualified in STAGE1_DISALLOWED

    @pytest.mark.parametrize('spelling', _denied_tool_spellings())
    def test_every_occurrence_sits_on_a_denying_line(self, spelling: str):
        for line in _lines_naming(STAGE1_SYSTEM_PROMPT, spelling):
            assert any(marker in line for marker in DENIAL_MARKERS), (
                f'{spelling!r} named on a line carrying no denial marker: {line!r}'
            )


class TestPreconditionSectionNamesTheDeniedTools:
    """(b, section half) The section states the reason, naming all four."""

    @pytest.mark.parametrize(
        'bare',
        [q.removeprefix(_MCP_ESCALATION_PREFIX) for q in DISALLOW_ESCALATION_READS],
    )
    def test_bare_name_present_in_section(self, bare: str):
        assert bare in _section(STAGE1_SYSTEM_PROMPT, PRECONDITION_HEADING)

    @pytest.mark.parametrize('spelling', _denied_tool_spellings())
    def test_section_occurrences_are_denying(self, spelling: str):
        section = _section(STAGE1_SYSTEM_PROMPT, PRECONDITION_HEADING)
        for line in _lines_naming(section, spelling):
            assert any(marker in line for marker in DENIAL_MARKERS), (
                f'{spelling!r} named non-denyingly inside the precondition '
                f'section: {line!r}'
            )


class TestExecutableClauseSurvives:
    """(c) The one clause Stage 1 CAN execute, with a derived threshold."""

    def test_names_the_gate_stamp(self):
        assert (
            'gate_escalated_at'
            in _section(STAGE1_SYSTEM_PROMPT, PRECONDITION_HEADING)
        )

    def test_threshold_is_interpolated_not_hardcoded(self):
        hours = int(STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS // 3600)
        assert f'{hours}h' in _section(
            STAGE1_SYSTEM_PROMPT, PRECONDITION_HEADING
        )
