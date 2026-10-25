"""Referential tests for the Stage-1 escalation-probe precondition (task 3052).

Deliverable (3) of task 3052.  Stage 1 has repeatedly filed
"no escalation was filed for X" / channel-dead / zero-yield findings against
conditions it is *structurally* unable to observe: all four escalation READ
tools are denied to it via ``cli_stage_runner.STAGE1_DISALLOWED`` ->
``DISALLOW_ESCALATION_READS``.  The precondition section states the
prohibition and keeps the one clause Stage 1 can actually execute (the
``metadata.gate_escalated_at`` age check).

This file asserts exactly TWO facts, both derived from live constants rather
than from the prompt's wording:

(a) ``DISALLOW_ESCALATION_READS`` really is folded into ``STAGE1_DISALLOWED``
    — the precondition's load-bearing premise.  Were it false, the section
    would be telling Stage 1 not to do something it is in fact allowed to do.
(b) The stall threshold in the section is interpolated from
    ``stage1_stall_detector.STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS``, so a
    threshold change cannot leave stale prompt text behind.

The section's PROSE is deliberately not pinned.  Step-11 originally asserted a
containment-and-denial-marker invariant over the assembled prompt; it was
removed on review as a prose pin — a full rewording that changed no identifier
would have failed it, and its hand-maintained substring list passed any line
containing "denied" anywhere, including one that licensed the tool, so it never
enforced the invariant it advertised.  Recorded here so a reader diffing
plan.json against this file does not read the absence as an oversight.
"""

from __future__ import annotations

from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_ESCALATION_READS,
    STAGE1_DISALLOWED,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_stall_detector import (
    STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS,
)

#: LOCATOR, not a contract: the heading ``_section`` slices on so the threshold
#: assertion reads the precondition's own text rather than the whole prompt.  A
#: reword that renames the heading should update this constant — it is not a
#: pin on the wording, and a failure here means "the locator moved", not "the
#: prompt regressed".
PRECONDITION_HEADING = '## Escalation-Probe Precondition (task 3052)'


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


class TestPromptNeverLicensesADeniedTool:
    """(a) The precondition's premise, read off the live constants.

    What survives here is referential, not textual: the section can only be
    right about Stage 1 being unable to probe the escalation queue while the
    escalation read tools really are on its disallow list.
    """

    def test_disallow_list_is_folded_into_stage1(self):
        for qualified in DISALLOW_ESCALATION_READS:
            assert qualified in STAGE1_DISALLOWED


class TestExecutableClauseSurvives:
    """(b) The one clause Stage 1 CAN execute, with a derived threshold."""

    def test_threshold_is_interpolated_not_hardcoded(self):
        hours = int(STAGE1_GATE_BACKLOG_STALL_THRESHOLD_SECS // 3600)
        assert f'{hours}h' in _section(
            STAGE1_SYSTEM_PROMPT, PRECONDITION_HEADING
        )
