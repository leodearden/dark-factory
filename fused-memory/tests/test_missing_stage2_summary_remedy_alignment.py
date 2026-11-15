"""Stage 1 and Stage 3 no longer name reconstruction as the remedy for a
ledger-missing Stage 2 summary (task 6295, follow-up to task 3734).

Only the harness's ledger write path (``summary_pool.write_cycle_summary``) can
create that row, so a remedy telling an agent to reconstruct it names a repair
nobody can perform.  These tests pin interface tokens and wiring, never
sentences, per the house norm stated in
``test_stage3_cross_project_routing_actionable.py`` and
``test_stage2_narrative_reconstruction_retired.py``.
"""

from __future__ import annotations

import re

from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT


def _sections(prompt: str) -> list[str]:
    """The ``## `` sections of a rendered prompt; each one's first line is its heading."""
    return prompt.split('\n## ')[1:]


def _section(prompt: str, heading: str) -> str:
    matches = [section for section in _sections(prompt) if section.splitlines()[0] == heading]
    assert len(matches) == 1, (
        f'Expected exactly one "## {heading}" section, found {len(matches)}.'
    )
    return matches[0]


class TestStage3MissingSummaryRemedy:
    @staticmethod
    def _missing_branch() -> str:
        bullets = _section(STAGE3_SYSTEM_PROMPT, 'Cycle-Summary Verification').split('\n- ')
        missing = [bullet for bullet in bullets if "`reason: 'missing'`" in bullet]
        assert len(missing) == 1, (
            "Expected exactly one reason='missing' bullet in Stage 3's "
            f'## Cycle-Summary Verification, found {len(missing)}; the pins '
            'below would otherwise pass against nothing.'
        )
        return missing[0]

    def test_missing_branch_keeps_actionable_routing(self):
        bullet = self._missing_branch()

        assert "category='missing_knowledge'" in bullet and 'actionable=true' in bullet, (
            "The reason='missing' finding must stay category='missing_knowledge', "
            'actionable=true: harness.py::_finding_is_live_actionable routes only '
            'actionable findings to a remediation pass, so a recurring ledger-write '
            f'gap would otherwise go silent. Bullet: {bullet}'
        )

    def test_missing_branch_names_a_remedy(self):
        bullet = self._missing_branch()

        assert "suggested_action='" in bullet, (
            "The reason='missing' bullet must still name a suggested_action; the "
            f'retired remedy is replaced, not dropped. Bullet: {bullet}'
        )

    def test_retired_reconstruct_remedy_is_gone(self):
        assert "suggested_action='reconstruct'" not in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT still names suggested_action='reconstruct'. No "
            'remediation-pass agent can re-run write_cycle_summary for a prior run, '
            'so the value names a repair nobody can perform (task 6295).'
        )


class TestStage1MissingSummaryRemedy:
    def test_the_stage2_summary_decision_section_exists(self):
        """Selected by content, not heading, because the heading is what was renamed."""
        decision_sections = [
            section for section in _sections(STAGE1_SYSTEM_PROMPT)
            if "stage='task_knowledge_sync'" in section
        ]

        assert len(decision_sections) == 1, (
            "Expected exactly one STAGE1_SYSTEM_PROMPT section keying the ledger "
            "pre-check to stage='task_knowledge_sync', found "
            f'{len(decision_sections)}; the absence pins below would otherwise '
            'pass against a prompt that lost the check.'
        )

    def test_retired_reconstruct_framing_is_gone(self):
        for retired in ('reconstruct-trigger', 'Already-Reconstructed'):
            assert retired not in STAGE1_SYSTEM_PROMPT, (
                f'STAGE1_SYSTEM_PROMPT still carries {retired!r}, the retired framing '
                'of the missing Stage 2 summary as something to reconstruct '
                '(task 6295).'
            )

    def test_heading_cross_references_resolve(self):
        referenced = re.findall(r'see ## ([^\n]+?) below', STAGE1_SYSTEM_PROMPT)
        headings = {section.splitlines()[0] for section in _sections(STAGE1_SYSTEM_PROMPT)}

        assert referenced, 'STAGE1_SYSTEM_PROMPT has no "see ## <heading> below" pointer.'
        for heading in referenced:
            assert heading in headings, (
                f'STAGE1_SYSTEM_PROMPT points the agent at "## {heading}", '
                'which no section carries.'
            )
