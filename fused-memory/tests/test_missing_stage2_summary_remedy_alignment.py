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
