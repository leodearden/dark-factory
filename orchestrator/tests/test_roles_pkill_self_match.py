"""Anchor test for `PKILL_SELF_MATCH_GUIDANCE` (task 5961).

Held to the preamble-tail contract in `_role_splice_contract.py`; only this
block's own remedy prose, and its one extra order pin, live here. Every
assertion is an existence, containment, count or index check against a NAMED
constant — never a prose pin, never a regex over wording.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.pkill_guidance import PKILL_SELF_MATCH_GUIDANCE
from orchestrator.agents.roles import BACKGROUND_WAIT_GUIDANCE


class TestPkillSelfMatchSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='PKILL_SELF_MATCH_GUIDANCE',
        constant=PKILL_SELF_MATCH_GUIDANCE,
        restore_remedy=(
            'Restore the task-5961 guidance: without it every Bash-capable role '
            'rediscovers the `pkill -f` self-kill by failure.'
        ),
        brace_remedy='Spell the example without braces; `$(...)` needs none.',
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: their `Bash(git:*)` grants refuse `pkill`, so the '
            'block is dead weight there.'
        ),
        carrier_note=(
            'A role that can run `pkill -f` needs the guidance, and one that '
            'cannot does not.'
        ),
    )

    def test_wait_section_pointers_cannot_dangle(self) -> None:
        """The block's "section above" pointers resolve to `BACKGROUND_WAIT_GUIDANCE`.

        The prose points at the exit-code section and at the wait section's
        termination tool, both members of that block. The structural twin of
        `orchestrator/tests/test_roles_wait_pattern.py::test_amender_reminder_cannot_dangle`.
        """
        self.block.contract.assert_lands_after(
            follows=BACKGROUND_WAIT_GUIDANCE,
            follows_name='BACKGROUND_WAIT_GUIDANCE',
            remedy=(
                'Its pointers to the exit-code and wait sections "above" would '
                'dangle. Keep both in the preamble, the wait block first.'
            ),
        )
