"""Anchor test for `CHAINED_COMMAND_STATUS_GUIDANCE` (task 5962).

Held to the preamble-tail contract in `_role_splice_contract.py`; only this
block's own remedy prose lives here. Every assertion is an existence,
containment, count or index check against a NAMED constant — never a prose
pin, never a regex over wording, never a byte-size figure.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.chained_command_guidance import CHAINED_COMMAND_STATUS_GUIDANCE


class TestChainedCommandStatusSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='CHAINED_COMMAND_STATUS_GUIDANCE',
        constant=CHAINED_COMMAND_STATUS_GUIDANCE,
        restore_remedy=(
            'Restore the task-5962 guidance: without it every Bash-capable role '
            'misreads a chained call\'s single exit status.'
        ),
        brace_remedy='Do not spell a status example with `${...}`; `$?` needs no braces.',
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: their `Bash(git:*)` grants cannot chain arbitrary '
            'steps, so the block is dead weight there.'
        ),
        carrier_note=(
            'A role that can chain steps in one `Bash` call needs the guidance, '
            'and one that cannot does not.'
        ),
    )
