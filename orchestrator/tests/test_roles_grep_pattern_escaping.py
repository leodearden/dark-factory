"""Anchor test for `GREP_PATTERN_ESCAPING_GUIDANCE` (task 5963).

Held to the preamble-tail contract in `_role_splice_contract.py`; only this
block's own remedy prose lives here. Every assertion is an existence,
containment, count or index check against a NAMED constant — never a prose
pin, never a regex over wording, never a byte-size figure.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.grep_pattern_guidance import GREP_PATTERN_ESCAPING_GUIDANCE


class TestGrepPatternEscapingSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='GREP_PATTERN_ESCAPING_GUIDANCE',
        constant=GREP_PATTERN_ESCAPING_GUIDANCE,
        restore_remedy=(
            'Restore the task-5963 guidance: without it every Bash-capable role '
            'misreads an eval parse error or an unclosed-group rejection as a '
            'no-match or a missing file.'
        ),
        brace_remedy=(
            'Do not list braces among the regex metacharacters, and do not spell '
            'a shell expansion as `${...}`.'
        ),
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: their `Bash(git:*)` grants never pass a search '
            'pattern through a shell, so the block is dead weight there.'
        ),
        carrier_note=(
            'A role that writes search patterns through a shell needs the '
            'guidance, and one that cannot does not.'
        ),
        order_note=(
            'The order is load-bearing too: its prose points at "the look-around '
            'rejection earlier in this prompt", so moving it earlier also dangles '
            'that pointer.'
        ),
    )
