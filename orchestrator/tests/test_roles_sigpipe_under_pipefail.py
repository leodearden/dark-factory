"""Anchor test for `SIGPIPE_UNDER_PIPEFAIL_GUIDANCE` (task 5965).

Held to the preamble-tail contract in `_role_splice_contract.py`; only this
block's own remedy prose lives here. Every assertion is an existence,
containment, count or index check against a NAMED constant — never a prose
pin, never a regex over wording, never a byte-size figure.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.sigpipe_guidance import SIGPIPE_UNDER_PIPEFAIL_GUIDANCE


class TestSigpipeUnderPipefailSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='SIGPIPE_UNDER_PIPEFAIL_GUIDANCE',
        constant=SIGPIPE_UNDER_PIPEFAIL_GUIDANCE,
        restore_remedy=(
            'Restore the task-5965 guidance: without it a Bash-capable role '
            'reads the pipefail 141 that `head` or `grep -q` hands back as a '
            'failure or a "no match".'
        ),
        brace_remedy=(
            'Name the PIPESTATUS array in prose; never spell a subscript, '
            'which needs braces.'
        ),
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: a `Bash(git:*)` grant cannot write or run a '
            'pipefail script, so the block is dead weight there.'
        ),
        carrier_note=(
            'architect and deep_reviewer belong in it although they lack `Edit`: '
            'they run scripts and read pipeline statuses.'
        ),
    )
