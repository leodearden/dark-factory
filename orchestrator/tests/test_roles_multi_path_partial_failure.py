"""Anchor test for `MULTI_PATH_PARTIAL_FAILURE_GUIDANCE` (task 5966).

Held to the preamble-tail contract in `_role_splice_contract.py`; only this
block's own remedy prose lives here. Every assertion is an existence,
containment, count or index check against a NAMED constant — never a prose
pin, never a regex over wording, never a byte-size figure.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.partial_failure_guidance import MULTI_PATH_PARTIAL_FAILURE_GUIDANCE


class TestMultiPathPartialFailureSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='MULTI_PATH_PARTIAL_FAILURE_GUIDANCE',
        constant=MULTI_PATH_PARTIAL_FAILURE_GUIDANCE,
        restore_remedy=(
            'Restore the task-5966 guidance: without it every Bash-capable role '
            'reads past the one error line in a multi-path wc, grep or ls and '
            'trusts a partial total.'
        ),
        brace_remedy='Describe any brace form in words; never spell it out.',
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: a `Bash(git:*)` grant cannot run wc, cat, grep '
            'or ls, so the block is dead weight there.'
        ),
    )
