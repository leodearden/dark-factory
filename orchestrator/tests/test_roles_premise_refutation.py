"""Anchor test for `PREMISE_REFUTATION_GUIDANCE` (task 5976).

Held to the preamble-tail contract in `_role_splice_contract.py`; only this
block's own remedy prose lives here. Every assertion is an existence,
containment, count or index check against a NAMED constant — never a prose
pin, never a regex over wording.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.premise_refutation_guidance import PREMISE_REFUTATION_GUIDANCE


class TestPremiseRefutationSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='PREMISE_REFUTATION_GUIDANCE',
        constant=PREMISE_REFUTATION_GUIDANCE,
        restore_remedy=(
            'Restore the task-5976 guidance: without it a role reads its own '
            'non-reproduction of a claim, run in the wrong execution context, '
            'as a refutation.'
        ),
        brace_remedy=(
            'Write placeholders as <context>, never with braces, so the block '
            'survives an interpolating splice site.'
        ),
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: a `Bash(git:*)` grant cannot run a check in a '
            "claim's execution context, so the block is dead weight there."
        ),
        carrier_note=(
            'Any role that can run a live check can misread its non-reproduction '
            'as a refutation.'
        ),
    )
