"""Anchor test for `BASH_CWD_ANCHOR_GUIDANCE` (task 5971).

Provenance: reify #7924, refiled cross-repo as task 5971; the same mechanism as
dark_factory codebook entry-cand-20260729-4. Held to the preamble-tail contract
in `_role_splice_contract.py`; only this block's own remedy prose lives here.
Every assertion is an existence, containment, count or index check against a
NAMED constant — never a prose pin, never a regex over wording, never a
byte-size figure.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.bash_cwd_guidance import BASH_CWD_ANCHOR_GUIDANCE


class TestBashCwdAnchorSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='BASH_CWD_ANCHOR_GUIDANCE',
        constant=BASH_CWD_ANCHOR_GUIDANCE,
        restore_remedy=(
            'Restore the task-5971 guidance: without it a Bash-capable role whose '
            'shell kept an earlier `cd` keeps guessing repo-root-relative paths '
            'that come back not-found.'
        ),
        brace_remedy='Describe any brace form in words; never spell it out.',
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            "widening the set. judge's `Bash(git:*)` grant cannot `cd`, so its "
            'cwd cannot drift, and the block would tell it to run `cd` and `pwd` '
            'it is denied. reviewer_comprehensive is PromptSpec-backed, and a '
            'pinned artifact can drop a splice.'
        ),
    )
