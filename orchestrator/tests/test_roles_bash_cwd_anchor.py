"""Anchor test for `BASH_CWD_ANCHOR_GUIDANCE` (tasks 5971 and 11554).

Provenance: reify #7924, refiled cross-repo as task 5971; the same mechanism as
dark_factory codebook entry-cand-20260729-4; dark_factory codebook
entry-cand-20261118-2 (task 11554). The block is ONE section composed of two
named halves, the lingering-`cd` rule and the deleted-cwd rule. It is held to
the preamble-tail contract in `_role_splice_contract.py`; only this block's own
remedy prose and its composition checks live here. Every assertion is an
existence, containment, count or index check against a NAMED constant — never
a prose pin, never a regex over wording, never a byte-size figure.
"""

from __future__ import annotations

import pytest
from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.bash_cwd_guidance import (
    BASH_CWD_ANCHOR_GUIDANCE,
    DELETED_CWD_GUIDANCE,
    LINGERING_CD_GUIDANCE,
)

_HALVES = [
    ('LINGERING_CD_GUIDANCE', LINGERING_CD_GUIDANCE),
    ('DELETED_CWD_GUIDANCE', DELETED_CWD_GUIDANCE),
]


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

    def test_both_halves_are_composed_into_the_block(self) -> None:
        """Each half's presence is falsifiable without a prose pin."""
        self.block.contract.assert_composes(
            _HALVES,
            remedy=(
                'Dropping LINGERING_CD_GUIDANCE loses the lingering-`cd` rule '
                '(task 5971): a role whose shell kept an earlier `cd` keeps '
                'guessing repo-root-relative paths. Dropping DELETED_CWD_GUIDANCE '
                'loses the deleted-cwd rule (task 11554): a role that removes the '
                'scratch directory its shell stands in reads the resulting '
                '"Exit code 1" as a failure, or has its next call refused unrun.'
            ),
        )

    @pytest.mark.parametrize(('name', 'half'), _HALVES, ids=[name for name, _ in _HALVES])
    def test_each_half_is_spliced_exactly_once(self, name: str, half: str) -> None:
        """A 0 count means the composite is missing; a 2 means a stale bare half survives.

        ``absent_ok=False`` is safe only because the composition is pinned by
        `test_both_halves_are_composed_into_the_block`, which makes a zero
        count diagnostic rather than a duplicate report of a missing splice.
        """
        self.block.contract.assert_spliced_exactly_once(
            constant=half,
            constant_name=name,
            absent_ok=False,
            remedy=(
                'A count of 0 means the composed BASH_CWD_ANCHOR_GUIDANCE splice '
                'is missing from that role; a count of 2 means a stale bare '
                f'`+ {name}` term, for example one re-added by a merge '
                'resolution, survives beside it. Delete the bare term.'
            ),
        )
