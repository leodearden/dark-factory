"""Anchor test for `PIPELINE_EXIT_STATUS_GUIDANCE` (tasks 5965 and 5915).

The block is ONE section composed of two named halves, the masked-status rule
and the pipefail-141 rule. It is held to the preamble-tail contract in
`_role_splice_contract.py`; only this block's own remedy prose and its
composition checks live here. Every assertion is an existence, containment,
count or index check against a NAMED constant: never a prose pin, never a
regex over wording, never a byte-size figure.
"""

from __future__ import annotations

import pytest
from _role_splice_contract import MARKDOWN_HEADING, PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.pipeline_status_guidance import (
    PIPELINE_EXIT_STATUS_GUIDANCE,
    PIPELINE_LAST_STAGE_STATUS,
    SIGPIPE_UNDER_PIPEFAIL,
)

_HALVES = [
    ('PIPELINE_LAST_STAGE_STATUS', PIPELINE_LAST_STAGE_STATUS),
    ('SIGPIPE_UNDER_PIPEFAIL', SIGPIPE_UNDER_PIPEFAIL),
]


class TestPipelineExitStatusSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='PIPELINE_EXIT_STATUS_GUIDANCE',
        constant=PIPELINE_EXIT_STATUS_GUIDANCE,
        restore_remedy=(
            'Restore the pipeline exit-status block: without it a Bash-capable '
            'role reads `$?` after `cmd | head` as cmd\'s status, or reads the '
            'pipefail 141 that `head` or `grep -q` hands back as a failure or a '
            '"no match".'
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

    def test_both_halves_are_composed_into_the_block(self) -> None:
        """Each half's presence is falsifiable without a prose pin."""
        self.block.contract.assert_composes(
            _HALVES,
            remedy=(
                'Dropping PIPELINE_LAST_STAGE_STATUS loses the masked-status rule '
                '(task 5915): a role reads `$?` after `cmd | head` as cmd\'s own '
                'status. Dropping SIGPIPE_UNDER_PIPEFAIL loses the pipefail-141 '
                'rule (task 5965): a role reads an early-closing reader\'s 141 as a '
                'failure or a "no match".'
            ),
        )

    @pytest.mark.parametrize(('name', 'half'), _HALVES)
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
                'A count of 0 means the composed PIPELINE_EXIT_STATUS_GUIDANCE '
                'splice is missing from that role; a count of 2 means a stale '
                f'bare `+ {name}` term, for example one re-added by a merge '
                'resolution, survives beside it. Delete the bare term.'
            ),
        )

    def test_block_opens_exactly_one_section(self) -> None:
        """The two halves read as ONE section (esc-5965-3), not two blocks."""
        sections = PIPELINE_EXIT_STATUS_GUIDANCE.count(MARKDOWN_HEADING)

        assert sections == 1, (
            f'PIPELINE_EXIT_STATUS_GUIDANCE opens {sections} `##` sections; it '
            'must be ONE pipeline exit-status block (esc-5965-3). Only '
            'PIPELINE_LAST_STAGE_STATUS opens the section; SIGPIPE_UNDER_PIPEFAIL '
            'continues it with a lead paragraph and must not reopen one.'
        )
