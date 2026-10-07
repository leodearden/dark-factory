"""Anchor/contract test for `PREMISE_REFUTATION_GUIDANCE` (task 5976).

A sibling of `test_roles_sigpipe_under_pipefail.py`. Every assertion here is
an existence, containment, count or index check against a NAMED constant — never
a prose pin, never a regex over wording. The mechanical half of that shape lives
in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.premise_refutation_guidance import PREMISE_REFUTATION_GUIDANCE


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'PREMISE_REFUTATION_GUIDANCE',
        PREMISE_REFUTATION_GUIDANCE,
        remedy=(
            'Restore the task-5976 guidance: without it a role reads its own '
            'non-reproduction of a claim, run in the wrong execution context, '
            'as a refutation.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'PREMISE_REFUTATION_GUIDANCE',
        PREMISE_REFUTATION_GUIDANCE,
        remedy=(
            'Write placeholders as <context>, never with braces, so the block '
            'survives an interpolating splice site.'
        ),
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert PREMISE_REFUTATION_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'PREMISE_REFUTATION_GUIDANCE does not start with MARKDOWN_HEADING, '
        'so at the tail of _BASH_CAPABLE_ROLE_PREAMBLE it would read as an '
        'unheaded continuation of whichever block precedes it. Give it a '
        'leading blank line and a `## ` heading.'
    )
