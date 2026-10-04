"""Anchor/contract test for `MULTI_PATH_PARTIAL_FAILURE_GUIDANCE` (task 5966).

A sibling of `test_roles_pasted_text_python_literal.py`. Every assertion here is
an existence, containment, count or index check against a NAMED constant —
never a prose pin, never a regex over wording, never a byte-size figure. The
mechanical half of that shape lives in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.partial_failure_guidance import MULTI_PATH_PARTIAL_FAILURE_GUIDANCE


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'MULTI_PATH_PARTIAL_FAILURE_GUIDANCE',
        MULTI_PATH_PARTIAL_FAILURE_GUIDANCE,
        remedy=(
            'Restore the task-5966 guidance: without it every Bash-capable role '
            'reads past the one error line in a multi-path wc, grep or ls and '
            'trusts a partial total.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'MULTI_PATH_PARTIAL_FAILURE_GUIDANCE',
        MULTI_PATH_PARTIAL_FAILURE_GUIDANCE,
        remedy='Describe any brace form in words; never spell it out.',
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert MULTI_PATH_PARTIAL_FAILURE_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'MULTI_PATH_PARTIAL_FAILURE_GUIDANCE does not start with MARKDOWN_HEADING, '
        'so at the tail of _BASH_CAPABLE_ROLE_PREAMBLE it would read as an '
        'unheaded continuation of whichever block precedes it. Give it a '
        'leading blank line and a `## ` heading.'
    )
