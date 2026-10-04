"""Anchor/contract test for `SIGPIPE_UNDER_PIPEFAIL_GUIDANCE` (task 5965).

A sibling of `test_roles_pasted_text_python_literal.py`. Every assertion here is
an existence, containment, count or index check against a NAMED constant — never
a prose pin, never a regex over wording, never a byte-size figure. The
mechanical half of that shape lives in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.sigpipe_guidance import SIGPIPE_UNDER_PIPEFAIL_GUIDANCE


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'SIGPIPE_UNDER_PIPEFAIL_GUIDANCE',
        SIGPIPE_UNDER_PIPEFAIL_GUIDANCE,
        remedy=(
            'Restore the task-5965 guidance: without it a Bash-capable role '
            'reads the pipefail 141 that `head` or `grep -q` hands back as a '
            'failure or a "no match".'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'SIGPIPE_UNDER_PIPEFAIL_GUIDANCE',
        SIGPIPE_UNDER_PIPEFAIL_GUIDANCE,
        remedy=(
            'Name the PIPESTATUS array in prose; never spell a subscript, '
            'which needs braces.'
        ),
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert SIGPIPE_UNDER_PIPEFAIL_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'SIGPIPE_UNDER_PIPEFAIL_GUIDANCE does not start with MARKDOWN_HEADING, '
        'so at the tail of _BASH_CAPABLE_ROLE_PREAMBLE it would read as an '
        'unheaded continuation of whichever block precedes it. Give it a '
        'leading blank line and a `## ` heading.'
    )
