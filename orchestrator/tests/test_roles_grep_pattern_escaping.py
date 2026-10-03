"""Anchor/contract test for `GREP_PATTERN_ESCAPING_GUIDANCE` (task 5963).

A sibling of `test_roles_chained_command_status.py`. Every assertion here is an
existence, containment, count or index check against a NAMED constant — never a
prose pin, never a regex over wording, never a byte-size figure. The mechanical
half of that shape lives in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.grep_pattern_guidance import GREP_PATTERN_ESCAPING_GUIDANCE


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'GREP_PATTERN_ESCAPING_GUIDANCE',
        GREP_PATTERN_ESCAPING_GUIDANCE,
        remedy=(
            'Restore the task-5963 guidance: without it every Bash-capable role '
            'misreads an eval parse error or an unclosed-group rejection as a '
            'no-match or a missing file.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'GREP_PATTERN_ESCAPING_GUIDANCE',
        GREP_PATTERN_ESCAPING_GUIDANCE,
        remedy=(
            'Do not list braces among the regex metacharacters, and do not spell '
            'a shell expansion as `${...}`.'
        ),
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert GREP_PATTERN_ESCAPING_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'GREP_PATTERN_ESCAPING_GUIDANCE does not start with MARKDOWN_HEADING, so '
        'at the tail of _BASH_CAPABLE_ROLE_PREAMBLE it would read as an unheaded '
        'continuation of whichever block precedes it. Give it a leading blank '
        'line and a `## ` heading.'
    )
