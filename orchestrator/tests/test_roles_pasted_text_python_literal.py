"""Anchor/contract test for `PASTED_TEXT_PYTHON_LITERAL_GUIDANCE` (task 5964).

A cross-repo refile of reify legibility-census candidate #7910, and a sibling of
`test_roles_grep_pattern_escaping.py`. Every assertion here is an existence,
containment, count or index check against a NAMED constant — never a prose pin,
never a regex over wording, never a byte-size figure. The mechanical half of
that shape lives in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.python_literal_guidance import PASTED_TEXT_PYTHON_LITERAL_GUIDANCE


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'PASTED_TEXT_PYTHON_LITERAL_GUIDANCE',
        PASTED_TEXT_PYTHON_LITERAL_GUIDANCE,
        remedy=(
            'Restore the task-5964 guidance: without it every Bash-capable role '
            'pastes foreign source into a plain Python literal and either loses '
            'a turn to a unicodeescape SyntaxError or silently edits with '
            'decoded escapes.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'PASTED_TEXT_PYTHON_LITERAL_GUIDANCE',
        PASTED_TEXT_PYTHON_LITERAL_GUIDANCE,
        remedy=(
            'Describe the brace-form `\\u` escape of Rust and JavaScript in '
            'words; never spell it out.'
        ),
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert PASTED_TEXT_PYTHON_LITERAL_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'PASTED_TEXT_PYTHON_LITERAL_GUIDANCE does not start with MARKDOWN_HEADING, '
        'so at the tail of _BASH_CAPABLE_ROLE_PREAMBLE it would read as an '
        'unheaded continuation of whichever block precedes it. Give it a '
        'leading blank line and a `## ` heading.'
    )
