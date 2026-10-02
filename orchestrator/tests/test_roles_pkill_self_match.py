"""Anchor/contract test for `PKILL_SELF_MATCH_GUIDANCE` (task 5961).

A sibling of `test_roles_grep_lookaround.py`, and the cross-repo refile of reify
legibility-census candidate #7922. The finding, the measurements behind it and
the prose constraints are recorded once, in the docstring of
`orchestrator/src/orchestrator/agents/pkill_guidance.py`; this module points
there rather than restating them.

Every assertion here is an existence, containment, count or index check
against a NAMED constant — never a prose pin, never a regex over wording. The
mechanical half of that shape lives in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import MARKDOWN_HEADING, assert_brace_free, assert_nonempty

from orchestrator.agents.pkill_guidance import PKILL_SELF_MATCH_GUIDANCE


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'PKILL_SELF_MATCH_GUIDANCE',
        PKILL_SELF_MATCH_GUIDANCE,
        remedy=(
            'Restore the task-5961 guidance: without it every Bash-capable role '
            'rediscovers the `pkill -f` self-kill by failure.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'PKILL_SELF_MATCH_GUIDANCE',
        PKILL_SELF_MATCH_GUIDANCE,
        remedy='Spell the example without braces; `$(...)` needs none.',
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert PKILL_SELF_MATCH_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'PKILL_SELF_MATCH_GUIDANCE does not start with MARKDOWN_HEADING, so at '
        'the tail of _BASH_CAPABLE_ROLE_PREAMBLE it would read as an unheaded '
        'continuation of whichever section precedes it. Give it a leading blank '
        'line and a `## ` heading.'
    )
