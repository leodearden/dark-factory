"""Anchor/contract test for `GREP_PATTERN_ESCAPING_GUIDANCE` (task 5963).

A sibling of `test_roles_chained_command_status.py`. Every assertion here is an
existence, containment, count or index check against a NAMED constant — never a
prose pin, never a regex over wording, never a byte-size figure. The mechanical
half of that shape lives in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.grep_pattern_guidance import GREP_PATTERN_ESCAPING_GUIDANCE
from orchestrator.agents.roles import GREP_LOOKAROUND_GUIDANCE

#: Roles holding UNQUALIFIED `Bash` with a literal system_prompt — the only ones
#: that write `grep`/`rg` patterns through a shell. `judge` is absent because its
#: grant is `Bash(git:*)` and its prompt restricts it to Read/Glob/Grep;
#: `reviewer_comprehensive` is absent because it is PromptSpec-backed and also
#: holds only `Bash(git:*)`.
_BASH_CAPABLE_UNPINNED_ROLES = frozenset({
    'architect',
    'debugger',
    'deep_reviewer',
    'implementer',
    'merger',
    'simple_task',
    'steward',
})

_CONTRACT = SpliceContract(
    constant_name='GREP_PATTERN_ESCAPING_GUIDANCE',
    constant=GREP_PATTERN_ESCAPING_GUIDANCE,
    roles=_BASH_CAPABLE_UNPINNED_ROLES,
    role_set_name='_BASH_CAPABLE_UNPINNED_ROLES',
    capability=lambda role: role.prompt_spec is None and 'Bash' in role.allowed_tools,
    capability_description='a literal system_prompt and unqualified `Bash`',
)


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


def test_role_set_matches_bash_capability():
    """Premise tripwire: the hand-maintained set equals the derived one."""
    _CONTRACT.assert_role_set_matches_capability(
        remedy=(
            "A role's `Bash` grant changed, or a role was added. Update "
            '_BASH_CAPABLE_UNPINNED_ROLES to match: a role that writes search '
            'patterns through a shell needs the guidance, and one that cannot '
            'does not.'
        ),
    )


def test_every_bash_capable_role_carries_guidance():
    """Every role that writes search patterns through a shell is told how both layers parse them."""
    _CONTRACT.assert_every_role_carries(
        remedy=(
            'Append GREP_PATTERN_ESCAPING_GUIDANCE to the tail of '
            'roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; that one composite reaches '
            'every Bash-capable role.'
        ),
    )


def test_no_other_role_carries_guidance():
    """The negative half: no role outside the set carries the block."""
    _CONTRACT.assert_no_other_role_carries(
        remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: their `Bash(git:*)` grants never pass a search '
            'pattern through a shell, so the block is dead weight there.'
        ),
    )


def test_guidance_appears_exactly_once_per_role():
    """No stale duplicate splice survives beside a new one.

    ``absent_ok=True`` leaves a missing splice to fail only
    `test_every_bash_capable_role_carries_guidance`: one root cause, one
    failing test.
    """
    _CONTRACT.assert_spliced_exactly_once(
        absent_ok=True,
        remedy='Keep exactly one GREP_PATTERN_ESCAPING_GUIDANCE term in the preamble.',
    )


def test_guidance_lands_after_the_grep_block():
    """The block lands anywhere after `GREP_LOOKAROUND_GUIDANCE` ends.

    (a) The order is load-bearing: the block's prose points at "the look-around
    rejection earlier in this prompt", which resolves only while
    `GREP_LOOKAROUND_GUIDANCE` precedes it.

    (b) Adjacency is deliberately NOT pinned (esc-5966-3): sibling census blocks
    append to the same tail of `_BASH_CAPABLE_ROLE_PREAMBLE` in any merge order.
    """
    _CONTRACT.assert_lands_after(
        follows=GREP_LOOKAROUND_GUIDANCE,
        follows_name='GREP_LOOKAROUND_GUIDANCE',
        remedy=(
            'Re-append it at the END of roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; '
            'moving it earlier breaks the adjacency pins of the blocks ahead of '
            'it and dangles its pointer to the look-around section.'
        ),
    )
