"""Anchor/contract test for `BASH_CWD_ANCHOR_GUIDANCE` (task 5971).

Provenance: reify #7924, refiled cross-repo as task 5971; the same mechanism as
dark_factory codebook entry-cand-20260729-4. A sibling of
`test_roles_multi_path_partial_failure.py`. Every assertion here is an
existence, containment, count or index check against a NAMED constant — never
a prose pin, never a regex over wording, never a byte-size figure. The
mechanical half of that shape lives in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.bash_cwd_guidance import BASH_CWD_ANCHOR_GUIDANCE
from orchestrator.agents.roles import GREP_LOOKAROUND_GUIDANCE

#: Roles holding UNQUALIFIED `Bash` with a literal system_prompt — the only ones
#: whose shell can `cd` and so drift. `judge` is absent because its grant is
#: `Bash(git:*)`, which cannot `cd`; `reviewer_comprehensive` is absent because
#: it is PromptSpec-backed and also holds only `Bash(git:*)`.
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
    constant_name='BASH_CWD_ANCHOR_GUIDANCE',
    constant=BASH_CWD_ANCHOR_GUIDANCE,
    roles=_BASH_CAPABLE_UNPINNED_ROLES,
    role_set_name='_BASH_CAPABLE_UNPINNED_ROLES',
    capability=lambda role: role.prompt_spec is None and 'Bash' in role.allowed_tools,
    capability_description='a literal system_prompt and unqualified `Bash`',
)


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'BASH_CWD_ANCHOR_GUIDANCE',
        BASH_CWD_ANCHOR_GUIDANCE,
        remedy=(
            'Restore the task-5971 guidance: without it a Bash-capable role whose '
            'shell kept an earlier `cd` keeps guessing repo-root-relative paths '
            'that come back not-found.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'BASH_CWD_ANCHOR_GUIDANCE',
        BASH_CWD_ANCHOR_GUIDANCE,
        remedy='Describe any brace form in words; never spell it out.',
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert BASH_CWD_ANCHOR_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'BASH_CWD_ANCHOR_GUIDANCE does not start with MARKDOWN_HEADING, so at the '
        'tail of _BASH_CAPABLE_ROLE_PREAMBLE it would read as an unheaded '
        'continuation of whichever block precedes it. Give it a leading blank '
        'line and a `## ` heading.'
    )


def test_role_set_matches_bash_capability():
    """Premise tripwire: the hand-maintained set equals the derived one."""
    _CONTRACT.assert_role_set_matches_capability(
        remedy=(
            "A role's `Bash` grant changed, or a role was added. Update "
            '_BASH_CAPABLE_UNPINNED_ROLES to match.'
        ),
    )


def test_every_bash_capable_role_carries_guidance():
    """Every role whose shell can `cd` is told how that moves its relative paths."""
    _CONTRACT.assert_every_role_carries(
        remedy=(
            'Append BASH_CWD_ANCHOR_GUIDANCE to the tail of '
            'roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; that one composite reaches '
            'every Bash-capable role.'
        ),
    )


def test_no_other_role_carries_guidance():
    """The negative half: no role outside the set carries the block."""
    _CONTRACT.assert_no_other_role_carries(
        remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            "widening the set. judge's `Bash(git:*)` grant cannot `cd`, so its "
            'cwd cannot drift, and the block would tell it to run `cd` and `pwd` '
            'it is denied. reviewer_comprehensive is PromptSpec-backed, and a '
            'pinned artifact can drop a splice.'
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
        remedy='Keep exactly one BASH_CWD_ANCHOR_GUIDANCE term in the preamble.',
    )


def test_guidance_lands_after_the_grep_block():
    """The block lands anywhere after `GREP_LOOKAROUND_GUIDANCE` ends.

    Order only, never adjacency: the APPEND-ONLY comment above
    `orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
    holds the rationale.
    """
    _CONTRACT.assert_lands_after(
        follows=GREP_LOOKAROUND_GUIDANCE,
        follows_name='GREP_LOOKAROUND_GUIDANCE',
        remedy=(
            'Re-append it at the END of roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; '
            'moving it earlier breaks the adjacency pins of the blocks ahead of '
            'it.'
        ),
    )
