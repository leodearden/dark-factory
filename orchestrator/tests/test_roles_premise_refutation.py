"""Anchor/contract test for `PREMISE_REFUTATION_GUIDANCE` (task 5976).

A sibling of `test_roles_sigpipe_under_pipefail.py`. Every assertion here is
an existence, containment, count or index check against a NAMED constant — never
a prose pin, never a regex over wording. The mechanical half of that shape lives
in `_role_splice_contract.py`.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.premise_refutation_guidance import PREMISE_REFUTATION_GUIDANCE
from orchestrator.agents.roles import GREP_LOOKAROUND_GUIDANCE

#: Roles holding UNQUALIFIED `Bash` with a literal system_prompt: they can run a
#: live check and mistake its non-reproduction for a refutation. `judge` holds
#: only `Bash(git:*)`; `reviewer_comprehensive` is PromptSpec-backed and also
#: holds only `Bash(git:*)`.
_LIVE_CHECK_ROLES = frozenset({
    'architect',
    'debugger',
    'deep_reviewer',
    'implementer',
    'merger',
    'simple_task',
    'steward',
})

_CONTRACT = SpliceContract(
    constant_name='PREMISE_REFUTATION_GUIDANCE',
    constant=PREMISE_REFUTATION_GUIDANCE,
    roles=_LIVE_CHECK_ROLES,
    role_set_name='_LIVE_CHECK_ROLES',
    capability=lambda role: role.prompt_spec is None and 'Bash' in role.allowed_tools,
    capability_description='a literal system_prompt and unqualified `Bash`',
)


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


def test_role_set_matches_live_check_capability():
    """Premise tripwire: the hand-maintained set equals the derived one."""
    _CONTRACT.assert_role_set_matches_capability(
        remedy=(
            "A role's `Bash` grant changed, or a role was added. Update "
            '_LIVE_CHECK_ROLES to match: any role that can run a live check '
            'can misread its non-reproduction as a refutation.'
        ),
    )


def test_every_live_check_role_carries_guidance():
    """Every role that can run a live check is told what its non-reproduction proves."""
    _CONTRACT.assert_every_role_carries(
        remedy=(
            'Append PREMISE_REFUTATION_GUIDANCE to the tail of '
            'roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; that one composite reaches '
            'every live-check role.'
        ),
    )


def test_no_other_role_carries_guidance():
    """The negative half: no role outside the set carries the block."""
    _CONTRACT.assert_no_other_role_carries(
        remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: a `Bash(git:*)` grant cannot run a check in a '
            "claim's execution context, so the block is dead weight there."
        ),
    )


def test_guidance_appears_exactly_once_per_role():
    """No stale duplicate splice survives beside a new one.

    ``absent_ok=True`` leaves a missing splice to fail only
    `test_every_live_check_role_carries_guidance`: one root cause, one
    failing test.
    """
    _CONTRACT.assert_spliced_exactly_once(
        absent_ok=True,
        remedy='Keep exactly one PREMISE_REFUTATION_GUIDANCE term in the preamble.',
    )


def test_guidance_lands_after_the_grep_block():
    """The block lands anywhere after `GREP_LOOKAROUND_GUIDANCE` ends.

    Adjacency is deliberately NOT pinned: sibling census blocks append to the
    same tail in any merge order. The APPEND-ONLY comment above
    `orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
    holds that rationale.
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
