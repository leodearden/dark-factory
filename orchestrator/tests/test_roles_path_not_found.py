"""Anchor/contract test for `PATH_NOT_FOUND_GUIDANCE` (task 5970, reify census #7920).

The block tells an agent that a not-found error on a path it expected to exist
falsifies the path, so the next call must search for the file by name rather
than hand the same guess to another tool.

Every assertion here is an existence, containment, count or index check against
a NAMED constant, never a prose pin. `_role_splice_contract.py` holds the
rationale for that rule and the mechanical half of each check.
"""

from __future__ import annotations

from dataclasses import replace

from _role_splice_contract import (
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.path_not_found_guidance import PATH_NOT_FOUND_GUIDANCE
from orchestrator.agents.roles import (
    GREP_LOOKAROUND_GUIDANCE,
    GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
    ROLES,
)

#: Every role with a literal system_prompt. DERIVED rather than
#: hand-maintained: two sibling modules already keep this set by hand.
_UNPINNED_ROLES = frozenset(name for name, role in ROLES.items() if role.prompt_spec is None)

#: The carriers of `_BASH_CAPABLE_ROLE_PREAMBLE`, whose tail this block joins.
_BASH_PREAMBLE_ROLES = frozenset(
    name for name in _UNPINNED_ROLES if 'Bash' in ROLES[name].allowed_tools
)

#: Roles building their own chain around the read-only Grep variant (`judge`).
_READ_ONLY_ROLES = _UNPINNED_ROLES - _BASH_PREAMBLE_ROLES

_CONTRACT = SpliceContract(
    constant_name='PATH_NOT_FOUND_GUIDANCE',
    constant=PATH_NOT_FOUND_GUIDANCE,
    roles=_UNPINNED_ROLES,
    role_set_name='_UNPINNED_ROLES',
    capability=lambda role: role.prompt_spec is None and 'Glob' in role.allowed_tools,
    capability_description='a literal (non-PromptSpec) system_prompt and `Glob`',
)


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'PATH_NOT_FOUND_GUIDANCE',
        PATH_NOT_FOUND_GUIDANCE,
        remedy=(
            'Restore the task-5970 guidance: without it an agent whose expected '
            'path came back not-found retries the same guess with another tool '
            'instead of searching for the file.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'PATH_NOT_FOUND_GUIDANCE',
        PATH_NOT_FOUND_GUIDANCE,
        remedy='Describe any brace form in words; never spell it out.',
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert PATH_NOT_FOUND_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'PATH_NOT_FOUND_GUIDANCE does not start with MARKDOWN_HEADING, so where '
        'it is spliced it would read as an unheaded continuation of whichever '
        'block precedes it. Give it a leading blank line and a `## ` heading.'
    )


def test_derived_role_sets_are_nonempty():
    """No splice test below can pass vacuously over an emptied derivation.

    `_READ_ONLY_ROLES` is deliberately not required non-empty: it empties
    legitimately if `judge` ever gains unqualified `Bash`, and the grep
    module's own tripwire fires for that change.
    """
    assert _UNPINNED_ROLES, (
        'No role has a literal system_prompt, so every splice test here passes '
        'over an empty set. Check the `prompt_spec is None` derivation.'
    )
    assert _BASH_PREAMBLE_ROLES, (
        'No literal-prompt role holds unqualified `Bash`, so the preamble-order '
        "test passes over an empty set. Check the `'Bash' in allowed_tools` "
        'derivation.'
    )


def test_every_carrier_can_run_the_recovery():
    """Every derived carrier holds `Glob`, the one tool the recovery names.

    Not a tautology: `_UNPINNED_ROLES` is derived from the literal prompt
    alone, while the capability also requires the `Glob` grant.
    """
    _CONTRACT.assert_role_set_matches_capability(
        remedy=(
            '_UNPINNED_ROLES is derived, not hand-maintained, so a `lost=` role '
            'is a literal-prompt role without `Glob`: it would be told to run a '
            'search it cannot run. Grant it `Glob`, or exclude it from the '
            'derivation and from the splice together.'
        ),
    )


def test_every_unpinned_role_carries_guidance():
    """Every role that can hit a not-found is told how to recover from it."""
    _CONTRACT.assert_every_role_carries(
        remedy=(
            'For a role holding unqualified `Bash`, append PATH_NOT_FOUND_GUIDANCE '
            'to the tail of roles.py::_BASH_CAPABLE_ROLE_PREAMBLE. For any other '
            "literal-prompt role, splice it into that role's own chain, as "
            'roles.py::JUDGE does after GREP_LOOKAROUND_GUIDANCE_READ_ONLY.'
        ),
    )


def test_no_other_role_carries_guidance():
    """The negative half: no PromptSpec-backed role carries the block."""
    _CONTRACT.assert_no_other_role_carries(
        remedy=(
            'Remove the splice. A PromptSpec-backed role such as '
            'reviewer_comprehensive can have its literal prompt overridden by a '
            'pinned artifact at runtime, so a splice there is unreliable. '
            'Reaching it needs a deliberate _REVIEWER_PROMPT_HARNESS_VERSION '
            'change, not a silent splice.'
        ),
    )


def test_guidance_appears_exactly_once_per_role():
    """No stale duplicate splice survives beside a new one.

    ``absent_ok=True`` leaves a missing splice to fail only
    `test_every_unpinned_role_carries_guidance`: one root cause, one failing
    test.
    """
    _CONTRACT.assert_spliced_exactly_once(
        absent_ok=True,
        remedy='Keep exactly one PATH_NOT_FOUND_GUIDANCE term per splice chain.',
    )


def test_guidance_lands_after_the_grep_block_in_bash_roles():
    """In the preamble roles the block lands anywhere after `GREP_LOOKAROUND_GUIDANCE`.

    Order, not adjacency: sibling census blocks append to the same tail in any
    merge order. The APPEND-ONLY comment above
    `orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
    holds that rationale.
    """
    replace(
        _CONTRACT, roles=_BASH_PREAMBLE_ROLES, role_set_name='_BASH_PREAMBLE_ROLES'
    ).assert_lands_after(
        follows=GREP_LOOKAROUND_GUIDANCE,
        follows_name='GREP_LOOKAROUND_GUIDANCE',
        remedy=(
            'Re-append it at the END of roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; '
            'moving it earlier breaks the adjacency pins of the blocks ahead of '
            'it.'
        ),
    )


def test_guidance_lands_after_the_read_only_grep_block():
    """In the read-only roles the block lands after `GREP_LOOKAROUND_GUIDANCE_READ_ONLY`."""
    replace(
        _CONTRACT, roles=_READ_ONLY_ROLES, role_set_name='_READ_ONLY_ROLES'
    ).assert_lands_after(
        follows=GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
        follows_name='GREP_LOOKAROUND_GUIDANCE_READ_ONLY',
        remedy=(
            'Splice it straight after GREP_LOOKAROUND_GUIDANCE_READ_ONLY in the '
            "role's own chain, as roles.py::JUDGE builds it."
        ),
    )
