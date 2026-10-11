"""Anchor/contract test for `FILE_LOOKUP_GUIDANCE` (task 5975, reify codebook entry-cand-20260827-17).

The sighting: a recursive `find` from a project root that holds a full copy of
the tree per task worktree. It visited every copy and ran into the Bash
timeout. The block says where a by-name search is safe and names the lookup
that cannot walk into those copies.

Every assertion here is an existence, containment, count or index check against
a NAMED constant, never a prose pin. `_role_splice_contract.py` holds the
rationale for that rule and the mechanical half of each check.
"""

from __future__ import annotations

from dataclasses import replace

from _role_splice_contract import (
    BASH_CAPABLE_UNPINNED_ROLES,
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
    bash_capable_unpinned_contract,
)

from orchestrator.agents.file_lookup_guidance import FILE_LOOKUP_GUIDANCE
from orchestrator.agents.roles import (
    GREP_LOOKAROUND_GUIDANCE,
    GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
    ROLES,
    AgentRole,
)

#: Every role with a literal system_prompt, derived from ROLES.
_UNPINNED_ROLES = frozenset(name for name, role in ROLES.items() if role.prompt_spec is None)

#: Roles building their own chain around the read-only Grep variant (`judge`).
_READ_ONLY_ROLES = _UNPINNED_ROLES - BASH_CAPABLE_UNPINNED_ROLES


def _can_run_every_prescribed_lookup(role: AgentRole) -> bool:
    """The block prescribes git's index, `Grep` and a narrowed `Glob`, so a carrier must run all three (task 5331)."""
    return (
        role.prompt_spec is None
        and 'Glob' in role.allowed_tools
        and 'Grep' in role.allowed_tools
        and ('Bash' in role.allowed_tools or 'Bash(git:*)' in role.allowed_tools)
    )


_CONTRACT = SpliceContract(
    constant_name='FILE_LOOKUP_GUIDANCE',
    constant=FILE_LOOKUP_GUIDANCE,
    roles=_UNPINNED_ROLES,
    role_set_name='_UNPINNED_ROLES',
    capability=_can_run_every_prescribed_lookup,
    capability_description=(
        'a literal (non-PromptSpec) system_prompt, `Glob`, `Grep` and a '
        'git-capable `Bash` grant'
    ),
)


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'FILE_LOOKUP_GUIDANCE',
        FILE_LOOKUP_GUIDANCE,
        remedy=(
            'Restore the task-5975 guidance: without it an agent looking for a '
            'file by name from a root holding worktree copies walks every copy '
            'until the Bash timeout kills the search.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'FILE_LOOKUP_GUIDANCE',
        FILE_LOOKUP_GUIDANCE,
        remedy='Write placeholders in angle brackets; never spell out a brace form.',
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert FILE_LOOKUP_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'FILE_LOOKUP_GUIDANCE does not start with MARKDOWN_HEADING, so where it '
        'is spliced it would read as an unheaded continuation of whichever '
        'block precedes it. Give it a leading blank line and a `## ` heading.'
    )


def test_unpinned_role_set_is_nonempty():
    """No splice test below can pass vacuously over an emptied derivation.

    `_READ_ONLY_ROLES` is deliberately not required non-empty: it empties
    legitimately if `judge` ever gains unqualified `Bash`, and the grep
    module's own tripwire fires for that change.
    """
    assert _UNPINNED_ROLES, (
        'No role has a literal system_prompt, so every splice test here passes '
        'over an empty set. Check the `prompt_spec is None` derivation.'
    )


def test_every_carrier_can_run_the_lookups():
    """Every derived carrier holds `Glob`, `Grep` and a git-capable `Bash`.

    Not a tautology: `_UNPINNED_ROLES` is derived from the literal prompt
    alone, while the capability also requires the three grants the block
    prescribes.
    """
    _CONTRACT.assert_role_set_matches_capability(
        remedy=(
            '_UNPINNED_ROLES is derived, not hand-maintained, so a `lost=` role '
            'is a literal-prompt role missing `Glob`, `Grep` or a git-capable '
            '`Bash`: it would be told to run a lookup it cannot run. Grant the '
            'missing tool, or exclude it from the derivation and from the '
            'splice together.'
        ),
    )


def test_every_unpinned_role_carries_guidance():
    """Every role that can search for a file by name is told where that is safe."""
    _CONTRACT.assert_every_role_carries(
        remedy=(
            'For a role holding unqualified `Bash`, append FILE_LOOKUP_GUIDANCE '
            'to the tail of roles.py::_BASH_CAPABLE_ROLE_PREAMBLE. For any other '
            "literal-prompt role, splice it into that role's own chain, as "
            'roles.py::JUDGE does.'
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
        remedy='Keep exactly one FILE_LOOKUP_GUIDANCE term per splice chain.',
    )


def test_guidance_lands_after_the_grep_block_in_bash_roles():
    """In the preamble roles the block lands anywhere after `GREP_LOOKAROUND_GUIDANCE`.

    Order, not adjacency: sibling census blocks append to the same tail in any
    merge order. The APPEND-ONLY comment above
    `orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
    holds that rationale.
    """
    bash_capable_unpinned_contract(
        'FILE_LOOKUP_GUIDANCE', FILE_LOOKUP_GUIDANCE
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
            'Splice it after GREP_LOOKAROUND_GUIDANCE_READ_ONLY in the '
            "role's own chain, as roles.py::JUDGE builds it."
        ),
    )
