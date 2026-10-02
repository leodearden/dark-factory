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

from _role_splice_contract import (
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.pkill_guidance import PKILL_SELF_MATCH_GUIDANCE
from orchestrator.agents.roles import BACKGROUND_WAIT_GUIDANCE, GREP_LOOKAROUND_GUIDANCE

#: Roles holding UNQUALIFIED `Bash` with a literal system_prompt — the only ones
#: that can run `pkill`/`pgrep` at all. `judge` is absent because its grant is
#: `Bash(git:*)`; `reviewer_comprehensive` is absent because it is
#: PromptSpec-backed and also holds only `Bash(git:*)`.
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
    constant_name='PKILL_SELF_MATCH_GUIDANCE',
    constant=PKILL_SELF_MATCH_GUIDANCE,
    roles=_BASH_CAPABLE_UNPINNED_ROLES,
    role_set_name='_BASH_CAPABLE_UNPINNED_ROLES',
    capability=lambda role: role.prompt_spec is None and 'Bash' in role.allowed_tools,
    capability_description='a literal system_prompt and unqualified `Bash`',
)


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


def test_role_set_matches_bash_capability():
    """Premise tripwire: the hand-maintained set equals the derived one."""
    _CONTRACT.assert_role_set_matches_capability(
        remedy=(
            "A role's `Bash` grant changed, or a role was added. Update "
            '_BASH_CAPABLE_UNPINNED_ROLES to match: a role that can run '
            '`pkill -f` needs the guidance, and one that cannot does not.'
        ),
    )


def test_every_bash_capable_role_carries_guidance():
    """Every role that can run `pkill -f` is told it self-matches."""
    _CONTRACT.assert_every_role_carries(
        remedy=(
            'Append PKILL_SELF_MATCH_GUIDANCE to the tail of '
            'roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; that one composite reaches '
            'every Bash-capable role.'
        ),
    )


def test_no_other_role_carries_guidance():
    """The negative half: no role outside the set carries the block."""
    _CONTRACT.assert_no_other_role_carries(
        remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: their `Bash(git:*)` grants refuse `pkill`, so the '
            'block is dead weight there.'
        ),
    )


def test_guidance_appears_exactly_once_per_role():
    """No stale duplicate splice survives beside a new one.

    ``absent_ok=True``: a missing splice fails only the containment test — one
    root cause, one failing test, as `test_roles_grep_lookaround.py` documents.
    """
    _CONTRACT.assert_spliced_exactly_once(
        absent_ok=True,
        remedy='Keep exactly one PKILL_SELF_MATCH_GUIDANCE term in the preamble.',
    )


def test_guidance_lands_after_the_grep_block():
    """The block lands anywhere after `GREP_LOOKAROUND_GUIDANCE` ends.

    (a) Landing there keeps every existing placement pin intact: the wait block
    stays the first `##` heading, and the tool-call-rejection, error-remedy and
    grep blocks each still abut their predecessor.

    (b) Adjacency is deliberately NOT pinned. Sibling census blocks append to
    the same tail of `_BASH_CAPABLE_ROLE_PREAMBLE` concurrently, so "immediately
    after X" would make whichever branch merges second fail its own test.

    ABSENT is recorded, never skipped, so this cannot pass vacuously.
    """
    offenders: dict[str, dict[str, object]] = {}
    for role in sorted(_CONTRACT.roles):
        prompt = _CONTRACT.all_roles[role].system_prompt
        idx = prompt.find(PKILL_SELF_MATCH_GUIDANCE)
        grep_idx = prompt.find(GREP_LOOKAROUND_GUIDANCE)
        if idx == -1 or grep_idx == -1:
            offenders[role] = {
                'offset': idx if idx != -1 else 'ABSENT',
                'follows_offset': grep_idx if grep_idx != -1 else 'ABSENT',
            }
            continue
        earliest_allowed = grep_idx + len(GREP_LOOKAROUND_GUIDANCE)
        if idx < earliest_allowed:
            offenders[role] = {'offset': idx, 'earliest_allowed': earliest_allowed}

    assert offenders == {}, (
        f'Roles placing PKILL_SELF_MATCH_GUIDANCE incorrectly: {offenders}. It '
        'must land anywhere after GREP_LOOKAROUND_GUIDANCE ends. Append it at '
        'the tail of roles.py::_BASH_CAPABLE_ROLE_PREAMBLE; moving it earlier '
        'breaks the adjacency pins of the blocks ahead of it.'
    )


def test_wait_section_pointers_cannot_dangle():
    """The block's "section above" pointers resolve to `BACKGROUND_WAIT_GUIDANCE`.

    The prose points at the exit-code section and at the wait section's
    termination tool, both members of that block. The structural twin of
    `orchestrator/tests/test_roles_wait_pattern.py::test_amender_reminder_cannot_dangle`.
    A role where either constant is absent is recorded (offset -1), never
    skipped.
    """
    offenders: dict[str, dict[str, int]] = {}
    for role in sorted(_CONTRACT.roles):
        prompt = _CONTRACT.all_roles[role].system_prompt
        wait_offset = prompt.find(BACKGROUND_WAIT_GUIDANCE)
        pkill_offset = prompt.find(PKILL_SELF_MATCH_GUIDANCE)
        if not 0 <= wait_offset < pkill_offset:
            offenders[role] = {'wait_offset': wait_offset, 'pkill_offset': pkill_offset}

    assert offenders == {}, (
        f'Roles where PKILL_SELF_MATCH_GUIDANCE does not follow '
        f'BACKGROUND_WAIT_GUIDANCE: {offenders}. Its pointers to the exit-code '
        'and wait sections "above" would dangle. Keep both in the preamble, '
        'the wait block first.'
    )
