"""Anchor/contract test for the `Grep` look-around guidance (task 5331).

Fourth sibling of `test_roles_wait_pattern.py` (task 3607),
`test_roles_tool_call_rejection.py` (tasks 4273/4578) and
`test_roles_error_remedy_hint.py` (task 4964) — same shape, same provenance
kind: a legibility-census finding (`metadata.source: legibility_census`)
about a wasted agent turn, turned into a named prompt constant spliced into a
machine-derived set of roles.

The finding, reproduced first-hand: `Grep(pattern=r'config\\.(?!git|
project_root|verify_env)[a-z_]+', ...)` is rejected with `error: look-around,
including look-ahead and look-behind, is not supported` /
`Consider enabling PCRE2 with the --pcre2 flag`. The search never runs, and
the printed remedy is unreachable through the tool that printed it — `Grep`
exposes no `--pcre2` parameter in any spelling, so no re-issue of that call
can succeed.

`Grep` is a Claude Code builtin: its CAUSE is upstream of this repository and
is not addressed here at all. The full rationale for what is and is not in
scope, which roles carry which variant, and the shell-shadowing measurements
behind the escape hatch lives ONCE, in the comment block above
`_GREP_ENGINE_LIMITS` in `orchestrator/src/orchestrator/agents/roles.py`.
This module points there rather than restating it.

Its real effect is on model behaviour and is not unit-testable, but silent
removal during a prompt refactor is a genuine regression — the repo sanctions
exactly this kind of "mandated token present in each role prompt" guard. Read
"token" STRICTLY as a named constant. Every assertion in this file is an
existence / containment / count / index check against a NAMED CONSTANT: never
a string literal asserted against the constants' prose, never a regex over
wording, never a byte-size figure. A prose pin has no correctness content in
either direction — it passes on prose reworded to say the opposite and fails
on a legitimate tightening — so it only taxes future prompt edits.

The mechanical half of that shape — the offender-collection loops, the
derived-vs-hardcoded role-set comparison, the count and index bookkeeping —
lives in `_role_splice_contract.py` (task 4405). That module's docstring is
the authoritative account of what the shared shape does and does not absorb;
most of it is NOT restated here. The tests below stay one thin function per
invariant, each delegating its body to the helper while keeping its own
docstring and its own remediation prose.
"""

from __future__ import annotations

import pytest
from _role_splice_contract import (
    MARKDOWN_HEADING,
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.roles import (
    _GREP_ENGINE_LIMITS,
    _GREP_PCRE_BASH_RECOURSE,
    _GREP_PCRE_READ_ONLY_RECOURSE,
    ERROR_REMEDY_HINT_GUIDANCE,
    GREP_LOOKAROUND_GUIDANCE,
    GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
)

#: Every constant this module guards, halves and composed splice units alike.
#: Parametrizing over the mapping rather than asserting the five inline is what
#: keeps a sixth constant from arriving unchecked: adding it here gives it the
#: non-empty and brace-free guards for free.
_ALL_CONSTANTS = {
    '_GREP_ENGINE_LIMITS': _GREP_ENGINE_LIMITS,
    '_GREP_PCRE_BASH_RECOURSE': _GREP_PCRE_BASH_RECOURSE,
    '_GREP_PCRE_READ_ONLY_RECOURSE': _GREP_PCRE_READ_ONLY_RECOURSE,
    'GREP_LOOKAROUND_GUIDANCE': GREP_LOOKAROUND_GUIDANCE,
    'GREP_LOOKAROUND_GUIDANCE_READ_ONLY': GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
}

#: The two PUBLIC splice units — the only two that are ever spliced into a role
#: prompt. The three halves above reach a prompt only through one of these.
_PUBLIC_SPLICE_UNITS = {
    'GREP_LOOKAROUND_GUIDANCE': GREP_LOOKAROUND_GUIDANCE,
    'GREP_LOOKAROUND_GUIDANCE_READ_ONLY': GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
}


@pytest.mark.parametrize('name', sorted(_ALL_CONSTANTS))
def test_grep_lookaround_constants_are_nonempty(name):
    """Each guarded constant is a non-empty string.

    NOT redundant with the containment tests, though it reads that way: those
    assert `CONSTANT in ROLES[role].system_prompt`, and the empty string is a
    substring of every string — so every one of those assertions holds
    vacuously against an emptied constant. This is the sole guard against the
    guidance being silently dropped in a prompt refactor, and it covers the
    three private halves as well as the two composed units, because emptying a
    half leaves the composed unit non-empty and every other test green.
    """
    assert_nonempty(
        name,
        _ALL_CONSTANTS[name],
        remedy=(
            'Restore the census-5331 guidance: this assertion is the sole guard '
            'against it being silently dropped, including from its own '
            'composition into a public splice unit.'
        ),
    )


@pytest.mark.parametrize('name', sorted(_ALL_CONSTANTS))
def test_grep_lookaround_constants_have_no_literal_braces(name):
    """No literal ``{``/``}`` in any half or composed unit.

    Role prompts are deliberately not f-strings — these constants reach a
    prompt by plain `+` concatenation — but they are held brace-free
    defensively so they stay interpolation-safe if a future splice site needs
    them. `CODE_QUALITY_GUIDANCE` is the live example of why that matters: it
    reaches a `str.format()` template and is brace-free BY CONTRACT.
    """
    assert_brace_free(
        name,
        _ALL_CONSTANTS[name],
        remedy=(
            'A literal brace raises at format time or mangles the rendered '
            'prompt at an interpolating splice site. Spell the example pattern '
            'without a brace quantifier.'
        ),
    )


@pytest.mark.parametrize('name', sorted(_PUBLIC_SPLICE_UNITS))
def test_public_splice_units_open_their_own_section(name):
    """Each public unit opens with its own ``\\n## `` heading.

    Structural, not a wording pin: the heading TEXT is never asserted, only
    that the unit STARTS with `MARKDOWN_HEADING`. Without this a unit spliced
    behind `ERROR_REMEDY_HINT_GUIDANCE` would read as an unheaded continuation
    of that block's last paragraph — a different claim than the one it makes.

    It is also the precondition for
    `_role_splice_contract.py::assert_placement`'s up-front arm, which
    compares the unit's offset against the prompt's first `##` heading: a unit
    that does not begin with a heading can never satisfy that comparison.
    """
    assert _PUBLIC_SPLICE_UNITS[name].startswith(MARKDOWN_HEADING), (
        f'{name} does not start with MARKDOWN_HEADING, so it reads as an '
        'unheaded continuation of whatever block precedes it rather than as '
        'its own section. Give it a leading blank line and a `## ` heading.'
    )


def test_the_two_recourse_halves_are_distinct():
    """The judge variant is a genuinely different sentence, not a copy.

    The whole point of the two-variant split is that `judge` holds
    `Bash(git:*)` rather than unqualified `Bash`, so the `grep -P` escape
    hatch the other seven roles are given would land it a permission denial.
    If the two recourse halves were ever made equal, the split would buy
    nothing while still costing two constants, two role sets and two contracts
    — and every containment test in this module would stay green, because both
    composed units would then be the same string.

    Asserts INEQUALITY of two named constants, not the content of either:
    rewording either half freely is a no-op here.
    """
    assert _GREP_PCRE_BASH_RECOURSE != _GREP_PCRE_READ_ONLY_RECOURSE, (
        '_GREP_PCRE_BASH_RECOURSE and _GREP_PCRE_READ_ONLY_RECOURSE are '
        'identical, so the two-variant split buys nothing. Either restore the '
        "judge-specific recourse (it must not prescribe a shell command judge's "
        '`Bash(git:*)` grant would refuse), or collapse the split back to one '
        'unit over one role set.'
    )


#: Roles holding UNQUALIFIED `Bash`, so the `command grep -rP` escape hatch is
#: a command they can actually run.
#: Capability: `role.prompt_spec is None and 'Bash' in role.allowed_tools`.
_BASH_CAPABLE_UNPINNED_ROLES = frozenset({
    'architect',
    'debugger',
    'deep_reviewer',
    'implementer',
    'merger',
    'simple_task',
    'steward',
})

#: Roles whose `Bash` grant is narrower than unqualified (`judge` holds
#: `Bash(git:*)`), so prescribing `grep -P` would walk them into a permission
#: denial rather than a result — they get the restructure-only variant.
#: Capability: `role.prompt_spec is None and 'Bash' not in role.allowed_tools`.
_READ_ONLY_UNPINNED_ROLES = frozenset({'judge'})

#: The union, derived rather than hand-maintained a third time: every role with
#: a literal (non-`PromptSpec`) system_prompt. `reviewer_comprehensive` is in
#: NEITHER set because it is the sole `PromptSpec`-built role, whose pinned
#: artifact can override the literal template at runtime — a splice there could
#: be silently dropped in exactly the sessions it protects. That is a DEFERRED
#: COVERAGE GAP, not an exemption on the merits (it holds `Grep` and can hit the
#: rejection like any other role); it is the same reading
#: `test_roles_tool_call_rejection.py::_UNPINNED_PROMPT_ROLES` already documents,
#: and closing it means splicing into the frozen reviewer template and bumping
#: `_REVIEWER_PROMPT_HARNESS_VERSION`, deliberately not done here.
_ALL_UNPINNED_ROLES = _BASH_CAPABLE_UNPINNED_ROLES | _READ_ONLY_UNPINNED_ROLES

_CONTRACTS = {
    'GREP_LOOKAROUND_GUIDANCE': SpliceContract(
        constant_name='GREP_LOOKAROUND_GUIDANCE',
        constant=GREP_LOOKAROUND_GUIDANCE,
        roles=_BASH_CAPABLE_UNPINNED_ROLES,
        role_set_name='_BASH_CAPABLE_UNPINNED_ROLES',
        capability=lambda role: role.prompt_spec is None and 'Bash' in role.allowed_tools,
        capability_description='a literal system_prompt and unqualified `Bash`',
    ),
    'GREP_LOOKAROUND_GUIDANCE_READ_ONLY': SpliceContract(
        constant_name='GREP_LOOKAROUND_GUIDANCE_READ_ONLY',
        constant=GREP_LOOKAROUND_GUIDANCE_READ_ONLY,
        roles=_READ_ONLY_UNPINNED_ROLES,
        role_set_name='_READ_ONLY_UNPINNED_ROLES',
        capability=lambda role: role.prompt_spec is None and 'Bash' not in role.allowed_tools,
        capability_description='a literal system_prompt but no unqualified `Bash`',
    ),
}

#: The recourse half each unit is built from. Keyed by the same names as
#: `_CONTRACTS`, and deliberately NOT carrying the shared `_GREP_ENGINE_LIMITS`
#: half: that one is shared by construction, and listing it per-variant here
#: would be the second copy the split exists to avoid.
_RECOURSE_HALF = {
    'GREP_LOOKAROUND_GUIDANCE': ('_GREP_PCRE_BASH_RECOURSE', _GREP_PCRE_BASH_RECOURSE),
    'GREP_LOOKAROUND_GUIDANCE_READ_ONLY': (
        '_GREP_PCRE_READ_ONLY_RECOURSE',
        _GREP_PCRE_READ_ONLY_RECOURSE,
    ),
}

_SHARED_HALF_CONTRACT = SpliceContract(
    constant_name='_GREP_ENGINE_LIMITS',
    constant=_GREP_ENGINE_LIMITS,
    roles=_ALL_UNPINNED_ROLES,
    role_set_name='_BASH_CAPABLE_UNPINNED_ROLES | _READ_ONLY_UNPINNED_ROLES',
    capability=lambda role: role.prompt_spec is None,
    capability_description='a literal (non-PromptSpec) system_prompt',
)


@pytest.mark.parametrize('name', sorted(_CONTRACTS))
def test_role_set_matches_its_bash_capability(name):
    """Drift tripwire: each hand-maintained set still equals its derived one.

    This is what makes the two-variant split self-maintaining rather than a
    snapshot. If `judge`'s grant is ever widened to unqualified `Bash`, or a
    new role arrives, the derived set diverges from the hand-maintained one and
    the failure names the role to move — instead of leaving a role silently
    carrying an escape hatch it cannot run, or denied one it could.

    Pins the PREMISE (which roles can run `grep -P`), not the splice, so it
    passes regardless of whether the guidance has been spliced anywhere yet.
    """
    _CONTRACTS[name].assert_role_set_matches_capability(
        remedy=(
            "A role's `Bash` grant changed, or a role was added. Move it between "
            '_BASH_CAPABLE_UNPINNED_ROLES and _READ_ONLY_UNPINNED_ROLES and give '
            'it the matching variant: the read-only variant exists precisely so '
            'no role is told to run a command its grant would refuse.'
        ),
    )


@pytest.mark.parametrize('name', sorted(_CONTRACTS))
def test_splice_unit_composes_its_two_halves(name):
    """Each public unit carries the shared facts half AND its own recourse half.

    Structural enforcement of the MUST-COMPOSE mandate: neither half can be
    spliced apart from the other by a future refactor. Both are asserted, not
    just the variant-specific one — dropping `_GREP_ENGINE_LIMITS` would delete
    the limitation itself (the finding's actual remedy) from every role prompt
    while the composed unit stayed non-empty and every containment, count and
    placement test in this module stayed green.

    The helper checks each half NON-EMPTY before it checks it contained, so an
    emptied half is caught rather than passing vacuously.
    """
    _CONTRACTS[name].assert_composes(
        [('_GREP_ENGINE_LIMITS', _GREP_ENGINE_LIMITS), _RECOURSE_HALF[name]],
        remedy=(
            'Dropping _GREP_ENGINE_LIMITS would remove the limitation itself — '
            'the part every `Grep`-holding role needs — leaving only a recourse '
            'for a failure the agent is no longer warned about. Dropping the '
            'recourse half would leave a role that knows the pattern will be '
            'rejected and has been told nothing about what to do instead.'
        ),
    )


@pytest.mark.parametrize('name', sorted(_CONTRACTS))
def test_every_role_in_the_set_carries_its_variant(name):
    """Every role in the set embeds its variant in its system_prompt."""
    _CONTRACTS[name].assert_every_role_carries(
        remedy=(
            'These roles hold `Grep` and would discover the look-around '
            'rejection by failure, then spend a second turn on the unreachable '
            '`--pcre2` remedy the rejection prints.'
        ),
    )


@pytest.mark.parametrize('name', sorted(_CONTRACTS))
def test_no_role_outside_the_set_carries_that_variant(name):
    """The negative half: a role outside the set must NOT carry the variant.

    The two tests above catch a role gaining the capability and a covered role
    losing the block. Neither catches the splice landing in the WRONG set —
    which is the specific defect the split exists to prevent: the Bash variant
    in `judge` tells a role to run a command its `Bash(git:*)` grant refuses,
    and the read-only variant in a Bash-capable role withholds an escape hatch
    that role could have run. It also catches a splice into
    `reviewer_comprehensive`, whose exclusion this test enforces without
    claiming to justify.
    """
    _CONTRACTS[name].assert_no_other_role_carries(
        remedy=(
            'If the variants were swapped, swap them back. If a role genuinely '
            'changed `Bash` grant, move it between _BASH_CAPABLE_UNPINNED_ROLES '
            'and _READ_ONLY_UNPINNED_ROLES. `reviewer_comprehensive` takes '
            'neither variant: a PromptSpec-backed role may silently drop a '
            'splice at runtime, and closing that gap needs a '
            '_REVIEWER_PROMPT_HARNESS_VERSION bump.'
        ),
    )


@pytest.mark.parametrize('name', sorted(_CONTRACTS))
def test_variant_appears_exactly_once_per_role(name):
    """No duplicate splice — the variant is carried once, and only once.

    Scoped to catching a stale duplicate left beside a new one, NOT to
    enforcing presence: that is
    `test_every_role_in_the_set_carries_its_variant`'s job.
    """
    # `absent_ok=True`: a role that has not yet received the splice then fails
    # exactly ONE test for that one root cause — the containment one, whose job
    # presence is — instead of two. The asymmetry with
    # `test_the_limitation_reaches_every_unpinned_role` below, which passes
    # False, is deliberate: a zero count for the SHARED half is diagnostic
    # rather than a duplicate report, because the composition is separately
    # pinned by `test_splice_unit_composes_its_two_halves`.
    _CONTRACTS[name].assert_spliced_exactly_once(
        absent_ok=True,
        remedy=(
            'A stale duplicate splice was probably left beside a new one — '
            'delete the extra copy.'
        ),
    )


@pytest.mark.parametrize('name', sorted(_CONTRACTS))
def test_variant_placement_is_structural(name):
    """Each variant lands immediately after `ERROR_REMEDY_HINT_GUIDANCE`.

    An index comparison against a named constant — no literal text, no magic
    number. Immediately-after is the ONLY position available, not a preference:

    - The 7 Bash-capable roles carry `BACKGROUND_WAIT_GUIDANCE`, and
      `test_roles_wait_pattern.py::test_combined_guidance_is_stated_up_front`
      requires that block's heading to remain the prompt's FIRST `##` heading
      within its char budget.
    - `judge` carries no wait block, and
      `test_roles_tool_call_rejection.py::test_guidance_placement_is_structural`
      requires `TOOL_CALL_REJECTION_GUIDANCE`'s heading to be its first `##`.

    So nothing may be spliced ahead of either landmark, and
    `ERROR_REMEDY_HINT_GUIDANCE` abuts both — appending after it is the one
    spot satisfying every pre-existing invariant with no per-role branching.
    No `char_budget` is passed: that secondary bound belongs to the wait
    block's own up-front invariant, not to a block spliced behind it.

    A role where the variant is absent entirely is recorded as an offender
    rather than skipped, so this can never pass vacuously.
    """
    _CONTRACTS[name].assert_placement(
        follows=ERROR_REMEDY_HINT_GUIDANCE,
        follows_name='ERROR_REMEDY_HINT_GUIDANCE',
        remedy=(
            'It cannot be moved ahead of the wait block or of '
            'TOOL_CALL_REJECTION_GUIDANCE to fix this — both are pinned as their '
            "prompts' FIRST `##` heading by the task-3607 and task-4273 "
            'invariants. Re-abut it to ERROR_REMEDY_HINT_GUIDANCE instead.'
        ),
    )


def test_the_limitation_reaches_every_unpinned_role():
    """The shared half reaches all eight unpinned roles, exactly once each.

    The finding's actual remedy, and the one claim neither per-variant contract
    states on its own: every role that holds `Grep` — which is every role — is
    told the limit BEFORE it writes the pattern. The per-variant tests only ever
    assert over their own set, so a role dropped from BOTH sets would leave
    every one of them green.

    Two assertions because the claim has two halves that are meaningless apart:
    that the two sets still PARTITION the unpinned roles (so the union is not
    quietly missing one), and that the shared half actually reaches each of
    them once.
    """
    _SHARED_HALF_CONTRACT.assert_role_set_matches_capability(
        remedy=(
            'A role gained or lost a literal system_prompt, so the two variant '
            'sets no longer partition the unpinned roles. Add it to whichever of '
            '_BASH_CAPABLE_UNPINNED_ROLES / _READ_ONLY_UNPINNED_ROLES its `Bash` '
            'grant selects — leaving it in neither silently denies it the '
            'limitation while every per-variant test stays green.'
        ),
    )
    # `absent_ok=False`, unlike the per-variant count test above: with the
    # composition pinned by `test_splice_unit_composes_its_two_halves`, a zero
    # count here means that role carries NEITHER variant, and a count of 2 means
    # a role was spliced with both.
    _SHARED_HALF_CONTRACT.assert_spliced_exactly_once(
        absent_ok=False,
        remedy=(
            'A count of 0 means the role carries neither variant and is not told '
            'the limitation at all; a count of 2 means it was spliced with both, '
            'so it has been given contradictory recourse.'
        ),
    )
