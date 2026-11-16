"""Anchor/contract test for `ERROR_REMEDY_HINT_GUIDANCE` (task 4964).

Sibling of `test_roles_tool_call_rejection.py` (tasks 4273 + 4578) — same
shape, same provenance kind: a legibility-census finding about a wasted
agent turn, turned into a named prompt constant spliced into a
machine-derived set of roles: census 2026-08-31 §1.1
(`plans/confusion-census-2026-08-31.md`), task 4964. An `Edit` call failed
with a multi-match error whose remedy clause said to set `replace_all` to
true, and the next turn called a bare `replace_all` as if it were a tool.

SCOPE. `Edit`'s multi-match error text is emitted by the harness, upstream of
this repository, so only the agent's RESPONSE to it is in scope.

THE EDITING RULE. The census did not establish that `replace_all: true` was
the right fix for that edit, so the prose presents both remedies as chosen on
intent and never says that setting the parameter is always right.

The trace, its codebook entry and the founding session live in task 4964's
commits, not here.

THIS IS A SEPARATE MODULE, not a fourth bullet added to
`test_roles_tool_call_rejection.py` the way task 4578 shared that module,
because (per the finding above, and this task's
design decision of the same name) the finding is a DIFFERENT error class:
a well-formed `Edit` call that parsed fine and failed on CONTENT, printing
a remedy, rather than one of `InputValidationError`'s
defect-in-the-call shapes. It gets its own `##` section and its own
constant, `ERROR_REMEDY_HINT_GUIDANCE`, spliced immediately after
`TOOL_CALL_REJECTION_GUIDANCE` rather than composed into it.

Its real effect is on model behaviour and is not unit-testable, but
silent removal during a prompt refactor is a genuine regression — the
repo sanctions exactly this kind of "mandated token present in each role
prompt" guard. Read "token" there STRICTLY as a named constant: the
sanctioned shape is `SOME_CONSTANT in ROLES[name].system_prompt`, never a
string literal asserted against a constant's prose. The latter has no
correctness content in either direction — it passes on prose reworded to
say the opposite and fails on a legitimate tightening — so it only taxes
future prompt edits. Every assertion in this file is an existence /
containment / count / index check against a NAMED CONSTANT: never a
string literal asserted against the constant's prose, never a regex over
wording, never a byte-size figure. `_role_splice_contract.py` holds the
mechanical half of each check.
"""

from __future__ import annotations

from _role_splice_contract import (
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
    assert_opens_own_section,
)

from orchestrator.agents.harness_guidance import (
    ERROR_REMEDY_HINT_GUIDANCE,
    TOOL_CALL_REJECTION_GUIDANCE,
)
from orchestrator.agents.roles import ROLES, AgentRole


def _carries_tool_call_rejection_guidance(role: AgentRole) -> bool:
    return TOOL_CALL_REJECTION_GUIDANCE in role.system_prompt


#: This block travels with this set (see test_carried_by_exactly_the_tool_call_rejection_role_set).
_TOOL_CALL_REJECTION_CARRIERS = frozenset(
    name for name, role in ROLES.items() if _carries_tool_call_rejection_guidance(role)
)

# `roles` is DERIVED from `capability`, so `assert_role_set_matches_capability`
# would compare the derivation with itself; it is deliberately never called here.
_CONTRACT = SpliceContract(
    constant_name='ERROR_REMEDY_HINT_GUIDANCE',
    constant=ERROR_REMEDY_HINT_GUIDANCE,
    roles=_TOOL_CALL_REJECTION_CARRIERS,
    role_set_name='_TOOL_CALL_REJECTION_CARRIERS',
    capability=_carries_tool_call_rejection_guidance,
    capability_description='TOOL_CALL_REJECTION_GUIDANCE in its system_prompt',
)


def test_error_remedy_hint_guidance_is_nonempty() -> None:
    """The mandated constant is a non-empty string.

    Mirrors `test_tool_call_rejection_guidance_is_nonempty`. NOT redundant
    with the containment tests added in step-3, though it reads that way:
    those assert `ERROR_REMEDY_HINT_GUIDANCE in ROLES[name].system_prompt`
    for each spliced role, and the empty string is a substring of every
    string — so every one of those assertions holds vacuously if this
    constant is ever emptied. This one-line assertion is the sole guard
    against the guidance being silently dropped in a prompt refactor.
    """
    assert_nonempty(
        'ERROR_REMEDY_HINT_GUIDANCE',
        ERROR_REMEDY_HINT_GUIDANCE,
        remedy='Restore the census-4964 guidance rather than letting a prompt refactor drop it.',
    )


def test_guidance_opens_with_its_own_markdown_heading() -> None:
    """The constant is its own `##` section, not a paragraph glued onto another.

    This is what makes the block able to sit as its own section immediately
    after `TOOL_CALL_REJECTION_GUIDANCE` (see step-3's placement test)
    rather than reading as an unheaded continuation of whatever precedes
    it.
    """
    assert_opens_own_section(
        'ERROR_REMEDY_HINT_GUIDANCE',
        ERROR_REMEDY_HINT_GUIDANCE,
        remedy=(
            'The constant must carry its own markdown heading to sit as its own '
            'section immediately after TOOL_CALL_REJECTION_GUIDANCE.'
        ),
    )


def test_guidance_is_brace_free() -> None:
    """No literal ``{``/``}`` in the constant.

    Defensive, not load-bearing, exactly as documented for
    `WAIT_PATTERN_GUIDANCE` and both halves of `TOOL_CALL_REJECTION_GUIDANCE`:
    role prompts reach these constants only by plain `+` concatenation,
    which is brace-safe by construction, but staying brace-free costs
    nothing and keeps the constant interpolation-safe if a future splice
    site does use an f-string.
    """
    assert_brace_free(
        'ERROR_REMEDY_HINT_GUIDANCE',
        ERROR_REMEDY_HINT_GUIDANCE,
        remedy=(
            'Role prompts are deliberately not f-strings, but this constant is '
            'held brace-free defensively so it stays interpolation-safe if a '
            'future splice site needs it.'
        ),
    )


def test_carried_by_exactly_the_tool_call_rejection_role_set() -> None:
    """`ERROR_REMEDY_HINT_GUIDANCE` and `TOOL_CALL_REJECTION_GUIDANCE` share one carrier set.

    The set of eligible roles is not re-derived or hand-maintained here —
    see this task's design decision on the coupling invariant. Rather than a
    second `_UNPINNED_PROMPT_ROLES`-style frozenset that could silently
    drift from the sibling module's copy, this test asserts the two blocks
    simply travel together.
    `test_roles_tool_call_rejection.py::test_unpinned_prompt_role_set_matches_prompt_spec_capability`
    already owns the tripwire pinning WHICH roles have unpinned prompts, and
    that module's `test_artifact_pinned_role_does_not_carry_guidance`
    already pins `reviewer_comprehensive` out of the
    `TOOL_CALL_REJECTION_GUIDANCE` carrier set — so an accidental splice
    into `reviewer_comprehensive` here shows up as `extra_new` below,
    without a separate negative test duplicating that coverage.
    """
    new_carriers = {
        name for name, role in ROLES.items() if ERROR_REMEDY_HINT_GUIDANCE in role.system_prompt
    }

    # Guard-before-containment, same discipline the sibling module's
    # test_missing_required_parameter_shape_is_composed_into_the_splice_unit
    # documents: if both blocks were ever dropped from every role, the
    # equality check below would hold VACUOUSLY (empty == empty) and this
    # test would pass on a total regression. This assertion is the sole
    # guard against that.
    shared = _TOOL_CALL_REJECTION_CARRIERS & new_carriers
    assert shared, (
        'No role carries both TOOL_CALL_REJECTION_GUIDANCE and '
        'ERROR_REMEDY_HINT_GUIDANCE. The equality check below still passes '
        'on two independently-empty sets — this assertion is the sole '
        'guard against both blocks being silently dropped from every role '
        'prompt at once.'
    )

    missing_new = sorted(_TOOL_CALL_REJECTION_CARRIERS - new_carriers)
    extra_new = sorted(new_carriers - _TOOL_CALL_REJECTION_CARRIERS)
    assert missing_new == [] and extra_new == [], (
        f'ERROR_REMEDY_HINT_GUIDANCE carrier set diverges from '
        f'TOOL_CALL_REJECTION_GUIDANCE: missing_new={missing_new} (carry '
        f'TOOL_CALL_REJECTION_GUIDANCE but not the new block) '
        f'extra_new={extra_new} (carry the new block without '
        'TOOL_CALL_REJECTION_GUIDANCE). A role gaining or losing an '
        'unpinned prompt is '
        'test_roles_tool_call_rejection.py::'
        'test_unpinned_prompt_role_set_matches_prompt_spec_capability to '
        'adjudicate — this test only enforces that the two blocks move '
        'together.'
    )


def test_guidance_appears_exactly_once_per_role() -> None:
    """No duplicate splice — the block is carried once, and only once, per role.

    Scoped to catching a stale duplicate splice left beside a new one, NOT
    to enforcing presence — that is
    `test_carried_by_exactly_the_tool_call_rejection_role_set`'s job. A role
    where the block is entirely absent is skipped here rather than flagged.
    Presence belongs to that test's `missing_new`, and
    `test_placement_immediately_follows_tool_call_rejection_guidance` also
    records it as ABSENT, so counting it here would make a third failure for
    one root cause. Mirrors the sibling module's
    `test_guidance_appears_exactly_once_per_role`.

    It iterates `_TOOL_CALL_REJECTION_CARRIERS`; a copy in any other role is
    reported as `extra_new` by
    `test_carried_by_exactly_the_tool_call_rejection_role_set`.
    """
    _CONTRACT.assert_spliced_exactly_once(
        absent_ok=True,
        remedy='Keep exactly one ERROR_REMEDY_HINT_GUIDANCE term per splice chain.',
    )


def test_placement_immediately_follows_tool_call_rejection_guidance() -> None:
    """The new block lands immediately after `TOOL_CALL_REJECTION_GUIDANCE`, always.

    Iterates `TOOL_CALL_REJECTION_GUIDANCE`'s own carrier set, not the new
    block's, so a carrier that gained an out-of-place copy is still checked.
    A carrier where the new block is absent is recorded as an `'ABSENT'`
    offender rather than skipped, as `SpliceContract.assert_placement`
    records it for every consumer. Such a role also fails
    `test_carried_by_exactly_the_tool_call_rejection_role_set`'s
    `missing_new` check. If `TOOL_CALL_REJECTION_GUIDANCE` vanishes from
    every role, the carrier set is empty and this test passes vacuously;
    that test's `shared` guard is what catches it.

    Why the new block cannot be placed EARLIER: two pre-existing invariants
    constrain the front of every role prompt. For the 7 roles carrying
    `BACKGROUND_WAIT_GUIDANCE`,
    `test_roles_wait_pattern.py::test_combined_guidance_is_stated_up_front`
    requires that block's heading to remain the prompt's FIRST `##` heading
    and to land within `_UP_FRONT_CHAR_BUDGET` chars, so splicing ahead of
    it would break both halves of that invariant. For `judge` (which
    carries no wait block),
    `test_roles_tool_call_rejection.py::test_guidance_placement_is_structural`
    requires `TOOL_CALL_REJECTION_GUIDANCE`'s heading to be the prompt's
    first `##` heading. Appending strictly after `TOOL_CALL_REJECTION_GUIDANCE`
    is the one position that satisfies both, uniformly, with no per-role
    branching.
    """
    _CONTRACT.assert_placement(
        follows=TOOL_CALL_REJECTION_GUIDANCE,
        follows_name='TOOL_CALL_REJECTION_GUIDANCE',
        remedy=(
            'It must immediately follow TOOL_CALL_REJECTION_GUIDANCE in every '
            'role that carries that block. A role reported ABSENT has dropped '
            'the new block entirely, which '
            'test_carried_by_exactly_the_tool_call_rejection_role_set also '
            'reports as missing_new: restore the splice rather than moving '
            'anything.'
        ),
    )
