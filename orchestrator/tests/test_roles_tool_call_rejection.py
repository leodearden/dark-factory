"""Anchor/contract test for `TOOL_CALL_REJECTION_GUIDANCE` (task 4273).

Sibling of `test_roles_wait_pattern.py` (task 3607) — same shape, same
provenance kind: a legibility-census finding about a wasted agent turn,
turned into a named prompt constant spliced into a machine-derived set of
roles. This one is census 2026-08-16 §1.1
(`plans/confusion-census-2026-08-16.md`): a `Read` tool call whose echoed
input carried an empty parameter slot between `offset` and `limit`
(`"offset": 1240, , "limit": 1`) was rejected by `InputValidationError` as
unparseable JSON, and three turns later the agent reissued the *identical*
malformed structure with only `limit` edited (1 -> 260) — rejected
identically. The retry encoded a misdiagnosis: the agent read the rejection
as a bad `limit` VALUE rather than a JSON SYNTAX defect.

The stray comma's CAUSE is upstream of this repository — tool-call
generation/serialization for a Claude Code builtin (`Read`) — and is not
addressed here at all. What IS in this repo's control, and what this
constant targets, is the RETRY facet: an agent that reads the rejection
correctly does not need three wasted turns to recover from it.

Its real effect is on model behaviour and is not unit-testable, but silent
removal during a prompt refactor is a genuine regression — the repo
sanctions exactly this kind of "mandated token present in each role prompt"
guard. Read "token" there STRICTLY as a named constant: the sanctioned shape
is `SOME_CONSTANT in ROLES[name].system_prompt`, never a string literal
asserted against a constant's prose. The latter has no correctness content
in either direction — it passes on prose reworded to say the opposite and
fails on a legitimate tightening — so it only taxes future prompt edits.
Every assertion in this file is an existence / containment / count / index
check against a NAMED CONSTANT: never a string literal asserted against the
constant's prose, never a regex over wording, never a byte-size figure.
"""

from __future__ import annotations

from orchestrator.agents.roles import (
    BACKGROUND_WAIT_GUIDANCE,
    ROLES,
    TOOL_CALL_REJECTION_GUIDANCE,
)

# Membership is decided by a PROMPT-CONSTRUCTION detail — `role.prompt_spec
# is None`, i.e. the role's system_prompt is literal Python text rather than
# built from a `PromptSpec` — but that detail is NOT the reason a role needs
# this guidance. The need is capability-based: every role that can issue a
# tool call and hit `InputValidationError` needs to know how to read the
# rejection, and `reviewer_comprehensive` (which holds `Read`, `Grep`, `Glob`
# and `Bash(git:*)` same as `judge`) is no exception. It is left OUT of this
# set only because it is the sole role whose `system_prompt` is built by
# `_reviewer_role` from a `PromptSpec`, whose heuristics half a pinned prompt
# artifact may override at runtime — so a splice into the literal template
# could be silently dropped in exactly the sessions it was meant to protect.
# That makes the exclusion a DEFERRED COVERAGE GAP, not a judgment that the
# role is exempt on the merits: closing it means splicing into the frozen
# `_REVIEWER_CONTRACT_TEMPLATE` instead, which requires bumping
# `_REVIEWER_PROMPT_HARNESS_VERSION` (a bump that invalidates every pinned
# reviewer artifact, which is why it is not done here) and moving the role
# into this set — not leaving it exempt. Symbol names only, no `roles.py:NNN`
# line citations: they drift the moment the cited symbol moves and this very
# task shifted every one of them by 53 lines from the numbers first drafted.
_UNPINNED_PROMPT_ROLES = frozenset({
    'architect',
    'debugger',
    'deep_reviewer',
    'implementer',
    'judge',
    'merger',
    'simple_task',
    'steward',
})

# `BACKGROUND_WAIT_GUIDANCE` opens with its own `\n## ` heading, so for a role
# that carries no wait block (`judge`, today), "spliced up front" is exactly
# "its heading IS the first `##` heading in the prompt" — see
# `test_roles_wait_pattern.py`'s own `_MARKDOWN_HEADING` for the precedent.
# Reused here as a structural landmark, not any particular heading's text, so
# renaming a section in roles.py cannot silently turn this check into a no-op.
_MARKDOWN_HEADING = '\n## '


def test_tool_call_rejection_guidance_is_nonempty() -> None:
    """The mandated constant is a non-empty string.

    Mirrors `test_wait_pattern_guidance_is_nonempty`. NOT redundant with the
    containment tests added in step-3, though it reads that way: those
    assert `TOOL_CALL_REJECTION_GUIDANCE in ROLES[name].system_prompt` for
    each spliced role, and the empty string is a substring of every string —
    so every one of those assertions holds vacuously if this constant is
    ever emptied. This one-line assertion is the sole guard against the
    guidance being silently dropped in a prompt refactor.
    """
    assert TOOL_CALL_REJECTION_GUIDANCE.strip(), (
        'TOOL_CALL_REJECTION_GUIDANCE is empty. Every containment test added '
        'for this constant still passes when it is — the empty string is a '
        'substring of anything — so this assertion is the sole guard against '
        'the census-4273 guidance being silently dropped in a prompt refactor.'
    )


def test_unpinned_prompt_role_set_matches_prompt_spec_capability() -> None:
    """Drift tripwire: the hand-maintained role set still equals the derived one.

    Mirrors `test_background_capable_role_set_matches_bash_capability`. Passes
    on the current tree regardless of whether the guidance has been spliced
    anywhere yet — it pins the PREMISE (which roles have a literal prompt),
    not the splice itself.
    """
    derived = {name for name, role in ROLES.items() if role.prompt_spec is None}

    assert derived == _UNPINNED_PROMPT_ROLES, (
        'A role gained or lost a literal (non-PromptSpec) system_prompt, so '
        'the set of roles eligible for TOOL_CALL_REJECTION_GUIDANCE has '
        f'changed: gained={sorted(derived - _UNPINNED_PROMPT_ROLES)} '
        f'lost={sorted(_UNPINNED_PROMPT_ROLES - derived)}. A newly '
        'unpinned-prompt role must be added to _UNPINNED_PROMPT_ROLES AND given '
        'TOOL_CALL_REJECTION_GUIDANCE; if it is genuinely exempt, justify the '
        'exclusion in the comment above the set.'
    )


def test_unpinned_prompt_roles_carry_tool_call_rejection_guidance() -> None:
    """Every unpinned-prompt role's system_prompt embeds the guidance block."""
    offenders = sorted(
        name
        for name in _UNPINNED_PROMPT_ROLES
        if TOOL_CALL_REJECTION_GUIDANCE not in ROLES[name].system_prompt
    )

    assert offenders == [], (
        f'Roles missing TOOL_CALL_REJECTION_GUIDANCE from their system_prompt: '
        f'{offenders}. These roles hold Read and can hit a rejected tool call '
        'with no guidance on how to read the rejection before retrying.'
    )


def test_artifact_pinned_role_does_not_carry_guidance() -> None:
    """The negative half: a role outside `_UNPINNED_PROMPT_ROLES` must NOT carry it.

    The two tests above catch a role GAINING an unpinned prompt and a covered
    role LOSING the block. Neither catches an accidental splice into an
    excluded role. Mirrors `test_excluded_roles_do_not_carry_combined_guidance`
    for the wait block.

    `reviewer_comprehensive`'s exclusion (see the comment above
    `_UNPINNED_PROMPT_ROLES`) is a DEFERRED coverage gap, not a claim that the
    role is exempt on the merits — this test enforces the gap, it does not
    justify it.
    """
    offenders = sorted(
        name
        for name in ROLES
        if name not in _UNPINNED_PROMPT_ROLES
        and TOOL_CALL_REJECTION_GUIDANCE in ROLES[name].system_prompt
    )

    assert offenders == [], (
        f'Roles carrying TOOL_CALL_REJECTION_GUIDANCE without an unpinned '
        f'prompt: {offenders}. A PromptSpec-backed role may silently drop a '
        'splice at runtime — either remove it, or if the role genuinely '
        'gained an unpinned prompt, add it to _UNPINNED_PROMPT_ROLES. '
        "`reviewer_comprehensive`'s absence from that set is a deferred "
        'coverage gap (closing it needs a _REVIEWER_PROMPT_HARNESS_VERSION '
        'bump), not an exemption on the merits.'
    )


def test_guidance_appears_exactly_once_per_role() -> None:
    """No duplicate splice — the block is carried once, and only once, per role.

    Scoped to catching a stale duplicate splice left beside a new one, NOT to
    enforcing presence — that is
    `test_unpinned_prompt_roles_carry_tool_call_rejection_guidance`'s job. A
    role where the block is entirely absent is skipped here rather than
    flagged, so a role that has not yet received the splice fails exactly one
    test for that one root cause instead of two.
    """
    offenders = {}
    for name in sorted(_UNPINNED_PROMPT_ROLES):
        count = ROLES[name].system_prompt.count(TOOL_CALL_REJECTION_GUIDANCE)
        if count == 0:
            continue
        if count != 1:
            offenders[name] = count

    assert offenders == {}, (
        f'Roles carrying more than one copy of TOOL_CALL_REJECTION_GUIDANCE: '
        f'{offenders}. A stale duplicate splice was probably left beside a new '
        'one — delete the extra copy.'
    )


def test_guidance_placement_is_structural() -> None:
    """The block lands in the one structurally correct spot for each role.

    Two placement rules, both index comparisons against named constants —
    no literal text and no magic number:

    - For a role whose prompt contains `BACKGROUND_WAIT_GUIDANCE` (7 today),
      the new block must land immediately after it: `find(GUIDANCE) ==
      find(BACKGROUND_WAIT_GUIDANCE) + len(BACKGROUND_WAIT_GUIDANCE)`. It
      cannot go BEFORE the wait block instead:
      `test_roles_wait_pattern.py::test_combined_guidance_is_stated_up_front`
      requires the wait block's own heading to remain the prompt's FIRST
      `##` heading AND to land within its `_UP_FRONT_CHAR_BUDGET` of 1500
      chars, so splicing ahead of it would break both halves of that
      task-3607 invariant.
    - For a role whose prompt does NOT contain `BACKGROUND_WAIT_GUIDANCE`
      (`judge` today, derived at runtime rather than hardcoded), the new
      block's own `\\n## ` heading must be the prompt's first `##` heading —
      the same "up front" structural landmark the wait block uses.

    A role where the block is absent entirely is recorded as an offender
    rather than skipped, so this test can never pass vacuously on a role
    that dropped the splice.
    """
    offenders = {}
    for name in sorted(_UNPINNED_PROMPT_ROLES):
        prompt = ROLES[name].system_prompt
        idx = prompt.find(TOOL_CALL_REJECTION_GUIDANCE)
        if idx == -1:
            offenders[name] = {'offset': 'ABSENT'}
            continue

        wait_idx = prompt.find(BACKGROUND_WAIT_GUIDANCE)
        if wait_idx != -1:
            expected = wait_idx + len(BACKGROUND_WAIT_GUIDANCE)
            if idx != expected:
                offenders[name] = {'offset': idx, 'expected_after_wait_block': expected}
        else:
            first_heading = prompt.find(_MARKDOWN_HEADING)
            if idx != first_heading:
                offenders[name] = {'offset': idx, 'first_heading': first_heading}

    assert offenders == {}, (
        f'Roles placing TOOL_CALL_REJECTION_GUIDANCE incorrectly: {offenders}. '
        'It must immediately follow BACKGROUND_WAIT_GUIDANCE where that block '
        'is present, or be the first `##`-headed section where it is absent.'
    )
