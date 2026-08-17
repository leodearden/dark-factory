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

from orchestrator.agents.roles import TOOL_CALL_REJECTION_GUIDANCE


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


def test_tool_call_rejection_guidance_has_no_literal_braces() -> None:
    """No literal `{`/`}` — same invariant `WAIT_PATTERN_GUIDANCE` holds.

    Defensive, NOT load-bearing: this constant reaches role prompts only by
    plain `+` concatenation, which is brace-safe by construction — role
    prompts are deliberately NOT f-strings (see the `MANDATED_STAGING_COMMAND`
    note in roles.py) precisely because they contain literal braces. Held
    brace-free anyway so it stays interpolation-safe if a future splice site
    needs it, exactly as `WAIT_PATTERN_GUIDANCE` is.
    """
    assert '{' not in TOOL_CALL_REJECTION_GUIDANCE and '}' not in TOOL_CALL_REJECTION_GUIDANCE, (
        'TOOL_CALL_REJECTION_GUIDANCE contains a literal brace. It is held '
        'brace-free so it stays safe at any future interpolating splice site '
        '— remove the brace or a future interpolation could break at runtime.'
    )
