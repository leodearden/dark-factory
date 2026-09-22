"""Anchor/contract test for the compound-command rewrite guidance (task 5683).

Fifth sibling of `test_roles_wait_pattern.py` (task 3607),
`test_roles_tool_call_rejection.py` (tasks 4273/4578),
`test_roles_error_remedy_hint.py` (task 4964) and
`test_roles_grep_lookaround.py` (task 5331) — same shape, same provenance
kind: a legibility-census finding (`metadata.source: legibility_census`) about
a wasted agent turn, turned into a named prompt constant spliced into a
machine-derived set of roles. `test_roles_grep_lookaround.py` is the template
this module follows most closely.

PROVENANCE, BY POINTER ONLY: census 2026-09-20 section 1.1, codebook candidate
`entry-cand-20260918-19`, task 5683. The mechanism, the probe table with the
commands that produced it, the host-hook identification, the out-of-repo-cause
carve-out, the operand-presence correction and the discrimination against the
three neighbouring blocks are recorded ONCE, next to the constant they
constrain, in the comment block above
`orchestrator/src/orchestrator/agents/roles.py::COMPOUND_COMMAND_REWRITE_GUIDANCE`.
This module points there rather than carrying a second copy that would have to
be re-corrected in step with the first (SPOT, heuristic 11).

Its real effect is on model behaviour and is not unit-testable, but silent
removal during a prompt refactor is a genuine regression — the repo sanctions
exactly this kind of "mandated token present in each role prompt" guard. Read
"token" STRICTLY as a named constant. THE STANDING RULE: every assertion in
this file is an existence / containment / count / index check against a NAMED
CONSTANT — never a string literal asserted against the constant's prose, never
a regex over wording, never a byte-size figure. A prose pin has no correctness
content in either direction — it passes on prose reworded to say the opposite
and fails on a legitimate tightening — so it only taxes future prompt edits.

The mechanical half of that shape — the offender-collection loops, the
derived-vs-hardcoded role-set comparison, the count and index bookkeeping —
lives in `_role_splice_contract.py` (task 4405). That module's docstring is the
authoritative account of what the shared shape does and does not absorb; it is
NOT restated here. The tests below stay one thin function per invariant, each
delegating its body to the helper while keeping its own docstring and its own
remediation prose.

The four hard constraints `roles.py` documents above `_GREP_ENGINE_LIMITS` bind
this module too. Two of them are visible here: no `mcp__<family>__<name>` token
in any comment or string, and every test cited as
`path/to/test_module.py::lowercase_function` rather than as a `Test` +
CamelCase identifier, which
`orchestrator/tests/test_cited_test_class_drift.py::test_every_cited_test_class_resolves`
statically requires to resolve to a real class.
"""

from __future__ import annotations

from _role_splice_contract import (
    MARKDOWN_HEADING,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.roles import COMPOUND_COMMAND_REWRITE_GUIDANCE


def test_guidance_is_nonempty():
    """The guidance constant is a non-empty string.

    NOT redundant with the containment tests, though it reads that way: those
    assert `CONSTANT in ROLES[role].system_prompt`, and the empty string is a
    substring of every string — so every one of those assertions holds
    vacuously against an emptied constant. This is the sole guard against the
    guidance being silently dropped in a prompt refactor.
    """
    assert_nonempty(
        'COMPOUND_COMMAND_REWRITE_GUIDANCE',
        COMPOUND_COMMAND_REWRITE_GUIDANCE,
        remedy=(
            'Restore the census-5683 guidance: this assertion is the sole guard '
            'against it being silently dropped, because every containment check '
            'in this module still passes when the constant is empty.'
        ),
    )


def test_guidance_opens_its_own_section():
    """The constant opens with its own ``\\n## `` heading.

    Structural, not a wording pin: the heading TEXT is never asserted, only
    that the constant STARTS with `MARKDOWN_HEADING`. Without this the block —
    which is spliced at the TAIL of the chain, behind
    `GREP_LOOKAROUND_GUIDANCE` — would read as an unheaded continuation of that
    block's last paragraph, a different claim than the one it makes.

    It is also the precondition for
    `orchestrator/tests/_role_splice_contract.py::assert_placement`'s up-front
    arm, which compares the constant's offset against the prompt's first `##`
    heading: a block that does not begin with a heading can never satisfy that
    comparison.
    """
    assert COMPOUND_COMMAND_REWRITE_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'COMPOUND_COMMAND_REWRITE_GUIDANCE does not start with MARKDOWN_HEADING, '
        'so it reads as an unheaded continuation of whatever block precedes it '
        'rather than as its own section. Give it a leading blank line and a '
        '`## ` heading.'
    )


def test_guidance_has_no_literal_braces():
    """No literal ``{``/``}`` in the constant.

    Role prompts are deliberately not f-strings — this constant reaches a
    prompt by plain `+` concatenation — but it is held brace-free defensively
    so it stays interpolation-safe if a future splice site needs it.
    `CODE_QUALITY_GUIDANCE` is the live example of why that matters: it reaches
    a `str.format()` template and is brace-free BY CONTRACT.
    """
    assert_brace_free(
        'COMPOUND_COMMAND_REWRITE_GUIDANCE',
        COMPOUND_COMMAND_REWRITE_GUIDANCE,
        remedy=(
            'A literal brace raises at format time or mangles the rendered '
            'prompt at an interpolating splice site. Spell the example command '
            'without one.'
        ),
    )
