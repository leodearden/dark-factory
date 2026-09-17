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
from _role_splice_contract import MARKDOWN_HEADING, assert_brace_free, assert_nonempty

from orchestrator.agents.roles import (
    _GREP_ENGINE_LIMITS,
    _GREP_PCRE_BASH_RECOURSE,
    _GREP_PCRE_READ_ONLY_RECOURSE,
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
