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
    SpliceContract,
    assert_brace_free,
    assert_nonempty,
)

from orchestrator.agents.roles import (
    COMPOUND_COMMAND_REWRITE_GUIDANCE,
    GREP_LOOKAROUND_GUIDANCE,
)


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


#: Roles holding UNQUALIFIED `Bash`, so a multi-line `python3 -c` script and
#: the heredoc escape are both things they can actually run.
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

#: `judge` is deliberately absent: its grant is `Bash(git:*)`, so it can run
#: neither `python3 -c` nor the heredoc escape, and the whole block is
#: inapplicable there rather than merely needing a narrower recourse — which is
#: why this constant has ONE variant where `GREP_LOOKAROUND_GUIDANCE` has two.
#: `reviewer_comprehensive` is absent for a different reason: it is the sole
#: `PromptSpec`-built role, whose pinned artifact can override the literal
#: template at runtime, so a splice there could be silently dropped in exactly
#: the sessions it protects. That is a DEFERRED COVERAGE GAP, not an exemption
#: on the merits — the same reading
#: `orchestrator/tests/test_roles_grep_lookaround.py::test_no_role_outside_the_set_carries_that_variant`
#: already documents, and closing it means splicing into the frozen reviewer
#: template and bumping `_REVIEWER_PROMPT_HARNESS_VERSION`.
_CONTRACT = SpliceContract(
    constant_name='COMPOUND_COMMAND_REWRITE_GUIDANCE',
    constant=COMPOUND_COMMAND_REWRITE_GUIDANCE,
    roles=_BASH_CAPABLE_UNPINNED_ROLES,
    role_set_name='_BASH_CAPABLE_UNPINNED_ROLES',
    capability=lambda role: role.prompt_spec is None and 'Bash' in role.allowed_tools,
    capability_description='a literal system_prompt and unqualified `Bash`',
)


def test_role_set_matches_its_bash_capability():
    """Drift tripwire: the hand-listed set still equals the derived one.

    This is what makes the carrier set self-maintaining rather than a snapshot.
    A role gaining or losing unqualified `Bash`, converting to a `PromptSpec`,
    or a new role arriving, diverges the derived set from the hand-maintained
    one and the failure names the set to edit.

    Pinned against the CAPABILITY rather than against a sibling constant's
    carriers. Coupling to a sibling would make this test report a defect
    whenever that sibling's own set legitimately changed, and would go silently
    vacuous if the sibling were ever removed; the capability is the thing that
    actually decides who needs this block.

    Pins the PREMISE (which roles can run the script forms this block talks
    about), not the splice, so it passes regardless of whether the guidance has
    been spliced anywhere yet.
    """
    _CONTRACT.assert_role_set_matches_capability(
        remedy=(
            "A role's `Bash` grant changed, a role was added, or a role became "
            'PromptSpec-backed. Add or remove it in _BASH_CAPABLE_UNPINNED_ROLES '
            'to match — a role that cannot run an unqualified `Bash` command '
            'cannot use either the offending form or the escape, so the block is '
            'dead weight there.'
        ),
    )


def test_every_role_in_the_set_carries_the_guidance():
    """Every role in the set embeds the guidance in its system_prompt."""
    _CONTRACT.assert_every_role_carries(
        remedy=(
            'These roles compose compound `Bash` commands routinely and would '
            'otherwise meet the rewrite by failure — reading an IndentationError '
            'that accuses their own script, with nothing reporting that the '
            'command was altered before it reached the shell.'
        ),
    )


def test_no_role_outside_the_set_carries_the_guidance():
    """The negative half: a role outside the set must NOT carry the guidance.

    The two tests above catch a role gaining the capability and a covered role
    losing the block. Neither catches a splice landing where the block does not
    belong, which ships silently and is paid on every invocation of that role.
    """
    _CONTRACT.assert_no_other_role_carries(
        remedy=(
            '`judge` takes NO variant of this block: its grant is `Bash(git:*)`, '
            'so it can run neither the multi-line `python3 -c` form the block '
            'warns about nor the heredoc escape it prescribes — a splice there is '
            'dead weight, not a fix, and the right response is to remove it '
            'rather than to add `judge` to the set. `reviewer_comprehensive` is '
            'out for the separate PromptSpec reason recorded above '
            '_BASH_CAPABLE_UNPINNED_ROLES; closing that gap needs a '
            '_REVIEWER_PROMPT_HARNESS_VERSION bump, not a splice.'
        ),
    )


def test_guidance_appears_exactly_once_per_role():
    """No duplicate splice — the guidance is carried once, and only once.

    Scoped to catching a stale duplicate left beside a new one, NOT to
    enforcing presence: that is
    `test_every_role_in_the_set_carries_the_guidance`'s job.
    """
    # `absent_ok=True`: a role that has not yet received the splice then fails
    # exactly ONE test for that one root cause — the containment one, whose job
    # presence is — instead of two.
    _CONTRACT.assert_spliced_exactly_once(
        absent_ok=True,
        remedy=(
            'A stale duplicate splice was probably left beside a new one — '
            'delete the extra copy.'
        ),
    )


def test_placement_is_structural():
    """The guidance lands immediately after `GREP_LOOKAROUND_GUIDANCE`.

    An index comparison against a named constant — no literal text, no magic
    number. The TAIL of the chain is the ONLY position available, not a
    preference, because three upstream invariants pin everything ahead of it:

    - `orchestrator/tests/test_roles_wait_pattern.py::test_combined_guidance_is_stated_up_front`
      requires `BACKGROUND_WAIT_GUIDANCE`'s heading to remain the prompt's
      FIRST `##` heading, within its char budget.
    - `orchestrator/tests/test_roles_tool_call_rejection.py::test_guidance_placement_is_structural`
      requires `TOOL_CALL_REJECTION_GUIDANCE`'s heading to be `judge`'s first
      `##`.
    - `orchestrator/tests/test_roles_grep_lookaround.py::test_variant_placement_is_structural`
      requires the grep block to abut `ERROR_REMEDY_HINT_GUIDANCE`.

    And one reason that is not a test at all: `_GREP_ENGINE_LIMITS`'s prose
    draws its own discrimination by pointing at "the section just above", which
    holds only while it abuts `ERROR_REMEDY_HINT_GUIDANCE`. Splicing this block
    between them would leave every test above green while silently redirecting
    that pointer at this block. Appending behind the grep block satisfies all
    four with no per-role branching.

    No `char_budget` is passed: that secondary bound belongs to the wait
    block's own up-front invariant, not to a block spliced behind it.

    The FOLLOWS arm is non-vacuous here because every role in the set carries
    the predecessor, and a role where this block is absent entirely is recorded
    as an offender rather than skipped — so this can never pass by default.
    """
    _CONTRACT.assert_placement(
        follows=GREP_LOOKAROUND_GUIDANCE,
        follows_name='GREP_LOOKAROUND_GUIDANCE',
        remedy=(
            'It cannot be moved ahead of the wait block, of '
            'TOOL_CALL_REJECTION_GUIDANCE, or between ERROR_REMEDY_HINT_GUIDANCE '
            'and the grep block to fix this — the first two are pinned as their '
            "prompts' FIRST `##` heading, and the third abutment is both pinned "
            'and depended on by _GREP_ENGINE_LIMITS\'s "the section just above". '
            'Re-append it behind GREP_LOOKAROUND_GUIDANCE instead.'
        ),
    )
