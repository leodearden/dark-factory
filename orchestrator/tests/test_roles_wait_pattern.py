"""Anchor/contract test for the census-R3 wait-pattern guidance (task 3607).

Sibling of ``test_roles_background_warning.py`` (task 2761). That task told
dispatched agents what NOT to do — never end your turn with a backgrounded
command still pending. It never told them what to do INSTEAD, so the census
(2026-08-02 §1.1) caught session after session improvising a wait: foreground
``sleep`` chains that the harness anti-polling guard blocks, back-to-back
``Read``/``Bash`` calls fired as an ad-hoc delay, hand-rolled ``until`` loops
that get SIGTERMed at the Bash timeout and lose everything they observed.

The two rules are mandated to travel TOGETHER — an agent given only 2761's
prohibition trades the abandoned-background footgun for the busy-loop one, and
an agent given only the wait pattern loses the don't-end-your-turn rule. This
file pins that STRUCTURALLY rather than editorially: ``BACKGROUND_WAIT_GUIDANCE``
is the single splice unit, and it is asserted to contain both halves, so no
future prompt refactor can splice one without the other.

Its real effect is on model behaviour and is not unit-testable, but silent
removal during a prompt refactor is a genuine safety regression — the repo
sanctions exactly this kind of "mandated token present in each role prompt"
guard. Each assertion is a one-line existence / containment / ordering check
against a named constant, not a prose or regex pin.
"""

from __future__ import annotations

from orchestrator.agents.roles import (
    BACKGROUND_TASK_WARNING,
    BACKGROUND_WAIT_GUIDANCE,
    ROLES,
    WAIT_PATTERN_GUIDANCE,
)


def test_wait_pattern_guidance_is_nonempty() -> None:
    """The mandated constant is a non-empty string."""
    assert WAIT_PATTERN_GUIDANCE.strip()


def test_combined_guidance_composes_both_rules() -> None:
    """``BACKGROUND_WAIT_GUIDANCE`` carries BOTH halves of the contract.

    This is the structural enforcement of the task's MUST-COMPOSE mandate: the
    splice unit cannot carry the wait-pattern rule without also carrying task
    2761's don't-end-your-turn-on-a-background-command rule.
    """
    assert BACKGROUND_TASK_WARNING in BACKGROUND_WAIT_GUIDANCE, (
        "BACKGROUND_WAIT_GUIDANCE no longer contains task 2761's "
        'BACKGROUND_TASK_WARNING. The two rules must NEVER be spliced apart: an '
        'agent told only how to wait, without the prohibition on ending its turn '
        'with a background command pending, simply trades one footgun for the other.'
    )
    assert WAIT_PATTERN_GUIDANCE in BACKGROUND_WAIT_GUIDANCE, (
        'BACKGROUND_WAIT_GUIDANCE no longer contains WAIT_PATTERN_GUIDANCE. The '
        'two rules must NEVER be spliced apart: an agent told only what not to do '
        '(task 2761), with no sanctioned wait pattern, improvises a busy-loop or a '
        'blocked sleep chain instead — the exact census-R3 defect this constant fixes.'
    )


def test_wait_pattern_guidance_has_no_literal_braces() -> None:
    """No literal ``{``/``}`` — same invariant ``BACKGROUND_TASK_WARNING`` holds.

    The constant is interpolated into ``build_amender_prompt``'s f-string, so a
    literal brace would raise at import/format time or silently mangle the
    rendered prompt.
    """
    assert '{' not in WAIT_PATTERN_GUIDANCE and '}' not in WAIT_PATTERN_GUIDANCE, (
        'WAIT_PATTERN_GUIDANCE contains a literal brace. It is interpolated into '
        "build_amender_prompt's f-string (briefing.py) — braces must be removed or "
        'the prompt breaks at runtime.'
    )
