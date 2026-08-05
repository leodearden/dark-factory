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

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.roles import (
    BACKGROUND_TASK_WARNING,
    BACKGROUND_WAIT_GUIDANCE,
    ROLES,
    WAIT_PATTERN_GUIDANCE,
)
from orchestrator.config import GitConfig, OrchestratorConfig

# A role can encounter background work iff it holds the UNQUALIFIED ``'Bash'``
# tool — i.e. it can launch a build, a full test suite, or a long verification
# run. ``reviewer_comprehensive`` and ``judge`` hold only ``'Bash(git:*)'``
# (read-only git, no long-running command is reachable) and are deliberately
# excluded: the guidance would be ~2.6 kB of dead weight in every one of their
# sessions.
_BACKGROUND_CAPABLE_ROLES = frozenset({
    'architect',
    'implementer',
    'debugger',
    'merger',
    'steward',
    'deep_reviewer',
    'simple_task',
})

# "Up front" == within the first kilobyte of the system prompt. Every target
# role's opening identity paragraph is under ~450 chars (``simple_task``'s
# ~6-line opener is the longest at ~420), so this pins "immediately after the
# identity paragraph, before the first ``##`` heading" without pinning any
# prose. The census-R3 defect was precisely that task 2761's guard sat at the
# TAIL of an 11 kB prompt, behind the escalation/memory/follow-up blocks.
_UP_FRONT_CHAR_BUDGET = 1000

# Every target role's system_prompt ends with ``_ESCALATION_INSTRUCTIONS``, so
# this heading is a stable ordering landmark for "ahead of the appended tail
# blocks" that does not depend on any block's own wording.
_TAIL_BLOCK_LANDMARK = '## Escalation'


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


def test_background_capable_role_set_matches_bash_capability() -> None:
    """Drift tripwire: the hand-maintained role set still equals the derived one.

    Mirrors ``test_roles_operator_tools.py``'s iterate-every-role invariant. The
    match is on the exact string ``'Bash'``, so a ``'Bash(git:*)'`` grant does
    NOT qualify.
    """
    derived = {name for name, role in ROLES.items() if 'Bash' in role.allowed_tools}

    assert derived == _BACKGROUND_CAPABLE_ROLES, (
        'A role gained or lost the unqualified `Bash` tool, so the set of roles '
        'that can encounter background work has changed: '
        f'gained={sorted(derived - _BACKGROUND_CAPABLE_ROLES)} '
        f'lost={sorted(_BACKGROUND_CAPABLE_ROLES - derived)}. '
        'A newly Bash-capable role must be added to _BACKGROUND_CAPABLE_ROLES AND '
        'given BACKGROUND_WAIT_GUIDANCE up front in its system_prompt; if it is '
        'genuinely exempt, justify the exclusion in the comment above the set.'
    )


def test_background_capable_roles_carry_combined_guidance() -> None:
    """Every Bash-capable role's system_prompt embeds the combined block."""
    offenders = sorted(
        name
        for name in _BACKGROUND_CAPABLE_ROLES
        if BACKGROUND_WAIT_GUIDANCE not in ROLES[name].system_prompt
    )

    assert offenders == [], (
        f'Roles missing BACKGROUND_WAIT_GUIDANCE from their system_prompt: {offenders}. '
        'These roles can launch a build or a full test suite, so they will hit '
        'background work with no sanctioned wait pattern to reach for.'
    )


def test_combined_guidance_appears_exactly_once_per_role() -> None:
    """No duplicate splice — the block is carried once, and only once.

    Catches a stale tail splice left behind beside the new up-front one, which
    would silently double ~2.6 kB of prompt in every session of that role. The
    ``BACKGROUND_TASK_WARNING`` count is checked separately because the combined
    unit CONTAINS it: a leftover bare ``+ BACKGROUND_TASK_WARNING`` tail would
    push that count to 2 while the combined count stayed at 1.
    """
    offenders = {}
    for name in sorted(_BACKGROUND_CAPABLE_ROLES):
        prompt = ROLES[name].system_prompt
        combined = prompt.count(BACKGROUND_WAIT_GUIDANCE)
        warning = prompt.count(BACKGROUND_TASK_WARNING)
        if combined != 1 or warning != 1:
            offenders[name] = {
                'BACKGROUND_WAIT_GUIDANCE': combined,
                'BACKGROUND_TASK_WARNING': warning,
            }

    assert offenders == {}, (
        f'Roles whose guidance splice count is not exactly 1: {offenders}. '
        'Expected 1 of each. A count of 2 for BACKGROUND_TASK_WARNING means a '
        'stale tail `+ BACKGROUND_TASK_WARNING` survives beside the up-front '
        'BACKGROUND_WAIT_GUIDANCE — delete the tail, it is now redundant.'
    )


def test_combined_guidance_is_stated_up_front() -> None:
    """The block is stated UP FRONT, not buried behind the appended tail blocks.

    This is the census-R3 defect itself: task 2761's guard was appended at the
    very end of an 11 kB prompt (and ``_FOLLOWUP_FILING_INSTRUCTIONS`` after it),
    where it does not read as an operating rule.
    """
    offenders = {}
    for name in sorted(_BACKGROUND_CAPABLE_ROLES):
        prompt = ROLES[name].system_prompt
        idx = prompt.find(BACKGROUND_WAIT_GUIDANCE)
        if idx == -1:
            # Absent entirely -- record rather than skip, so this test can never
            # pass vacuously on a role that dropped the block.
            offenders[name] = {'offset': 'ABSENT'}
            continue
        landmark = prompt.find(_TAIL_BLOCK_LANDMARK)
        if idx >= _UP_FRONT_CHAR_BUDGET or (landmark != -1 and idx >= landmark):
            offenders[name] = {'offset': idx, _TAIL_BLOCK_LANDMARK: landmark}

    assert offenders == {}, (
        f'Roles stating the guidance too late: {offenders} '
        f'(budget={_UP_FRONT_CHAR_BUDGET} chars, and it must precede '
        f'{_TAIL_BLOCK_LANDMARK!r}). Splice BACKGROUND_WAIT_GUIDANCE immediately '
        'after the opening identity paragraph, before the first `##` heading.'
    )


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    """Minimal BriefingAssembler over a stub OrchestratorConfig (no I/O),
    mirroring the fixture in test_roles_background_warning.py."""
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    return BriefingAssembler(config)


@pytest.mark.asyncio
async def test_amender_prompt_carries_combined_guidance(
    briefing: BriefingAssembler,
) -> None:
    """The built amender turn-prompt reinforces BOTH rules, not just 2761's half.

    The amender has no dedicated role — it runs under IMPLEMENTER's system
    prompt, which now carries the block up front. The turn-prompt injection is
    the deliberate at-the-failure-site reinforcement: Reify-5164's amender
    backgrounded a 2700s verify and ended its turn at exactly the "Run
    verification before finishing" step this block sits under. An amender told
    only "don't end your turn" and not "here is how to wait" is the case this
    task exists to close.

    ``_get_memory_context`` is patched to a stub so no real fused-memory HTTP
    call fires (mirrors the resume golden test).
    """
    with patch.object(
        BriefingAssembler, '_get_memory_context', return_value='# Context\n\n_stub_',
    ):
        prompt = await briefing.build_amender_prompt(
            plan={'task_id': '1', 'title': 't', 'analysis': 'a'},
            iteration_log=[],
            suggestions=[],
            locked_modules=['x'],
            task_id='1',
        )

    assert BACKGROUND_WAIT_GUIDANCE in prompt, (
        'The amender turn-prompt does not carry BACKGROUND_WAIT_GUIDANCE. '
        'build_amender_prompt must interpolate the combined unit, not '
        'BACKGROUND_TASK_WARNING alone — the wait-pattern half is what tells the '
        'amender how to run a long verify without abandoning it.'
    )
