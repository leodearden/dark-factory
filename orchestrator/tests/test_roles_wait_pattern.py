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

Turn prompts whose role already carries the full block inject the short
``WAIT_PATTERN_REMINDER`` pointer instead, so no session receives the same
~3.2 kB twice; the tests below pin both the pointer AND the precondition that
makes it sound (its receiving role really does carry the block).

Its real effect is on model behaviour and is not unit-testable, but silent
removal during a prompt refactor is a genuine safety regression — the repo
sanctions exactly this kind of "mandated token present in each role prompt"
guard. Each assertion is an existence / containment / ordering / count check
against a named constant, not a prose or regex pin — including the placement
check, which compares against the prompt's first ``##`` heading rather than any
particular heading's text.
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
    WAIT_PATTERN_REMINDER,
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

# Secondary, deliberately LOOSE bound on how much identity paragraph may
# precede the block. The real invariant is the structural one below (the
# guidance's own heading must be the prompt's FIRST ``##`` heading); this only
# catches a pathologically long preamble that is technically heading-free.
# Widest current margin is ``simple_task`` at 417 chars, so this leaves ~3x
# headroom and will not fire merely because someone added a sentence.
_UP_FRONT_CHAR_BUDGET = 1500

# ``BACKGROUND_WAIT_GUIDANCE`` opens with its own ``\n## `` heading, so
# "spliced between the identity paragraph and the role's first real section"
# is exactly "its heading IS the first ``##`` heading in the prompt". That is a
# structural property of the splice, not a prose pin: it keeps holding when
# every heading in the file is renamed. It replaces an earlier ``'## Escalation'``
# landmark that silently no-opped when the heading was not found.
_MARKDOWN_HEADING = '\n## '


def test_wait_pattern_guidance_is_nonempty() -> None:
    """The mandated constant is a non-empty string."""
    assert WAIT_PATTERN_GUIDANCE.strip()


def test_background_threshold_is_the_session_idle_bound_not_the_bash_cap() -> None:
    """The background-vs-foreground threshold tracks the WATCHDOG, not the Bash cap.

    ``shared.cli_invoke``'s working-regime watchdog kills a whole session once
    ``idle_elapsed >= max(working_idle_secs, timeout_seconds)`` with no NEW
    transcript turn, and a blocking foreground ``Bash`` call emits no turn while
    it runs. So the operative ceiling on a foreground command is the role's idle
    bound (stock 1800s = 30 min), NOT the harness's 3900000 ms (65 min) Bash cap.
    Guidance that used 3900000 as the decision threshold actively instructed
    agents to size a 45-minute foreground verify to a 2700000 timeout and lose
    the entire session at 30 minutes — the same lose-all-observations failure the
    guidance's own ``until``-loop bullet warns against. Pinned because the wrong
    number READS more authoritative (it is a real harness fact, just not the
    binding one) and will be "restored" by a future editor otherwise.
    """
    idle_secs = OrchestratorConfig.model_fields['timeouts'].default_factory().working_idle_secs  # type: ignore[misc]
    assert idle_secs == 1800, (
        f'timeouts.working_idle_secs default moved to {idle_secs}s. The wait-pattern '
        'guidance quotes 1800s / 30 minutes as the session idle bound and derives its '
        '~25-minute background threshold from it — update both constants in roles.py '
        'to stay under the new bound, then update this assertion.'
    )

    for name, text in (
        ('WAIT_PATTERN_GUIDANCE', WAIT_PATTERN_GUIDANCE),
        ('WAIT_PATTERN_REMINDER', WAIT_PATTERN_REMINDER),
    ):
        assert '25 minutes' in text, (
            f'{name} no longer states the ~25-minute background threshold. Sizing a '
            'long run to a large foreground `timeout` does not buy that time: the '
            'working-regime watchdog kills the session after 30 minutes with no new '
            'transcript turn. Do not replace this with the 3900000 harness cap.'
        )
        assert '30 minutes' in text, (
            f'{name} no longer states the 30-minute session idle bound that JUSTIFIES '
            'the threshold. A bare number with no stated reason gets "corrected" back '
            'to the 3900000 Bash cap by the next editor.'
        )
        assert 'BashOutput' in text, (
            f'{name} no longer names `BashOutput`. Polling is what makes a backgrounded '
            'long run safe — each poll is a new transcript turn and resets the idle '
            'clock — so the escape hatch must be named alongside the threshold.'
        )


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


@pytest.mark.parametrize(
    'name',
    ['WAIT_PATTERN_GUIDANCE', 'WAIT_PATTERN_REMINDER'],
)
def test_wait_pattern_constants_have_no_literal_braces(name: str) -> None:
    """No literal ``{``/``}`` — same invariant ``BACKGROUND_TASK_WARNING`` holds.

    Both constants reach an f-string: ``WAIT_PATTERN_REMINDER`` is interpolated
    into ``build_amender_prompt`` directly, and ``WAIT_PATTERN_GUIDANCE`` is
    concatenated into ``BACKGROUND_WAIT_GUIDANCE``, which is spliced into role
    prompts. A literal brace would raise at format time or silently mangle the
    rendered prompt.
    """
    value = {
        'WAIT_PATTERN_GUIDANCE': WAIT_PATTERN_GUIDANCE,
        'WAIT_PATTERN_REMINDER': WAIT_PATTERN_REMINDER,
    }[name]

    assert '{' not in value and '}' not in value, (
        f'{name} contains a literal brace. It reaches an f-string in '
        'briefing.py — braces must be removed or the prompt breaks at runtime.'
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

    "Up front" is asserted STRUCTURALLY: the block's own ``##`` heading must be
    the first ``##`` heading in the prompt, i.e. nothing but the opening
    identity paragraph precedes it. No heading text is pinned, so renaming any
    section in ``roles.py`` cannot silently turn this check into a no-op.
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
        first_heading = prompt.find(_MARKDOWN_HEADING)
        if idx != first_heading or idx >= _UP_FRONT_CHAR_BUDGET:
            offenders[name] = {'offset': idx, 'first_heading': first_heading}

    assert offenders == {}, (
        f'Roles stating the guidance too late: {offenders}. The block must be '
        "spliced immediately after the opening identity paragraph, so that its "
        'own `##` heading is the FIRST `##` heading in the prompt (offset == '
        f'first_heading) and lands within {_UP_FRONT_CHAR_BUDGET} chars. A '
        'first_heading BELOW offset means some other section now precedes it.'
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


def test_amender_reminder_cannot_dangle() -> None:
    """The pointer's precondition: its receiving role carries the full block.

    ``build_amender_prompt`` injects the short ``WAIT_PATTERN_REMINDER`` rather
    than restating all of ``BACKGROUND_WAIT_GUIDANCE``, which is only sound
    because the amender is invoked under IMPLEMENTER (``workflow.py``, the sole
    ``_invoke(IMPLEMENTER, ...)`` amendment call site) and IMPLEMENTER carries
    the full block up front. Pin that precondition here: if IMPLEMENTER ever
    loses the block, the pointer refers to rules the amender was never given.
    """
    assert BACKGROUND_WAIT_GUIDANCE in ROLES['implementer'].system_prompt, (
        "IMPLEMENTER lost BACKGROUND_WAIT_GUIDANCE, so build_amender_prompt's "
        'WAIT_PATTERN_REMINDER now points at rules the amender never receives. '
        'Either restore the block to IMPLEMENTER, or inline the full '
        'BACKGROUND_WAIT_GUIDANCE into the amender turn-prompt again.'
    )


@pytest.mark.asyncio
async def test_amender_prompt_reinforces_the_wait_rules(
    briefing: BriefingAssembler,
) -> None:
    """The built amender turn-prompt reinforces the rules at the failure site.

    The amender has no dedicated role — it runs under IMPLEMENTER's system
    prompt, which carries the combined block up front. This injection is the
    deliberate at-the-failure-site reinforcement: Reify-5164's amender
    backgrounded a 2700s verify and ended its turn at exactly the "Run
    verification before finishing" step the reminder sits under.

    It asserts the POINTER, not the whole block: interpolating
    ``BACKGROUND_WAIT_GUIDANCE`` verbatim here would hand the amender the same
    ~3.2 kB twice in one session (once via IMPLEMENTER's system prompt, once
    here) — the cross-prompt twin of the double-splice that
    ``test_combined_guidance_appears_exactly_once_per_role`` forbids within a
    single prompt. The no-duplication half is asserted explicitly so a future
    edit cannot quietly reintroduce it.

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

    assert WAIT_PATTERN_REMINDER in prompt, (
        'The amender turn-prompt does not carry WAIT_PATTERN_REMINDER. The '
        '"run verification before finishing" step is Reify-5164 exact failure '
        'site — it must restate the wait rules there, not rely solely on the '
        'system prompt.'
    )
    assert BACKGROUND_WAIT_GUIDANCE not in prompt, (
        'The amender turn-prompt inlines the full BACKGROUND_WAIT_GUIDANCE. '
        "IMPLEMENTER's system prompt already carries it up front, so this "
        'duplicates ~3.2 kB verbatim in every amendment pass. Use the short '
        'WAIT_PATTERN_REMINDER pointer instead.'
    )
