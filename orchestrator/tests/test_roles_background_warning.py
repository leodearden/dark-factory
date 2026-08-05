"""Anchor/contract test for the Layer-1 background-task guard (task 2761).

Models test_roles_staging_command.py: assert the mandated
``BACKGROUND_TASK_WARNING`` constant is present verbatim in every assembled
prompt that must carry it — the ``implementer`` and ``debugger`` role system
prompts, and the built amender turn-prompt (the amender has no dedicated role,
so it inherits IMPLEMENTER's system prompt; the turn-prompt injection reinforces
the guard at the exact failure site — its "Run verification before finishing"
step is where Reify-5164's amender backgrounded a 2700s verify and ended its
turn).

Its real effect is on model behaviour and is not unit-testable, but silent
removal during a prompt refactor is a genuine safety regression — the repo
sanctions exactly this kind of "mandated token present in each role prompt"
guard.  Each assertion is a one-line existence check, not a prose pin.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.roles import (
    BACKGROUND_TASK_WARNING,
    ROLES,
    WAIT_PATTERN_REMINDER,
)
from orchestrator.config import GitConfig, OrchestratorConfig


def test_background_warning_is_nonempty() -> None:
    """The mandated constant is a non-empty string."""
    assert BACKGROUND_TASK_WARNING.strip()


def test_implementer_system_prompt_carries_warning() -> None:
    """IMPLEMENTER.system_prompt embeds the warning verbatim (also covers the
    amender, which is invoked under the IMPLEMENTER role)."""
    assert BACKGROUND_TASK_WARNING in ROLES['implementer'].system_prompt


def test_debugger_system_prompt_carries_warning() -> None:
    """DEBUGGER.system_prompt embeds the warning verbatim."""
    assert BACKGROUND_TASK_WARNING in ROLES['debugger'].system_prompt


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    """Minimal BriefingAssembler over a stub OrchestratorConfig (no I/O),
    mirroring the fixture in test_briefing.py."""
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
async def test_amender_prompt_reinforces_warning_at_failure_site(
    briefing: BriefingAssembler,
) -> None:
    """The built amender turn-prompt reinforces the warning at the failure site.

    2761's guarantee is that the amender SESSION carries this rule. It still
    does, via two assertions here: IMPLEMENTER's system prompt (which the
    amender runs under) carries the warning verbatim, and the turn prompt
    restates it at the "Run verification before finishing" step.

    Task 3607 changed the turn-prompt injection from the verbatim warning to
    the short ``WAIT_PATTERN_REMINDER`` pointer, because IMPLEMENTER now carries
    the full block UP FRONT rather than buried at the tail — so inlining it
    again here would hand the amender the same text twice in one session. The
    reminder is held to a minimum size by ``test_roles_wait_pattern.py``, so the
    reinforcement stays substantive rather than decaying to a bare "see above".

    ``_get_memory_context`` is patched to a stub so no real fused-memory HTTP
    call fires (mirrors the resume golden test)."""
    assert BACKGROUND_TASK_WARNING in ROLES['implementer'].system_prompt, (
        'The amender runs under IMPLEMENTER; its system prompt must carry the '
        "2761 warning, since the turn prompt now only points back at it."
    )

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
        'The amender turn-prompt no longer reinforces the wait rules at the '
        '"Run verification before finishing" step — Reify-5164 exact failure '
        'site. Restore WAIT_PATTERN_REMINDER (or the full '
        'BACKGROUND_WAIT_GUIDANCE) there.'
    )
