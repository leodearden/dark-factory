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
from orchestrator.agents.roles import BACKGROUND_TASK_WARNING, ROLES
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
async def test_amender_prompt_carries_warning(briefing: BriefingAssembler) -> None:
    """The built amender turn-prompt reinforces the warning at the failure site.

    ``_get_memory_context`` is patched to a stub so no real fused-memory HTTP
    call fires (mirrors the resume golden test)."""
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
    assert BACKGROUND_TASK_WARNING in prompt
