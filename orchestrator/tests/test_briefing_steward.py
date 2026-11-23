"""Unit tests for build_steward_initial_prompt.

Scoped to the prompt builder — no MCP, no workflow. Patches
``BriefingAssembler._get_memory_context`` so the tests don't hit the
fused-memory HTTP endpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.landed_sha_searches import LandedShaSearches, already_merged_step
from orchestrator.agents.steward_digest import (
    EarlierAttempts,
    HandledL0,
    render_earlier_attempts,
    render_handled_digest,
)
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.workflow_types import StewardResolved

_TASK = {
    'id': 'df_task_42',
    'title': 'Add widget',
    'description': 'Simple widget addition.',
    'status': 'blocked',
    'dependencies': [],
    'priority': 'medium',
    'metadata': {},
}
_ESCALATION = {
    'id': 'esc-42-1',
    'category': 'task_failure',
    'summary': 'Execution iterations exhausted',
    'detail': 'Agent ran out of iterations.',
    'severity': 'blocking',
}


_NO_EARLIER_ATTEMPTS = EarlierAttempts()


def _fixture_git() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


def _assembler(tmp_path: Path, git: GitConfig) -> BriefingAssembler:
    return BriefingAssembler(OrchestratorConfig(project_root=tmp_path, git=git))


async def _initial_prompt(
    briefing: BriefingAssembler,
    worktree: Path,
    handled: tuple[HandledL0, ...] = (),
    pending_escalations: list[dict] | None = None,
    earlier_attempts: EarlierAttempts = _NO_EARLIER_ATTEMPTS,
) -> str:
    with patch.object(
        BriefingAssembler,
        '_get_memory_context',
        return_value='# Context\n\n_stub_',
    ):
        return await briefing.build_steward_initial_prompt(
            task=_TASK,
            escalation=_ESCALATION,
            pending_escalations=pending_escalations or [],
            worktree=worktree,
            handled=handled,
            earlier_attempts=earlier_attempts,
        )


def _steward_owned_tail(prompt: str) -> str:
    return prompt[prompt.index('\n# Escalation\n\n') + 1:]


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    return _assembler(tmp_path, _fixture_git())


@pytest.mark.asyncio
class TestBuildStewardInitialPrompt:
    async def test_steward_initial_prompt_instructs_done_provenance(
        self, briefing, tmp_path
    ):
        """Steward's initial prompt must instruct it to pass done_provenance.

        Phase 2 enforcement: when the steward marks a task done (already-on-main
        case), it must supply done_provenance={'commit': <sha>} or
        {'note': <text>}. Minimal substring check per design decision 6 in the
        plan — prose wording may evolve freely as long as the token is present.
        """
        prompt = await _initial_prompt(briefing, tmp_path)

        assert 'done_provenance' in prompt, (
            'Steward prompt must reference done_provenance so the LLM agent '
            'knows to pass it when calling set_task_status(done) after the '
            'enforcement flag is flipped.'
        )

    @pytest.mark.parametrize(
        'git',
        [
            pytest.param(_fixture_git(), id='fixture-git'),
            pytest.param(GitConfig(main_branch='trunk', branch_prefix='feat/'), id='trunk-feat'),
        ],
    )
    async def test_already_merged_step_is_rendered_for_this_task_and_project(
        self, tmp_path: Path, git: GitConfig,
    ) -> None:
        briefing = _assembler(tmp_path, git)

        prompt = await _initial_prompt(briefing, tmp_path)

        expected = already_merged_step(LandedShaSearches.for_task(str(_TASK['id']), git))
        assert f'\n2. {expected}\n3. ' in prompt


_DIGEST_HEADING = '## Escalations This Steward Already Handled'
_HANDLED = (HandledL0('esc-42-0', 'task_failure', StewardResolved(resolution_text='fixed the import')),)
_OTHER_PENDING = {'id': 'esc-42-2', 'category': 'review_suggestion', 'summary': 'Rename the helper'}


@pytest.mark.asyncio
class TestHandledDigest:
    async def test_an_empty_digest_leaves_the_steward_owned_tail_byte_identical(
        self, briefing: BriefingAssembler, tmp_path: Path,
    ) -> None:
        prompt = await _initial_prompt(briefing, tmp_path)

        already_merged = already_merged_step(
            LandedShaSearches.for_task(str(_TASK['id']), _fixture_git()),
        )
        expected = (
            '# Escalation\n'
            '\n'
            '- **ID:** `esc-42-1`\n'
            '- **Category:** task_failure\n'
            '- **Severity:** blocking\n'
            '- **Summary:** Execution iterations exhausted\n'
            '- **Detail:** Agent ran out of iterations.\n'
            '\n'
            '# Parameters\n'
            '\n'
            f'- **project_id:** `{briefing.project_id}`\n'
            f'- **project_root:** `{tmp_path}`\n'
            f'- **worktree:** `{tmp_path}`\n'
            '\n'
            '# Action\n'
            '\n'
            '1. Understand the escalation and the task context.\n'
            f'2. {already_merged}\n'
            '3. Read the relevant code.\n'
            '4. Handle the escalation — fix the issue, or triage suggestions.\n'
            '5. Run tests to verify any code changes.\n'
            '6. Call `resolve_issue` with a summary of what you did.\n'
        )
        assert _steward_owned_tail(prompt) == expected
        assert _DIGEST_HEADING not in prompt

    async def test_the_digest_sits_after_the_pending_block_and_before_parameters(
        self, briefing: BriefingAssembler, tmp_path: Path,
    ) -> None:
        prompt = await _initial_prompt(
            briefing, tmp_path, handled=_HANDLED, pending_escalations=[_OTHER_PENDING],
        )

        digest = render_handled_digest(_HANDLED)
        assert digest in prompt
        at = prompt.index(digest)
        assert prompt.index('\n# Escalation\n\n') < at
        assert prompt.index('## Other Pending Escalations') < at
        assert at + len(digest) <= prompt.index('# Parameters')

    async def test_the_digest_follows_the_escalation_block_with_no_pending(
        self, briefing: BriefingAssembler, tmp_path: Path,
    ) -> None:
        prompt = await _initial_prompt(briefing, tmp_path, handled=_HANDLED)

        tail = _steward_owned_tail(prompt)
        assert '- **Detail:** Agent ran out of iterations.\n' + render_handled_digest(_HANDLED) in tail
        assert render_handled_digest(_HANDLED) + '\n# Parameters\n' in tail


_EARLIER = EarlierAttempts(timed_out=1)


@pytest.mark.asyncio
class TestEarlierAttempts:
    async def test_earlier_attempts_follow_the_escalation_block_before_the_pending_block(
        self, briefing: BriefingAssembler, tmp_path: Path,
    ) -> None:
        prompt = await _initial_prompt(
            briefing, tmp_path, handled=_HANDLED, pending_escalations=[_OTHER_PENDING],
            earlier_attempts=_EARLIER,
        )

        tail = _steward_owned_tail(prompt)
        attempts = render_earlier_attempts(_EARLIER)
        assert '- **Detail:** Agent ran out of iterations.\n' + attempts + '\n## Other Pending' in tail
        assert tail.index(attempts) < tail.index(render_handled_digest(_HANDLED))

    async def test_earlier_attempts_alone_end_before_parameters(
        self, briefing: BriefingAssembler, tmp_path: Path,
    ) -> None:
        prompt = await _initial_prompt(briefing, tmp_path, earlier_attempts=_EARLIER)

        assert render_earlier_attempts(_EARLIER) + '\n# Parameters\n' in _steward_owned_tail(prompt)
