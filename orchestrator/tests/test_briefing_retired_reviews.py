"""The architect's briefing on review rounds answered on a quarantined branch (task 3663).

A quarantine renamed the reviewed branch aside, so the amend-round responses
committed on it are not on the new branch. The plan and revalidation
briefings name the parked branch and list each in-scope suggestion it
amended, so the architect can plan their re-derivation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import memory_transport

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.review_lineage import RetiredReview

_TASK = {'id': '9', 'title': 'T', 'description': 'D'}
_PLAN = {'task_id': '9', 'title': 'T', 'files': [], 'steps': []}
_BRANCH = 'refs/heads/task/9-20261112T000000Z'
_TIP = 'f00dfacecafe1234'
_ARCHIVE = Path('/meta/task-9/retired-reviews/9-20261112T000000Z')
_SUGGESTIONS = (
    {
        'reviewer': 'reviewer_comprehensive',
        'location': 'src/foo/bar.py:42',
        'category': 'correctness',
        'description': 'the retry loop swallows the timeout',
        'suggested_fix': 'reraise TimeoutError after the last attempt',
    },
    {
        'reviewer': 'reviewer_comprehensive',
        'location': 'src/foo/baz.py:7',
        'category': 'clarity',
        'description': 'the helper name hides that it writes to disk',
        'suggested_fix': 'rename it to persist_manifest',
    },
)
_RETIRED = RetiredReview(
    branch=_BRANCH,
    tip_sha=_TIP,
    parking='/q/9-20261112T000000Z',
    quarantined_at='2026-11-12T00:00:00+00:00',
    amendment_rounds_total=1,
    review_cycles_total=0,
    archive=_ARCHIVE,
    suggestions=_SUGGESTIONS,
)


@pytest.fixture
def briefing(tmp_path: Path) -> Iterator[BriefingAssembler]:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    with memory_transport(AsyncMock(side_effect=ConnectionRefusedError('no memory in tests'))):
        yield BriefingAssembler(config)


async def _architect(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_architect_prompt(_TASK, context='', **kwargs)


async def _revalidation(briefing: BriefingAssembler, **kwargs) -> str:
    return await briefing.build_revalidation_prompt(_TASK, _PLAN, [], context='', **kwargs)


_BUILDERS = [_architect, _revalidation]


@pytest.mark.asyncio
@pytest.mark.parametrize('build', _BUILDERS)
async def test_the_section_names_the_parked_branch_and_its_archive(briefing, build):
    prompt = await build(briefing, retired_reviews=(_RETIRED,))

    assert _BRANCH in prompt
    assert _TIP in prompt
    assert str(_ARCHIVE) in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize('build', _BUILDERS)
async def test_the_section_lists_every_suggestion_and_its_fix(briefing, build):
    prompt = await build(briefing, retired_reviews=(_RETIRED,))

    for suggestion in _SUGGESTIONS:
        assert suggestion['description'] in prompt
        assert suggestion['suggested_fix'] in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize('build', _BUILDERS)
async def test_the_section_asks_for_a_step_or_a_design_decision(briefing, build):
    prompt = await build(briefing, retired_reviews=(_RETIRED,))

    assert 'Review Rounds Answered On A Quarantined Branch' in prompt
    assert 'add_design_decision' in prompt
    assert f'git log --oneline main..{_TIP}' in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize('build', _BUILDERS)
async def test_nothing_retired_leaves_the_prompt_byte_identical(briefing, build):
    assert await build(briefing, retired_reviews=()) == await build(briefing)
