"""``metadata.followup_items`` renders numbered in every role's task block (task 8414).

plans/follow-up-filing-shape-prd.md §3 decision 1: the items are the task's charter, so they reach
the architect through the briefing, never only through curator prose.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from _briefing_helpers import briefing as briefing
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, observed_item

from orchestrator.agents.briefing import BriefingAssembler

_SECTION = '## Follow-up Items'

_OBSERVATIONS = [
    ('The retry loop swallows the timeout', ['src/app/retry.py']),
    ('The cache key ignores the tenant', ['src/app/cache.py', 'src/app/keys.py']),
    ('The audit row omits the actor', ['src/app/audit.py']),
]


def _stored_items(origin_task: str = '8000') -> list[dict[str, Any]]:
    return [
        observed_item(summary=summary, files=files, origin_task=origin_task).model_dump(mode='json')
        for summary, files in _OBSERVATIONS
    ]


def _task(metadata: object) -> dict[str, Any]:
    return {
        'id': '8500',
        'title': 'Follow-ups in src/app',
        'description': 'Three observations in one area.',
        'metadata': metadata,
    }


def _bundle_task() -> dict[str, Any]:
    return _task({
        'files': sorted({path for _summary, files in _OBSERVATIONS for path in files}),
        FOLLOWUP_ITEMS_METADATA_KEY: _stored_items(),
    })


def _section(prompt: str) -> str:
    """The follow-up section alone: the role templates carry numbered lists of their own."""
    assert prompt.count(_SECTION) == 1, prompt
    rest = prompt[prompt.index(_SECTION) + len(_SECTION):]
    next_heading = re.search(r'^#{1,2} ', rest, flags=re.MULTILINE)
    return rest[: next_heading.start()] if next_heading else rest


def _numbered_lines(prompt: str) -> list[str]:
    return [line for line in _section(prompt).splitlines() if re.match(r'\d+\. ', line)]


@pytest.mark.asyncio
async def test_the_architect_prompt_numbers_every_item_in_stored_order(
    briefing: BriefingAssembler,
) -> None:
    prompt = await briefing.build_architect_prompt(_bundle_task(), worktree=None, context='')

    lines = _numbered_lines(prompt)
    assert len(lines) == len(_OBSERVATIONS)
    for number, (line, (summary, files)) in enumerate(zip(lines, _OBSERVATIONS, strict=True), start=1):
        assert line.startswith(f'{number}. ')
        assert summary in line
        assert all(path in line for path in files)
        assert '8000' in line


@pytest.mark.asyncio
async def test_the_section_survives_the_architect_files_anti_anchor(briefing: BriefingAssembler) -> None:
    prompt = await briefing.build_architect_prompt(_bundle_task(), worktree=None, context='')

    assert '**Files:**' not in prompt
    assert len(_numbered_lines(prompt)) == len(_OBSERVATIONS)


@pytest.mark.asyncio
async def test_the_simple_task_prompt_carries_the_section(briefing: BriefingAssembler) -> None:
    prompt = await briefing.build_simple_task_prompt(_bundle_task(), worktree=None, context='')

    assert _OBSERVATIONS[2][0] in _numbered_lines(prompt)[2]


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [None, {}, {'files': ['src/app/retry.py']}])
async def test_a_task_without_items_has_no_section(briefing: BriefingAssembler, metadata: object) -> None:
    prompt = await briefing.build_architect_prompt(_task(metadata), worktree=None, context='')

    assert _SECTION not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'malformed',
    [
        pytest.param('not an item', id='non-dict'),
        pytest.param({'summary': 'no files or key', 'origin_task': '8000'}, id='invalid-item'),
    ],
)
async def test_malformed_entries_are_counted_beside_the_valid_ones(
    briefing: BriefingAssembler, malformed: object,
) -> None:
    valid = _stored_items()[0]
    task = _task({'files': ['src/app/retry.py'], FOLLOWUP_ITEMS_METADATA_KEY: [malformed, valid]})

    prompt = await briefing.build_architect_prompt(task, worktree=None, context='')

    lines = _numbered_lines(prompt)
    assert len(lines) == 1
    assert lines[0].startswith('1. ')
    assert _OBSERVATIONS[0][0] in lines[0]
    assert '1 malformed entry' in _section(prompt)


@pytest.mark.asyncio
async def test_a_non_list_value_renders_a_malformed_notice(briefing: BriefingAssembler) -> None:
    task = _task({FOLLOWUP_ITEMS_METADATA_KEY: 'one summary as a bare string'})

    prompt = await briefing.build_architect_prompt(task, worktree=None, context='')

    assert 'malformed' in _section(prompt)
    assert _numbered_lines(prompt) == []
