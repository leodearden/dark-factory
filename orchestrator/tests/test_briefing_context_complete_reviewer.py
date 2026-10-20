"""The context-complete (D2) reviewer briefing carries what the incumbent never sees.

The incumbent ``build_reviewer_prompt`` hands the reviewer a diff cut at
50,000 characters and nothing else, so a path-ordered cut drops whole
directories and the reviewer judges a change without the task it serves.
These tests drive the public seam
``BriefingAssembler.build_context_complete_reviewer_prompt`` with structured
``agents.review_briefing`` values and assert what reaches the reviewer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _briefing_helpers import (
    _mcp_search_envelope,
    _result,
    _search_arguments,
    briefing,  # noqa: F401 — re-export: pytest fixture used by test methods
    memory_transport,
)

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.review_briefing import (
    CONTEXT_COMPLETE_INLINE_DIFF_CHARS,
    ChangedFile,
    ReviewChange,
    ReviewConvention,
    ReviewSubject,
)

REVIEWER = 'reviewer_comprehensive'
BASE_SHA = '0123456789abcdef0123456789abcdef01234567'
AMENDMENT_HEADING = '# Amendment Re-Review Scope'
CONVENTIONS_HEADING = '# Project Conventions'


def _task() -> dict:
    return {
        'id': '4242',
        'title': 'Tighten the merge-lane park grace',
        'description': 'Shorten the grace the merge worker waits on a held index lock.',
        'details': 'The grace currently matches the pre-commit budget.',
        'testStrategy': 'Drive the worker against a held lock and time the stand-off.',
        'metadata': {'files': ['orchestrator/src/orchestrator/merge_worker.py']},
    }


def _plan() -> dict:
    return {
        'task_id': '4242',
        'title': 'Tighten the park grace',
        'analysis': 'The stand-off reads one config knob; lowering it is local.',
        'files': [
            'orchestrator/src/orchestrator/merge_worker.py',
            'orchestrator/tests/test_merge_worker.py',
        ],
        'design_decisions': [
            {'decision': 'Keep the knob in GitConfig', 'rationale': 'One home for git timing.'},
            {'decision': 'Log every stand-off', 'rationale': 'Operators read the log.'},
        ],
        'steps': [{'id': 'step-1', 'type': 'test', 'description': 'RED: the grace is shorter'}],
    }


def _hunk(path: str, added_lines: list[str]) -> str:
    body = '\n'.join(f'+{line}' for line in added_lines)
    return (
        f'diff --git a/{path} b/{path}\n'
        f'--- a/{path}\n'
        f'+++ b/{path}\n'
        f'@@ -0,0 +1,{len(added_lines)} @@\n'
        f'{body}\n'
    )


SMALL_FILES = (
    ChangedFile('orchestrator/src/orchestrator/merge_worker.py', 3, 1),
    ChangedFile('orchestrator/tests/test_merge_worker.py', 12, 0),
)
SMALL_DIFF = _hunk(SMALL_FILES[0].path, ['grace = 30', 'log.info("stand-off")', 'return grace']) + _hunk(
    SMALL_FILES[1].path, [f'assert case_{n}' for n in range(12)],
)


def _subject(
    *,
    diff: str = SMALL_DIFF,
    files: tuple[ChangedFile, ...] = SMALL_FILES,
    plan: dict | None = None,
    plan_absent_reason: str = '',
    conventions: tuple[ReviewConvention, ...] = (),
    task: dict | None = None,
) -> ReviewSubject:
    return ReviewSubject(
        task=task if task is not None else _task(),
        plan=plan,
        plan_absent_reason=plan_absent_reason,
        change=ReviewChange(base_sha=BASE_SHA, diff=diff, files=files),
        conventions=conventions,
    )


async def _build(briefing: BriefingAssembler, subject: ReviewSubject, **kwargs) -> str:
    kwargs.setdefault('context', 'CTX')
    return await briefing.build_context_complete_reviewer_prompt(REVIEWER, subject, **kwargs)


def _section(prompt: str, heading: str) -> str:
    """The text of the top-level section *heading*, up to the next top-level heading."""
    start = prompt.index(heading)
    end = prompt.find('\n# ', start + len(heading))
    return prompt[start:] if end < 0 else prompt[start:end]


def _line_naming(prompt: str, path: str) -> str:
    """The first line of the prompt that names *path* (the changed-file list precedes the diff)."""
    return next(line for line in prompt.splitlines() if path in line)


@pytest.mark.asyncio
class TestSubjectSnapshot:
    """A subject briefs every reviewer from the records as they were when it was built."""

    async def test_mutating_the_callers_records_after_construction_changes_no_briefing(
        self, briefing: BriefingAssembler,
    ):
        task, plan = _task(), _plan()
        subject = _subject(task=task, plan=plan)
        task['title'] = 'MUTATED TITLE'
        task['metadata']['files'].append('MUTATED_FILE.py')
        plan['analysis'] = 'MUTATED ANALYSIS'
        plan['files'].append('MUTATED_PLAN_FILE.py')

        prompt = await _build(briefing, subject)

        assert _task()['title'] in prompt
        assert _plan()['analysis'] in prompt
        assert 'MUTATED' not in prompt

    async def test_the_subjects_records_are_read_only(self) -> None:
        subject = _subject(plan=_plan())

        with pytest.raises(TypeError):
            subject.task['title'] = 'x'  # type: ignore[index]
        assert subject.plan is not None
        with pytest.raises(TypeError):
            subject.plan['analysis'] = 'x'  # type: ignore[index]


@pytest.mark.asyncio
class TestTaskAndPlan:
    async def test_every_briefed_task_field_reaches_the_reviewer(self, briefing: BriefingAssembler):
        task = _task()

        prompt = await _build(briefing, _subject(plan=_plan()))

        for field in ('title', 'description', 'details', 'testStrategy'):
            assert task[field] in prompt, field

    async def test_the_test_strategy_renders_on_its_own_line(self, briefing: BriefingAssembler):
        prompt = await _build(briefing, _subject())

        assert f'**Test strategy:** {_task()["testStrategy"]}' in prompt.splitlines()

    @pytest.mark.parametrize('test_strategy', ['', None])
    async def test_an_empty_or_absent_test_strategy_renders_nothing(
        self, briefing: BriefingAssembler, test_strategy: str | None,
    ):
        task = _task()
        if test_strategy is None:
            del task['testStrategy']
        else:
            task['testStrategy'] = test_strategy

        prompt = await _build(briefing, _subject(task=task))

        assert '**Test strategy:**' not in prompt

    async def test_the_plan_analysis_files_and_decisions_reach_the_reviewer(
        self, briefing: BriefingAssembler,
    ):
        plan = _plan()

        prompt = await _build(briefing, _subject(plan=plan))

        assert plan['analysis'] in prompt
        for path in plan['files']:
            assert path in prompt
        for entry in plan['design_decisions']:
            assert entry['decision'] in prompt
            assert entry['rationale'] in prompt

    async def test_an_absent_plan_states_why_and_still_builds(self, briefing: BriefingAssembler):
        reason = 'the plan was reclaimed with the worktree'

        prompt = await _build(briefing, _subject(plan=None, plan_absent_reason=reason))

        assert reason in prompt
        assert _task()['title'] in prompt


def _sixty_k_change() -> tuple[str, tuple[ChangedFile, ...]]:
    """A diff whose source hunk alone overruns a 50,000-char path-ordered cut."""
    source_lines = [f'value_{n:05d} = compute_the_value_for_slot({n})  # padding' for n in range(1200)]
    diff = (
        _hunk('orchestrator/src/a.py', source_lines)
        + _hunk('orchestrator/tests/test_a.py', ['def test_a():', '    assert a.value_00001'])
        + _hunk('plans/p.md', ['# Plan', 'Ship a.py.', 'Done.'])
    )
    files = (
        ChangedFile('orchestrator/src/a.py', 1200, 0),
        ChangedFile('orchestrator/tests/test_a.py', 2, 7),
        ChangedFile('plans/p.md', 3, 0),
        ChangedFile('docs/diagram.png', None, None),
    )
    return diff, files


@pytest.mark.asyncio
class TestChangedFiles:
    async def test_the_fixture_overruns_the_incumbent_cut(self):
        diff, _ = _sixty_k_change()

        assert diff.index('orchestrator/tests/test_a.py') > 50_000
        assert len(diff) < CONTEXT_COMPLETE_INLINE_DIFF_CHARS

    async def test_every_changed_path_is_listed_with_its_counts(self, briefing: BriefingAssembler):
        diff, files = _sixty_k_change()

        prompt = await _build(briefing, _subject(diff=diff, files=files))

        for changed in files[:3]:
            line = _line_naming(prompt, changed.path)
            assert f'+{changed.added}' in line, line
            assert f'-{changed.deleted}' in line, line

    async def test_a_binary_file_is_labelled_binary(self, briefing: BriefingAssembler):
        diff, files = _sixty_k_change()

        prompt = await _build(briefing, _subject(diff=diff, files=files))

        assert 'binary' in _line_naming(prompt, 'docs/diagram.png')

    async def test_the_per_file_command_names_the_base(self, briefing: BriefingAssembler):
        prompt = await _build(briefing, _subject())

        assert f'git diff {BASE_SHA}...HEAD -- ' in prompt


@pytest.mark.asyncio
class TestInlineDiff:
    async def test_a_diff_under_the_cap_is_inlined_whole(self, briefing: BriefingAssembler):
        diff, files = _sixty_k_change()

        prompt = await _build(briefing, _subject(diff=diff, files=files))

        assert diff in prompt
        assert '[diff truncated]' not in prompt

    async def test_a_diff_over_the_cap_is_omitted_wholesale_with_its_size(
        self, briefing: BriefingAssembler,
    ):
        line_count = CONTEXT_COMPLETE_INLINE_DIFF_CHARS // 40 + 100
        oversized = [f'oversized_marker_{n:06d} = slot({n})  # pad' for n in range(line_count)]
        diff = _hunk('orchestrator/src/big.py', oversized) + _hunk('plans/late.md', ['late'])
        files = (
            ChangedFile('orchestrator/src/big.py', line_count, 0),
            ChangedFile('plans/late.md', 1, 0),
        )
        assert len(diff) > CONTEXT_COMPLETE_INLINE_DIFF_CHARS

        prompt = await _build(briefing, _subject(diff=diff, files=files))

        assert 'oversized_marker_' not in prompt
        assert '+late' not in prompt
        assert f'{len(diff):,}' in prompt
        for changed in files:
            assert f'+{changed.added}' in _line_naming(prompt, changed.path)
        assert '[diff truncated]' not in prompt


@pytest.mark.asyncio
class TestConventionsAndAction:
    async def test_each_convention_rule_and_why_appear(self, briefing: BriefingAssembler):
        conventions = (
            ReviewConvention('Cite cross-file refs as path::symbol', 'Line pins drift.'),
            ReviewConvention('Never run git stash', 'refs/stash is shared across worktrees.'),
        )

        prompt = await _build(briefing, _subject(conventions=conventions))

        section = _section(prompt, CONVENTIONS_HEADING)
        for convention in conventions:
            assert convention.rule in section
            assert convention.why in section

    async def test_no_conventions_renders_no_section(self, briefing: BriefingAssembler):
        prompt = await _build(briefing, _subject(conventions=()))

        assert CONVENTIONS_HEADING not in prompt

    async def test_the_verdict_call_names_the_reviewer(self, briefing: BriefingAssembler):
        prompt = await _build(briefing, _subject())

        assert f'submit_review_verdict(reviewer="{REVIEWER}"' in prompt


@pytest.mark.asyncio
class TestSharedWithTheIncumbent:
    async def test_the_amendment_section_matches_the_incumbent(self, briefing: BriefingAssembler):
        suggestions = [
            {'location': 'src/foo.py:42', 'description': 'tighten the error message here'},
            {'location': None, 'description': 'name the constant'},
        ]

        incumbent = await briefing.build_reviewer_prompt(
            REVIEWER, SMALL_DIFF, context='CTX', amendment_suggestions=suggestions,
        )
        candidate = await _build(briefing, _subject(), amendment_suggestions=suggestions)

        assert AMENDMENT_HEADING in candidate
        assert (
            candidate[candidate.index(AMENDMENT_HEADING):]
            == incumbent[incumbent.index(AMENDMENT_HEADING):]
        )

    async def test_no_suggestions_renders_no_amendment_section(self, briefing: BriefingAssembler):
        prompt = await _build(briefing, _subject())

        assert AMENDMENT_HEADING not in prompt

    async def test_no_context_issues_a_task_scoped_memory_query(self, briefing: BriefingAssembler):
        entry = _result('1', 'A recalled fact.', source_store='mem0')
        entry['category'] = 'preferences_and_norms'
        entry['created_at'] = '2026-08-15T22:22:49+00:00'
        mcp = AsyncMock(return_value=_mcp_search_envelope([entry]))

        with memory_transport(mcp):
            prompt = await briefing.build_context_complete_reviewer_prompt(
                REVIEWER, _subject(), context=None,
            )

        assert 'A recalled fact.' in prompt
        arguments = _search_arguments(mcp)
        assert arguments
        for args in arguments:
            assert args['caller_agent_id'] == 'claude-task-4242-reviewer'
            assert args['caller_task_id'] == '4242'
