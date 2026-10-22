"""Tests for ``reconciliation/unlanded_path_check.py`` (task 5626).

A done task's Mem0 memories can name a path the task branch created and then
deleted before it merged (solar_challenge task 107). The check asks git for
those paths, matches them in the task's memories, and emits one advisory
Stage-1 flag per task for Stage 2 to act on.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from itertools import count
from pathlib import Path

import pytest
from _fm_helpers import _init_git_repo
from _merged_branch_repo import commit_file, delete_file, git, merge_no_ff

from fused_memory.models.reconciliation import EventSource, EventType, ReconciliationEvent
from fused_memory.reconciliation.cli_stage_runner import FINDING_ITEM_SCHEMA
from fused_memory.reconciliation.flag_dedup import (
    ABSENCE_FLAG_TYPES,
    STALE_METADATA_FLAG_TYPES,
    compute_flag_signature,
)
from fused_memory.reconciliation.unlanded_path_check import (
    UNLANDED_PATH_FLAG_CATEGORY,
    UNLANDED_PATH_FLAG_TYPE,
    DoneTransition,
    UnlandedPathCheckResult,
    build_unlanded_path_flag,
    check_unlanded_path_claims,
    mentions_path,
    select_done_transitions,
)
from fused_memory.services.unlanded_branch_paths import (
    ProbeOutcome,
    UnlandedPath,
    UnlandedPathsProbe,
)

GUARD = 'tests/unit/guard.py'
BRIEFING = 'review/briefing.yaml'
OTHER = 'src/pkg/helper.py'
MERGE_SHA = 'abcdef0123' + '4' * 30
ADD_SHA = '1111111111' + 'a' * 30
EDIT_SHA = '2222222222' + 'b' * 30
DROP_SHA = '3333333333' + 'c' * 30

_event_ids = count()


def _event(event_type: EventType, payload: dict) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=f'evt-{next(_event_ids)}',
        type=event_type,
        source=EventSource.agent,
        project_id='test_project',
        timestamp=datetime.now(UTC),
        payload=payload,
    )


def _status(task_id: object, new_status: str, provenance: object = None) -> ReconciliationEvent:
    payload: dict = {'task_id': task_id, 'old_status': 'in-progress', 'new_status': new_status}
    if provenance is not None:
        payload['done_provenance'] = provenance
    return _event(EventType.task_status_changed, payload)


def _merged(commit: str) -> dict:
    return {'kind': 'merged', 'commit': commit}


class TestSelectDoneTransitions:
    def test_keeps_only_done_status_changes(self):
        events = [
            _status('1', 'done', _merged(MERGE_SHA)),
            _status('2', 'blocked'),
            _status('3', 'in-progress'),
            _event(EventType.memory_added, {'task_id': '4', 'new_status': 'done'}),
        ]

        assert select_done_transitions(events) == (DoneTransition(task_id='1', commit=MERGE_SHA),)

    def test_int_task_id_is_coerced_to_str(self):
        (transition,) = select_done_transitions([_status(107, 'done', _merged(MERGE_SHA))])

        assert transition.task_id == '107'

    @pytest.mark.parametrize(
        'provenance',
        [
            None,
            {'kind': 'operational-verified', 'evidence': 'ran the script'},
            {'kind': 'merged', 'commit': ''},
            {'kind': 'merged', 'commit': 12345},
            {'kind': 'found_on_main', 'commit': MERGE_SHA, 'note': 'landed by task 106'},
            'merged at abc',
        ],
    )
    def test_commit_is_none_without_the_tasks_own_merge_commit(self, provenance):
        (transition,) = select_done_transitions([_status('5', 'done', provenance)])

        assert transition.commit is None

    def test_last_event_per_task_wins_in_first_appearance_order(self):
        events = [
            _status('9', 'done', _merged(ADD_SHA)),
            _status('8', 'done', _merged(EDIT_SHA)),
            _status('9', 'done', _merged(DROP_SHA)),
        ]

        assert select_done_transitions(events) == (
            DoneTransition(task_id='9', commit=DROP_SHA),
            DoneTransition(task_id='8', commit=EDIT_SHA),
        )

    def test_done_transition_is_frozen(self):
        transition = DoneTransition(task_id='1', commit=None)

        with pytest.raises(dataclasses.FrozenInstanceError):
            transition.commit = MERGE_SHA  # type: ignore[misc]


class TestMentionsPath:
    @pytest.mark.parametrize(
        'content',
        [
            f'Added {GUARD} as the guard.',
            f'Added `{GUARD}` as the guard.',
            f'See {GUARD}::test_x for the pin.',
            f'The guard lives in {GUARD}.',
            f'Wrote /home/x/repo/{GUARD} today.',
        ],
    )
    def test_mentions(self, content):
        assert mentions_path(content, GUARD)

    @pytest.mark.parametrize(
        'content',
        [
            f'Compiled {GUARD}c into the cache.',
            f'Created {GUARD}/extra as a directory.',
            f'Edited my{GUARD} instead.',
            'Nothing about any guard here.',
        ],
    )
    def test_non_mentions(self, content):
        assert not mentions_path(content, GUARD)

    def test_regex_metacharacters_in_the_path_are_literal(self):
        assert not mentions_path('touched aab/c.py', 'a+b/c.py')
        assert mentions_path('touched a+b/c.py', 'a+b/c.py')


def _probe(*unlanded: UnlandedPath) -> UnlandedPathsProbe:
    return UnlandedPathsProbe(outcome=ProbeOutcome.ANALYZED, merge_commit=MERGE_SHA, unlanded=unlanded)


GUARD_UNLANDED = UnlandedPath(path=GUARD, branch_commits=(ADD_SHA, EDIT_SHA, DROP_SHA))
OTHER_UNLANDED = UnlandedPath(path=OTHER, branch_commits=(EDIT_SHA,))


class TestBuildUnlandedPathFlag:
    def _flag(self, task_id: object = '107') -> dict:
        probe = _probe(GUARD_UNLANDED, OTHER_UNLANDED)
        offenders = [('m1', (GUARD_UNLANDED,)), ('m2', (GUARD_UNLANDED, OTHER_UNLANDED))]
        return build_unlanded_path_flag(task_id, probe, offenders)

    def test_identity_fields(self):
        flag = self._flag(107)

        assert UNLANDED_PATH_FLAG_TYPE == 'memory_cites_unlanded_path'
        assert UNLANDED_PATH_FLAG_CATEGORY == 'memory_stale'
        assert flag['flag_type'] == UNLANDED_PATH_FLAG_TYPE
        assert flag['category'] == UNLANDED_PATH_FLAG_CATEGORY
        assert flag['task_id'] == '107'
        assert flag['severity'] == 'moderate'
        assert flag['actionable'] is True
        assert flag['category'] in FINDING_ITEM_SCHEMA['properties']['category']['enum']
        assert set(flag) <= set(FINDING_ITEM_SCHEMA['properties'])
        assert compute_flag_signature(flag) == ('107', UNLANDED_PATH_FLAG_TYPE)

    def test_cited_memories_in_offender_order(self):
        assert self._flag()['cited_memories'] == [
            {'memory_id': 'm1', 'store': 'mem0'},
            {'memory_id': 'm2', 'store': 'mem0'},
        ]

    def test_description_names_the_evidence(self):
        description = self._flag()['description']

        for needle in ('m1', 'm2', GUARD, OTHER, MERGE_SHA[:10], ADD_SHA[:10], EDIT_SHA[:10], DROP_SHA[:10]):
            assert needle in description

    def test_suggested_action_names_both_branches(self):
        action = self._flag()['suggested_action']

        assert 'update_memory' in action
        assert 'dismiss' in action

    def test_flag_type_survives_the_done_task_filters(self):
        assert UNLANDED_PATH_FLAG_TYPE not in STALE_METADATA_FLAG_TYPES
        assert UNLANDED_PATH_FLAG_TYPE not in ABSENCE_FLAG_TYPES


class TestUnlandedPathCheckResult:
    def test_empty_stats(self):
        assert UnlandedPathCheckResult.empty().stats() == {
            'unlanded_path_check_checked': 0,
            'unlanded_path_check_skipped': 0,
            'unlanded_path_check_errors': 0,
            'unlanded_path_check_flags_emitted': 0,
        }

    def test_flags_emitted_counts_the_flags(self):
        result = UnlandedPathCheckResult(flags=({'a': 1}, {'b': 2}), checked=3, skipped=1, errors=1)

        assert result.stats() == {
            'unlanded_path_check_checked': 3,
            'unlanded_path_check_skipped': 1,
            'unlanded_path_check_errors': 1,
            'unlanded_path_check_flags_emitted': 2,
        }

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            UnlandedPathCheckResult.empty().checked = 1  # type: ignore[misc]


# ── check_unlanded_path_claims against a real merged repo ───────────────────

PROJECT = 'test_project'


class FakeMemoryService:
    """Duck-typed stand-in for the two public MemoryService readers, with their keyword signatures.

    *memories_by_task* maps a task id to ``{memory_id: content}``; a ``None``
    content makes ``get_memory_by_id`` answer not-found for that id.
    """

    def __init__(
        self,
        memories_by_task: dict[str, dict[str, str | None]] | None = None,
        *,
        list_error: BaseException | None = None,
        read_errors: dict[str, BaseException] | None = None,
    ) -> None:
        self.memories_by_task = memories_by_task or {}
        self.list_error = list_error
        self.read_errors = read_errors or {}
        self.list_calls: list[dict] = []
        self.read_calls: list[dict] = []

    async def get_memories_by_metadata(self, *, project_id: str, filters: dict, limit: int = 1000) -> list[dict]:
        self.list_calls.append({'project_id': project_id, 'filters': filters, 'limit': limit})
        if self.list_error is not None:
            raise self.list_error
        task_id = filters['task_id']
        return [
            {'id': memory_id, 'created_at': '2026-09-18T13:08:50+00:00', 'metadata': {'task_id': task_id}}
            for memory_id in self.memories_by_task.get(task_id, {})
        ]

    async def get_memory_by_id(self, *, project_id: str, memory_id: str) -> dict | None:
        self.read_calls.append({'project_id': project_id, 'memory_id': memory_id})
        if memory_id in self.read_errors:
            raise self.read_errors[memory_id]
        for memories in self.memories_by_task.values():
            content = memories.get(memory_id)
            if content is not None:
                return {'id': memory_id, 'content': content, 'metadata': {}}
        return None


INCIDENT_MEMORIES: dict[str, str | None] = {
    'm1': f'Task 107 delivered {GUARD} and a briefing edit.',
    'm2': f'Task 107 edited {BRIEFING}.',
    'm3': None,
}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    root = tmp_path / 'repo'
    root.mkdir()
    _init_git_repo(root)
    commit_file(root, BRIEFING, 'v1\n', 'seed briefing')
    return root


def _incident_merge(repo: Path, task_id: str) -> str:
    """The solar_challenge 107 shape: the branch adds, edits and deletes GUARD, and edits BRIEFING."""
    branch = f'task/{task_id}'
    git(repo, 'checkout', '-q', '-b', branch, 'main')
    commit_file(repo, GUARD, 'one\n', 'add guard')
    commit_file(repo, BRIEFING, f'{task_id}\n', 'edit briefing')
    commit_file(repo, GUARD, 'two\n', 'edit guard')
    delete_file(repo, GUARD, 'drop guard')
    return merge_no_ff(repo, branch)


def _clean_merge(repo: Path, task_id: str) -> str:
    branch = f'task/{task_id}'
    git(repo, 'checkout', '-q', '-b', branch, 'main')
    commit_file(repo, f'src/feature_{task_id}.py', 'x\n', 'add feature')
    return merge_no_ff(repo, branch)


async def _check(memory: FakeMemoryService, root: Path, events: list[ReconciliationEvent]) -> UnlandedPathCheckResult:
    return await check_unlanded_path_claims(memory, PROJECT, str(root), events)


@pytest.mark.asyncio
async def test_incident_flags_only_the_memory_naming_the_unlanded_path(repo: Path) -> None:
    merge = _incident_merge(repo, '107')
    memory = FakeMemoryService({'107': INCIDENT_MEMORIES})

    result = await _check(memory, repo, [_status('107', 'done', _merged(merge))])

    (flag,) = result.flags
    assert flag['task_id'] == '107'
    assert flag['cited_memories'] == [{'memory_id': 'm1', 'store': 'mem0'}]
    assert GUARD in flag['description']
    assert result.checked == 1
    assert result.errors == 0
    assert result.stats()['unlanded_path_check_flags_emitted'] == 1
    assert [(c['project_id'], c['filters']) for c in memory.list_calls] == [(PROJECT, {'task_id': '107'})]
    assert {c['project_id'] for c in memory.read_calls} == {PROJECT}


@pytest.mark.asyncio
async def test_clean_merge_costs_no_memory_reads(repo: Path) -> None:
    merge = _clean_merge(repo, '108')
    memory = FakeMemoryService({'108': {'m1': f'Mentions {GUARD} anyway.'}})

    result = await _check(memory, repo, [_status('108', 'done', _merged(merge))])

    assert result.checked == 1
    assert result.flags == ()
    assert memory.list_calls == []
    assert memory.read_calls == []


@pytest.mark.asyncio
async def test_commitless_and_non_merge_transitions_are_skipped(repo: Path) -> None:
    seed = git(repo, 'rev-list', '--max-parents=0', 'HEAD')
    memory = FakeMemoryService({'201': {'m1': GUARD}, '202': {'m2': GUARD}})
    events = [
        _status('201', 'done', {'kind': 'operational-verified', 'evidence': 'ran it'}),
        _status('202', 'done', _merged(seed)),
    ]

    result = await _check(memory, repo, events)

    assert result.skipped == 2
    assert result.checked == 0
    assert result.flags == ()
    assert memory.list_calls == []
    assert memory.read_calls == []


@pytest.mark.asyncio
async def test_found_on_main_borrowing_another_tasks_merge_is_skipped(repo: Path) -> None:
    borrowed = _incident_merge(repo, '106')
    memory = FakeMemoryService({'107': INCIDENT_MEMORIES})
    found = {'kind': 'found_on_main', 'commit': borrowed, 'note': 'landed by task 106'}

    result = await _check(memory, repo, [_status('107', 'done', found)])

    assert result == UnlandedPathCheckResult(flags=(), checked=0, skipped=1, errors=0)
    assert memory.list_calls == []


@pytest.mark.asyncio
async def test_non_git_project_root_is_an_error_not_a_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    plain = tmp_path / 'plain'
    plain.mkdir()
    memory = FakeMemoryService({'107': INCIDENT_MEMORIES})

    result = await _check(memory, plain, [_status('107', 'done', _merged(MERGE_SHA))])

    assert result.errors == 1
    assert result.flags == ()
    assert memory.list_calls == []


@pytest.mark.asyncio
async def test_memory_listing_failure_is_an_error_not_a_raise(repo: Path) -> None:
    merge = _incident_merge(repo, '107')
    memory = FakeMemoryService({'107': INCIDENT_MEMORIES}, list_error=TimeoutError('qdrant read timed out'))

    result = await _check(memory, repo, [_status('107', 'done', _merged(merge))])

    assert result.errors == 1
    assert result.flags == ()


@pytest.mark.asyncio
async def test_one_unreadable_memory_does_not_cost_the_flag(repo: Path) -> None:
    merge = _incident_merge(repo, '107')
    memory = FakeMemoryService(
        {'107': {'m-bad': f'also names {GUARD}', 'm1': f'delivered {GUARD}'}},
        read_errors={'m-bad': TimeoutError('qdrant read timed out')},
    )

    result = await _check(memory, repo, [_status('107', 'done', _merged(merge))])

    assert result.errors == 1
    (flag,) = result.flags
    assert flag['cited_memories'] == [{'memory_id': 'm1', 'store': 'mem0'}]


@pytest.mark.asyncio
async def test_two_done_tasks_one_unlanded(repo: Path) -> None:
    incident = _incident_merge(repo, '107')
    clean = _clean_merge(repo, '108')
    memory = FakeMemoryService({'107': INCIDENT_MEMORIES, '108': {'m9': 'clean work'}})
    events = [_status('107', 'done', _merged(incident)), _status('108', 'done', _merged(clean))]

    result = await _check(memory, repo, events)

    assert result.checked == 2
    assert [flag['task_id'] for flag in result.flags] == ['107']


@pytest.mark.asyncio
async def test_cancellation_propagates(repo: Path) -> None:
    merge = _incident_merge(repo, '107')
    memory = FakeMemoryService({'107': INCIDENT_MEMORIES}, list_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _check(memory, repo, [_status('107', 'done', _merged(merge))])


@pytest.mark.asyncio
async def test_no_done_transitions_is_empty(repo: Path) -> None:
    memory = FakeMemoryService({'107': INCIDENT_MEMORIES})
    events = [_status('107', 'blocked'), _event(EventType.memory_added, {'task_id': '107'})]

    assert await _check(memory, repo, events) == UnlandedPathCheckResult.empty()
