"""Tests for ``reconciliation/unlanded_path_check.py`` (task 5626).

A done task's Mem0 memories can name a path the task branch created and then
deleted before it merged (solar_challenge task 107). The check asks git for
those paths, matches them in the task's memories, and emits one advisory
Stage-1 flag per task for Stage 2 to act on.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from itertools import count

import pytest

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
    mentions_path,
    select_done_transitions,
)
from fused_memory.services.unlanded_branch_paths import (
    ProbeOutcome,
    UnlandedPath,
    UnlandedPathsProbe,
)

GUARD = 'tests/unit/guard.py'
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
            'merged at abc',
        ],
    )
    def test_commit_is_none_without_a_usable_provenance_commit(self, provenance):
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
