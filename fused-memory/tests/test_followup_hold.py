"""The follow-up holder marker and its predicate (plans/follow-up-filing-shape-prd.md §4.3, decision 10).

A holder is the task whose metadata carries ``followup_holder``. The predicate reads a task row as
the store hands it out: metadata as a dict or as its JSON string, possibly absent or unreadable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError
from shared.followup_hold_records import FOLLOWUP_HOLD_METADATA_KEY, FOLLOWUP_HOLDER_METADATA_KEY

from fused_memory.middleware.followup_hold import (
    FollowupHolderProtectedError,
    HolderProtection,
    PromotionDeferral,
    carry_hold_record,
    citation_ref,
    held_by,
    hold_at_create,
    hold_scope,
    is_holder,
    may_strip_holder_edge,
    protected_status,
    protected_update,
    releases_holder,
)

_MARKER = {'project_id': 'dark_factory', 'minted_at': '2026-11-08T00:00:00+00:00'}


def test_the_marker_key_is_the_contract_fixed_name() -> None:
    assert FOLLOWUP_HOLDER_METADATA_KEY == 'followup_holder'


@pytest.mark.parametrize('metadata', [
    {FOLLOWUP_HOLDER_METADATA_KEY: _MARKER, 'files': ['a.py']},
    json.dumps({FOLLOWUP_HOLDER_METADATA_KEY: _MARKER}),
], ids=['dict', 'json-string'])
def test_a_row_carrying_the_marker_is_a_holder(metadata: Any) -> None:
    assert is_holder({'id': 7, 'status': 'pending', 'metadata': metadata})


@pytest.mark.parametrize('task', [
    {'id': 7, 'metadata': {'files': ['a.py']}},
    {'id': 7, 'metadata': {FOLLOWUP_HOLDER_METADATA_KEY: None}},
    {'id': 7, 'metadata': json.dumps({FOLLOWUP_HOLDER_METADATA_KEY: None})},
    {'id': 7},
    {'id': 7, 'metadata': ''},
    {'id': 7, 'metadata': None},
    {'id': 7, 'metadata': json.dumps([FOLLOWUP_HOLDER_METADATA_KEY])},
    {'id': 7, 'metadata': 'followup_holder: yes'},
], ids=[
    'key-absent', 'value-none', 'json-value-none', 'metadata-absent', 'metadata-empty-string',
    'metadata-none', 'non-object-json', 'garbage-string',
])
def test_a_row_without_a_readable_marker_is_not_a_holder(task: dict[str, Any]) -> None:
    assert not is_holder(task)


_PROBE = (
    'import json, sys\n'
    'import fused_memory.middleware.followup_hold\n'
    'print(json.dumps([name for name in ('
    "'fused_memory.middleware.task_interceptor', 'fused_memory.middleware.task_curator'"
    ') if name in sys.modules]))\n'
)


def test_importing_followup_hold_loads_neither_the_interceptor_nor_the_curator() -> None:
    """curator_types imports this module, so it must stay below both drivers.

    ``-I`` drops PYTHONPATH, the user site and the cwd, so this measures the module's own import
    closure rather than whatever the pytest process loaded.
    """
    result = subprocess.run(
        [sys.executable, '-I', '-c', _PROBE], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [], (
        f'importing followup_hold loaded {result.stdout.strip()}; it must import only the stdlib '
        'and lower layers.'
    )


_FOLLOWUP = {'source': 'agent-followup', 'spawn_context': 'agent_followup', 'spawned_from': '101'}
_REVIEW = {'spawn_context': 'review_suggestions'}


@pytest.mark.parametrize(('metadata', 'priority'), [
    (_FOLLOWUP, 'low'),
    (_FOLLOWUP, 'polish'),
    (_REVIEW, 'low'),
    ({**_FOLLOWUP, 'complexity': 'simple'}, 'low'),
    ({**_FOLLOWUP, 'task_kind': 'normal'}, 'low'),
    (json.dumps(_FOLLOWUP), 'low'),
], ids=['followup-low', 'followup-polish', 'review-low', 'simple', 'task-kind-normal', 'json-string'])
def test_a_low_followup_is_in_hold_scope(metadata: Any, priority: str) -> None:
    assert hold_scope(metadata, priority)


@pytest.mark.parametrize(('metadata', 'priority'), [
    (_FOLLOWUP, 'medium'),
    (_FOLLOWUP, 'high'),
    (_FOLLOWUP, None),
    ({'source': 'reconciliation'}, 'low'),
    ({'spawn_context': 'agent_followup'}, 'low'),
    ({**_FOLLOWUP, 'task_kind': 'deterministic'}, 'low'),
    ({**_FOLLOWUP, 'execution_class': 'operational'}, 'low'),
    ({**_FOLLOWUP, 'prd_path': 'plans/x-prd.md'}, 'low'),
    ({**_FOLLOWUP, 'x_finding_key': 'fk-x'}, 'low'),
    ({**_FOLLOWUP, 'x_finding_key': ['fk-x']}, 'low'),
    ({**_FOLLOWUP, 'human_decomposed': True}, 'low'),
    (json.dumps({**_FOLLOWUP, 'human_decomposed': True}), 'low'),
    (None, 'low'),
    ('', 'low'),
], ids=[
    'medium', 'high', 'no-priority', 'other-source', 'followup-context-without-source',
    'deterministic', 'execution-class', 'prd-path', 'finding-key', 'finding-key-list',
    'planning-birth', 'planning-birth-json', 'metadata-none', 'metadata-empty',
])
def test_a_task_outside_the_hold_rule_is_not_held(metadata: Any, priority: str | None) -> None:
    assert not hold_scope(metadata, priority)


_HELD_AT = '2026-11-19T04:30:00+00:00'


@pytest.mark.parametrize(('dependencies', 'expected'), [
    (None, '9'),
    ('', '9'),
    ('3,4', '3,4,9'),
    ('3,9', '3,9'),
    ('9', '9'),
], ids=['none', 'empty', 'two-others', 'already-names-holder', 'only-holder'])
def test_hold_at_create_appends_the_holder_edge(dependencies: str | None, expected: str) -> None:
    kwargs = {'title': 't', 'priority': 'low', 'dependencies': dependencies}

    held_kwargs, _ = hold_at_create(kwargs, dict(_FOLLOWUP), 9, held_at=_HELD_AT)

    assert held_kwargs['dependencies'] == expected
    assert held_kwargs['title'] == 't'
    assert held_kwargs['priority'] == 'low'


def test_hold_at_create_appends_the_edge_when_dependencies_are_absent() -> None:
    held_kwargs, _ = hold_at_create({'title': 't'}, dict(_FOLLOWUP), 9, held_at=_HELD_AT)

    assert held_kwargs == {'title': 't', 'dependencies': '9'}


@pytest.mark.parametrize('metadata', [dict(_FOLLOWUP), json.dumps(_FOLLOWUP)], ids=['dict', 'json-string'])
def test_hold_at_create_stamps_the_record_beside_every_other_key(metadata: Any) -> None:
    _, held_metadata = hold_at_create({'title': 't'}, metadata, 9, held_at=_HELD_AT)

    assert held_metadata == {**_FOLLOWUP, FOLLOWUP_HOLD_METADATA_KEY: {'holder': 9, 'held_at': _HELD_AT}}


def test_hold_at_create_leaves_its_inputs_unmutated() -> None:
    kwargs = {'title': 't', 'dependencies': '3'}
    metadata = dict(_FOLLOWUP)

    hold_at_create(kwargs, metadata, 9, held_at=_HELD_AT)

    assert kwargs == {'title': 't', 'dependencies': '3'}
    assert metadata == _FOLLOWUP


_HOLDER_ROW = {'id': 9, 'status': 'pending', 'metadata': {FOLLOWUP_HOLDER_METADATA_KEY: _MARKER}}
_PLAIN_ROW = {'id': 10, 'status': 'pending', 'metadata': dict(_FOLLOWUP)}


@pytest.mark.parametrize('status', ['done', 'cancelled', 'deferred'])
def test_leaving_pending_for_a_protected_status_is_refused(status: str) -> None:
    assert protected_status(_HOLDER_ROW, status, release_holder=False) is HolderProtection.STATUS


@pytest.mark.parametrize('status', ['done', 'cancelled', 'deferred'])
def test_an_explicit_release_lifts_the_status_protection(status: str) -> None:
    assert protected_status(_HOLDER_ROW, status, release_holder=True) is None


@pytest.mark.parametrize('release_holder', ['yes', 'true', 1], ids=['yes', 'true', 'one'])
def test_only_a_literal_true_releases(release_holder: Any) -> None:
    assert protected_status(_HOLDER_ROW, 'cancelled', release_holder=release_holder) is HolderProtection.STATUS


@pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked'])
def test_an_unprotected_status_is_not_judged(status: str) -> None:
    assert protected_status(_HOLDER_ROW, status, release_holder=False) is None


@pytest.mark.parametrize('status', ['done', 'cancelled', 'deferred', 'pending'])
def test_a_non_holder_row_is_never_protected(status: str) -> None:
    assert protected_status(_PLAIN_ROW, status, release_holder=False) is None


_WATCH = 'The follow-up holder never dispatches.'
_STORED_HOLDER = {
    'task_kind': 'deterministic', 'always_escalates': True, 'execution_class': 'decision',
    'deferred_watch': _WATCH, FOLLOWUP_HOLDER_METADATA_KEY: _MARKER,
}


@pytest.mark.parametrize(('payload', 'replace', 'expected'), [
    ({FOLLOWUP_HOLDER_METADATA_KEY: None}, False, 'MARKER'),
    ({FOLLOWUP_HOLDER_METADATA_KEY: {**_MARKER, 'project_id': 'other'}}, False, 'MARKER'),
    ({'deferred_watch': None}, False, 'DEFERRED_WATCH'),
    ({'deferred_watch': ''}, False, 'DEFERRED_WATCH'),
    ({'trigger_met': True}, False, 'TRIGGER_MET'),
    ({key: value for key, value in _STORED_HOLDER.items() if key != FOLLOWUP_HOLDER_METADATA_KEY}, True, 'MARKER'),
    ({key: value for key, value in _STORED_HOLDER.items() if key != 'deferred_watch'}, True, 'DEFERRED_WATCH'),
    ({**_STORED_HOLDER, 'trigger_met': True}, True, 'TRIGGER_MET'),
    (json.dumps({'trigger_met': True}), False, 'TRIGGER_MET'),
], ids=[
    'merge-strips-marker', 'merge-alters-marker', 'merge-strips-watch', 'merge-blanks-watch',
    'merge-sets-trigger-met', 'replace-drops-marker', 'replace-drops-watch',
    'replace-sets-trigger-met', 'json-payload',
])
def test_an_update_that_would_ungate_the_holder_is_refused(payload: Any, replace: bool, expected: str) -> None:
    assert protected_update(_STORED_HOLDER, payload, replace=replace) is HolderProtection[expected]


@pytest.mark.parametrize(('payload', 'replace'), [
    ({**_STORED_HOLDER, 'x_note': 'kept'}, True),
    ({'x_note': 'kept', 'files': ['a.py']}, False),
    ({FOLLOWUP_HOLDER_METADATA_KEY: _MARKER, 'deferred_watch': 'reworded'}, False),
    ({'trigger_met': False}, False),
], ids=['replace-echo-plus-x-key', 'merge-unrelated', 'merge-echo-marker-reword-watch', 'trigger-met-false'])
def test_an_update_that_keeps_the_holder_gated_is_allowed(payload: Any, replace: bool) -> None:
    assert protected_update(_STORED_HOLDER, payload, replace=replace) is None


@pytest.mark.parametrize(('payload', 'replace'), [
    ({'trigger_met': True}, False),
    ({'deferred_watch': None}, False),
    ({FOLLOWUP_HOLDER_METADATA_KEY: None}, False),
    ({}, True),
], ids=['trigger-met', 'strip-watch', 'marker-none', 'replace-empty'])
def test_an_update_to_a_non_holder_is_never_judged(payload: Any, replace: bool) -> None:
    stored = {**_FOLLOWUP, 'deferred_watch': _WATCH}

    assert protected_update(stored, payload, replace=replace) is None


def test_the_refusal_carries_the_holder_and_its_held_count() -> None:
    error = FollowupHolderProtectedError(
        holder_id='9', protection=HolderProtection.STATUS, attempted_status='cancelled', held_count=2,
    )

    wire = error.to_error_dict()

    assert {key: wire[key] for key in wire if key != 'hint'} == {
        'success': False,
        'error': 'followup_holder_protected',
        'error_type': 'FollowupHolderProtectedError',
        'holder_id': '9',
        'protection': 'protected_status',
        'attempted_status': 'cancelled',
        'held_count': 2,
    }
    assert 'release_holder=True' in wire['hint']
    assert error.event_payload() == {
        'holder_id': '9', 'attempted_status': 'cancelled', 'protection': 'protected_status', 'held_count': 2,
    }


_HOLD_RECORD = {'holder': 9, 'held_at': _HELD_AT}


def _held_row(status: str = 'pending', *, dependencies: list[Any] | None = None, record: Any = None) -> dict[str, Any]:
    return {
        'id': 12,
        'status': status,
        'dependencies': [3, 9] if dependencies is None else dependencies,
        'metadata': {**_FOLLOWUP, FOLLOWUP_HOLD_METADATA_KEY: _HOLD_RECORD if record is None else record},
    }


@pytest.mark.parametrize('status', ['pending', 'deferred', 'in-progress', 'blocked'])
def test_a_row_with_its_record_and_holder_edge_is_held_by_that_holder(status: str) -> None:
    assert held_by(_held_row(status)) == 9


def test_a_row_whose_metadata_is_a_json_string_is_read_the_same() -> None:
    row = {**_held_row(), 'metadata': json.dumps(_held_row()['metadata'])}

    assert held_by(row) == 9


def test_string_dependency_ids_name_the_holder_edge() -> None:
    assert held_by(_held_row(dependencies=['3', '9'])) == 9


@pytest.mark.parametrize('row', [
    {'id': 12, 'status': 'pending', 'dependencies': [9], 'metadata': dict(_FOLLOWUP)},
    _held_row(dependencies=[3]),
    _held_row(dependencies=[]),
    {**_held_row(), 'dependencies': None},
    _held_row('done'),
    _held_row('cancelled'),
], ids=['record-absent', 'edge-absent', 'no-dependencies', 'dependencies-none', 'done', 'cancelled'])
def test_a_row_without_both_record_and_edge_or_settled_is_not_held(row: dict[str, Any]) -> None:
    assert held_by(row) is None


def test_a_promoted_row_keeps_its_record_but_is_not_held() -> None:
    promoted = {**_HOLD_RECORD, 'promoted': {'by': 'citation', 'at': _HELD_AT, 'ref': 'fk-1'}}

    assert held_by(_held_row(dependencies=[3], record=promoted)) is None


@pytest.mark.parametrize('record', [
    {'holder': 'x', 'held_at': _HELD_AT},
    {**_HOLD_RECORD, 'extra': 1},
    'held',
], ids=['holder-not-int', 'extra-key', 'not-an-object'])
def test_a_malformed_record_is_not_held_and_warns(record: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level('WARNING', logger='fused_memory.middleware.followup_hold'):
        assert held_by(_held_row(record=record)) is None

    assert any('task 12 carries' in message for message in caplog.messages)


@pytest.mark.parametrize(('payload', 'expected'), [
    ({'x_finding_key': 'fk-1'}, 'fk-1'),
    ({'x_finding_key': ['fk-1', 'fk-2']}, 'fk-1'),
    ({'prd_path': 'plans/x-prd.md'}, 'plans/x-prd.md'),
    ({'x_finding_key': 'fk-1', 'prd_path': 'plans/x-prd.md'}, 'fk-1'),
    (json.dumps({'x_finding_key': ['fk-1']}), 'fk-1'),
    ({}, None),
    (None, None),
    ({'x_finding_key': []}, None),
    ({'x_finding_key': ''}, None),
    ({'prd_path': ''}, None),
    ({'x_finding_key': 7}, None),
    ({'prd_path': ['plans/x-prd.md']}, None),
], ids=[
    'finding-key', 'finding-key-list', 'prd-path', 'finding-key-wins', 'json-string', 'empty',
    'none', 'empty-list', 'empty-string', 'empty-prd-path', 'non-string-key', 'non-string-prd-path',
])
def test_citation_ref_names_the_first_citation(payload: Any, expected: str | None) -> None:
    assert citation_ref(payload) == expected


@pytest.mark.parametrize('status', ['done', 'cancelled', 'deferred'])
def test_an_explicit_release_of_the_holder_releases_its_held_tasks(status: str) -> None:
    assert releases_holder(_HOLDER_ROW, status, release_holder=True)


@pytest.mark.parametrize(('task', 'status', 'release_holder'), [
    (_HOLDER_ROW, 'cancelled', False),
    (_HOLDER_ROW, 'cancelled', 'true'),
    (_PLAIN_ROW, 'cancelled', True),
    (_HOLDER_ROW, 'pending', True),
], ids=['not-released', 'truthy-string', 'non-holder', 'pending'])
def test_anything_short_of_an_explicit_release_releases_nothing(
    task: dict[str, Any], status: str, release_holder: Any,
) -> None:
    assert not releases_holder(task, status, release_holder=release_holder)


@pytest.mark.parametrize('agent_id', [None, 'claude-task-8418-architect'], ids=['header-less', 'agent-session'])
def test_a_human_actor_may_strip_the_holder_edge(agent_id: str | None) -> None:
    assert may_strip_holder_edge(agent_id)


@pytest.mark.parametrize('agent_id', [
    'orchestrator', 'harness-x', 'steward-1', 'recon-stage-2', 'escalation-watcher',
    'orchestrator-deterministic',
])
def test_a_machine_actor_may_not_strip_the_holder_edge(agent_id: str) -> None:
    assert not may_strip_holder_edge(agent_id)


def test_a_promotion_deferral_refuses_an_extra_key() -> None:
    with pytest.raises(ValidationError):
        PromotionDeferral.model_validate({'guard': 'g', 'reason': 'r', 'extra': 'x'})


_STORED_HELD = {FOLLOWUP_HOLD_METADATA_KEY: _HOLD_RECORD, 'files': ['a.py']}


@pytest.mark.parametrize(('payload', 'replace', 'expected'), [
    ({'x_finding_key': ['fk-1']}, True, {'x_finding_key': ['fk-1'], FOLLOWUP_HOLD_METADATA_KEY: _HOLD_RECORD}),
    ({}, True, {FOLLOWUP_HOLD_METADATA_KEY: _HOLD_RECORD}),
    ({FOLLOWUP_HOLD_METADATA_KEY: None}, False, {FOLLOWUP_HOLD_METADATA_KEY: _HOLD_RECORD}),
    (
        {FOLLOWUP_HOLD_METADATA_KEY: {**_HOLD_RECORD, 'holder': 3}, 'files': ['b.py']}, False,
        {FOLLOWUP_HOLD_METADATA_KEY: _HOLD_RECORD, 'files': ['b.py']},
    ),
], ids=['replace-drops-it', 'replace-with-nothing', 'merge-nulls-it', 'merge-rewrites-it'])
def test_an_update_that_would_drop_or_rewrite_the_hold_record_carries_it(
    payload: dict[str, Any], replace: bool, expected: dict[str, Any],
) -> None:
    assert carry_hold_record(_STORED_HELD, payload, replace=replace) == expected


@pytest.mark.parametrize(('stored', 'payload', 'replace'), [
    (_STORED_HELD, {'x_finding_key': ['fk-1']}, False),
    (_STORED_HELD, {**_STORED_HELD, 'files': ['b.py']}, True),
    (json.dumps(_STORED_HELD), {'files': ['b.py']}, False),
    ({'files': ['a.py']}, {'files': ['b.py']}, True),
    (None, {}, True),
], ids=['merge-without-it', 'replace-echoing-it', 'json-stored', 'never-held', 'no-metadata'])
def test_an_update_that_leaves_the_hold_record_carries_nothing(stored: Any, payload: dict[str, Any], replace: bool) -> None:
    assert carry_hold_record(stored, payload, replace=replace) is None
