"""The feed contract's value shapes (plans/client-feed-prd.md §7.1-7.5), built the way a JSON reader builds them."""

import copy
import re
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError
from shared.task_statuses import TaskStatus as FactoryTaskStatus

from client_feed import contract

SHA = 'a' * 40
AT = '2026-10-14T12:00:00Z'

PAYLOADS: dict[str, dict[str, Any]] = {
    'heartbeat': {
        'head': SHA,
        'as_of': AT,
        'engagement_version': 3,
        'watermark_seq': 0,
        'degraded': [{'source': 'archive', 'reason': 'archive dir unreadable'}],
    },
    'factory_state': {
        'scheduler': 'running',
        'since': AT,
        'last_event_at': AT,
        'idle_seconds': 12,
    },
    'prd_upserted': {
        'prd_path': 'plans/ledger-prd.md',
        'title': 'Ledger',
        'excerpt': 'A ledger for the client.',
        'task_counts': {'pending': 2, 'done': 1},
    },
    'task_upserted': {
        'task_id': 5,
        'title': 'Write the ledger',
        'status': 'blocked',
        'waiting_on': 'leo',
        'updated_at': AT,
        'depends_on': [3],
        'prd_path': 'plans/ledger-prd.md',
        'prd_task_label': 'alpha',
        'landed_commit': None,
    },
    'task_phase': {'task_id': 5, 'phase': 'plan', 'entered': True},
    'verify_run': {
        'run_key': 'task-5-verify-1',
        'task_id': 5,
        'kind': 'task',
        'passed': True,
        'duration_ms': 1200,
        'commit': SHA,
        'legs': [{'name': 'pytest', 'category': 'test', 'rc': 0}],
    },
    'test_results': {
        'run_key': 'task-5-verify-1',
        'counts': {'passed': 2, 'failed': 1, 'skipped': 0, 'errored': 0, 'truncated': 1},
        'duration_ms': 900,
        'cases': [
            {
                'id': 'tests/test_ledger.py::test_ok',
                'outcome': 'passed',
                'duration_ms': 10,
                'criterion': 'C-1',
                'exception_type': None,
            },
            {
                'id': 'tests/test_ledger.py::test_bad',
                'outcome': 'failed',
                'duration_ms': 20,
                'criterion': None,
                'exception_type': 'AssertionError',
            },
        ],
    },
    'merge_landed': {
        'commit': SHA,
        'task_ids': [5],
        'titles': ['Write the ledger'],
        'files_changed': 3,
        'insertions': 40,
        'deletions': 2,
        'review': {
            'cycles': 2,
            'blocking_count': 0,
            'suggestion_count': 3,
            'by_category': {'naming': 2, 'tests': 1},
        },
    },
    'branch_snapshot': {
        'branches': [{'name': 'task/6', 'task_id': 6, 'ahead': 2, 'last_commit_at': AT}],
    },
    'tree_snapshot': {
        'commit': SHA,
        'totals': {'files': 2, 'production_lines': 30, 'test_lines': 10, 'docs_lines': 0},
        'files': [
            {
                'path': 'src/ledger.py',
                'kind': 'production',
                'lines': 30,
                'bytes': 900,
                'last_changed_at': AT,
            },
            {
                'path': 'tests/test_ledger.py',
                'kind': 'test',
                'lines': 10,
                'bytes': 300,
                'last_changed_at': None,
            },
        ],
        'budget': {
            'proportion': 0.25,
            'counted_lines': 30,
            'budget_lines': 7,
            'spent_lines': 0,
            'floor_lines': 50,
        },
    },
    'file_revealed': {
        'path': 'src/ledger.py',
        'commit': SHA,
        'lines': 30,
        'content': 'x = 1\n',
        'deleted': False,
        'renamed_from': None,
    },
    'reveal_refused': {'path': '../etc/passwd', 'reason': 'not_in_tree'},
    'criteria_results': {
        'run_kind': 'verify',
        'commit': SHA,
        'run_at': AT,
        'results': [
            {'id': 'C-1', 'outcome': 'passed', 'tests': ['tests/test_ledger.py::test_ok']},
        ],
    },
    'criteria_declared': {
        'criteria': [
            {
                'id': 'C-1',
                'prd': 'plans/ledger-prd.md',
                'capability': 'Ledger',
                'statement': 'Every entry shows the running balance after it.',
                'kind': 'success',
            },
            {
                'id': 'C-2',
                'prd': 'plans/ledger-prd.md',
                'capability': 'Ledger',
                'statement': 'A balance is never rounded before the last entry.',
                'kind': 'failure',
            },
        ],
    },
}

MODEL_NAMES: dict[str, str] = {
    'heartbeat': 'Heartbeat',
    'factory_state': 'FactoryState',
    'prd_upserted': 'PrdUpserted',
    'task_upserted': 'TaskUpserted',
    'task_phase': 'TaskPhase',
    'verify_run': 'VerifyRun',
    'test_results': 'VerifyTestResults',
    'merge_landed': 'MergeLanded',
    'branch_snapshot': 'BranchSnapshot',
    'tree_snapshot': 'TreeSnapshot',
    'file_revealed': 'FileRevealed',
    'reveal_refused': 'RevealRefused',
    'criteria_results': 'CriteriaResults',
    'criteria_declared': 'CriteriaDeclared',
}

EMPTY_STATE: dict[str, Any] = {
    'project_id': 'demo-client',
    'feed_version': '1.0',
    'watermark_seq': 0,
    'as_of': None,
    'head': None,
    'engagement_version': None,
    'degraded': [],
    'factory': None,
    'prds': {},
    'tasks': {},
    'in_flight': [],
    'branches': None,
    'tree': None,
    'revealed': [],
    'criteria': {},
    'criteria_declared': {},
    'recent': [],
}


def event_dict(wire_type: str, seq: int = 1, **overrides: Any) -> dict[str, Any]:
    return {
        'seq': seq,
        'at': AT,
        'project_id': 'demo-client',
        'feed_version': '1.0',
        'type': wire_type,
        **copy.deepcopy(PAYLOADS[wire_type]),
        **overrides,
    }


def build(wire_type: str, **overrides: Any):
    return contract.EVENT_MODELS[wire_type].model_validate(event_dict(wire_type, **overrides))


def first_error_message(exc: ValidationError) -> str:
    return exc.errors()[0]['msg']


NESTED_SAMPLES: list[tuple[str, dict[str, Any]]] = [
    ('Degradation', PAYLOADS['heartbeat']['degraded'][0]),
    ('VerifyLeg', PAYLOADS['verify_run']['legs'][0]),
    ('CaseCounts', PAYLOADS['test_results']['counts']),
    ('CaseResult', PAYLOADS['test_results']['cases'][0]),
    ('ReviewCounts', PAYLOADS['merge_landed']['review']),
    ('Branch', PAYLOADS['branch_snapshot']['branches'][0]),
    ('TreeTotals', PAYLOADS['tree_snapshot']['totals']),
    ('TreeFile', PAYLOADS['tree_snapshot']['files'][0]),
    ('RevealBudget', PAYLOADS['tree_snapshot']['budget']),
    ('CriterionResult', PAYLOADS['criteria_results']['results'][0]),
    ('Criterion', PAYLOADS['criteria_declared']['criteria'][0]),
    ('InFlightTask', {'task_id': 7, 'phase': 'plan', 'since': AT}),
    ('RevealedFile', {'path': 'src/ledger.py', 'commit': SHA, 'lines': 30}),
    ('RevealRequested', {'seq': 1, 'at': AT, 'type': 'reveal_requested', 'path': 'src/a.py'}),
    (
        'EngagementProjection',
        {
            'engagement_version': 3,
            'project_id': 'demo-client',
            'stage': 'build',
            'delivered_commit': None,
        },
    ),
    ('FeedState', EMPTY_STATE),
]


def test_event_models_cover_exactly_the_named_wire_types() -> None:
    assert set(contract.EVENT_MODELS) == set(MODEL_NAMES)
    for wire_type, name in MODEL_NAMES.items():
        assert contract.EVENT_MODELS[wire_type] is getattr(contract, name)


def test_feed_version_is_one_dot_zero_and_its_major_is_derived() -> None:
    assert contract.FEED_VERSION == '1.0'
    assert contract.FEED_MAJOR == contract.feed_major(contract.FEED_VERSION) == 1
    assert contract.feed_major('2.7') == 2


@pytest.mark.parametrize('wire_type', sorted(MODEL_NAMES))
def test_event_round_trips_through_json(wire_type: str) -> None:
    event = build(wire_type)
    assert type(event).model_validate_json(event.model_dump_json()) == event


@pytest.mark.parametrize('wire_type', sorted(MODEL_NAMES))
def test_event_rejects_an_unknown_key(wire_type: str) -> None:
    with pytest.raises(ValidationError, match='surprise'):
        build(wire_type, surprise=1)


@pytest.mark.parametrize('wire_type', sorted(MODEL_NAMES))
def test_event_refuses_attribute_assignment(wire_type: str) -> None:
    event = build(wire_type)
    with pytest.raises(ValidationError):
        event.seq = 99


@pytest.mark.parametrize(('name', 'sample'), NESTED_SAMPLES, ids=[n for n, _ in NESTED_SAMPLES])
def test_record_model_validates_its_sample_and_rejects_an_unknown_key(
    name: str, sample: dict[str, Any]
) -> None:
    model: type[BaseModel] = getattr(contract, name)
    instance = model.model_validate(sample)
    assert model.model_validate_json(instance.model_dump_json()) == instance
    with pytest.raises(ValidationError, match='surprise'):
        model.model_validate({**sample, 'surprise': 1})


@pytest.mark.parametrize(('name', 'sample'), NESTED_SAMPLES, ids=[n for n, _ in NESTED_SAMPLES])
def test_record_model_is_frozen(name: str, sample: dict[str, Any]) -> None:
    model: type[BaseModel] = getattr(contract, name)
    instance = model.model_validate(sample)
    first_field = next(iter(type(instance).model_fields))
    with pytest.raises(ValidationError):
        setattr(instance, first_field, getattr(instance, first_field))


@pytest.mark.parametrize('text', ['2026-10-14T12:00:00Z', '2026-10-14T12:00:00+00:00'])
def test_utc_timestamp_accepts_utc(text: str) -> None:
    offset = TypeAdapter(contract.UtcTimestamp).validate_python(text).utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


@pytest.mark.parametrize('text', ['2026-10-14T12:00:00', '2026-10-14T12:00:00+02:00'])
def test_utc_timestamp_rejects_naive_and_offset_times(text: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(contract.UtcTimestamp).validate_python(text)


@pytest.mark.parametrize('sha', ['0123456789abcdef' * 2 + '01234567', 'f' * 64])
def test_commit_sha_accepts_40_and_64_lowercase_hex(sha: str) -> None:
    assert TypeAdapter(contract.CommitSha).validate_python(sha) == sha


@pytest.mark.parametrize('sha', ['a' * 39, 'a' * 41, 'A' * 40, 'g' * 40, 'a' * 63, ''])
def test_commit_sha_rejects_anything_else(sha: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(contract.CommitSha).validate_python(sha)


def test_repo_path_accepts_a_relative_path() -> None:
    assert TypeAdapter(contract.RepoPath).validate_python('src/a.py') == 'src/a.py'


@pytest.mark.parametrize(
    'path', ['../x', 'a/../b', '/etc/passwd', '', 'a//b', './a', '.', '..', 'a/']
)
def test_repo_path_rejects_escapes_absolutes_and_empty_segments(path: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(contract.RepoPath).validate_python(path)


@pytest.mark.parametrize('path', ['../x', '/etc/passwd'])
def test_refusals_and_requests_carry_the_raw_requested_path(path: str) -> None:
    refused = contract.RevealRefused.model_validate(event_dict('reveal_refused', path=path))
    assert refused.path == path
    requested = contract.RevealRequested.model_validate(
        {'seq': 1, 'at': AT, 'type': 'reveal_requested', 'path': path}
    )
    assert requested.path == path


@pytest.mark.parametrize(
    ('deleted', 'content'), [(True, None), (False, 'x = 1\n'), (False, '')]
)
def test_file_revealed_accepts_a_consistent_deleted_and_content(
    deleted: bool, content: str | None
) -> None:
    build('file_revealed', deleted=deleted, content=content)


@pytest.mark.parametrize(('deleted', 'content'), [(True, 'x = 1\n'), (False, None)])
def test_file_revealed_refuses_an_inconsistent_deleted_and_content(
    deleted: bool, content: str | None
) -> None:
    with pytest.raises(ValidationError) as caught:
        build('file_revealed', deleted=deleted, content=content)
    message = first_error_message(caught.value)
    assert 'deleted' in message
    assert 'content' in message
    assert str(deleted) in message


def test_test_results_refuses_counts_that_do_not_cover_the_cases() -> None:
    counts = {'passed': 4, 'failed': 3, 'skipped': 0, 'errored': 0, 'truncated': 1}
    with pytest.raises(ValidationError) as caught:
        build('test_results', counts=counts)
    message = first_error_message(caught.value)
    assert re.search(r'\b7\b', message)
    assert re.search(r'\b3\b', message)
    assert 'cases' in message
    assert 'truncated' in message


def test_merge_landed_refuses_titles_that_do_not_pair_with_task_ids() -> None:
    with pytest.raises(ValidationError) as caught:
        build('merge_landed', titles=['one', 'two'], task_ids=[5])
    message = first_error_message(caught.value)
    assert 'titles' in message
    assert 'task_ids' in message
    assert re.search(r'\b2\b', message)
    assert re.search(r'\b1\b', message)


def test_criteria_declared_refuses_a_duplicate_id() -> None:
    first, second = PAYLOADS['criteria_declared']['criteria']
    with pytest.raises(ValidationError) as caught:
        build('criteria_declared', criteria=[first, {**second, 'id': 'C-1'}])
    message = first_error_message(caught.value)
    assert 'C-1' in message
    assert 'unique' in message


def test_criteria_declared_accepts_an_empty_declaration() -> None:
    declaration = contract.CriteriaDeclared.model_validate(
        event_dict('criteria_declared', criteria=[])
    )
    assert declaration.criteria == ()


@pytest.mark.parametrize(
    'override',
    [
        {'kind': 'maybe'},
        {'prd': '../x.md'},
        {'prd': '/abs.md'},
        {'id': ''},
        {'statement': ''},
        {'capability': ''},
    ],
    ids=['unknown-kind', 'escaping-prd', 'absolute-prd', 'empty-id', 'empty-statement', 'empty-capability'],
)
def test_a_criterion_refuses_a_bad_field(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        contract.Criterion.model_validate({**PAYLOADS['criteria_declared']['criteria'][0], **override})


@pytest.mark.parametrize(('wire_type', 'field'), [('verify_run', 'legs'), ('merge_landed', 'review')])
def test_not_recorded_is_the_one_accepted_string(wire_type: str, field: str) -> None:
    event = build(wire_type, **{field: 'not_recorded'})
    assert getattr(event, field) == contract.NOT_RECORDED == 'not_recorded'
    for other in ('not recorded', 'unknown', 'passed', ''):
        with pytest.raises(ValidationError):
            build(wire_type, **{field: other})


@pytest.mark.parametrize('seq', [True, 1.0, '1', 0, -1])
def test_seq_is_a_strict_positive_int(seq: object) -> None:
    with pytest.raises(ValidationError):
        build('heartbeat', seq=seq)


@pytest.mark.parametrize('project_id', ['Bad Id', '', '-lead', 'x' * 65, 'UPPER'])
def test_project_id_is_a_slug(project_id: str) -> None:
    with pytest.raises(ValidationError):
        build('heartbeat', project_id=project_id)


@pytest.mark.parametrize('feed_version', ['1', 'v1.0', '1.0.0', ''])
def test_feed_version_is_major_dot_minor(feed_version: str) -> None:
    with pytest.raises(ValidationError):
        build('heartbeat', feed_version=feed_version)


def test_an_event_refuses_another_wire_type() -> None:
    with pytest.raises(ValidationError):
        contract.Heartbeat.model_validate(event_dict('heartbeat', type='factory_state'))


def test_task_status_vocabulary_matches_the_factory() -> None:
    assert {s.value for s in contract.TaskStatus} == {s.value for s in FactoryTaskStatus}


@pytest.mark.parametrize(
    ('enum_name', 'values'),
    [
        ('Source', {'tasks', 'events', 'git', 'archive', 'prds', 'criteria', 'engagement', 'inbox'}),
        ('SchedulerState', {'running', 'paused'}),
        ('WaitingOn', {'factory', 'leo'}),
        ('VerifyKind', {'task', 'merge'}),
        ('CaseOutcome', {'passed', 'failed', 'skipped', 'errored'}),
        ('FileKind', {'production', 'test', 'docs', 'generated'}),
        (
            'RefusalReason',
            {'not_in_tree', 'not_visible', 'not_revealable', 'over_budget', 'symlink'},
        ),
        ('RunKind', {'verify', 'staging'}),
        ('CriterionOutcome', {'passed', 'failed', 'not_run'}),
        ('CriterionKind', {'success', 'failure'}),
    ],
)
def test_feed_owned_vocabularies(enum_name: str, values: set[str]) -> None:
    assert {member.value for member in getattr(contract, enum_name)} == values


def test_feed_state_validates_an_empty_state_and_rejects_unknown_keys() -> None:
    state = contract.FeedState.model_validate(EMPTY_STATE)
    assert state.watermark_seq == 0
    assert state.recent == ()
    with pytest.raises(ValidationError, match='surprise'):
        contract.FeedState.model_validate({**EMPTY_STATE, 'surprise': 1})


def test_feed_state_round_trips_with_every_section_filled() -> None:
    task = event_dict('task_upserted', seq=4)
    prd = event_dict('prd_upserted', seq=3)
    criteria = event_dict('criteria_results', seq=9)
    state = contract.FeedState.model_validate(
        {
            **EMPTY_STATE,
            'watermark_seq': 9,
            'as_of': AT,
            'head': SHA,
            'engagement_version': 3,
            'degraded': PAYLOADS['heartbeat']['degraded'],
            'factory': event_dict('factory_state', seq=2),
            'prds': {prd['prd_path']: prd},
            'tasks': {task['task_id']: task},
            'in_flight': [{'task_id': 5, 'phase': 'plan', 'since': AT}],
            'branches': event_dict('branch_snapshot', seq=6),
            'tree': event_dict('tree_snapshot', seq=7),
            'revealed': [{'path': 'src/ledger.py', 'commit': SHA, 'lines': 30}],
            'criteria': {'verify': criteria},
            'criteria_declared': {'C-1': PAYLOADS['criteria_declared']['criteria'][0]},
            'recent': [prd, task, event_dict('reveal_refused', seq=8), criteria],
        }
    )
    assert contract.FeedState.model_validate_json(state.model_dump_json()) == state
    assert [type(event).__name__ for event in state.recent] == [
        'PrdUpserted',
        'TaskUpserted',
        'RevealRefused',
        'CriteriaResults',
    ]


@pytest.mark.parametrize(
    ('wire_type', 'key'),
    [
        ('task_upserted', 5),
        ('prd_upserted', 'plans/ledger-prd.md'),
        ('file_revealed', 'src/ledger.py'),
        ('verify_run', 'task-5-verify-1'),
        ('merge_landed', SHA),
        ('test_results', 'task-5-verify-1'),
    ],
)
def test_keyed_events_expose_their_natural_key(wire_type: str, key: object) -> None:
    assert build(wire_type).natural_key == key


@pytest.mark.parametrize(
    'wire_type',
    [
        'heartbeat',
        'factory_state',
        'task_phase',
        'branch_snapshot',
        'tree_snapshot',
        'reveal_refused',
        'criteria_results',
        'criteria_declared',
    ],
)
def test_unkeyed_events_have_no_natural_key(wire_type: str) -> None:
    assert build(wire_type).natural_key is None
