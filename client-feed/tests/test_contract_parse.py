"""Parsing one feed, inbox or engagement text into a contract model, or into a typed FeedError."""

import json
import re
from typing import Any

import pytest

from client_feed import contract
from client_feed.contract import (
    ContractViolation,
    EngagementProjection,
    EventHeader,
    FeedError,
    RevealRequested,
    UnsupportedFeedVersion,
    parse_engagement,
    parse_feed_line,
    parse_inbox_line,
    parse_state,
    split_jsonl,
)

SHA = 'b' * 40
AT = '2026-10-14T12:00:00Z'

MINIMAL_PAYLOADS: dict[str, dict[str, Any]] = {
    'heartbeat': {
        'head': None,
        'as_of': AT,
        'engagement_version': None,
        'watermark_seq': 0,
        'degraded': [],
    },
    'factory_state': {
        'scheduler': 'paused',
        'since': None,
        'last_event_at': None,
        'idle_seconds': None,
    },
    'prd_upserted': {'prd_path': 'plans/a-prd.md', 'title': 'A', 'excerpt': '', 'task_counts': {}},
    'task_upserted': {
        'task_id': 1,
        'title': 'T',
        'status': 'pending',
        'waiting_on': None,
        'updated_at': AT,
        'depends_on': [],
        'prd_path': None,
        'prd_task_label': None,
        'landed_commit': None,
    },
    'task_phase': {'task_id': 1, 'phase': 'plan', 'entered': True},
    'verify_run': {
        'run_key': 'r1',
        'task_id': None,
        'kind': 'merge',
        'passed': False,
        'duration_ms': None,
        'commit': None,
        'legs': 'not_recorded',
    },
    'test_results': {
        'run_key': 'r1',
        'counts': {'passed': 0, 'failed': 0, 'skipped': 0, 'errored': 0, 'truncated': 0},
        'duration_ms': None,
        'cases': [],
    },
    'merge_landed': {
        'commit': SHA,
        'task_ids': [],
        'titles': [],
        'files_changed': 0,
        'insertions': 0,
        'deletions': 0,
        'review': 'not_recorded',
    },
    'branch_snapshot': {'branches': []},
    'tree_snapshot': {
        'commit': SHA,
        'totals': {'files': 0, 'production_lines': 0, 'test_lines': 0, 'docs_lines': 0},
        'files': [],
        'budget': {
            'proportion': 0.25,
            'counted_lines': 0,
            'budget_lines': 0,
            'spent_lines': 0,
            'floor_lines': 50,
        },
    },
    'file_revealed': {
        'path': 'src/a.py',
        'commit': SHA,
        'lines': 0,
        'content': None,
        'deleted': True,
        'renamed_from': None,
    },
    'reveal_refused': {'path': '/etc/passwd', 'reason': 'not_in_tree'},
    'criteria_results': {'run_kind': 'staging', 'commit': None, 'run_at': AT, 'results': []},
}

ENGAGEMENT = {
    'engagement_version': 2,
    'project_id': 'demo-client',
    'stage': 'build',
    'delivered_commit': None,
}


def envelope(wire_type: str, *, seq: int = 1, feed_version: str = '1.0') -> dict[str, Any]:
    return {
        'seq': seq,
        'at': AT,
        'project_id': 'demo-client',
        'feed_version': feed_version,
        'type': wire_type,
    }


def known_line(wire_type: str, **overrides: Any) -> dict[str, Any]:
    return {**envelope(wire_type), **MINIMAL_PAYLOADS[wire_type], **overrides}


def test_the_minimal_payloads_cover_every_known_type() -> None:
    assert set(MINIMAL_PAYLOADS) == set(contract.EVENT_MODELS)


@pytest.mark.parametrize('wire_type', sorted(MINIMAL_PAYLOADS))
def test_a_known_line_parses_to_its_model(wire_type: str) -> None:
    line = known_line(wire_type)
    parsed = parse_feed_line(json.dumps(line), line_no=1)
    model = contract.EVENT_MODELS[wire_type]
    assert type(parsed) is model
    assert parsed == model.model_validate(line)


def test_a_known_type_at_a_higher_minor_parses_to_its_model() -> None:
    line = {**known_line('task_phase'), 'feed_version': '1.3'}
    assert parse_feed_line(json.dumps(line), line_no=1) == contract.TaskPhase.model_validate(line)


def test_an_unknown_type_under_the_supported_major_parses_to_its_header() -> None:
    line = {
        **envelope('future_minor_event', seq=12, feed_version='1.7'),
        'payload': {'anything': [1, 2]},
        'more': 'unvalidated',
    }
    parsed = parse_feed_line(json.dumps(line), line_no=12)
    assert type(parsed) is EventHeader
    assert parsed == EventHeader.model_validate(envelope('future_minor_event', seq=12, feed_version='1.7'))
    assert (parsed.seq, parsed.project_id, parsed.feed_version, parsed.type) == (
        12,
        'demo-client',
        '1.7',
        'future_minor_event',
    )


@pytest.mark.parametrize(
    'line',
    [
        {**known_line('heartbeat'), 'feed_version': '2.0'},
        {**envelope('future_major_event', feed_version='2.0'), 'payload': 1},
    ],
    ids=['known-type', 'unknown-type'],
)
def test_another_major_is_refused(line: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedFeedVersion) as caught:
        parse_feed_line(json.dumps(line), line_no=7)
    error = caught.value
    assert (error.line_no, error.found, error.supported_major) == (7, '2.0', 1)
    message = str(error)
    assert re.search(r'\b7\b', message)
    assert '2.0' in message
    assert re.search(r'\b1\b', message)


def test_a_known_type_with_an_extra_key_is_a_violation_not_a_header() -> None:
    line = known_line('heartbeat', surprise=1)
    with pytest.raises(ContractViolation) as caught:
        parse_feed_line(json.dumps(line), line_no=4)
    assert caught.value.line_no == 4
    assert 'surprise' in caught.value.detail
    assert re.search(r'\b4\b', str(caught.value))


@pytest.mark.parametrize(
    'text',
    [
        '{"seq": 1,',
        '[1, 2]',
        '',
        '   ',
        '"heartbeat"',
        json.dumps({key: value for key, value in known_line('heartbeat').items() if key != 'seq'}),
    ],
    ids=['not-json', 'array', 'blank', 'whitespace', 'string', 'missing-seq'],
)
def test_a_malformed_line_is_a_violation_at_its_line(text: str) -> None:
    with pytest.raises(ContractViolation) as caught:
        parse_feed_line(text, line_no=9)
    assert caught.value.line_no == 9


@pytest.mark.parametrize(
    'override',
    [
        {'seq': 0},
        {'project_id': 'Bad Id'},
        {'at': '2026-10-14T12:00:00'},
        {'feed_version': 'one'},
        {'type': ''},
    ],
    ids=['seq-zero', 'bad-project-id', 'naive-at', 'bad-version', 'empty-type'],
)
def test_the_header_is_validated_even_for_an_unknown_type(override: dict[str, Any]) -> None:
    line = {**envelope('future_minor_event'), 'payload': 1, **override}
    with pytest.raises(ContractViolation) as caught:
        parse_feed_line(json.dumps(line), line_no=3)
    assert caught.value.line_no == 3


def test_a_line_separator_inside_a_record_does_not_split_it() -> None:
    a = json.dumps({'content': 'first second'}, ensure_ascii=False)
    b = json.dumps({'content': 'third\x85fourth '}, ensure_ascii=False)
    assert ' ' in a and '\x85' in b and ' ' in b
    assert tuple(split_jsonl(a + '\n' + b + '\n')) == (a, b)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('', ()),
        ('{}', ('{}',)),
        ('{}\n', ('{}',)),
        ('{}\n\n{}\n', ('{}', '', '{}')),
        ('{}\n\n', ('{}', '')),
    ],
)
def test_only_the_final_newline_is_dropped(text: str, expected: tuple[str, ...]) -> None:
    assert tuple(split_jsonl(text)) == expected


def test_both_refusals_are_feed_errors() -> None:
    assert issubclass(ContractViolation, FeedError)
    assert issubclass(UnsupportedFeedVersion, FeedError)
    assert issubclass(FeedError, Exception)


def test_an_inbox_request_parses() -> None:
    request = {'seq': 1, 'at': AT, 'type': 'reveal_requested', 'path': '../x'}
    assert parse_inbox_line(json.dumps(request), line_no=1) == RevealRequested.model_validate(request)


@pytest.mark.parametrize(
    'text',
    [
        json.dumps({'seq': 1, 'at': AT, 'type': 'reveal_requested', 'path': 'a', 'x': 1}),
        json.dumps({'seq': 1, 'at': AT, 'type': 'reveal_granted', 'path': 'a'}),
        json.dumps(
            {'seq': 1, 'at': AT, 'type': 'reveal_requested', 'path': 'a', 'project_id': 'p'}
        ),
        'not json',
    ],
    ids=['extra-key', 'other-type', 'feed-envelope-key', 'not-json'],
)
def test_a_bad_inbox_line_is_a_violation(text: str) -> None:
    with pytest.raises(ContractViolation) as caught:
        parse_inbox_line(text, line_no=2)
    assert caught.value.line_no == 2


@pytest.mark.parametrize('delivered_commit', [None, SHA])
def test_an_engagement_projection_parses(delivered_commit: str | None) -> None:
    text = json.dumps({**ENGAGEMENT, 'delivered_commit': delivered_commit})
    parsed = parse_engagement(text)
    assert parsed == EngagementProjection.model_validate(
        {**ENGAGEMENT, 'delivered_commit': delivered_commit}
    )
    assert parsed.stage == 'build'


@pytest.mark.parametrize(
    'text',
    [
        json.dumps({key: value for key, value in ENGAGEMENT.items() if key != 'engagement_version'}),
        json.dumps({**ENGAGEMENT, 'extra': True}),
        '{',
    ],
    ids=['missing-version', 'extra-key', 'not-json'],
)
def test_a_bad_engagement_projection_is_a_violation(text: str) -> None:
    with pytest.raises(ContractViolation) as caught:
        parse_engagement(text)
    assert caught.value.line_no is None


def test_a_state_parses_and_a_bad_one_is_a_whole_file_violation() -> None:
    state = {
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
        'recent': [],
    }
    assert parse_state(json.dumps(state)).project_id == 'demo-client'
    with pytest.raises(ContractViolation) as caught:
        parse_state(json.dumps({**state, 'watermark_seq': -1}))
    assert caught.value.line_no is None
    assert 'watermark_seq' in caught.value.detail
