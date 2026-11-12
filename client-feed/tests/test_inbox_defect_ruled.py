"""B37: the inbox contract's third member, defect_ruled (PRD §14.2 decision 14, §14.3), against hand-written lines; the portal's vendored in-v3 fixture arrives with σ3 (task 7407)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _in_dir import write_engagement
from _publisher_pass import REVEALABLE_PATH, inbox_offset, reveal_answers, reveal_policy

from client_feed.contract import (
    INBOX_FILENAME,
    ContractViolation,
    DefectRuled,
    RulingKind,
    parse_inbox_line,
)
from client_feed.feed_io import OutDir
from client_feed.inbox import read_inbox
from client_feed.policy import Policy
from client_feed.publisher import run_pass

PROJECT = 'demo-client'
T0 = datetime(2026, 11, 3, 9, 0, tzinfo=UTC)
NOW = datetime(2026, 11, 3, 13, 0, tzinfo=UTC)
AT = '2026-11-03T12:00:00Z'

RULED: dict[str, Any] = {
    'seq': 3,
    'at': AT,
    'type': 'defect_ruled',
    'block': 2,
    'report': 1,
    'kind': 'not_a_defect',
}
REPORTED: dict[str, Any] = {
    'seq': 2,
    'at': AT,
    'type': 'defect_reported',
    'block': 2,
    'report': 1,
    'criterion_id': 'C-12',
    'restater': {'prompt_version': 'restate-v1'},
    'outcome': {'kind': 'restated', 'observed': 'o', 'expected': 'e', 'steps': ['s']},
}


def requested(seq: int, path: str) -> dict[str, Any]:
    return {'seq': seq, 'at': AT, 'type': 'reveal_requested', 'path': path}


def write_inbox(in_dir: Path, *records: dict[str, Any]) -> Path:
    inbox = in_dir / INBOX_FILENAME
    inbox.write_text(''.join(f'{json.dumps(record)}\n' for record in records))
    return inbox


@pytest.mark.parametrize('kind', ['not_a_defect', 'not_covered', 'change_request'])
def test_a_ruled_line_parses_to_defect_ruled(kind: str) -> None:
    parsed = parse_inbox_line(json.dumps({**RULED, 'kind': kind}), line_no=1)

    assert isinstance(parsed, DefectRuled)
    assert parsed.kind is RulingKind(kind)
    assert (parsed.seq, parsed.block, parsed.report) == (3, 2, 1)


@pytest.mark.parametrize(
    ('record', 'field'),
    [
        pytest.param({**RULED, 'note': 'out of scope'}, 'note', id='note'),
        pytest.param({**RULED, 'text': 'raw portal words'}, 'text', id='text'),
        pytest.param({**RULED, 'kind': 'duplicate'}, 'kind', id='fourth-kind'),
        pytest.param({**RULED, 'report': 0}, 'report', id='report-zero'),
        pytest.param({**RULED, 'block': 0}, 'block', id='block-zero'),
        pytest.param({**RULED, 'report': True}, 'report', id='report-bool'),
        pytest.param(
            {key: value for key, value in RULED.items() if key != 'kind'}, 'kind', id='no-kind'
        ),
        pytest.param({**RULED, 'project_id': PROJECT}, 'project_id', id='feed-envelope-key'),
    ],
)
def test_a_ruled_line_off_the_contract_is_a_violation_naming_its_field(
    record: dict[str, Any], field: str
) -> None:
    with pytest.raises(ContractViolation) as caught:
        parse_inbox_line(json.dumps(record), line_no=2)

    assert caught.value.line_no == 2
    assert field in caught.value.detail


@pytest.fixture
def in_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'in'
    write_engagement(path, project_id=PROJECT)
    return path


def test_a_ruled_line_is_read_beside_requests_and_reports_in_inbox_order(in_dir: Path) -> None:
    inbox = write_inbox(
        in_dir, requested(1, 'src/ledger.py'), REPORTED, RULED, requested(4, 'src/reports.py')
    )

    read = read_inbox(in_dir, offset=0)

    assert [line.seq for line in read.lines] == [1, 2, 3, 4]
    assert read.rejected == ()
    assert [request.seq for request in read.requests] == [1, 4]
    assert [report.seq for report in read.reports] == [2]
    assert [(ruling.block, ruling.report, ruling.kind) for ruling in read.rulings] == [
        (2, 1, RulingKind.NOT_A_DEFECT)
    ]
    assert read.offset == inbox.stat().st_size


def test_a_ruled_line_carrying_a_note_is_rejected_by_offset_and_the_tail_moves_on(
    in_dir: Path,
) -> None:
    first = requested(1, 'src/ledger.py')
    inbox = write_inbox(
        in_dir, first, {**RULED, 'seq': 2, 'note': 'out of scope'}, requested(3, 'src/ledger.py')
    )

    read = read_inbox(in_dir, offset=0)

    assert [line.seq for line in read.lines] == [1, 3]
    assert read.rulings == ()
    assert [(rejected.offset, 'note' in rejected.detail) for rejected in read.rejected] == [
        (len(f'{json.dumps(first)}\n'.encode()), True)
    ]
    assert read.offset == inbox.stat().st_size


@pytest.fixture
def policy(tmp_path: Path, in_dir: Path) -> Policy:
    return reveal_policy(tmp_path, in_dir, project_id=PROJECT, committed_at=T0)


def test_the_reveal_path_answers_only_the_requests_around_a_ruled_line(
    policy: Policy, in_dir: Path
) -> None:
    inbox = write_inbox(
        in_dir, requested(1, REVEALABLE_PATH), {**RULED, 'seq': 2}, requested(3, 'src/absent.py')
    )

    report = run_pass(policy, OutDir.open(policy.out_dir, project_id=PROJECT), now=NOW)

    assert report.degraded == ()
    assert [(line.type, line.path) for line in reveal_answers(policy.out_dir)] == [
        ('file_revealed', REVEALABLE_PATH),
        ('reveal_refused', 'src/absent.py'),
    ]
    assert inbox_offset(policy.out_dir) == inbox.stat().st_size


def test_a_pass_over_an_inbox_of_only_a_ruled_line_moves_the_offset_past_it(
    policy: Policy, in_dir: Path
) -> None:
    inbox = write_inbox(in_dir, RULED)

    report = run_pass(policy, OutDir.open(policy.out_dir, project_id=PROJECT), now=NOW)

    assert report.degraded == ()
    assert reveal_answers(policy.out_dir) == []
    assert inbox_offset(policy.out_dir) == inbox.stat().st_size
