"""The factory's in-v2 stand-in, read as the portal will write it, and B23's rejection cases."""

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _git_repo import make_repo
from _in_dir import write_engagement
from _stores import RunsDb, TasksDb

from client_feed.contract import (
    INBOX_FILENAME,
    ContractViolation,
    DefectReported,
    FileRevealed,
    Quarantined,
    Restated,
    RevealRefused,
    RevealRequested,
    parse_inbox_line,
    split_jsonl,
)
from client_feed.feed_io import CURSOR_FILENAME, Cursor, OutDir, read_feed_lines
from client_feed.inbox import read_inbox
from client_feed.policy import Policy
from client_feed.publisher import run_pass
from client_feed.validate import validate_out_dir

IN_V2 = Path(__file__).parent / 'fixtures' / 'in-v2-factory'
PROJECT = 'demo-client'
T0 = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)
RESTATED_LINE = 1
QUARANTINED_LINE = 2


def fixture_lines() -> tuple[str, ...]:
    return split_jsonl((IN_V2 / INBOX_FILENAME).read_text())


def mutated(index: int, mutate: Callable[[dict[str, Any]], object]) -> str:
    record = json.loads(fixture_lines()[index])
    mutate(record)
    return json.dumps(record)


@pytest.fixture
def in_dir(tmp_path: Path) -> Path:
    copied = tmp_path / 'in'
    copied.mkdir()
    shutil.copyfile(IN_V2 / INBOX_FILENAME, copied / INBOX_FILENAME)
    return copied


def test_every_fixture_line_is_a_valid_inbox_line() -> None:
    parsed = [parse_inbox_line(text, line_no=n) for n, text in enumerate(fixture_lines(), 1)]

    assert [type(line) for line in parsed] == [
        RevealRequested,
        DefectReported,
        DefectReported,
        RevealRequested,
    ]
    restated, quarantined = parsed[RESTATED_LINE], parsed[QUARANTINED_LINE]
    assert isinstance(restated, DefectReported) and isinstance(restated.outcome, Restated)
    assert isinstance(quarantined, DefectReported) and isinstance(quarantined.outcome, Quarantined)


@pytest.mark.parametrize(
    ('index', 'mutate', 'field'),
    [
        pytest.param(
            RESTATED_LINE, lambda r: r.update(text='raw client words'), 'text', id='raw-text'
        ),
        pytest.param(
            RESTATED_LINE, lambda r: r['outcome'].update(observed=''), 'observed', id='empty-observed'
        ),
        pytest.param(
            RESTATED_LINE,
            lambda r: r['outcome'].update(reasons=['not_restatable']),
            'reasons',
            id='restated-with-reasons',
        ),
        pytest.param(
            QUARANTINED_LINE,
            lambda r: r['outcome'].update(observed='o'),
            'observed',
            id='quarantined-with-observed',
        ),
        pytest.param(RESTATED_LINE, lambda r: r.pop('outcome'), 'outcome', id='no-outcome'),
        pytest.param(
            RESTATED_LINE, lambda r: r['outcome'].pop('kind'), 'outcome', id='outcome-without-kind'
        ),
        pytest.param(
            QUARANTINED_LINE,
            lambda r: r['outcome']['reasons'].append('made_up'),
            'reasons',
            id='unknown-reason',
        ),
        pytest.param(
            RESTATED_LINE,
            lambda r: r.update(criterion_id='C:12'),
            'criterion_id',
            id='colon-criterion-id',
        ),
    ],
)
def test_a_defect_line_off_the_contract_is_refused_naming_its_field(
    index: int, mutate: Callable[[dict[str, Any]], object], field: str
) -> None:
    with pytest.raises(ContractViolation) as caught:
        parse_inbox_line(mutated(index, mutate), line_no=index + 1)

    assert field in caught.value.detail


def test_the_fixture_inbox_reads_as_requests_and_reports_in_order(in_dir: Path) -> None:
    read = read_inbox(in_dir, offset=0)

    assert [request.path for request in read.requests] == ['src/ledger.py', 'src/reports.py']
    assert [(report.block, report.report) for report in read.reports] == [(2, 1), (2, 2)]
    assert [line.seq for line in read.lines] == [1, 2, 3, 4]
    assert read.rejected == ()
    assert read.offset == (in_dir / INBOX_FILENAME).stat().st_size


def test_a_defect_line_carrying_raw_text_is_rejected_and_the_rest_are_read(in_dir: Path) -> None:
    lines = list(fixture_lines())
    lines[RESTATED_LINE] = mutated(RESTATED_LINE, lambda r: r.update(text='raw client words'))
    inbox = in_dir / INBOX_FILENAME
    inbox.write_text(''.join(f'{line}\n' for line in lines))

    read = read_inbox(in_dir, offset=0)

    assert [rejected.offset for rejected in read.rejected] == [len(lines[0].encode()) + 1]
    assert [line.seq for line in read.lines] == [1, 3, 4]
    assert read.offset == inbox.stat().st_size


@pytest.fixture
def policy(tmp_path: Path, in_dir: Path) -> Policy:
    repo = make_repo(tmp_path / 'repo')
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    repo.write('src/ledger.py', 'balance = 100\n')
    repo.commit('c1', when=T0)
    TasksDb.create(repo.root)
    RunsDb.create(repo.root)
    write_engagement(in_dir, project_id=PROJECT)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    return Policy.model_validate(
        {
            'project_id': PROJECT,
            'project_root': str(repo.root),
            'out_dir': str(out_dir),
            'in_dir': str(in_dir),
            'visible_globs': ['src/**'],
            'reveal_proportion': 1.0,
        }
    )


def test_a_publisher_pass_answers_both_requests_and_counts_nothing_under_inbox(
    policy: Policy, in_dir: Path
) -> None:
    report = run_pass(policy, OutDir.open(policy.out_dir, project_id=PROJECT), now=NOW)

    assert report.degraded == ()
    answers = [
        (line.type, line.path)
        for line in read_feed_lines(policy.out_dir)
        if isinstance(line, FileRevealed | RevealRefused)
    ]
    assert answers == [('file_revealed', 'src/ledger.py'), ('reveal_refused', 'src/reports.py')]
    cursor = Cursor.model_validate_json((policy.out_dir / CURSOR_FILENAME).read_text())
    assert cursor.inbox_offset == (in_dir / INBOX_FILENAME).stat().st_size
    assert validate_out_dir(policy.out_dir, expected_project_id=PROJECT).ok
