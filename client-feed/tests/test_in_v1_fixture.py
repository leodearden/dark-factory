"""The portal's vendored in-v1 files, read and answered by this publisher as the portal writes them."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _git_repo import make_repo
from _stores import RunsDb, TasksDb

from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    INBOX_FILENAME,
    FileRevealed,
    RefusalReason,
    RevealRefused,
    Source,
    parse_engagement,
)
from client_feed.feed_io import CURSOR_FILENAME, Cursor, OutDir, read_feed_lines
from client_feed.inbox import read_inbox
from client_feed.policy import Policy
from client_feed.publisher import PassReport, run_pass
from client_feed.validate import validate_out_dir

IN_V1 = Path(__file__).parent / 'fixtures' / 'in-v1'
PROJECT = 'demo-client'
T0 = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 21, 10, 0, tzinfo=UTC)
LEDGER = ''.join(f'entry_{index} = {index}\n' for index in range(60))


@pytest.fixture
def in_dir(tmp_path: Path) -> Path:
    copied = tmp_path / 'in'
    copied.mkdir()
    shutil.copyfile(IN_V1 / INBOX_FILENAME, copied / INBOX_FILENAME)
    shutil.copyfile(IN_V1 / ENGAGEMENT_FILENAME, copied / ENGAGEMENT_FILENAME)
    return copied


@pytest.fixture
def policy(tmp_path: Path, in_dir: Path) -> Policy:
    repo = make_repo(tmp_path / 'repo')
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    repo.write('src/ledger.py', LEDGER)
    repo.commit('c1', when=T0)
    TasksDb.create(repo.root)
    RunsDb.create(repo.root)
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


def _rebase_delivered_to_null(in_dir: Path) -> None:
    path = in_dir / ENGAGEMENT_FILENAME
    vendored = parse_engagement(path.read_text())
    path.write_text(vendored.model_copy(update={'delivered_commit': None}).model_dump_json())


def _pass(policy: Policy) -> PassReport:
    return run_pass(policy, OutDir.open(policy.out_dir, project_id=PROJECT), now=NOW)


def _inbox_offset(policy: Policy) -> int:
    return Cursor.model_validate_json((policy.out_dir / CURSOR_FILENAME).read_text()).inbox_offset


def test_the_vendored_inbox_holds_two_requests_and_nothing_rejected(in_dir: Path) -> None:
    read = read_inbox(in_dir, offset=0)

    assert [request.path for request in read.requests] == ['src/ledger.py', 'src/reports.py']
    assert read.rejected == ()
    assert read.offset == (in_dir / INBOX_FILENAME).stat().st_size


def test_each_vendored_request_is_answered_once_the_engagement_is_usable(
    policy: Policy, in_dir: Path
) -> None:
    _rebase_delivered_to_null(in_dir)

    report = _pass(policy)

    assert report.degraded == ()
    answers = [
        line for line in read_feed_lines(policy.out_dir) if isinstance(line, FileRevealed | RevealRefused)
    ]
    assert [(line.type, line.path) for line in answers] == [
        ('file_revealed', 'src/ledger.py'),
        ('reveal_refused', 'src/reports.py'),
    ]
    revealed, refused = answers
    assert isinstance(revealed, FileRevealed) and revealed.content == LEDGER
    assert isinstance(refused, RevealRefused) and refused.reason is RefusalReason.NOT_IN_TREE
    assert _inbox_offset(policy) == (in_dir / INBOX_FILENAME).stat().st_size
    assert validate_out_dir(policy.out_dir, expected_project_id=PROJECT).ok


def test_the_vendored_fixture_delivered_commit_disables_reveals_and_holds_the_inbox(
    policy: Policy,
) -> None:
    report = _pass(policy)

    assert [degradation.source for degradation in report.degraded] == [Source.ENGAGEMENT]
    assert not any(
        isinstance(line, FileRevealed | RevealRefused) for line in read_feed_lines(policy.out_dir)
    )
    assert _inbox_offset(policy) == 0
