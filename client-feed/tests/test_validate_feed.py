"""validate-feed on the committed boundary fixtures (PRD B4, B5) and on altered copies of them.

The fixtures are what the portal repo vendors. feed-v1/state.json is the fold of feed-v1/feed.jsonl at
seq 26; feed-v1-foreign-project/state.json is the fold of that feed's lines 1 and 3. Regenerate feed-v1's
only after a deliberate feed change, from client-feed/:
uv run python -c "from pathlib import Path; from client_feed.contract import parse_feed_line as p; from client_feed.fold import fold; d = Path('tests/fixtures/feed-v1'); d.joinpath('state.json').write_text(fold([p(t, line_no=i) for i, t in enumerate(d.joinpath('feed.jsonl').read_text().splitlines(), 1)], project_id='demo-client', upto_seq=26).model_dump_json(indent=2) + '\\n')"
"""

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from client_feed.cli import main
from client_feed.contract import (
    EVENT_MODELS,
    FEED_FILENAME,
    STATE_FILENAME,
    EventHeader,
    FeedLine,
    FeedState,
    Heartbeat,
    WaitingOn,
    parse_feed_line,
)
from client_feed.fold import fold
from client_feed.validate import Problem, ProblemKind, ValidationReport, validate_out_dir

FIXTURES = Path(__file__).parent / 'fixtures'
FEED_V1 = FIXTURES / 'feed-v1'
FOREIGN = FIXTURES / 'feed-v1-foreign-project'


def feed_texts(out_dir: Path) -> list[str]:
    return (out_dir / FEED_FILENAME).read_text(encoding='utf-8').splitlines()


def feed_lines(out_dir: Path) -> list[FeedLine]:
    return [parse_feed_line(text, line_no=i) for i, text in enumerate(feed_texts(out_dir), 1)]


def committed_state(out_dir: Path) -> FeedState:
    return FeedState.model_validate_json((out_dir / STATE_FILENAME).read_text(encoding='utf-8'))


def altered_copy(tmp_path: Path, source: Path = FEED_V1) -> Path:
    return Path(shutil.copytree(source, tmp_path / 'out'))


def rewrite_feed(out_dir: Path, edit: Callable[[list[dict[str, Any]]], None]) -> None:
    objects = [json.loads(text) for text in feed_texts(out_dir)]
    edit(objects)
    (out_dir / FEED_FILENAME).write_text(
        ''.join(json.dumps(obj, separators=(',', ':')) + '\n' for obj in objects), encoding='utf-8'
    )


def only_problem(report: ValidationReport) -> Problem:
    assert not report.ok
    assert len(report.problems) == 1, report.problems
    return report.problems[0]


def test_every_fixture_line_parses() -> None:
    assert len(feed_lines(FEED_V1)) == len(feed_texts(FEED_V1)) == 28


def test_the_fixture_carries_every_known_type_and_one_unknown() -> None:
    lines = feed_lines(FEED_V1)
    assert {ln.type for ln in lines if not isinstance(ln, EventHeader)} == set(EVENT_MODELS)
    assert sum(isinstance(ln, EventHeader) for ln in lines) == 1


def test_the_committed_state_is_the_fold_at_its_watermark() -> None:
    state = committed_state(FEED_V1)
    assert fold(feed_lines(FEED_V1), project_id='demo-client', upto_seq=state.watermark_seq) == state


def test_the_watermark_leaves_lines_for_a_resuming_reader() -> None:
    state = committed_state(FEED_V1)
    lines = feed_lines(FEED_V1)
    assert state.watermark_seq < lines[-1].seq
    assert fold(lines, project_id='demo-client') != state


def test_the_committed_state_holds_the_facts_the_feed_was_written_to_show() -> None:
    state = committed_state(FEED_V1)
    assert set(state.tasks) == {1, 2, 3, 4}
    assert state.tasks[1].status == 'done'
    assert state.tasks[1].landed_commit is not None
    assert state.tasks[3].waiting_on is WaitingOn.LEO
    assert [(t.task_id, t.phase) for t in state.in_flight] == [(2, 'implement')]
    assert [r.path for r in state.revealed] == ['src/ledger.py']
    assert set(state.criteria) == {'verify'}
    assert not any(isinstance(event, Heartbeat) for event in state.recent)
    assert [d.source for d in state.degraded] == ['archive']


def test_the_fixture_validates(capsys: pytest.CaptureFixture[str]) -> None:
    report = validate_out_dir(FEED_V1)
    assert report.ok, report.problems
    assert (report.lines_checked, report.watermark_seq) == (28, 26)
    assert main(['validate-feed', '--out-dir', str(FEED_V1)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['ok'] is True
    assert printed['problems'] == []


def test_a_foreign_project_line_is_the_one_problem(capsys: pytest.CaptureFixture[str]) -> None:
    problem = only_problem(validate_out_dir(FOREIGN))
    assert (problem.kind, problem.seq) == (ProblemKind.PROJECT_MISMATCH, 2)
    assert 'canary-b' in problem.detail
    assert 'demo-client' in problem.detail
    assert main(['validate-feed', '--out-dir', str(FOREIGN)]) == 1
    assert 'canary-b' in capsys.readouterr().out


def test_an_expected_project_other_than_the_state_is_refused() -> None:
    problem = only_problem(validate_out_dir(FEED_V1, expected_project_id='other-client'))
    assert problem.kind == ProblemKind.PROJECT_MISMATCH
    assert 'other-client' in problem.detail
    assert 'demo-client' in problem.detail
    argv = ['validate-feed', '--out-dir', str(FEED_V1), '--project-id', 'other-client']
    assert main(argv) == 1


def test_every_line_with_an_added_key_is_named(tmp_path: Path) -> None:
    out_dir = altered_copy(tmp_path)

    def add_keys(objects: list[dict[str, Any]]) -> None:
        objects[4]['surprise'] = 1
        objects[8]['surprise'] = 2

    rewrite_feed(out_dir, add_keys)
    report = validate_out_dir(out_dir)
    assert [(p.kind, p.line) for p in report.problems] == [
        (ProblemKind.CONTRACT_VIOLATION, 5),
        (ProblemKind.CONTRACT_VIOLATION, 9),
    ]
    assert all('surprise' in p.detail for p in report.problems)


def test_a_line_of_another_major_is_named(tmp_path: Path) -> None:
    out_dir = altered_copy(tmp_path)

    def bump(objects: list[dict[str, Any]]) -> None:
        objects[6]['feed_version'] = '2.0'

    rewrite_feed(out_dir, bump)
    problem = only_problem(validate_out_dir(out_dir))
    assert (problem.kind, problem.line) == (ProblemKind.UNSUPPORTED_VERSION, 7)
    assert '2.0' in problem.detail


def test_swapped_lines_are_a_corrupt_feed(tmp_path: Path) -> None:
    out_dir = altered_copy(tmp_path)

    def swap(objects: list[dict[str, Any]]) -> None:
        objects[10], objects[11] = objects[11], objects[10]

    rewrite_feed(out_dir, swap)
    problem = only_problem(validate_out_dir(out_dir))
    assert (problem.kind, problem.seq) == (ProblemKind.CORRUPT_FEED, 11)
    assert '11' in problem.detail
    assert '12' in problem.detail


def test_an_edited_state_is_a_state_mismatch_naming_the_field(tmp_path: Path) -> None:
    out_dir = altered_copy(tmp_path)
    state = json.loads((out_dir / STATE_FILENAME).read_text(encoding='utf-8'))
    state['tasks']['3']['title'] = 'Edited by hand'
    (out_dir / STATE_FILENAME).write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    problem = only_problem(validate_out_dir(out_dir))
    assert problem.kind == ProblemKind.STATE_MISMATCH
    assert 'tasks' in problem.detail
    assert 'recent' not in problem.detail


@pytest.mark.parametrize('missing', [FEED_FILENAME, STATE_FILENAME])
def test_a_missing_file_is_named(tmp_path: Path, missing: str) -> None:
    out_dir = altered_copy(tmp_path)
    (out_dir / missing).unlink()
    problem = only_problem(validate_out_dir(out_dir))
    assert problem.kind == ProblemKind.MISSING_FILE
    assert missing in problem.detail


@pytest.mark.parametrize('text', ['{', '[]', '{"project_id": "demo-client"}'])
def test_a_malformed_state_is_a_whole_file_contract_violation(tmp_path: Path, text: str) -> None:
    out_dir = altered_copy(tmp_path)
    (out_dir / STATE_FILENAME).write_text(text, encoding='utf-8')
    problem = only_problem(validate_out_dir(out_dir))
    assert (problem.kind, problem.line) == (ProblemKind.CONTRACT_VIOLATION, None)


def test_a_missing_out_dir_argument_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as caught:
        main(['validate-feed'])
    assert caught.value.code == 2


def test_the_installed_console_script_validates_both_fixtures() -> None:
    script = Path(sys.executable).with_name('client-feed')
    assert script.is_file(), f'console script not installed at {script}; run uv sync --all-packages'
    valid = subprocess.run(
        [str(script), 'validate-feed', '--out-dir', str(FEED_V1)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    foreign = subprocess.run(
        [str(script), 'validate-feed', '--out-dir', str(FOREIGN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert foreign.returncode == 1, foreign.stdout + foreign.stderr
    assert 'canary-b' in foreign.stdout
