"""validate-feed on the committed boundary fixtures (PRD B4, B5) and on altered copies of them.

The fixtures are what the portal repo vendors. feed-v1/state.json is the fold of feed-v1/feed.jsonl at
seq 28; feed-v1-foreign-project/state.json is the fold of that feed's lines 1 and 3. Regenerate feed-v1's
only after a deliberate feed change, from client-feed/:
uv run python -c "from pathlib import Path; from client_feed.contract import parse_feed_line as p, split_jsonl; from client_feed.fold import fold; d = Path('tests/fixtures/feed-v1'); d.joinpath('state.json').write_text(fold([p(t, line_no=i) for i, t in enumerate(split_jsonl(d.joinpath('feed.jsonl').read_text()), 1)], project_id='demo-client', upto_seq=28).model_dump_json(indent=2) + '\\n')"
"""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from client_feed.cli import main
from client_feed.contract import (
    EVENT_MODELS,
    FEED_FILENAME,
    STATE_FILENAME,
    Degradation,
    EventHeader,
    FeedLine,
    FeedState,
    FiledDefect,
    FileRevealed,
    Heartbeat,
    RunKind,
    Source,
    WaitingOn,
    parse_feed_line,
    split_jsonl,
)
from client_feed.envelope import Stamp
from client_feed.fold import fold
from client_feed.validate import Problem, ProblemKind, ValidationReport, validate_out_dir
from client_feed.value_events import prd_events

FIXTURES = Path(__file__).parent / 'fixtures'
FEED_V1 = FIXTURES / 'feed-v1'
FEED_V1_1 = FIXTURES / 'feed-v1.1'
FOREIGN = FIXTURES / 'feed-v1-foreign-project'
BOUNDARY_FIXTURES = pytest.mark.parametrize('fixture', [FEED_V1, FEED_V1_1], ids=lambda p: p.name)


def feed_texts(out_dir: Path) -> tuple[str, ...]:
    return tuple(split_jsonl((out_dir / FEED_FILENAME).read_text(encoding='utf-8')))


def feed_lines(out_dir: Path) -> list[FeedLine]:
    return [parse_feed_line(text, line_no=i) for i, text in enumerate(feed_texts(out_dir), 1)]


def pass_ends(lines: list[FeedLine]) -> list[Heartbeat]:
    return [ln for ln in lines if isinstance(ln, Heartbeat)]


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
    assert len(feed_lines(FEED_V1)) == len(feed_texts(FEED_V1)) == 31


def test_the_fixture_carries_every_known_type_but_the_one_1_1_added_and_one_unknown() -> None:
    lines = feed_lines(FEED_V1)
    known = {ln.type for ln in lines if not isinstance(ln, EventHeader)}
    assert known == set(EVENT_MODELS) - {'defect_filed'}
    assert sum(isinstance(ln, EventHeader) for ln in lines) == 1


@BOUNDARY_FIXTURES
def test_the_committed_state_is_the_fold_at_its_watermark(fixture: Path) -> None:
    state = committed_state(fixture)
    assert fold(feed_lines(fixture), project_id='demo-client', upto_seq=state.watermark_seq) == state


@BOUNDARY_FIXTURES
def test_every_heartbeat_of_the_fixture_carries_the_watermark_the_previous_pass_committed(
    fixture: Path,
) -> None:
    heartbeats = pass_ends(feed_lines(fixture))
    committed_before_each = [0, *(hb.seq for hb in heartbeats[:-1])]
    assert [hb.watermark_seq for hb in heartbeats] == committed_before_each


@BOUNDARY_FIXTURES
def test_every_pass_of_the_fixture_ends_with_each_prd_counting_its_tasks_as_the_publisher_would(
    fixture: Path,
) -> None:
    lines = feed_lines(fixture)
    assert committed_state(fixture).watermark_seq in {hb.seq for hb in pass_ends(lines)}
    for hb in pass_ends(lines):
        folded = fold(lines, project_id='demo-client', upto_seq=hb.seq)
        corrections = prd_events(
            tasks=folded.tasks.values(),
            state=folded,
            summaries={},
            stamp=Stamp(project_id='demo-client', first_seq=hb.seq + 1),
            now=hb.at,
        )
        assert corrections == (), (
            f'the pass ending at heartbeat seq {hb.seq} leaves stale PRD counts; the publisher would say '
            f'{[(prd.prd_path, prd.task_counts) for prd in corrections]}'
        )


@BOUNDARY_FIXTURES
def test_the_watermark_leaves_lines_for_a_resuming_reader(fixture: Path) -> None:
    state = committed_state(fixture)
    lines = feed_lines(fixture)
    assert state.watermark_seq < lines[-1].seq
    assert fold(lines, project_id='demo-client') != state


def test_every_line_of_the_1_1_fixture_parses_at_1_1() -> None:
    lines = feed_lines(FEED_V1_1)
    assert len(lines) == len(feed_texts(FEED_V1_1))
    assert {ln.feed_version for ln in lines} == {'1.1'}


def test_the_two_boundary_fixtures_together_carry_every_known_type() -> None:
    v1_1_types = {ln.type for ln in feed_lines(FEED_V1_1)}
    v1_types = {ln.type for ln in feed_lines(FEED_V1) if not isinstance(ln, EventHeader)}
    assert 'defect_filed' in v1_1_types
    assert v1_types | v1_1_types == set(EVENT_MODELS)


def test_the_committed_1_1_state_holds_one_gated_filing_and_the_report_still_unfiled() -> None:
    state = committed_state(FEED_V1_1)
    assert state.defects_filed == {
        'portal:demo-client:2:1': FiledDefect(block=2, report=1, task_id=6, gated=True)
    }
    assert state.degraded == (
        Degradation(
            source=Source.FILER,
            reason='1 defect report(s) unfiled past 600s: portal:demo-client:2:2',
        ),
    )
    assert set(state.tasks) == {5, 6}
    assert state.tasks[6].depends_on == (5,)


def test_the_resuming_reader_of_the_1_1_fixture_sees_the_second_filing_and_the_filer_clear() -> None:
    whole = fold(feed_lines(FEED_V1_1), project_id='demo-client')
    assert whole.defects_filed == {
        'portal:demo-client:2:1': FiledDefect(block=2, report=1, task_id=6, gated=True),
        'portal:demo-client:2:2': FiledDefect(block=2, report=2, task_id=8, gated=True),
    }
    assert whole.degraded == ()


def test_the_1_1_fixture_validates() -> None:
    report = validate_out_dir(FEED_V1_1)
    assert report.ok, report.problems
    assert main(['validate-feed', '--out-dir', str(FEED_V1_1)]) == 0


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
    assert {cid: c.kind for cid, c in state.criteria_declared.items()} == {
        'C-1': 'success',
        'C-2': 'failure',
        'C-3': 'success',
    }
    assert set(state.criteria_declared) == {r.id for r in state.criteria[RunKind.VERIFY].results}


def test_the_fixture_validates(capsys: pytest.CaptureFixture[str]) -> None:
    report = validate_out_dir(FEED_V1)
    assert report.ok, report.problems
    assert (report.lines_checked, report.watermark_seq) == (31, 28)
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


def test_a_line_separator_inside_a_string_is_not_a_line_break(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = altered_copy(tmp_path)
    lines = feed_lines(out_dir)
    revealed = next(ln for ln in lines if isinstance(ln, FileRevealed))
    assert revealed.seq == 22
    last = lines[-1]
    line = revealed.model_copy(
        update={
            'seq': last.seq + 1,
            'at': last.at + timedelta(minutes=1),
            'content': 'first\u2028second\x85third\u2029',
        }
    ).model_dump_json()
    assert '\u2028' in line and '\x85' in line
    with (out_dir / FEED_FILENAME).open('a', encoding='utf-8') as feed:
        feed.write(line + '\n')
    report = validate_out_dir(out_dir)
    assert report.ok, report.problems
    assert report.lines_checked == len(feed_lines(FEED_V1)) + 1
    assert main(['validate-feed', '--out-dir', str(out_dir)]) == 0


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


@pytest.mark.parametrize('unreadable', [FEED_FILENAME, STATE_FILENAME])
def test_an_unreadable_file_is_a_reported_problem(
    tmp_path: Path, unreadable: str, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = altered_copy(tmp_path)
    (out_dir / unreadable).unlink()
    (out_dir / unreadable).mkdir()
    problem = only_problem(validate_out_dir(out_dir))
    assert problem.kind == ProblemKind.UNREADABLE_FILE
    assert unreadable in problem.detail
    assert main(['validate-feed', '--out-dir', str(out_dir)]) == 1
    assert json.loads(capsys.readouterr().out)['problems'][0]['kind'] == 'unreadable_file'


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file')
def test_a_file_without_read_permission_is_unreadable(tmp_path: Path) -> None:
    out_dir = altered_copy(tmp_path)
    (out_dir / STATE_FILENAME).chmod(0)
    problem = only_problem(validate_out_dir(out_dir))
    assert problem.kind == ProblemKind.UNREADABLE_FILE
    assert STATE_FILENAME in problem.detail


def test_swapped_lines_past_the_watermark_are_a_corrupt_feed(tmp_path: Path) -> None:
    out_dir = altered_copy(tmp_path)
    second_last = feed_lines(out_dir)[-2].seq

    def swap_the_last_two(objects: list[dict[str, Any]]) -> None:
        objects[-2], objects[-1] = objects[-1], objects[-2]

    rewrite_feed(out_dir, swap_the_last_two)
    problem = only_problem(validate_out_dir(out_dir))
    assert committed_state(out_dir).watermark_seq < second_last
    assert (problem.kind, problem.seq) == (ProblemKind.CORRUPT_FEED, second_last)


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


def test_the_installed_console_script_validates_every_fixture() -> None:
    script = Path(sys.executable).with_name('client-feed')
    assert script.is_file(), f'console script not installed at {script}; run uv sync --all-packages'
    for fixture in (FEED_V1, FEED_V1_1):
        valid = subprocess.run(
            [str(script), 'validate-feed', '--out-dir', str(fixture)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert valid.returncode == 0, fixture.name + valid.stdout + valid.stderr
    foreign = subprocess.run(
        [str(script), 'validate-feed', '--out-dir', str(FOREIGN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert foreign.returncode == 1, foreign.stdout + foreign.stderr
    assert 'canary-b' in foreign.stdout
