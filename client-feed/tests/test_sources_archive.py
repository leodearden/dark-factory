"""Reading one verify run back from the orchestrator's durable archive (data/verify-logs/<task id>/)."""

import gzip
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from client_feed.contract import NOT_RECORDED, VerifyLeg
from client_feed.sources.archive import ArchivedRun, read_archived_run

TASK_ID = 7
BASE = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
SECRET_CMD = 'uv run pytest /home/someone/private/tests'
SECRET_HINT = 'AssertionError: secret customer value 0.393'


def at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


def task_dir(root: Path, task_id: int = TASK_ID) -> Path:
    path = root / 'data' / 'verify-logs' / str(task_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stamp(when: datetime) -> str:
    return when.strftime('%Y%m%dT%H%M%S_%fZ')


def qualified(qualifiers: str) -> str:
    return f'.{qualifiers}' if qualifiers else ''


def write_plan(root: Path, when: datetime, *, attempt: int = 1) -> Path:
    path = task_dir(root) / f'attempt-{attempt}.plan-{stamp(when)}.json'
    path.write_text(json.dumps({'legs': [{'label': 'test', 'reason': 'full suite'}]}))
    return path


def summary_payload(category: str, commands: Sequence[tuple[str, int]]) -> dict:
    """The orchestrator's verify.py::_build_summary_payload shape."""
    runs = [
        {
            'label': label,
            'cmd': SECRET_CMD,
            'rc': rc,
            'timed_out': False,
            'started_at': '2026-10-01T10:00:00+00:00',
            'duration_secs': 1.5,
            'segments': None,
            'load': None,
            'slot_wait_secs': None,
        }
        for label, rc in commands
    ]
    worst = max(runs, key=lambda run: run['rc'])
    return {
        'role': 'task',
        'category': category,
        'cause_hint': SECRET_HINT,
        'label': worst['label'],
        'rc': worst['rc'],
        'timed_out': False,
        'cmd': SECRET_CMD,
        'started_at': '2026-10-01T10:00:00+00:00',
        'duration_secs': 4.5,
        'commands': runs,
    }


def write_summary(
    root: Path,
    when: datetime,
    *,
    qualifiers: str = 'alpha',
    category: str = 'test_failure',
    commands: Sequence[tuple[str, int]] = (('test', 1), ('lint', 0), ('type', 0)),
    attempt: int = 1,
    infix: str = 'summary',
) -> Path:
    name = f'attempt-{attempt}{qualified(qualifiers)}.{infix}-{stamp(when)}.json'
    path = task_dir(root) / name
    path.write_text(json.dumps(summary_payload(category, commands)))
    return path


def junit_xml(*testcases: str, suite_time: str | None = '1.5', wrapped: bool = True) -> bytes:
    time_attribute = '' if suite_time is None else f' time="{suite_time}"'
    suite = f'<testsuite name="pytest"{time_attribute}>{"".join(testcases)}</testsuite>'
    document = f'<testsuites name="pytest tests">{suite}</testsuites>' if wrapped else suite
    return f'<?xml version="1.0" encoding="utf-8"?>{document}'.encode()


def passed(name: str, classname: str = 'tests.test_a') -> str:
    return f'<testcase classname="{classname}" name="{name}" time="0.01" />'


def write_junit(
    root: Path, when: datetime, content: bytes, *, qualifiers: str = 'alpha', attempt: int = 1
) -> Path:
    name = f'attempt-{attempt}{qualified(qualifiers)}.junit-{stamp(when)}.xml.gz'
    path = task_dir(root) / name
    path.write_bytes(gzip.compress(content))
    return path


def read(root: Path, ended_at: datetime, *, max_cases: int = 100) -> ArchivedRun:
    return read_archived_run(root, TASK_ID, ended_at=ended_at, max_cases=max_cases)


def case_ids(run: ArchivedRun) -> set[str]:
    assert run.tests is not None
    return {case.id for case in run.tests.cases}


def test_no_task_directory_is_not_recorded(tmp_path: Path) -> None:
    run = read(tmp_path, at(60))

    assert run.legs == NOT_RECORDED
    assert run.tests is None


def test_a_directory_holding_only_a_plan_is_not_recorded(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))

    run = read(tmp_path, at(60))

    assert run.legs == NOT_RECORDED
    assert run.tests is None


def test_a_junit_without_a_summary_has_tests_but_unrecorded_legs(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_junit(tmp_path, at(5), junit_xml(passed('test_x')))

    run = read(tmp_path, at(60))

    assert run.legs == NOT_RECORDED
    assert case_ids(run) == {'tests.test_a::test_x'}


def test_legs_are_the_summary_commands_under_the_run_category(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_summary(tmp_path, at(5), category='test_failure')

    run = read(tmp_path, at(60))

    assert run.legs == (
        VerifyLeg(name='test', category='test_failure', rc=1),
        VerifyLeg(name='lint', category='test_failure', rc=0),
        VerifyLeg(name='type', category='test_failure', rc=0),
    )
    assert run.tests is None
    assert SECRET_CMD not in repr(run)
    assert SECRET_HINT not in repr(run)
    assert '/home/' not in repr(run)


def test_the_run_is_bounded_by_its_plan_and_its_end(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_summary(tmp_path, at(5), category='older_attempt')
    write_junit(tmp_path, at(5), junit_xml(passed('test_old')))
    write_plan(tmp_path, at(60))
    write_summary(tmp_path, at(65), category='this_run')
    write_junit(tmp_path, at(65), junit_xml(passed('test_new')))
    write_plan(tmp_path, at(130))
    write_summary(tmp_path, at(135), category='later_run')
    write_junit(tmp_path, at(135), junit_xml(passed('test_late')))

    run = read(tmp_path, at(120))

    assert run.legs != NOT_RECORDED
    assert {leg.category for leg in run.legs} == {'this_run'}
    assert case_ids(run) == {'tests.test_a::test_new'}


def test_a_file_stamped_exactly_at_the_end_belongs_to_the_run(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_junit(tmp_path, at(5), junit_xml(passed('test_x')))

    assert case_ids(read(tmp_path, at(5))) == {'tests.test_a::test_x'}


def test_without_a_plan_every_file_up_to_the_end_is_a_candidate(tmp_path: Path) -> None:
    write_junit(tmp_path, at(0), junit_xml(passed('test_alpha_first')), qualifiers='alpha')
    write_junit(tmp_path, at(15), junit_xml(passed('test_beta')), qualifiers='beta')
    write_junit(tmp_path, at(30), junit_xml(passed('test_alpha_rerun')), qualifiers='alpha')
    write_junit(tmp_path, at(90), junit_xml(passed('test_after')), qualifiers='beta')

    run = read(tmp_path, at(60))

    assert case_ids(run) == {'tests.test_a::test_beta', 'tests.test_a::test_alpha_rerun'}


def test_a_rerun_within_the_run_replaces_the_earlier_file(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_summary(tmp_path, at(1), category='first')
    write_junit(tmp_path, at(1), junit_xml(passed('test_first')))
    write_summary(tmp_path, at(2), category='second')
    write_junit(tmp_path, at(2), junit_xml(passed('test_second')))

    run = read(tmp_path, at(60))

    assert run.legs != NOT_RECORDED
    assert {leg.category for leg in run.legs} == {'second'}
    assert case_ids(run) == {'tests.test_a::test_second'}


def test_each_module_junit_of_the_run_is_read(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_junit(tmp_path, at(1), junit_xml(passed('test_a')), qualifiers='alpha')
    write_junit(tmp_path, at(2), junit_xml(passed('test_b', 'tests.test_b')), qualifiers='beta')
    write_junit(tmp_path, at(3), junit_xml(passed('test_c', 'tests.test_c')), qualifiers='')

    run = read(tmp_path, at(60))

    assert case_ids(run) == {
        'tests.test_a::test_a',
        'tests.test_b::test_b',
        'tests.test_c::test_c',
    }


def test_files_outside_the_grammar_and_symlinks_are_ignored(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_junit(tmp_path, at(1), junit_xml(passed('test_real')))
    write_summary(tmp_path, at(2), qualifiers='remote-laptop', infix='pass-summary')
    directory = task_dir(tmp_path)
    (directory / f'attempt-1.alpha.test-{stamp(at(2))}.log').write_text(SECRET_HINT)
    (directory / 'notes.txt').write_text('notes')
    (directory / f'attempt-1.alpha.junit-{stamp(at(2))}.xml').write_bytes(
        junit_xml(passed('test_plain_xml'))
    )
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    linked_junit = elsewhere / 'report.xml.gz'
    linked_junit.write_bytes(gzip.compress(junit_xml(passed('test_linked'))))
    (directory / f'attempt-1.gamma.junit-{stamp(at(3))}.xml.gz').symlink_to(linked_junit)
    linked_summary = elsewhere / 'summary.json'
    linked_summary.write_text(json.dumps(summary_payload('linked', [('test', 0)])))
    (directory / f'attempt-1.gamma.summary-{stamp(at(3))}.json').symlink_to(linked_summary)

    run = read(tmp_path, at(60))

    assert run.legs == NOT_RECORDED
    assert case_ids(run) == {'tests.test_a::test_real'}


def test_module_summaries_contribute_their_legs_in_stamp_order(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    write_summary(
        tmp_path, at(3), qualifiers='alpha', category='passed', commands=[('test', 0), ('lint', 0)]
    )
    write_summary(
        tmp_path, at(2), qualifiers='remote-laptop', category='test_failure', commands=[('test', 1)]
    )

    run = read(tmp_path, at(60))

    assert run.legs == (
        VerifyLeg(name='test', category='test_failure', rc=1),
        VerifyLeg(name='test', category='passed', rc=0),
        VerifyLeg(name='lint', category='passed', rc=0),
    )


@pytest.mark.parametrize('task_id', [8, 70])
def test_another_tasks_directory_is_never_read(tmp_path: Path, task_id: int) -> None:
    write_plan(tmp_path, at(0))
    write_junit(tmp_path, at(1), junit_xml(passed('test_x')))

    run = read_archived_run(tmp_path, task_id, ended_at=at(60), max_cases=100)

    assert run.legs == NOT_RECORDED
    assert run.tests is None
