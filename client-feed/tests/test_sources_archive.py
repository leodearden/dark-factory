"""Reading one verify run back from the orchestrator's durable archive (data/verify-logs/<task id>/)."""

import gzip
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from client_feed.contract import (
    FEED_VERSION,
    NOT_RECORDED,
    CaseCounts,
    CaseOutcome,
    CaseResult,
    VerifyLeg,
    VerifyTestResults,
)
from client_feed.sources.archive import (
    ArchivedRun,
    ArchivedTests,
    ArchiveProblem,
    ArchiveReadError,
    read_archived_run,
)

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


def read(
    root: Path, ended_at: datetime, *, started_at: datetime = BASE, max_cases: int = 100
) -> ArchivedRun:
    return read_archived_run(
        root, TASK_ID, started_at=started_at, ended_at=ended_at, max_cases=max_cases
    )


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


def test_files_stamped_exactly_at_the_start_and_the_end_belong_to_the_run(tmp_path: Path) -> None:
    write_junit(tmp_path, at(10), junit_xml(passed('test_first')), qualifiers='alpha')
    write_junit(tmp_path, at(20), junit_xml(passed('test_last')), qualifiers='beta')

    run = read(tmp_path, at(20), started_at=at(10))

    assert case_ids(run) == {'tests.test_a::test_first', 'tests.test_a::test_last'}


def test_without_a_plan_every_file_within_the_run_is_a_candidate(tmp_path: Path) -> None:
    write_junit(tmp_path, at(0), junit_xml(passed('test_before')), qualifiers='beta')
    write_junit(tmp_path, at(10), junit_xml(passed('test_alpha_first')), qualifiers='alpha')
    write_junit(tmp_path, at(15), junit_xml(passed('test_beta')), qualifiers='beta')
    write_junit(tmp_path, at(30), junit_xml(passed('test_alpha_rerun')), qualifiers='alpha')
    write_junit(tmp_path, at(90), junit_xml(passed('test_after')), qualifiers='beta')

    run = read(tmp_path, at(60), started_at=at(5))

    assert case_ids(run) == {'tests.test_a::test_beta', 'tests.test_a::test_alpha_rerun'}


def test_a_later_run_that_archived_no_plan_does_not_read_the_earlier_runs_files(
    tmp_path: Path,
) -> None:
    write_plan(tmp_path, at(0))
    write_summary(tmp_path, at(5), category='local_run')
    write_junit(tmp_path, at(5), junit_xml(passed('test_local')))
    write_summary(tmp_path, at(120), qualifiers='remote-laptop', infix='pass-summary')

    run = read(tmp_path, at(120), started_at=at(110))

    assert run.legs == NOT_RECORDED
    assert run.tests is None


NAIVE = datetime(2026, 10, 1, 10, 0)


@pytest.mark.parametrize(
    ('started_at', 'ended_at', 'named'),
    [(NAIVE, at(60), 'started_at'), (at(0), NAIVE, 'ended_at'), (at(60), at(0), 'after')],
    ids=['naive-start', 'naive-end', 'start-after-end'],
)
def test_an_unusable_run_window_is_refused_by_name(
    tmp_path: Path, started_at: datetime, ended_at: datetime, named: str
) -> None:
    with pytest.raises(ValueError, match=named):
        read(tmp_path, ended_at, started_at=started_at)


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

    run = read_archived_run(tmp_path, task_id, started_at=at(0), ended_at=at(60), max_cases=100)

    assert run.legs == NOT_RECORDED
    assert run.tests is None


SECRET_MESSAGE = 'assert 0.2 > 0.393'
SECRET_TEXT = 'SECRET-TRACEBACK-TEXT /home/someone/private/test_a.py:12'


def red(name: str, tag: str, type_attribute: str | None = None) -> str:
    type_part = '' if type_attribute is None else f' type="{type_attribute}"'
    return (
        f'<testcase classname="tests.test_a" name="{name}" time="0.5">'
        f'<{tag} message="{SECRET_MESSAGE}"{type_part}>{SECRET_TEXT}</{tag}>'
        '</testcase>'
    )


def skipped(name: str, skip_type: str) -> str:
    return (
        f'<testcase classname="tests.test_a" name="{name}" time="0.0">'
        f'<skipped type="{skip_type}" message="{SECRET_MESSAGE}">{SECRET_TEXT}</skipped>'
        '</testcase>'
    )


def archived_tests(tmp_path: Path, *reports: bytes, max_cases: int = 100) -> ArchivedTests:
    write_plan(tmp_path, at(0))
    for index, report in enumerate(reports):
        write_junit(tmp_path, at(1 + index), report, qualifiers=f'module{index}')
    tests = read(tmp_path, at(60), max_cases=max_cases).tests
    assert tests is not None
    return tests


def by_id(tests: ArchivedTests) -> dict[str, CaseResult]:
    return {case.id: case for case in tests.cases}


def test_b7_a_failed_case_reports_its_kind_never_its_message(tmp_path: Path) -> None:
    tests = archived_tests(
        tmp_path,
        junit_xml(passed('test_ok'), red('test_ratio', 'failure'), passed('test_ok_too')),
    )

    ratio = by_id(tests)['tests.test_a::test_ratio']
    assert tests.counts.failed == 1
    assert tests.counts.passed == 2
    assert ratio.outcome is CaseOutcome.FAILED
    assert ratio.exception_type == 'failure'
    assert SECRET_MESSAGE not in repr(tests)
    assert SECRET_TEXT not in repr(tests)
    assert '0.393' not in repr(tests)


def test_a_failure_type_attribute_is_the_exception_type(tmp_path: Path) -> None:
    tests = archived_tests(tmp_path, junit_xml(red('test_ratio', 'failure', 'AssertionError')))

    assert tests.cases[0].exception_type == 'AssertionError'


@pytest.mark.parametrize(
    ('testcase', 'outcome', 'exception_type'),
    [
        (red('test_x', 'error'), CaseOutcome.ERRORED, 'error'),
        (red('test_x', 'error', 'ImportError'), CaseOutcome.ERRORED, 'ImportError'),
        (skipped('test_x', 'pytest.skip'), CaseOutcome.SKIPPED, None),
        (skipped('test_x', 'pytest.xfail'), CaseOutcome.SKIPPED, None),
        (passed('test_x'), CaseOutcome.PASSED, None),
    ],
    ids=['error', 'typed-error', 'skip', 'xfail', 'passed'],
)
def test_outcomes(
    tmp_path: Path, testcase: str, outcome: CaseOutcome, exception_type: str | None
) -> None:
    case = archived_tests(tmp_path, junit_xml(testcase)).cases[0]

    assert (case.outcome, case.exception_type) == (outcome, exception_type)
    assert SECRET_TEXT not in repr(case)


def test_the_first_outcome_child_decides(tmp_path: Path) -> None:
    testcase = (
        '<testcase classname="tests.test_a" name="test_x">'
        '<skipped type="pytest.skip" message="m" /><error message="teardown" />'
        '</testcase>'
    )

    assert archived_tests(tmp_path, junit_xml(testcase)).cases[0].outcome is CaseOutcome.SKIPPED


@pytest.mark.parametrize(
    ('attributes', 'case_id', 'duration_ms'),
    [
        ('classname="tests.test_a.TestK" name="test_y[p]" time="0.0405"', 'tests.test_a.TestK::test_y[p]', 40),
        ('classname="" name="test_bare" time="2"', 'test_bare', 2000),
        ('name="test_no_class"', 'test_no_class', None),
    ],
    ids=['classname', 'empty-classname', 'no-classname-no-time'],
)
def test_case_ids_and_durations(
    tmp_path: Path, attributes: str, case_id: str, duration_ms: int | None
) -> None:
    case = archived_tests(tmp_path, junit_xml(f'<testcase {attributes} />')).cases[0]

    assert (case.id, case.duration_ms) == (case_id, duration_ms)


def test_the_run_duration_sums_its_suites(tmp_path: Path) -> None:
    tests = archived_tests(
        tmp_path,
        junit_xml(passed('test_a'), suite_time='1.5'),
        junit_xml(passed('test_b'), suite_time='2.25', wrapped=False),
    )

    assert tests.duration_ms == 3750
    assert len(tests.cases) == 2


def test_a_run_whose_suites_carry_no_time_has_no_duration(tmp_path: Path) -> None:
    tests = archived_tests(tmp_path, junit_xml(passed('test_a'), suite_time=None))

    assert tests.duration_ms is None


def test_red_cases_lead_and_the_listing_is_cut_to_the_cap(tmp_path: Path) -> None:
    report = junit_xml(
        passed('test_p1'),
        red('test_f1', 'failure'),
        passed('test_p2'),
        red('test_e1', 'error'),
        passed('test_p3'),
    )

    tests = archived_tests(tmp_path, report, max_cases=2)

    assert [case.id for case in tests.cases] == ['tests.test_a::test_f1', 'tests.test_a::test_e1']
    assert tests.counts == CaseCounts(passed=3, failed=1, skipped=0, errored=1, truncated=3)


def test_an_uncut_listing_still_leads_with_red(tmp_path: Path) -> None:
    report = junit_xml(passed('test_p1'), red('test_e1', 'error'), red('test_f1', 'failure'))

    tests = archived_tests(tmp_path, report, max_cases=100)

    assert [case.id for case in tests.cases] == [
        'tests.test_a::test_e1',
        'tests.test_a::test_f1',
        'tests.test_a::test_p1',
    ]
    assert tests.counts.truncated == 0


def test_archived_tests_make_a_valid_test_results_event(tmp_path: Path) -> None:
    report = junit_xml(
        passed('test_p1'), red('test_f1', 'failure'), skipped('test_s1', 'pytest.skip')
    )
    tests = archived_tests(tmp_path, report, max_cases=1)

    event = VerifyTestResults(
        seq=1,
        at=at(60),
        project_id='p',
        feed_version=FEED_VERSION,
        run_key='k',
        counts=tests.counts,
        duration_ms=tests.duration_ms,
        cases=tests.cases,
    )

    assert event.counts.truncated == 2


def _assert_names_the_file(error: ArchiveReadError, tmp_path: Path, problem: ArchiveProblem) -> None:
    assert error.problem is problem
    assert str(error.relpath).startswith(f'data/verify-logs/{TASK_ID}/attempt-1.')
    assert f'data/verify-logs/{TASK_ID}/attempt-1.' in str(error)
    assert str(tmp_path) not in str(error)


def _read_error(tmp_path: Path) -> ArchiveReadError:
    with pytest.raises(ArchiveReadError) as caught:
        read(tmp_path, at(60))
    return caught.value


@pytest.mark.parametrize(
    'content',
    [b'<testsuite/> but not gzip', gzip.compress(junit_xml(passed('test_a')))[:30]],
    ids=['not-gzip', 'truncated-gzip'],
)
def test_an_unreadable_junit_raises_a_typed_error(tmp_path: Path, content: bytes) -> None:
    write_plan(tmp_path, at(0))
    path = write_junit(tmp_path, at(1), b'')
    path.write_bytes(content)

    _assert_names_the_file(_read_error(tmp_path), tmp_path, ArchiveProblem.UNREADABLE)


@pytest.mark.parametrize(
    'content',
    [
        b'<testsuites><testsuite>',
        junit_xml('<testcase name="test_a" time="fast" />'),
        junit_xml('<testcase name="test_a" time="-1" />'),
        junit_xml(passed('test_a'), suite_time='soon'),
    ],
    ids=['unclosed', 'non-numeric-case-time', 'negative-case-time', 'non-numeric-suite-time'],
)
def test_a_malformed_junit_raises_a_typed_error(tmp_path: Path, content: bytes) -> None:
    write_plan(tmp_path, at(0))
    write_junit(tmp_path, at(1), content)

    _assert_names_the_file(_read_error(tmp_path), tmp_path, ArchiveProblem.MALFORMED_JUNIT)


@pytest.mark.parametrize(
    'content',
    ['{"category": "test_failure"}', 'not json', '[]', '{"category": "x", "commands": [{"label": "t", "rc": "1"}]}'],
    ids=['no-commands', 'not-json', 'not-an-object', 'string-rc'],
)
def test_a_malformed_summary_raises_a_typed_error(tmp_path: Path, content: str) -> None:
    write_plan(tmp_path, at(0))
    write_summary(tmp_path, at(1)).write_text(content)

    _assert_names_the_file(_read_error(tmp_path), tmp_path, ArchiveProblem.MALFORMED_SUMMARY)


def test_the_run_names_its_junit_files_relative_to_the_root(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    alpha = write_junit(tmp_path, at(1), junit_xml(passed('test_a')), qualifiers='alpha')
    beta = write_junit(tmp_path, at(2), junit_xml(passed('test_b')), qualifiers='beta')

    tests = read(tmp_path, at(60)).tests

    assert tests is not None
    assert tests.junit_relpaths == (alpha.relative_to(tmp_path), beta.relative_to(tmp_path))


def test_a_collection_map_beside_a_junit_is_never_read_as_a_run_file(tmp_path: Path) -> None:
    write_plan(tmp_path, at(0))
    junit_stem = f'attempt-1.alpha.junit-{stamp(at(1))}'
    for name in (f'{junit_stem}.tests.json', f'{junit_stem}.xml.gz.tests.json'):
        (task_dir(tmp_path) / name).write_text('not json')

    run = read(tmp_path, at(60))

    assert run.legs == NOT_RECORDED
    assert run.tests is None
