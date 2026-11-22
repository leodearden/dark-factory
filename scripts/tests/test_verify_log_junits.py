"""Contract of scripts/verify_log_junits.py, over a tmp archive laid out like data/verify-logs/<task>/."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest
import verify_log_junits as vlj

_TIP = 'a' * 40
_BASE = 'b' * 40
_HEAD = 'c' * 40


@pytest.mark.parametrize(
    ('filename', 'expected'),
    [
        (
            'attempt-2.junit-20261120T145849_791075Z.xml.gz',
            (vlj.ArtefactKind.JUNIT, 2, None, None, datetime(2026, 11, 20, 14, 58, 49, 791075, UTC)),
        ),
        (
            'attempt-1.tests_scripts.junit-20261120T145849_791075Z.xml.gz',
            (vlj.ArtefactKind.JUNIT, 1, 'tests_scripts', None, datetime(2026, 11, 20, 14, 58, 49, 791075, UTC)),
        ),
        (
            'attempt-1.orchestrator.seg1.junit-20261120T145849_000001Z.xml.gz',
            (vlj.ArtefactKind.JUNIT, 1, 'orchestrator', 'seg1', datetime(2026, 11, 20, 14, 58, 49, 1, UTC)),
        ),
        (
            'attempt-3.shared.summary-20260816T080255Z.json',
            (vlj.ArtefactKind.SUMMARY, 3, 'shared', None, datetime(2026, 8, 16, 8, 2, 55, tzinfo=UTC)),
        ),
        (
            'attempt-1.dispatch-20261121T223341_122445Z.json',
            (vlj.ArtefactKind.DISPATCH, 1, None, None, datetime(2026, 11, 21, 22, 33, 41, 122445, UTC)),
        ),
    ],
)
def test_an_archive_name_parses_into_typed_fields(
    filename: str, expected: tuple[object, ...]
) -> None:
    parsed = vlj.ArchiveName.parse(filename)

    assert parsed is not None
    assert (parsed.kind, parsed.attempt, parsed.infix, parsed.segment, parsed.stamp) == expected


@pytest.mark.parametrize(
    'filename',
    [
        'attempt-1.test.log',
        'report.xml',
        'attempt-1.orchestrator.test-20261122T140136_274198Z.log',
        'attempt-1.remote-laptop.pass-summary-20261121T223046_694414Z.json',
        'attempt-1.plan-20261121T223047_268913Z.json',
        'attempt-1.shared.junit-20261121T223337_836122Z.json',
        'attempt-1.shared.summary-20261121T223337_836122Z.xml.gz',
    ],
)
def test_a_name_outside_the_grammar_is_none(filename: str) -> None:
    assert vlj.ArchiveName.parse(filename) is None


def _testcase(classname: str, name: str, seconds: float, child: str = '') -> str:
    return f'<testcase classname={quoteattr(classname)} name={quoteattr(name)} time="{seconds}">{child}</testcase>'


def _write_junit(path: Path, *testcases: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ''.join(testcases)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="{len(testcases)}">{body}</testsuite></testsuites>'
    )
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write(xml)
    return path


def _write_dispatch(path: Path, *, base: str = _BASE, tip: str = _TIP) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'base_sha': base, 'tip_sha': tip, 'decision': 'local'}), encoding='utf-8')
    return path


def _commits(*present: str) -> Callable[[str], bool]:
    return lambda sha: sha in present


_TIP_AND_BASE_EXIST = _commits(_TIP, _BASE)


def _read(
    root: Path, *, min_seconds: float = 1.0, commit_exists: Callable[[str], bool] = _TIP_AND_BASE_EXIST
) -> vlj.JunitCorpus:
    return vlj.read_runs(root, min_seconds=min_seconds, commit_exists=commit_exists, head_sha=_HEAD)


_CRASH = (
    '<error message="failed on setup with &quot;worker \'gw1\' crashed while running '
    "'tests/test_worker_id_fixture.py::test_worker_id_survives_the_lanes_serial_confirm_rerun'&quot;\" />"
)


def test_rows_at_or_above_the_floor_are_kept_and_the_rest_counted(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / '4809' / 'attempt-1.fused-memory.junit-20261115T091851_734589Z.xml.gz',
        _testcase('tests.test_a.TestA', 'test_slow', 12.5),
        _testcase('tests.test_a.TestA', 'test_edge', 1.0),
        _testcase('tests.test_a', 'test_fast', 0.2),
        _testcase('tests.test_b', 'test_quick', 0.01),
    )

    corpus = _read(tmp_path)

    assert corpus.skipped == ()
    (run,) = corpus.runs
    assert (run.task_id, run.attempt, run.infix, run.segment, run.path) == ('4809', 1, 'fused-memory', None, junit)
    assert run.stamp == datetime(2026, 11, 15, 9, 18, 51, 734589, UTC)
    assert [(case.classname, case.name, case.seconds) for case in run.cases] == [
        ('tests.test_a.TestA', 'test_slow', 12.5),
        ('tests.test_a.TestA', 'test_edge', 1.0),
    ]
    assert run.rows_below_floor == 2


def test_an_xdist_worker_crash_row_is_censored_and_other_outcomes_are_kept(tmp_path: Path) -> None:
    _write_junit(
        tmp_path / '4809' / 'attempt-1.fused-memory.junit-20261115T091851_734589Z.xml.gz',
        _testcase('tests.test_worker_id_fixture', 'test_worker_id_survives_the_lanes_serial_confirm_rerun', 0.026, _CRASH),
        _testcase('tests.test_a', 'test_failed', 3.0, '<failure message="assert 1 == 2">trace</failure>'),
        _testcase('tests.test_a', 'test_errored', 3.0, '<error message="fixture blew up">trace</error>'),
        _testcase('tests.test_a', 'test_passed', 3.0),
    )

    (run,) = _read(tmp_path, min_seconds=0.0).runs

    assert {case.name: case.outcome for case in run.cases} == {
        'test_worker_id_survives_the_lanes_serial_confirm_rerun': vlj.CaseOutcome.CENSORED,
        'test_failed': vlj.CaseOutcome.FAILED,
        'test_errored': vlj.CaseOutcome.FAILED,
        'test_passed': vlj.CaseOutcome.PASSED,
    }


def test_a_censored_row_is_kept_whatever_the_floor(tmp_path: Path) -> None:
    """Its junit time is not a duration, so the floor cannot rule it out."""
    _write_junit(
        tmp_path / '4809' / 'attempt-1.fused-memory.junit-20261115T091851_734589Z.xml.gz',
        _testcase('tests.test_worker_id_fixture', 'test_crashed', 0.026, _CRASH),
    )

    (run,) = _read(tmp_path, min_seconds=30.0).runs

    assert [case.outcome for case in run.cases] == [vlj.CaseOutcome.CENSORED]


def _junit_and_dispatches(tmp_path: Path, *dispatch_stamps: str) -> None:
    _write_junit(
        tmp_path / '7000' / 'attempt-2.shared.junit-20261121T223337_836122Z.xml.gz',
        _testcase('tests.test_a', 'test_slow', 5.0),
    )
    for stamp in dispatch_stamps:
        _write_dispatch(tmp_path / '7000' / f'attempt-2.dispatch-{stamp}.json', tip=f'{stamp}-tip', base=f'{stamp}-base')


def test_the_tree_is_the_first_dispatch_at_or_after_the_junit(tmp_path: Path) -> None:
    _junit_and_dispatches(tmp_path, '20261121T223000Z', '20261121T223341_122445Z', '20261121T225945_899745Z')

    (run,) = _read(tmp_path, commit_exists=lambda sha: True).runs

    assert run.tree == vlj.TreeRef('20261121T223341_122445Z-tip', vlj.TreeSource.TIP)


def test_another_attempts_dispatch_is_never_used(tmp_path: Path) -> None:
    _junit_and_dispatches(tmp_path)
    _write_dispatch(tmp_path / '7000' / 'attempt-1.dispatch-20261121T223341_122445Z.json')

    (run,) = _read(tmp_path).runs

    assert run.tree == vlj.TreeRef(_HEAD, vlj.TreeSource.HEAD_FALLBACK)


def test_the_base_is_used_when_only_the_base_commit_exists(tmp_path: Path) -> None:
    _write_junit(
        tmp_path / '7000' / 'attempt-1.shared.junit-20261121T223337Z.xml.gz',
        _testcase('tests.test_a', 'test_slow', 5.0),
    )
    _write_dispatch(tmp_path / '7000' / 'attempt-1.dispatch-20261121T223341_122445Z.json')

    (run,) = _read(tmp_path, commit_exists=_commits(_BASE)).runs

    assert run.tree == vlj.TreeRef(_BASE, vlj.TreeSource.BASE)


def test_head_is_the_fallback_when_neither_dispatched_commit_exists(tmp_path: Path) -> None:
    _write_junit(
        tmp_path / '7000' / 'attempt-1.shared.junit-20261121T223337Z.xml.gz',
        _testcase('tests.test_a', 'test_slow', 5.0),
    )
    _write_dispatch(tmp_path / '7000' / 'attempt-1.dispatch-20261121T223341_122445Z.json')

    (run,) = _read(tmp_path, commit_exists=_commits()).runs

    assert run.tree == vlj.TreeRef(_HEAD, vlj.TreeSource.HEAD_FALLBACK)


def test_head_is_the_fallback_when_no_dispatch_follows_the_junit(tmp_path: Path) -> None:
    _junit_and_dispatches(tmp_path, '20261121T223000Z')

    (run,) = _read(tmp_path, commit_exists=lambda sha: True).runs

    assert run.tree == vlj.TreeRef(_HEAD, vlj.TreeSource.HEAD_FALLBACK)


def test_unreadable_junits_are_counted_skips_and_every_junit_is_accounted_for(tmp_path: Path) -> None:
    task_dir = tmp_path / '8000'
    _write_junit(task_dir / 'attempt-1.shared.junit-20261121T223337Z.xml.gz', _testcase('tests.test_a', 'test_slow', 5.0))
    corrupt_gz = task_dir / 'attempt-1.scripts.junit-20261121T223338Z.xml.gz'
    corrupt_gz.write_bytes(b'\x1f\x8b\x08\x00not really gzip')
    not_xml = task_dir / 'attempt-1.orchestrator.junit-20261121T223339Z.xml.gz'
    with gzip.open(not_xml, 'wt', encoding='utf-8') as handle:
        handle.write('this is a pytest log, not xml')
    (task_dir / 'attempt-1.shared.test-20261121T223337Z.log').write_text('not selected', encoding='utf-8')

    corpus = _read(tmp_path)

    assert [run.infix for run in corpus.runs] == ['shared']
    assert {skip.path: bool(skip.reason) for skip in corpus.skipped} == {corrupt_gz: True, not_xml: True}
    assert len(corpus.runs) + len(corpus.skipped) == 3
