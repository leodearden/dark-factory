"""Contract of scripts/timeout_headroom_selfcheck.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import timeout_headroom_selfcheck as selfcheck
from shared.testing_ceiling_anchors import CeilingAnchor, Population, PopulationKind
from shared.testing_timeout_headroom import UNDER_LOAD_HEADROOM_FACTOR
from shared.testing_timeout_markers import SiteKind, TimeoutSite
from shared.whole_tree_guard_family import min_leg_budget_for_family
from verify_budget_census import Corpus, Record, RecordPath
from verify_log_junits import CaseOutcome, JunitCase, JunitRun, TreeRef, TreeSource

_SPELLED = Population(PopulationKind.SPELLED_THROUGH, constant='BIG')
_ANCHOR = CeilingAnchor(
    anchor_id='big_family',
    ceiling_secs=800,
    measured_worst_secs=60.0,
    population=_SPELLED,
    provenance='shared/src/shared/testing_ceiling_anchors.py::BIG_MEASURED_WORST_SECS',
)
_BIG_SITE = TimeoutSite('<module>', SiteKind.MODULE_PYTESTMARK, None, 3, 'BIG', 'BIG')
_OTHER_SITE = TimeoutSite('<module>', SiteKind.MODULE_PYTESTMARK, None, 3, 'OTHER', 'OTHER')


def _run(task_id: str = '7011', infix: str = 'fused-memory') -> JunitRun:
    return JunitRun(
        task_id=task_id,
        attempt=1,
        infix=infix,
        segment=None,
        stamp=datetime(2026, 11, 20, 14, 58, 49, tzinfo=UTC),
        path=Path(f'/archive/{task_id}/attempt-1.{infix}.junit-20261120T145849Z.xml.gz'),
        tree=TreeRef('a' * 40, TreeSource.TIP),
        cases=(),
        rows_below_floor=0,
    )


def _observation(
    seconds: float,
    site: TimeoutSite | None = _BIG_SITE,
    *,
    name: str = 'test_slow',
    outcome: CaseOutcome = CaseOutcome.PASSED,
    run: JunitRun | None = None,
) -> selfcheck.Observation:
    return selfcheck.Observation(
        run=run or _run(),
        case=JunitCase('tests.test_m.TestScan', name, seconds, outcome),
        closest=site,
    )


def _report(*observations: selfcheck.Observation) -> selfcheck.AnchorReport:
    (report,) = selfcheck.judge_anchors((_ANCHOR,), observations)
    return report


# ---------------------------------------------------------------------------
# judge_anchors
# ---------------------------------------------------------------------------


def test_an_anchor_breaches_when_its_worst_member_exceeds_the_admissible_worst() -> None:
    report = _report(_observation(10.0), _observation(150.0, name='test_worst'))

    assert report.finding is not None
    assert report.finding.kind is selfcheck.FindingKind.PER_TEST_HEADROOM
    assert report.worst is not None
    assert report.worst.seconds == 150.0


def test_a_member_exactly_at_the_admissible_worst_does_not_breach() -> None:
    report = _report(_observation(_ANCHOR.admissible_worst_secs))

    assert report.finding is None
    assert report.members == 1


def test_non_members_are_ignored() -> None:
    report = _report(_observation(500.0, _OTHER_SITE), _observation(500.0, None), _observation(20.0))

    assert report.finding is None
    assert report.members == 1
    assert report.worst is not None
    assert report.worst.seconds == 20.0


def test_censored_members_are_excluded_from_the_worst_and_listed() -> None:
    report = _report(
        _observation(0.026, name='test_crashed', outcome=CaseOutcome.CENSORED),
        _observation(30.0),
    )

    assert report.finding is None
    assert report.worst is not None
    assert report.worst.seconds == 30.0
    assert [row.test_id for row in report.censored] == ['tests.test_m.TestScan::test_crashed']


def test_an_anchor_with_no_members_reports_no_worst() -> None:
    report = _report(_observation(500.0, _OTHER_SITE))

    assert (report.members, report.worst, report.finding) == (0, None, None)


def test_a_per_test_finding_carries_the_anchor_and_the_worst_row() -> None:
    worst = _observation(550.75, name='test_worker_id_survives', run=_run('7011'))
    report = _report(_observation(10.0), worst)

    assert report.finding is not None
    evidence = report.finding.record()['evidence']
    assert evidence == {
        'anchor_id': 'big_family',
        'ceiling_secs': 800,
        'headroom_factor': UNDER_LOAD_HEADROOM_FACTOR,
        'recorded_worst_secs': 60.0,
        'recorded_worst_home': _ANCHOR.provenance,
        'observed_worst_secs': 550.75,
        'required_ceiling_secs': 4440,
        'worst_row': {
            'seconds': 550.75,
            'task_id': '7011',
            'attempt': 1,
            'leg': 'fused-memory',
            'test_id': 'tests.test_m.TestScan::test_worker_id_survives',
            'junit_path': '/archive/7011/attempt-1.fused-memory.junit-20261120T145849Z.xml.gz',
            'tree_sha': 'a' * 40,
            'tree_source': 'tip',
        },
    }
    assert (report.finding.subject, report.finding.bound, report.finding.required) == ('big_family', 800, 4440)


# ---------------------------------------------------------------------------
# worst_green_legs
# ---------------------------------------------------------------------------


def _command(label: str, duration: object, *, rc: object = 0, timed_out: object = False, cmd: str = 'pytest') -> dict:
    return {'label': label, 'cmd': cmd, 'rc': rc, 'timed_out': timed_out, 'duration_secs': duration}


def _record(infix: str, *commands: dict, task_id: str = '9000', role: str | None = 'merge') -> Record:
    where = RecordPath(
        path=Path(f'/archive/{task_id}/attempt-1.{infix}.summary-20261120T145849Z.json'),
        task_id=task_id,
        attempt=1,
        module_prefix=infix,
        role=None,
        corpus='archive',
        archived_at='20261120T145849Z',
    )
    payload: dict = {'commands': list(commands)}
    if role is not None:
        payload['role'] = role
    return Record(where=where, payload=payload)


def test_the_worst_green_leg_is_the_longest_passing_untimed_out_test_command() -> None:
    corpus = Corpus(
        records=(
            _record('shared', _command('test', 300.0), _command('type', 900.0), task_id='1'),
            _record('shared', _command('test', 655.5, cmd='pytest tests/test_a.py'), task_id='2', role='task'),
            _record('shared', _command('test', 990.0, rc=1), task_id='3'),
            _record('shared', _command('test', 995.0, timed_out=True), task_id='4'),
            _record('shared', _command('test', True), _command('test', float('nan')), task_id='5'),
            _record('shared', _command('test', 999.0, rc=False), task_id='6'),
            _record('orchestrator', _command('test', 2000.0), task_id='7'),
        ),
        skipped=(),
    )

    worst = selfcheck.worst_green_legs(corpus, ['shared', 'tests_scripts'])

    assert set(worst) == {'shared'}
    assert (worst['shared'].seconds, worst['shared'].task_id, worst['shared'].cmd) == (
        655.5,
        '2',
        'pytest tests/test_a.py',
    )
    assert worst['shared'].role == 'task'


# ---------------------------------------------------------------------------
# judge_legs
# ---------------------------------------------------------------------------


def _green(infix: str, seconds: float) -> selfcheck.GreenRun:
    return selfcheck.GreenRun(
        infix=infix,
        seconds=seconds,
        task_id='9000',
        attempt=1,
        summary_path=f'/archive/9000/attempt-1.{infix}.summary-20261120T145849Z.json',
        cmd='pytest',
        role='merge',
    )


def _legs(budgets: dict[str, int], family: set[str], worst: dict[str, selfcheck.GreenRun]) -> dict[str, selfcheck.LegReport]:
    return {report.prefix: report for report in selfcheck.judge_legs(budgets, frozenset(family), worst)}


def test_a_family_leg_breaches_iff_its_budget_is_below_the_family_floor_of_its_worst() -> None:
    floor = min_leg_budget_for_family(655.5)
    reports = _legs(
        {'shared': floor - 100, 'escalation': floor},
        {'shared', 'escalation'},
        {'shared': _green('shared', 655.5), 'escalation': _green('escalation', 655.5)},
    )

    assert reports['shared'].status is selfcheck.LegStatus.BREACH
    assert reports['shared'].finding is not None
    assert reports['shared'].required_budget_secs == floor
    assert reports['escalation'].status is selfcheck.LegStatus.OK
    assert reports['escalation'].finding is None


def test_a_leg_outside_the_family_never_breaches() -> None:
    reports = _legs({'cockpit': 10}, set(), {'cockpit': _green('cockpit', 5000.0)})

    assert reports['cockpit'].status is selfcheck.LegStatus.NOT_IN_FAMILY
    assert reports['cockpit'].finding is None


def test_a_family_leg_with_no_green_run_is_reported_no_run_never_zero() -> None:
    reports = _legs({'tests/scripts': 100}, {'tests/scripts'}, {})

    report = reports['tests/scripts']
    assert report.status is selfcheck.LegStatus.NO_RUN
    assert (report.worst_run, report.required_budget_secs, report.finding) == (None, None, None)


def test_a_leg_is_matched_to_its_runs_through_the_sanitised_infix() -> None:
    reports = _legs({'tests/scripts': 100}, {'tests/scripts'}, {'tests_scripts': _green('tests_scripts', 300.0)})

    assert reports['tests/scripts'].status is selfcheck.LegStatus.BREACH


def test_a_leg_finding_cites_the_recorded_figure_without_reading_it() -> None:
    reports = _legs({'shared': 100}, {'shared'}, {'shared': _green('shared', 655.5)})

    finding = reports['shared'].finding
    assert finding is not None
    record = finding.record()
    assert record['kind'] == 'leg_budget'
    assert record['subject'] == 'shared'
    assert record['evidence']['recorded_worst_home'] == (
        'tests/scripts/test_module_verify_budgets.py::MEASURED_MODULE_SUITE_WORST_SECS'
    )
    assert record['evidence']['observed_worst_green_secs'] == 655.5
    assert record['evidence']['required_budget_secs'] == min_leg_budget_for_family(655.5)


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


def _any_findings() -> list[selfcheck.Finding]:
    anchor_finding = _report(_observation(550.75)).finding
    leg_finding = _legs({'shared': 100}, {'shared'}, {'shared': _green('shared', 655.5)})['shared'].finding
    assert anchor_finding is not None
    assert leg_finding is not None
    return [anchor_finding, leg_finding]


@pytest.mark.parametrize('index', [0, 1])
def test_every_finding_record_round_trips_through_json(index: int) -> None:
    record = _any_findings()[index].record()

    assert json.loads(json.dumps(record)) == record
    assert record['slot'] == _any_findings()[index].slot()


def test_the_slot_is_equal_for_equal_kind_subject_bound_and_required() -> None:
    first = _report(_observation(550.75, name='test_one')).finding
    same_measurement_other_test = _report(_observation(550.70, name='test_two', run=_run('8000'))).finding
    worse = _report(_observation(700.0)).finding
    assert first is not None
    assert same_measurement_other_test is not None
    assert worse is not None

    assert first.slot() == same_measurement_other_test.slot()
    assert first.slot() != worse.slot()


def test_the_slot_changes_when_the_bound_changes() -> None:
    resized = CeilingAnchor(
        anchor_id=_ANCHOR.anchor_id,
        ceiling_secs=1200,
        measured_worst_secs=_ANCHOR.measured_worst_secs,
        population=_SPELLED,
        provenance=_ANCHOR.provenance,
    )
    before = _report(_observation(550.75)).finding
    (after_report,) = selfcheck.judge_anchors((resized,), (_observation(550.75),))
    assert before is not None
    assert after_report.finding is not None

    assert before.slot() != after_report.finding.slot()
    assert before.slot().startswith('timeout-headroom:per_test_headroom:big_family:')
