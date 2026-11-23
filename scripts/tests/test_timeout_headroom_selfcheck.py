"""Contract of scripts/timeout_headroom_selfcheck.py."""

from __future__ import annotations

import gzip
import json
import subprocess
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
import timeout_headroom_selfcheck as selfcheck
from escalation.queue import EscalationQueue
from shared.testing_ceiling_anchors import (
    CeilingAnchor,
    Population,
    PopulationKind,
    registered_anchors,
)
from shared.testing_timeout_headroom import UNDER_LOAD_HEADROOM_FACTOR
from shared.testing_timeout_markers import SiteKind, TimeoutSite
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT, min_leg_budget_for_family
from verify_budget_census import Corpus, Record, RecordPath, sanitise_prefix
from verify_log_junits import CaseOutcome, JunitCase, JunitRun, TreeRef, TreeSource

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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


def test_the_slot_is_equal_for_equal_kind_subject_and_bound_whatever_the_required_value() -> None:
    first = _report(_observation(550.75, name='test_one')).finding
    better = _report(_observation(300.0, name='test_two', run=_run('8000'))).finding
    worse = _report(_observation(700.0)).finding
    assert first is not None
    assert better is not None
    assert worse is not None

    assert len({first.required, better.required, worse.required}) == 3
    assert first.slot() == better.slot() == worse.slot()


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


def test_a_finding_whose_slot_is_not_one_file_name_component_is_refused_at_construction() -> None:
    spans_directories = CeilingAnchor(
        anchor_id='a/b',
        ceiling_secs=_ANCHOR.ceiling_secs,
        measured_worst_secs=_ANCHOR.measured_worst_secs,
        population=_SPELLED,
        provenance=_ANCHOR.provenance,
    )

    with pytest.raises(ValueError, match='one file-name component') as refused:
        selfcheck.judge_anchors((spans_directories,), (_observation(550.75),))

    assert 'timeout-headroom:per_test_headroom:a/b:' in str(refused.value)


# ---------------------------------------------------------------------------
# run -- over a tmp project: a git repo holding its own verify archive.
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 11, 22, 1, 30, tzinfo=UTC)
_FAMILY_MODULE = """
    import pytest
    from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

    pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

    def test_scan() -> None:
        pass
    """


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _project(tmp_path: Path, *, scan_seconds: float) -> Path:
    root = tmp_path / 'project'
    (root / 'pkg' / 'tests').mkdir(parents=True)
    (root / 'pkg' / 'tests' / 'test_m.py').write_text(textwrap.dedent(_FAMILY_MODULE), encoding='utf-8')
    (root / 'pkg' / 'orchestrator.yaml').write_text(
        'test_command: "pytest pkg/tests"\nverify_command_timeout_secs: 900\n', encoding='utf-8'
    )
    (root / 'dark-factory-orchestrator.yaml').write_text(f'project_root: {root}\n', encoding='utf-8')
    _git(root, 'init', '-q')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', 'family member')
    tip = _git(root, 'rev-parse', 'HEAD')
    task_dir = root / 'data' / 'verify-logs' / '7011'
    task_dir.mkdir(parents=True)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        f'<testcase classname="tests.test_m" name="test_scan" time="{scan_seconds}" />'
        '</testsuite></testsuites>'
    )
    with gzip.open(task_dir / 'attempt-1.pkg.junit-20261120T145849_000001Z.xml.gz', 'wt', encoding='utf-8') as handle:
        handle.write(xml)
    (task_dir / 'attempt-1.dispatch-20261120T150000Z.json').write_text(
        json.dumps({'base_sha': tip, 'tip_sha': tip}), encoding='utf-8'
    )
    return root


def _leg_summary(root: Path, *, infix: str, seconds: float, cmd: str) -> None:
    task_dir = root / 'data' / 'verify-logs' / '9000'
    task_dir.mkdir(parents=True, exist_ok=True)
    command = {'label': 'test', 'rc': 0, 'timed_out': False, 'duration_secs': seconds, 'cmd': cmd}
    (task_dir / f'attempt-1.{infix}.summary-20261120T145849Z.json').write_text(
        json.dumps({'commands': [command]}), encoding='utf-8'
    )


@dataclass
class _Escalations:
    """A plain recording escalation port: (tool, args) -> canned answer.

    With a ``queue``, ``escalate_info`` mints its id through the real
    ``EscalationQueue.make_id``, so a slot the server cannot mint fails here as it does there.
    """

    history: Mapping[str, list[dict]] = field(default_factory=dict)
    fail_on: str | None = None
    answer_error: bool = False
    queue: EscalationQueue | None = None
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    minted: list[str] = field(default_factory=list)

    def __call__(self, tool: str, args: dict[str, object]) -> Mapping[str, object]:
        self.calls.append((tool, args))
        if tool == self.fail_on:
            raise ConnectionError('escalation server unreachable')
        if self.answer_error:
            return {'error': 'boom'}
        if tool == 'get_task_escalations':
            return {'result': list(self.history.get(str(args['task_id']), []))}
        if tool == 'escalate_info' and self.queue is not None:
            self.minted.append(self.queue.make_id(str(args['task_id'])))
            return {'id': self.minted[-1]}
        return {'id': 'esc-new-1'}

    def tools(self) -> list[str]:
        return [tool for tool, _ in self.calls]


def _minting(tmp_path: Path) -> _Escalations:
    return _Escalations(queue=EscalationQueue(tmp_path / 'escalations'))


_NO_LEGS = selfcheck.LegFacts(budgets={}, family=frozenset())


def _config(root: Path, *, dry_run: bool = False) -> selfcheck.RunConfig:
    return selfcheck.RunConfig(
        project_root=root,
        archive_root=root / 'data' / 'verify-logs',
        stamp_path=root / 'data' / 'timeout-headroom-selfcheck' / 'state.json',
        dry_run=dry_run,
    )


def _selfcheck_run(
    root: Path, escalations: _Escalations, *, dry_run: bool = False, legs: selfcheck.LegFacts = _NO_LEGS
) -> selfcheck.RunResult:
    return selfcheck.run(_config(root, dry_run=dry_run), selfcheck.Ports(escalation=escalations), _NOW, legs=legs)


def test_a_breach_with_no_history_files_exactly_one_escalate_info_in_its_slot(tmp_path: Path) -> None:
    root = _project(tmp_path, scan_seconds=200.0)
    escalations = _minting(tmp_path)

    result = _selfcheck_run(root, escalations)

    assert result.ok
    (finding,) = result.findings
    assert finding.subject == 'whole_tree_family'
    filed = [args for tool, args in escalations.calls if tool == 'escalate_info']
    assert len(filed) == 1
    (args,) = filed
    assert args['task_id'] == finding.slot()
    assert args['agent_role'] == 'timeout-headroom-selfcheck'
    assert args['category'] == 'risk_identified'
    evidence = args['evidence']
    assert isinstance(evidence, list) and evidence
    assert all(set(entry) == {'observation', 'measured_at', 'ref'} for entry in evidence)
    assert result.filed == (finding.slot(),)


def test_a_breaching_leg_whose_prefix_spans_directories_is_filed(tmp_path: Path) -> None:
    root = _project(tmp_path, scan_seconds=20.0)
    _leg_summary(root, infix='tests_scripts', seconds=300.0, cmd='pytest tests/scripts')
    escalations = _minting(tmp_path)
    legs = selfcheck.LegFacts(budgets={'tests/scripts': 100}, family=frozenset({'tests/scripts'}))

    result = _selfcheck_run(root, escalations, legs=legs)

    assert result.ok, result.failure
    (finding,) = result.findings
    slot = finding.slot()
    assert slot in result.filed
    assert '/' not in slot
    assert escalations.minted == [f'esc-{slot}-1']
    assert finding.record()['evidence']['prefix'] == 'tests/scripts'


def test_a_breach_whose_slot_has_history_is_skipped_and_reported(tmp_path: Path) -> None:
    root = _project(tmp_path, scan_seconds=200.0)
    slot = _selfcheck_run(root, _Escalations(), dry_run=True).findings[0].slot()
    escalations = _Escalations(history={slot: [{'id': 'esc-7011-1', 'status': 'resolved'}]})

    result = _selfcheck_run(root, escalations)

    assert result.ok
    assert 'escalate_info' not in escalations.tools()
    assert (result.filed, result.skipped) == ((), (slot,))


@pytest.mark.parametrize('second_night_seconds', [300.0, 160.0], ids=['worse', 'better'])
def test_an_unresized_breach_is_filed_once_across_nights_whose_worst_drifts(
    tmp_path: Path, second_night_seconds: float
) -> None:
    history: dict[str, list[dict]] = {}
    first_night = _selfcheck_run(_project(tmp_path / 'night-1', scan_seconds=200.0), _Escalations(history=history))
    for slot in first_night.filed:
        history[slot] = [{'id': f'esc-{slot}-1', 'status': 'pending'}]
    second = _Escalations(history=history)

    second_night = _selfcheck_run(_project(tmp_path / 'night-2', scan_seconds=second_night_seconds), second)

    assert first_night.findings[0].required != second_night.findings[0].required
    assert len(first_night.filed) == 1
    assert 'escalate_info' not in second.tools()
    assert second_night.skipped == first_night.filed


def test_a_dry_run_calls_no_write_tool_and_writes_no_stamp(tmp_path: Path) -> None:
    root = _project(tmp_path, scan_seconds=200.0)
    escalations = _Escalations()

    result = _selfcheck_run(root, escalations, dry_run=True)

    assert result.ok
    assert len(result.findings) == 1
    assert 'escalate_info' not in escalations.tools()
    assert not _config(root).stamp_path.exists()


def test_a_completed_run_writes_the_stamp(tmp_path: Path) -> None:
    root = _project(tmp_path, scan_seconds=200.0)

    result = _selfcheck_run(root, _minting(tmp_path))

    stamp = json.loads(_config(root).stamp_path.read_text(encoding='utf-8'))
    assert stamp['ok'] is True
    assert stamp['measured_at'] == _NOW.isoformat()
    assert {entry['anchor_id'] for entry in stamp['report']['anchors']} == {
        'whole_tree_family',
        'bare_magicmock_repo_scan',
        'ini_default',
    }
    assert stamp['report']['legs'] == []
    assert stamp['filed'] == list(result.filed)
    assert stamp['skipped'] == []


def test_a_quiet_night_files_nothing_and_still_stamps(tmp_path: Path) -> None:
    root = _project(tmp_path, scan_seconds=20.0)
    escalations = _minting(tmp_path)

    result = _selfcheck_run(root, escalations)

    assert result.ok
    assert result.findings == ()
    assert escalations.calls == []
    assert json.loads(_config(root).stamp_path.read_text(encoding='utf-8'))['ok'] is True


@pytest.mark.parametrize('escalations', [_Escalations(fail_on='get_task_escalations'), _Escalations(answer_error=True)])
def test_an_escalation_fault_fails_the_run_at_its_stage_and_stamps_it(tmp_path: Path, escalations: _Escalations) -> None:
    root = _project(tmp_path, scan_seconds=200.0)

    result = _selfcheck_run(root, escalations)

    assert not result.ok
    assert result.failure is not None
    assert result.failure.stage is selfcheck.FailureStage.ESCALATION
    stamp = json.loads(_config(root).stamp_path.read_text(encoding='utf-8'))
    assert stamp['ok'] is False
    assert stamp['failure']['stage'] == 'escalation'


def test_main_dry_run_prints_the_structured_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _project(tmp_path, scan_seconds=200.0)

    code = selfcheck.main(['--project-root', str(root), '--dry-run', '--json'])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['ok'] is True
    assert printed['dry_run'] is True
    assert [finding['subject'] for finding in printed['findings']] == ['whole_tree_family']
    assert [leg['prefix'] for leg in printed['report']['legs']] == ['pkg']
    assert not _config(root).stamp_path.exists()


def _this_checkout() -> Path:
    return Path(_git(Path(__file__).resolve().parent, 'rev-parse', '--show-toplevel'))


def test_the_legs_adapters_read_this_checkouts_registered_legs() -> None:
    repo_root = _this_checkout()

    budgets = selfcheck.registered_leg_budgets(repo_root)
    family = selfcheck.family_legs(repo_root, budgets)

    assert isinstance(budgets['shared'], int)
    assert isinstance(budgets['orchestrator'], int)
    assert 'shared' in family


def _admitted_site(population: Population) -> TimeoutSite | None:
    if population.kind is PopulationKind.UNMARKED:
        return None
    constant = population.constant
    return TimeoutSite('<module>', SiteKind.MODULE_PYTESTMARK, None, 3, constant or '', constant)


def test_every_registered_leg_and_anchor_breach_has_a_slot_the_escalation_server_can_mint(tmp_path: Path) -> None:
    leg_reports = [
        report
        for prefix in selfcheck.registered_leg_budgets(_this_checkout())
        for report in selfcheck.judge_legs(
            {prefix: 1}, frozenset({prefix}), {sanitise_prefix(prefix): _green(sanitise_prefix(prefix), 655.5)}
        )
    ]
    anchor_reports = [
        report
        for anchor in registered_anchors()
        for report in selfcheck.judge_anchors(
            (anchor,), (_observation(anchor.admissible_worst_secs + 1.0, _admitted_site(anchor.population)),)
        )
    ]
    findings = [report.finding for report in (*leg_reports, *anchor_reports)]
    assert None not in findings
    queue = EscalationQueue(tmp_path / 'q')

    minted = {finding.slot(): queue.make_id(finding.slot()) for finding in findings if finding is not None}

    assert {slot: esc_id for slot, esc_id in minted.items() if not esc_id.startswith(f'esc-{slot}-')} == {}
