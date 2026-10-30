"""scripts/offline_lane_soak_report.py: ζ's mechanical checks and figures, driven through ``main`` (task 6867).

Every store is written by the lane's real writers (``offline_lane_soak_fixtures``);
the task store is a fake, injected through ``main``'s ``task_reader`` keyword.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import offline_lane_soak_report as soak_report
import pytest
from offline_lane_soak_fixtures import LINT, T0, UNIT, SoakProject, effect, failure
from orchestrator.flake_ledger import UNKNOWN_TEST_ID, FlakeCallSite, FlakeVerdict, ledger_db_path
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Outage,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import CommandRecord, unconfirmable_node
from orchestrator.offline_lane_filing import LANE_METADATA_KEY
from orchestrator.offline_lane_records import LanePass, Phase, ProbeKind, ProbeResult
from shared.project_layout import escalation_queue_dir

NOW = T0 + timedelta(days=7)
OUTAGE_PROBE = ProbeResult(ProbeKind.OUTAGE, 75, 'qdrant', 'qdrant: get collections: ConnectError', 1.5)
CHECKS = (
    'records_well_formed',
    'filings_resolve',
    'filings_bounded_by_threshold',
    'ledger_rows_per_pass',
    'deflake_tasks_carry_evidence',
    'no_stranded_filing',
    'offered_debts_owned_or_due',
)


def _report(
    project: SoakProject, capsys: pytest.CaptureFixture[str], *args: str, now: datetime = NOW,
) -> tuple[int, dict[str, Any], str]:
    code = soak_report.main(
        ['--project-root', str(project.root), '--json', *args], task_reader=project.tasks, now=now,
    )
    out, err = capsys.readouterr()
    return code, json.loads(out), err


def _assert_nothing_to_judge(code: int, report: dict[str, Any], err: str, names: str) -> None:
    assert code == 2
    assert names in report['unjudgeable']
    assert [(c['name'], c['verdict'], c['examined']) for c in report['checks']] == [
        (name, 'no_data', 0) for name in CHECKS
    ]
    for name in CHECKS:
        assert f'[NO_DATA] {name}: examined 0; rule: ' in err


# --- nothing to judge ----------------------------------------------------------------


def test_an_empty_project_root_is_nothing_to_judge(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)

    code, report, err = _report(project, capsys)

    _assert_nothing_to_judge(code, report, err, 'run records')


def _red_pass(project: SoakProject) -> None:
    project.add_pass(UNIT, at=T0, detect=(failure('tests/test_a.py::test_a'),), effect=effect(tests_counted=()))


def test_records_without_a_ledger_are_nothing_to_judge_and_no_ledger_is_made(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    _red_pass(project)

    code, report, err = _report(project, capsys)

    _assert_nothing_to_judge(code, report, err, 'flake ledger')
    assert not ledger_db_path(tmp_path).exists()


def test_a_ledger_that_is_not_sqlite_is_nothing_to_judge(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    _red_pass(project)
    project.db.write_bytes(b'this is not an sqlite database, only some bytes\n' * 64)

    code, report, err = _report(project, capsys)

    _assert_nothing_to_judge(code, report, err, 'flake ledger')
    assert 'unreadable' in report['unjudgeable']


def test_no_escalation_queue_is_nothing_to_judge_and_none_is_made(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    _red_pass(project)
    project.provision_ledger()
    project.save_command(CommandRecord.empty(UNIT))

    code, report, err = _report(project, capsys)

    _assert_nothing_to_judge(code, report, err, 'escalation queue')
    assert not escalation_queue_dir(tmp_path).exists()
    assert ledger_db_path(tmp_path).exists()


def test_no_episode_store_is_nothing_to_judge(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    _red_pass(project)
    project.provision_ledger()
    project.escalations()

    code, report, err = _report(project, capsys)

    _assert_nothing_to_judge(code, report, err, 'episode store')


def test_records_none_of_which_carries_an_effect_block_are_nothing_to_judge(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    project.add_pass(UNIT, at=T0, detect=(failure('tests/test_a.py::test_a'),), blocks=False)
    project.provision(UNIT)

    code, report, err = _report(project, capsys)

    _assert_nothing_to_judge(code, report, err, 'effect block')


def test_no_pass_since_the_cut_is_nothing_to_judge(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    _red_pass(project)
    project.provision(UNIT)

    code, report, err = _report(project, capsys, '--since', (T0 + timedelta(days=1)).isoformat())

    _assert_nothing_to_judge(code, report, err, 'no pass record since')


# --- (a) records_well_formed -----------------------------------------------------------


def _check(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(check for check in report['checks'] if check['name'] == name)


def _figure(report: dict[str, Any], name: str) -> Any:
    return next(figure['value'] for figure in report['figures'] if figure['name'] == name)


def _green_project(tmp_path: Path) -> tuple[SoakProject, list[LanePass]]:
    """Three green unit passes and one green lint pass, every store present: nothing for the other checks."""
    project = SoakProject(tmp_path)
    project.provision(UNIT, LINT)
    passes = [project.add_pass(UNIT, at=T0 + timedelta(hours=hours), effect=effect()) for hours in (0, 1, 2)]
    passes.append(project.add_pass(LINT, at=T0 + timedelta(hours=1, minutes=30), effect=effect()))
    return project, passes


def test_well_formed_records_pass_and_the_run_exits_zero(tmp_path: Path, capsys):
    project, _ = _green_project(tmp_path)

    code, report, err = _report(project, capsys)

    assert code == 0
    assert report['unjudgeable'] is None
    assert (_check(report, 'records_well_formed')['verdict'], _check(report, 'records_well_formed')['examined']) == (
        'pass', 4,
    )
    assert '[PASS] records_well_formed: examined 4; rule: ' in err


def test_a_record_overwritten_with_non_json_fails_naming_the_pass(tmp_path: Path, capsys):
    project, passes = _green_project(tmp_path)
    passes[2].record_path.write_text('{truncated', encoding='utf-8')

    code, report, err = _report(project, capsys)

    assert code == 1
    assert '[FAIL] records_well_formed: examined 4;' in err
    (failure_line,) = _check(report, 'records_well_formed')['failures']
    assert passes[2].pass_id in failure_line
    assert 'no readable run record' in failure_line


def test_an_effect_block_its_reader_rejects_fails_naming_the_pass(tmp_path: Path, capsys):
    project, passes = _green_project(tmp_path)
    project.records.set_pass_blocks(passes[2], effect={'tasks_filed': 3})

    code, report, _ = _report(project, capsys)

    assert code == 1
    (failure_line,) = _check(report, 'records_well_formed')['failures']
    assert passes[2].pass_id in failure_line
    assert 'effect block' in failure_line


def test_a_finished_sub_run_whose_output_is_gone_fails_naming_the_pass_and_phase(tmp_path: Path, capsys):
    project, passes = _green_project(tmp_path)
    passes[1].output_path(Phase.DETECT).unlink()

    code, report, _ = _report(project, capsys)

    assert code == 1
    (failure_line,) = _check(report, 'records_well_formed')['failures']
    assert passes[1].pass_id in failure_line
    assert 'detect' in failure_line


def test_interrupted_passes_are_counted_and_do_not_fail(tmp_path: Path, capsys):
    project, _ = _green_project(tmp_path)
    header_only = project.add_pass(UNIT, at=T0 + timedelta(hours=3), detect=None, blocks=False)
    no_effect = project.add_pass(UNIT, at=T0 + timedelta(hours=4), detect=(failure('tests/test_a.py::test_a'),),
                                 blocks=False)

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'records_well_formed')
    assert (check['verdict'], check['examined'], check['failures']) == ('pass', 6, [])
    assert [note for note in check['notes'] if 'interrupted' in note] == [
        f'interrupted: {UNIT}/{header_only.pass_id}', f'interrupted: {UNIT}/{no_effect.pass_id}',
    ]
    assert _figure(report, 'interrupted_passes') == [
        {'command': UNIT, 'pass_id': header_only.pass_id}, {'command': UNIT, 'pass_id': no_effect.pass_id},
    ]


def test_a_pass_from_before_the_cut_over_is_outside_the_default_window(tmp_path: Path, capsys):
    project, _ = _green_project(tmp_path)
    project.add_pass(UNIT, at=T0 - timedelta(hours=1), detect=(failure('tests/test_a.py::test_a'),), blocks=False)

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'records_well_formed')
    assert (check['examined'], check['notes']) == (4, [])
    assert report['since'] == T0.isoformat()


def test_command_restricts_the_examined_passes(tmp_path: Path, capsys):
    project, _ = _green_project(tmp_path)

    _, report, _ = _report(project, capsys, '--command', LINT)

    assert _check(report, 'records_well_formed')['examined'] == 1
    assert report['commands'] == [LINT]


def test_since_excludes_earlier_passes(tmp_path: Path, capsys):
    project, _ = _green_project(tmp_path)

    _, report, _ = _report(project, capsys, '--since', (T0 + timedelta(hours=1)).isoformat())

    assert _check(report, 'records_well_formed')['examined'] == 3


def test_passes_per_command_per_day_count_by_utc_date(tmp_path: Path, capsys):
    project, _ = _green_project(tmp_path)
    project.add_pass(UNIT, at=T0 + timedelta(hours=20), effect=effect())

    _, report, _ = _report(project, capsys)

    assert _figure(report, 'passes_per_command_per_day') == {
        LINT: {'2026-10-20': 1},
        UNIT: {'2026-10-20': 3, '2026-10-21': 1},
    }


# --- (b) filings_resolve and (c) filings_bounded_by_threshold --------------------------

NODE_A = 'tests/test_a.py::test_a'
NODE_B = 'tests/test_b.py::test_b'


def _reproduced(*nodes: str) -> Confirmed:
    return Confirmed(ConfirmMode.NARROWED, nodes, (), ())


def _red(project: SoakProject, at: datetime, *nodes: str, **effect_fields: Any) -> LanePass:
    """A confirmed red pass of *nodes* with its ledger rows; its effect counts them."""
    lane_pass = project.add_pass(
        UNIT, at=at, detect=[failure(node) for node in nodes], confirm=[failure(node) for node in nodes],
        outcome=_reproduced(*nodes), effect=effect(tests_counted=nodes, **effect_fields),
    )
    project.occurrence(lane_pass, nodes, FlakeVerdict.fails_in_isolation)
    return lane_pass


def _filing_project(tmp_path: Path, *nodes: str) -> tuple[SoakProject, LanePass, tuple[str, ...]]:
    """*nodes* reproduced on two passes, each filed on the second: one fix task per node."""
    project = SoakProject(tmp_path)
    project.provision(UNIT)
    first = _red(project, T0, *nodes, episodes_started=nodes)
    second = project.add_pass(
        UNIT, at=T0 + timedelta(hours=1), detect=[failure(node) for node in nodes],
        confirm=[failure(node) for node in nodes], blocks=False,
    )
    project.occurrence(second, nodes, FlakeVerdict.fails_in_isolation)
    filed = tuple(project.file_fix_task(UNIT, [node], [first, second]) for node in nodes)
    project.set_blocks(second, outcome=_reproduced(*nodes), effect=effect(
        tests_counted=nodes, episodes_extended=nodes, threshold_reached=nodes, tasks_filed=filed,
    ))
    return project, second, filed


def test_every_filed_task_resolves_with_its_digest_and_lane_metadata(tmp_path: Path, capsys):
    project, _, filed = _filing_project(tmp_path, NODE_A, NODE_B)

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'filings_resolve')
    assert (check['verdict'], check['examined']) == ('pass', len(filed))


def test_a_filed_task_the_store_answers_absent_fails_naming_it_and_its_pass(tmp_path: Path, capsys):
    project, filing_pass, (task_id,) = _filing_project(tmp_path, NODE_A)
    project.tasks.answer_absent(task_id)

    code, report, err = _report(project, capsys)

    assert code == 1
    assert '[FAIL] filings_resolve: examined 1;' in err
    (failure_line,) = _check(report, 'filings_resolve')['failures']
    assert task_id in failure_line
    assert filing_pass.pass_id in failure_line


def test_a_filed_task_with_empty_details_fails(tmp_path: Path, capsys):
    project, _, (task_id,) = _filing_project(tmp_path, NODE_A)
    project.tasks.tasks[task_id]['details'] = ''

    code, report, _ = _report(project, capsys)

    assert code == 1
    (failure_line,) = _check(report, 'filings_resolve')['failures']
    assert task_id in failure_line
    assert 'details' in failure_line


def test_a_filed_task_without_lane_metadata_fails(tmp_path: Path, capsys):
    project, _, (task_id,) = _filing_project(tmp_path, NODE_A)
    del project.tasks.tasks[task_id]['metadata'][LANE_METADATA_KEY]

    code, report, _ = _report(project, capsys)

    assert code == 1
    (failure_line,) = _check(report, 'filings_resolve')['failures']
    assert task_id in failure_line
    assert LANE_METADATA_KEY in failure_line


def test_a_task_read_that_fails_is_nothing_to_judge(tmp_path: Path, capsys):
    project, _, (task_id,) = _filing_project(tmp_path, NODE_A)
    project.tasks.fail(task_id)

    code, report, err = _report(project, capsys)

    _assert_nothing_to_judge(code, report, err, 'task store')
    assert task_id in report['unjudgeable']


def test_filings_within_the_threshold_crossings_pass_and_report_both_counts(tmp_path: Path, capsys):
    project, _, _ = _filing_project(tmp_path, NODE_A, NODE_B)

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'filings_bounded_by_threshold')
    assert check['verdict'] == 'pass'
    assert check['notes'] == ['2 distinct task(s) filed; 2 (pass, test) threshold crossing(s)']


def test_more_filings_than_crossings_fail_naming_both_counts(tmp_path: Path, capsys):
    project, filing_pass, (task_id,) = _filing_project(tmp_path, NODE_A)
    extra = project.file_fix_task(UNIT, [NODE_A], [filing_pass])
    project.set_blocks(filing_pass, outcome=_reproduced(NODE_A), effect=effect(
        tests_counted=(NODE_A,), episodes_extended=(NODE_A,), threshold_reached=(NODE_A,),
        tasks_filed=(task_id, extra),
    ))

    code, report, err = _report(project, capsys)

    assert code == 1
    assert '[FAIL] filings_bounded_by_threshold:' in err
    assert _check(report, 'filings_bounded_by_threshold')['failures'] == [
        '2 distinct task(s) filed; 1 (pass, test) threshold crossing(s)',
    ]


def test_a_window_with_no_filing_and_no_crossing_has_no_data_to_bound(tmp_path: Path, capsys):
    project, _ = _green_project(tmp_path)

    code, report, _ = _report(project, capsys)

    assert code == 0
    for name in ('filings_resolve', 'filings_bounded_by_threshold'):
        assert (_check(report, name)['verdict'], _check(report, name)['examined']) == ('no_data', 0)


# --- (d) ledger_rows_per_pass ----------------------------------------------------------


def _reproduced_and_cleared(project: SoakProject, at: datetime, *, cleared_row: FlakeCallSite | None) -> LanePass:
    """A confirmed pass reproducing NODE_A and clearing NODE_B, with NODE_A's row and, at *cleared_row*, NODE_B's."""
    lane_pass = project.add_pass(
        UNIT, at=at, detect=[failure(NODE_A), failure(NODE_B)], confirm=[failure(NODE_A)],
        outcome=Confirmed(ConfirmMode.NARROWED, (NODE_A,), (NODE_B,), ()), effect=effect(tests_counted=(NODE_A,)),
    )
    project.occurrence(lane_pass, [NODE_A], FlakeVerdict.fails_in_isolation)
    if cleared_row is not None:
        project.occurrence(lane_pass, [NODE_B], FlakeVerdict.passes_in_isolation, call_site=cleared_row)
    return lane_pass


def _ledger_project(tmp_path: Path, *, last_cleared_row: FlakeCallSite | None) -> tuple[SoakProject, LanePass]:
    project = SoakProject(tmp_path)
    project.provision(UNIT)
    _reproduced_and_cleared(project, T0, cleared_row=FlakeCallSite.offline_lane)
    last = _reproduced_and_cleared(project, T0 + timedelta(hours=1), cleared_row=last_cleared_row)
    return project, last


def test_every_reproduced_and_cleared_test_with_its_row_passes(tmp_path: Path, capsys):
    project, _ = _ledger_project(tmp_path, last_cleared_row=FlakeCallSite.offline_lane)

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'ledger_rows_per_pass')
    assert (check['verdict'], check['examined']) == ('pass', 4)


def test_a_missing_row_fails_naming_the_pass_and_the_test(tmp_path: Path, capsys):
    project, last = _ledger_project(tmp_path, last_cleared_row=None)

    code, report, err = _report(project, capsys)

    assert code == 1
    assert '[FAIL] ledger_rows_per_pass: examined 4;' in err
    (failure_line,) = _check(report, 'ledger_rows_per_pass')['failures']
    assert last.pass_id in failure_line
    assert NODE_B in failure_line


def test_a_row_at_another_call_site_does_not_count(tmp_path: Path, capsys):
    project, last = _ledger_project(tmp_path, last_cleared_row=FlakeCallSite.merge_gate)

    code, report, _ = _report(project, capsys)

    assert code == 1
    (failure_line,) = _check(report, 'ledger_rows_per_pass')['failures']
    assert last.pass_id in failure_line
    assert NODE_B in failure_line


def test_the_unconfirmable_pseudo_test_and_the_unknown_id_are_never_required(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    project.provision(UNIT)
    lane_pass = project.add_pass(
        UNIT, at=T0, detect=[failure(NODE_A)], confirm=[failure(NODE_A)],
        outcome=_reproduced(NODE_A, UNKNOWN_TEST_ID, unconfirmable_node(UNIT)), effect=effect(),
    )
    project.occurrence(lane_pass, [NODE_A], FlakeVerdict.fails_in_isolation)

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'ledger_rows_per_pass')
    assert (check['verdict'], check['examined']) == ('pass', 1)


def test_unconfirmable_outage_and_green_passes_examine_nothing(tmp_path: Path, capsys):
    project = SoakProject(tmp_path)
    project.provision(UNIT)
    project.add_pass(UNIT, at=T0, effect=effect())
    project.add_pass(
        UNIT, at=T0 + timedelta(hours=1), detect=[failure(NODE_A)],
        outcome=Unconfirmable(UnconfirmableReason.DETECT_RED_NAMED_NO_TEST, 'exit 1 naming no test'), effect=effect(),
    )
    project.add_pass(
        UNIT, at=T0 + timedelta(hours=2), detect=[failure(NODE_A)], probe=OUTAGE_PROBE,
        outcome=Outage(OUTAGE_PROBE), effect=effect(),
    )

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'ledger_rows_per_pass')
    assert (check['verdict'], check['examined']) == ('no_data', 0)
