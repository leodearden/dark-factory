"""scripts/offline_lane_soak_report.py: ζ's mechanical checks and figures, driven through ``main`` (task 6867).

Every store is written by the lane's real writers (``offline_lane_soak_fixtures``);
the task store is a fake, injected through ``main``'s ``task_reader`` keyword.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import offline_lane_soak_report as soak_report
import pytest
from offline_lane_soak_fixtures import (
    FAILURE_LINE,
    HEAD,
    LINT,
    OTHER_HEAD,
    T0,
    UNIT,
    SoakProject,
    command_record,
    effect,
    episode,
    failure,
)
from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    DebtTrigger,
    FlakeCallSite,
    FlakeVerdict,
    ledger_db_path,
)
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Outage,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    EndedEpisode,
    EndReason,
    OfferedDebt,
    unconfirmable_node,
)
from orchestrator.offline_lane_escalations import (
    LANE_ESCALATION_ROLE,
    StuckFiling,
    file_stuck_filing,
    resolve_stuck_filing,
)
from orchestrator.offline_lane_filing import LANE_METADATA_KEY
from orchestrator.offline_lane_records import LanePass, Phase, ProbeKind, ProbeResult, records_root
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


# --- (e) deflake_tasks_carry_evidence and (g) offered_debts_owned_or_due ---------------


def _debt_project(
    tmp_path: Path, *, owned: tuple[str, ...] = (NODE_A, NODE_B), due: tuple[str, ...] = (),
    due_at: datetime = NOW - timedelta(minutes=5),
) -> tuple[SoakProject, LanePass, dict[str, str]]:
    """NODE_A and NODE_B red at T0; at T0+1h NODE_A clears and NODE_B recovers, and both are offered as debt.

    Each node in *owned* gets its debt opened by the real ``open_debt``; each in *due* is only enqueued.
    """
    project = SoakProject(tmp_path)
    project.provision(UNIT)
    first = _red(project, T0, NODE_A, NODE_B, episodes_started=(NODE_A, NODE_B))
    offered = ((NODE_A, DebtTrigger.cleared), (NODE_B, DebtTrigger.recovered))
    offering = project.add_pass(
        UNIT, at=T0 + timedelta(hours=1), detect=[failure(NODE_A)], confirm=[],
        outcome=Confirmed(ConfirmMode.NARROWED, (), (NODE_A,), ()),
        effect=effect(debts_offered=tuple(OfferedDebt(node, trigger, HEAD) for node, trigger in offered)),
    )
    project.occurrence(offering, [NODE_A], FlakeVerdict.passes_in_isolation)
    owners: dict[str, str] = {}
    for node, trigger in offered:
        if node in owned:
            row = project.open_lane_debt(node, trigger, offering, earlier=[first])
            assert row.owner_task_id is not None
            owners[node] = row.owner_task_id
        elif node in due:
            project.due(node, trigger, offering, enqueued_at=due_at, earlier=[first])
    return project, offering, owners


def test_lane_owned_debts_carry_trigger_and_evidence_and_are_owned(tmp_path: Path, capsys):
    project, _, _ = _debt_project(tmp_path)

    code, report, _ = _report(project, capsys)

    assert code == 0
    for name in ('deflake_tasks_carry_evidence', 'offered_debts_owned_or_due'):
        assert (_check(report, name)['verdict'], _check(report, name)['examined']) == ('pass', 2)


def test_an_owner_without_its_trigger_fails_naming_the_owner(tmp_path: Path, capsys):
    project, _, owners = _debt_project(tmp_path)
    del project.tasks.tasks[owners[NODE_B]]['metadata']['flake_debt_trigger']

    code, report, err = _report(project, capsys)

    assert code == 1
    assert '[FAIL] deflake_tasks_carry_evidence: examined 2;' in err
    (failure_line,) = _check(report, 'deflake_tasks_carry_evidence')['failures']
    assert owners[NODE_B] in failure_line
    assert 'flake_debt_trigger' in failure_line


def test_an_owner_whose_text_lacks_every_failure_line_of_its_test_fails(tmp_path: Path, capsys):
    project, _, owners = _debt_project(tmp_path)
    owner = project.tasks.tasks[owners[NODE_A]]
    owner['description'] = owner['description'].replace(FAILURE_LINE, 'redacted')
    owner['details'] = owner['details'].replace(FAILURE_LINE, 'redacted')

    code, report, _ = _report(project, capsys)

    assert code == 1
    (failure_line,) = _check(report, 'deflake_tasks_carry_evidence')['failures']
    assert owners[NODE_A] in failure_line
    assert NODE_A in failure_line


def test_an_owner_of_a_debt_opened_at_another_call_site_is_not_examined(tmp_path: Path, capsys):
    project, _, owners = _debt_project(tmp_path)
    project.tasks.tasks[owners[NODE_B]]['metadata']['flake_debt_call_site'] = FlakeCallSite.merge_gate.value

    code, report, _ = _report(project, capsys)

    assert code == 0
    assert _check(report, 'deflake_tasks_carry_evidence')['examined'] == 1


def test_a_debt_still_due_is_owed_and_has_no_owner_to_examine(tmp_path: Path, capsys):
    project, _, _ = _debt_project(tmp_path, owned=(NODE_A,), due=(NODE_B,))

    code, report, _ = _report(project, capsys)

    assert code == 0
    assert _check(report, 'deflake_tasks_carry_evidence')['examined'] == 1
    check = _check(report, 'offered_debts_owned_or_due')
    assert (check['verdict'], check['examined']) == ('pass', 2)


def test_an_offered_debt_with_neither_owner_nor_due_item_fails_naming_the_pass_and_test(tmp_path: Path, capsys):
    project, offering, _ = _debt_project(tmp_path, owned=(NODE_A,))

    code, report, err = _report(project, capsys)

    assert code == 1
    assert '[FAIL] offered_debts_owned_or_due: examined 2;' in err
    (failure_line,) = _check(report, 'offered_debts_owned_or_due')['failures']
    assert offering.pass_id in failure_line
    assert NODE_B in failure_line


def test_a_stale_due_item_is_reported_and_alone_does_not_fail(tmp_path: Path, capsys):
    project, _, _ = _debt_project(tmp_path, owned=(NODE_A,), due=(NODE_B,), due_at=NOW - timedelta(hours=1))

    code, report, _ = _report(project, capsys)

    assert code == 0
    check = _check(report, 'offered_debts_owned_or_due')
    assert check['verdict'] == 'pass'
    assert [note for note in check['notes'] if NODE_B in note and 'stale' in note]


# --- (f) no_stranded_filing ------------------------------------------------------------


def _crossing_project(tmp_path: Path) -> tuple[SoakProject, LanePass, LanePass]:
    """NODE_A reproduced at T0 and again at T0+1h, where it reaches the threshold; nothing filed in any effect."""
    project = SoakProject(tmp_path)
    project.provision(UNIT)
    first = _red(project, T0, NODE_A, episodes_started=(NODE_A,))
    crossing = _red(project, T0 + timedelta(hours=1), NODE_A, episodes_extended=(NODE_A,), threshold_reached=(NODE_A,))
    return project, first, crossing


def _stuck_filing(project: SoakProject) -> str:
    stuck = StuckFiling(command=UNIT, nodes=(NODE_A,), due_since=NOW - timedelta(hours=2), reason='submit_task failed')
    return file_stuck_filing(project.escalations(), stuck)


def _stranded(report: dict[str, Any]) -> dict[str, Any]:
    return _check(report, 'no_stranded_filing')


def test_a_crossing_covered_by_a_task_its_effect_filed_is_owned(tmp_path: Path, capsys):
    project, _, _ = _filing_project(tmp_path, NODE_A)

    code, report, _ = _report(project, capsys)

    assert code == 0
    assert (_stranded(report)['verdict'], _stranded(report)['examined']) == ('pass', 1)


def test_a_crossing_covered_by_a_task_only_its_live_episode_names_is_owned(tmp_path: Path, capsys):
    project, first, crossing = _crossing_project(tmp_path)
    task_id = project.file_fix_task(UNIT, [NODE_A], [first, crossing])
    project.save_command(command_record(UNIT, [episode(NODE_A, [first, crossing], covering_task_id=task_id)]))

    code, report, _ = _report(project, capsys)

    assert code == 0
    assert (_stranded(report)['verdict'], _stranded(report)['examined']) == ('pass', 1)


def test_a_due_filing_with_a_pending_stuck_record_is_owned(tmp_path: Path, capsys):
    project, first, crossing = _crossing_project(tmp_path)
    escalation_id = _stuck_filing(project)
    project.save_command(command_record(
        UNIT, [episode(NODE_A, [first, crossing], filing_due_since=NOW - timedelta(hours=2))],
        stuck_filing_escalation_id=escalation_id,
    ))

    code, report, _ = _report(project, capsys)

    assert code == 0
    assert _stranded(report)['verdict'] == 'pass'


def test_a_filing_due_for_less_than_recheck_is_pending_not_stranded(tmp_path: Path, capsys):
    project, first, crossing = _crossing_project(tmp_path)
    project.save_command(command_record(
        UNIT, [episode(NODE_A, [first, crossing], filing_due_since=NOW - timedelta(minutes=10))],
    ))

    code, report, _ = _report(project, capsys)

    assert code == 0
    assert _stranded(report)['verdict'] == 'pass'
    assert [note for note in _stranded(report)['notes'] if 'pending' in note and NODE_A in note]


def test_a_shorter_recheck_makes_the_same_due_filing_stranded(tmp_path: Path, capsys):
    project, first, crossing = _crossing_project(tmp_path)
    project.save_command(command_record(
        UNIT, [episode(NODE_A, [first, crossing], filing_due_since=NOW - timedelta(minutes=10))],
    ))

    code, report, _ = _report(project, capsys, '--recheck-secs', '300')

    assert code == 1
    assert '300s' in _stranded(report)['rule']


def test_a_crossing_with_neither_a_covering_task_nor_an_open_stuck_record_is_stranded(tmp_path: Path, capsys):
    project, first, crossing = _crossing_project(tmp_path)
    project.save_command(command_record(
        UNIT, [episode(NODE_A, [first, crossing], filing_due_since=NOW - timedelta(hours=2))],
    ))

    code, report, err = _report(project, capsys)

    assert code == 1
    assert '[FAIL] no_stranded_filing: examined 1;' in err
    (failure_line,) = _stranded(report)['failures']
    assert UNIT in failure_line
    assert NODE_A in failure_line
    assert crossing.pass_id in failure_line


def test_a_resolved_stuck_record_does_not_own_the_filing(tmp_path: Path, capsys):
    project, first, crossing = _crossing_project(tmp_path)
    escalation_id = _stuck_filing(project)
    resolve_stuck_filing(project.escalations(), escalation_id)
    project.save_command(command_record(
        UNIT, [episode(NODE_A, [first, crossing], filing_due_since=NOW - timedelta(hours=2))],
        stuck_filing_escalation_id=escalation_id,
    ))

    code, report, _ = _report(project, capsys)

    assert code == 1
    assert _stranded(report)['verdict'] == 'fail'


# --- figures ---------------------------------------------------------------------------

NODE_C = 'tests/test_c.py::test_c'
NODE_D = 'tests/test_d.py::test_d'
NODE_E = 'tests/test_e.py::test_e'
UP_PROBE = ProbeResult(ProbeKind.UP, 0, None, 'every dependency answered', 0.4)
FIGURES = (
    'passes_per_command_per_day',
    'episodes_by_length',
    'filing_delay',
    'lane_debts',
    'due_list',
    'undecided_tests',
    'due_offers_per_pass',
    'clearing_passes_per_day',
    'held_reds_ended_elsewhere',
    'regression_escalations',
    'outages',
    'interrupted_passes',
    'record_directory_bytes',
)


@dataclass(frozen=True)
class _Seeded:
    project: SoakProject
    started: LanePass
    filing: LanePass
    recovering: LanePass
    outage: LanePass
    interrupted: LanePass
    fix_task: str
    owner: str
    outage_escalation: str
    regression: str


def _seed_filing(project: SoakProject, started: LanePass) -> tuple[LanePass, str, str]:
    """T0+1h: NODE_A crosses and is filed, NODE_B clears and is offered as debt, NODE_C is undecided."""
    filing = project.add_pass(
        UNIT, at=T0 + timedelta(hours=1), detect=[failure(NODE_A), failure(NODE_B)], confirm=[failure(NODE_A)],
        blocks=False,
    )
    project.occurrence(filing, [NODE_A], FlakeVerdict.fails_in_isolation)
    project.occurrence(filing, [NODE_B], FlakeVerdict.passes_in_isolation)
    fix_task = project.file_fix_task(UNIT, [NODE_A], [started, filing])
    project.set_blocks(filing, outcome=Confirmed(ConfirmMode.NARROWED, (NODE_A,), (NODE_B,), ()), effect=effect(
        tests_counted=(NODE_A,), episodes_extended=(NODE_A,), threshold_reached=(NODE_A,), tasks_filed=(fix_task,),
        episodes_ended=(EndedEpisode(NODE_B, EndReason.CLEARED, 1, (HEAD,)),),
        debts_offered=(OfferedDebt(NODE_B, DebtTrigger.cleared, HEAD),), debts_undecided=(NODE_C,),
    ))
    owner = project.open_lane_debt(NODE_B, DebtTrigger.cleared, filing, earlier=[started]).owner_task_id
    assert owner is not None
    project.tasks.tasks[owner]['status'] = 'in-progress'
    return filing, fix_task, owner


def _seed_recovery(project: SoakProject, started: LanePass, filing: LanePass) -> LanePass:
    """T0+2h on another head: NODE_A and NODE_C pass there (A offered as recovered), NODE_E passes on its own head."""
    recovering = project.add_pass(UNIT, at=T0 + timedelta(hours=2), head=OTHER_HEAD, effect=effect(
        episodes_ended=(
            EndedEpisode(NODE_A, EndReason.PASSED, 2, (HEAD,)),
            EndedEpisode(NODE_C, EndReason.PASSED, 1, (HEAD,)),
            EndedEpisode(NODE_E, EndReason.PASSED, 1, (OTHER_HEAD,)),
        ),
        debts_offered=(OfferedDebt(NODE_A, DebtTrigger.recovered, HEAD),),
    ))
    project.due(NODE_A, DebtTrigger.recovered, recovering, enqueued_at=NOW - timedelta(hours=1),
                earlier=[started, filing])
    return recovering


def _seed_outage(project: SoakProject) -> tuple[LanePass, str]:
    """T0+3h: a unit outage, its record released; T0+3h30: a lint red whose probe says every dependency is up."""
    outage = project.add_pass(
        UNIT, at=T0 + timedelta(hours=3), detect=[failure(NODE_A)], probe=OUTAGE_PROBE, blocks=False,
    )
    escalation_id = project.file_outage(outage, OUTAGE_PROBE)
    project.set_blocks(outage, outcome=Outage(OUTAGE_PROBE), effect=effect(escalations_raised=(escalation_id,)))
    project.release(escalation_id, 'the outage probe passed')
    project.add_pass(
        LINT, at=T0 + timedelta(hours=3, minutes=30), detect=[failure(NODE_E)], probe=UP_PROBE,
        outcome=Unconfirmable(UnconfirmableReason.DETECT_RED_NAMED_NO_TEST, 'exit 1 naming no test'), effect=effect(),
    )
    return outage, escalation_id


def _seeded(tmp_path: Path) -> _Seeded:
    project = SoakProject(tmp_path)
    project.provision(UNIT, LINT)
    started = _red(project, T0, NODE_A, NODE_B, episodes_started=(NODE_A, NODE_B))
    filing, fix_task, owner = _seed_filing(project, started)
    recovering = _seed_recovery(project, started, filing)
    outage, outage_escalation = _seed_outage(project)
    clearing = project.add_pass(
        UNIT, at=T0 + timedelta(days=1), detect=[failure(NODE_C)], confirm=[],
        outcome=Confirmed(ConfirmMode.NARROWED, (), (NODE_C,), ()), effect=effect(),
    )
    project.occurrence(clearing, [NODE_C], FlakeVerdict.passes_in_isolation)
    interrupted = project.add_pass(UNIT, at=T0 + timedelta(days=1, hours=1), detect=None, blocks=False)
    project.save_command(command_record(UNIT, [episode(NODE_D, [started, filing, clearing])]))
    regression = project.regression_escalation(FlakeCallSite.offline_lane, NODE_B, at=T0 + timedelta(hours=5))
    project.regression_escalation(FlakeCallSite.merge_gate, NODE_B, at=T0 + timedelta(hours=5))
    project.regression_escalation(FlakeCallSite.offline_lane, NODE_C, at=T0 - timedelta(days=1))
    return _Seeded(project, started, filing, recovering, outage, interrupted, fix_task, owner, outage_escalation,
                   regression)


def _seeded_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[_Seeded, dict[str, Any]]:
    seeded = _seeded(tmp_path)
    _, report, _ = _report(seeded.project, capsys)
    return seeded, report


def test_every_figure_is_reported_with_its_rule(tmp_path: Path, capsys):
    _, report = _seeded_report(tmp_path, capsys)

    assert sorted(figure['name'] for figure in report['figures']) == sorted(FIGURES)
    assert all(isinstance(figure['rule'], str) and figure['rule'] for figure in report['figures'])


def test_episodes_by_length_count_ended_episodes_by_reason_and_live_ones(tmp_path: Path, capsys):
    _, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'episodes_by_length') == {
        'ended': {'cleared': {'1': 1}, 'passed': {'1': 2, '2': 1}},
        'live': {'3': 1},
    }


def test_filing_delay_runs_from_the_episode_start_to_the_filing_pass(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'filing_delay') == [{
        'task_id': seeded.fix_task, 'command': UNIT, 'episode_pass': seeded.started.pass_id,
        'filing_pass': seeded.filing.pass_id, 'delay_secs': 3600.0,
    }]


def test_lane_debts_name_the_trigger_evidence_head_owner_and_its_status(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'lane_debts') == [
        {
            'pass_id': seeded.filing.pass_id, 'test': NODE_B, 'trigger': 'cleared', 'evidence_head': HEAD,
            'owner': seeded.owner, 'owner_status': 'in-progress',
        },
        {
            'pass_id': seeded.recovering.pass_id, 'test': NODE_A, 'trigger': 'recovered', 'evidence_head': HEAD,
            'owner': None, 'owner_status': None,
        },
    ]


def test_the_due_list_shows_each_item_and_the_stale_count(tmp_path: Path, capsys):
    _, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'due_list') == {
        'items': [{
            'test': NODE_A, 'call_site': FlakeCallSite.offline_lane.value, 'age_secs': 3600.0, 'attempts': 0,
            'last_error': None,
        }],
        'stale': 1,
    }


def test_undecided_tests_and_due_offers_are_listed_per_pass(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'undecided_tests') == [
        {'command': UNIT, 'pass_id': seeded.filing.pass_id, 'tests': [NODE_C]},
    ]
    assert _figure(report, 'due_offers_per_pass') == {'0': 4, '1': 2}


def test_clearing_passes_count_confirmed_outcomes_that_cleared_by_utc_date(tmp_path: Path, capsys):
    _, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'clearing_passes_per_day') == {UNIT: {'2026-10-20': 1, '2026-10-21': 1}}


def test_held_reds_ended_elsewhere_are_flagged_by_their_recovered_offer(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'held_reds_ended_elsewhere') == [
        {
            'command': UNIT, 'pass_id': seeded.recovering.pass_id, 'test': NODE_A, 'head': OTHER_HEAD,
            'recovered_debt_offered': True,
        },
        {
            'command': UNIT, 'pass_id': seeded.recovering.pass_id, 'test': NODE_C, 'head': OTHER_HEAD,
            'recovered_debt_offered': False,
        },
    ]


def test_regression_escalations_are_the_offline_lane_ones_filed_in_the_window(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    (regression,) = _figure(report, 'regression_escalations')
    assert regression['id'] == seeded.regression
    assert regression['status'] == 'pending'
    assert regression['timestamp'] == (T0 + timedelta(hours=5)).isoformat()


def test_outages_list_their_passes_probe_results_and_released_records(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'outages') == {
        'passes': [{'command': UNIT, 'pass_id': seeded.outage.pass_id}],
        'probe_results': {LINT: {'up': 1}, UNIT: {'outage': 1}},
        'escalations': [{
            'id': seeded.outage_escalation, 'command': UNIT, 'pass_id': seeded.outage.pass_id, 'status': 'resolved',
            'resolved_by': LANE_ESCALATION_ROLE, 'resolution': 'the outage probe passed',
        }],
    }


def test_interrupted_passes_and_the_record_directory_size_are_reported(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    assert _figure(report, 'interrupted_passes') == [{'command': UNIT, 'pass_id': seeded.interrupted.pass_id}]
    stored = records_root(tmp_path).rglob('*')
    assert _figure(report, 'record_directory_bytes') == sum(path.stat().st_size for path in stored if path.is_file())


def test_without_json_stdout_names_each_figure_and_its_rule(tmp_path: Path, capsys):
    seeded, report = _seeded_report(tmp_path, capsys)

    soak_report.main(['--project-root', str(tmp_path)], task_reader=seeded.project.tasks, now=NOW)
    out = capsys.readouterr().out

    for figure in report['figures']:
        assert figure['name'] in out
        assert figure['rule'] in out
