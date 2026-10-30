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
from orchestrator.flake_ledger import ledger_db_path
from orchestrator.offline_lane_episode_state import CommandRecord
from orchestrator.offline_lane_records import LanePass, Phase
from shared.project_layout import escalation_queue_dir

NOW = T0 + timedelta(days=7)
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
