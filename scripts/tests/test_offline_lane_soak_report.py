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
from offline_lane_soak_fixtures import T0, UNIT, SoakProject, effect, failure
from orchestrator.flake_ledger import ledger_db_path
from orchestrator.offline_lane_episode_state import CommandRecord
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
