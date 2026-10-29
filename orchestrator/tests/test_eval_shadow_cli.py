"""`orchestrator eval-shadow report`: the per-project CLI over the C5 shadow report."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pytest
from _shadow_report_seed import (
    CANDIDATE,
    OPENED_AT,
    PROJECT_ID,
    SeededProject,
    make_project,
    seed_production_cost,
    write_config,
)
from click.testing import CliRunner, Result
from shared.project_layout import runs_db
from test_shadow_report import _COLUMNS, _OTHER, _cells_of, _row, _seed_two_lines, _table

from orchestrator.cli import main


@pytest.fixture
def project(tmp_path) -> SeededProject:
    return make_project(tmp_path)


def _report(*args: str) -> Result:
    return CliRunner().invoke(main, ['eval-shadow', 'report', *args])


def _tags(cell: str) -> list[str]:
    return [] if cell == '-' else cell.split(',')


@pytest.mark.parametrize(('n_min', 'underpowered'), [(4, False), (5, True)])
def test_the_table_prints_the_contracted_columns_and_reads_n_min_from_the_config(
    project, tmp_path, n_min, underpowered,
):
    _seed_two_lines(project)

    result = _report('--config', str(write_config(tmp_path, project.project_root, n_min)))

    assert result.exit_code == 0, result.output
    [header] = [line for line in result.stdout.splitlines() if _cells_of(line)[0] == 'shape']
    assert tuple(_cells_of(header)) == _COLUMNS
    assert ('UNDERPOWERED' in _row(result.stdout, CANDIDATE)['tags']) is underpowered
    assert 'UNDERPOWERED' in _row(result.stdout, _OTHER)['tags']


def test_production_spend_is_read_from_the_selected_projects_store(project, tmp_path):
    paired = _seed_two_lines(project)
    asyncio.run(seed_production_cost(
        project.db_path, PROJECT_ID, paired.task_id, 4.0, OPENED_AT + timedelta(hours=1),
    ))

    result = _report('--config', str(write_config(tmp_path, project.project_root, 3)))

    assert result.exit_code == 0, result.output
    assert _row(result.stdout, CANDIDATE)['$/usable inc'] == '1.00'


def test_the_json_form_parses_and_carries_the_tables_lines(project, tmp_path):
    _seed_two_lines(project)
    config = str(write_config(tmp_path, project.project_root, 3))

    text = _report('--config', config)
    as_json = _report('--config', config, '--json')

    assert as_json.exit_code == 0, as_json.output
    payload = json.loads(as_json.stdout)
    assert [(line['candidate'], line['tags']) for line in payload['lines']] == [
        (row['candidate'], _tags(row['tags'])) for row in _table(text.stdout)
    ]


def test_a_missing_result_fails_both_forms_and_names_the_cell(project, tmp_path):
    gone = _seed_two_lines(project)
    assert gone.result_path is not None
    gone.result_path.unlink()
    config = str(write_config(tmp_path, project.project_root, 3))

    text = _report('--config', config)
    as_json = _report('--config', config, '--json')

    assert text.exit_code != 0
    assert 'result_missing' in text.stdout
    assert gone.cell_id in text.stdout
    assert as_json.exit_code != 0
    payload = json.loads(as_json.stdout)
    assert payload['complete'] is False
    missing = [gap['cell_id'] for line in payload['lines'] for gap in line['result_missing']]
    assert missing == [gone.cell_id]


def test_a_candidate_filter_limits_both_forms_to_that_candidate(project, tmp_path):
    _seed_two_lines(project)
    config = str(write_config(tmp_path, project.project_root, 3))

    text = _report('--config', config, '--candidate', _OTHER)
    as_json = _report('--config', config, '--candidate', _OTHER, '--json')

    assert [row['candidate'] for row in _table(text.stdout)] == [_OTHER]
    assert [line['candidate'] for line in json.loads(as_json.stdout)['lines']] == [_OTHER]


def test_a_project_that_has_no_runs_db_is_refused_and_left_without_one(tmp_path):
    result = _report('--config', str(write_config(tmp_path, tmp_path, 3)))

    assert result.exit_code == 1
    assert str(runs_db(tmp_path)) in result.output
    assert not runs_db(tmp_path).exists()


def test_a_shape_without_a_report_rule_is_refused_naming_the_supported_shapes(project, tmp_path):
    result = _report(
        '--config', str(write_config(tmp_path, project.project_root, 3)), '--shape', 'architect',
    )

    assert result.exit_code != 0
    assert 'implementer' in result.output


def test_the_group_and_its_report_command_are_listed():
    assert 'eval-shadow' in CliRunner().invoke(main, ['--help']).output
    group = CliRunner().invoke(main, ['eval-shadow', '--help'])
    assert group.exit_code == 0, group.output
    assert 'report' in group.output


def test_no_usable_config_is_refused_as_every_sibling_refuses_it(tmp_path, monkeypatch):
    absent = tmp_path / 'absent.yaml'
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(absent))

    result = _report()

    assert result.exit_code == 1
    assert f'Config file not found: {absent}' in result.output
