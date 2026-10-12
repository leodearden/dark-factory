"""Each script reads its project's stores where shared.project_layout says they live.

B2's scripts rows (plans/review-all-s14-shared-leaves-prd.md, task 6713). Every
row drives one reader through a public seam for a temp project root and checks
that it lands on the layout's path for that root.
"""
from __future__ import annotations

import _task_db_scan
import census_tagger_debris
import merge_lane_throughput
import suite_census_evidence
import task_event_timeline
from sitting import fleet_state
from suite_census_fixtures import flake_db, git_tree

from shared import project_layout


def test_fleet_state_reads_runs_db_through_the_one_tier1_locator(tmp_path):
    root = tmp_path / 'proj'

    assert fleet_state.runs_db_path(root) == project_layout.runs_db(root)
    assert fleet_state.runs_db_path is _task_db_scan.runs_db_path


def test_task_event_timeline_opens_the_layouts_runs_db_for_a_project_root(tmp_path, capsys):
    root = tmp_path / 'proj'

    exit_code = task_event_timeline.main(['--project-root', str(root), '1'])

    assert exit_code == task_event_timeline.EXIT_UNREADABLE
    assert str(project_layout.runs_db(root)) in capsys.readouterr().err


def test_census_reads_the_layouts_task_store_and_event_log(tmp_path, make_tasks_db):
    root = tmp_path / 'proj'
    store = project_layout.tasks_db(root)
    store.parent.mkdir(parents=True)
    make_tasks_db([{'id': 1}], directory=store.parent)

    coverage = census_tagger_debris.census_project(str(root)).coverage

    assert coverage.total_tasks == 1
    assert coverage.event_log_path == str(project_layout.runs_db(root))


def test_merge_lane_throughput_resolves_the_layouts_runs_db(tmp_path):
    root = tmp_path / 'proj'

    assert merge_lane_throughput.resolve_runs_db(root) == project_layout.runs_db(root)


def test_suite_census_finds_flakes_at_the_layouts_runs_db(tmp_path):
    root = git_tree(tmp_path / 'proj', {'tests/test_a.py': 'def test_a():\n    pass\n'})
    flake_db(
        project_layout.runs_db(root),
        [('2026-09-01T00:00:00+00:00', 'tests/test_a.py::test_a', 'passes_in_isolation')],
    )

    windows = {w.source: w for w in suite_census_evidence.pytest_evidence(root, root).windows}

    flake = windows['flake_occurrence']
    assert (flake.present, flake.artefacts) == (True, 1)
    assert flake.pattern.startswith('data/orchestrator/runs.db: ')
