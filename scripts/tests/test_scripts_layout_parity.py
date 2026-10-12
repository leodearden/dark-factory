"""Each script reads its project's stores where shared.project_layout says they live.

B2's scripts rows (plans/review-all-s14-shared-leaves-prd.md, task 6713). Every
row drives one reader through a public seam for a temp project root and checks
that it lands on the layout's path for that root.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import _task_db_scan
import census_tagger_debris
import merge_lane_throughput
import suite_census_evidence
import sweep_toolcall_markup
import task_event_timeline
from orchestrator.session_registry import normalize_escalations_dir
from sitting import fleet_state, inventory
from suite_census_fixtures import flake_db, git_tree

from shared import project_layout

SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def _load_hyphenated(stem: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(stem.replace('-', '_'), SCRIPTS_DIR / f'{stem}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_sitting_inventorys_orchestrator_queue_is_the_layouts(tmp_path):
    root = tmp_path / 'proj'

    assert inventory.QUEUE_SUBDIRS[0] == Path(project_layout.ESCALATION_QUEUE_RELPATH)
    assert root / inventory.QUEUE_SUBDIRS[0] == project_layout.escalation_queue_dir(root)
    assert fleet_state.project_queue_dirs(root)[0] == normalize_escalations_dir(
        project_layout.escalation_queue_dir(root)
    )


def test_sweep_enumerates_records_in_the_layouts_queue_and_not_a_sibling(tmp_path):
    root = tmp_path / 'proj'
    record = project_layout.escalation_queue_dir(root) / 'esc-1-1.json'
    decoy = root / 'data' / 'escalations-decoy' / 'esc-2-1.json'
    for path in (record, decoy):
        path.parent.mkdir(parents=True)
        path.write_text('{"id": "esc-1-1", "status": "resolved"}')

    targets = sweep_toolcall_markup.discover_targets(root)

    assert [t.path for t in targets if t.lane == sweep_toolcall_markup.LANE_ESCALATIONS] == [
        record
    ]


def test_member_chain_sweep_finds_each_projects_layout_queue(tmp_path, monkeypatch):
    src = tmp_path / 'src'
    queues = [project_layout.escalation_queue_dir(src / name) for name in ('b', 'a')]
    for queue in queues:
        queue.mkdir(parents=True)
    (src / 'c' / 'data').mkdir(parents=True)
    monkeypatch.setenv('HOME', str(tmp_path))

    found = _load_hyphenated('member-chain-sweep').default_queues()

    assert [q for q in found if q.startswith(str(src))] == sorted(str(q) for q in queues)
