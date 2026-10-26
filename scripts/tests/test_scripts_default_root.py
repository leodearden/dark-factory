"""A script's default project root is the MAIN checkout, even run from a linked worktree.

Each row imports one script from a copy placed in a linked worktree of a temp
main repository, evaluates a public no-argument expression, and expects an
answer under that temp MAIN checkout. The scripts derive their defaults from
``shared.project_layout.default_project_root`` at use -- never a home-path
literal, and never the directory the copy happens to sit in. A script whose
parser is public also leaves its flag at None when parsed, so the default is
resolved at use rather than when the parser is built. reclaim_orphaned_worktrees
exposes its default only in its report, so it is probed through its CLI.

check_consolidation_closure's rows live in
fused-memory/tests/test_check_consolidation_closure_cli.py: that script imports
fused_memory, which the venv running this module does not install.
Contract: plans/review-all-s14-shared-leaves-prd.md (decision 5, task 6713).
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
from linked_worktree_copy import linked_worktree_copy, probe_env
from reclaim_orphaned_worktrees import PARKING_ROOT_NAME
from shared.project_layout import escalation_queue_dir, runs_db, tasks_db

REPO_ROOT = Path(__file__).resolve().parents[2]


class Row(NamedTuple):
    module: str
    expression: str
    expected: Callable[[Path], object]


ARCHIVE_ROOT_RELATIVE = Path('data', 'orchestrator', 'agent-transcripts')

ROWS = [
    Row('_task_db_scan', 'm.resolve_project_roots(env={})', lambda main: [str(main)]),
    Row(
        '_task_db_scan',
        'm.tasks_db_path(m.resolve_project_roots(env={})[0])',
        lambda main: str(tasks_db(main)),
    ),
    Row('audit_model_admission', 'm.default_runs_db()', lambda main: str(runs_db(main))),
    Row(
        'escalation_ladder',
        'm.default_escalations_dir()',
        lambda main: str(escalation_queue_dir(main)),
    ),
    Row('merge-pytest-n-ab-analysis', 'm.default_db()', lambda main: str(runs_db(main))),
    Row('merge-pytest-n-ab-analysis', 'm.default_report_dir()', lambda main: str(main / 'plans')),
    Row(
        'audit_offcontract_review_findings',
        'm.default_root()',
        lambda main: str(main / '.worktrees' / '.task-meta'),
    ),
    Row(
        'gc_agent_transcripts',
        'm.default_archive_root()',
        lambda main: str(main / ARCHIVE_ROOT_RELATIVE),
    ),
    Row(
        'migrate_transcript_archive_gunzip',
        'm.default_archive_root()',
        lambda main: str(main / ARCHIVE_ROOT_RELATIVE),
    ),
    Row('merge_lane_throughput', 'm.resolve_project_roots(None)', lambda main: [str(main)]),
]


class ParsedFlag(NamedTuple):
    module: str
    dest: str


UNRESOLVED_AT_PARSE = [
    ParsedFlag('gc_agent_transcripts', 'root'),
    ParsedFlag('migrate_transcript_archive_gunzip', 'root'),
    ParsedFlag('reclaim_orphaned_worktrees', 'repo'),
    ParsedFlag('merge_lane_throughput', 'project_root'),
]


@pytest.fixture(scope='module')
def main_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp('default-root').resolve() / 'main'


@pytest.fixture(scope='module')
def worktree(main_checkout: Path) -> Path:
    return linked_worktree_copy(main_checkout, REPO_ROOT)


def _loader(module: str) -> str:
    if module.isidentifier():
        return f'import {module} as m'
    return (
        'import importlib.util, sys\n'
        f'spec = importlib.util.spec_from_file_location("m", "{module}.py")\n'
        'm = importlib.util.module_from_spec(spec)\n'
        'sys.modules["m"] = m\n'
        'spec.loader.exec_module(m)'
    )


def _probe(tree: Path, module: str, expression: str) -> object:
    source = f'{_loader(module)}\nimport json\nprint(json.dumps({expression}, default=str))'
    proc = subprocess.run(
        [sys.executable, '-s', '-c', source],
        cwd=tree / 'scripts',
        env=probe_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f'probe exited {proc.returncode}\n--- stderr ---\n{proc.stderr}'
    return json.loads(proc.stdout)


@pytest.mark.parametrize('row', ROWS, ids=[f'{row.module}:{row.expression}' for row in ROWS])
def test_default_resolves_under_the_main_checkout(
    main_checkout: Path, worktree: Path, row: Row
) -> None:
    assert _probe(worktree, row.module, row.expression) == row.expected(main_checkout)


@pytest.mark.parametrize(
    'row', UNRESOLVED_AT_PARSE, ids=[f'{row.module}:--{row.dest}' for row in UNRESOLVED_AT_PARSE]
)
def test_parsing_leaves_the_default_root_for_main_to_resolve(row: ParsedFlag) -> None:
    parser = importlib.import_module(row.module).build_parser()
    assert getattr(parser.parse_args([]), row.dest) is None


def test_reclaim_sweeps_the_main_checkouts_quarantine_base(main_checkout: Path, worktree: Path) -> None:
    proc = subprocess.run(
        [sys.executable, '-s', 'reclaim_orphaned_worktrees.py', '--check'],
        cwd=worktree / 'scripts',
        env=probe_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f'reclaim exited {proc.returncode}\n--- stderr ---\n{proc.stderr}'
    assert json.loads(proc.stdout)['root'] == str(main_checkout / PARKING_ROOT_NAME)
