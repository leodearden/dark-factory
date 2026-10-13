"""A script's default project root is the MAIN checkout, even run from a linked worktree.

Each row imports one script from a copy placed in a linked worktree of a temp
main repository, evaluates a public no-argument expression, and expects an
answer under that temp MAIN checkout. The scripts derive their defaults from
``shared.project_layout.default_project_root`` at use -- never a home-path
literal, and never the directory the copy happens to sit in. A script whose
parser is public also leaves its flag at None when parsed, so the default is
resolved at use rather than when the parser is built.

check_consolidation_closure's rows live in
fused-memory/tests/test_check_consolidation_closure_cli.py: that script imports
fused_memory, which the venv running this module does not install.
Contract: plans/review-all-s14-shared-leaves-prd.md (decision 5, task 6713).
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
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
    Row('reclaim_orphaned_worktrees', 'm.default_repo()', lambda main: str(main)),
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


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture(scope='module')
def main_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    main = tmp_path_factory.mktemp('default-root').resolve() / 'main'
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(main)], check=True, capture_output=True)
    _git(main, 'config', 'user.email', 'scripts-default-root@example.com')
    _git(main, 'config', 'user.name', 'Scripts Default Root Test')
    _git(main, 'config', 'commit.gpgsign', 'false')
    _git(main, 'commit', '-q', '--allow-empty', '--no-verify', '-m', 'seed')
    return main


@pytest.fixture(scope='module')
def worktree(main_checkout: Path) -> Path:
    tree = main_checkout / '.worktrees' / 'w'
    _git(main_checkout, 'worktree', 'add', '-q', str(tree), '-b', 'w')
    shutil.copy2(REPO_ROOT / 'pyproject.toml', tree / 'pyproject.toml')
    (tree / 'scripts').mkdir()
    for script in (REPO_ROOT / 'scripts').glob('*.py'):
        shutil.copy2(script, tree / 'scripts' / script.name)
    shutil.copytree(
        REPO_ROOT / 'shared' / 'src' / 'shared',
        tree / 'shared' / 'src' / 'shared',
        ignore=shutil.ignore_patterns('__pycache__'),
    )
    return tree


def _env() -> dict[str, str]:
    dropped = {'PYTHONPATH', 'PYTHONSAFEPATH', 'VIRTUAL_ENV'}
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env['PYTHONNOUSERSITE'] = '1'
    return env


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
        env=_env(),
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
