"""Each leg's task-role pytest floor narrows to its whole-tree guard family on the REAL tree.

The narrowing is ``orchestrator/src/orchestrator/verify_plan.py::_task_role_production_pytest_run``
(docs/task-role-pytest-floor.md).  It falls back to the owning-module full
suite when a module has no guard family, or when the module's own ``-m``
addopts deselect the whole narrowed selection.  Unit tests pin both fallbacks
on synthetic inputs; only the real module configs over the real tree can show
that neither fires for a given leg.

The plan is derived through the public ``derive_verify_plan`` with the same
two seams ``verify.py`` supplies in production.  The lister mirrors the contract
of ``orchestrator/src/orchestrator/verify.py::_worktree_pkg_test_lister``: a
module prefix maps to every ``.py`` file under ``<root>/<prefix>/tests``, as
root-relative POSIX paths.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pytest
from git_listing import listed_files
from orchestrator.config import ModuleConfig
from orchestrator.verify_evidence import Leg
from orchestrator.verify_plan import (
    PlannedRun,
    ScopeKind,
    derive_verify_plan,
    module_of,
    planned_run_leg,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT, is_whole_tree_guard_module

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group('test_task_role_floor_narrows_every_leg'),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str | None:
    try:
        return (_REPO_ROOT / path).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None


def _list_pkg_tests(prefix: str) -> list[str]:
    tests_dir = _REPO_ROOT / prefix / 'tests'
    if not tests_dir.is_dir():
        return []
    return [path.relative_to(_REPO_ROOT).as_posix() for path in tests_dir.rglob('*.py')]


def _first_production_file(prefix: str) -> str:
    candidates = sorted(
        relative for relative in (
            path.relative_to(_REPO_ROOT).as_posix()
            for path in listed_files(_REPO_ROOT, f'{prefix}/*.py')
        )
        if 'tests' not in PurePosixPath(relative).parts[:-1]
        and PurePosixPath(relative).name != 'conftest.py'
    )
    assert candidates, f'no tracked production .py file under {prefix}/'
    return candidates[0]


def _task_role_pytest_run(mc: ModuleConfig, touched: str) -> PlannedRun:
    plan = derive_verify_plan([touched], [mc], None, _read, role='task', list_pkg_tests=_list_pkg_tests)
    (run,) = [
        run for run in plan.runs
        if module_of(run) == mc.prefix and planned_run_leg(run) is Leg.TEST
    ]
    return run


@pytest.mark.parametrize(('prefix', 'min_family_size'), [
    pytest.param('orchestrator', 10, id='orchestrator'),
    pytest.param('shared', 20, id='shared'),
    pytest.param('fused-memory', 68, id='fused-memory'),
    pytest.param('scripts', 63, id='scripts'),
])
def test_the_task_role_floor_narrows_to_the_guard_family(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
    prefix: str,
    min_family_size: int,
) -> None:
    mc = discover_module_configs()[prefix]
    touched = _first_production_file(prefix)

    run = _task_role_pytest_run(mc, touched)

    assert run.scope_kind is ScopeKind.FILE_SCOPED, run.reason
    assert 'whole-tree guard family' in run.reason, run.reason
    assert '— not narrowed' not in run.reason, run.reason
    outside_the_family = [
        target for target in run.scoped_targets
        if not target.startswith(f'{prefix}/tests/') or not is_whole_tree_guard_module(_read(target))
    ]
    assert not outside_the_family, outside_the_family
    assert len(run.scoped_targets) >= min_family_size, (
        f'{prefix} narrowed to only {len(run.scoped_targets)} guard-family module(s) for '
        f'{touched} (floor {min_family_size}): {list(run.scoped_targets)}'
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        'known gap, task 8503: the scripts leg enumerates its family from scripts/tests '
        'only, though its test_command also collects tests/scripts. Fixing it XPASSes '
        'this test; then drop the xfail.'
    ),
)
def test_the_scripts_floor_runs_the_tests_scripts_guards(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    mc = discover_module_configs()['scripts']

    run = _task_role_pytest_run(mc, _first_production_file('scripts'))

    assert any(target.startswith('tests/scripts/') for target in run.scoped_targets), (
        list(run.scoped_targets)
    )
