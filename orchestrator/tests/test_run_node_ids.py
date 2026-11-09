"""``orchestrator.verify.run_node_ids`` against a REAL pytest, through its public seam."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT

from orchestrator import verify as verify_module
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_node_ids

_PROBE = '''\
import pytest


def test_ok():
    pass


def test_bad():
    assert False


@pytest.mark.skip(reason='probe')
def test_skip():
    pass


class TestBox:
    def test_z(self):
        pass


@pytest.mark.parametrize('p', ['a/b'])
def test_p(p):
    pass
'''

OK = 'pkg/tests/test_probe.py::test_ok'
BAD = 'pkg/tests/test_probe.py::test_bad'
SKIP = 'pkg/tests/test_probe.py::test_skip'
PARAM = 'pkg/tests/test_probe.py::test_p[a/b]'
MODULE_RELATIVE_Z = 'tests/test_probe.py::TestBox::test_z'
ABSENT_FILE = 'pkg/tests/test_absent.py::test_x'


def _make_tree(tmp_path: Path) -> Path:
    """Module ``pkg`` with its own ini, so the rootdir is the module dir."""
    (tmp_path / 'pkg' / 'tests').mkdir(parents=True)
    (tmp_path / 'pkg' / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path / 'pkg' / 'tests' / 'test_probe.py').write_text(_PROBE, encoding='utf-8')
    return tmp_path


def _make_config(tree: Path) -> OrchestratorConfig:
    """As test_merge_gate_rerun_end_to_end.py::_make_config: a bare ``pytest`` resolves."""
    return OrchestratorConfig(
        project_root=tree,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        merge_verify_breadth='full',
        verify_admission_enabled=False,
        verify_cold_preprovision_command='',
        verify_env={
            'PATH': f'{Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}',
        },
    )


def _module_config(prefix: str) -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix, test_command='pytest', lint_command=None, type_check_command=None,
    )


@pytest.mark.xdist_group('run_node_ids_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_each_id_gets_its_own_status(tmp_path):
    tree = _make_tree(tmp_path)
    ids = [OK, BAD, SKIP, PARAM, MODULE_RELATIVE_Z, ABSENT_FILE]

    statuses = asyncio.run(run_node_ids(tree, _make_config(tree), [_module_config('pkg')], ids))

    assert dict(statuses) == {
        OK: 'pass',
        BAD: 'fail',
        SKIP: 'skipped',
        PARAM: 'pass',
        MODULE_RELATIVE_Z: 'pass',
        ABSENT_FILE: 'error',
    }
    assert not list((tree / '.df-verify-junit').glob('report*.xml'))


@pytest.mark.xdist_group('run_node_ids_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_an_uncollectable_id_errors_alone(tmp_path):
    tree = _make_tree(tmp_path)
    missing = 'pkg/tests/test_probe.py::test_missing'

    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), [_module_config('pkg')], [OK, BAD, SKIP, missing],
    ))

    assert dict(statuses) == {OK: 'pass', BAD: 'fail', SKIP: 'skipped', missing: 'error'}


@pytest.mark.xdist_group('run_node_ids_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_a_collection_error_errors_only_its_own_id(tmp_path):
    tree = _make_tree(tmp_path)
    (tree / 'pkg' / 'tests' / 'test_broken.py').write_text(
        'import no_such_module_t7918\ndef test_never():\n    pass\n', encoding='utf-8',
    )
    never = 'pkg/tests/test_broken.py::test_never'

    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), [_module_config('pkg')], [OK, never],
    ))

    assert dict(statuses) == {OK: 'pass', never: 'error'}


@pytest.mark.xdist_group('run_node_ids_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_no_ids_returns_an_empty_mapping(tmp_path):
    tree = _make_tree(tmp_path)

    statuses = asyncio.run(run_node_ids(tree, _make_config(tree), [_module_config('pkg')], []))

    assert dict(statuses) == {}


_OK_AND_BAD = 'def test_ok():\n    pass\n\n\ndef test_bad():\n    assert False\n'
A_OK = 'pkg_a/tests/test_a.py::test_ok'
A_BAD = 'pkg_a/tests/test_a.py::test_bad'
B_OK = 'pkg_b/tests/test_b.py::test_ok'
B_BAD = 'pkg_b/tests/test_b.py::test_bad'
TWO_MODULE_STATUSES = {A_OK: 'pass', A_BAD: 'fail', B_OK: 'pass', B_BAD: 'fail'}


def _make_two_module_tree(tmp_path: Path) -> Path:
    """``pkg_a`` has its own ini (rootdir = module dir); ``pkg_b`` has none (rootdir = tree)."""
    for module, test_file in (('pkg_a', 'test_a.py'), ('pkg_b', 'test_b.py')):
        (tmp_path / module / 'tests').mkdir(parents=True)
        (tmp_path / module / 'tests' / test_file).write_text(_OK_AND_BAD, encoding='utf-8')
    (tmp_path / 'pkg_a' / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    return tmp_path


def _run_recording_intervals(
    tree: Path, *, concurrent_groups: bool,
) -> tuple[dict[str, str], list[tuple[Path, float, float]]]:
    """``run_node_ids`` over both modules, recording each group run's report and interval."""
    real_run_verification = verify_module.run_verification
    runs: list[tuple[Path, float, float]] = []

    async def recorder(*args, **kwargs):
        start = time.monotonic()
        try:
            return await real_run_verification(*args, **kwargs)
        finally:
            runs.append((kwargs['junit_report'], start, time.monotonic()))

    with patch.object(verify_module, 'run_verification', side_effect=recorder):
        statuses = asyncio.run(run_node_ids(
            tree, _make_config(tree), [_module_config('pkg_a'), _module_config('pkg_b')],
            [A_OK, A_BAD, B_OK, B_BAD], concurrent_groups=concurrent_groups,
        ))
    return dict(statuses), runs


@pytest.mark.xdist_group('run_node_ids_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_concurrent_groups_overlap_with_distinct_reports(tmp_path):
    tree = _make_two_module_tree(tmp_path)

    statuses, runs = _run_recording_intervals(tree, concurrent_groups=True)

    assert statuses == TWO_MODULE_STATUSES
    reports = {report for report, _, _ in runs}
    assert len(reports) == 2, runs
    merge_verify_reports = {
        verify_module._prepare_junit_report_path(tree, prefix) for prefix in ('pkg_a', 'pkg_b')
    }
    assert not reports & merge_verify_reports, reports
    assert max(start for _, start, _ in runs) < min(end for _, _, end in runs), runs


@pytest.mark.xdist_group('run_node_ids_real_pytest')
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_default_runs_groups_one_at_a_time(tmp_path):
    tree = _make_two_module_tree(tmp_path)

    statuses, runs = _run_recording_intervals(tree, concurrent_groups=False)

    assert statuses == TWO_MODULE_STATUSES
    (_, first_start, first_end), (_, second_start, _) = sorted(runs, key=lambda run: run[1])
    assert first_end <= second_start, runs
