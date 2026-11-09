"""``orchestrator.verify.run_node_ids`` against a REAL pytest, through its public seam."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT

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
def test_no_ids_returns_an_empty_mapping(tmp_path):
    tree = _make_tree(tmp_path)

    statuses = asyncio.run(run_node_ids(tree, _make_config(tree), [_module_config('pkg')], []))

    assert dict(statuses) == {}
