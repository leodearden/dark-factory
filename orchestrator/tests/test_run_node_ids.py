"""``orchestrator.verify.run_node_ids`` against a REAL pytest, through its public seam."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_node_ids

pytestmark = [
    pytest.mark.xdist_group('run_node_ids_real_pytest'),
    pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT),
]

_PROBE = '''\
import pytest


@pytest.fixture
def broken_fixture():
    raise RuntimeError('setup fails')


@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError('teardown fails')


def test_ok():
    pass


def test_bad():
    assert False


@pytest.mark.skip(reason='probe')
def test_skip():
    pass


def test_setup_error(broken_fixture):
    pass


def test_fail_then_teardown_error(broken_teardown):
    assert False


def test_pass_then_teardown_error(broken_teardown):
    pass


class TestBox:
    def test_z(self):
        pass


@pytest.mark.parametrize('p', ['a/b'])
def test_p(p):
    pass


@pytest.mark.parametrize('q', [1, 2])
def test_q(q):
    assert q == 1


@pytest.mark.parametrize('baseline', ['present', 'absent'])
def test_ratchet(baseline):
    if baseline == 'absent':
        pytest.skip('baseline absent')
'''

OK = 'pkg/tests/test_probe.py::test_ok'
BAD = 'pkg/tests/test_probe.py::test_bad'
SKIP = 'pkg/tests/test_probe.py::test_skip'
SETUP_ERROR = 'pkg/tests/test_probe.py::test_setup_error'
FAIL_THEN_TEARDOWN_ERROR = 'pkg/tests/test_probe.py::test_fail_then_teardown_error'
PASS_THEN_TEARDOWN_ERROR = 'pkg/tests/test_probe.py::test_pass_then_teardown_error'
BOX_Z = 'pkg/tests/test_probe.py::TestBox::test_z'
PARAM = 'pkg/tests/test_probe.py::test_p[a/b]'
PARAMETRIZED_ALL_PASS = 'pkg/tests/test_probe.py::test_p'
PARAMETRIZED_ONE_FAILS = 'pkg/tests/test_probe.py::test_q'
PARAMETRIZED_ONE_SKIPS = 'pkg/tests/test_probe.py::test_ratchet'
MODULE_RELATIVE_Z = 'tests/test_probe.py::TestBox::test_z'
ABSENT_FILE = 'pkg/tests/test_absent.py::test_x'
MISSING = 'pkg/tests/test_probe.py::test_missing'

def _make_tree(tmp_path: Path) -> Path:
    """Module ``pkg`` with its own ini, so the rootdir is the module dir."""
    (tmp_path / 'pkg' / 'tests').mkdir(parents=True)
    (tmp_path / 'pkg' / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path / 'pkg' / 'tests' / 'test_probe.py').write_text(_PROBE, encoding='utf-8')
    return tmp_path


def _count_sessions(module_dir: Path) -> Path:
    """Have each pytest session rooted at *module_dir* append one line to the returned log."""
    log = module_dir.parent / f'{module_dir.name}-sessions.log'
    (module_dir / 'conftest.py').write_text(
        'def pytest_sessionstart(session):\n'
        f'    with open({str(log)!r}, "a", encoding="utf-8") as log:\n'
        '        log.write("session\\n")\n',
        encoding='utf-8',
    )
    return log


def _sessions(log: Path) -> int:
    return len(log.read_text(encoding='utf-8').splitlines()) if log.exists() else 0


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


def _module_config(prefix: str, test_command: str = 'pytest') -> ModuleConfig:
    return ModuleConfig(
        prefix=prefix, test_command=test_command, lint_command=None, type_check_command=None,
    )


def test_each_id_gets_its_own_status(tmp_path):
    tree = _make_tree(tmp_path)
    ids = [
        OK, BAD, SKIP, SETUP_ERROR, FAIL_THEN_TEARDOWN_ERROR, PASS_THEN_TEARDOWN_ERROR, PARAM,
        PARAMETRIZED_ALL_PASS, PARAMETRIZED_ONE_FAILS, PARAMETRIZED_ONE_SKIPS, MODULE_RELATIVE_Z,
        ABSENT_FILE,
    ]

    statuses = asyncio.run(run_node_ids(tree, _make_config(tree), [_module_config('pkg')], ids))

    assert dict(statuses) == {
        OK: 'pass',
        BAD: 'fail',
        SKIP: 'skipped',
        SETUP_ERROR: 'error',
        FAIL_THEN_TEARDOWN_ERROR: 'fail',
        PASS_THEN_TEARDOWN_ERROR: 'error',
        PARAM: 'pass',
        PARAMETRIZED_ALL_PASS: 'pass',
        PARAMETRIZED_ONE_FAILS: 'fail',
        PARAMETRIZED_ONE_SKIPS: 'skipped',
        MODULE_RELATIVE_Z: 'pass',
        ABSENT_FILE: 'error',
    }
    assert not list((tree / '.df-verify-junit').glob('report*.xml'))


def test_a_run_that_reports_its_ids_is_not_rerun(tmp_path):
    tree = _make_tree(tmp_path)
    sessions = _count_sessions(tree / 'pkg')

    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), [_module_config('pkg')],
        [OK, SETUP_ERROR, PARAMETRIZED_ALL_PASS],
    ))

    assert dict(statuses) == {OK: 'pass', SETUP_ERROR: 'error', PARAMETRIZED_ALL_PASS: 'pass'}
    assert _sessions(sessions) == 1


def test_an_uncollectable_id_errors_alone(tmp_path):
    tree = _make_tree(tmp_path)

    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), [_module_config('pkg')], [OK, BAD, SKIP, MISSING],
    ))

    assert dict(statuses) == {OK: 'pass', BAD: 'fail', SKIP: 'skipped', MISSING: 'error'}


def test_a_refused_run_costs_fewer_sessions_than_ids(tmp_path):
    tree = _make_tree(tmp_path)
    sessions = _count_sessions(tree / 'pkg')
    ids = [OK, BAD, SKIP, BOX_Z, MISSING]

    statuses = asyncio.run(run_node_ids(tree, _make_config(tree), [_module_config('pkg')], ids))

    assert dict(statuses) == {
        OK: 'pass', BAD: 'fail', SKIP: 'skipped', BOX_Z: 'pass', MISSING: 'error',
    }
    assert _sessions(sessions) < len(ids)


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


def test_no_ids_returns_an_empty_mapping(tmp_path):
    tree = _make_tree(tmp_path)

    statuses = asyncio.run(run_node_ids(tree, _make_config(tree), [_module_config('pkg')], []))

    assert dict(statuses) == {}


def test_an_unpreparable_report_path_errors_every_id(tmp_path):
    tree = _make_tree(tmp_path)
    (tree / '.df-verify-junit').write_text('a file where the report directory belongs\n')

    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), [_module_config('pkg')], [OK, BAD],
    ))

    assert dict(statuses) == {OK: 'error', BAD: 'error'}


_OK_AND_BAD = 'def test_ok():\n    pass\n\n\ndef test_bad():\n    assert False\n'
A_OK = 'pkg_a/tests/test_a.py::test_ok'
A_BAD = 'pkg_a/tests/test_a.py::test_bad'
B_OK = 'pkg_b/tests/test_b.py::test_ok'
B_BAD = 'pkg_b/tests/test_b.py::test_bad'
TWO_MODULE_IDS = [A_OK, A_BAD, B_OK, B_BAD]
TWO_MODULE_STATUSES = {A_OK: 'pass', A_BAD: 'fail', B_OK: 'pass', B_BAD: 'fail'}
TWO_MODULES = ('pkg_a', 'pkg_b')


def _ok_and_bad(module: str) -> str:
    return _OK_AND_BAD


def _make_two_module_tree(
    tmp_path: Path, test_file_body: Callable[[str], str] = _ok_and_bad,
) -> Path:
    """``pkg_a`` has its own ini (rootdir = module dir); ``pkg_b`` has none (rootdir = tree)."""
    for module in TWO_MODULES:
        (tmp_path / module / 'tests').mkdir(parents=True)
        test_file = tmp_path / module / 'tests' / f'test_{module[-1]}.py'
        test_file.write_text(test_file_body(module), encoding='utf-8')
    (tmp_path / 'pkg_a' / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    return tmp_path


_RECORDING_BODY = '''\
import json
import pathlib
import time


def _record(request, start):
    entry = {
        'module': MODULE, 'test': request.node.name, 'start': start, 'end': time.time(),
        'report': request.config.option.xmlpath,
    }
    with open(LOG, 'a', encoding='utf-8') as log:
        log.write(json.dumps(entry) + '\\n')


def test_ok(request):
    start = time.time()
    pathlib.Path(OWN_MARK).touch()
    if PARTNER_MARK is not None:
        deadline = time.monotonic() + 60
        while not pathlib.Path(PARTNER_MARK).exists():
            assert time.monotonic() < deadline, 'the partner module never ran alongside'
            time.sleep(0.05)
    _record(request, start)


def test_bad(request):
    _record(request, time.time())
    assert False
'''


def _run_two_recording_modules(
    tmp_path: Path, *, concurrent_groups: bool,
) -> tuple[dict[str, str], list[dict]]:
    """``run_node_ids`` over both modules, whose tests record when they ran and their report.

    Under *concurrent_groups* each module's ``test_ok`` waits for the other's
    to start, so it passes only when the two module runs overlap.
    """
    log = tmp_path / 'runs.jsonl'
    marks = tmp_path / 'marks'
    marks.mkdir()

    def recording_test_file(module: str) -> str:
        [partner] = [other for other in TWO_MODULES if other != module]
        partner_mark = str(marks / partner) if concurrent_groups else None
        return (
            f'LOG = {str(log)!r}\nMODULE = {module!r}\n'
            f'OWN_MARK = {str(marks / module)!r}\nPARTNER_MARK = {partner_mark!r}\n'
            + _RECORDING_BODY
        )

    tree = _make_two_module_tree(tmp_path, recording_test_file)
    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), [_module_config(module) for module in TWO_MODULES],
        TWO_MODULE_IDS, concurrent_groups=concurrent_groups,
    ))
    records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    return dict(statuses), records


def test_concurrent_groups_overlap_with_distinct_reports(tmp_path):
    statuses, records = _run_two_recording_modules(tmp_path, concurrent_groups=True)

    assert statuses == TWO_MODULE_STATUSES, records
    ok_runs = [record for record in records if record['test'] == 'test_ok']
    assert max(run['start'] for run in ok_runs) < min(run['end'] for run in ok_runs), ok_runs
    reports = {record['report'] for record in records}
    assert len(reports) == 2, records
    merge_verify_reports = {
        str(tmp_path.resolve() / '.df-verify-junit' / f'report.{module}.xml')
        for module in TWO_MODULES
    }
    assert not reports & merge_verify_reports, reports


def test_default_runs_groups_one_at_a_time(tmp_path):
    statuses, records = _run_two_recording_modules(tmp_path, concurrent_groups=False)

    assert statuses == TWO_MODULE_STATUSES
    spans = sorted(
        (
            min(r['start'] for r in records if r['module'] == module),
            max(r['end'] for r in records if r['module'] == module),
        )
        for module in TWO_MODULES
    )
    (_, first_end), (second_start, _) = spans
    assert first_end <= second_start, records


@pytest.mark.parametrize('concurrent_groups', [False, True], ids=['serial', 'concurrent'])
def test_a_module_that_writes_no_report_errors_only_its_own_ids(tmp_path, concurrent_groups):
    tree = _make_two_module_tree(tmp_path)
    module_configs = [
        _module_config('pkg_a'), _module_config('pkg_b', test_command='no-such-pytest-t7918'),
    ]

    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), module_configs, TWO_MODULE_IDS,
        concurrent_groups=concurrent_groups,
    ))

    assert dict(statuses) == {A_OK: 'pass', A_BAD: 'fail', B_OK: 'error', B_BAD: 'error'}


def test_ids_are_mapped_and_keyed_one_by_one(tmp_path):
    tree = _make_two_module_tree(tmp_path)
    a_ok_module_relative = 'tests/test_a.py::test_ok'
    unmappable = 'pkg_c/tests/test_c.py::test_ok'

    statuses = asyncio.run(run_node_ids(
        tree, _make_config(tree), [_module_config(module) for module in TWO_MODULES],
        [A_OK, a_ok_module_relative, unmappable, B_BAD],
    ))

    assert dict(statuses) == {
        A_OK: 'pass', a_ok_module_relative: 'pass', unmappable: 'error', B_BAD: 'fail',
    }
