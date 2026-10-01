"""Contract tests for ``nested_pytest_session``, the throwaway-session harness.

This file pins the harness's structural behaviour only: the nested session runs
against its own copy of ``df_pytest_isolation`` in its own rootdir, the conftest
template binds what it names, and the harness refuses inputs that would void
that isolation. The guard modules' end-to-end classes, which drive every real
nested run, remain the main regression detectors for it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from nested_pytest_session import (  # noqa: E402
    NESTED_SESSION_TIMEOUT_SECS,
    binding_conftest,
    run_nested_pytest,
)

_PROBE_SETUP = (
    'import df_pytest_isolation\n'
    "df_pytest_isolation.HARNESS_SPLICE_PROBE = 'spliced'\n"
)

_PROBE_TESTS = '''\
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_the_minimal_ini_makes_the_tmp_tree_the_rootdir(request):
    assert request.config.inipath == HERE / 'pytest.ini'
    assert request.config.rootpath == HERE


def test_the_session_imports_the_copy_not_the_repo_module():
    copy = Path(sys.modules['df_pytest_isolation'].__file__).resolve()
    assert copy.parent == HERE


def test_setup_is_spliced_in_before_the_binding():
    assert sys.modules['df_pytest_isolation'].HARNESS_SPLICE_PROBE == 'spliced'


def test_the_binding_registers_the_named_fixture(request):
    assert '_df_fleet_dir_redirect' in request.fixturenames
'''

_TRIVIAL_TEST = 'def test_passes():\n    pass\n'


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def test_a_bound_session_runs_against_its_own_copy_in_its_own_tree(tmp_path: Path) -> None:
    result = run_nested_pytest(
        tmp_path / 'bound',
        {
            'conftest.py': binding_conftest('_df_fleet_dir_redirect', setup=_PROBE_SETUP),
            'test_probe.py': _PROBE_TESTS,
        },
    )

    assert result.returncode == 0, _output(result)
    assert '4 passed' in _output(result), _output(result)


def test_explicit_targets_replace_the_default_whole_tree_target(tmp_path: Path) -> None:
    result = run_nested_pytest(
        tmp_path / 'targeted',
        {'a/test_one.py': _TRIVIAL_TEST, 'b/test_two.py': _TRIVIAL_TEST},
        targets=('a/test_one.py',),
    )

    assert result.returncode == 0, _output(result)
    assert '1 passed' in _output(result), _output(result)


def test_a_preexisting_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / 'taken'
    root.mkdir()

    with pytest.raises(FileExistsError):
        run_nested_pytest(root, {'test_one.py': _TRIVIAL_TEST})

    assert list(root.iterdir()) == []


@pytest.mark.parametrize('relpath', ['pytest.ini', 'df_pytest_isolation.py'])
def test_a_source_may_not_replace_the_harness_own_files(
    tmp_path: Path, relpath: str,
) -> None:
    root = tmp_path / 'clobbering'

    with pytest.raises(ValueError, match=relpath):
        run_nested_pytest(root, {relpath: '', 'test_one.py': _TRIVIAL_TEST})

    assert not root.exists()


def test_the_nested_cap_is_below_this_runs_per_test_timeout(
    pytestconfig: pytest.Config,
) -> None:
    pytest_timeout = pytest.importorskip('pytest_timeout')
    axe = pytest_timeout.get_env_settings(pytestconfig).timeout
    if not axe:
        pytest.skip('no per-test timeout configured, nothing can truncate the nested run')

    assert axe > NESTED_SESSION_TIMEOUT_SECS, (
        f'NESTED_SESSION_TIMEOUT_SECS={NESTED_SESSION_TIMEOUT_SECS} is not below '
        f'this run\'s per-test timeout of {axe}s, so a wedged nested run would be '
        'killed by the per-test axe instead of surfacing as TimeoutExpired with '
        'its captured output.'
    )
