"""Contract tests for ``nested_pytest_session``, the throwaway-session harness.

This file pins the harness's structural behaviour only: the nested session runs
against its own copy of ``df_pytest_isolation`` in its own rootdir, the conftest
template binds what it names, and the harness refuses inputs that would void
that isolation. The guard modules' end-to-end classes, which drive every real
nested run, remain the main regression detectors for it.
"""

from __future__ import annotations

import getpass
import subprocess
from pathlib import Path

import pytest
from nested_pytest_session import (
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

_TEMPROOT_RECORDER = '''\
import os
from pathlib import Path


def test_records_the_inherited_temproot(tmp_path):
    record = Path(__file__).resolve().parent / 'inherited-temproot'
    record.write_text(os.environ.get('PYTEST_DEBUG_TEMPROOT', ''))
'''


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob('*'))


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


def test_a_nested_session_leaves_the_hosts_default_temp_root_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = tmp_path / 'host-temp'
    (host / f'pytest-of-{getpass.getuser()}' / 'garbage-seeded' / 'leftover').mkdir(parents=True)
    before = _listing(host)
    monkeypatch.setenv('PYTEST_DEBUG_TEMPROOT', str(host))
    root = tmp_path / 'temproot-probe'

    result = run_nested_pytest(root, {'test_records_temproot.py': _TEMPROOT_RECORDER})

    assert result.returncode == 0, _output(result)
    assert '1 passed' in _output(result), _output(result)
    assert (root / 'inherited-temproot').read_text() == str(host), (
        'nested session did not inherit PYTEST_DEBUG_TEMPROOT'
    )
    after = _listing(host)
    assert after == before, (
        "nested session touched the host's default temp root\n"
        f'before: {before}\nafter:  {after}'
    )


@pytest.mark.parametrize(
    'spelling', [('--basetemp={}',), ('--basetemp', '{}')], ids=['joined', 'separate'],
)
def test_a_basetemp_smuggled_through_targets_is_refused_before_any_session_starts(
    tmp_path: Path, spelling: tuple[str, ...],
) -> None:
    smuggled = tmp_path / 'elsewhere'
    targets = ('test_one.py', *(arg.format(smuggled) for arg in spelling))
    root = tmp_path / 'smuggling'

    with pytest.raises(ValueError, match='--basetemp') as refused:
        run_nested_pytest(root, {'test_one.py': _TRIVIAL_TEST}, targets=targets)

    assert targets[1] in str(refused.value)
    assert not root.exists() and not smuggled.exists(), 'a session started anyway'


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
