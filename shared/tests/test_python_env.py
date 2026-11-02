"""Which environment variables select a Python environment, and an environ with none of them.

The names below are the external contract, so every one is spelled as a literal
here rather than imported from ``shared.python_env``.
"""

from __future__ import annotations

import pytest

from shared.python_env import without_python_env_selection

ORCH_VENV = '/srv/orch/.venv'

PYTHON_ENV_SELECTION_NAMES = (
    'VIRTUAL_ENV',
    'UV_PROJECT_ENVIRONMENT',
    'UV_PROJECT',
    'UV_ACTIVE',
    'UV_FROZEN',
    'UV_NO_SYNC',
    'UV_RUN_RECURSION_DEPTH',
    'CONDA_PREFIX',
    'CONDA_DEFAULT_ENV',
    'PYTHONHOME',
)

ORDINARY_ENV = {
    'HOME': '/home/someone',
    'LANG': 'en_US.UTF-8',
    'CARGO_INCREMENTAL': '0',
    'CLAUDE_CODE_OAUTH_TOKEN': 'session-' + 's' * 40,
    'ORCH_CONFIG_PATH': '/srv/orch/dark-factory-orchestrator.yaml',
}


@pytest.mark.parametrize('name', PYTHON_ENV_SELECTION_NAMES)
def test_each_python_env_selection_name_is_dropped(name):
    environ = {name: '/srv/orch/something', **ORDINARY_ENV}
    assert name not in without_python_env_selection(environ)


class TestVenvBinOnPath:
    def test_a_leading_venv_bin_is_removed_and_the_rest_keep_their_order(self):
        environ = {
            'VIRTUAL_ENV': ORCH_VENV,
            'PATH': f'{ORCH_VENV}/bin:/home/someone/.cargo/bin:/usr/bin:/bin',
        }
        assert without_python_env_selection(environ)['PATH'] == (
            '/home/someone/.cargo/bin:/usr/bin:/bin'
        )

    def test_a_non_leading_venv_bin_is_removed(self):
        environ = {'VIRTUAL_ENV': ORCH_VENV, 'PATH': f'/usr/bin:{ORCH_VENV}/bin:/bin'}
        assert without_python_env_selection(environ)['PATH'] == '/usr/bin:/bin'

    def test_a_trailing_slash_on_virtual_env_still_matches(self):
        environ = {'VIRTUAL_ENV': ORCH_VENV + '/', 'PATH': f'{ORCH_VENV}/bin:/usr/bin'}
        assert without_python_env_selection(environ)['PATH'] == '/usr/bin'

    def test_without_virtual_env_path_is_byte_identical(self):
        path = '/home/someone/proj/.venv/bin:/usr/bin::/bin/'
        assert without_python_env_selection({'PATH': path})['PATH'] == path

    def test_without_path_the_result_has_no_path(self):
        assert 'PATH' not in without_python_env_selection({'VIRTUAL_ENV': ORCH_VENV})


def test_ordinary_names_pass_through_unchanged():
    environ = {'VIRTUAL_ENV': ORCH_VENV, **ORDINARY_ENV}
    assert without_python_env_selection(environ) == ORDINARY_ENV


def test_returns_a_new_dict_and_leaves_the_input_unmutated():
    environ = {
        'VIRTUAL_ENV': ORCH_VENV,
        'UV_RUN_RECURSION_DEPTH': '1',
        'PATH': f'{ORCH_VENV}/bin:/usr/bin',
        **ORDINARY_ENV,
    }
    snapshot = dict(environ)
    child = without_python_env_selection(environ)
    assert isinstance(child, dict)
    assert child is not environ
    assert environ == snapshot
