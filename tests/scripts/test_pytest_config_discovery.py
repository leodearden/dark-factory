"""The contract of ``tests/scripts/pytest_config_discovery.py``, driven through its public seam.

Each workspace here is literal TOML written under ``tmp_path``; no subprocess runs.
"""
from __future__ import annotations

import pathlib
import re

import pytest
from pytest_config_discovery import (
    ROOT_CONFIG_NAME,
    discovered_pytest_configs,
    workspace_members,
)


def _write_pyproject(directory: pathlib.Path, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'pyproject.toml').write_text(text, encoding='utf-8')


def test_discovers_the_root_and_every_member_that_declares_a_pytest_table(
    tmp_path: pathlib.Path,
) -> None:
    _write_pyproject(
        tmp_path,
        '[tool.uv.workspace]\n'
        'members = ["alpha", "beta", "gamma"]\n'
        '\n'
        '[tool.pytest.ini_options]\n'
        'timeout = 5\n',
    )
    _write_pyproject(tmp_path / 'alpha', '[tool.pytest.ini_options]\ntimeout_method = "thread"\n')
    _write_pyproject(tmp_path / 'beta', '[tool.pytest.ini_options]\n')
    _write_pyproject(tmp_path / 'gamma', '[project]\nname = "gamma"\n')

    assert discovered_pytest_configs(tmp_path, min_expected=3) == {
        ROOT_CONFIG_NAME: {'timeout': 5},
        'alpha': {'timeout_method': 'thread'},
        'beta': {},
    }


def test_workspace_members_returns_the_declared_members_in_order(tmp_path: pathlib.Path) -> None:
    _write_pyproject(tmp_path, '[tool.uv.workspace]\nmembers = ["beta", "alpha"]\n')
    _write_pyproject(tmp_path / 'alpha', '[project]\nname = "alpha"\n')
    _write_pyproject(tmp_path / 'beta', '[project]\nname = "beta"\n')

    assert workspace_members(tmp_path) == ['beta', 'alpha']


@pytest.mark.parametrize(
    'member',
    [
        pytest.param('delta', id='directory-without-pyproject'),
        pytest.param('pkgs/*', id='glob-member'),
    ],
)
def test_a_member_without_a_pyproject_fails_loudly(tmp_path: pathlib.Path, member: str) -> None:
    _write_pyproject(
        tmp_path,
        '[tool.uv.workspace]\n'
        f'members = ["alpha", "{member}"]\n'
        '\n'
        '[tool.pytest.ini_options]\n'
        'timeout = 5\n',
    )
    _write_pyproject(tmp_path / 'alpha', '[tool.pytest.ini_options]\ntimeout = 5\n')
    (tmp_path / 'delta').mkdir()
    _write_pyproject(tmp_path / 'pkgs' / 'one', '[tool.pytest.ini_options]\ntimeout = 5\n')

    with pytest.raises(AssertionError) as refused:
        discovered_pytest_configs(tmp_path, min_expected=1)

    message = str(refused.value)
    assert repr(member) in message, message
    assert str(tmp_path / member / 'pyproject.toml') in message, message


@pytest.mark.parametrize(
    'root_pyproject',
    [
        pytest.param('[tool.pytest.ini_options]\ntimeout = 5\n', id='no-workspace-table'),
        pytest.param(
            '[tool.uv.workspace]\nmembers = []\n\n[tool.pytest.ini_options]\ntimeout = 5\n',
            id='empty-members',
        ),
    ],
)
def test_a_workspace_without_members_fails_loudly(
    tmp_path: pathlib.Path, root_pyproject: str,
) -> None:
    _write_pyproject(tmp_path, root_pyproject)

    with pytest.raises(AssertionError) as refused:
        discovered_pytest_configs(tmp_path, min_expected=1)

    assert '[tool.uv.workspace].members' in str(refused.value)


def test_discovering_fewer_than_the_floor_fails_loudly(tmp_path: pathlib.Path) -> None:
    _write_pyproject(
        tmp_path,
        '[tool.uv.workspace]\n'
        'members = ["alpha"]\n'
        '\n'
        '[tool.pytest.ini_options]\n'
        'timeout = 5\n',
    )
    _write_pyproject(tmp_path / 'alpha', '[project]\nname = "alpha"\n')

    with pytest.raises(AssertionError) as refused:
        discovered_pytest_configs(tmp_path, min_expected=7)

    message = str(refused.value)
    assert re.search(r'\b7\b', message), message
    assert str([ROOT_CONFIG_NAME]) in message, message


def test_this_checkouts_discovery_clears_the_default_floor() -> None:
    configs = discovered_pytest_configs()

    assert ROOT_CONFIG_NAME in configs
