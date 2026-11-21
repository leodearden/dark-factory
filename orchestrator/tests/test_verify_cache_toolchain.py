"""The toolchain a verify leg runs under, as cache-key inputs (task 5652)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling

from orchestrator.verify_cache import toolchain

_PYVENV_CFG = 'home = /x\nimplementation = CPython\nversion_info = 3.13.9\n'

_UV_LOCK = """\
version = 1

[[package]]
name = "ruff"
version = "0.15.9"

[[package]]
name = "click"
version = "8.1.7"

[[package]]
name = "pyright"
version = "1.1.408"
"""

_PACKAGE_LOCK = json.dumps({
    'name': 'fixture',
    'lockfileVersion': 3,
    'packages': {'': {}, 'node_modules/pyright': {'version': '1.1.408'}},
})


def _fingerprint(**overrides: object) -> toolchain.ToolchainFingerprint:
    fields: dict = {
        'role': 'merge',
        'interpreter': 'CPython 3.13.9',
        'uv_lock_blob': 'a' * 40,
        'tool_versions': (('ruff', 'uv.lock', '0.15.9'),),
        'verify_env': (('DF_VERIFY_ROLE', 'merge'), ('K', 'V')),
        'preprovision_command': 'uv sync --all-packages',
        'orchestrator_code': 'c' * 40,
    }
    return toolchain.ToolchainFingerprint(**{**fields, **overrides})


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, env=git_env_with_ceiling(repo),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _worktree_with_venv(tmp_path: Path) -> Path:
    worktree = tmp_path / 'worktree'
    (worktree / '.venv').mkdir(parents=True)
    (worktree / '.venv' / 'pyvenv.cfg').write_text(_PYVENV_CFG)
    return worktree


class TestParsers:
    def test_the_interpreter_is_implementation_and_version(self):
        assert toolchain.interpreter_from_pyvenv_cfg(_PYVENV_CFG) == 'CPython 3.13.9'

    def test_a_cfg_without_version_info_is_unavailable(self):
        with pytest.raises(toolchain.ToolchainUnavailable, match='version_info'):
            toolchain.interpreter_from_pyvenv_cfg('home = /x\nimplementation = CPython\n')

    def test_tool_versions_come_from_both_lockfiles_sorted(self):
        assert toolchain.tool_versions_from_lockfiles(_UV_LOCK, _PACKAGE_LOCK) == (
            ('pyright', 'package-lock.json', '1.1.408'),
            ('pyright', 'uv.lock', '1.1.408'),
            ('ruff', 'uv.lock', '0.15.9'),
        )

    def test_an_absent_lockfile_contributes_nothing(self):
        assert toolchain.tool_versions_from_lockfiles(None, _PACKAGE_LOCK) == (
            ('pyright', 'package-lock.json', '1.1.408'),
        )
        assert toolchain.tool_versions_from_lockfiles(None, None) == ()

    def test_a_malformed_uv_lock_is_unavailable_naming_it(self):
        with pytest.raises(toolchain.ToolchainUnavailable, match='uv.lock'):
            toolchain.tool_versions_from_lockfiles('[[package', None)

    def test_a_malformed_package_lock_is_unavailable_naming_it(self):
        with pytest.raises(toolchain.ToolchainUnavailable, match='package-lock.json'):
            toolchain.tool_versions_from_lockfiles(None, '{not json')


class TestFingerprintInputs:
    def test_inputs_are_sorted_toolchain_names_including_the_role(self):
        inputs = _fingerprint().as_inputs()

        assert list(inputs) == sorted(inputs)
        assert all(name.startswith('toolchain.') for name, _value in inputs)
        assert ('toolchain.role', 'merge') in inputs

    @pytest.mark.parametrize('override', [
        {'role': 'task'},
        {'preprovision_command': ''},
        {'verify_env': (('DF_VERIFY_ROLE', 'merge'), ('K', 'W'))},
        {'interpreter': 'CPython 3.14.0'},
        {'tool_versions': (('ruff', 'uv.lock', '0.15.10'),)},
        {'orchestrator_code': 'd' * 40},
    ])
    def test_any_one_field_changes_the_inputs(self, override):
        assert _fingerprint(**override).as_inputs() != _fingerprint().as_inputs()

    def test_equal_fields_give_equal_inputs(self):
        assert _fingerprint().as_inputs() == _fingerprint().as_inputs()

    def test_verify_env_order_is_not_identity(self):
        reordered = _fingerprint(verify_env=(('K', 'V'), ('DF_VERIFY_ROLE', 'merge')))

        assert reordered == _fingerprint()
        assert reordered.as_inputs() == _fingerprint().as_inputs()

    @pytest.mark.parametrize('role', ['', 'Merge', 'merge '])
    def test_a_role_outside_the_verify_roles_is_rejected_naming_it(self, role):
        with pytest.raises(ValueError, match=f'role {role!r}'):
            _fingerprint(role=role)

    def test_a_fingerprint_needs_the_orchestrator_code_identity(self):
        with pytest.raises(ValueError, match='orchestrator code'):
            _fingerprint(orchestrator_code='')

    def test_a_duplicate_verify_env_name_is_rejected_naming_it(self):
        with pytest.raises(ValueError, match="verify_env name.*'K'"):
            _fingerprint(verify_env=(('K', 'V'), ('K', 'W')))


class TestReadToolchain:
    @pytest.mark.asyncio
    async def test_a_python_tree_is_fingerprinted_from_tree_and_worktree(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo', RepoSeed(
            files=(('uv.lock', _UV_LOCK), ('package-lock.json', _PACKAGE_LOCK)),
            message='lockfiles',
        ))
        tree = _git(repo, 'rev-parse', 'HEAD^{tree}')

        fingerprint = await toolchain.read_toolchain(
            _worktree_with_venv(tmp_path), repo, tree,
            role='task', verify_env={'K': 'V'}, preprovision_command='uv sync', orchestrator_code='c' * 40,
        )

        assert dataclasses.astuple(fingerprint) == (
            'task',
            'CPython 3.13.9',
            _git(repo, 'rev-parse', f'{tree}:uv.lock'),
            (
                ('pyright', 'package-lock.json', '1.1.408'),
                ('pyright', 'uv.lock', '1.1.408'),
                ('ruff', 'uv.lock', '0.15.9'),
            ),
            (('K', 'V'),),
            'uv sync',
            'c' * 40,
        )

    @pytest.mark.asyncio
    async def test_a_python_tree_without_a_venv_is_unavailable(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo', RepoSeed(files=(('uv.lock', _UV_LOCK),), message='lock'))
        worktree = tmp_path / 'cold-worktree'
        worktree.mkdir()

        with pytest.raises(toolchain.ToolchainUnavailable, match='pyvenv.cfg'):
            await toolchain.read_toolchain(
                worktree, repo, _git(repo, 'rev-parse', 'HEAD^{tree}'),
                role='merge', verify_env={}, preprovision_command='', orchestrator_code='c' * 40,
            )

    @pytest.mark.asyncio
    async def test_a_non_python_tree_needs_no_interpreter(self, tmp_path):
        repo = seed_repo(tmp_path / 'repo', RepoSeed(files=(('Cargo.toml', '[workspace]\n'),), message='rust'))
        worktree = tmp_path / 'rust-worktree'
        worktree.mkdir()

        fingerprint = await toolchain.read_toolchain(
            worktree, repo, _git(repo, 'rev-parse', 'HEAD^{tree}'),
            role='merge', verify_env={}, preprovision_command='', orchestrator_code='c' * 40,
        )

        assert fingerprint.interpreter is None
        assert fingerprint.uv_lock_blob is None
