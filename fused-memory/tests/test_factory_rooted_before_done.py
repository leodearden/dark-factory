"""Factory-rooted before_done scripts at the submit boundary (task 6386).

``before_done.script_root='factory'`` resolves a predicate's script against the
dark-factory tooling checkout instead of the task's own project, so one
factory-owned predicate can be filed against any project. Only the public
seams are driven: ``deterministic_task_error`` and the ``submit_task`` tool.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from shared.repo_paths import DARK_FACTORY_ROOT_ENV
from shared.task_metadata import BeforeDone

from fused_memory.middleware.deterministic_task_guard import deterministic_task_error
from fused_memory.server.tools import create_mcp_server

SCRIPT = 'scripts/check.sh'


def _install_script(root: Path, relpath: str = SCRIPT, *, mode: int = 0o755) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(mode)
    return path


def _before_done(**overrides: Any) -> dict[str, Any]:
    return {'script': SCRIPT, 'timeout_secs': 60, **overrides}


def _guard(
    before_done: dict[str, Any],
    project_root: Path,
    factory_root_resolver: Callable[[], Path | None],
) -> dict[str, Any] | None:
    return deterministic_task_error(
        'deterministic',
        {'before_done': before_done},
        str(project_root),
        factory_root_resolver=factory_root_resolver,
    )


def _never_consulted() -> Path | None:
    raise AssertionError('the factory root resolver must not run for a project-rooted script')


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / 'foreign'
    root.mkdir()
    return root


@pytest.fixture
def factory_root(tmp_path: Path) -> Path:
    root = tmp_path / 'factory'
    _install_script(root)
    return root


class TestScriptRootGuardTable:
    def test_factory_predicate_accepted_without_a_project_copy(
        self, project_root: Path, factory_root: Path,
    ) -> None:
        bd = _before_done(script_root='factory', kind='predicate')

        assert _guard(bd, project_root, lambda: factory_root) is None

    @pytest.mark.parametrize(
        'kind_fields', [{'kind': 'deploy'}, {}], ids=['explicit-deploy', 'default-deploy'],
    )
    def test_factory_deploy_refused(
        self, project_root: Path, factory_root: Path, kind_fields: dict[str, str],
    ) -> None:
        bd = _before_done(script_root='factory', **kind_fields)

        result = _guard(bd, project_root, lambda: factory_root)

        assert result is not None
        assert result['error_type'] == 'ValidationError'
        assert 'script_root' in result['error']
        assert 'predicate' in result['error']

    @pytest.mark.parametrize('kind', ['deploy', 'predicate'])
    def test_project_rooted_script_under_project_root_accepted(
        self, project_root: Path, factory_root: Path, kind: str,
    ) -> None:
        _install_script(project_root)
        bd = _before_done(script_root='project', kind=kind)

        assert _guard(bd, project_root, lambda: factory_root) is None

    @pytest.mark.parametrize('kind', ['deploy', 'predicate'])
    def test_project_rooted_script_only_under_factory_root_refused(
        self, project_root: Path, factory_root: Path, kind: str,
    ) -> None:
        bd = _before_done(script_root='project', kind=kind)

        result = _guard(bd, project_root, lambda: factory_root)

        assert result is not None
        assert 'does not exist under project_root' in result['error']

    def test_factory_script_only_under_project_root_refused(
        self, project_root: Path, factory_root: Path,
    ) -> None:
        _install_script(project_root, 'scripts/project-only.sh')
        bd = _before_done(
            script='scripts/project-only.sh', script_root='factory', kind='predicate',
        )

        result = _guard(bd, project_root, lambda: factory_root)

        assert result is not None
        assert 'does not exist under the dark-factory root' in result['error']

    def test_factory_script_escaping_the_factory_root_refused(
        self, tmp_path: Path, project_root: Path, factory_root: Path,
    ) -> None:
        _install_script(tmp_path, 'escape.sh')
        bd = _before_done(script='../escape.sh', script_root='factory', kind='predicate')

        result = _guard(bd, project_root, lambda: factory_root)

        assert result is not None
        assert 'resolves outside the dark-factory root' in result['error']

    def test_non_executable_factory_script_refused(
        self, project_root: Path, factory_root: Path,
    ) -> None:
        _install_script(factory_root, 'scripts/noexec.sh', mode=0o644)
        bd = _before_done(script='scripts/noexec.sh', script_root='factory', kind='predicate')

        result = _guard(bd, project_root, lambda: factory_root)

        assert result is not None
        assert 'executable' in result['error']

    def test_unresolvable_factory_root_refused(self, project_root: Path) -> None:
        bd = _before_done(script_root='factory', kind='predicate')

        result = _guard(bd, project_root, lambda: None)

        assert result is not None
        assert result['error_type'] == 'ValidationError'
        assert "script_root='factory'" in result['error']
        assert 'dark-factory root could not be resolved' in result['error']
        assert DARK_FACTORY_ROOT_ENV in result['hint']

    def test_unknown_script_root_refused_naming_the_valid_values(
        self, project_root: Path, factory_root: Path,
    ) -> None:
        bd = _before_done(script_root='elsewhere', kind='predicate')

        result = _guard(bd, project_root, lambda: factory_root)

        assert result is not None
        assert 'elsewhere' in result['error']
        assert "'factory'" in result['error']
        assert "'project'" in result['error']

    @pytest.mark.parametrize(
        'root_fields', [{'script_root': 'project'}, {}], ids=['explicit', 'default'],
    )
    def test_project_rooted_script_never_consults_the_resolver(
        self, project_root: Path, root_fields: dict[str, str],
    ) -> None:
        _install_script(project_root)
        bd = _before_done(kind='predicate', **root_fields)

        assert _guard(bd, project_root, _never_consulted) is None


class TestGuardAndModelAgree:
    """One rule, two enforcement points: the guard accepts exactly what the
    shared ``BeforeDone`` model accepts across ``script_root`` x ``kind``."""

    @pytest.mark.parametrize('script_root', ['project', 'factory', 'elsewhere'])
    @pytest.mark.parametrize('kind', ['deploy', 'predicate'])
    def test_parity(
        self, project_root: Path, factory_root: Path, script_root: str, kind: str,
    ) -> None:
        _install_script(project_root)
        bd = _before_done(kind=kind, script_root=script_root)
        try:
            BeforeDone.model_validate(bd)
            model_accepts = True
        except ValidationError:
            model_accepts = False

        guard_accepts = _guard(bd, project_root, lambda: factory_root) is None

        assert guard_accepts == model_accepts


class TestSubmitTaskFactoryRootedPredicate:
    """The user-observable signal: a factory-owned predicate files against a
    project that does not carry the script, resolved by the DEFAULT resolver
    to the dark-factory checkout these tests run from. The script is the
    resolver's own validity marker, so it exists wherever that resolver
    succeeds."""

    FACTORY_SCRIPT = 'scripts/watcher-rearm.sh'

    @pytest.fixture
    def git_project_root(self, project_root: Path) -> Path:
        subprocess.run(['git', 'init', '-q', str(project_root)], check=True)
        return project_root

    @pytest.fixture
    def interceptor(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        monkeypatch.delenv(DARK_FACTORY_ROOT_ENV, raising=False)
        ti = AsyncMock()
        ti.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
        return ti

    async def _submit(
        self, interceptor: AsyncMock, project_root: Path, before_done: dict[str, Any],
    ) -> Any:
        server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
        return await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': str(project_root),
                'title': 'A factory-owned predicate',
                'description': 'A factory-owned predicate filed against another project.',
                'task_kind': 'deterministic',
                'metadata': {'before_done': before_done},
            },
        )

    def _predicate(self, **overrides: Any) -> dict[str, Any]:
        return {
            'script': self.FACTORY_SCRIPT,
            'script_root': 'factory',
            'kind': 'predicate',
            'timeout_secs': 60,
            **overrides,
        }

    @pytest.mark.asyncio
    async def test_factory_rooted_predicate_is_accepted_and_forwarded(
        self, interceptor: AsyncMock, git_project_root: Path,
    ) -> None:
        result = await self._submit(interceptor, git_project_root, self._predicate())

        assert 'error' not in result, result
        interceptor.submit_task.assert_awaited_once()
        forwarded = interceptor.submit_task.call_args.kwargs['metadata']
        assert forwarded['before_done']['script_root'] == 'factory'

    @pytest.mark.asyncio
    async def test_without_script_root_the_script_is_looked_up_in_the_project(
        self, interceptor: AsyncMock, git_project_root: Path,
    ) -> None:
        bd = self._predicate()
        del bd['script_root']

        result = await self._submit(interceptor, git_project_root, bd)

        assert 'does not exist under project_root' in result['error']
        interceptor.submit_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_factory_rooted_deploy_is_refused(
        self, interceptor: AsyncMock, git_project_root: Path,
    ) -> None:
        result = await self._submit(
            interceptor, git_project_root, self._predicate(kind='deploy'),
        )

        assert 'script_root' in result['error']
        interceptor.submit_task.assert_not_awaited()
