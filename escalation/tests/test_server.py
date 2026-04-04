"""Tests for escalation server — merge_request helpers and integration."""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from escalation.server import (
    _filter_module_configs,
    _get_task_files,
)


def _mc(prefix: str) -> SimpleNamespace:
    """Create a lightweight stand-in for ModuleConfig."""
    return SimpleNamespace(prefix=prefix)


# --- Stubs for orchestrator types (not importable in escalation venv) ---


@dataclasses.dataclass
class _StubMergeRequest:
    task_id: str
    branch: str
    worktree: Path
    pre_rebased: bool
    task_files: list[str] | None
    module_configs: list
    config: object
    result: asyncio.Future


@dataclasses.dataclass
class _StubMergeOutcome:
    status: str = 'done'
    reason: str = ''
    conflict_details: str | None = None


def _mock_orchestrator_modules():
    """Inject mock orchestrator modules into sys.modules for deferred imports."""
    mock_config_mod = MagicMock()
    # Expose public name — _load_config_for_worktree should use discover_module_configs
    mock_config_mod.discover_module_configs = MagicMock(return_value={})
    mock_mq_mod = MagicMock()
    mock_mq_mod.MergeRequest = _StubMergeRequest
    mock_mq_mod.MergeOutcome = _StubMergeOutcome

    return patch.dict(sys.modules, {
        'orchestrator': MagicMock(),
        'orchestrator.config': mock_config_mod,
        'orchestrator.merge_queue': mock_mq_mod,
    }), mock_config_mod, mock_mq_mod


class TestFilterModuleConfigs:
    """_filter_module_configs returns only configs whose prefix matches task files."""

    def test_none_task_files_returns_all(self):
        configs = {'escalation': _mc('escalation'), 'orchestrator': _mc('orchestrator')}
        result = _filter_module_configs(configs, None)
        assert len(result) == 2

    def test_empty_task_files_returns_all(self):
        configs = {'escalation': _mc('escalation')}
        result = _filter_module_configs(configs, [])
        assert len(result) == 1

    def test_matching_prefix_included(self):
        configs = {
            'escalation': _mc('escalation'),
            'orchestrator': _mc('orchestrator'),
        }
        result = _filter_module_configs(
            configs, ['escalation/src/escalation/server.py'],
        )
        assert len(result) == 1
        assert result[0].prefix == 'escalation'

    def test_no_matching_prefix_returns_empty(self):
        configs = {'orchestrator': _mc('orchestrator')}
        result = _filter_module_configs(
            configs, ['dashboard/src/app.py'],
        )
        assert result == []

    def test_multiple_prefixes_matched(self):
        configs = {
            'escalation': _mc('escalation'),
            'orchestrator': _mc('orchestrator'),
            'shared': _mc('shared'),
        }
        result = _filter_module_configs(configs, [
            'escalation/src/escalation/server.py',
            'orchestrator/src/orchestrator/config.py',
        ])
        prefixes = {mc.prefix for mc in result}
        assert prefixes == {'escalation', 'orchestrator'}


class TestGetTaskFiles:
    """_get_task_files runs git diff and returns changed file paths."""

    @pytest.mark.asyncio
    async def test_returns_file_list_on_success(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b'escalation/src/escalation/server.py\nescalation/tests/test_server.py\n',
            b'',
        )
        mock_proc.returncode = 0

        with patch('escalation.server.asyncio.create_subprocess_exec', return_value=mock_proc):
            result = await _get_task_files(Path('/fake/worktree'))

        assert result == [
            'escalation/src/escalation/server.py',
            'escalation/tests/test_server.py',
        ]

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_diff(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'\n', b'')
        mock_proc.returncode = 0

        with patch('escalation.server.asyncio.create_subprocess_exec', return_value=mock_proc):
            result = await _get_task_files(Path('/fake/worktree'))

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'', b'fatal: bad revision')
        mock_proc.returncode = 128

        with patch('escalation.server.asyncio.create_subprocess_exec', return_value=mock_proc):
            result = await _get_task_files(Path('/fake/worktree'))

        assert result is None

    @pytest.mark.asyncio
    async def test_passes_worktree_as_cwd(self):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'a.py\n', b'')
        mock_proc.returncode = 0

        with patch('escalation.server.asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await _get_task_files(Path('/my/worktree'))

        mock_exec.assert_called_once()
        _, kwargs = mock_exec.call_args
        assert kwargs['cwd'] == '/my/worktree'

    @pytest.mark.asyncio
    async def test_returns_none_and_kills_on_timeout(self):
        """_get_task_files returns None and kills the process when git diff hangs.

        Patches asyncio.wait_for to raise TimeoutError immediately so the test
        does not actually wait 30 seconds.  When the real implementation wraps
        proc.communicate() in asyncio.wait_for(timeout=30.0), this patch fires
        and the TimeoutError handler must kill the process and return None.
        """
        mock_proc = AsyncMock()
        # communicate() returns normally when called for cleanup after kill
        mock_proc.communicate.return_value = (b'', b'')
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with (
            patch('escalation.server.asyncio.create_subprocess_exec', return_value=mock_proc),
            patch('escalation.server.asyncio.wait_for', side_effect=asyncio.TimeoutError),
        ):
            result = await _get_task_files(Path('/my/worktree'))

        assert result is None
        mock_proc.kill.assert_called_once()


class TestLoadConfigForWorktree:
    """_load_config_for_worktree loads project config and re-discovers module configs."""

    def test_uses_config_yaml_when_present(self, tmp_path):
        config_dir = tmp_path / 'orchestrator'
        config_dir.mkdir()
        config_file = config_dir / 'config.yaml'
        config_file.write_text('test: true')

        mock_config = MagicMock()
        sys_patch, mock_config_mod, _ = _mock_orchestrator_modules()
        mock_config_mod.load_config.return_value = mock_config

        with sys_patch:
            from escalation.server import _load_config_for_worktree as load_fn
            result = load_fn(tmp_path)

        mock_config_mod.load_config.assert_called_once_with(config_file)
        assert result is mock_config

    def test_passes_none_when_config_absent(self, tmp_path):
        mock_config = MagicMock()
        sys_patch, mock_config_mod, _ = _mock_orchestrator_modules()
        mock_config_mod.load_config.return_value = mock_config

        with sys_patch:
            from escalation.server import _load_config_for_worktree as load_fn
            result = load_fn(tmp_path)

        mock_config_mod.load_config.assert_called_once_with(None)
        assert result is mock_config

    def test_rediscovers_module_configs_from_worktree(self, tmp_path):
        """reload_module_configs is called with the worktree path (not _module_configs directly)."""
        mock_config = MagicMock()
        sys_patch, mock_config_mod, _ = _mock_orchestrator_modules()
        mock_config_mod.load_config.return_value = mock_config

        with sys_patch:
            from escalation.server import _load_config_for_worktree as load_fn
            load_fn(tmp_path)

        mock_config.reload_module_configs.assert_called_once_with(tmp_path)

    def test_does_not_access_private_module_configs(self, tmp_path):
        """_load_config_for_worktree never sets config._module_configs directly."""
        mock_config = MagicMock(spec=[
            'reload_module_configs', 'module_configs', 'for_module',
        ])
        sys_patch, mock_config_mod, _ = _mock_orchestrator_modules()
        mock_config_mod.load_config.return_value = mock_config

        with sys_patch:
            from escalation.server import _load_config_for_worktree as load_fn
            load_fn(tmp_path)

        # _module_configs should never be set as an attribute
        assert '_module_configs' not in mock_config.__dict__


class TestMergeRequestIntegration:
    """merge_request tool passes computed module_configs, task_files, and config."""

    @pytest.mark.asyncio
    async def test_merge_request_populates_scoping_fields(self):
        """MergeRequest receives task_files from git diff and filtered module_configs."""
        from escalation.queue import EscalationQueue
        from escalation.server import create_server

        captured: list = []

        async def capture_put(req):
            captured.append(req)
            req.result.set_result(_StubMergeOutcome(status='done', reason='merged'))

        mock_queue = MagicMock()
        mock_queue.put = capture_put

        esc_mc = _mc('escalation')
        task_files = ['escalation/src/escalation/server.py']

        mock_config = MagicMock()
        mock_config.module_configs = {'escalation': esc_mc, 'orchestrator': _mc('orchestrator')}

        sys_patch, _, _ = _mock_orchestrator_modules()

        with (
            sys_patch,
            patch('escalation.server._load_config_for_worktree', return_value=mock_config),
            patch('escalation.server._get_task_files', return_value=task_files),
        ):
            server = create_server(
                EscalationQueue(Path('/tmp/fake-queue')),
                merge_queue=mock_queue,
            )
            await server.call_tool(
                'merge_request',
                {'task_id': '426', 'branch': '426', 'worktree': '/fake/wt'},
            )

        assert len(captured) == 1
        req = captured[0]
        assert req.task_files == ['escalation/src/escalation/server.py']
        assert len(req.module_configs) == 1
        assert req.module_configs[0].prefix == 'escalation'
        assert req.config is mock_config

    @pytest.mark.asyncio
    async def test_merge_request_no_queue_returns_error(self):
        from escalation.queue import EscalationQueue
        from escalation.server import create_server

        server = create_server(EscalationQueue(Path('/tmp/fake-queue')), merge_queue=None)
        result = await server.call_tool(
            'merge_request',
            {'task_id': '1', 'branch': '1', 'worktree': '/fake/wt'},
        )
        assert any('error' in str(c) for c in result.content)
