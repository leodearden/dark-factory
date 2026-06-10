"""Tests for multi-host verify wiring in merge_queue.py (task 1716).

Covers:
  - _build_remote_runners helper (step-5 RED / step-6 GREEN)
  - _run_post_merge_verify pool wiring + dispatching-host scope derivation
    + cold-shadow guard (step-7 RED / step-8 GREEN)
  - _run_drift_check + _maybe_run_drift_check + land-hook integration
    (step-9 RED / step-10 GREEN)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import RemoteRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, verify_runners=None, main_branch='main', drift_every_n=20):
    """Build an OrchestratorConfig with optional verify_runners."""
    kwargs = {
        'git': GitConfig(main_branch=main_branch),
        'verify_drift_check_every_n_lands': drift_every_n,
    }
    if verify_runners is not None:
        kwargs['verify_runners'] = verify_runners
    return OrchestratorConfig(**kwargs)


def _make_runner_cfg(name, *, ssh_host='h.local', git_remote='r', config_path=None, enabled=True):
    return VerifyRunnerConfig(
        name=name,
        ssh_host=ssh_host,
        git_remote=git_remote,
        config_path=config_path,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# step-5: _build_remote_runners helper — RED
# ---------------------------------------------------------------------------


class TestBuildRemoteRunners:
    """Tests for _build_remote_runners(config, cwd, *, quarantine=None) -> list[RemoteRunner]."""

    def _call(self, config, cwd='/repo', *, quarantine=None):
        from orchestrator.merge_queue import _build_remote_runners
        return _build_remote_runners(config, cwd, quarantine=quarantine)

    def test_empty_verify_runners_returns_empty_list(self):
        """config.verify_runners=[] → returns []."""
        config = _make_config(verify_runners=[])
        result = self._call(config)
        assert result == []

    def test_two_enabled_runners_returns_two_remote_runners(self):
        """Two enabled VerifyRunnerConfig → two RemoteRunners in order."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('r1', ssh_host='h1.local', git_remote='remote1'),
            _make_runner_cfg('r2', ssh_host='h2.local', git_remote='remote2'),
        ])
        result = self._call(config, cwd='/myrepo')
        assert len(result) == 2
        assert all(isinstance(r, RemoteRunner) for r in result)
        assert result[0].name == 'r1'
        assert result[1].name == 'r2'

    def test_remote_runners_have_is_local_false(self):
        """Returned RemoteRunners have is_local=False."""
        config = _make_config(verify_runners=[_make_runner_cfg('r1')])
        result = self._call(config)
        assert result[0].is_local is False

    def test_runner_built_with_correct_ssh_host(self):
        """RemoteRunner stores the config ssh_host."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('r1', ssh_host='my-host.example.com', git_remote='origin'),
        ])
        result = self._call(config)
        assert result[0]._ssh_host == 'my-host.example.com'

    def test_runner_built_with_correct_git_remote(self):
        """RemoteRunner stores the config git_remote."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('r1', git_remote='my-remote'),
        ])
        result = self._call(config)
        assert result[0]._git_remote == 'my-remote'

    def test_runner_built_with_passed_cwd(self):
        """RemoteRunner._cwd matches the passed cwd argument."""
        config = _make_config(verify_runners=[_make_runner_cfg('r1')])
        result = self._call(config, cwd='/custom/repo')
        assert str(result[0]._cwd) == '/custom/repo'

    def test_runner_built_with_config_path(self):
        """RemoteRunner._config_path matches config.config_path."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('r1', config_path='/etc/orch.yaml'),
        ])
        result = self._call(config)
        assert result[0]._config_path == '/etc/orch.yaml'

    def test_runner_built_with_main_branch_from_config(self):
        """RemoteRunner._main_branch == config.git.main_branch."""
        config = _make_config(verify_runners=[_make_runner_cfg('r1')], main_branch='trunk')
        result = self._call(config)
        assert result[0]._main_branch == 'trunk'

    def test_disabled_runner_excluded(self):
        """A runner with enabled=False is excluded from the result."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('active', enabled=True),
            _make_runner_cfg('disabled', enabled=False),
        ])
        result = self._call(config)
        assert len(result) == 1
        assert result[0].name == 'active'

    def test_quarantine_excludes_named_runner(self):
        """quarantine={'r1'} excludes r1, keeps r2."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('r1'),
            _make_runner_cfg('r2'),
        ])
        result = self._call(config, quarantine={'r1'})
        assert len(result) == 1
        assert result[0].name == 'r2'

    def test_quarantine_none_includes_all_enabled(self):
        """quarantine=None (default) includes all enabled runners."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('r1'),
            _make_runner_cfg('r2'),
        ])
        result = self._call(config, quarantine=None)
        assert len(result) == 2

    def test_quarantine_empty_set_includes_all_enabled(self):
        """quarantine=set() (empty) includes all enabled runners."""
        config = _make_config(verify_runners=[
            _make_runner_cfg('r1'),
            _make_runner_cfg('r2'),
        ])
        result = self._call(config, quarantine=set())
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Helpers for _run_post_merge_verify / _run_cold_shadow_verify tests
# ---------------------------------------------------------------------------


def _make_pass_result(**kw):
    return VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok', **kw)


def _make_merge_request(config, *, task_files=None, worktree=None):
    """Build a MergeRequest for use in _run_post_merge_verify tests."""
    import asyncio
    from pathlib import Path
    from orchestrator.merge_queue import MergeRequest

    try:
        future = asyncio.get_running_loop().create_future()
    except RuntimeError:
        import concurrent.futures
        future = concurrent.futures.Future()

    return MergeRequest(
        task_id='task-42',
        branch='task/42',
        worktree=worktree or Path('/repo/task-42'),
        pre_rebased=False,
        task_files=task_files,
        module_configs=[],
        config=config,
        result=future,
    )


def _make_git_ops_mock():
    """Build a minimal async mock for GitOps."""
    mock = MagicMock()
    mock.get_main_sha = AsyncMock(return_value='main-sha')
    mock.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)  # plenty of disk
    mock.cleanup_merge_worktree = AsyncMock()
    mock.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    return mock


# ---------------------------------------------------------------------------
# step-7: _run_post_merge_verify pool wiring + scope derivation + cold-shadow guard — RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunPostMergeVerifyPoolWiring:
    """_run_post_merge_verify builds a multi-runner pool when verify_runners are configured."""

    async def test_pool_includes_remote_runner(self, tmp_path):
        """With an enabled verify_runner, the pool dispatches to the remote."""
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_config(verify_runners=[_make_runner_cfg('laptop')])
        req = _make_merge_request(config, task_files=['src/foo.py'], worktree=tmp_path)
        git_ops = _make_git_ops_mock()

        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        emitted = []

        class FakeEventStore:
            def emit(self, event_type, *, task_id=None, phase=None, data=None, **kw):
                emitted.append({'event_type': event_type, 'data': data or {}})

        with patch('orchestrator.merge_queue._build_remote_runners', return_value=[fake_remote]):
            outcome = await _run_post_merge_verify(
                git_ops, req, tmp_path,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
                event_store=FakeEventStore(),
                merge_sha='abc123',
            )

        assert outcome is None  # verify passed
        merge_verify_events = [e for e in emitted if hasattr(e['event_type'], 'value') and e['event_type'].value == 'merge_verify']
        assert len(merge_verify_events) >= 1
        assert merge_verify_events[0]['data']['runner'] == 'laptop'

    async def test_dispatching_host_derives_task_files_when_enabled_runners(self, tmp_path):
        """With enabled runner + task_files=None, derivation runs on dispatching host."""
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_config(verify_runners=[_make_runner_cfg('laptop')])
        req = _make_merge_request(config, task_files=None, worktree=tmp_path)
        git_ops = _make_git_ops_mock()

        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=_make_pass_result())

        spec_calls = []
        orig_build_spec = None

        import orchestrator.verify_runner as _vr
        orig_build = _vr.build_merge_verify_spec

        def spy_build_spec(config, module_configs, task_files, **kw):
            spec_calls.append(task_files)
            return orig_build(config, module_configs, task_files, **kw)

        with patch('orchestrator.merge_queue._build_remote_runners', return_value=[fake_remote]), \
             patch('orchestrator.merge_queue.build_merge_verify_spec', side_effect=spy_build_spec), \
             patch('orchestrator.merge_queue._derive_task_files_from_git',
                   new=AsyncMock(return_value=['src/x.py'])):
            await _run_post_merge_verify(
                git_ops, req, tmp_path,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
                merge_sha='abc123',
            )

        assert len(spec_calls) >= 1
        assert spec_calls[0] == ('src/x.py',)

    async def test_gate_no_derivation_when_no_enabled_runners(self, tmp_path):
        """With no enabled runners + task_files=None, _run_post_merge_verify does NOT
        proactively call _derive_task_files_from_git (Lever C gate — byte-identical path)."""
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_config(verify_runners=[])  # Lever C off
        req = _make_merge_request(config, task_files=None, worktree=tmp_path)
        git_ops = _make_git_ops_mock()

        spec_calls = []
        import orchestrator.verify_runner as _vr
        orig_build = _vr.build_merge_verify_spec

        def spy_build_spec(conf, module_configs, task_files, **kw):
            spec_calls.append(task_files)
            return orig_build(conf, module_configs, task_files, **kw)

        # Track proactive calls to _derive_task_files_from_git from _run_post_merge_verify.
        # We patch run_scoped_verification so the local runner doesn't actually run verify
        # (which would also call _derive internally), isolating the upstream derivation.
        derive_mock = AsyncMock(return_value=['src/x.py'])

        with patch('orchestrator.merge_queue.build_merge_verify_spec', side_effect=spy_build_spec), \
             patch('orchestrator.merge_queue.run_scoped_verification',
                   new=AsyncMock(return_value=_make_pass_result())), \
             patch('orchestrator.merge_queue._derive_task_files_from_git',
                   new=derive_mock):
            await _run_post_merge_verify(
                git_ops, req, tmp_path,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
                merge_sha='abc123',
            )

        # The proactive derivation must NOT have been called when Lever C is off
        derive_mock.assert_not_called()
        # spec must have received task_files=None (byte-identical local-only path)
        assert spec_calls[0] is None


@pytest.mark.asyncio
class TestColdShadowVerifyLocalOnly:
    """_run_cold_shadow_verify stays local-only even when verify_runners are configured."""

    async def test_cold_shadow_does_not_call_build_remote_runners(self, tmp_path):
        """_run_cold_shadow_verify never calls _build_remote_runners (trust-anchor guard)."""
        from orchestrator.merge_queue import _run_cold_shadow_verify

        config = _make_config(verify_runners=[_make_runner_cfg('laptop')])
        req = _make_merge_request(config, task_files=['src/foo.py'], worktree=tmp_path)
        git_ops = _make_git_ops_mock()

        build_remote_spy = MagicMock(return_value=[])

        with patch('orchestrator.merge_queue._build_remote_runners', build_remote_spy), \
             patch('orchestrator.merge_queue.run_scoped_verification',
                   new=AsyncMock(return_value=_make_pass_result())):
            await _run_cold_shadow_verify(git_ops, req, 'abc123', None)

        build_remote_spy.assert_not_called()

    async def test_cold_shadow_runs_on_local_runner(self, tmp_path):
        """_run_cold_shadow_verify uses a LocalRunner (not remote) even with verify_runners set."""
        from orchestrator.merge_queue import _run_cold_shadow_verify
        from orchestrator.verify_runner import LocalRunner

        config = _make_config(verify_runners=[_make_runner_cfg('laptop')])
        req = _make_merge_request(config, task_files=None, worktree=tmp_path)
        git_ops = _make_git_ops_mock()

        local_runner_instances = []
        orig_local_runner = LocalRunner

        class SpyLocalRunner(orig_local_runner):
            def __init__(self, *args, **kwargs):
                local_runner_instances.append(self)
                super().__init__(*args, **kwargs)

        with patch('orchestrator.merge_queue.LocalRunner', SpyLocalRunner), \
             patch('orchestrator.merge_queue.run_scoped_verification',
                   new=AsyncMock(return_value=_make_pass_result())):
            await _run_cold_shadow_verify(git_ops, req, 'abc123', None)

        assert len(local_runner_instances) >= 1
        assert all(r.is_local for r in local_runner_instances)
