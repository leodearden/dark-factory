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
