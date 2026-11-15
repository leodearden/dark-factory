"""Task 6565 design D1: the sha the process's verify wire path was imported from."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.deployed_sha import (
    DeployedSha,
    capture_deployed_sha,
    checkout_head,
    resolve_code_checkout,
    wire_path_is_dirty,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, 'add', '-A')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', message)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A git repo with one tracked file in each wire path and one outside it."""
    repo = tmp_path / 'df'
    for rel in ('orchestrator/src/orchestrator/verify.py', 'shared/src/shared/x.py', 'docs/a.md'):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text('v1\n')
    _git(tmp_path, 'init', '-q', str(repo))
    _commit_all(repo, 'base')
    return repo


class TestResolveCodeCheckout:
    def test_returns_the_root_of_the_checkout_holding_this_code(self):
        from orchestrator import deployed_sha

        root = resolve_code_checkout()
        assert root is not None
        assert (root / '.git').exists()
        assert Path(deployed_sha.__file__).resolve().is_relative_to(root)

    def test_finds_git_root_from_nested_start(self, tmp_path: Path):
        (tmp_path / '.git').mkdir()
        nested = tmp_path / 'a' / 'b' / 'c'
        nested.mkdir(parents=True)
        assert resolve_code_checkout(start=nested) == tmp_path

    def test_returns_none_when_no_ancestor_has_git(self, tmp_path: Path, monkeypatch):
        """A stray ``/tmp/.git`` exists on some hosts, so every probe is forced to miss."""
        monkeypatch.setattr(Path, 'exists', lambda self: False)
        assert resolve_code_checkout(start=tmp_path) is None


class TestCaptureDeployedSha:
    def test_still_head_gives_a_determinate_sha(self, checkout: Path):
        calls: list[str] = []

        result = capture_deployed_sha(checkout, lambda: calls.append('imported'))

        assert calls == ['imported']
        assert result == DeployedSha(
            sha=_git(checkout, 'rev-parse', 'HEAD'), indeterminate=False, dirty_wire_path=False,
        )

    def test_head_moving_during_the_import_is_indeterminate(self, checkout: Path):
        """A landing between the two reads: the sha is the later HEAD, flagged."""
        before = _git(checkout, 'rev-parse', 'HEAD')

        def land_during_import() -> None:
            (checkout / 'orchestrator/src/orchestrator/verify.py').write_text('v2\n')
            _commit_all(checkout, 'landing')

        result = capture_deployed_sha(checkout, land_during_import)

        assert result is not None
        assert result.indeterminate is True
        assert result.sha == _git(checkout, 'rev-parse', 'HEAD') != before
        assert result.flagged

    def test_no_checkout_still_imports_and_gives_none(self):
        calls: list[str] = []
        assert capture_deployed_sha(None, lambda: calls.append('imported')) is None
        assert calls == ['imported']

    def test_unreadable_checkout_still_imports_and_gives_none(self, tmp_path: Path):
        calls: list[str] = []
        assert capture_deployed_sha(tmp_path, lambda: calls.append('imported')) is None
        assert calls == ['imported']


class TestWirePathDirty:
    def test_clean_tree_is_not_dirty(self, checkout: Path):
        assert wire_path_is_dirty(checkout) is False

    def test_untracked_file_in_the_wire_path_is_not_dirty(self, checkout: Path):
        (checkout / 'orchestrator/src/orchestrator/new_module.py').write_text('x\n')
        assert wire_path_is_dirty(checkout) is False

    @pytest.mark.parametrize('rel', ['orchestrator/src/orchestrator/verify.py', 'shared/src/shared/x.py'])
    def test_tracked_edit_in_the_wire_path_is_dirty(self, checkout: Path, rel: str):
        (checkout / rel).write_text('edited\n')
        assert wire_path_is_dirty(checkout) is True

    def test_tracked_edit_outside_the_wire_path_is_not_dirty(self, checkout: Path):
        (checkout / 'docs/a.md').write_text('edited\n')
        assert wire_path_is_dirty(checkout) is False

    def test_capture_records_the_dirty_flag(self, checkout: Path):
        (checkout / 'shared/src/shared/x.py').write_text('edited\n')
        result = capture_deployed_sha(checkout, lambda: None)
        assert result is not None
        assert result.dirty_wire_path is True
        assert result.indeterminate is False
        assert result.sha == checkout_head(checkout)


def test_deployed_sha_wire_form_is_json_native():
    deployed = DeployedSha(sha='b' * 40, indeterminate=False, dirty_wire_path=True)
    assert deployed.to_wire() == {'sha': 'b' * 40, 'indeterminate': False, 'dirty_wire_path': True}
    assert deployed.flagged


def test_importing_the_harness_loads_every_wire_module():
    """D1's premise: the bracketed import pins the whole verify wire path.

    Run in a fresh interpreter, since this test process has imported far more.
    """
    wire = [
        'orchestrator.verify', 'orchestrator.verify_runner', 'orchestrator.flake_ledger',
        'orchestrator.verify_admission_record', 'orchestrator.task_verify_spec',
        'orchestrator.verify_spec_shared',
    ]
    probe = f'import sys, orchestrator.harness; print([m for m in {wire!r} if m not in sys.modules])'
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]'


def test_build_remote_runners_hands_every_runner_the_deployed_sha():
    from orchestrator.verify_hosts import build_remote_runners

    deployed = DeployedSha(sha='c' * 40, indeterminate=False, dirty_wire_path=False)
    config = OrchestratorConfig(
        git=GitConfig(main_branch='main'),
        verify_env_probes=[],
        verify_runners=[
            VerifyRunnerConfig(
                name=name, ssh_host='h.local', git_remote='r', df_checkout_path='/df',
            )
            for name in ('r1', 'r2')
        ],
    )

    runners = build_remote_runners(config, '/repo', deployed_sha=deployed)

    assert [r.deployed_sha for r in runners] == [deployed, deployed]


class TestHarnessAnnouncesTheDeployedSha:
    """Leo 2026-10-20: a flagged sha is logged and emitted; the laptop stays in use."""

    @staticmethod
    def _harness(mock_orch_config, deployed: DeployedSha | None):
        from unittest.mock import MagicMock, patch

        from orchestrator.harness import Harness

        with patch('orchestrator.harness.McpLifecycle'), \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            harness = Harness(mock_orch_config, deployed_sha=deployed)
        event_store = MagicMock()
        harness.event_store = event_store
        return harness, event_store

    def test_flagged_sha_warns_and_emits_the_event(self, mock_orch_config, caplog):
        from orchestrator.event_store import EventType

        deployed = DeployedSha(sha='f' * 40, indeterminate=True, dirty_wire_path=True)
        harness, event_store = self._harness(mock_orch_config, deployed)

        with caplog.at_level('WARNING', logger='orchestrator.harness'):
            harness._announce_deployed_sha()

        event_store.emit.assert_called_once_with(
            EventType.deployed_sha_indeterminate, data=deployed.to_wire(),
        )
        assert any('f' * 40 in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize('deployed', [
        DeployedSha(sha='f' * 40, indeterminate=False, dirty_wire_path=False),
        None,
    ])
    def test_unflagged_or_unknown_sha_emits_nothing(self, mock_orch_config, deployed):
        harness, event_store = self._harness(mock_orch_config, deployed)
        harness._announce_deployed_sha()
        event_store.emit.assert_not_called()
