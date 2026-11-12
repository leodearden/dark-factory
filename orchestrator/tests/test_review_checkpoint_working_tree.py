"""A review's Phase 1 is qualified by the working-tree state of the checkout it measured.

Drives the public ``ReviewCheckpoint.run_focused`` against a real git repo,
with Phase 1 replaced by a fake that walks the tree like a linter, so a dirty
file on disk is what reddens Phase 1.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    git_env_with_ceiling,
    pydantic_spec,
    stamp_stock_routing_config,
)
from _recording_event_store import _RecordingEventStore
from df_pytest_isolation import git_redirect_env
from shared.cli_invoke import AgentResult, AllAccountsCappedException

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.verify import VerifyResult

# Compelled by test_whole_tree_scan_timeout_guard.py, whose detector matches the
# fake linter's rglob('*.py'); that sweep covers only a few-file scratch repo.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_LINT_MARKER = 'LINT_BAD'
_BARE_FAIL_SUMMARY = 'Failures: lint issues'


def _git(cwd: Path, *args: str) -> None:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    subprocess.run(['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)


@pytest.fixture
def review_repo() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix='df-review-tree-') as scratch:
        repo = seed_repo(
            Path(scratch) / 'repo',
            RepoSeed(files=(('scripts/ok.py', 'x = 1\n'),), message='init'),
        )
        assert '/tmp/pytest' not in str(repo), (
            f'{repo} contains "/tmp/pytest", which ReviewCheckpoint._run_review refuses '
            'as a project_root; this fixture must live outside pytest basetemp'
        )
        yield repo


@pytest.fixture
def plain_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix='df-review-plain-') as scratch:
        plain = Path(scratch) / 'plain'
        (plain / 'scripts').mkdir(parents=True)
        yield plain


def _printed_as(rel: str, module_dirs: Iterable[str]) -> str:
    return next(
        (rel.removeprefix(f'{d}/') for d in module_dirs if rel.startswith(f'{d}/')), rel,
    )


async def _tree_walking_lint(
    project_root: Path, config: OrchestratorConfig, **_admission: object,
) -> VerifyResult:
    offenders = sorted(
        _printed_as(path.relative_to(project_root).as_posix(), config.module_configs_or_empty)
        for path in project_root.rglob('*.py')
        if '.git' not in path.relative_to(project_root).parts
        and _LINT_MARKER in path.read_text()
    )
    if not offenders:
        return VerifyResult(
            passed=True, summary='OK', test_output='', lint_output='', type_output='',
        )
    lint_output = ''.join(
        f'F821 Undefined name `{_LINT_MARKER}`\n  --> {rel}:1:1\n' for rel in offenders
    ) + f'Found {len(offenders)} errors.\n'
    return VerifyResult(
        passed=False, summary=_BARE_FAIL_SUMMARY,
        test_output='', lint_output=lint_output, type_output='',
    )


def _make_checkpoint(
    project_root: Path, tmp_path: Path, *, module_dirs: Iterable[str] = (),
) -> ReviewCheckpoint:
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = project_root
    config.models.deep_reviewer = 'opus'
    config.max_turns.deep_reviewer = 50
    config.budgets.deep_reviewer = 10.0
    config.effort.deep_reviewer = 'max'
    config.backends.deep_reviewer = 'claude'
    config.fused_memory.project_id = 'test-project'
    config.escalation.host = 'localhost'
    config.escalation.port = 9999
    config.module_configs_or_empty = {d: ModuleConfig(prefix=d) for d in module_dirs}
    config.review.briefing_path = 'df-6687-no-such-briefing.yaml'
    config.review.reports_dir = str(tmp_path / 'reports')
    stamp_stock_routing_config(config)

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}

    cp = ReviewCheckpoint(config, mcp=mcp, usage_gate=None)
    cp.event_store = cast(EventStore, _RecordingEventStore())
    return cp


@pytest.fixture
def reviewer(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    reviewer = AsyncMock(return_value=AgentResult(
        success=True, output='', subtype='success',
        structured_output={'findings': [], 'summary': 'ok'},
    ))
    monkeypatch.setattr('orchestrator.review_checkpoint.invoke_with_cap_retry', reviewer)
    monkeypatch.setattr(
        'orchestrator.review_checkpoint.run_full_verification', _tree_walking_lint,
    )
    return reviewer


def _prompt(reviewer: AsyncMock) -> str:
    assert reviewer.await_args is not None, 'the deep reviewer was never invoked'
    return reviewer.await_args.kwargs['prompt']


def _summary_line(prompt: str) -> str:
    return next(line for line in prompt.splitlines() if line.startswith('**Summary:**'))


def _saved_report(tmp_path: Path, review_id: str) -> dict:
    return json.loads((tmp_path / 'reports' / f'review-{review_id}.json').read_text())


@pytest.mark.asyncio
class TestReviewPhase1WorkingTree:
    async def test_untracked_lint_failure_is_labelled_everywhere_it_is_reported(
        self, review_repo: Path, reviewer: AsyncMock, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (review_repo / 'scripts' / 'bad.py').write_text(f'{_LINT_MARKER}\n')
        caplog.set_level(logging.INFO, logger='orchestrator.review_checkpoint')

        report = await _make_checkpoint(review_repo, tmp_path).run_focused()

        prompt = _prompt(reviewer)
        summary_line = _summary_line(prompt)
        assert summary_line != f'**Summary:** {_BARE_FAIL_SUMMARY}'
        assert 'scripts/bad.py' in summary_line
        assert 'untracked' in summary_line
        assert (
            prompt.index('## Phase 1 Results')
            < prompt.index('### Working-tree state')
            < prompt.index('## Review Briefing')
        )
        saved = _saved_report(tmp_path, report.review_id)
        assert saved['phase1_summary'] != _BARE_FAIL_SUMMARY
        assert saved['phase1_working_tree']['named_in_failure'] == [
            {'path': 'scripts/bad.py', 'change': 'untracked'},
        ]
        assert report.phase1_working_tree is not None
        assert [n.path for n in report.phase1_working_tree.named_in_failure] == [
            'scripts/bad.py',
        ]
        assert any(
            'Phase 1 complete' in record.getMessage()
            and 'scripts/bad.py' in record.getMessage()
            for record in caplog.records
        )

    async def test_committed_lint_failure_stays_unqualified(
        self, review_repo: Path, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        (review_repo / 'scripts' / 'bad.py').write_text(f'{_LINT_MARKER}\n')
        _git(review_repo, 'add', 'scripts/bad.py')
        _git(review_repo, 'commit', '-q', '-m', 'commit the lint error')

        report = await _make_checkpoint(review_repo, tmp_path).run_focused()

        prompt = _prompt(reviewer)
        assert _summary_line(prompt) == f'**Summary:** {_BARE_FAIL_SUMMARY}'
        assert '### Working-tree state' not in prompt
        saved = _saved_report(tmp_path, report.review_id)
        assert saved['phase1_working_tree']['named_in_failure'] == []

    async def test_committed_failure_ending_in_a_dirty_path_names_nothing(
        self, review_repo: Path, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        committed = review_repo / 'tools' / 'scripts' / 'bad.py'
        committed.parent.mkdir(parents=True)
        committed.write_text(f'{_LINT_MARKER}\n')
        _git(review_repo, 'add', 'tools/scripts/bad.py')
        _git(review_repo, 'commit', '-q', '-m', 'commit the lint error')
        (review_repo / 'scripts' / 'bad.py').write_text('y = 1\n')

        report = await _make_checkpoint(review_repo, tmp_path).run_focused()

        saved = _saved_report(tmp_path, report.review_id)
        assert saved['phase1_working_tree']['named_in_failure'] == []
        assert report.phase1_working_tree is not None
        assert report.phase1_working_tree.named_in_failure == ()
        assert 'scripts/bad.py' not in _summary_line(_prompt(reviewer))

    async def test_uncommitted_edit_named_in_failure_is_labelled_uncommitted(
        self, review_repo: Path, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        (review_repo / 'scripts' / 'ok.py').write_text(f'{_LINT_MARKER}\n')

        report = await _make_checkpoint(review_repo, tmp_path).run_focused()

        saved = _saved_report(tmp_path, report.review_id)
        assert saved['phase1_working_tree']['named_in_failure'] == [
            {'path': 'scripts/ok.py', 'change': 'uncommitted'},
        ]

    async def test_pass_with_uncommitted_edit_is_qualified(
        self, review_repo: Path, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        (review_repo / 'scripts' / 'ok.py').write_text('x = 2\n')

        await _make_checkpoint(review_repo, tmp_path).run_focused()

        prompt = _prompt(reviewer)
        assert _summary_line(prompt) != '**Summary:** OK'
        assert '### Working-tree state' in prompt

    async def test_capped_review_still_carries_the_working_tree(
        self, review_repo: Path, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        reviewer.side_effect = AllAccountsCappedException(
            retries=1, elapsed_secs=1.0, label='x',
        )
        (review_repo / 'scripts' / 'bad.py').write_text(f'{_LINT_MARKER}\n')

        report = await _make_checkpoint(review_repo, tmp_path).run_focused()

        assert report.phase1_working_tree is not None
        assert [n.path for n in report.phase1_working_tree.named_in_failure] == [
            'scripts/bad.py',
        ]

    async def test_module_relative_failure_is_attributed_through_the_module_dir(
        self, review_repo: Path, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        bad = review_repo / 'pkg' / 'src' / 'bad.py'
        bad.parent.mkdir(parents=True)
        bad.write_text(f'{_LINT_MARKER}\n')

        report = await _make_checkpoint(review_repo, tmp_path, module_dirs=['pkg']).run_focused()

        assert 'pkg/src/bad.py' not in report.phase1.lint_output
        saved = _saved_report(tmp_path, report.review_id)
        assert saved['phase1_working_tree']['named_in_failure'] == [
            {'path': 'pkg/src/bad.py', 'change': 'untracked'},
        ]

    async def test_checkout_git_cannot_describe_leaves_phase1_unqualified(
        self, plain_dir: Path, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        (plain_dir / 'scripts' / 'bad.py').write_text(f'{_LINT_MARKER}\n')

        report = await _make_checkpoint(plain_dir, tmp_path).run_focused()

        prompt = _prompt(reviewer)
        assert _summary_line(prompt) == f'**Summary:** {_BARE_FAIL_SUMMARY}'
        assert '### Working-tree state' not in prompt
        assert report.phase1_working_tree is None
        assert 'phase1_working_tree' not in _saved_report(tmp_path, report.review_id)
