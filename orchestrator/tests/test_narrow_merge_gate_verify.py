"""The merge-role narrow gate decision: verify.narrow_merge_gate_decision over a real merge worktree (task 5411)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from _landing_git_scenarios import LandingRepo

from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.event_store import EventType
from orchestrator.narrow_gate_class import GateMiss, MergeGateResolution, MissReason
from orchestrator.narrow_gate_plan import NarrowGateDecision
from orchestrator.verify import VerifyResult, narrow_merge_gate_decision, run_scoped_verification
from orchestrator.verify_cmd import VerifyCmd, parse_config_command
from orchestrator.verify_plan import PlannedRun, ScopeKind, VerifyPlan

_GUARD = 'pkg/tests/test_guard.py'
_SOURCE = 'pkg/src/pkg/mod.py'
_GUARD_SOURCE = (
    'import pytest\n'
    'from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
    'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
)
_MODULE_YAML = yaml.safe_dump({
    'test_command': 'cd pkg && uv run pytest',
    'lint_command': 'cd pkg && uv run ruff check',
    'type_check_command': 'cd pkg && uv run pyright',
})
_GUARD_SCRIPT = '#!/bin/sh\nexit 0\n'


class _Scene:
    """A repo whose main carries one registered module, and merges built in a linked worktree."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        root = tmp_path / 'repo'
        root.mkdir()
        self.repo = LandingRepo.init(root)
        self.repo.commit('base', {
            'README.md': 'hello\n',
            'docs/a.md': 'a\n',
            'pkg/orchestrator.yaml': _MODULE_YAML,
            _SOURCE: '"""Module."""\nx = 1\n',
            _GUARD: _GUARD_SOURCE,
            'pkg/tests/test_plain.py': 'def test_plain():\n    pass\n',
        })
        self._merges = 0

    def on_main(self, files: dict[str, str]) -> None:
        self.repo.commit('main moves', files)

    def merge(self, files: dict[str, str]) -> tuple[Path, str, str]:
        """Branch from main, commit *files*, and merge it --no-ff in a new linked worktree."""
        self._merges += 1
        branch = f'task/{self._merges}'
        base = self.repo.sha('main')
        self.repo.git('checkout', '-q', '-b', branch, 'main')
        self.repo.commit(branch, files)
        self.repo.git('checkout', '-q', 'main')
        worktree = self.tmp_path / f'wt{self._merges}'
        self.repo.git('worktree', 'add', '-q', '--detach', str(worktree), 'main')
        self.repo.git('-C', str(worktree), 'merge', '--no-ff', '-q', '-m', f'merge {branch}', branch)
        return worktree, base, self.repo.git('-C', str(worktree), 'rev-parse', 'HEAD')

    def config(self, *, enabled: bool = True, full_gate_globs: list[str] | None = None) -> OrchestratorConfig:
        path = self.tmp_path / 'dark-factory-orchestrator.yaml'
        path.write_text(yaml.safe_dump({
            'project_root': str(self.repo.root),
            'git': {'merge_config_only_full_gate_globs': full_gate_globs or []},
            'narrow_merge_gate': {'enabled': enabled, 'docs_paths': ['docs/**', '*.md']},
        }))
        return load_config(path)


@pytest.fixture
def scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Scene:
    built = _Scene(tmp_path)
    # The autouse config isolation pins ORCH_PROJECT_ROOT to tmp_path, which outranks the yaml's
    # project_root; module discovery must walk the repo, not the merge worktrees beside it.
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(built.repo.root))
    return built


def _full(reason: MissReason, path: str | None = None) -> tuple[MergeGateResolution, GateMiss, None]:
    return MergeGateResolution.FULL, GateMiss(reason, path), None


def _outcome(decision: NarrowGateDecision) -> tuple[MergeGateResolution, GateMiss | None, object]:
    return decision.resolution, decision.miss, decision.plan


def _run(decision: NarrowGateDecision, tool: str) -> PlannedRun:
    assert decision.plan is not None
    [run] = [r for r in decision.plan.runs if r.module_prefix == 'pkg' and r.reason.startswith(f'{tool}:')]
    return run


async def _decide(
    scene: _Scene, files: dict[str, str], *, enabled: bool = True, full_gate_globs: list[str] | None = None,
) -> NarrowGateDecision:
    worktree, base, merge_sha = scene.merge(files)
    config = scene.config(enabled=enabled, full_gate_globs=full_gate_globs)
    return await narrow_merge_gate_decision(worktree, config, base_sha=base, merge_sha=merge_sha)


@pytest.mark.asyncio
class TestNarrowMergeGateDecision:

    async def test_a_disabled_gate_is_the_full_gate(self, scene: _Scene) -> None:
        decision = await _decide(scene, {'docs/x.md': 'x\n'}, enabled=False)

        assert _outcome(decision) == _full(MissReason.DISABLED)

    async def test_a_docs_merge_takes_the_docs_gate_with_the_guard_family(self, scene: _Scene) -> None:
        decision = await _decide(scene, {'docs/x.md': 'x\n'})

        assert (decision.resolution, decision.miss) == (MergeGateResolution.DOCS, None)
        assert decision.changed_paths == ('docs/x.md',)
        pytest_run = _run(decision, 'pytest')
        assert (pytest_run.scope_kind, pytest_run.scoped_targets) == (ScopeKind.FILE_SCOPED, (_GUARD,))
        assert _run(decision, 'lint').scope_kind is ScopeKind.SKIPPED

    async def test_a_comment_only_merge_takes_the_comments_gate(self, scene: _Scene) -> None:
        decision = await _decide(scene, {_SOURCE: '"""Reworded."""\nx = 1  # why\n'})

        assert decision.resolution is MergeGateResolution.COMMENTS
        lint = _run(decision, 'lint')
        assert (lint.scope_kind, lint.scoped_targets) == (ScopeKind.FILE_SCOPED, (_SOURCE,))

    async def test_a_changed_path_missing_from_disk_is_absent_evidence(self, scene: _Scene) -> None:
        worktree, base, merge_sha = scene.merge({'docs/x.md': 'x\n'})
        (worktree / 'docs' / 'x.md').unlink()

        decision = await narrow_merge_gate_decision(worktree, scene.config(), base_sha=base, merge_sha=merge_sha)

        assert _outcome(decision) == _full(MissReason.EVIDENCE_ABSENT, 'docs/x.md')

    async def test_a_guard_test_missing_from_disk_is_absent_evidence(self, scene: _Scene) -> None:
        worktree, base, merge_sha = scene.merge({'docs/x.md': 'x\n'})
        (worktree / _GUARD).unlink()

        decision = await narrow_merge_gate_decision(worktree, scene.config(), base_sha=base, merge_sha=merge_sha)

        assert _outcome(decision) == _full(MissReason.EVIDENCE_ABSENT, _GUARD)

    async def test_a_worktree_not_at_the_merge_is_absent_evidence(self, scene: _Scene) -> None:
        worktree, base, _merge_sha = scene.merge({'docs/x.md': 'x\n'})

        decision = await narrow_merge_gate_decision(worktree, scene.config(), base_sha=base, merge_sha=base)

        assert _outcome(decision) == _full(MissReason.EVIDENCE_ABSENT)

    async def test_a_2838_full_gate_glob_forces_the_full_gate(self, scene: _Scene) -> None:
        decision = await _decide(scene, {'docs/x.md': 'x\n'}, full_gate_globs=['docs/*'])

        assert _outcome(decision) == _full(MissReason.FORCED_FULL_GATE)

    async def test_the_verify_pipeline_guard_forces_the_full_gate(self, scene: _Scene) -> None:
        scene.on_main({'scripts/verify-pipeline-guard.sh': _GUARD_SCRIPT})
        scene.repo.git('update-index', '--chmod=+x', 'scripts/verify-pipeline-guard.sh')
        scene.repo.git('commit', '-q', '-m', 'executable guard')
        (scene.repo.root / 'scripts' / 'verify-pipeline-guard.sh').chmod(0o755)

        decision = await _decide(scene, {'docs/x.md': 'x\n'})

        assert _outcome(decision) == _full(MissReason.FORCED_FULL_GATE)

    async def test_a_registry_with_nothing_to_run_is_an_empty_command_set(self, scene: _Scene) -> None:
        scene.on_main({_GUARD: 'def test_unmarked():\n    pass\n'})

        decision = await _decide(scene, {'docs/x.md': 'x\n'})

        assert _outcome(decision) == _full(MissReason.EMPTY_COMMAND_SET)

    async def test_a_code_change_is_the_full_gate(self, scene: _Scene) -> None:
        decision = await _decide(scene, {_SOURCE: '"""Module."""\nx = 2\n'})

        assert _outcome(decision) == _full(MissReason.CODE_CHANGED, _SOURCE)

    async def test_an_unknown_base_is_an_unreadable_delta(self, scene: _Scene) -> None:
        worktree, _base, merge_sha = scene.merge({'docs/x.md': 'x\n'})

        decision = await narrow_merge_gate_decision(worktree, scene.config(), base_sha='0' * 40, merge_sha=merge_sha)

        assert _outcome(decision) == _full(MissReason.DELTA_UNREADABLE)

    async def test_a_vanished_worktree_degrades_to_the_full_gate(self, scene: _Scene) -> None:
        _worktree, base, merge_sha = scene.merge({'docs/x.md': 'x\n'})

        decision = await narrow_merge_gate_decision(
            scene.tmp_path / 'absent', scene.config(), base_sha=base, merge_sha=merge_sha,
        )

        assert decision.resolution is MergeGateResolution.FULL


class _RecordingEventStore:
    def __init__(self) -> None:
        self.events: list[tuple[Any, dict[str, Any]]] = []

    def emit(self, event_type: Any, *, data: dict[str, Any] | None = None, **_kw: Any) -> None:
        self.events.append((event_type, dict(data or {})))

    def of(self, event_type: EventType) -> list[dict[str, Any]]:
        return [data for kind, data in self.events if kind == event_type]


class _Execution:
    """One merge-role run_scoped_verification over a merge, with every command recorded instead of run."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.events = _RecordingEventStore()

    async def _record(self, cmd: str, *_args: Any, **_kwargs: Any) -> tuple[int, str, bool]:
        self.calls.append(cmd)
        return 0, '', False

    async def verify(
        self, worktree: Path, config: OrchestratorConfig, task_files: list[str], **narrow: Any,
    ) -> VerifyResult:
        with patch('orchestrator.verify._run_cmd', side_effect=self._record):
            return await run_scoped_verification(
                worktree, config, list(config.module_configs_or_empty.values()), task_files,
                role='merge', is_merge_verify=True, event_store=self.events,  # type: ignore[arg-type]
                **narrow,
            )

    def escalations(self) -> list[str]:
        return [data['resolution'] for data in self.events.of(EventType.trivial_pass_escalated)]


def _invocation(cmd: VerifyCmd) -> tuple[object, ...]:
    """What a command runs, on what: execution adds only reporting flags (junit, the defect plugin)."""
    return cmd.tool, cmd.uv_project, cmd.cwd_rel, cmd.targets


def _prescribed(plan: VerifyPlan) -> list[tuple[object, ...]]:
    return sorted(_invocation(run.cmd) for run in plan.runs if run.cmd is not None)


def _executed(calls: list[str]) -> list[tuple[object, ...]]:
    return sorted(_invocation(parse_config_command(call)) for call in calls)


async def _merged_decision(scene: _Scene, files: dict[str, str]) -> tuple[Path, OrchestratorConfig, NarrowGateDecision]:
    worktree, base, merge_sha = scene.merge(files)
    config = scene.config()
    return worktree, config, await narrow_merge_gate_decision(worktree, config, base_sha=base, merge_sha=merge_sha)


def _passed(**fields: Any) -> VerifyResult:
    return VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok', **fields)


@pytest.mark.asyncio
class TestNarrowGateExecution:

    @pytest.mark.parametrize(('files', 'resolution'), [
        ({'docs/x.md': 'x\n'}, MergeGateResolution.DOCS),
        ({_SOURCE: '"""Reworded."""\nx = 1  # why\n'}, MergeGateResolution.COMMENTS),
    ])
    async def test_a_narrow_decision_runs_exactly_its_plan(
        self, scene: _Scene, files: dict[str, str], resolution: MergeGateResolution,
    ) -> None:
        worktree, config, decision = await _merged_decision(scene, files)
        assert decision.resolution is resolution and decision.plan is not None
        execution = _Execution()

        result = await execution.verify(worktree, config, list(decision.changed_paths), narrow_gate=decision)

        assert _executed(execution.calls) == _prescribed(decision.plan)
        assert (result.passed, result.trivial, result.narrow_gate) == (True, False, resolution)
        assert result.plan == decision.plan.to_dict()
        assert execution.escalations() == []

    async def test_without_a_decision_a_docs_merge_still_escalates_to_the_full_gate(self, scene: _Scene) -> None:
        worktree, config, decision = await _merged_decision(scene, {'docs/x.md': 'x\n'})
        execution = _Execution()

        result = await execution.verify(worktree, config, list(decision.changed_paths))

        assert execution.escalations() == ['full_gate']
        assert result.narrow_gate is None

    async def test_a_full_decision_is_not_narrowed_on(self, scene: _Scene) -> None:
        worktree, config, _decision = await _merged_decision(scene, {'docs/x.md': 'x\n'})
        execution = _Execution()

        result = await execution.verify(
            worktree, config, ['docs/x.md'], narrow_gate=NarrowGateDecision.full(GateMiss(MissReason.DISABLED)),
        )

        assert execution.escalations() == ['full_gate']
        assert result.narrow_gate is None


class TestWholeTreeEvidence:

    def test_an_ordinary_pass_is_whole_tree_evidence(self) -> None:
        assert _passed().is_whole_tree_evidence is True

    def test_a_trivial_pass_is_not(self) -> None:
        assert _passed(trivial=True).is_whole_tree_evidence is False

    @pytest.mark.parametrize('resolution', [MergeGateResolution.DOCS, MergeGateResolution.COMMENTS])
    def test_a_narrow_gate_pass_is_not(self, resolution: MergeGateResolution) -> None:
        assert _passed(narrow_gate=resolution).is_whole_tree_evidence is False
