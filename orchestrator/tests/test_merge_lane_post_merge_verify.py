"""The contract of ``merge_lane/post_merge_verify.py`` (PRD κ, task 5040).

``run_post_merge_verify`` verifies one merge and answers whether it may land:
``None`` to land, a blocked ``MergeOutcome`` (with the merge worktree cleaned)
to withhold it.  It is driven here on a real git merge worktree, through its
public signature alone, with the verify scripted by an injected
``FakeVerifier`` and a remote host by a fake remote runner.  Every expected
value is today's behaviour; none is new.
"""
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import yaml
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    RecordingEscalations,
    VerifyScript,
    delegates_to,
    fails,
    passes,
    raises,
)
from _narrow_gate_fixtures import REGISTERED_MODULE, SOURCE

from orchestrator.config import GitConfig, OrchestratorConfig, load_config
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.post_merge_verify import (
    TRIVIAL_PASS_MAIN_RED_REASON_PREFIX,
    run_post_merge_verify,
)
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest, QueuedBranch
from orchestrator.narrow_gate_class import MergeGateResolution, MissReason
from orchestrator.narrow_gate_plan import NarrowGateDecision
from orchestrator.verify import VerifyResult, cached_main_baseline_failing_ids, seed_main_baseline
from orchestrator.verify_cmd import render
from orchestrator.verify_runner import (
    FLOCK_CONTENTION_CATEGORY,
    RunnerUnavailable,
    make_flock_contention_result,
)

TASK_ID = 'pmv-1'


def _git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


class _Scene:
    """A real repo with one task branch merged into a fresh merge worktree."""

    def __init__(self, git_ops: GitOps, req: MergeRequest, merge_wt: Path, merge_sha: str) -> None:
        self.git_ops = git_ops
        self.req = req
        self.merge_wt = merge_wt
        self.merge_sha = merge_sha

    async def verify(self, verifier: FakeVerifier, **options: Any) -> MergeOutcome | None:
        kwargs: dict[str, Any] = {
            'timeouts': {}, 'enospc_retries': {}, 'max_timeouts': 3, 'max_enospc': 1,
        }
        kwargs.update(options)
        return await run_post_merge_verify(
            self.git_ops, self.req, self.merge_wt,
            merge_sha=self.merge_sha, verifier=verifier, **kwargs,
        )


async def _merged_scene(repo: Path, config: OrchestratorConfig, files: dict[str, str]) -> _Scene:
    """*files* committed on task/<TASK_ID> and merged into a fresh merge worktree."""
    git_ops = GitOps(_git_config(), repo)
    worktree = (await git_ops.create_worktree(TASK_ID)).path
    for path, text in files.items():
        (worktree / path).parent.mkdir(parents=True, exist_ok=True)
        (worktree / path).write_text(text)
    await git_ops.commit(worktree, 'Add feature')
    merged = await git_ops.merge_to_main(worktree, f'task/{TASK_ID}')
    assert merged.success and merged.merge_worktree is not None, merged
    assert merged.merge_commit is not None
    req = MergeRequest(
        task_id=TASK_ID,
        branch=QueuedBranch.parse(f'task/{TASK_ID}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )
    return _Scene(git_ops, req, merged.merge_worktree, merged.merge_commit)


@pytest_asyncio.fixture
async def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo')
    config = OrchestratorConfig(
        project_root=repo, git=_git_config(), escalate_preexisting_main_break=False,
        verify_cross_check_remote_green=True,
    )
    return await _merged_scene(repo, config, {'feature.txt': 'feature\n'})


class _ScriptedRemote(FakeRemoteRunner):
    """A remote verify host that answers *result*."""

    def __init__(self, result: VerifyResult, name: str = 'laptop') -> None:
        super().__init__(name)
        self.result = result

    async def run_merge_verify(self, merge_sha: str, spec: Any) -> VerifyResult:
        self.calls.append(merge_sha)
        return self.result


class _ContentionEscalations(RecordingEscalations):
    """Answers the flock alarm's open-L2 dedup probe: none open."""

    def get_by_task(self, task_id: str, **filters: Any) -> list[Any]:
        return []


def _enospc() -> VerifyScript:
    return VerifyScript(result=VerifyResult(
        passed=False, test_output='E   OSError: [Errno 28] No space left on device',
        lint_output='', type_output='', summary='disk full',
    ))


@pytest.mark.asyncio
class TestRunPostMergeVerify:
    async def test_the_verifier_is_required(self, scene: _Scene) -> None:
        without_a_verifier: dict[str, Any] = {
            'timeouts': {}, 'enospc_retries': {}, 'max_timeouts': 3, 'max_enospc': 1,
        }
        with pytest.raises(TypeError, match='verifier'):
            await run_post_merge_verify(
                scene.git_ops, scene.req, scene.merge_wt, **without_a_verifier,
            )

    async def test_a_pass_lands_and_keeps_the_worktree(self, scene: _Scene) -> None:
        verifier = FakeVerifier(passes())

        assert await scene.verify(verifier) is None
        assert verifier.verified == [TASK_ID]
        assert scene.merge_wt.exists()

    async def test_a_categorised_failure_blocks_with_its_frozen_prefix(self, scene: _Scene) -> None:
        verifier = FakeVerifier(fails(category='test_failure', summary='2 failed'))

        outcome = await scene.verify(verifier)

        assert outcome is not None and outcome.status == 'blocked'
        assert outcome.reason.startswith(
            'Post-merge verification failed: 2 failed [category: test_failure]',
        ), outcome.reason
        assert outcome.failure_category == 'test_failure'
        assert not scene.merge_wt.exists()

    async def test_the_disk_guard_short_circuits_without_a_verify(self, scene: _Scene) -> None:
        verifier = FakeVerifier(disk_reason='Transient infrastructure failure (disk pressure): low')

        outcome = await scene.verify(verifier)

        assert outcome is not None
        assert outcome.reason == 'Transient infrastructure failure (disk pressure): low'
        assert outcome.verify_skipped is True
        assert verifier.verified == []
        assert not scene.merge_wt.exists()

    async def test_an_enospc_red_retries_once_inside_its_budget(self, scene: _Scene) -> None:
        verifier = FakeVerifier(sequence=[_enospc(), passes()])
        enospc_retries: dict[str, int] = {}

        outcome = await scene.verify(verifier, enospc_retries=enospc_retries)

        assert outcome is None
        assert verifier.verified == [TASK_ID, TASK_ID]
        assert enospc_retries == {TASK_ID: 1}

    async def test_a_spent_enospc_budget_holds_as_transient_infra(self, scene: _Scene) -> None:
        verifier = FakeVerifier(_enospc())

        outcome = await scene.verify(verifier, enospc_retries={TASK_ID: 1})

        assert outcome is not None
        assert outcome.reason.startswith('Transient infrastructure failure (disk pressure): ')
        assert verifier.verified == [TASK_ID]

    async def test_a_flock_contention_red_keeps_its_own_arm(self, scene: _Scene) -> None:
        remote = _ScriptedRemote(make_flock_contention_result('laptop', 111, 222))
        escalations = _ContentionEscalations()

        outcome = await scene.verify(
            FakeVerifier(), runner=remote, escalation_queue=escalations,
        )

        assert outcome is not None
        assert outcome.reason.startswith('Post-merge verification blocked: ')
        assert outcome.reason.endswith(f'[category: {FLOCK_CONTENTION_CATEGORY}]')
        assert outcome.failure_category == FLOCK_CONTENTION_CATEGORY
        [alarm] = escalations.filed
        assert alarm.category == 'verify_worktree_contention'
        assert alarm.level == 2


@pytest.mark.asyncio
class TestCrossCheckOfARemoteGreen:
    async def test_a_local_fail_withholds_the_land_halts_and_quarantines(
        self, scene: _Scene,
    ) -> None:
        quarantined: list[str] = []
        halts: list[str] = []
        escalations = RecordingEscalations()

        outcome = await scene.verify(
            FakeVerifier(fails(category='test_failure', summary='red on the anchor')),
            runner=FakeRemoteRunner('laptop'),
            quarantine_divergent=quarantined.append,
            halt_hook=halts.append,
            escalation_queue=escalations,
        )

        assert outcome is not None and outcome.status == 'blocked'
        assert 'red on the anchor' in outcome.reason
        assert quarantined == ['laptop']
        assert len(halts) == 1 and 'suspected red main' in halts[0]
        [divergence] = escalations.filed
        assert divergence.category == 'verify_cross_check_mismatch'
        assert divergence.level == 2

    async def test_an_unavailable_anchor_is_inconclusive_and_trusts_the_remote(
        self, scene: _Scene,
    ) -> None:
        quarantined: list[str] = []
        halts: list[str] = []

        outcome = await scene.verify(
            FakeVerifier(raises(RunnerUnavailable('anchor down'))),
            runner=FakeRemoteRunner('laptop'),
            quarantine_divergent=quarantined.append,
            halt_hook=halts.append,
        )

        assert outcome is None
        assert quarantined == []
        assert halts == []

    async def test_an_agreeing_anchor_lands(self, scene: _Scene) -> None:
        verifier = FakeVerifier(passes())
        remote = FakeRemoteRunner('laptop')

        assert await scene.verify(verifier, runner=remote) is None
        assert remote.calls == [scene.merge_sha]
        assert verifier.verified == [TASK_ID]


class _RecordingEventStore:
    run_id = 'pmv-run'

    def __init__(self) -> None:
        self.events: list[tuple[Any, dict[str, Any]]] = []

    def emit(self, event_type: Any, *, data: dict[str, Any] | None = None, **_kw: Any) -> int | None:
        self.events.append((event_type, dict(data or {})))
        return None

    def of(self, event_type: EventType) -> list[dict[str, Any]]:
        return [data for kind, data in self.events if kind == event_type]


class _ScopedRecorder:
    """A scoped verify that records the narrow decision it was handed and answers as the executor would."""

    def __init__(self) -> None:
        self.decisions: list[NarrowGateDecision | None] = []

    async def __call__(self, *_args: Any, **options: Any) -> VerifyResult:
        decision = options.get('narrow_gate')
        self.decisions.append(decision)
        return VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
            failing_test_ids=[], skipped_test_ids=[],
            narrow_gate=None if decision is None else decision.resolution,
        )


class _NarrowLane:
    """A repo whose main registers one module, with the narrow merge gate configured for its docs."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.repo = seed_repo(tmp_path / 'repo', RepoSeed(
            files=(('README.md', '# Test\n'), ('docs/a.md', 'a\n'), *REGISTERED_MODULE.items()),
            message='Initial commit',
        ))
        self.events = _RecordingEventStore()
        self.recorder = _ScopedRecorder()
        self.results: list[VerifyResult] = []

    def config(self, *, enabled: bool) -> OrchestratorConfig:
        path = self.tmp_path / 'dark-factory-orchestrator.yaml'
        path.write_text(yaml.safe_dump({
            'project_root': str(self.repo),
            'git': _git_config().model_dump(include={
                'main_branch', 'branch_prefix', 'remote', 'worktree_dir', 'push_after_advance',
            }),
            'escalate_preexisting_main_break': False,
            'verify_cross_check_remote_green': True,
            'narrow_merge_gate': {'enabled': enabled, 'docs_paths': ['docs/**']},
        }))
        return load_config(path)

    async def merge(self, files: dict[str, str], *, enabled: bool = True) -> tuple[_Scene, str]:
        """The merged scene and the main sha it was built on."""
        base = await GitOps(_git_config(), self.repo).get_main_sha()
        return await _merged_scene(self.repo, self.config(enabled=enabled), files), base

    async def verify(self, scene: _Scene, **options: Any) -> MergeOutcome | None:
        return await scene.verify(
            FakeVerifier(delegates_to(self.recorder)), event_store=self.events,
            on_result=self.results.append, **options,
        )

    def resolutions(self) -> list[dict[str, Any]]:
        return self.events.of(EventType.merge_gate_resolution)


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _NarrowLane:
    built = _NarrowLane(tmp_path)
    # The autouse config isolation pins ORCH_PROJECT_ROOT to tmp_path, which outranks the yaml's
    # project_root; module discovery must walk the repo.
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(built.repo))
    return built


_DOCS = {'docs/x.md': 'x\n'}


def _rendered(decision: NarrowGateDecision | None) -> list[str]:
    assert decision is not None and decision.plan is not None
    return [render(run.cmd) for run in decision.plan.runs if run.cmd is not None]


@pytest.mark.asyncio
class TestNarrowMergeGate:
    """Task 5411: a docs or comments merge verifies on the local host under its narrow command set."""

    async def test_a_docs_merge_lands_on_a_local_narrow_verify(self, lane: _NarrowLane) -> None:
        scene, base = await lane.merge(_DOCS)

        assert await lane.verify(scene, narrow_gate_base=base) is None
        [decision] = lane.recorder.decisions
        assert decision is not None and decision.resolution is MergeGateResolution.DOCS

    async def test_a_narrow_verify_never_reaches_the_remote_nor_its_cross_check(self, lane: _NarrowLane) -> None:
        scene, base = await lane.merge(_DOCS)
        remote = _ScriptedRemote(passes().result)

        assert await lane.verify(scene, narrow_gate_base=base, runner=remote) is None
        assert remote.calls == []
        [decision] = lane.recorder.decisions
        assert decision is not None and decision.resolution is MergeGateResolution.DOCS

    async def test_the_resolution_is_recorded_with_what_ran(self, lane: _NarrowLane) -> None:
        scene, base = await lane.merge(_DOCS)

        await lane.verify(scene, narrow_gate_base=base)

        [event] = lane.resolutions()
        [decision] = lane.recorder.decisions
        assert _rendered(decision)
        assert {key: event[key] for key in (
            'resolution', 'changed_paths', 'commands', 'passed', 'base_sha', 'merge_sha',
        )} == {
            'resolution': 'docs_gate', 'changed_paths': ['docs/x.md'], 'commands': _rendered(decision),
            'passed': True, 'base_sha': base, 'merge_sha': scene.merge_sha,
        }

    async def test_a_narrow_pass_neither_feeds_the_shadow_nor_seeds_the_baseline(self, lane: _NarrowLane) -> None:
        scene, base = await lane.merge(_DOCS)

        assert await lane.verify(scene, narrow_gate_base=base) is None
        assert lane.results == []
        assert cached_main_baseline_failing_ids(scene.merge_sha) is None

    async def test_a_narrow_pass_over_a_known_red_main_is_withheld(self, lane: _NarrowLane) -> None:
        scene, base = await lane.merge(_DOCS)
        seed_main_baseline(await scene.git_ops.get_main_sha(), frozenset({'pkg/tests/test_x.py::test_red'}))

        outcome = await lane.verify(scene, narrow_gate_base=base)

        assert outcome is not None and outcome.status == 'blocked'
        assert outcome.reason.startswith(TRIVIAL_PASS_MAIN_RED_REASON_PREFIX), outcome.reason
        assert 'docs_gate' in outcome.reason

    async def test_a_code_merge_takes_the_full_gate_and_records_why(self, lane: _NarrowLane) -> None:
        scene, base = await lane.merge({SOURCE: '"""Module."""\nx = 2\n'})
        remote = _ScriptedRemote(passes().result)

        assert await lane.verify(scene, narrow_gate_base=base, runner=remote) is None
        assert remote.calls == [scene.merge_sha]
        assert lane.recorder.decisions == [None]
        [event] = lane.resolutions()
        assert (event['resolution'], event['miss_reason'], event['commands']) == (
            'full_gate', MissReason.CODE_CHANGED, None,
        )

    async def test_without_a_base_the_lane_is_unchanged(self, lane: _NarrowLane) -> None:
        scene, _base = await lane.merge(_DOCS)

        assert await lane.verify(scene) is None
        assert lane.recorder.decisions == [None]
        assert lane.resolutions() == []
        assert len(lane.results) == 1

    async def test_a_disabled_gate_leaves_the_lane_unchanged(self, lane: _NarrowLane) -> None:
        scene, base = await lane.merge(_DOCS, enabled=False)

        assert await lane.verify(scene, narrow_gate_base=base) is None
        assert lane.recorder.decisions == [None]
        assert lane.resolutions() == []


_HOMES = {
    'orchestrator.merge_lane.post_merge_verify': (
        'run_post_merge_verify', 'merge_boundary_module_configs',
        'placeholder_local_runner', 'TRIVIAL_PASS_MAIN_RED_REASON_PREFIX',
    ),
    'orchestrator.merge_lane.verify_failure_outcome': (
        'red_verify_outcome', 'RedVerifyContext', 'DryRunInvestigationHandles',
        'spawn_merge_verify_dry_run', 'verify_hit_enospc', 'transient_infra_reason',
        'TRANSIENT_INFRA_REASON_PREFIX', 'MAIN_HEALTH_PROBE_PENDING_NOTE',
    ),
    'orchestrator.merge_lane.verify_cross_check': ('CrossCheck', 'cross_check_remote_green'),
}
_GONE_FROM_WORKER = (
    '_run_post_merge_verify', '_verify_hit_enospc', '_DryRunInvestigationHandles',
    '_spawn_merge_verify_dry_run', '_merge_boundary_module_configs',
    '_placeholder_local_runner', '_local_merge_verify_runner', '_record_flake_observation',
    '_classify_disposition_for_outcome', '_render_skew_surfaces',
    '_alarm_verify_worktree_contention', '_CROSS_CHECK_SENTINEL',
    'MAIN_HEALTH_PROBE_PENDING_NOTE', 'TRIVIAL_PASS_MAIN_RED_REASON_PREFIX',
    'TRANSIENT_INFRA_REASON_PREFIX', '_transient_infra_reason',
)


@pytest.mark.parametrize(
    ('module', 'name'), [(m, n) for m, names in _HOMES.items() for n in names],
)
def test_each_post_merge_verify_name_lives_in_its_concern_module(module: str, name: str) -> None:
    assert hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize('name', _GONE_FROM_WORKER)
def test_the_worker_no_longer_defines_the_moved_name(name: str) -> None:
    assert not hasattr(importlib.import_module('orchestrator.merge_lane.worker'), name)
