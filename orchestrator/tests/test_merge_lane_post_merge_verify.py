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
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    RecordingEscalations,
    VerifyScript,
    fails,
    passes,
    raises,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.post_merge_verify import run_post_merge_verify
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest, QueuedBranch
from orchestrator.verify import VerifyResult
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


@pytest_asyncio.fixture
async def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo')
    git_ops = GitOps(_git_config(), repo)
    config = OrchestratorConfig(
        project_root=repo, git=_git_config(), escalate_preexisting_main_break=False,
        verify_cross_check_remote_green=True,
    )
    worktree = (await git_ops.create_worktree(TASK_ID)).path
    (worktree / 'feature.txt').write_text('feature\n')
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
