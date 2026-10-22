"""The post-rebase re-verify gate records every decision it makes (task 5064).

``gates.py::_reverify_rebased_tree`` decides whether a rebased tree may
advance without a fresh verify. Each decision, skip or fire, is a
``merge_reverify_gate`` row, and a fire's re-verify writes its own
``merge_verify`` row beside it.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
import subprocess
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import FakeVerifier, fails, lane_scene_config, make_lane, merge_through_lane
from _merge_lane_verifier_doubles import ScriptedVerifier
from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeOutcome, MergeRequest, QueuedBranch
from orchestrator.merge_lane.gates import _reverify_rebased_tree, note_queue_verified_main_tip
from orchestrator.merge_lane.reverify_gate_decision import (
    MAX_RECORDED_OVERLAP_PATHS,
    GateClearance,
    ReverifyGateDecision,
    ReverifyGateReason,
    emit_reverify_gate_decision,
)
from orchestrator.merge_lane.worker import _run_post_merge_verify
from orchestrator.verify import VerifyResult

_SHAS = {
    'merge_sha': 'a' * 40,
    'verified_merge_sha': 'b' * 40,
    'rebased_from': 'c' * 40,
    'rebased_onto': 'd' * 40,
}


def _decision(
    reason: ReverifyGateReason,
    *,
    overlap: tuple[str, ...] = (),
    blockers: tuple[str, ...] = (),
    task_id: str = 't-5064',
) -> ReverifyGateDecision:
    return ReverifyGateDecision(
        task_id=task_id, reason=reason,
        merge_sha=_SHAS['merge_sha'],
        verified_merge_sha=_SHAS['verified_merge_sha'],
        rebased_from=_SHAS['rebased_from'],
        rebased_onto=_SHAS['rebased_onto'],
        overlap=overlap, blockers=blockers,
    )


_CONSISTENT_DECISIONS = {
    ReverifyGateReason.same_tree: _decision(ReverifyGateReason.same_tree),
    ReverifyGateReason.disjoint: _decision(ReverifyGateReason.disjoint),
    ReverifyGateReason.disjoint_untrusted: _decision(
        ReverifyGateReason.disjoint_untrusted,
        blockers=('whole-tree merge gate (merge_verify_breadth=full)',),
    ),
    ReverifyGateReason.overlap: _decision(ReverifyGateReason.overlap, overlap=('x.txt',)),
    ReverifyGateReason.overlap_unknown: _decision(ReverifyGateReason.overlap_unknown),
    ReverifyGateReason.same_tree_history_leg: _decision(ReverifyGateReason.same_tree_history_leg),
}

_SKIPS = {ReverifyGateReason.same_tree, ReverifyGateReason.disjoint}


class TestReverifyGateDecisionRecord:
    def test_every_reason_has_a_consistent_decision_under_test(self) -> None:
        assert set(_CONSISTENT_DECISIONS) == set(ReverifyGateReason)

    @pytest.mark.parametrize('reason', list(ReverifyGateReason), ids=str)
    def test_event_data_carries_the_decision(self, reason: ReverifyGateReason) -> None:
        decision = _CONSISTENT_DECISIONS[reason]

        assert decision.event_data() == {
            'reason': str(reason),
            'reverify': reason not in _SKIPS,
            **_SHAS,
            'overlap': list(decision.overlap),
            'overlap_total': len(decision.overlap),
            'blockers': list(decision.blockers),
        }

    def test_recorded_overlap_is_bounded_and_says_so(self) -> None:
        paths = tuple(f'f{n:02d}.py' for n in range(MAX_RECORDED_OVERLAP_PATHS + 2))
        decision = _decision(ReverifyGateReason.overlap, overlap=paths)

        data = decision.event_data()

        assert data['overlap'] == list(paths[:MAX_RECORDED_OVERLAP_PATHS])
        assert data['overlap_total'] == len(paths)

    @pytest.mark.parametrize(
        ('reason', 'overlap', 'blockers'),
        [
            pytest.param(ReverifyGateReason.overlap, (), (), id='overlap-without-paths'),
            pytest.param(ReverifyGateReason.disjoint, ('x.txt',), (), id='disjoint-with-paths'),
            pytest.param(
                ReverifyGateReason.disjoint_untrusted, (), (), id='untrusted-without-blockers',
            ),
            pytest.param(
                ReverifyGateReason.overlap, ('x.txt',), ('unverified drift',),
                id='overlap-with-blockers',
            ),
        ],
    )
    def test_an_inconsistent_decision_is_refused_by_name(
        self, reason: ReverifyGateReason, overlap: tuple[str, ...], blockers: tuple[str, ...],
    ) -> None:
        with pytest.raises(ValueError) as refused:
            _decision(reason, overlap=overlap, blockers=blockers, task_id='t-refused')

        message = str(refused.value)
        for fact in ('t-refused', str(reason), repr(overlap), repr(blockers)):
            assert fact in message, message

    @pytest.mark.parametrize('reason', list(ReverifyGateReason), ids=str)
    def test_only_a_skip_yields_a_clearance(self, reason: ReverifyGateReason) -> None:
        expected = {
            ReverifyGateReason.same_tree: GateClearance.same_tree,
            ReverifyGateReason.disjoint: GateClearance.disjoint,
        }.get(reason)

        assert reason.skip_clearance is expected
        assert reason.reverifies is (expected is None)
        assert _CONSISTENT_DECISIONS[reason].reverifies is reason.reverifies

    def test_the_history_leg_clearance_is_earned_by_running_the_leg_never_by_a_skip(
        self,
    ) -> None:
        assert GateClearance.history_leg.value == 'history_leg'
        assert all(
            reason.skip_clearance is not GateClearance.history_leg
            for reason in ReverifyGateReason
        )

    def test_emit_writes_one_runs_db_row(self, tmp_path: Path) -> None:
        store = EventStore(tmp_path / 'runs.db', 'run-5064')
        decision = _CONSISTENT_DECISIONS[ReverifyGateReason.overlap]

        emit_reverify_gate_decision(store, decision)

        [row] = store.fetch_events_by_type(EventType.merge_reverify_gate)
        assert row['task_id'] == decision.task_id
        assert row['phase'] == 'merge'
        assert row['data'] == decision.event_data()

    def test_emit_without_a_store_is_a_no_op(self) -> None:
        emit_reverify_gate_decision(None, _CONSISTENT_DECISIONS[ReverifyGateReason.disjoint])


# ─── The gate itself, over a real repo ──────────────────────────────────────

_SHARED_LINES = 20
_SEED = RepoSeed(
    files=(
        ('README.md', '# Test\n'),
        ('shared.txt', ''.join(f'shared line {n}\n' for n in range(1, _SHARED_LINES + 1))),
    ),
    message='Initial commit',
)
_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_GATE_LOGGER = 'orchestrator.merge_lane'

Drift = Callable[[Path], str]


def _git(cwd: Path, *args: str) -> str:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    completed = subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _edit_shared_line(root: Path, line_no: int, text: str) -> str:
    path = root / 'shared.txt'
    lines = path.read_text().splitlines(keepends=True)
    lines[line_no - 1] = f'{text}\n'
    path.write_text(''.join(lines))
    _git(root, 'add', '--', 'shared.txt')
    _git(root, 'commit', '-m', f'edit shared line {line_no}')
    return _git(root, 'rev-parse', 'HEAD')


def _drift_on_the_branchs_file(repo: Path) -> str:
    return _edit_shared_line(repo, 18, 'shared line 18, edited on main')


def _drift_elsewhere(repo: Path) -> str:
    (repo / 'unrelated.txt').write_text('drift\n')
    _git(repo, 'add', '--', 'unrelated.txt')
    _git(repo, 'commit', '-m', 'drift: unrelated direct commit')
    return _git(repo, 'rev-parse', 'HEAD')


def _queue_landed(drift: Drift) -> Drift:
    def landed(repo: Path) -> str:
        sha = drift(repo)
        note_queue_verified_main_tip(sha)
        return sha
    return landed


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    store: EventStore

    def config(self, **overrides: Any) -> OrchestratorConfig:
        return lane_scene_config(self.repo, _GIT, **overrides)

    def gate_rows(self) -> list[dict[str, Any]]:
        return [
            {'task_id': row['task_id'], **row['data']}
            for row in self.store.fetch_events_by_type(EventType.merge_reverify_gate)
        ]

    def verify_rows(self, merge_sha: str) -> list[dict[str, Any]]:
        return [
            row['data'] for row in self.store.fetch_events_by_type(EventType.merge_verify)
            if row['data'].get('merge_sha') == merge_sha
        ]


def _scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    return _Scene(
        repo=repo, git_ops=GitOps(_GIT, repo),
        store=EventStore(tmp_path / 'runs.db', 'run-5064'),
    )


def _request(config: OrchestratorConfig, task_id: str, worktree: Path) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, _GIT.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


@dataclasses.dataclass(frozen=True)
class _Rebased:
    """A verified merge commit, rebased onto a main that moved under its verify."""

    request: MergeRequest
    merge_wt: Path
    verified: str
    rebased_from: str
    rebased_onto: str
    rebased_sha: str


@contextlib.asynccontextmanager
async def _rebased_merge(
    scene: _Scene, config: OrchestratorConfig, drift: Drift,
) -> AsyncIterator[_Rebased]:
    task_id = 'gate-5064'
    branch_wt = (await scene.git_ops.create_worktree(task_id)).path
    _edit_shared_line(branch_wt, 2, 'shared line 2, edited by the branch')
    merged = await scene.git_ops.merge_to_main(branch_wt, task_id)
    assert merged.success and merged.merge_worktree is not None, merged
    merge_wt = merged.merge_worktree
    try:
        rebased_from = _git(scene.repo, 'rev-parse', 'main')
        verified = _git(merge_wt, 'rev-parse', 'HEAD')
        rebased_onto = drift(scene.repo)
        _git(merge_wt, 'rebase', rebased_onto)
        yield _Rebased(
            request=_request(config, task_id, branch_wt),
            merge_wt=merge_wt,
            verified=verified,
            rebased_from=rebased_from,
            rebased_onto=rebased_onto,
            rebased_sha=_git(merge_wt, 'rev-parse', 'HEAD'),
        )
    finally:
        if merge_wt.exists():
            await scene.git_ops.cleanup_merge_worktree(merge_wt)


async def _gate(
    scene: _Scene,
    rebased: _Rebased,
    verifier: FakeVerifier,
    *,
    merge_sha: str | None = None,
    rebased_from: str | None = None,
) -> MergeOutcome | GateClearance:
    return await _reverify_rebased_tree(
        scene.git_ops, rebased.request, rebased.merge_wt,
        verified_merge_sha=rebased.verified,
        rebased_from=rebased.rebased_from if rebased_from is None else rebased_from,
        rebased_onto=rebased.rebased_onto,
        timeouts={},
        enospc_retries={},
        max_timeouts=3,
        max_enospc=1,
        merge_sha=rebased.rebased_sha if merge_sha is None else merge_sha,
        run_post_merge_verify=functools.partial(_run_post_merge_verify, verifier=verifier),
        event_store=scene.store,
    )


@pytest.mark.asyncio
class TestReverifyGateRecordsItsDecision:
    async def test_an_overlap_fire_records_its_decision_and_one_verify(
        self, tmp_path: Path,
    ) -> None:
        scene = _scene(tmp_path)
        verifier = FakeVerifier()
        async with _rebased_merge(
            scene, scene.config(merge_verify_breadth='scoped'), _drift_on_the_branchs_file,
        ) as rebased:
            clearance = await _gate(scene, rebased, verifier)

        assert clearance is GateClearance.reverified, clearance
        assert verifier.entered_count == 1
        assert scene.gate_rows() == [{
            'task_id': rebased.request.task_id,
            'reason': 'overlap',
            'reverify': True,
            'merge_sha': rebased.rebased_sha,
            'verified_merge_sha': rebased.verified,
            'rebased_from': rebased.rebased_from,
            'rebased_onto': rebased.rebased_onto,
            'overlap': ['shared.txt'],
            'overlap_total': 1,
            'blockers': [],
        }]
        [verify] = scene.verify_rows(rebased.rebased_sha)
        assert verify['passed'] is True

    async def test_a_trusted_disjoint_skip_is_recorded_at_info_with_no_verify(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        scene = _scene(tmp_path)
        verifier = FakeVerifier()
        async with _rebased_merge(
            scene, scene.config(merge_verify_breadth='scoped'), _queue_landed(_drift_elsewhere),
        ) as rebased:
            with caplog.at_level(logging.INFO, logger=_GATE_LOGGER):
                clearance = await _gate(scene, rebased, verifier)

        assert clearance is GateClearance.disjoint, clearance
        assert verifier.entered_count == 0
        [row] = scene.gate_rows()
        assert (row['reason'], row['reverify'], row['overlap'], row['blockers']) == (
            'disjoint', False, [], [],
        )
        assert scene.verify_rows(rebased.rebased_sha) == []
        assert any(
            record.levelno == logging.INFO
            and rebased.request.task_id in record.getMessage()
            and 'disjoint' in record.getMessage()
            for record in caplog.records
        ), [record.getMessage() for record in caplog.records]

    async def test_an_untrusted_disjoint_fire_records_the_failed_premise(
        self, tmp_path: Path,
    ) -> None:
        scene = _scene(tmp_path)
        async with _rebased_merge(
            scene, scene.config(merge_verify_breadth='scoped'), _drift_elsewhere,
        ) as rebased:
            clearance = await _gate(scene, rebased, FakeVerifier())

        assert clearance is GateClearance.reverified, clearance
        [row] = scene.gate_rows()
        assert (row['reason'], row['reverify'], row['overlap']) == (
            'disjoint_untrusted', True, [],
        )
        [blocker] = row['blockers']
        assert 'unverified drift' in blocker
        assert len(scene.verify_rows(rebased.rebased_sha)) == 1

    async def test_a_failed_overlap_probe_fires_with_no_paths_recorded(
        self, tmp_path: Path,
    ) -> None:
        scene = _scene(tmp_path)
        async with _rebased_merge(
            scene, scene.config(merge_verify_breadth='scoped'), _drift_on_the_branchs_file,
        ) as rebased:
            clearance = await _gate(scene, rebased, FakeVerifier(), rebased_from='')

        assert clearance is GateClearance.reverified, clearance
        [row] = scene.gate_rows()
        assert (row['reason'], row['reverify'], row['overlap'], row['overlap_total']) == (
            'overlap_unknown', True, [], 0,
        )
        assert len(scene.verify_rows(rebased.rebased_sha)) == 1

    async def test_a_red_fire_records_its_decision_and_its_failing_verify(
        self, tmp_path: Path,
    ) -> None:
        scene = _scene(tmp_path)
        async with _rebased_merge(
            scene, scene.config(merge_verify_breadth='scoped'), _drift_on_the_branchs_file,
        ) as rebased:
            outcome = await _gate(
                scene, rebased, FakeVerifier(default=fails(category='pytest', summary='red')),
            )

        assert not isinstance(outcome, GateClearance), outcome
        assert outcome.status == 'blocked', outcome
        [row] = scene.gate_rows()
        assert row['reason'] == 'overlap'
        verifies = scene.verify_rows(rebased.rebased_sha)
        assert verifies and all(verify['passed'] is False for verify in verifies), verifies

    async def test_a_same_tree_skip_is_recorded_with_no_verify(self, tmp_path: Path) -> None:
        scene = _scene(tmp_path)
        verifier = FakeVerifier()
        async with _rebased_merge(
            scene,
            scene.config(merge_verify_breadth='full', merge_gate_history_pure=True),
            _drift_elsewhere,
        ) as rebased:
            same_tree_sha = _git(
                scene.repo, 'commit-tree', f'{rebased.verified}^{{tree}}',
                '-p', rebased.rebased_onto, '-m', 'same tree',
            )
            clearance = await _gate(scene, rebased, verifier, merge_sha=same_tree_sha)

        assert clearance is GateClearance.same_tree, clearance
        assert verifier.entered_count == 0
        [row] = scene.gate_rows()
        assert (row['reason'], row['reverify'], row['merge_sha'], row['verified_merge_sha']) == (
            'same_tree', False, same_tree_sha, rebased.verified,
        )
        assert scene.verify_rows(same_tree_sha) == []

    async def test_a_same_tree_is_not_a_skip_when_the_gate_reads_history(
        self, tmp_path: Path,
    ) -> None:
        scene = _scene(tmp_path)
        async with _rebased_merge(
            scene,
            scene.config(merge_verify_breadth='full', merge_gate_history_pure=False),
            _drift_elsewhere,
        ) as rebased:
            same_tree_sha = _git(
                scene.repo, 'commit-tree', f'{rebased.verified}^{{tree}}',
                '-p', rebased.rebased_onto, '-m', 'same tree',
            )
            clearance = await _gate(scene, rebased, FakeVerifier(), merge_sha=same_tree_sha)

        assert clearance is GateClearance.reverified, clearance
        [row] = scene.gate_rows()
        assert row['reason'] == 'disjoint_untrusted', row
        assert len(scene.verify_rows(same_tree_sha)) == 1


# ─── Both production call sites hand the worker's store to the gate ─────────


class _MovesMainOnItsFirstVerify:
    """A merge verify that passes, moving main under itself on its first call."""

    def __init__(self, repo: Path, drift: Drift) -> None:
        self._repo = repo
        self._drift = drift
        self.calls = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> VerifyResult:
        self.calls += 1
        if self.calls == 1:
            self._drift(self._repo)
        return VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='', summary='ok',
        )


async def _through_the_lane(
    scene: _Scene, request: MergeRequest, verify: _MovesMainOnItsFirstVerify,
) -> MergeOutcome:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=ScriptedVerifier(verify), event_store=scene.store,
    )
    return await merge_through_lane(lane, queue, request)


@pytest.mark.asyncio
class TestSingleBranchGateIsRecorded:
    async def test_an_overlap_fire_through_the_lane_writes_its_decision_and_its_verify(
        self, tmp_path: Path,
    ) -> None:
        scene = _scene(tmp_path)
        config = scene.config(merge_verify_breadth='scoped', merge_verify_min_free_disk_bytes=0)
        branch_wt = (await scene.git_ops.create_worktree('lane-5064')).path
        _edit_shared_line(branch_wt, 2, 'shared line 2, edited by the branch')
        verify = _MovesMainOnItsFirstVerify(scene.repo, _drift_on_the_branchs_file)

        outcome = await _through_the_lane(
            scene, _request(config, 'lane-5064', branch_wt), verify,
        )

        assert outcome.status == 'done', outcome
        landed = _git(scene.repo, 'rev-parse', 'main')
        [row] = scene.gate_rows()
        assert (row['task_id'], row['reason'], row['reverify'], row['overlap']) == (
            'lane-5064', 'overlap', True, ['shared.txt'],
        )
        assert row['merge_sha'] == landed
        [reverify] = scene.verify_rows(landed)
        assert reverify['passed'] is True

    async def test_a_same_tree_skip_through_the_lane_is_recorded(self, tmp_path: Path) -> None:
        scene = _scene(tmp_path)
        config = scene.config(merge_verify_breadth='full', merge_gate_history_pure=True)
        head_wt = (await scene.git_ops.create_worktree('head')).path
        head_commit = _edit_shared_line(head_wt, 2, 'shared line 2, edited by the head')
        follower_wt = (await scene.git_ops.create_worktree('follower')).path
        _git(follower_wt, 'merge', '--ff-only', head_commit)
        _edit_shared_line(follower_wt, 10, 'shared line 10, edited by the follower')

        def lands_the_heads_patch(repo: Path) -> str:
            _git(repo, 'cherry-pick', '-x', head_commit)
            return _git(repo, 'rev-parse', 'HEAD')

        outcome = await _through_the_lane(
            scene, _request(config, 'follower', follower_wt),
            _MovesMainOnItsFirstVerify(scene.repo, lands_the_heads_patch),
        )

        assert outcome.status == 'done', outcome
        [row] = scene.gate_rows()
        assert (row['task_id'], row['reason'], row['reverify']) == ('follower', 'same_tree', False)
        assert row['merge_sha'] == _git(scene.repo, 'rev-parse', 'main')
