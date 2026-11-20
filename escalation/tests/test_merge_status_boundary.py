"""merge_status's boundary suite, rows B1-B9, across a simulated restart.

The executable form of ``plans/merge-status-durable-non-landed-prd.md``'s
Contract (B+H) and Boundary-test sketch, rows B1-B9.  B10 lives in
``orchestrator/tests/test_merge_status_boundary_b10.py`` because its subject is
EventStore's cross-run read.

Real git, real producers, real ``escalation.git_authority.probe_landing``: the
prior run's state is written by the orchestrator's own finalize and journal
writers, and the simulated restart is a fresh run_id plus a fresh journal store
over the prior run's files.  The unit-scope tier tests are
``escalation/tests/test_merge_status_tiers.py``.
"""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import re
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import pytest
from _merge_tool_calls import call_merge_status
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import (
    InFlightMergeRegistry,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    WaiterRecord,
    coalesce_or_enqueue_merge_request,
    enqueue_merge_request,
    landing_evidence,
)
from orchestrator.merge_lane import worker as merge_worker
from orchestrator.merge_lane.intake import register_and_enqueue_merge_request
from orchestrator.merge_lane.queue_store import MergeQueueStore
from shared.merge_state import (
    EPISTEMIC_STATES,
    LIVE_STATES,
    MERGE_REASONS,
    OUTCOME_STATES,
    MergeReason,
    MergeState,
)
from shared.merge_subject import format_merge_subject

from escalation import git_authority
from escalation.git_authority import GitAuthorityOutcome, GitAuthorityVerdict
from escalation.merge_status_answer import hint_for
from escalation.queue import EscalationQueue
from escalation.server import create_server

pytestmark = pytest.mark.asyncio

_RUN_PRIOR = 'run-prior'
_RUN_CURRENT = 'run-current'

#: I1: the only tiers whose evidence may set an outcome state.
_PRESENT_TRUTH_SOURCES = frozenset({'live', 'event_store', 'git'})

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _git_succeeds(repo: Path, *args: str) -> bool:
    return subprocess.run(['git', *args], cwd=repo, capture_output=True).returncode == 0


@dataclass(frozen=True)
class _World:
    """One project on disk: a git repo plus the files a run leaves behind."""

    repo: Path
    config: OrchestratorConfig
    git_ops: GitOps
    db: Path
    journal_path: Path
    escalations: Path


@pytest.fixture
def world(tmp_path: Path) -> _World:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@test.com')
    _git(repo, 'config', 'user.name', 'Test')
    (repo / 'README.md').write_text('boundary suite\n')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-m', 'initial commit')
    config = OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
        ),
    )
    return _World(
        repo=repo,
        config=config,
        git_ops=GitOps(config.git, config.project_root),
        db=tmp_path / 'runs.db',
        journal_path=tmp_path / 'merge_queue.json',
        escalations=tmp_path / 'escalations',
    )


def _unmerged_branch(world: _World, task_id: str) -> None:
    """task/<task_id> carries one commit main does not."""
    _git(world.repo, 'checkout', '-q', '-b', f'task/{task_id}', 'main')
    (world.repo / f'{task_id}.txt').write_text(f'work for {task_id}\n')
    _git(world.repo, 'add', '-A')
    _git(world.repo, 'commit', '-q', '-m', f'work for {task_id}')
    _git(world.repo, 'checkout', '-q', 'main')


def _rebase_landing(world: _World, task_id: str) -> str:
    """Land task/<task_id>'s work as a rebased copy under a merge marker, delete the ref, return the original tip."""
    _unmerged_branch(world, task_id)
    original_tip = _git(world.repo, 'rev-parse', f'task/{task_id}')
    (world.repo / 'main-moved.txt').write_text(f'main moved past task/{task_id}\n')
    _git(world.repo, 'add', '-A')
    _git(world.repo, 'commit', '-q', '-m', 'main moves on')
    rebased = f'rebased-{task_id}'
    _git(world.repo, 'checkout', '-q', '-b', rebased, 'main')
    _git(world.repo, 'cherry-pick', original_tip)
    _git(world.repo, 'checkout', '-q', 'main')
    _git(
        world.repo, 'merge', '-q', '--no-ff', rebased,
        '-m', format_merge_subject(f'task/{task_id}', 'main'),
    )
    _git(world.repo, 'branch', '-q', '-D', rebased, f'task/{task_id}')
    return original_tip


_INSIDE_PROBE: ContextVar[bool] = ContextVar('inside_probe_landing', default=False)

_GIT_OPS_LANDING_PRIMITIVES = (
    'is_ancestor',
    'find_merge_marker',
    'find_equivalent_commit',
    'find_task_citation_commit',
    'commit_effect_present_in_main',
    'branch_content_in_main',
)

_MODULE_LANDING_PRIMITIVES = (
    (landing_evidence, 'patch_content_contained'),
    (landing_evidence, 'branch_work_landed'),
    (merge_worker, 'patch_content_contained'),
)


@dataclass(frozen=True)
class _LandingPrimitiveSpy:
    """Passthrough wrappers recording whether each landing-primitive call ran inside probe_landing (I5)."""

    calls: list[tuple[str, bool]] = field(default_factory=list)
    verdicts: list[GitAuthorityVerdict] = field(default_factory=list)

    @classmethod
    def install(cls, monkeypatch: pytest.MonkeyPatch, git_ops: GitOps) -> _LandingPrimitiveSpy:
        spy = cls()
        monkeypatch.setattr(
            git_authority, 'probe_landing', spy._attributing(git_authority.probe_landing),
        )
        for name in _GIT_OPS_LANDING_PRIMITIVES:
            monkeypatch.setattr(
                git_ops, name, spy._recording(f'GitOps.{name}', getattr(git_ops, name)),
            )
        for module, name in _MODULE_LANDING_PRIMITIVES:
            label = f'{module.__name__}.{name}'
            monkeypatch.setattr(module, name, spy._recording(label, getattr(module, name)))
        return spy

    def _attributing(
        self, probe: Callable[..., Awaitable[GitAuthorityVerdict]],
    ) -> Callable[..., Awaitable[GitAuthorityVerdict]]:
        async def probe_landing(*args: Any, **kwargs: Any) -> GitAuthorityVerdict:
            token = _INSIDE_PROBE.set(True)
            try:
                verdict = await probe(*args, **kwargs)
            finally:
                _INSIDE_PROBE.reset(token)
            self.verdicts.append(verdict)
            return verdict

        return probe_landing

    def _recording(
        self, label: str, primitive: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        async def recorded(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((label, _INSIDE_PROBE.get()))
            return await primitive(*args, **kwargs)

        return recorded

    @property
    def calls_inside_probe(self) -> list[str]:
        return [label for label, inside in self.calls if inside]

    @property
    def calls_outside_probe(self) -> list[str]:
        return [label for label, inside in self.calls if not inside]


def _request(world: _World, task_id: str) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, world.config.git.branch_prefix),
        worktree=world.repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=world.config,
        result=asyncio.get_running_loop().create_future(),
    )


async def _finalize_in_prior_run(
    world: _World, task_id: str, outcome: MergeOutcome,
) -> MergeRequest:
    """The prior run submits *task_id* and settles it; the real finalize callback writes the row."""
    req = _request(world, task_id)
    await enqueue_merge_request(asyncio.Queue(), req, EventStore(world.db, _RUN_PRIOR))
    req.result.set_result(outcome)
    await asyncio.sleep(0)
    return req


def _journal_in_prior_run(world: _World, task_id: str) -> MergeRequest:
    """The prior run journals an owned request and dies before it finalizes."""
    req = _request(world, task_id)
    MergeQueueStore(world.journal_path).record(req)
    return req


def _live_entry(req: MergeRequest, state: str) -> dict[str, Any]:
    """The keys merge_status reads from orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.snapshot's entries."""
    return {
        'task_id': req.task_id,
        'branch': req.branch.bare_id,
        'state': state,
        'enqueued_at': req.enqueued_at,
        'position': 0,
        'waiter_alive': not req.result.cancelled(),
        'request_id': req.request_id,
    }


def _restart(
    world: _World,
    *,
    live_entries: Sequence[Mapping[str, Any]] = (),
    git_ops: GitOps | None = None,
) -> Any:
    """The restarted process: a fresh run_id and a fresh journal store over the prior run's files."""
    worker = (
        SimpleNamespace(snapshot=lambda: {'entries': list(live_entries)})
        if live_entries else None
    )
    harness = SimpleNamespace(
        _merge_worker=worker,
        _merge_store=MergeQueueStore(world.journal_path),
        git_ops=world.git_ops if git_ops is None else git_ops,
    )
    return create_server(
        EscalationQueue(world.escalations),
        orch_config=world.config,
        event_store=EventStore(world.db, _RUN_CURRENT),
        harness=harness,
    )


def _assert_wire_contract(response: Mapping[str, Any]) -> None:
    state = response['state']
    reason = response.get('reason')
    last = response['last_outcome']
    assert state in frozenset(MergeState), f'I4: state is not a MergeState value: {response}'
    if state in OUTCOME_STATES:
        assert response['source'] in _PRESENT_TRUTH_SOURCES, (
            f'I1: outcome state from a history source: {response}'
        )
        assert 'hint' not in response, f'an outcome state carries no hint: {response}'
    if last is not None and last['is_current_run'] is False and response['source'] != 'git':
        # A git-proven state beside equal history is corroboration:
        # escalation/src/escalation/merge_status_answer.py (module docstring).
        assert last['state'] != state, f'I2: a prior-run row set the state: {response}'
    assert (state == MergeState.unknown) == (reason == MergeReason.probe_failed), (
        f'I3: unknown iff probe_failed: {response}'
    )
    if state in EPISTEMIC_STATES:
        assert reason in frozenset(MergeReason), f'epistemic state without a MergeReason: {response}'
        assert response['hint'] == hint_for(MergeReason(reason)), f'hint not derived from reason: {response}'
    if last is not None:
        assert isinstance(last['run_id'], str), f'last_outcome.run_id: {response}'
        assert isinstance(last['finished_at'], str), f'last_outcome.finished_at: {response}'
        assert isinstance(last['is_current_run'], bool), f'last_outcome.is_current_run: {response}'


async def _poll(server: Any, **key: str) -> dict[str, Any]:
    response = await call_merge_status(server, **key)
    _assert_wire_contract(response)
    return response


async def _absorb_by_attach(
    world: _World, primary: MergeRequest, store: EventStore, registry: InFlightMergeRegistry,
) -> str:
    await register_and_enqueue_merge_request(asyncio.Queue(), primary, store, registry)
    loser = WaiterRecord(
        request_id='mr-attached-loser',
        future=asyncio.get_running_loop().create_future(),
        source='workflow',
    )
    assert registry.attach(primary.branch.bare_id, loser)
    return loser.request_id


async def _absorb_by_door_coalesce(
    world: _World, primary: MergeRequest, store: EventStore, registry: InFlightMergeRegistry,
) -> str:
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    loser = _request(world, primary.task_id)
    assert (await coalesce_or_enqueue_merge_request(queue, primary, store, registry)).dispatched
    assert (await coalesce_or_enqueue_merge_request(queue, loser, store, registry)).in_flight
    return loser.request_id


_ABSORB_MODES: Mapping[
    str, Callable[[_World, MergeRequest, EventStore, InFlightMergeRegistry], Awaitable[str]]
] = MappingProxyType({'attach': _absorb_by_attach, 'door_coalesce': _absorb_by_door_coalesce})


async def _finalize_absorbing_in_prior_run(
    world: _World, task_id: str, outcome: MergeOutcome, *, mode: str,
) -> tuple[MergeRequest, str]:
    """The prior run's primary absorbs a loser, then settles; returns the primary and the loser's id."""
    primary = _request(world, task_id)
    absorb = _ABSORB_MODES[mode]
    loser_id = await absorb(world, primary, EventStore(world.db, _RUN_PRIOR), InFlightMergeRegistry())
    primary.result.set_result(outcome)
    await asyncio.sleep(0)
    return primary, loser_id


@dataclass(frozen=True)
class _Scene:
    """One row after the prior run: the restarted server, the key the row polls, the ids it asserts against."""

    server: Any
    key: Mapping[str, str]
    facts: Mapping[str, str]


_VERIFY_FAILED = MergeOutcome('blocked', reason='post-merge verify failed')
_REBASE_CONFLICT = MergeOutcome('conflict', reason='rebase conflict')


async def _scene_b1(world: _World) -> _Scene:
    _unmerged_branch(world, '101')
    prior = await _finalize_in_prior_run(world, '101', _VERIFY_FAILED)
    return _Scene(
        _restart(world), {'task_id': '101'}, {'task_id': '101', 'request_id': prior.request_id},
    )


async def _scene_b2(world: _World) -> _Scene:
    await _finalize_in_prior_run(world, '102', _REBASE_CONFLICT)
    resubmitted = _request(world, '102')
    MergeQueueStore(world.journal_path).record(resubmitted)
    server = _restart(world, live_entries=[_live_entry(resubmitted, 'queued')])
    return _Scene(server, {'task_id': '102'}, {'resubmitted_request_id': resubmitted.request_id})


async def _scene_b3(world: _World) -> _Scene:
    _unmerged_branch(world, '103')
    journaled = _journal_in_prior_run(world, '103')
    return _Scene(
        _restart(world), {'task_id': '103'}, {'task_id': '103', 'request_id': journaled.request_id},
    )


async def _scene_b4(world: _World) -> _Scene:
    return _Scene(_restart(world), {'task_id': '104'}, {})


async def _scene_b5(world: _World) -> _Scene:
    original_tip = _rebase_landing(world, '105')
    await _finalize_in_prior_run(world, '105', _VERIFY_FAILED)
    return _Scene(
        _restart(world), {'task_id': '105'}, {'task_id': '105', 'original_tip': original_tip},
    )


async def _scene_b6(world: _World) -> _Scene:
    train_id = 'coalesce-1062-5ca1ab1e'
    superseded = MergeOutcome('superseded', superseded_by=train_id)
    await _finalize_in_prior_run(world, '1061', superseded)
    newest = await _finalize_in_prior_run(world, '1062', superseded)
    return _Scene(
        _restart(world),
        {'request_id': train_id},
        {'train_id': train_id, 'newest_member_request_id': newest.request_id},
    )


async def _scene_b7(world: _World, *, mode: str = 'attach') -> _Scene:
    primary, loser_id = await _finalize_absorbing_in_prior_run(
        world, '107', _REBASE_CONFLICT, mode=mode,
    )
    return _Scene(
        _restart(world), {'request_id': loser_id}, {'primary_request_id': primary.request_id},
    )


async def _scene_b8(world: _World) -> _Scene:
    await _finalize_in_prior_run(world, '108', _VERIFY_FAILED)
    vanished = GitOps(world.config.git, world.repo.parent / 'vanished')
    return _Scene(_restart(world, git_ops=vanished), {'task_id': '108'}, {})


_SCENES: Mapping[str, Callable[[_World], Awaitable[_Scene]]] = MappingProxyType({
    'B1': _scene_b1,
    'B2': _scene_b2,
    'B3': _scene_b3,
    'B4': _scene_b4,
    'B5': _scene_b5,
    'B6': _scene_b6,
    'B7': _scene_b7,
    'B8': _scene_b8,
})


class TestB1PriorRunBlockedAfterRestart:
    """B1 is THE REGRESSION PIN — must not be waived: a prior run's blocked is never this run's state."""

    @pytest.mark.parametrize('key', ['task_id', 'request_id'])
    async def test_b1_a_prior_run_blocked_answers_stale_record_never_blocked(
        self, world: _World, key: str,
    ) -> None:
        scene = await _scene_b1(world)

        response = await _poll(scene.server, **{key: scene.facts[key]})

        assert response['state'] != 'blocked'
        assert response['state'] not in OUTCOME_STATES
        assert response['state'] == MergeState.stale_record
        assert response['reason'] == MergeReason.prior_run_terminal
        assert response['source'] == 'event_store_cross_run'
        assert response['hint'] == hint_for(MergeReason.prior_run_terminal)
        last = response['last_outcome']
        assert last['state'] == 'blocked'
        assert last['is_current_run'] is False
        assert last['run_id'] == _RUN_PRIOR
        assert last['request_id'] == scene.facts['request_id']


class TestB2CurrentRunOnlySetsState:
    async def test_b2_a_live_resubmission_sets_state_and_the_prior_conflict_stays_history(
        self, world: _World,
    ) -> None:
        scene = await _scene_b2(world)

        response = await _poll(scene.server, **scene.key)

        assert response['state'] != 'conflict'
        assert response['state'] == MergeState.queued
        assert response['source'] == 'live'
        assert response['request_id'] == scene.facts['resubmitted_request_id']
        last = response['last_outcome']
        assert last['state'] == 'conflict'
        assert last['run_id'] == _RUN_PRIOR
        assert last['is_current_run'] is False


class TestB3JournaledNotYetRecovered:
    @pytest.mark.parametrize('key', ['task_id', 'request_id'])
    async def test_b3_a_journaled_request_with_no_live_entry_answers_journaled(
        self, world: _World, key: str,
    ) -> None:
        scene = await _scene_b3(world)

        response = await _poll(scene.server, **{key: scene.facts[key]})

        assert response['state'] == MergeState.journaled
        assert response['source'] == 'journal'
        assert response['reason'] == MergeReason.journaled_not_live
        assert response['request_id'] == scene.facts['request_id']
        assert response['last_outcome'] is None


class TestB4NeverSubmitted:
    async def test_b4_nothing_anywhere_answers_no_record_never_submitted(
        self, world: _World,
    ) -> None:
        scene = await _scene_b4(world)

        response = await _poll(scene.server, **scene.key)

        assert response['state'] == MergeState.no_record
        assert response['reason'] == MergeReason.never_submitted
        assert response['source'] == 'none'
        assert response['last_outcome'] is None
        assert response['hint'] == hint_for(MergeReason.never_submitted)


class TestB5RebaseLandingIsConsumedFromGit:
    async def test_b5_a_rebase_landing_answers_done_from_git_with_no_landing_call_of_its_own(
        self, world: _World, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = await _scene_b5(world)
        task_id, original_tip = scene.facts['task_id'], scene.facts['original_tip']
        assert not _git_succeeds(world.repo, 'merge-base', '--is-ancestor', original_tip, 'main')
        assert _git(world.repo, 'show', f'main:{task_id}.txt') == _git(
            world.repo, 'show', f'{original_tip}:{task_id}.txt',
        )
        assert not _git_succeeds(
            world.repo, 'rev-parse', '--verify', '--quiet', f'refs/heads/task/{task_id}',
        )
        spy = _LandingPrimitiveSpy.install(monkeypatch, world.git_ops)

        response = await _poll(scene.server, **scene.key)

        assert response['state'] == MergeState.done
        assert response['kind'] == 'found_on_main'
        assert response['source'] == 'git'
        (verdict,) = spy.verdicts
        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert response['merge_sha'] == verdict.merge_sha
        last = response['last_outcome']
        assert last['state'] == 'blocked'
        assert last['is_current_run'] is False
        assert spy.calls_outside_probe == []
        assert spy.calls_inside_probe


async def test_b6_a_coalesce_train_id_resolves_through_the_reverse_index(world: _World) -> None:
    scene = await _scene_b6(world)

    response = await _poll(scene.server, **scene.key)

    assert response['state'] != MergeState.no_record
    assert response['state'] == MergeState.stale_record
    assert response['reason'] == MergeReason.coalesce_train_id
    assert response['request_id'] == scene.facts['train_id']
    last = response['last_outcome']
    assert last['state'] == 'superseded'
    assert last['superseded_by'] == scene.facts['train_id']
    assert last['request_id'] == scene.facts['newest_member_request_id']


@pytest.mark.parametrize('mode', sorted(_ABSORB_MODES))
async def test_b7_an_absorbed_loser_resolves_to_its_primary_after_restart(
    world: _World, mode: str,
) -> None:
    scene = await _scene_b7(world, mode=mode)

    response = await _poll(scene.server, **scene.key)

    assert response['state'] == MergeState.stale_record
    assert response['reason'] == MergeReason.prior_run_terminal
    last = response['last_outcome']
    assert last['request_id'] == scene.facts['primary_request_id']
    assert last['state'] == 'conflict'
    assert last['is_current_run'] is False
    assert last['run_id'] == _RUN_PRIOR


async def test_b8_a_raising_git_probe_is_the_only_unknown(world: _World) -> None:
    scene = await _scene_b8(world)

    response = await _poll(scene.server, **scene.key)

    assert response['state'] != MergeState.stale_record
    assert response['state'] == MergeState.unknown
    assert response['reason'] == MergeReason.probe_failed
    assert response['source'] == 'git'
    assert response['hint'] == hint_for(MergeReason.probe_failed)
    assert response['last_outcome']['run_id'] == _RUN_PRIOR


@pytest.mark.parametrize('row', sorted(_SCENES))
async def test_unknown_is_produced_by_b8_alone(world: _World, row: str) -> None:
    scene = await _SCENES[row](world)

    response = await _poll(scene.server, **scene.key)

    assert (response['state'] == MergeState.unknown) == (row == 'B8'), (row, response)


_MERGE_QUEUE_SKILL = 'skills/merge-queue/SKILL.md'
_UNBLOCK_SKILL = 'skills/unblock/SKILL.md'
_UNBLOCK_LOW_RISK_SKILL = 'skills/unblock-low-risk/SKILL.md'
_CONSUMER_SKILLS = frozenset({_MERGE_QUEUE_SKILL, _UNBLOCK_SKILL, _UNBLOCK_LOW_RISK_SKILL})
_SHARED_SWITCH = 'skills/_shared/merge-status-reasons.md'
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding='utf-8')


def _load_alpha_guard() -> ModuleType:
    """A private copy of alpha's guard module, never registered in sys.modules."""
    path = _REPO_ROOT / 'scripts' / 'tests' / 'test_merge_state_vocabulary_consistency.py'
    spec = importlib.util.spec_from_file_location('_merge_state_guard_for_b9', path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def alpha_guard() -> ModuleType:
    return _load_alpha_guard()


#: The closed disposition vocabulary a route span or switch row states, in bold.
_DISPOSITIONS = (
    'WAIT', 'ACCEPT', 'REPORT', 'RESUBMIT', 'FOLLOW THE MEMBERS', 'ABORT', 'HAND OFF',
    'VERIFY', 'NOT IN FLIGHT HERE', 'FAIL CLOSED',
)
_DISPOSITION_RE = re.compile(r'\*\*(' + '|'.join(map(re.escape, _DISPOSITIONS)) + r')\*\*')
_ROUTABLE_PARTITIONS: Mapping[str, frozenset[MergeState]] = MappingProxyType({
    'LIVE_STATES': LIVE_STATES,
})
_SWITCH_ROW_RE = re.compile(r'^\|\s*`([a-z_]+)`\s*\|')


def _families(guard: ModuleType) -> tuple[Any, Any]:
    """The route and switch span families, built from *guard*'s own SpanFamily."""
    route = guard.SpanFamily(
        begin='merge-status-route:begin',
        end='merge-status-route:end',
        name_re=re.compile(r'answer=([A-Za-z_]+(?:,[A-Za-z_]+)*)(?=\s|-->|$)'),
        known_names=None,
        registry='`_CONSUMER_SKILLS` in escalation/tests/test_merge_status_boundary.py',
    )
    switch = guard.SpanFamily(
        begin='merge-status-switch:begin',
        end='merge-status-switch:end',
        name_re=re.compile(r'of=([A-Z][A-Z0-9_]*)'),
        known_names=frozenset({'MERGE_REASONS'}),
        registry='`_SHARED_SWITCH` in escalation/tests/test_merge_status_boundary.py',
    )
    return route, switch


def _single_disposition(text: str, *, source: str, where: str) -> str:
    found = sorted(set(_DISPOSITION_RE.findall(text)))
    assert len(found) == 1, (
        f'{source}: {where} must state exactly one bold disposition of '
        f'{list(_DISPOSITIONS)}; found {found}'
    )
    return found[0]


def _switch_in(guard: ModuleType, text: str, *, source: str) -> dict[MergeReason, str]:
    """The shared switch's action class for every MergeReason, one table row each."""
    _, switch_family = _families(guard)
    spans = guard.extract_spans(text, source=source, family=switch_family)
    assert len(spans) == 1, f'{source}: {len(spans)} merge-status-switch spans; exactly one expected'
    (_, body), = spans
    actions: dict[MergeReason, list[str]] = {}
    for line in body.splitlines():
        match = _SWITCH_ROW_RE.match(line)
        if match is None or match.group(1) not in MERGE_REASONS:
            continue
        reason = MergeReason(match.group(1))
        actions.setdefault(reason, []).append(
            _single_disposition(line, source=source, where=f'the switch row for {reason}'),
        )
    missing = sorted(MERGE_REASONS - actions.keys())
    duplicated = sorted(reason for reason, found in actions.items() if len(found) > 1)
    assert not missing and not duplicated, (
        f'{source}: the merge-status-switch table needs exactly one row per MERGE_REASONS '
        f'member; missing {missing}, duplicated {duplicated}'
    )
    return {reason: found for reason, (found,) in actions.items()}


def _state_keys(state: MergeState) -> frozenset[str]:
    """The route keys a reason-less answer consults: its state and each routable partition holding it."""
    return frozenset({state, *(name for name, states in _ROUTABLE_PARTITIONS.items() if state in states)})


def _reason_keys(switch: Mapping[MergeReason, str], reason: MergeReason) -> frozenset[str]:
    """The route keys an answer carrying *reason* consults: the reason and its switch action class."""
    return frozenset({reason, switch[reason]})


def _candidates(
    switch: Mapping[MergeReason, str], state: MergeState, reason: MergeReason | None,
) -> frozenset[str]:
    """The route keys that answer (state, reason)."""
    return _state_keys(state) if reason is None else _reason_keys(switch, reason)


def _routable_keys(switch: Mapping[MergeReason, str]) -> frozenset[str]:
    """Every route key some merge_status answer consults.

    Only an outcome state is answered without a reason; every epistemic state carries
    one (shared/src/shared/merge_state.py::MergeReason), so it is routed by that reason.
    """
    return frozenset().union(
        *map(_state_keys, OUTCOME_STATES),
        *(_reason_keys(switch, reason) for reason in MergeReason),
    )


def _routes_in(
    guard: ModuleType, text: str, switch: Mapping[MergeReason, str], *, source: str,
) -> dict[str, str]:
    """Every merge-status-route key in *text*, mapped to its span's disposition."""
    route_family, _ = _families(guard)
    routable = _routable_keys(switch)
    routes: dict[str, str] = {}
    for keys, body in guard.extract_spans(text, source=source, family=route_family):
        where = f'the route span answer={keys}'
        disposition = _single_disposition(body, source=source, where=where)
        for key in keys.split(','):
            assert key in routable, (
                f'{source}: {where} routes {key!r}, which no merge_status answer consults. '
                f'A route key is an outcome state, a routable partition '
                f'{sorted(_ROUTABLE_PARTITIONS)}, a MergeReason value or a switch action '
                f'class {sorted(set(switch.values()))}; an epistemic state always carries '
                f'a reason and is routed by it'
            )
            assert key not in routes, f'{source}: {where} routes {key!r} a second time'
            routes[key] = disposition
    return routes


def _resolve(
    routes: Mapping[str, str],
    switch: Mapping[MergeReason, str],
    state: MergeState,
    reason: MergeReason | None,
    *,
    source: str,
) -> str:
    """The disposition of the one route key that answers (state, reason)."""
    candidates = _candidates(switch, state, reason)
    matched = sorted(key for key in candidates if key in routes)
    assert len(matched) == 1, (
        f'{source}: state={state!s}, reason={reason!s} must be routed by exactly one of '
        f'{sorted(candidates)}; routed by {matched}'
    )
    return routes[matched[0]]


def _per_skill(*, merge_queue: str, unblock: str, unblock_low_risk: str) -> Mapping[str, str]:
    return MappingProxyType({
        _MERGE_QUEUE_SKILL: merge_queue,
        _UNBLOCK_SKILL: unblock,
        _UNBLOCK_LOW_RISK_SKILL: unblock_low_risk,
    })


_WAIT_EVERYWHERE = _per_skill(merge_queue='WAIT', unblock='WAIT', unblock_low_risk='WAIT')
_PRIOR_RUN_TERMINAL = _per_skill(merge_queue='REPORT', unblock='REPORT', unblock_low_risk='ABORT')

#: Each row's server answer and each consumer's disposition of it (task 4832's dispositions).
_CONSUMER_BRANCHES = (
    ('B1', MergeState.stale_record, MergeReason.prior_run_terminal, _PRIOR_RUN_TERMINAL),
    ('B2', MergeState.queued, None, _WAIT_EVERYWHERE),
    ('B3', MergeState.journaled, MergeReason.journaled_not_live, _WAIT_EVERYWHERE),
    (
        'B4', MergeState.no_record, MergeReason.never_submitted,
        _per_skill(merge_queue='RESUBMIT', unblock='RESUBMIT', unblock_low_risk='ABORT'),
    ),
    (
        'B5', MergeState.done, None,
        _per_skill(merge_queue='ACCEPT', unblock='ACCEPT', unblock_low_risk='ACCEPT'),
    ),
    (
        'B6', MergeState.stale_record, MergeReason.coalesce_train_id,
        _per_skill(
            merge_queue='FOLLOW THE MEMBERS', unblock='FOLLOW THE MEMBERS', unblock_low_risk='ABORT',
        ),
    ),
    ('B7', MergeState.stale_record, MergeReason.prior_run_terminal, _PRIOR_RUN_TERMINAL),
    ('B8', MergeState.unknown, MergeReason.probe_failed, _WAIT_EVERYWHERE),
)


@pytest.mark.parametrize(
    ('row', 'state', 'reason', 'expected'),
    [pytest.param(*branch, id=branch[0]) for branch in _CONSUMER_BRANCHES],
)
async def test_consumer_skill_branches_on_the_server_answer(
    world: _World,
    alpha_guard: ModuleType,
    row: str,
    state: MergeState,
    reason: MergeReason | None,
    expected: Mapping[str, str],
) -> None:
    scene = await _SCENES[row](world)
    response = await _poll(scene.server, **scene.key)
    assert (response['state'], response.get('reason')) == (state, reason), (row, response)

    switch = _switch_in(alpha_guard, _read(_SHARED_SWITCH), source=_SHARED_SWITCH)
    dispositions = {
        skill: _resolve(
            _routes_in(alpha_guard, _read(skill), switch, source=skill),
            switch, state, reason, source=skill,
        )
        for skill in sorted(_CONSUMER_SKILLS)
    }

    assert dispositions == dict(expected), row


async def test_every_route_a_consumer_skill_carries_is_checked_by_a_row(
    alpha_guard: ModuleType,
) -> None:
    switch = _switch_in(alpha_guard, _read(_SHARED_SWITCH), source=_SHARED_SWITCH)
    checked = frozenset().union(
        *(_candidates(switch, state, reason) for _, state, reason, _ in _CONSUMER_BRANCHES),
    )

    unchecked = {
        skill: keys
        for skill in sorted(_CONSUMER_SKILLS)
        if (keys := sorted(set(_routes_in(alpha_guard, _read(skill), switch, source=skill)) - checked))
    }

    assert not unchecked, (
        f'route keys no _CONSUMER_BRANCHES row consults, by skill: {unchecked}. Such a '
        f'span looks pinned but is checked by nothing: add a row whose answer consults '
        f'the key, or unwrap the span'
    )


async def test_every_scene_has_a_consumer_row() -> None:
    assert {row for row, *_ in _CONSUMER_BRANCHES} == set(_SCENES)


_FIXTURE_SKILL = 'fixture-skill.md'

_FIXTURE_SWITCH: Mapping[MergeReason, str] = MappingProxyType({
    MergeReason.probe_failed: 'WAIT',
    MergeReason.journaled_not_live: 'WAIT',
    MergeReason.prior_run_terminal: 'REPORT',
    MergeReason.never_submitted: 'NOT IN FLIGHT HERE',
    MergeReason.coalesce_train_id: 'FOLLOW THE MEMBERS',
    MergeReason.landing_unattributable: 'VERIFY',
})

_ROUTES_GREEN = """\
<!-- merge-status-route:begin answer=LIVE_STATES,WAIT -->
A live state or a wait-state reason — **WAIT**: keep polling. Plain ABORT and WAIT_REASONS are prose.
<!-- merge-status-route:end -->
<!-- merge-status-route:begin answer=prior_run_terminal -->
→ **REPORT** `last_outcome` to the human and stop.
<!-- merge-status-route:end -->
<!-- merge-status-route:begin answer=never_submitted -->
→ **RESUBMIT** (go back to step 3).
<!-- merge-status-route:end -->
"""

_ROUTES_TWO_KEYWORDS = """\
<!-- merge-status-route:begin answer=prior_run_terminal -->
→ **REPORT** `last_outcome`, or **ABORT**.
<!-- merge-status-route:end -->
"""

_ROUTES_NO_KEYWORD = """\
<!-- merge-status-route:begin answer=prior_run_terminal -->
→ report `last_outcome`; ABORT in plain prose and **WAIT_REASONS** are not keywords.
<!-- merge-status-route:end -->
"""

_ROUTES_DUPLICATE_KEY = """\
<!-- merge-status-route:begin answer=never_submitted -->
→ **RESUBMIT**.
<!-- merge-status-route:end -->
<!-- merge-status-route:begin answer=done,never_submitted -->
→ **ACCEPT**.
<!-- merge-status-route:end -->
"""

_ROUTES_UNKNOWN_KEY = """\
<!-- merge-status-route:begin answer=never_submited -->
→ **RESUBMIT**.
<!-- merge-status-route:end -->
"""

_ROUTES_SPACED_KEYS = """\
<!-- merge-status-route:begin answer=prior_run_terminal, never_submitted -->
→ **ABORT**.
<!-- merge-status-route:end -->
"""

_ROUTES_EPISTEMIC_STATE = """\
<!-- merge-status-route:begin answer=stale_record -->
→ **RESUBMIT**.
<!-- merge-status-route:end -->
"""

_ROUTES_CONSUMER_DISPOSITION = """\
<!-- merge-status-route:begin answer=ACCEPT -->
→ **ACCEPT**.
<!-- merge-status-route:end -->
"""


async def test_routes_in_maps_every_key_of_every_route_span(alpha_guard: ModuleType) -> None:
    routes = _routes_in(alpha_guard, _ROUTES_GREEN, _FIXTURE_SWITCH, source=_FIXTURE_SKILL)

    assert routes == {
        'LIVE_STATES': 'WAIT',
        'WAIT': 'WAIT',
        'prior_run_terminal': 'REPORT',
        'never_submitted': 'RESUBMIT',
    }


@pytest.mark.parametrize(
    ('text', 'offender'),
    [
        pytest.param(_ROUTES_TWO_KEYWORDS, 'prior_run_terminal', id='two-keywords'),
        pytest.param(_ROUTES_NO_KEYWORD, 'prior_run_terminal', id='no-keyword'),
        pytest.param(_ROUTES_DUPLICATE_KEY, 'never_submitted', id='duplicate-key'),
        pytest.param(_ROUTES_UNKNOWN_KEY, 'never_submited', id='unknown-key'),
        pytest.param(_ROUTES_SPACED_KEYS, 'merge-status-route:begin', id='spaced-keys'),
        pytest.param(_ROUTES_EPISTEMIC_STATE, 'stale_record', id='epistemic-state'),
        pytest.param(_ROUTES_CONSUMER_DISPOSITION, 'ACCEPT', id='consumer-disposition'),
    ],
)
async def test_routes_in_rejects_every_malformed_route(
    alpha_guard: ModuleType, text: str, offender: str,
) -> None:
    with pytest.raises(AssertionError) as excinfo:
        _routes_in(alpha_guard, text, _FIXTURE_SWITCH, source=_FIXTURE_SKILL)

    assert _FIXTURE_SKILL in str(excinfo.value)
    assert offender in str(excinfo.value)


_SWITCH_ROWS = {
    'probe_failed': '| `probe_failed` | `unknown` | a probe raised | **WAIT**: retry on the backoff |',
    'journaled_not_live': '| `journaled_not_live` | `journaled` | journal holds it | **WAIT**. Never resubmit |',
    'prior_run_terminal': '| `prior_run_terminal` | `stale_record` | another run | **REPORT** `last_outcome` |',
    'never_submitted': '| `never_submitted` | `no_record` | no tier holds it | **NOT IN FLIGHT HERE**: consumer decides |',
    'coalesce_train_id': '| `coalesce_train_id` | `stale_record` | a train | **FOLLOW THE MEMBERS** |',
    'landing_unattributable': '| `landing_unattributable` | any but `unknown` | on main, unattributed | **VERIFY** at once |',
}


def _switch_fixture(*rows: str, before: str = '', after: str = '') -> str:
    table = '\n'.join((
        '| `reason` | rides on | meaning | action |',
        '|---|---|---|---|',
        *rows,
        '| any reason not listed here | — | the vocabulary grew | **FAIL CLOSED**: stop |',
    ))
    return (
        f'{before}<!-- merge-status-switch:begin of=MERGE_REASONS -->\n\n{table}\n\n'
        f'<!-- merge-status-switch:end -->\n{after}'
    )


_SWITCH_AFTER_REASON_VOCAB = _switch_fixture(
    *_SWITCH_ROWS.values(),
    before=(
        '<!-- merge-state-vocab:begin partition=MERGE_REASONS\n'
        '     The guard checks this line, not the table below. -->\n'
        'The reasons are `probe_failed`, `journaled_not_live`, `prior_run_terminal`, '
        '`never_submitted`, `coalesce_train_id` and `landing_unattributable`.\n'
        '<!-- merge-state-vocab:end -->\n\n'
    ),
    after='\n| `probe_failed` | a later table outside the switch | **ABORT** |\n',
)


async def test_switch_in_maps_every_reason_to_its_action_class(alpha_guard: ModuleType) -> None:
    switch = _switch_in(alpha_guard, _switch_fixture(*_SWITCH_ROWS.values()), source=_FIXTURE_SKILL)

    assert switch == dict(_FIXTURE_SWITCH)


async def test_switch_in_reads_only_the_switch_span_beside_the_reason_vocabulary(
    alpha_guard: ModuleType,
) -> None:
    switch = _switch_in(alpha_guard, _SWITCH_AFTER_REASON_VOCAB, source=_FIXTURE_SKILL)

    assert switch[MergeReason.probe_failed] == 'WAIT'
    assert set(switch) == set(MergeReason)


@pytest.mark.parametrize(
    ('text', 'offender'),
    [
        pytest.param(
            _switch_fixture(*(row for reason, row in _SWITCH_ROWS.items() if reason != 'landing_unattributable')),
            'landing_unattributable',
            id='missing-row',
        ),
        pytest.param(
            _switch_fixture(*_SWITCH_ROWS.values(), _SWITCH_ROWS['probe_failed']),
            'probe_failed',
            id='duplicated-row',
        ),
    ],
)
async def test_switch_in_requires_exactly_one_row_per_reason(
    alpha_guard: ModuleType, text: str, offender: str,
) -> None:
    with pytest.raises(AssertionError) as excinfo:
        _switch_in(alpha_guard, text, source=_FIXTURE_SKILL)

    assert _FIXTURE_SKILL in str(excinfo.value)
    assert offender in str(excinfo.value)


_FIXTURE_ROUTES: Mapping[str, str] = MappingProxyType({
    'LIVE_STATES': 'WAIT',
    'WAIT': 'WAIT',
    'done': 'ACCEPT',
    'prior_run_terminal': 'REPORT',
})


@pytest.mark.parametrize(
    ('state', 'reason', 'expected'),
    [
        pytest.param(MergeState.stale_record, MergeReason.prior_run_terminal, 'REPORT', id='by-reason'),
        pytest.param(MergeState.journaled, MergeReason.journaled_not_live, 'WAIT', id='by-switch-class'),
        pytest.param(MergeState.queued, None, 'WAIT', id='by-live-partition'),
        pytest.param(MergeState.done, None, 'ACCEPT', id='by-state'),
    ],
)
async def test_resolve_routes_an_answer_through_exactly_one_key(
    state: MergeState, reason: MergeReason | None, expected: str,
) -> None:
    resolved = _resolve(_FIXTURE_ROUTES, _FIXTURE_SWITCH, state, reason, source=_FIXTURE_SKILL)

    assert resolved == expected


@pytest.mark.parametrize(
    ('routes', 'state', 'reason'),
    [
        pytest.param(_FIXTURE_ROUTES, MergeState.no_record, MergeReason.never_submitted, id='no-route'),
        pytest.param(
            {**_FIXTURE_ROUTES, 'journaled_not_live': 'ABORT'},
            MergeState.journaled,
            MergeReason.journaled_not_live,
            id='two-routes',
        ),
    ],
)
async def test_resolve_rejects_an_answer_routed_zero_or_two_times(
    routes: Mapping[str, str], state: MergeState, reason: MergeReason,
) -> None:
    with pytest.raises(AssertionError) as excinfo:
        _resolve(routes, _FIXTURE_SWITCH, state, reason, source=_FIXTURE_SKILL)

    message = str(excinfo.value)
    assert _FIXTURE_SKILL in message
    assert reason in message and _FIXTURE_SWITCH[reason] in message


async def test_branch_key_in_the_skills_full_ref_form_resolves_a_prior_run_row(
    world: _World,
) -> None:
    _unmerged_branch(world, '110')
    prior = await _finalize_in_prior_run(world, '110', _VERIFY_FAILED)

    response = await _poll(_restart(world), branch='task/110')

    assert response['state'] == MergeState.stale_record
    assert response['last_outcome'] is not None
    assert response['last_outcome']['request_id'] == prior.request_id


#: The _SCENES rows whose scene.key is {'task_id': ...}.
_TASK_ID_KEYED_ROWS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B8')


@pytest.mark.parametrize('row', sorted(_SCENES))
async def test_task_id_keyed_rows_names_every_scene_keyed_by_task_id(
    world: _World, row: str,
) -> None:
    scene = await _SCENES[row](world)

    assert ('task_id' in scene.key) == (row in _TASK_ID_KEYED_ROWS), (
        f'{row} polls by {sorted(scene.key)}: update _TASK_ID_KEYED_ROWS'
    )


@pytest.mark.parametrize('row', _TASK_ID_KEYED_ROWS)
async def test_a_branch_key_in_either_spelling_answers_what_the_task_id_key_answers(
    world: _World, row: str,
) -> None:
    scene = await _SCENES[row](world)
    task_id = scene.key['task_id']

    by_task_id = await _poll(scene.server, task_id=task_id)
    by_bare_branch = await _poll(scene.server, branch=task_id)
    by_full_ref = await _poll(scene.server, branch=f'task/{task_id}')

    assert by_bare_branch == by_task_id, f'{row}: branch={task_id!r} diverged from task_id'
    assert by_full_ref == by_task_id, f'{row}: branch=task/{task_id} diverged from task_id'


_SEED_MEMBER = 'seeded_drift'


@dataclass(frozen=True)
class _VocabularySeed:
    """A MergeState member added to *target*, by *assignment* ('' lands it by complement)."""

    target: str
    assignment: str


_SEEDS: Mapping[str, _VocabularySeed] = MappingProxyType({
    'epistemic_by_default': _VocabularySeed('EPISTEMIC_STATES', ''),
    'terminal': _VocabularySeed(
        'TERMINAL_STATES', f'TERMINAL_STATES = TERMINAL_STATES | {{MergeState.{_SEED_MEMBER}}}\n',
    ),
    'live': _VocabularySeed(
        'LIVE_STATES', f'LIVE_STATES = LIVE_STATES | {{MergeState.{_SEED_MEMBER}}}\n',
    ),
})

_MEMBER_ANCHOR = "    unknown = 'unknown'\n"
_PARTITION_ANCHOR = 'OUTCOME_STATES = LIVE_STATES | TERMINAL_STATES\n'


def _seeded_vocabulary(tmp_path: Path, seed: str) -> Path:
    """This tree's merge_state.py with a member added and assigned as *seed* says, written beside the test."""
    source = _REPO_ROOT / 'shared' / 'src' / 'shared' / 'merge_state.py'
    text = source.read_text(encoding='utf-8')
    for anchor in (_MEMBER_ANCHOR, _PARTITION_ANCHOR):
        assert text.count(anchor) == 1, f'seed anchor {anchor!r} must occur exactly once in {source}'
    text = text.replace(_MEMBER_ANCHOR, f"{_MEMBER_ANCHOR}    {_SEED_MEMBER} = '{_SEED_MEMBER}'\n")
    text = text.replace(_PARTITION_ANCHOR, _SEEDS[seed].assignment + _PARTITION_ANCHOR)
    seeded = tmp_path / f'merge_state_{seed}.py'
    seeded.write_text(text, encoding='utf-8')
    return seeded


def _failing_sites(guard: ModuleType) -> dict[str, str]:
    """Run alpha's per-site drift assertion over every pinned site; the failures, by site."""
    failures: dict[str, str] = {}
    for site in guard.PINNED_SITES:
        try:
            guard.test_every_span_matches_its_partition(site)
        except AssertionError as exc:
            failures[site] = str(exc)
    return failures


async def test_b9_control_alphas_guard_passes_on_the_unseeded_vocabulary() -> None:
    assert _failing_sites(_load_alpha_guard()) == {}


async def test_b9_a_member_added_without_updating_the_skill_lists_fails_alphas_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _load_alpha_guard()
    load_vocabulary = guard.load_vocabulary
    failed: set[str] = set()
    for seed, spec in _SEEDS.items():
        seeded = _seeded_vocabulary(tmp_path, seed)
        partitions = load_vocabulary(path=seeded)
        assert _SEED_MEMBER in partitions[spec.target], (seed, sorted(partitions[spec.target]))
        monkeypatch.setattr(guard, 'load_vocabulary', functools.partial(load_vocabulary, path=seeded))

        failures = _failing_sites(guard)

        expected = {
            site for site, (declared, _) in guard.PINNED_SITES.items()
            if any(_SEED_MEMBER in partitions[name] for name in declared)
        }
        assert expected, seed
        assert set(failures) == expected, (seed, sorted(failures), sorted(expected))
        assert all(_SEED_MEMBER in message for message in failures.values()), (seed, failures)
        failed |= set(failures)
    assert not _CONSUMER_SKILLS - failed, sorted(_CONSUMER_SKILLS - failed)
