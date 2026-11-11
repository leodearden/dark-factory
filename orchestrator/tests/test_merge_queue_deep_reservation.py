"""A deep chain's links are reserved for its tip verify (task 7720).

Live merger and verifier loops over a real tmp git repo. Head H verifies on
the remote 'head-host'; S, merged speculatively behind it, dispatches on
'local' and chains the followers buffered behind it. The contract these rows
pin is ``plans/deep-merge-ahead-prd.md`` (dispatch contract), implemented in
``merge_lane/chain_reservation.py`` and ``merge_lane/lookahead_hold.py``.

Every observation is public: ``snapshot()``, request futures, main's history,
the fakes' call ledgers and the event store.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_deep_scene import _permit_census, _spy_chain_lane_release
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    VerifyScript,
    fails,
    hangs_until,
    healthy_verify_clock,
    lane_scene_config,
    lane_state,
    make_lane,
    raises,
    running_lane,
)
from _orch_helpers import wait_responsive
from test_merge_lane_coalesce_admission import _TrainHooks

from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import MergeLane, MergeOutcome, MergeRequest, QueuedBranch
from orchestrator.merge_lane import worker as merge_queue

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')
_HEAD_HOST = 'head-host'
_POLL_SECS = 0.005


@dataclasses.dataclass
class _Scene:
    git_ops: GitOps
    config: OrchestratorConfig
    events: EventStore
    lane_releases: list[tuple]


def _scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    merge_deep: MergeDeepConfig, **overrides: Any,
) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    config = lane_scene_config(
        repo, _GIT, verify_host_policy='prefer_remote', verify_cross_check_remote_green=False,
        merge_deep=merge_deep, **overrides,
    )
    assert config.merge_deep.model_dump() == merge_deep.model_dump(), (
        'the operational yaml merged into the scene config: '
        f'{config.merge_deep!r} != {merge_deep!r}'
    )
    return _Scene(
        git_ops=GitOps(_GIT, repo),
        config=config,
        events=EventStore(db_path=tmp_path / 'events.db', run_id='deep-reservation'),
        lane_releases=_spy_chain_lane_release(monkeypatch),
    )


def _deep(chain_cap: int = 3, *, reservation_timeout: float = 1e9) -> MergeDeepConfig:
    return MergeDeepConfig(
        chain_cap=chain_cap, chain_build_timeout_secs=300,
        chain_reservation_timeout_secs=reservation_timeout,
    )


async def _request(scene: _Scene, name: str, **identity: Any) -> MergeRequest:
    worktree = (await scene.git_ops.create_worktree(name)).path
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await scene.git_ops.commit(worktree, f'{name}: own work')
    return MergeRequest(
        task_id=name,
        branch=QueuedBranch.parse(name, _GIT.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        **identity,
    )


async def _until(predicate: Callable[[], bool], *, label: str) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(_POLL_SECS)
    await wait_responsive(_poll(), label=label)


def _reservations(lane: MergeLane) -> list[dict[str, Any]]:
    return lane.snapshot()['chain_reservations']


async def _landed_task_files(scene: _Scene) -> list[str]:
    """The task files main's first-parent history added, oldest first."""
    _, log, _ = await _run(
        ['git', 'log', '--first-parent', '--reverse', '--format=%H', 'main'],
        cwd=scene.git_ops.project_root,
    )
    landed: list[str] = []
    for sha in log.split()[1:]:
        _, names, _ = await _run(
            ['git', 'diff', '--name-only', '--diff-filter=A', f'{sha}^1', sha],
            cwd=scene.git_ops.project_root,
        )
        landed.extend(n.removesuffix('.txt') for n in names.split() if n.endswith('.txt'))
    return landed


class _FirstVerifyScripted(FakeVerifier):
    """S's FIRST verify (the chain tip) follows its script; every later verify passes."""

    def __init__(self, first: dict[str, VerifyScript]) -> None:
        super().__init__()
        self._first = first

    def next_script(self, task_id: str | None) -> VerifyScript:
        scripted = self._first.pop(task_id or '', None)
        return scripted if scripted is not None else super().next_script(task_id)


@dataclasses.dataclass
class _Round:
    """A running lane mid-round: H verifies on 'head-host', P behind it on
    'p-host', and S, behind P, has dispatched on 'local' with its followers.

    P holds the lane's second speculation permit until it lands, so the
    merger cannot take a follower before S's dispatch snapshots it; once
    :meth:`land_head_and_p` lands H and P, the merger ticks freely while S's
    tip verify still runs.
    """

    lane: MergeLane
    queue: asyncio.Queue[Any]
    verifier: FakeVerifier
    reqs: dict[str, MergeRequest]
    release_head: asyncio.Event
    release_p: asyncio.Event
    remotes: list[FakeRemoteRunner]

    def put(self, *requests: MergeRequest) -> None:
        for request in requests:
            self.queue.put_nowait(request)

    async def land_head_and_p(self) -> None:
        """Land H, then P: P's verify ends only once H has landed."""
        for name, release in (('H', self.release_head), ('P', self.release_p)):
            release.set()
            outcome = await wait_responsive(self.reqs[name].result, label=f'{name} landed')
            assert outcome.status == 'done', (name, outcome)


@contextlib.asynccontextmanager
async def _deep_round(
    scene: _Scene,
    followers: tuple[str, ...],
    *,
    s_script: VerifyScript,
    extra: tuple[str, ...] = (),
    awaited: bool = False,
    **lane_kwargs: Any,
) -> AsyncIterator[_Round]:
    """Run a lane until S has dispatched with *followers* buffered behind it;
    S's first verify follows *s_script*."""
    names = ('H', 'P', 'S', *followers, *extra)
    reqs = {
        name: await _request(scene, name, awaited_by_workflow=awaited) for name in names
    }
    head_entered, p_entered = asyncio.Event(), asyncio.Event()
    release_head, release_p = asyncio.Event(), asyncio.Event()
    remotes = [
        FakeRemoteRunner(_HEAD_HOST, release=release_head, entered=head_entered),
        FakeRemoteRunner('p-host', release=release_p, entered=p_entered),
    ]
    verifier = _FirstVerifyScripted({'S': s_script})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=verifier, clock=healthy_verify_clock(),
        speculation_depth=2, remote_runners=remotes, event_store=scene.events,
        **lane_kwargs,
    )
    run_round = _Round(lane, queue, verifier, reqs, release_head, release_p, remotes)
    async with running_lane(lane):
        run_round.put(reqs['H'])
        await wait_responsive(head_entered.wait(), label='H verify under way on head-host')
        run_round.put(reqs['P'])
        await wait_responsive(p_entered.wait(), label='P verify under way on p-host')
        run_round.put(reqs['S'], *(reqs[name] for name in followers))
        yield run_round


async def _outcomes(run_round: _Round, names: tuple[str, ...]) -> dict[str, MergeOutcome]:
    return {
        name: await wait_responsive(run_round.reqs[name].result, label=f'{name} outcome')
        for name in names
    }


def _assert_no_live_permit(lane: MergeLane) -> None:
    census = _permit_census(lane)
    assert (census['spec_live'], census['cap_live']) == (frozenset(), frozenset()), census


def _verify_events_for(scene: _Scene, task_ids: set[str]) -> list[dict[str, Any]]:
    return [
        event for event in scene.events.fetch_events_by_type(EventType.merge_verify)
        if event.get('task_id') in task_ids
    ]


class TestAGreenTipLandsTheReservedPrefix:
    """A1: the links stay reserved for the whole tip verify, and land via the chain."""

    async def test_reserved_links_land_via_the_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        async with _deep_round(
            scene, ('F1', 'F2'), extra=('X',),
            s_script=hangs_until(release_s, entered=s_entered),
        ) as rnd:
            lane, reqs = rnd.lane, rnd.reqs
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")
            (row,) = _reservations(lane)
            assert (row['owner_task_id'], row['link_task_ids']) == ('S', ['F1', 'F2'])

            await rnd.land_head_and_p()
            rnd.put(reqs['X'])
            await _until(
                lambda: lane_state(lane, reqs['X'].request_id) not in (None, 'queued'),
                label='the merger took X while S verifies',
            )
            assert [r['link_task_ids'] for r in _reservations(lane)] == [['F1', 'F2']]
            for name in ('F1', 'F2'):
                assert lane_state(lane, reqs[name].request_id) == 'queued', name
                assert not reqs[name].result.done(), name

            release_s.set()
            outcomes = await _outcomes(rnd, ('S', 'F1', 'F2', 'X'))
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert [outcomes[n].landed_via_chain for n in ('S', 'F1', 'F2')] == [1, 1, 1]
            assert await _landed_task_files(scene) == ['H', 'P', 'S', 'F1', 'F2', 'X']
            assert rnd.verifier.verified.count('S') == 1
            assert not {'F1', 'F2'} & set(rnd.verifier.verified)
            assert _verify_events_for(scene, {'F1', 'F2'}) == []
            assert len(scene.lane_releases) == 1, scene.lane_releases
            assert _reservations(lane) == []
            assert lane.speculation_accounting_violations() == []
            _assert_no_live_permit(lane)


class TestACoalescingAbsorbsNoReservedLink:
    """A2: the coalescer never builds a train out of a chain's reserved links."""

    async def test_a_train_during_the_tip_verify_takes_only_unreserved_singles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Y0 is the merger's look-ahead pick once P lands; Y1 and Y2, still
        buffered beside the reserved F1 and F2, are what the coalescer may take."""
        scene = _scene(
            tmp_path, monkeypatch, merge_deep=_deep(), merge_train_coalesce_enabled=True,
        )
        hooks = _TrainHooks(parked={'Y1', 'Y2'})
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        async with _deep_round(
            scene, ('F1', 'F2'), extra=('Y0', 'Y1', 'Y2'), awaited=True,
            s_script=hangs_until(release_s, entered=s_entered),
            train_callback_factory=hooks.factory,
        ) as rnd:
            reqs = rnd.reqs
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")
            rnd.put(reqs['Y0'], reqs['Y1'], reqs['Y2'])
            await rnd.land_head_and_p()
            await _until(
                lambda: bool(scene.events.fetch_events_by_type(EventType.train_coalesced)),
                label='a train coalesced while S verifies',
            )
            trains = [
                event['data']['member_task_ids']
                for event in scene.events.fetch_events_by_type(EventType.train_coalesced)
            ]
            assert trains == [['Y1', 'Y2']]

            release_s.set()
            outcomes = await _outcomes(rnd, ('S', 'F1', 'F2'))
            assert [outcomes[n].landed_via_chain for n in ('S', 'F1', 'F2')] == [1, 1, 1]

    async def test_the_links_are_reserved_while_the_chain_builds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The reservation starts at the chain snapshot, provisional (no tip, no
        deadline) for the whole build, so no consumer can take a candidate
        between the snapshot and the tip verify."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        build_entered, release_build = asyncio.Event(), asyncio.Event()
        real_build = merge_queue.build_chain

        async def _gated_build(*args: Any, **kwargs: Any) -> Any:
            build_entered.set()
            await release_build.wait()
            return await real_build(*args, **kwargs)

        monkeypatch.setattr('orchestrator.merge_lane.worker.build_chain', _gated_build)
        release_s = asyncio.Event()
        async with _deep_round(scene, ('F1', 'F2'), s_script=hangs_until(release_s)) as rnd:
            await wait_responsive(build_entered.wait(), label="S's chain build under way")
            assert _reservations(rnd.lane) == [{
                'owner_task_id': 'S', 'link_task_ids': ['F1', 'F2'],
                'tip': None, 'seconds_left': None,
            }]
            release_build.set()
            await _until(
                lambda: bool(_reservations(rnd.lane)) and _reservations(rnd.lane)[0]['tip'],
                label='the reservation confirmed onto the built chain',
            )
            (row,) = _reservations(rnd.lane)
            assert row['link_task_ids'] == ['F1', 'F2']
            assert row['seconds_left'] is not None
            release_s.set()
            await rnd.land_head_and_p()
            outcomes = await _outcomes(rnd, ('S', 'F1', 'F2'))
            assert [outcomes[n].landed_via_chain for n in ('S', 'F1', 'F2')] == [1, 1, 1]
            assert _reservations(rnd.lane) == []



def _hang_then(release: asyncio.Event, entered: asyncio.Event, then: VerifyScript) -> VerifyScript:
    return dataclasses.replace(then, release=release, entered=entered)


class TestEAChainRoundThatDoesNotAdoptReleasesItsLinks:
    """E: a tip that fails, raises or is abandoned only releases the reservation.

    No outcome is rendered for a link by that round: each link later lands on
    its own, verified on its own, and nothing is blocked.
    """

    async def _assert_links_go_their_own_way(
        self, scene: _Scene, rnd: _Round, *, s_status: str = 'done',
    ) -> None:
        reqs = rnd.reqs
        await _until(lambda: _reservations(rnd.lane) == [], label='the reservation released')
        for name in ('F1', 'F2'):
            assert not reqs[name].result.done(), name
        await rnd.land_head_and_p()
        names = ('F1', 'F2') if s_status != 'done' else ('S', 'F1', 'F2')
        outcomes = await _outcomes(rnd, names)
        assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(names, 'done')
        assert {n: o.landed_via_chain for n, o in outcomes.items()} == dict.fromkeys(names)
        for name in ('F1', 'F2'):
            own = [e for e in _verify_events_for(scene, {name}) if e['data'].get('chain_items') == 1]
            assert own, f'{name} was never verified on its own'
        assert await _landed_task_files(scene) == ['H', 'P', *names]
        assert _reservations(rnd.lane) == []
        _assert_no_live_permit(rnd.lane)

    async def test_a_red_tip_releases_the_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        red = _hang_then(release_s, s_entered, fails(category='test', summary='red tip'))
        async with _deep_round(scene, ('F1', 'F2'), s_script=red) as rnd:
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")
            release_s.set()
            await self._assert_links_go_their_own_way(scene, rnd)

    async def test_a_raised_tip_releases_the_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        boom = _hang_then(release_s, s_entered, raises(RuntimeError('tip verify exploded')))
        async with _deep_round(scene, ('F1', 'F2'), s_script=boom) as rnd:
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")
            release_s.set()
            await self._assert_links_go_their_own_way(scene, rnd)

    async def test_an_abandoned_round_releases_the_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        s_entered = asyncio.Event()
        async with _deep_round(
            scene, ('F1', 'F2'), s_script=hangs_until(asyncio.Event(), entered=s_entered),
        ) as rnd:
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")
            rnd.reqs['S'].result.cancel()
            await self._assert_links_go_their_own_way(scene, rnd, s_status='abandoned')
