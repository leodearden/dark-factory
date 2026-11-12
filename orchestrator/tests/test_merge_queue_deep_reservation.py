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
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_deep_scene import _permit_census, _spy_chain_lane_release
from _merge_lane_fakes import (
    FakeClock,
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
from _orch_helpers import MERGE_GATE_BARRIER_TIMEOUT, MERGE_RESULT_TIMEOUT, wait_responsive
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


async def _until(
    predicate: Callable[[], bool], *, label: str, timeout: float = MERGE_RESULT_TIMEOUT,
) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(_POLL_SECS)
    await wait_responsive(_poll(), timeout=timeout, label=label)


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
    clock: FakeClock

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
    clock = healthy_verify_clock()
    lane = make_lane(
        scene.git_ops, queue, verifier=verifier, clock=clock,
        speculation_depth=2, remote_runners=remotes, event_store=scene.events,
        **lane_kwargs,
    )
    run_round = _Round(lane, queue, verifier, reqs, release_head, release_p, remotes, clock)
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
    """No landed item still holds a permit. The one speculation permit that may
    stay live is the merger's, retained for a late arrival behind the last
    item it queued (task 1862); its next dequeue transfers or releases it."""
    census = _permit_census(lane)
    held_by_merger = lane.snapshot()['speculation']['held_by_merger']
    assert (len(census['spec_live']), census['cap_live']) == (held_by_merger, frozenset()), census
    assert lane.speculation_accounting_violations() == []


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
        landed = await _landed_task_files(scene)
        assert (landed[:2], sorted(landed[2:])) == (['H', 'P'], sorted(names))
        assert _reservations(rnd.lane) == []
        assert rnd.lane.speculation_accounting_violations() == []

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


async def _tip_passed_and_landed_its_head(
    rnd: _Round, release_s: asyncio.Event, s_entered: asyncio.Event,
) -> str:
    """S's tip passes while H and P are held, and H lands on the tip's
    authority; P, still verifying on 'p-host', stands between the passed tip
    and its finalize. Returns the reservation's tip."""
    lane, reqs = rnd.lane, rnd.reqs
    await wait_responsive(
        s_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="S's tip verify under way",
    )
    release_s.set()
    outcome = await wait_responsive(reqs['H'].result, label="H landed on S's tip authority")
    assert outcome.status == 'done', outcome
    assert [r['link_task_ids'] for r in _reservations(lane)] == [['F1', 'F2']]
    assert not any(reqs[name].result.done() for name in ('F1', 'F2'))
    p_host = next(remote for remote in rnd.remotes if remote.name == 'p-host')
    assert not reqs['P'].result.done() and p_host.live, "P's verify is no longer live on p-host"
    (row,) = _reservations(lane)
    return row['tip']


class TestEAPassedTipTornDownByTheCascadeReleasesItsLinks:
    """E: a tip that passed, torn down by the head-failure cascade before its
    own finalize, ends its reservation; the old tip never reaches main.

    Once released, a link may be chained again by a later round of its own,
    so how F1 and F2 land is not pinned; that each lands once is.
    """

    async def test_an_abandoned_p_cascades_over_the_passed_tip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        async with _deep_round(
            scene, ('F1', 'F2'), s_script=hangs_until(release_s, entered=s_entered),
        ) as rnd:
            old_tip = await _tip_passed_and_landed_its_head(rnd, release_s, s_entered)
            rnd.reqs['P'].result.cancel()
            await _until(
                lambda: _reservations(rnd.lane) == [],
                label="the cascade ended the passed tip's reservation",
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
            )
            rnd.release_head.set()
            rnd.release_p.set()
            outcomes = await _outcomes(rnd, ('S', 'F1', 'F2'))
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert outcomes['S'].landed_via_chain is None
            landed = await _landed_task_files(scene)
            assert (landed[:1], sorted(landed[1:])) == (['H'], ['F1', 'F2', 'S'])
            assert not await _reaches_main(scene, old_tip)
            assert _reservations(rnd.lane) == []
            assert rnd.lane.speculation_accounting_violations() == []


async def _git_retrying_index_lock(repo: Path, *argv: str) -> None:
    """Run one git command, retrying while the live lane holds the repo's index lock."""
    async def _attempt_until_unlocked() -> None:
        while True:
            rc, _, err = await _run(['git', *argv], cwd=repo)
            if rc == 0:
                return
            assert 'index.lock' in err, err
            await asyncio.sleep(_POLL_SECS)
    await wait_responsive(
        _attempt_until_unlocked(),
        timeout=MERGE_GATE_BARRIER_TIMEOUT,
        label=f'the live lane released the index lock for git {argv[0]}',
    )


async def _move_main_by_a_foreign_commit(scene: _Scene) -> None:
    """Move main the way a human direct commit does, disjoint from every task
    file, which ``_landed_task_files`` reads as ``*.txt``. Its one file is
    staged alone: ``add -A`` would race the lane's live merge worktrees."""
    repo = scene.git_ops.project_root
    (repo / 'FOREIGN.md').write_text('landed elsewhere\n')
    await _git_retrying_index_lock(repo, 'add', 'FOREIGN.md')
    await _git_retrying_index_lock(repo, 'commit', '-m', 'Foreign landing')


def _gate_decided_reverify(scene: _Scene, task_id: str) -> bool:
    return any(
        event['task_id'] == task_id and event['data'].get('reverify') is True
        for event in scene.events.fetch_events_by_type(EventType.merge_reverify_gate)
    )


async def _reaches_main(scene: _Scene, sha: str) -> bool:
    rc, _, err = await _run(
        ['git', 'merge-base', '--is-ancestor', sha, 'main'], cwd=scene.git_ops.project_root,
    )
    assert rc in (0, 1), err
    return rc == 0


class TestEBPassedTipTornDownBehindAGateDispatchedHeadReleasesItsLinks:
    """E: a tip that passed, torn down when the head in front of it is gate
    re-verified as a dispatch of its rebased commit (task 8525), ends its
    reservation; nothing lands twice, and the old tip never reaches main.

    'head-host' stays held until the reservation is seen gone, so the rebased
    head or S's re-dispatch waits there and S cannot land first: S's own
    finalize, which ends any reservation S owns, is not what releases it.
    """

    async def test_a_gate_dispatched_rebased_p_tears_down_the_passed_tip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        async with _deep_round(
            scene, ('F1', 'F2'), s_script=hangs_until(release_s, entered=s_entered),
        ) as rnd:
            lane, reqs = rnd.lane, rnd.reqs
            old_tip = await _tip_passed_and_landed_its_head(rnd, release_s, s_entered)
            await _move_main_by_a_foreign_commit(scene)
            rnd.release_p.set()
            await _until(
                lambda: _gate_decided_reverify(scene, 'P'),
                label="P's gate decided a full re-verify of its rebased commit",
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
            )
            await _until(
                lambda: (
                    all(r['tip'] != old_tip for r in _reservations(lane))
                    and not reqs['S'].result.done()
                ),
                label="the gate's dispatch ended the passed tip's reservation before S landed",
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
            )
            rnd.release_head.set()
            names = ('H', 'P', 'S', 'F1', 'F2')
            outcomes = await _outcomes(rnd, names)
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(names, 'done')
            assert sorted(await _landed_task_files(scene)) == sorted(names)
            assert not await _reaches_main(scene, old_tip)
            assert _reservations(lane) == []
            assert lane.speculation_accounting_violations() == []
            assert lane.two_layer_invariants(await _main_sha(scene)) == []


class TestBTheReservationIsTimeLimited:
    """B: a confirmed reservation lasts merge_deep.chain_reservation_timeout_secs."""

    async def test_an_expired_reservation_hands_the_links_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """B1: on expiry the links go back with a WARNING while the tip verify
        continues; the passing tip then lands only what is still buffered, and
        nothing lands twice."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep(reservation_timeout=600))
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        caplog.set_level(logging.WARNING, logger='orchestrator.merge_lane')
        async with _deep_round(
            scene, ('F1', 'F2'), s_script=hangs_until(release_s, entered=s_entered),
        ) as rnd:
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")

            def _expiry_warned() -> bool:
                return any(
                    rec.name == 'orchestrator.merge_lane' and rec.levelno == logging.WARNING
                    and 'expired' in rec.getMessage()
                    and all(name in rec.getMessage() for name in ('S', 'F1', 'F2'))
                    for rec in caplog.records
                )

            await _until(_expiry_warned, label='the reservation expired with a WARNING')
            assert _reservations(rnd.lane) == []
            await rnd.land_head_and_p()
            await _until(
                lambda: bool(_verify_events_for(scene, {'F1'})),
                label="F1 verified by a round of its own, not by S's tip",
            )

            release_s.set()
            outcomes = await _outcomes(rnd, ('S', 'F1', 'F2'))
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            landed = await _landed_task_files(scene)
            assert sorted(landed) == ['F1', 'F2', 'H', 'P', 'S'], landed
            assert rnd.lane.speculation_accounting_violations() == []

    async def test_a_reservation_holds_before_its_deadline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """B2: an hour into a two-hour bound the row is still there, and its
        deadline was started by the confirm, on the lane's monotonic clock."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep(reservation_timeout=7200))
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        async with _deep_round(
            scene, ('F1', 'F2'), s_script=hangs_until(release_s, entered=s_entered),
        ) as rnd:
            before_build = rnd.clock.monotonic()
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")
            after_confirm = rnd.clock.monotonic()
            await _until(
                lambda: rnd.clock.monotonic() > after_confirm + 3600, label='an hour of lane time',
            )
            now = rnd.clock.monotonic()
            (row,) = _reservations(rnd.lane)
            assert row['link_task_ids'] == ['F1', 'F2']
            assert 7200 - (now - before_build) <= row['seconds_left'] <= 7200 - (now - after_confirm)
            release_s.set()
            await rnd.land_head_and_p()
            outcomes = await _outcomes(rnd, ('S', 'F1', 'F2'))
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert sorted(await _landed_task_files(scene)) == ['F1', 'F2', 'H', 'P', 'S']

    async def test_expiry_survives_a_failing_heartbeat_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """B3: a heartbeat whose event write raises on every tick still expires
        the reservation; the expiry does not depend on the log step."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep(reservation_timeout=600))
        events = _HeartbeatWriteFails(db_path=tmp_path / 'events-hb.db', run_id='hb')
        scene.events = events
        release_s, s_entered = asyncio.Event(), asyncio.Event()
        caplog.set_level(logging.WARNING, logger='orchestrator.merge_lane')
        async with _deep_round(
            scene, ('F1', 'F2'), s_script=hangs_until(release_s, entered=s_entered),
        ) as rnd:
            await wait_responsive(s_entered.wait(), label="S's tip verify under way")
            await _until(
                lambda: events.failed_writes > 0, label='a heartbeat write failed',
            )
            await _until(lambda: _reservations(rnd.lane) == [], label='the reservation expired')
            release_s.set()
            await rnd.land_head_and_p()
            outcomes = await _outcomes(rnd, ('S', 'F1', 'F2'))
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
        assert any(
            'expired' in rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING
        )


class _HeartbeatWriteFails(EventStore):
    """An event store whose merge-heartbeat write raises, as a locked database would."""

    failed_writes = 0

    def emit(self, event_type: EventType, **fields: Any) -> None:
        if event_type is EventType.merge_heartbeat:
            self.failed_writes += 1
            raise sqlite3.OperationalError('database is locked')
        super().emit(event_type, **fields)


@dataclasses.dataclass
class _Lane:
    lane: MergeLane
    queue: asyncio.Queue[Any]
    clock: FakeClock

    def put(self, *requests: MergeRequest) -> None:
        for request in requests:
            self.queue.put_nowait(request)


@contextlib.asynccontextmanager
async def _running(
    scene: _Scene, *, verifier: FakeVerifier, remotes: list[FakeRemoteRunner],
    speculation_depth: int, stop_timeout: float = 60.0,
) -> AsyncIterator[_Lane]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    clock = healthy_verify_clock()
    lane = make_lane(
        scene.git_ops, queue, verifier=verifier, clock=clock,
        speculation_depth=speculation_depth, remote_runners=remotes,
        event_store=scene.events,
    )
    async with running_lane(lane, stop_timeout=stop_timeout):
        yield _Lane(lane, queue, clock)


async def _requests(scene: _Scene, *names: str) -> dict[str, MergeRequest]:
    return {name: await _request(scene, name) for name in names}


async def _first_parent(scene: _Scene, sha: str) -> str:
    _, parent, _ = await _run(
        ['git', 'rev-parse', f'{sha}^1'], cwd=scene.git_ops.project_root,
    )
    return parent.strip()


async def _lane_time_passes(run: _Lane, secs: float) -> None:
    """Let the running lane's loops carry its clock *secs* on."""
    start = run.clock.monotonic()
    await _until(lambda: run.clock.monotonic() > start + secs, label=f'{secs}s of lane time')


_AWAITING_DISPATCH = ('awaiting_verify', 'awaiting_host')


class TestDTheLookAheadWaitsForTheChainDecision:
    """D: the merger's look-ahead waits until S's dispatch has decided its chain."""

    async def test_a_free_permit_does_not_take_the_followers_before_s_dispatches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """D1: speculation_depth 3 leaves a permit free after S, yet S1 and
        S2 stay buffered until S's dispatch has chained them."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'H', 'S', 'S1', 'S2')
        head_entered, release_head = asyncio.Event(), asyncio.Event()
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        verifier = _FirstVerifyScripted({'S': hangs_until(release_s, entered=s_entered)})
        remotes = [FakeRemoteRunner(_HEAD_HOST, release=release_head, entered=head_entered)]
        async with _running(
            scene, verifier=verifier, remotes=remotes, speculation_depth=3,
        ) as run:
            run.put(reqs['H'])
            await wait_responsive(head_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on head-host')
            run.put(reqs['S'], reqs['S1'], reqs['S2'])
            await wait_responsive(s_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="S's tip verify under way")
            rows = [(r['owner_task_id'], r['link_task_ids']) for r in _reservations(run.lane)]
            assert rows == [('S', ['S1', 'S2'])]
            assert run.lane.unfrozen_suffix() == (reqs['S1'].request_id, reqs['S2'].request_id)

            release_head.set()
            release_s.set()
            outcomes = {
                name: await wait_responsive(reqs[name].result, label=f'{name} outcome')
                for name in ('H', 'S', 'S1', 'S2')
            }
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert [outcomes[n].landed_via_chain for n in ('S', 'S1', 'S2')] == [1, 1, 1]
            assert verifier.verified == ['S']
            assert await _landed_task_files(scene) == ['H', 'S', 'S1', 'S2']
            _assert_no_live_permit(run.lane)

    async def test_without_chains_the_look_ahead_is_not_held(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """D2: at chain_cap 0 the look-ahead merges S1 at once, and S1 is
        verified on its own while S's verify still runs."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep(chain_cap=0))
        reqs = await _requests(scene, 'H', 'S', 'S1', 'S2')
        head_entered, release_head = asyncio.Event(), asyncio.Event()
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier()
        remotes = [
            FakeRemoteRunner(_HEAD_HOST, release=release_head, entered=head_entered),
            FakeRemoteRunner('s-host', release=release_s, entered=s_entered),
        ]
        async with _running(
            scene, verifier=verifier, remotes=remotes, speculation_depth=3,
        ) as run:
            run.put(reqs['H'])
            await wait_responsive(head_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on head-host')
            run.put(reqs['S'], reqs['S1'], reqs['S2'])
            await wait_responsive(s_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="S's verify under way on s-host")
            await wait_responsive(verifier.await_entry(1), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='S1 verified on local')
            assert verifier.verified == ['S1']
            assert not release_s.is_set()
            assert _reservations(run.lane) == []

            release_head.set()
            release_s.set()
            outcomes = {
                name: await wait_responsive(reqs[name].result, label=f'{name} outcome')
                for name in ('H', 'S', 'S1', 'S2')
            }
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert {n: o.landed_via_chain for n, o in outcomes.items()} == dict.fromkeys(outcomes)
            assert await _landed_task_files(scene) == ['H', 'S', 'S1', 'S2']
            assert _reservations(run.lane) == []

    async def test_with_no_free_host_the_follower_waits_for_the_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """D3: S waits for a host and the merger waits for S; S1 is not lost,
        and S chains it once H's verify frees the host."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'H', 'S', 'S1')
        h_entered, release_h = asyncio.Event(), asyncio.Event()
        verifier = _FirstVerifyScripted({'H': hangs_until(release_h, entered=h_entered)})
        async with _running(scene, verifier=verifier, remotes=[], speculation_depth=2) as run:
            run.put(reqs['H'])
            await wait_responsive(h_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on local')
            run.put(reqs['S'], reqs['S1'])
            await _until(
                lambda: lane_state(run.lane, reqs['S'].request_id) in _AWAITING_DISPATCH,
                label='S merged and waiting for a host',
            )
            await _lane_time_passes(run, 300)
            assert lane_state(run.lane, reqs['S'].request_id) in _AWAITING_DISPATCH
            assert run.lane.unfrozen_suffix() == (reqs['S1'].request_id,)
            assert lane_state(run.lane, reqs['S1'].request_id) == 'queued'

            release_h.set()
            outcomes = {
                name: await wait_responsive(reqs[name].result, label=f'{name} outcome')
                for name in ('H', 'S', 'S1')
            }
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert [outcomes[n].landed_via_chain for n in ('S', 'S1')] == [1, 1]
            assert verifier.verified == ['H', 'S']
            assert await _landed_task_files(scene) == ['H', 'S', 'S1']

    async def test_arrivals_while_s_waits_for_a_host_are_chained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """D5: F1 and F2 arrive after S is merged and before H frees the one
        host; the held merger buffers them, and S's dispatch chains them."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'H', 'S', 'F1', 'F2')
        h_entered, release_h = asyncio.Event(), asyncio.Event()
        verifier = _FirstVerifyScripted({'H': hangs_until(release_h, entered=h_entered)})
        async with _running(scene, verifier=verifier, remotes=[], speculation_depth=2) as run:
            run.put(reqs['H'])
            await wait_responsive(h_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on local')
            run.put(reqs['S'])
            await _until(
                lambda: lane_state(run.lane, reqs['S'].request_id) in _AWAITING_DISPATCH,
                label='S merged and waiting for a host',
            )
            run.put(reqs['F1'], reqs['F2'])
            followers = (reqs['F1'].request_id, reqs['F2'].request_id)
            await _until(
                lambda: run.lane.unfrozen_suffix() == followers,
                label='the held merger buffered the arrivals',
            )
            assert lane_state(run.lane, reqs['S'].request_id) in _AWAITING_DISPATCH

            release_h.set()
            outcomes = {
                name: await wait_responsive(reqs[name].result, label=f'{name} outcome')
                for name in ('H', 'S', 'F1', 'F2')
            }
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert [outcomes[n].landed_via_chain for n in ('S', 'F1', 'F2')] == [1, 1, 1]
            assert verifier.verified == ['H', 'S']
            assert await _landed_task_files(scene) == ['H', 'S', 'F1', 'F2']

    async def test_a_declined_chain_releases_the_merger_at_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """D4: with nothing buffered behind S its dispatch builds no chain, the
        merger goes on at once, and a late arrival still attaches behind S."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'H', 'S', 'L')
        head_entered, release_head = asyncio.Event(), asyncio.Event()
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        l_entered = asyncio.Event()
        s_host = FakeRemoteRunner('s-host', release=release_s, entered=s_entered)
        l_host = FakeRemoteRunner('l-host', entered=l_entered)
        remotes = [
            FakeRemoteRunner(_HEAD_HOST, release=release_head, entered=head_entered),
            s_host, l_host,
        ]
        async with _running(
            scene, verifier=FakeVerifier(), remotes=remotes, speculation_depth=2,
        ) as run:
            run.put(reqs['H'])
            await wait_responsive(head_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on head-host')
            run.put(reqs['S'])
            await wait_responsive(s_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="S's verify under way on s-host")
            assert _reservations(run.lane) == []
            run.put(reqs['L'])
            await wait_responsive(l_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="L's verify under way on l-host")
            assert await _first_parent(scene, l_host.calls[0]) == s_host.calls[0]

            release_head.set()
            release_s.set()
            outcomes = {
                name: await wait_responsive(reqs[name].result, label=f'{name} outcome')
                for name in ('H', 'S', 'L')
            }
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert await _landed_task_files(scene) == ['H', 'S', 'L']
            assert len(l_host.calls) == 1

    async def test_stop_releases_a_held_merger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """D4: a merger held on S's chain decision does not hold up the lane's stop."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'H', 'S', 'S1')
        h_entered = asyncio.Event()
        verifier = _FirstVerifyScripted({'H': hangs_until(asyncio.Event(), entered=h_entered)})
        async with _running(
            scene, verifier=verifier, remotes=[], speculation_depth=2, stop_timeout=15.0,
        ) as run:
            run.put(reqs['H'])
            await wait_responsive(h_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on local')
            run.put(reqs['S'], reqs['S1'])
            await _until(
                lambda: lane_state(run.lane, reqs['S'].request_id) in _AWAITING_DISPATCH,
                label='S merged and waiting for a host',
            )
            await _lane_time_passes(run, 300)
            assert lane_state(run.lane, reqs['S1'].request_id) == 'queued'
        assert all(reqs[name].result.done() for name in ('H', 'S', 'S1'))


async def _main_sha(scene: _Scene) -> str:
    _, sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=scene.git_ops.project_root)
    return sha.strip()


class TestCAnItemBehindTheChainStacksOnTheChainTip:
    """C: the look-ahead skips the reserved links and stacks the next item on the chain tip."""

    async def test_an_item_behind_the_chain_verifies_on_a_free_host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """C1: X goes to the free 'spare' host while S's tip still verifies, merged
        onto the chain tip, so it lands right after the chain, verified once."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'H', 'S', 'F1', 'F2', 'X')
        head_entered, release_head = asyncio.Event(), asyncio.Event()
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        x_entered = asyncio.Event()
        s_host = FakeRemoteRunner('s-host', release=release_s, entered=s_entered)
        spare = FakeRemoteRunner('spare', entered=x_entered)
        remotes = [
            FakeRemoteRunner(_HEAD_HOST, release=release_head, entered=head_entered),
            s_host, spare,
        ]
        verifier = FakeVerifier()
        async with _running(
            scene, verifier=verifier, remotes=remotes, speculation_depth=3,
        ) as run:
            run.put(reqs['H'])
            await wait_responsive(head_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on head-host')
            run.put(reqs['S'], reqs['F1'], reqs['F2'])
            await wait_responsive(s_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="S's tip verify under way on s-host")
            (row,) = _reservations(run.lane)
            assert (row['owner_task_id'], row['link_task_ids']) == ('S', ['F1', 'F2'])
            assert s_host.calls == [row['tip']]

            run.put(reqs['X'])
            await wait_responsive(x_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="X's verify under way on spare")
            assert not release_s.is_set()
            assert await _first_parent(scene, spare.calls[0]) == row['tip']
            assert run.lane.check_frozen_prefix_invariant(await _main_sha(scene)) == []
            assert run.lane.verify_base_divergences() == ()

            release_s.set()
            release_head.set()
            outcomes = {
                name: await wait_responsive(reqs[name].result, label=f'{name} outcome')
                for name in ('H', 'S', 'F1', 'F2', 'X')
            }
            assert {n: o.status for n, o in outcomes.items()} == dict.fromkeys(outcomes, 'done')
            assert [outcomes[n].landed_via_chain for n in ('S', 'F1', 'F2')] == [1, 1, 1]
            assert await _landed_task_files(scene) == ['H', 'S', 'F1', 'F2', 'X']
            assert len(spare.calls) == 1
            assert verifier.verified == []
            _assert_no_live_permit(run.lane)


class TestTheMergerStacksOnWhatTheDispatchVerifies:
    """The decision the merger waits for is what S's successors stack on, as
    S's dispatch left it: a dispatch-time re-merge replaces S's commit."""

    async def test_a_re_merged_s_is_what_the_merger_stacks_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """H fails while S waits for the one host; S is re-merged onto main at
        its dispatch, and the merger's pending base is that re-merge, not S's
        first merge on H's dead commit."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'H', 'S')
        h_entered, release_h = asyncio.Event(), asyncio.Event()
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        verifier = _FirstVerifyScripted({
            'H': _hang_then(release_h, h_entered, fails(category='test', summary='red head')),
            'S': hangs_until(release_s, entered=s_entered),
        })
        async with _running(scene, verifier=verifier, remotes=[], speculation_depth=2) as run:
            run.put(reqs['H'])
            await wait_responsive(h_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='H verify under way on local')
            run.put(reqs['S'])
            await _until(
                lambda: lane_state(run.lane, reqs['S'].request_id) in _AWAITING_DISPATCH,
                label='S merged on H and waiting for the host',
            )
            release_h.set()
            assert (await wait_responsive(reqs['H'].result, label='H outcome')).status == 'blocked'
            await wait_responsive(s_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label="S's re-merge verifying on local")

            def _pending() -> dict[str, Any]:
                return run.lane.snapshot()['speculation']

            await _until(
                lambda: _pending()['pending_predecessor_task_id'] == 'S',
                label='the merger waiting for a late arrival behind S',
            )
            pending_base = _pending()['pending_spec_base']
            assert await _first_parent(scene, pending_base) == await _main_sha(scene)
            release_s.set()
            outcome = await wait_responsive(reqs['S'].result, label='S outcome')
            assert outcome.status == 'done', outcome
            assert await _landed_task_files(scene) == ['S']
