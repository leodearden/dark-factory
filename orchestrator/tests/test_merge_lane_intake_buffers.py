"""The merger-side intake: lane buffers, the redispatch park and the suffix state.

PRD ``plans/merge-lane-quality-prd.md`` task ι, partitioned by esc-5305-1.
``MergeIntake`` is built here through its public constructor with the ports the
worker's composition root hands it: a lifecycle registry, the lane ``HaltState``,
the admission-halt predicate and re-reading git/frozen-prefix callables. Every
case observes only what that surface reports back: the buffers, the park, the
registry the port recorded into, request futures and stored events.

Boundary row 4b then drives the worker's coalesce-train pass, which now reads
the lane buffers only through ``MergeIntake``, on a real repository.
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import README_SEED, seed_repo
from _merge_lane_fakes import lane_scene_config, make_lane, running_lane
from test_merge_lane_coalesce_admission import _TrainHooks

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.halt import HaltState
from orchestrator.merge_lane.intake_buffers import IntakeLifecycle, MergeIntake
from orchestrator.merge_lane.types import (
    DecidedItem,
    ItemLifecycleState,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
)
from orchestrator.suffix_graph import EMPTY_SUFFIX_CONFLICT_GRAPH

pytestmark = pytest.mark.asyncio

LANES = ('high', 'normal')
_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
State = ItemLifecycleState


class RecordingLifecycle:
    """The lifecycle port over a plain registry, recording every hop it was asked for."""

    def __init__(self) -> None:
        self.states: dict[str, ItemLifecycleState] = {}
        self.live: dict[str, MergeRequest] = {}
        self.hops: list[tuple[str, ItemLifecycleState, ItemLifecycleState]] = []

    def current(self, request_id: str) -> ItemLifecycleState | None:
        return self.states.get(request_id)

    def register(self, request: MergeRequest, initial: ItemLifecycleState) -> str:
        assert request.request_id not in self.states, request.request_id
        self.states[request.request_id] = initial
        self.live[request.request_id] = request
        return request.request_id

    def advance_if_at(
        self, request_id: str, from_state: ItemLifecycleState, to_state: ItemLifecycleState,
        *, live_obj: MergeRequest | None = None,
    ) -> bool:
        if self.states.get(request_id) != from_state:
            return False
        self.note_transition(request_id, from_state, to_state, live_obj=live_obj)
        return True

    def note_transition(
        self, request_id: str, from_state: ItemLifecycleState, to_state: ItemLifecycleState,
        *, live_obj: MergeRequest | None = None,
    ) -> None:
        self.hops.append((request_id, from_state, to_state))
        self.states[request_id] = to_state
        if live_obj is not None:
            self.live[request_id] = live_obj

    def live_request(self, request_id: str) -> MergeRequest | None:
        return self.live.get(request_id)

    def retire(self, request_id: str) -> None:
        current = self.states.get(request_id)
        if current is not None and current != ItemLifecycleState.TERMINAL:
            self.note_transition(request_id, current, ItemLifecycleState.TERMINAL)
        self.live.pop(request_id, None)

    def port(self) -> IntakeLifecycle:
        return IntakeLifecycle(
            current=self.current,
            register=self.register,
            advance_if_at=self.advance_if_at,
            note_transition=self.note_transition,
            live_request=self.live_request,
        )


@dataclasses.dataclass
class Scene:
    intake: MergeIntake
    queue: asyncio.Queue[Any]
    lifecycle: RecordingLifecycle
    halt: HaltState
    events: EventStore
    config: OrchestratorConfig
    admission: list[bool]
    reserved: set[str]


def any_writer(structure: str) -> None:
    """The owner port of a worker with no merger task yet: every writer may write."""


def refuse_writer(structure: str) -> None:
    """The owner port when the caller is not the running merger task."""
    raise AssertionError(f'{structure} mutated from non-owner coroutine')


@pytest.fixture
def scene(tmp_path: Path) -> Scene:
    return build_scene(tmp_path, assert_merger_owns=any_writer)


def build_scene(
    tmp_path: Path, *, assert_merger_owns: Callable[[str], None], repo: Path | None = None,
) -> Scene:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lifecycle = RecordingLifecycle()
    halt = HaltState(LANES, on_resume=lambda: None)
    events = EventStore(db_path=tmp_path / 'events.db', run_id='intake-buffers')
    root = repo or tmp_path
    git_ops = GitOps(_GIT, root)
    admission = [False]
    reserved: set[str] = set()
    intake = MergeIntake(
        queue=queue,
        lanes=LANES,
        lifecycle=lifecycle.port(),
        halt=halt,
        admission_halted=lambda: admission[0],
        merge_store=None,
        event_store=events,
        escalation_queue=None,
        git_ops=lambda: git_ops,
        frozen_requests=lambda: (),
        frozen_prefix_tip=lambda sha: sha,
        retire_item=lifecycle.retire,
        assert_merger_owns=assert_merger_owns,
        is_reserved=lambda request_id: request_id in reserved,
    )
    return Scene(
        intake, queue, lifecycle, halt, events,
        OrchestratorConfig(project_root=root), admission, reserved,
    )


def request(scene: Scene, task_id: str, *, lane: str = 'normal', **identity: Any) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, _GIT.branch_prefix),
        worktree=scene.config.project_root,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        lane='high' if lane == 'high' else 'normal',
        **identity,
    )


def parked(req: MergeRequest) -> DecidedItem:
    return DecidedItem(
        request=req, immediate_outcome=MergeOutcome('done'), base_sha='b' * 40,
        speculative=False,
    )


def coalesced_sources(scene: Scene) -> list[str]:
    return [
        event['data']['source']
        for event in scene.events.fetch_events_by_type(EventType.merge_coalesced)
    ]


def task_ids(requests: tuple[MergeRequest, ...]) -> list[str]:
    return [req.task_id for req in requests]


# ---------------------------------------------------------------------------
# (a)-(c) Buffering, the non-blocking drain and the pick
# ---------------------------------------------------------------------------


async def test_buffering_a_fresh_request_registers_it_and_parks_it_in_its_lane(
    scene: Scene,
) -> None:
    req = request(scene, '101')

    scene.intake.buffer_owned_request(req)

    assert scene.lifecycle.hops == [(req.request_id, State.QUEUED, State.LANE_BUFFERED)]
    assert scene.lifecycle.current(req.request_id) is State.LANE_BUFFERED
    assert scene.intake.buffered('normal') == (req,)
    assert scene.intake.buffered('high') == ()


async def test_the_drain_buffers_every_queued_request_and_reports_the_sentinel(
    scene: Scene,
) -> None:
    first, second = request(scene, '101'), request(scene, '102', lane='high')
    scene.queue.put_nowait(first)
    scene.queue.put_nowait(second)

    assert scene.intake.drain_into_lanes() is False
    assert scene.intake.buffered('normal') == (first,)
    assert scene.intake.buffered('high') == (second,)

    third, after = request(scene, '103'), request(scene, '104')
    for item in (third, None, after):
        scene.queue.put_nowait(item)

    assert scene.intake.drain_into_lanes() is True
    assert scene.intake.buffered('normal') == (first, third)
    assert scene.queue.get_nowait() is after


async def test_the_pick_prefers_high_and_moves_the_request_to_merging(scene: Scene) -> None:
    normal, high = request(scene, '101'), request(scene, '102', lane='high')
    scene.intake.buffer_owned_request(normal)
    scene.intake.buffer_owned_request(high)

    assert scene.intake.pop_next_pickable() is high
    assert scene.intake.pop_next_pickable() is normal
    assert scene.intake.pop_next_pickable() is None
    assert scene.lifecycle.hops[-2:] == [
        (high.request_id, State.LANE_BUFFERED, State.MERGING),
        (normal.request_id, State.LANE_BUFFERED, State.MERGING),
    ]


async def test_the_pick_skips_a_halted_lane(scene: Scene) -> None:
    normal, high = request(scene, '101'), request(scene, '102', lane='high')
    scene.intake.buffer_owned_request(normal)
    scene.intake.buffer_owned_request(high)
    scene.halt.halt_lane('high', 'test halt')

    assert scene.intake.pop_next_pickable() is normal
    assert scene.intake.pop_next_pickable() is None
    assert scene.intake.buffered('high') == (high,)


async def test_the_pick_holds_everything_while_admission_is_halted(scene: Scene) -> None:
    req = request(scene, '101')
    scene.intake.buffer_owned_request(req)
    scene.admission[0] = True

    assert scene.intake.pop_next_pickable() is None
    assert scene.intake.buffered('normal') == (req,)

    scene.admission[0] = False
    assert scene.intake.pop_next_pickable() is req


# ---------------------------------------------------------------------------
# (c2) A deep chain's reserved links are invisible to the pick (task 7720)
# ---------------------------------------------------------------------------


async def test_the_pick_skips_a_reserved_head_and_leaves_it_buffered(scene: Scene) -> None:
    head, behind = request(scene, '101'), request(scene, '102')
    scene.intake.buffer_owned_request(head)
    scene.intake.buffer_owned_request(behind)
    scene.reserved.add(head.request_id)

    assert scene.intake.pop_next_pickable() is behind
    assert scene.intake.pop_next_pickable() is None
    assert scene.intake.buffered('normal') == (head,)
    assert scene.lifecycle.current(head.request_id) is State.LANE_BUFFERED
    assert (head.request_id, State.LANE_BUFFERED, State.MERGING) not in scene.lifecycle.hops


async def test_a_lane_of_only_reserved_requests_falls_through(scene: Scene) -> None:
    high, normal = request(scene, '101', lane='high'), request(scene, '102')
    scene.intake.buffer_owned_request(high)
    scene.intake.buffer_owned_request(normal)
    scene.reserved.add(high.request_id)

    assert scene.intake.pop_next_pickable() is normal
    assert scene.intake.pop_next_pickable() is None
    assert scene.intake.buffered('high') == (high,)


async def test_a_reserved_older_neighbour_does_not_block_the_item_behind_it(
    tmp_path: Path,
) -> None:
    repo = seed_repo(tmp_path / 'repo', README_SEED)
    scene = build_scene(tmp_path, assert_merger_owns=any_writer, repo=repo)
    git_ops = GitOps(_GIT, repo)
    younger = request(scene, '101', merge_first_enqueued_at=300.0)
    older = request(scene, '102', merge_first_enqueued_at=100.0)
    for req in (younger, older):
        worktree = (await git_ops.create_worktree(req.task_id)).path
        (worktree / 'clique.py').write_text(f'{req.task_id}\n')
        await git_ops.commit(worktree, f'{req.task_id}: clique')
        scene.intake.buffer_owned_request(req)
    await scene.intake.recompute_suffix_conflict_graph()
    graph = scene.intake.suffix_conflict_graph
    assert graph.footprint_neighbors(younger.request_id) == {older.request_id}

    scene.reserved.add(older.request_id)

    assert scene.intake.pop_next_pickable() is younger
    assert scene.intake.buffered('normal') == (older,)


# ---------------------------------------------------------------------------
# (d) A re-entrant drain of a request already past QUEUED
# ---------------------------------------------------------------------------


async def test_a_distinct_twin_of_a_live_request_is_coalesced_as_already_merged(
    scene: Scene,
) -> None:
    original = request(scene, '101')
    scene.lifecycle.register(original, State.VERIFYING)
    twin = request(scene, '101', request_id=original.request_id)

    scene.intake.buffer_owned_request(twin)

    assert twin.result.result().status == 'already_merged'
    assert not original.result.done()
    assert scene.intake.buffered('normal') == ()
    assert scene.lifecycle.current(original.request_id) is State.VERIFYING
    assert coalesced_sources(scene) == ['duplicate_submission']


async def test_a_redrain_of_the_live_original_leaves_its_future_pending(
    scene: Scene,
) -> None:
    original = request(scene, '101')
    scene.lifecycle.register(original, State.VERIFYING)

    scene.intake.buffer_owned_request(original)

    assert not original.result.done()
    assert scene.intake.buffered('normal') == ()
    assert scene.lifecycle.hops == []
    assert coalesced_sources(scene) == ['live_original_redrain']


# ---------------------------------------------------------------------------
# (e) The redispatch park
# ---------------------------------------------------------------------------


async def test_the_redispatch_park_pops_front_requests_first(scene: Scene) -> None:
    back, front = parked(request(scene, '101')), parked(request(scene, '102'))
    assert scene.intake.has_pending_redispatch is False

    scene.intake.request_redispatch(back)
    scene.intake.request_redispatch(front, front=True)

    assert scene.intake.has_pending_redispatch is True
    assert scene.intake.pending_redispatch() == (front, back)
    assert scene.intake.next_redispatch() is front
    assert scene.intake.next_redispatch() is back
    assert scene.intake.has_pending_redispatch is False


# ---------------------------------------------------------------------------
# (f) The lane-buffer requests the other concerns make
# ---------------------------------------------------------------------------


async def test_a_taken_request_is_restored_to_its_exact_position(scene: Scene) -> None:
    reqs = [request(scene, task_id) for task_id in ('101', '102', '103')]
    for req in reqs:
        scene.intake.buffer_owned_request(req)

    taken = scene.intake.take_buffered('102')

    assert taken == ('normal', 1, reqs[1])
    assert task_ids(scene.intake.buffered('normal')) == ['101', '103']
    assert scene.intake.take_buffered('102') is None

    assert taken is not None
    lane, index, req = taken
    scene.intake.restore_buffered(lane, index, req)
    assert scene.intake.buffered('normal') == tuple(reqs)


async def test_removing_buffered_requests_drops_exactly_those(scene: Scene) -> None:
    normal = [request(scene, task_id) for task_id in ('101', '102')]
    high = request(scene, '103', lane='high')
    for req in (*normal, high):
        scene.intake.buffer_owned_request(req)

    scene.intake.remove_buffered({normal[0].request_id, high.request_id})

    assert scene.intake.buffered('normal') == (normal[1],)
    assert scene.intake.buffered('high') == ()


async def test_replacing_a_lane_rebuilds_it_in_the_given_order(scene: Scene) -> None:
    first, second = request(scene, '101'), request(scene, '102')
    scene.intake.buffer_owned_request(first)

    scene.intake.replace_lane('normal', [second, first])

    assert scene.intake.buffered('normal') == (second, first)
    scene.intake.replace_lane('normal', [])
    assert scene.intake.buffered('normal') == ()


async def test_a_lane_replacement_off_the_merger_task_is_refused(tmp_path: Path) -> None:
    scene = build_scene(tmp_path, assert_merger_owns=refuse_writer)

    with pytest.raises(AssertionError, match='_lane_buffers'):
        scene.intake.replace_lane('normal', [request(scene, '101')])

    assert scene.intake.buffered('normal') == ()


async def test_the_shutdown_drain_empties_every_lane_in_pick_order(scene: Scene) -> None:
    normal, high = request(scene, '101'), request(scene, '102', lane='high')
    scene.intake.buffer_owned_request(normal)
    scene.intake.buffer_owned_request(high)

    assert scene.intake.drain_lanes_for_shutdown() == (high, normal)
    assert scene.intake.buffered('high') == scene.intake.buffered('normal') == ()


# ---------------------------------------------------------------------------
# (g) The suffix state
# ---------------------------------------------------------------------------


async def test_the_suffix_state_starts_empty_and_remembers_main(scene: Scene) -> None:
    assert scene.intake.suffix_conflict_graph is EMPTY_SUFFIX_CONFLICT_GRAPH
    assert scene.intake.last_known_main_sha is None

    scene.intake.note_main_sha('a' * 40)

    assert scene.intake.last_known_main_sha == 'a' * 40


# ---------------------------------------------------------------------------
# Boundary row 4b: two waiting singles coalesce into one train
# ---------------------------------------------------------------------------


async def test_two_awaited_singles_coalesce_into_one_train_through_the_intake(
    tmp_path: Path,
) -> None:
    """Built on ``test_merge_lane_coalesce_admission.py``'s scene: the worker's
    train pass reads and rewrites the normal lane only through ``MergeIntake``."""
    repo = seed_repo(tmp_path / 'repo', README_SEED)
    git_ops = GitOps(_GIT, repo)
    events = EventStore(db_path=tmp_path / 'lane-events.db', run_id='row-4b')
    config = lane_scene_config(repo, _GIT, merge_train_coalesce_enabled=True)
    names = ('wf-a', 'wf-b')
    singles = []
    for name in names:
        worktree = (await git_ops.create_worktree(name)).path
        (worktree / f'{name}.txt').write_text(f'{name}\n')
        await git_ops.commit(worktree, f'{name}: own work')
        singles.append(MergeRequest(
            task_id=name, branch=QueuedBranch.parse(name, _GIT.branch_prefix),
            worktree=worktree, pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=asyncio.get_running_loop().create_future(),
            awaited_by_workflow=True,
        ))
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for single in singles:
        queue.put_nowait(single)
    hooks = _TrainHooks(parked=set(names))
    lane = make_lane(git_ops, queue, event_store=events, train_callback_factory=hooks.factory)

    async with running_lane(lane) as run:
        outcomes = [await run.outcome(single) for single in singles]
        await hooks.wait_landed()

    (train,) = [event['data'] for event in events.fetch_events_by_type(EventType.train_coalesced)]
    assert train['train_id'].startswith('coalesce-')
    assert train['member_task_ids'] == list(names)
    assert [(o.status, o.superseded_by) for o in outcomes] == [
        ('superseded', train['train_id']), ('superseded', train['train_id']),
    ]
    assert sorted(hooks.marked_done) == sorted(names)
