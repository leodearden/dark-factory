"""Tests making the two hardcoded-from_state ItemLifecycle hops
(``_buffer_owned_request``'s QUEUED->LANE_BUFFERED and
``_finalize_inflight``'s VERIFYING->FINALIZING) re-entry-tolerant, so a
duplicate/re-entrant submission coalesces as a benign no-op instead of
firing a ``merge_lifecycle_transition_rejected`` escalation (task 2852;
SHARPER RCA, reify cluster mr-8cafbaf0 / mr-99585bb8 / original incident
mr-c69ca480).

Both offender sites hardcode the transition's ``from_state`` instead of
reading the :class:`~orchestrator.merge_queue.ItemLifecycle` registry's
CURRENT state first (the pattern already used by ``_note_requeue``) — so a
duplicate/re-entrant driver that touches a request_id already deep in the
pipeline gets an illegal-edge rejection (non-corrupting, but persistently
noisy) instead of being recognized and coalesced.

Steps covered:
  step-1  RED   — SpeculativeMergeWorker._advance_if_at fire + tolerant
                  no-op contract
  step-2  GREEN — _advance_if_at added
  step-3  impl  — _finalize_inflight wired through _advance_if_at (no new
                  test here — proven by step-1(b) at the exact seam; normal-
                  path firing regression-covered by test_merge_speculation.py)
  step-4  RED   — _buffer_owned_request twin-drain repro (headline incident)
  step-5  GREEN — _buffer_owned_request restructured + _coalesce_reentrant_drain
                  (drop-only)
  step-6  RED   — fresh/requeue regressions + twin-future-resolved +
                  observability
  step-7  GREEN — _coalesce_reentrant_drain resolves the twin's future +
                  emits a merge_coalesced event

Amendment (reviewer_comprehensive, task 2852): ``_advance_if_at`` gained a
``tolerate`` parameter so a *current* state is only coalesced as a benign
no-op when the caller explicitly names it as a known-downstream state for
that hop — an untolerated mismatch still escalates, preserving detection of
a genuine wiring/ordering bug (``TestAdvanceIfAtToleratesOnlyNamedStates``).
Also adds a regression proving a twin sharing a request_id with an original
already at FINALIZING (the exact mr-99585bb8 double-finalize state) is
coalesced upstream of ``_finalize_inflight`` entirely, so that method's own
downstream hardcoded-from_state hops are never reached by this incident
class (``test_twin_drain_at_finalizing_state_never_reaches_finalize_inflight``).

This module imports orchestrator.merge_queue LOCALLY inside each test
method (not at module scope) so a not-yet-implemented symbol (e.g.
``_advance_if_at``, before step-2) never breaks collection of the rest of
the file during the RED steps — mirrors
test_merge_queue_lifecycle_registry.py's convention.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import MergeRequest, QueuedBranch

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — copied from
# test_merge_queue_lifecycle_registry.py; there is no shared worker conftest).
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


class _FakeEscalationQueue:
    """Minimal fake escalation queue (copied from
    test_merge_queue_lifecycle_registry.py — per-file duplication
    convention).
    """

    def __init__(self, *, open_l1: bool = False):
        self._open_l1 = open_l1
        self._seq = 0
        self.submitted: list = []

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return self._open_l1

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc) -> None:
        self.submitted.append(esc)

    def open_it(self):
        """Simulate a prior open L1 (for dedup tests)."""
        self._open_l1 = True


def _make_worker(git_ops: GitOps, *, escalation_queue: Any = None, event_store: Any = None):
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring).

    Mirrors test_merge_queue_lifecycle_registry.py's ``_make_worker``,
    extended with an optional ``event_store`` passthrough for the
    observability test (step-6(d) / step-7).
    """
    from orchestrator.merge_queue import SpeculativeMergeWorker

    return SpeculativeMergeWorker(
        git_ops, asyncio.Queue(), event_store=event_store, escalation_queue=escalation_queue,
    )


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop.

    Duplicated from test_merge_queue_lifecycle_registry.py (per-file
    duplication convention — see this file's module docstring), extended
    with an optional ``request_id`` override so twin-submission tests can
    force two DISTINCT ``MergeRequest`` objects to share one request_id —
    mirroring the journal-recovery ``reconstruct_merge_request`` path that
    produced the original incident's twin.
    """
    kwargs: dict[str, Any] = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        **kwargs,
    )


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: _advance_if_at shared re-entry-tolerant primitive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAdvanceIfAt:
    """``SpeculativeMergeWorker._advance_if_at(request_id, from_state,
    to_state, *, live_obj=None) -> bool`` — the shared re-entry-tolerant
    forward-hop primitive (task 2852 step-1/step-2), mirroring
    ``_note_requeue``'s read-``current()``-first discipline: fires the
    transition (via ``_note_transition``) iff the registry's current state
    equals *from_state*; otherwise logs a WARNING and no-ops — NOT an
    escalation.

    RED until step-2 GREEN adds ``_advance_if_at`` — AttributeError on
    current main.
    """

    async def test_fires_when_current_matches_from_state(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        item = _make_request('advance-fire', 'advance-fire', tmp_path, config)
        rid = item.request_id
        worker._register_item(item, initial=ItemLifecycleState.QUEUED)

        result = worker._advance_if_at(
            rid, ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED, live_obj=item,
        )

        assert result is True
        assert worker._lifecycle.current(rid) == ItemLifecycleState.LANE_BUFFERED
        assert worker._live_items[rid] is item
        assert fake_eq.submitted == []

    async def test_tolerant_noop_at_finalize_seam_when_already_advanced(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The exact double-finalize condition (SHARPER RCA mr-99585bb8):
        the registry is already at FINALIZING when ``_finalize_inflight``'s
        hop fires a second time for the same request_id.

        Amendment (reviewer_comprehensive, task 2852): passes ``tolerate=``
        matching ``_finalize_inflight``'s real call site exactly (the four
        states it documents as reachable here) — a bare mismatch is no
        longer unconditionally tolerated (see
        ``TestAdvanceIfAtToleratesOnlyNamedStates`` below).
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        entry = _make_request('advance-noop', 'advance-noop', tmp_path, config)
        rid = entry.request_id
        worker._register_item(entry, initial=ItemLifecycleState.FINALIZING)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._advance_if_at(
                rid, ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING,
                live_obj=entry,
                tolerate=frozenset({
                    ItemLifecycleState.FINALIZING,
                    ItemLifecycleState.MERGING,
                    ItemLifecycleState.GATE_REVERIFY,
                    ItemLifecycleState.TERMINAL,
                }),
            )

        assert result is False
        assert worker._lifecycle.current(rid) == ItemLifecycleState.FINALIZING
        assert fake_eq.submitted == [], (
            f'tolerant no-op must NOT escalate: {fake_eq.submitted!r}'
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        assert rid in warnings[0].message


@pytest.mark.asyncio
class TestAdvanceIfAtToleratesOnlyNamedStates:
    """Amendment (reviewer_comprehensive, task 2852): ``_advance_if_at``
    must NOT treat every ``current != from_state`` mismatch as a benign
    duplicate — only a *current* the caller explicitly names via
    ``tolerate`` (or omits, meaning "none"). Anything else is a genuinely
    earlier/off-path state — a real wiring/ordering bug — and must still
    fall through to :meth:`_note_transition` and escalate, exactly as it
    did before ``_advance_if_at`` existed. This scopes the noise-suppression
    to the actual duplicate/re-entry case the SHARPER RCA targeted instead
    of silently swallowing an unrelated ordering violation.
    """

    async def test_off_path_current_not_in_tolerate_still_escalates(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """current=DISPATCHING when the caller expects VERIFYING (and only
        tolerates FINALIZING/MERGING/GATE_REVERIFY/TERMINAL, mirroring
        ``_finalize_inflight``'s real call) is off-path, not downstream —
        the entry has not even reached verify yet. Must escalate exactly
        like the pre-``_advance_if_at`` hardcoded call would have.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        entry = _make_request('advance-offpath', 'advance-offpath', tmp_path, config)
        rid = entry.request_id
        worker._register_item(entry, initial=ItemLifecycleState.DISPATCHING)

        result = worker._advance_if_at(
            rid, ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING,
            live_obj=entry,
            tolerate=frozenset({
                ItemLifecycleState.FINALIZING,
                ItemLifecycleState.MERGING,
                ItemLifecycleState.GATE_REVERIFY,
                ItemLifecycleState.TERMINAL,
            }),
        )

        assert result is False
        assert worker._lifecycle.current(rid) == ItemLifecycleState.DISPATCHING, (
            'a rejected transition must leave the registry state unchanged'
        )
        assert len(fake_eq.submitted) == 1, (
            f'an off-path/untolerated mismatch must still escalate: {fake_eq.submitted!r}'
        )
        assert fake_eq.submitted[0].category == 'merge_lifecycle_transition_rejected'

    async def test_off_path_current_with_no_tolerate_argument_still_escalates(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Default ``tolerate=frozenset()`` (the ``_buffer_owned_request``
        call site never passes ``tolerate`` — its own pre-branching already
        guarantees ``current == QUEUED``) keeps the strict pre-existing
        behavior: ANY mismatch escalates.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        entry = _make_request('advance-strict', 'advance-strict', tmp_path, config)
        rid = entry.request_id
        worker._register_item(entry, initial=ItemLifecycleState.VERIFYING)

        result = worker._advance_if_at(
            rid, ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED, live_obj=entry,
        )

        assert result is False
        assert worker._lifecycle.current(rid) == ItemLifecycleState.VERIFYING
        assert len(fake_eq.submitted) == 1
        assert fake_eq.submitted[0].category == 'merge_lifecycle_transition_rejected'

    async def test_tolerates_named_downstream_state_beyond_the_exact_hardcoded_case(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """current=MERGING (named in ``tolerate`` but distinct from the
        FINALIZING case already covered by
        ``TestAdvanceIfAt.test_tolerant_noop_at_finalize_seam_when_already_advanced``)
        must ALSO coalesce as a benign no-op — proving ``tolerate`` is a
        real set, not just a single-value special case.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        entry = _make_request('advance-tolerate-merging', 'advance-tolerate-merging', tmp_path, config)
        rid = entry.request_id
        worker._register_item(entry, initial=ItemLifecycleState.MERGING)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._advance_if_at(
                rid, ItemLifecycleState.VERIFYING, ItemLifecycleState.FINALIZING,
                live_obj=entry,
                tolerate=frozenset({
                    ItemLifecycleState.FINALIZING,
                    ItemLifecycleState.MERGING,
                    ItemLifecycleState.GATE_REVERIFY,
                    ItemLifecycleState.TERMINAL,
                }),
            )

        assert result is False
        assert worker._lifecycle.current(rid) == ItemLifecycleState.MERGING
        assert fake_eq.submitted == [], (
            f'a named-tolerated downstream state must NOT escalate: {fake_eq.submitted!r}'
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        assert rid in warnings[0].message


# ---------------------------------------------------------------------------
# step-4 RED / step-5 GREEN: _buffer_owned_request duplicate/twin-drain
# (SHARPER-RCA offender #2 + original incident mr-c69ca480)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBufferOwnedRequestDuplicateDrain:
    """``_buffer_owned_request`` (SHARPER-RCA offender #2, mr-8cafbaf0 — and
    the original incident mr-c69ca480): draining a DISTINCT ``MergeRequest``
    object that shares its request_id with an item already live in the
    registry at a non-QUEUED state (e.g. VERIFYING) is a duplicate/re-entrant
    submission — mirroring the journal-recovery ``reconstruct_merge_request``
    path that produced the original incident's twin. It must coalesce as a
    benign no-op: no escalation, no divergent second pipeline item buffered
    under the shared request_id, and the live original left untouched.

    RED until step-5 GREEN restructures ``_buffer_owned_request`` with the
    registry-state three-way discrimination + ``_coalesce_reentrant_drain``.
    """

    async def test_twin_drain_does_not_escalate_or_buffer_and_leaves_original_live(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)

        original = _make_request('mr-twin-task', 'mr-twin-branch', tmp_path, config)
        rid = original.request_id
        worker._register_item(original, initial=ItemLifecycleState.VERIFYING)

        twin = _make_request(
            'mr-twin-task', 'mr-twin-branch', tmp_path, config, request_id=rid,
        )
        assert twin is not original, 'twin must be a DISTINCT object sharing the request_id'

        worker._buffer_owned_request(twin)

        assert fake_eq.submitted == [], (
            f'a recognized duplicate/re-entrant drain must not escalate: {fake_eq.submitted!r}'
        )
        for lane, buf in worker._lane_buffers.items():
            assert twin not in buf, f'twin must not be buffered in lane {lane!r}: {buf!r}'
        assert worker._lifecycle.current(rid) == ItemLifecycleState.VERIFYING, (
            'registry must be untouched by the coalesced duplicate'
        )
        assert worker._live_items[rid] is original, (
            'the live original must not be displaced by the twin'
        )

    async def test_twin_drain_at_finalizing_state_never_reaches_finalize_inflight(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Amendment (reviewer_comprehensive, task 2852): the SHARPER RCA's
        second cluster (mr-99585bb8) was a double-finalize — a request_id
        whose registry state had already reached FINALIZING. This proves
        that scenario is now closed off UPSTREAM of ``_finalize_inflight``
        entirely: a twin sharing a request_id with an original already at
        FINALIZING is coalesced right here at the drain chokepoint and
        never gets buffered, dispatched, or given its own InflightEntry —
        so it can never reach ``_finalize_inflight``'s downstream hardcoded-
        from_state hops (FINALIZING -> MERGING, etc.) at all. This is why
        those downstream hops do not need their own ``_advance_if_at``
        wiring for THIS incident class: there is no second entry object left
        to drive them re-entrantly once the twin is coalesced here.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)

        original = _make_request('mr-twin-finalizing', 'mr-twin-finalizing-branch', tmp_path, config)
        rid = original.request_id
        worker._register_item(original, initial=ItemLifecycleState.FINALIZING)

        twin = _make_request(
            'mr-twin-finalizing', 'mr-twin-finalizing-branch', tmp_path, config, request_id=rid,
        )

        worker._buffer_owned_request(twin)

        assert fake_eq.submitted == [], (
            f'a duplicate drain onto an already-FINALIZING request_id must not '
            f'escalate: {fake_eq.submitted!r}'
        )
        for lane, buf in worker._lane_buffers.items():
            assert twin not in buf, f'twin must not be buffered in lane {lane!r}: {buf!r}'
        assert worker._lifecycle.current(rid) == ItemLifecycleState.FINALIZING, (
            'registry must be untouched by the coalesced duplicate'
        )
        assert worker._live_items[rid] is original, (
            'the live original must not be displaced by the twin — no second '
            'InflightEntry is ever created for the twin, so _finalize_inflight '
            'is never invoked a second time for this request_id'
        )
        assert twin.result.done() and twin.result.result().status == 'already_merged'


# ---------------------------------------------------------------------------
# step-6 RED (twin-future-resolved / observability) + regression guards
# (fresh-path / legit-requeue) — step-7 GREEN extends
# _coalesce_reentrant_drain to resolve the twin's future + emit
# merge_coalesced.
# ---------------------------------------------------------------------------


class _FakeEventStore:
    """Minimal fake event store (copied from test_merge_request_ledger.py /
    test_merge_queue_multihost_wiring.py — per-file duplication convention).
    """

    def __init__(self) -> None:
        self.emitted: list = []

    def emit(self, event_type, *, task_id=None, phase=None, data=None, **kw):  # noqa: ARG002
        self.emitted.append({'event_type': event_type, 'task_id': task_id, 'data': data or {}})


@pytest.mark.asyncio
class TestBufferOwnedRequestFreshAndRequeueRegression:
    """Regression guards proving the fresh-request and legit-requeue paths
    through the step-5-restructured ``_buffer_owned_request`` still behave
    exactly as before (task 2852 step-6(a)/(b)) — only the NEW
    already-advanced-past-QUEUED branch changed. Both pass immediately
    after step-5 (no RED here).
    """

    async def test_fresh_request_still_buffers_and_advances_to_lane_buffered(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        req = _make_request('fresh-task', 'fresh-branch', tmp_path, config)
        rid = req.request_id

        worker._buffer_owned_request(req)

        assert worker._lifecycle.current(rid) == ItemLifecycleState.LANE_BUFFERED
        assert any(req in buf for buf in worker._lane_buffers.values()), (
            'fresh request must be appended to its lane buffer'
        )
        assert fake_eq.submitted == []

    async def test_legit_requeue_same_object_redrains_cleanly(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Mirrors test_merge_queue_lifecycle_registry.py:602-638's
        legit-requeue re-drain: the SAME object left registered at QUEUED
        (as ``_note_requeue`` leaves it) re-drains through
        ``_buffer_owned_request`` without being dropped as a duplicate.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        req = _make_request('requeue-task', 'requeue-branch', tmp_path, config)
        rid = req.request_id
        worker._register_item(req, initial=ItemLifecycleState.QUEUED)

        worker._buffer_owned_request(req)

        assert worker._lifecycle.current(rid) == ItemLifecycleState.LANE_BUFFERED
        assert worker._live_items[rid] is req, 'the same object must not be dropped'
        assert fake_eq.submitted == []


@pytest.mark.asyncio
class TestCoalescedTwinFutureAndObservability:
    """The coalesced twin's Future is defensively resolved to a benign
    ``already_merged`` MergeOutcome (so no hypothetical waiter hangs) and a
    ``merge_coalesced`` observability event is emitted (task 2852
    step-6/step-7).

    RED until step-7 GREEN extends ``_coalesce_reentrant_drain`` — the
    step-5 drop-only version resolves no future and emits no event.
    """

    async def test_twin_future_resolved_to_already_merged(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        original = _make_request('twin-future-task', 'twin-future-branch', tmp_path, config)
        rid = original.request_id
        worker._register_item(original, initial=ItemLifecycleState.VERIFYING)
        twin = _make_request(
            'twin-future-task', 'twin-future-branch', tmp_path, config, request_id=rid,
        )

        worker._buffer_owned_request(twin)

        assert twin.result.done(), 'no (hypothetical) waiter on the twin future may hang'
        assert twin.result.result().status == 'already_merged'

    async def test_twin_drain_emits_exactly_one_merge_coalesced_event(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.event_store import EventType
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        fake_es = _FakeEventStore()
        worker = _make_worker(git_ops, escalation_queue=fake_eq, event_store=fake_es)
        original = _make_request('twin-event-task', 'twin-event-branch', tmp_path, config)
        rid = original.request_id
        worker._register_item(original, initial=ItemLifecycleState.VERIFYING)
        twin = _make_request(
            'twin-event-task', 'twin-event-branch', tmp_path, config, request_id=rid,
        )

        worker._buffer_owned_request(twin)

        coalesced = [e for e in fake_es.emitted if e['event_type'] == EventType.merge_coalesced]
        assert len(coalesced) == 1, (
            f'expected exactly one merge_coalesced event, got: {fake_es.emitted!r}'
        )
        assert coalesced[0]['data']['source'] == 'duplicate_submission'
        assert coalesced[0]['task_id'] == twin.task_id


# ---------------------------------------------------------------------------
# task 3082 step-7 RED / step-8 GREEN: the coalesce must never fabricate
# success for a SELF-requeue.
#
# `_coalesce_reentrant_drain` resolves the drained item's future to
# `already_merged` UNCONDITIONALLY. That is genuinely benign for the task-2852
# journal-recovery TWIN (a DISTINCT object with a fresh, unobserved future), but
# when the arriving item IS the live original — a requeue-wiring bug — it hands a
# REAL waiter a fabricated success AND silences the only watchdog that would
# have caught the wedge: the request ledger's `sweep_resolved` drops any
# `done()` entry before `stuck_entries` sees it, so `merge_request_stuck` never
# fires.
#
# The discriminator is FUTURE identity, not object identity of the item:
# `_live_items[rid]` legitimately changes SHAPE across the pipeline
# (MergeRequest -> SpeculativeItem -> InflightEntry) while carrying the SAME
# `request.result`, so an `is`-on-the-item check would fail open exactly when
# the entry is deepest in the pipeline — which is when this bug bites.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCoalesceNeverResolvesALiveRequestsFuture:
    """A re-drain of the LIVE ORIGINAL must leave its future PENDING and
    escalate loudly, while a genuine duplicate twin keeps today's benign
    ``already_merged`` resolution verbatim (task 3082 step-7 RED / step-8
    GREEN).

    RED until step-8 discriminates live-original from twin by future identity.
    """

    @staticmethod
    def _drive_live_original(worker: Any, req: MergeRequest) -> None:
        """Re-drain the SAME object that is already live in the registry."""
        from orchestrator.merge_queue import ItemLifecycleState

        worker._register_item(req, initial=ItemLifecycleState.VERIFYING)
        worker._buffer_owned_request(req)

    async def test_live_original_redrain_does_not_resolve_its_future(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        req = _make_request('df3082-live-original', 'df3082-live-original', tmp_path, config)
        rid = req.request_id

        self._drive_live_original(worker, req)

        assert not req.result.done(), (
            f'the LIVE original\'s real waiter must not be handed a fabricated '
            f'outcome: {req.result.result() if req.result.done() else None!r}'
        )
        assert not any(req in buf for buf in worker._lane_buffers.values()), (
            'a re-entrant drain must still not be buffered as a divergent second item'
        )
        assert worker._live_items[rid] is req, f'{worker._live_items.get(rid)!r}'
        current = worker._lifecycle.current(rid)
        assert current == ItemLifecycleState.VERIFYING, (
            f'the live original\'s registry state must be untouched; reads {current!r}'
        )

    async def test_live_original_redrain_escalates_loudly(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """A re-entry of a genuinely live request is a requeue-wiring bug, not a
        benign twin: escalate loudly rather than degrade.
        """
        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        req = _make_request('df3082-live-escalate', 'df3082-live-escalate', tmp_path, config)
        rid = req.request_id

        self._drive_live_original(worker, req)

        assert len(fake_eq.submitted) == 1, (
            f'expected exactly one escalation for a live-original re-drain, got: '
            f'{fake_eq.submitted!r}'
        )
        esc = fake_eq.submitted[0]
        assert esc.category != 'merge_lifecycle_transition_rejected', (
            f'this is a distinct condition from a rejected transition: {esc.category!r}'
        )
        assert esc.category == 'merge_coalesce_live_original', (
            f'unexpected category: {esc.category!r}'
        )
        blob = f'{esc.summary}\n{esc.detail}'
        assert rid in blob, f'the escalation must name the request_id: {blob!r}'
        assert req.branch.bare_id in blob, f'the escalation must name the branch: {blob!r}'
        assert 'requeue' in blob.lower(), (
            f'the escalation must identify this as a requeue-wiring bug, not a '
            f'benign twin: {blob!r}'
        )

    async def test_live_original_redrain_still_emits_merge_coalesced(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Observability must not regress just because the future resolution is
        suppressed.
        """
        from orchestrator.event_store import EventType

        fake_eq = _FakeEscalationQueue(open_l1=False)
        fake_es = _FakeEventStore()
        worker = _make_worker(git_ops, escalation_queue=fake_eq, event_store=fake_es)
        req = _make_request('df3082-live-event', 'df3082-live-event', tmp_path, config)

        self._drive_live_original(worker, req)

        coalesced = [e for e in fake_es.emitted if e['event_type'] == EventType.merge_coalesced]
        assert len(coalesced) == 1, (
            f'expected exactly one merge_coalesced event, got: {fake_es.emitted!r}'
        )
        assert coalesced[0]['data']['source'] == 'duplicate_submission'

    async def test_coalesced_live_request_still_trips_merge_request_stuck(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """The watchdog the bug silenced: leaving the future PENDING re-arms
        ``merge_request_stuck``, because the ledger's ``sweep_resolved`` drops
        any ``done()`` entry before ``stuck_entries`` sees it.

        Reads merge_request_ledger.py only — that file is scope-fenced by this
        task and must NOT be modified.
        """
        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        req = _make_request('df3082-live-stuck', 'df3082-live-stuck', tmp_path, config)

        t0 = 1_000_000.0
        worker._request_ledger.on_dequeue(req, now=t0)
        self._drive_live_original(worker, req)
        worker._check_request_liveness(t0 + 2000.0, threshold_s=1000.0)

        stuck = [e for e in fake_eq.submitted if e.category == 'merge_request_stuck']
        assert len(stuck) == 1, (
            f'a coalesce-dropped LIVE request must still age out into a '
            f'merge_request_stuck alarm: {fake_eq.submitted!r}'
        )
        assert req.request_id in stuck[0].summary

    async def test_distinct_twin_future_is_still_resolved_to_already_merged(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """NO-REGRESSION GUARD pinning the exact boundary of the fix: only a
        future SHARED with the live registry object is spared. The task-2852
        journal-recovery twin — a DISTINCT object with its own fresh, unobserved
        future — keeps today's benign ``already_merged`` resolution and fires no
        wiring-bug escalation.
        """
        from orchestrator.merge_queue import ItemLifecycleState

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = _make_worker(git_ops, escalation_queue=fake_eq)
        original = _make_request('df3082-twin', 'df3082-twin', tmp_path, config)
        rid = original.request_id
        worker._register_item(original, initial=ItemLifecycleState.VERIFYING)
        twin = _make_request('df3082-twin', 'df3082-twin', tmp_path, config, request_id=rid)

        worker._buffer_owned_request(twin)

        assert twin.result.done(), 'no (hypothetical) waiter on the twin future may hang'
        assert twin.result.result().status == 'already_merged'
        assert not original.result.done(), (
            'the live original\'s future must stay pending regardless'
        )
        assert fake_eq.submitted == [], (
            f'a genuine journal-recovery twin is benign and must not escalate: '
            f'{fake_eq.submitted!r}'
        )
