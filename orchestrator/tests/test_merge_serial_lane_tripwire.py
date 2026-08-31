"""Tests for the merge serial-lane C4 tripwire (task 2930, PRD η).

docs/prds/merge-worktree-lifecycle-integrity.md §4 C4 / §9 row 10: dispatching
a SECOND concurrent LOCAL merge verify while the ``_MERGE_AHEAD_BOUND``-derived
per-host in-flight bound is 1 must log a WARNING and emit a telemetry event
(``EventType.merge_serial_lane_breached``).  **No hard block** — this is a
cheap DETECTION net for a future request-identity leak of the task/5326 class
(two journal entries rehydrated for one branch, both enqueued, bypassing the
``InFlightMergeRegistry`` coalesce gate).

Each test class imports the symbols under test LOCALLY inside its test methods
(not at module scope) so a not-yet-implemented name never breaks collection of
the rest of this file during earlier RED steps — mirrors the convention
documented in test_merge_skew_tripwire.py:10-13.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_types import (
    InflightEntry,
    ItemLifecycleState,
    QueuedBranch,
    RealMergeItem,
)
from orchestrator.verify_runner import HostAllocator, HostLease

# ---------------------------------------------------------------------------
# Config builder (per-file duplication — PRD D9: no cross-test-module imports
# of private fixtures; mirrors test_merge_queue_persistent_worktree.py:59+)
# ---------------------------------------------------------------------------


def _make_persistent_config(root: Path, *, persistent: bool) -> OrchestratorConfig:
    """Build OrchestratorConfig with the persistent_merge_worktree knob set."""
    git = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        persistent_merge_worktree=persistent,
    )
    return OrchestratorConfig(project_root=root, git=git)


# ---------------------------------------------------------------------------
# Step 01 — per_host_inflight_bound (INV-5 extraction)
# ---------------------------------------------------------------------------


class TestPerHostInflightBound:
    """``per_host_inflight_bound(merge_ahead_bound, num_hosts)`` is THE formula.

    One definition of ``ceil(max(1, bound) / max(1, hosts))``, shared by the
    fail-closed startup guard (``enforce_persistent_worktree_serial_lane``) and
    the C4 runtime tripwire — INV-5 (no-lockstep-duplication): two sites that
    must agree byte-for-byte are one site plus a call.
    """

    @pytest.mark.parametrize(
        ('bound', 'hosts', 'expected'),
        [
            (1, 1, 1),
            (2, 1, 2),
            (2, 2, 1),
            (3, 2, 2),  # ceil semantics — an uneven split rounds UP
            (4, 4, 1),
            (4, 2, 2),
        ],
    )
    def test_ceil_semantics(self, bound: int, hosts: int, expected: int) -> None:
        """The worst-case per-host in-flight count is ceil(bound / hosts)."""
        from orchestrator.merge_liveness import per_host_inflight_bound  # noqa: PLC0415

        assert per_host_inflight_bound(bound, hosts) == expected

    @pytest.mark.parametrize(
        ('bound', 'hosts'),
        [
            (0, 1),
            (-5, 1),
            (1, 0),
            (1, -3),
            (0, 0),
        ],
    )
    def test_degenerate_inputs_clamp_to_at_least_one(self, bound: int, hosts: int) -> None:
        """Degenerate inputs fail SAFE: >= 1, never ZeroDivisionError, never 0.

        Matches the clamp behaviour the guard has always had inline.  A
        spuriously permissive 0 would make the fail-closed startup guard stop
        refusing (and the tripwire stop firing) on a nonsense config.
        """
        from orchestrator.merge_liveness import per_host_inflight_bound  # noqa: PLC0415

        assert per_host_inflight_bound(bound, hosts) == 1


class TestSerialLaneGuardUsesSharedBound:
    """INV-5 anti-drift pin: the startup guard ROUTES THROUGH the helper.

    The guard's verdict must be genuinely derived from
    ``per_host_inflight_bound``'s return value, not from a duplicate inline
    expression that could silently drift away from the tripwire's copy.
    """

    def test_guard_delegates_to_per_host_inflight_bound(self, tmp_path: Path) -> None:
        """Patching the helper to return 1 makes the guard accept bound=2/hosts=1.

        Un-refactored, ``ceil(2/1) == 2`` would raise.  A no-raise here proves
        the raise decision reads the helper's return value.
        """
        from orchestrator.merge_liveness import (  # noqa: PLC0415
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_persistent_config(tmp_path, persistent=True)
        fake = MagicMock(return_value=1)
        with patch('orchestrator.merge_liveness.per_host_inflight_bound', fake):
            result = enforce_persistent_worktree_serial_lane(
                cfg, merge_ahead_bound=2, num_hosts=1
            )

        assert result is None, (
            'guard must not raise when the shared bound helper returns 1 — '
            'a raise here means the guard still recomputes the formula inline'
        )
        fake.assert_called_once_with(2, 1)


# ---------------------------------------------------------------------------
# Step 03 — check_serial_lane_tripwire (pure decision)
# ---------------------------------------------------------------------------


class TestCheckSerialLaneTripwire:
    """``check_serial_lane_tripwire`` is the PURE C4 decision.

    No I/O, no logging, no emission — it returns a :class:`SerialLaneAssessment`
    and the caller decides what to do.  ``local_inflight`` is the count of LOCAL
    verifies in flight INCLUDING the dispatch under consideration, so at bound=1
    the FIRST local dispatch is ``1 > 1`` → False (the positive control holds by
    construction, not via a suppression rule) and the SECOND is ``2 > 1`` → True.
    """

    def test_single_local_dispatch_at_bound_1_is_not_breached(self) -> None:
        """POSITIVE CONTROL: one local verify at bound=1 is the normal case."""
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        assessment = check_serial_lane_tripwire(1, merge_ahead_bound=1, num_hosts=1)
        assert assessment.breached is False

    def test_second_concurrent_local_dispatch_is_breached(self) -> None:
        """The C4 condition (§9 row 10): 2 local verifies at per-host bound 1."""
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        assessment = check_serial_lane_tripwire(2, merge_ahead_bound=1, num_hosts=1)
        assert assessment.breached is True

    def test_third_local_dispatch_is_also_breached(self) -> None:
        """Fires for EVERY excess dispatch, not just the second."""
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        assessment = check_serial_lane_tripwire(3, merge_ahead_bound=1, num_hosts=1)
        assert assessment.breached is True

    def test_idle_lane_is_not_breached(self) -> None:
        """Zero local verifies in flight cannot breach a bound of 1."""
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        assessment = check_serial_lane_tripwire(0, merge_ahead_bound=1, num_hosts=1)
        assert assessment.breached is False

    def test_multi_host_k2_one_local_verify_is_not_breached(self) -> None:
        """K=2 across 2 hosts → per-host bound 1; one local verify is legal.

        Mirrors the harness's ``num_hosts=_k`` wiring
        (orchestrator/src/orchestrator/harness.py::Harness._start_merge_worker),
        where the per-host bound is always 1.
        """
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        assessment = check_serial_lane_tripwire(1, merge_ahead_bound=2, num_hosts=2)
        assert assessment.breached is False
        assert assessment.per_host_bound == 1

    def test_multi_host_k2_two_local_verifies_is_breached(self) -> None:
        """K=2 across 2 hosts still allows only ONE verify per host."""
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        assessment = check_serial_lane_tripwire(2, merge_ahead_bound=2, num_hosts=2)
        assert assessment.breached is True

    def test_assessment_carries_the_structured_facts_it_used(self) -> None:
        """INV-2: the payload facts are on the assessment, at the values used."""
        from orchestrator.merge_liveness import (  # noqa: PLC0415
            SerialLaneAssessment,
            check_serial_lane_tripwire,
        )

        assessment = check_serial_lane_tripwire(3, merge_ahead_bound=4, num_hosts=2)
        assert isinstance(assessment, SerialLaneAssessment)
        assert assessment.local_inflight == 3
        assert assessment.per_host_bound == 2  # ceil(4/2)
        assert assessment.merge_ahead_bound == 4
        assert assessment.num_hosts == 2
        assert assessment.breached is True  # 3 > 2

    def test_bound_defaults_to_engine_constant_resolved_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitted bound reaches back to merge_queue._MERGE_AHEAD_BOUND AT CALL TIME.

        A def-time default would need a top-level ``import
        orchestrator.merge_queue`` in merge_liveness (module-load deadlock — the
        shim needs merge_liveness fully defined first) AND would defeat this
        monkeypatch, which the suite already relies on for
        ``enforce_persistent_worktree_serial_lane``.
        """
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        monkeypatch.setattr('orchestrator.merge_queue._MERGE_AHEAD_BOUND', 4)
        assessment = check_serial_lane_tripwire(2)
        assert assessment.merge_ahead_bound == 4
        assert assessment.breached is False  # 2 > ceil(4/1)=4 is False

    def test_unpatched_bound_default_is_the_real_engine_constant(self) -> None:
        """Unpatched, the reach-back yields the real _MERGE_AHEAD_BOUND (1)."""
        from orchestrator.merge_liveness import check_serial_lane_tripwire  # noqa: PLC0415

        assessment = check_serial_lane_tripwire(2)
        assert assessment.merge_ahead_bound == 1
        assert assessment.breached is True


# ---------------------------------------------------------------------------
# Step 07 — alarm_serial_lane_breach (acting wrapper)
# ---------------------------------------------------------------------------


class TestAlarmSerialLaneBreach:
    """``alarm_serial_lane_breach`` is the acting half: WARNING + telemetry.

    Never raises, never blocks, returns None — there is no veto channel a
    caller could accidentally honour as a hard block (PRD C4: "no hard block").
    """

    def test_breach_logs_one_warning_and_emits_one_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Breached → exactly one WARNING naming the numbers, and one event."""
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        from orchestrator.merge_liveness import (  # noqa: PLC0415
            alarm_serial_lane_breach,
            check_serial_lane_tripwire,
        )

        assessment = check_serial_lane_tripwire(2, merge_ahead_bound=1, num_hosts=1)
        rec = _RecordingEventStore()
        # merge_liveness.py:58 binds logger = getLogger('orchestrator.merge_queue'),
        # NOT __name__ — filter on that name.
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = alarm_serial_lane_breach(
                assessment,
                event_store=rec,
                task_id='5326',
                branch='task/5326',
                request_id='mr-29dfdbc2',
                host='local',
            )

        assert result is None

        # Match on a SPECIFIC substring, never "some WARNING exists".
        hits = [r for r in caplog.records if 'serial-lane' in r.getMessage()]
        assert len(hits) == 1, f'expected exactly one tripwire WARNING; got {hits!r}'
        msg = hits[0].getMessage()
        assert hits[0].levelno == logging.WARNING
        for fact in ('local_inflight=2', 'per_host_bound=1', 'task/5326'):
            assert fact in msg, f'WARNING must carry {fact!r}; got: {msg!r}'

        # INV-2: the payload carries every fact the emitter held in a variable,
        # so a consumer never has to log-scrape.
        events = [e for e in rec.events if e[0] == 'merge_serial_lane_breached']
        assert len(events) == 1
        _etype, payload = events[0]
        assert payload['task_id'] == '5326'
        assert payload['data'] == {
            'local_inflight': 2,
            'per_host_bound': 1,
            'merge_ahead_bound': 1,
            'num_hosts': 1,
            'branch': 'task/5326',
            'request_id': 'mr-29dfdbc2',
            'host': 'local',
        }

    def test_not_breached_is_totally_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """POSITIVE CONTROL (PRD §9 row 10 negative half): no event, no WARNING.

        The bare ``== []`` assertion is safe at THIS layer only: the alarm is
        called directly, with no worker and hence no ``_note_transition`` in
        play to pollute either channel.  Step 09's integration-layer positive
        control MUST filter both channels — do not "helpfully" loosen this one
        or tighten that one to match.
        """
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        from orchestrator.merge_liveness import (  # noqa: PLC0415
            alarm_serial_lane_breach,
            check_serial_lane_tripwire,
        )

        assessment = check_serial_lane_tripwire(1, merge_ahead_bound=1, num_hosts=1)
        rec = _RecordingEventStore()
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = alarm_serial_lane_breach(
                assessment,
                event_store=rec,
                task_id='5326',
                branch='task/5326',
                request_id='mr-29dfdbc2',
                host='local',
            )

        assert result is None
        assert rec.events == []
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_no_event_store_still_logs_the_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A store-less worker still gets the loud log (fail-open on telemetry only)."""
        from orchestrator.merge_liveness import (  # noqa: PLC0415
            alarm_serial_lane_breach,
            check_serial_lane_tripwire,
        )

        assessment = check_serial_lane_tripwire(2, merge_ahead_bound=1, num_hosts=1)
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = alarm_serial_lane_breach(
                assessment, event_store=None, task_id='5326', branch='task/5326'
            )

        assert result is None
        hits = [r for r in caplog.records if 'serial-lane' in r.getMessage()]
        assert len(hits) == 1, 'the WARNING must not depend on an event store'

    def test_hostile_event_store_cannot_propagate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A store whose emit() raises must never wedge the dispatch path."""
        from orchestrator.merge_liveness import (  # noqa: PLC0415
            alarm_serial_lane_breach,
            check_serial_lane_tripwire,
        )

        assessment = check_serial_lane_tripwire(2, merge_ahead_bound=1, num_hosts=1)
        hostile = MagicMock()
        hostile.emit = MagicMock(side_effect=RuntimeError('boom'))
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = alarm_serial_lane_breach(
                assessment, event_store=hostile, task_id='5326', branch='task/5326'
            )

        # No veto channel exists for a caller to accidentally honour as a block.
        assert result is None
        hostile.emit.assert_called_once()


# ---------------------------------------------------------------------------
# Step 09 — the tripwire wired into the real dispatch chokepoint
#
# Real-git fixtures duplicated per-file (PRD D9: no cross-test-module imports
# of private fixtures) — mirrors test_merge_queue_concurrent_verify.py's
# ``_setup_repo`` / ``git_repo`` / ``git_config`` / ``git_ops`` / ``config`` /
# ``_make_branch_with_file``.
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
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
    """Single-host (no verify_runners) OrchestratorConfig."""
    return OrchestratorConfig(project_root=git_repo, git=git_config)


async def _make_branch_with_file(
    git_ops: GitOps, branch_name: str, filename: str, content: str
) -> Path:
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


async def _make_merged_item(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str
) -> RealMergeItem:
    """Build a genuinely-merged RealMergeItem with a real branch and request_id."""
    branch = f'task/{task_id}'
    worktree = await _make_branch_with_file(git_ops, branch, f'f_{task_id}.py', f'x = {task_id!r}\n')
    request = MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )
    merge_result = await git_ops.merge_to_main(worktree, branch)
    assert merge_result.success and merge_result.merge_worktree is not None
    return RealMergeItem(
        request=request,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=await git_ops.get_main_sha(),
        speculative=False,
    )


_TRIPWIRE_LOG_MARK = 'serial-lane tripwire BREACHED'
_TRIPWIRE_FAILOPEN_MARK = 'serial-lane tripwire check failed'


@pytest.mark.asyncio
class TestSerialLaneTripwireWiredIntoDispatch:
    """THE USER-OBSERVABLE SIGNAL — PRD §9 row 10, driven through the real
    ``SpeculativeMergeWorker._inflight_append`` chokepoint.

    Calling ``_inflight_append`` directly on a bare worker is the blessed
    pattern: ``_assert_single_writer`` short-circuits when ``expected_task is
    None``, and its docstring explicitly blesses the direct-call unit-test
    convention.

    HAZARD — ``_note_transition`` contaminates BOTH caplog AND the event store.
    On an unregistered/illegal request_id it logs at WARNING and calls
    ``_alarm_illegal_lifecycle_transition`` with ``event_store=self._event_store``,
    which can emit its OWN event into the very recording store these tests
    inspect.  So every item is registered through the lifecycle at DISPATCHING
    first (making DISPATCHING -> VERIFYING a legal move), AND — belt and braces
    — both channels are filtered by identity anyway.  NEVER assert "no events at
    all" or "no WARNING at all" at this layer.  (Step 07's unit-layer positive
    control CAN assert that, because it calls the alarm directly with no worker
    in play; the two are deliberately different.)
    """

    @staticmethod
    def _tripwire_events(rec) -> list:
        return [e for e in rec.events if e[0] == 'merge_serial_lane_breached']

    @staticmethod
    def _tripwire_logs(caplog: pytest.LogCaptureFixture) -> list:
        return [r for r in caplog.records if _TRIPWIRE_LOG_MARK in r.getMessage()]

    def _local_entry(
        self,
        worker,
        item: RealMergeItem,
        *,
        host: str = 'local',
        is_local: bool = True,
    ) -> InflightEntry:
        """Register *item* at DISPATCHING and build a frozen-countable entry.

        ``verify_task`` MUST be a non-None PENDING placeholder —
        ``_frozen_inflight_entries()`` only counts entries whose
        ``verify_task is not None``.  ``make_placeholder_future()`` is used
        rather than a real coroutine task because pyproject's
        ``filterwarnings`` escalates an unawaited-coroutine leak to a test
        error, and a bare Future has nothing to leak.
        """
        worker._register_item(item, initial=ItemLifecycleState.DISPATCHING)
        placeholder = make_placeholder_future()
        self._placeholders.append(placeholder)
        return InflightEntry(
            item=item,
            lease=HostLease(name=host, runner=object(), is_local=is_local),
            verify_task=placeholder,  # type: ignore[arg-type]
            merge_wt=item.merge_wt,
            was_speculative=False,
            started_at=time.time(),
        )

    @pytest.fixture(autouse=True)
    def _placeholder_teardown(self):
        self._placeholders: list = []
        yield
        for fut in self._placeholders:
            if not fut.done():
                fut.cancel()

    def _bare_worker(self, git_ops: GitOps, rec):
        """A worker at the production single-host shape: bound=1, num_hosts=1."""
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), event_store=rec)
        # speculation_depth stays at its default (_MERGE_AHEAD_BOUND = 1) and
        # _host_allocator stays None so num_hosts resolves to 1.
        assert worker._speculation_depth == 1
        assert worker._host_allocator is None
        return worker

    async def test_single_local_dispatch_emits_nothing(
        self, git_ops: GitOps, config: OrchestratorConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """POSITIVE CONTROL: one local dispatch at bound=1 must not trip."""
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        rec = _RecordingEventStore()
        worker = self._bare_worker(git_ops, rec)
        item_a = await _make_merged_item(git_ops, config, '9001')

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._inflight_append(self._local_entry(worker, item_a))

        assert self._tripwire_events(rec) == []
        assert self._tripwire_logs(caplog) == []

    async def test_second_concurrent_local_dispatch_fires_warning_and_event(
        self, git_ops: GitOps, config: OrchestratorConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PRD §9 row 10: the second concurrent local verify is the C4 breach."""
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        rec = _RecordingEventStore()
        worker = self._bare_worker(git_ops, rec)
        item_a = await _make_merged_item(git_ops, config, '9002')
        item_b = await _make_merged_item(git_ops, config, '9003')

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._inflight_append(self._local_entry(worker, item_a))
            worker._inflight_append(self._local_entry(worker, item_b))

        events = self._tripwire_events(rec)
        assert len(events) == 1, f'expected exactly one tripwire event; got {events!r}'
        _etype, payload = events[0]
        # The event is OBSERVED in the store, not inferred from a mock call.
        assert payload['task_id'] == '9003', 'the event must name the OFFENDING dispatch'
        assert payload['data'] == {
            'local_inflight': 2,
            'per_host_bound': 1,
            'merge_ahead_bound': 1,
            'num_hosts': 1,
            'branch': '9003',
            'request_id': item_b.request.request_id,
            'host': 'local',
        }

        logs = self._tripwire_logs(caplog)
        assert len(logs) == 1
        assert logs[0].levelno == logging.WARNING

    async def test_tripwire_does_not_block_the_dispatch(
        self, git_ops: GitOps, config: OrchestratorConfig
    ) -> None:
        """PRD C4 'no hard block': both entries land, nothing raised, returns None."""
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        rec = _RecordingEventStore()
        worker = self._bare_worker(git_ops, rec)
        item_a = await _make_merged_item(git_ops, config, '9004')
        item_b = await _make_merged_item(git_ops, config, '9005')
        entry_a = self._local_entry(worker, item_a)
        entry_b = self._local_entry(worker, item_b)

        assert worker._inflight_append(entry_a) is None
        assert worker._inflight_append(entry_b) is None

        assert len(worker._inflight) == 2
        assert list(worker._inflight) == [entry_a, entry_b]
        # inflight_by_host is the LOSSLESS occupancy view (the sibling by_host
        # collapses two entries sharing a host, last-writer-wins).
        assert worker.snapshot()['occupancy']['inflight_by_host']['local'] == ['9004', '9005']

    async def test_third_local_dispatch_fires_again(
        self, git_ops: GitOps, config: OrchestratorConfig
    ) -> None:
        """Every breach reports — no dedup, no rate-limit, no streak gate."""
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        rec = _RecordingEventStore()
        worker = self._bare_worker(git_ops, rec)
        for tid in ('9006', '9007', '9008'):
            worker._inflight_append(
                self._local_entry(worker, await _make_merged_item(git_ops, config, tid))
            )

        events = self._tripwire_events(rec)
        assert len(events) == 2, 'the 2nd AND 3rd dispatch must each report'
        assert [e[1]['data']['local_inflight'] for e in events] == [2, 3]

    async def test_remote_lease_dispatches_do_not_trip(
        self, git_ops: GitOps, config: OrchestratorConfig
    ) -> None:
        """Only LOCAL leases count — the sanctioned local+remote overlap is legal."""
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        rec = _RecordingEventStore()
        worker = self._bare_worker(git_ops, rec)
        item_a = await _make_merged_item(git_ops, config, '9009')
        item_b = await _make_merged_item(git_ops, config, '9010')
        worker._inflight_append(self._local_entry(worker, item_a, host='laptop', is_local=False))
        worker._inflight_append(self._local_entry(worker, item_b, host='desktop', is_local=False))
        assert self._tripwire_events(rec) == [], 'two REMOTE verifies are not a serial-lane breach'

        rec2 = _RecordingEventStore()
        worker2 = self._bare_worker(git_ops, rec2)
        item_c = await _make_merged_item(git_ops, config, '9011')
        item_d = await _make_merged_item(git_ops, config, '9012')
        worker2._inflight_append(self._local_entry(worker2, item_c))
        worker2._inflight_append(self._local_entry(worker2, item_d, host='laptop', is_local=False))
        assert self._tripwire_events(rec2) == [], 'local+remote overlap is the legal K=2 shape'

    async def test_real_host_allocator_does_not_break_the_hook(
        self, git_ops: GitOps, config: OrchestratorConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """REGRESSION GUARD — do not drop.

        Every other case leaves ``_host_allocator = None``, so none of them
        executes the branch that reads the allocator.  ``HostAllocator.host_names``
        is a **@property**: a ``host_names()`` CALL would raise ``TypeError:
        'list' object is not callable``, and the hook's deliberate fail-open
        ``except Exception`` would SWALLOW it — leaving the tripwire permanently
        dead in production while every other test here stayed green.  A
        test-passing, production-dead detector is the worst possible outcome for
        a detection net.
        """
        from _recording_event_store import _RecordingEventStore  # noqa: PLC0415

        rec = _RecordingEventStore()
        worker = self._bare_worker(git_ops, rec)
        worker._host_allocator = HostAllocator([])  # single local slot, no remotes
        assert worker._host_allocator.host_names == ['local']

        item_a = await _make_merged_item(git_ops, config, '9013')
        item_b = await _make_merged_item(git_ops, config, '9014')
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._inflight_append(self._local_entry(worker, item_a))
            worker._inflight_append(self._local_entry(worker, item_b))

        events = self._tripwire_events(rec)
        assert len(events) == 1, (
            'the breach must still fire with a REAL HostAllocator wired in — '
            'zero events here means the hook died inside its fail-open arm'
        )
        assert events[0][1]['data']['num_hosts'] == 1

        # The fail-open log line appearing IS the proof the hook silently died.
        assert [r for r in caplog.records if _TRIPWIRE_FAILOPEN_MARK in r.getMessage()] == []
