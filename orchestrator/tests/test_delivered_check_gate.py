"""Tests for the delivered-check dep-gate (capability-delivered-checks PRD,
task delta — plans/capability-delivered-checks-prd.md).

Covers ``orchestrator.delivered_checks`` (the grep/script runner) and the
scheduler-side dep-gate that consumes it: the ``delivered_check_cache`` arm
on ``Scheduler._deps_satisfied`` / ``Scheduler._eligible_for_dispatch``, the
per-tick sweep (``Scheduler._compute_delivered_check_cache`` +
``Scheduler._phase_delivered_check_gate``), and the hold-visibility event
(``Scheduler._note_delivered_hold`` / ``EventType.delivered_check_gate_held``).

Structurally mirrors ``test_scheduler.py``'s ``TestDepsSatisfiedExternalGate``
/ ``TestExternalDepGateHeld_DepsLive`` / ``TestAcquireNextExternalDepGate`` —
the delivered-check gate is a deliberate structural clone of the
cross-project external-dep gate (task 1580); see the plan's design
decisions for the point-by-point mirror rationale.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _recording_event_store import _RecordingEventStore
from pydantic import ValidationError

from orchestrator.config import DeliveredChecksConfig, OrchestratorConfig, RELOADABLE_FIELDS
from orchestrator.delivered_checks import DeliveredCheckResult, run_delivered_check
from orchestrator.event_store import EventType
from orchestrator.scheduler import Scheduler, TickContext

# ---------------------------------------------------------------------------
# TestRunnerGrepKind (task 2580 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestRunnerGrepKind:
    """``run_delivered_check``'s grep-kind branch: argv shape + rc mapping."""

    def _fake_runner(self, rc: int, out: str = '', err: str = ''):
        """Build an injected fake runner recording every argv it's called with."""
        calls: list[list[str]] = []

        async def _runner(argv, **kwargs):
            calls.append(argv)
            return (rc, out, err)

        return _runner, calls

    @pytest.mark.asyncio
    async def test_argv_shape_with_paths(self):
        runner, calls = self._fake_runner(rc=0)
        check = {
            'name': 'cap',
            'kind': 'grep',
            'pattern': 'FooBar',
            'expect': 'present',
            'paths': ['src/a.py', 'src/b.py'],
        }

        await run_delivered_check(check, project_root='/proj', ref='main', runner=runner)

        assert calls == [
            ['git', '-C', '/proj', 'grep', '-E', 'FooBar', 'main', '--', 'src/a.py', 'src/b.py']
        ]

    @pytest.mark.asyncio
    async def test_argv_omits_dashdash_when_paths_empty(self):
        runner, calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'FooBar', 'expect': 'present'}

        await run_delivered_check(check, project_root='/proj', ref='main', runner=runner)

        assert calls == [['git', '-C', '/proj', 'grep', '-E', 'FooBar', 'main']]

    @pytest.mark.asyncio
    async def test_default_ref_is_main(self):
        runner, calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'FooBar', 'expect': 'present'}

        # ref= omitted entirely — default must be 'main'.
        await run_delivered_check(check, project_root='/proj', runner=runner)

        assert calls == [['git', '-C', '/proj', 'grep', '-E', 'FooBar', 'main']]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('rc', 'expected'),
        [
            (0, DeliveredCheckResult.DELIVERED),
            (1, DeliveredCheckResult.FAILED),
            (2, DeliveredCheckResult.ERRORED),
            (128, DeliveredCheckResult.ERRORED),
        ],
    )
    async def test_expect_present_rc_mapping(self, rc, expected):
        """expect=present: DELIVERED on rc==0, FAILED on rc==1, ERRORED on rc>=2."""
        runner, _calls = self._fake_runner(rc=rc)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'x', 'expect': 'present'}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('rc', 'expected'),
        [
            (0, DeliveredCheckResult.FAILED),
            (1, DeliveredCheckResult.DELIVERED),
            (2, DeliveredCheckResult.ERRORED),
            (128, DeliveredCheckResult.ERRORED),
        ],
    )
    async def test_expect_absent_rc_mapping(self, rc, expected):
        """expect=absent: DELIVERED on rc==1, FAILED on rc==0, ERRORED on rc>=2."""
        runner, _calls = self._fake_runner(rc=rc)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'x', 'expect': 'absent'}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is expected


# ---------------------------------------------------------------------------
# TestRunnerScriptKind (task 2580 — step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------


class TestRunnerScriptKind:
    """``run_delivered_check``'s script-kind branch: argv/cwd shape + exit/timeout mapping."""

    def _fake_runner(self, rc: int = 0, out: str = '', err: str = ''):
        calls: list[tuple] = []

        async def _runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return (rc, out, err)

        return _runner, calls

    @pytest.mark.asyncio
    async def test_argv_and_cwd_shape(self):
        runner, calls = self._fake_runner(rc=0)
        check = {
            'name': 'cap',
            'kind': 'script',
            'script': 'scripts/check.sh',
            'args': ['--flag', 'value'],
            'timeout_secs': 30,
        }

        await run_delivered_check(check, project_root='/proj', runner=runner)

        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv == ['/proj/scripts/check.sh', '--flag', 'value']
        assert kwargs.get('cwd') == Path('/proj')

    @pytest.mark.asyncio
    async def test_exit_zero_delivered(self):
        runner, _calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is DeliveredCheckResult.DELIVERED

    @pytest.mark.asyncio
    async def test_nonzero_exit_failed(self):
        runner, _calls = self._fake_runner(rc=1)
        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is DeliveredCheckResult.FAILED

    @pytest.mark.asyncio
    async def test_timeout_errored(self):
        """A runner that raises TimeoutError (simulating asyncio.wait_for's
        own timeout firing, mirroring test_deterministic_runner.py's
        seam-injection precedent) must map to ERRORED — no real subprocess,
        no real wait, is needed to exercise this path."""
        called = False

        async def _runner(argv, **kwargs):
            nonlocal called
            called = True
            raise TimeoutError('simulated timeout')

        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=_runner)

        assert called, 'the injected runner must actually be invoked'
        assert result is DeliveredCheckResult.ERRORED

    @pytest.mark.asyncio
    async def test_missing_executable_errored(self):
        """A runner that raises OSError (e.g. FileNotFoundError for a
        missing/non-executable script) must map to ERRORED."""
        called = False

        async def _runner(argv, **kwargs):
            nonlocal called
            called = True
            raise FileNotFoundError('missing executable')

        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=_runner)

        assert called, 'the injected runner must actually be invoked'
        assert result is DeliveredCheckResult.ERRORED


# ---------------------------------------------------------------------------
# TestDepsSatisfiedDeliveredGate (task 2580 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestDepsSatisfiedDeliveredGate:
    """Unit tests for the ``delivered_check_cache`` boolean gate added to
    ``Scheduler._deps_satisfied``.  Mirrors ``TestDepsSatisfiedExternalGate``
    (test_scheduler.py): the gate is PURE (no side effects, no escalation
    calls).  It is opt-in — passing delivered_check_cache=None (the
    default) reproduces byte-identical legacy behaviour, and it is only
    consulted for a TERMINAL local dep whose ``tasks_by_id`` record carries
    a truthy ``metadata.delivered_checks``.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler.finish_startup()
        return scheduler

    _CHECKS = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]

    def _dependent(self, dep_id: str = '20') -> dict:
        return {
            'id': '10',
            'status': 'pending',
            'dependencies': [{'id': dep_id}],
            'metadata': {},
        }

    def _dep(self, dep_id: str = '20', status: str = 'done', with_checks: bool = True) -> dict:
        metadata = {'delivered_checks': self._CHECKS} if with_checks else {}
        return {'id': dep_id, 'status': status, 'dependencies': [], 'metadata': metadata}

    # --- delivered_check_cache unset (default None) → byte-identical -------

    def test_default_none_byte_identical_done_dep_with_checks(self, scheduler: Scheduler):
        """Cache unset (default None): a done dep with checks is satisfied
        purely off status, exactly like legacy behaviour — the arm never
        even looks at metadata.delivered_checks."""
        task = self._dependent()
        dep = self._dep(status='done')
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep}
        assert scheduler._deps_satisfied(task, status_map, tasks_by_id) is True

    # --- row 3 (predicate): done dep, cache True → satisfied ---------------

    def test_done_dep_checks_cached_true_satisfied(self, scheduler: Scheduler):
        task = self._dependent()
        dep = self._dep(status='done')
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep}
        assert (
            scheduler._deps_satisfied(
                task, status_map, tasks_by_id, delivered_check_cache={'20': True}
            )
            is True
        )

    # --- row 4 (predicate): done dep, cache False → NOT satisfied ----------

    def test_done_dep_checks_cached_false_not_satisfied(self, scheduler: Scheduler):
        task = self._dependent()
        dep = self._dep(status='done')
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep}
        assert (
            scheduler._deps_satisfied(
                task, status_map, tasks_by_id, delivered_check_cache={'20': False}
            )
            is False
        )

    # --- row 7 (predicate): dep absent from cache → NOT satisfied ----------

    def test_done_dep_checks_absent_from_cache_not_satisfied(self, scheduler: Scheduler):
        """A dep carrying checks but absent from the cache (errored /
        over-budget / not yet evaluated) is NOT satisfied — fail-safe wait."""
        task = self._dependent()
        dep = self._dep(status='done')
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep}
        assert (
            scheduler._deps_satisfied(task, status_map, tasks_by_id, delivered_check_cache={})
            is False
        )

    # --- row 10 (predicate half): cancelled dep with checks — trusts main --

    def test_cancelled_dep_checks_cached_false_not_satisfied(self, scheduler: Scheduler):
        """A cancelled dep still carrying checks is gated exactly like a
        done one — the predicate trusts main, not the status label."""
        task = self._dependent()
        dep = self._dep(status='cancelled')
        status_map = {'20': 'cancelled'}
        tasks_by_id = {'20': dep}
        assert (
            scheduler._deps_satisfied(
                task, status_map, tasks_by_id, delivered_check_cache={'20': False}
            )
            is False
        )

    def test_cancelled_dep_checks_cached_true_satisfied(self, scheduler: Scheduler):
        task = self._dependent()
        dep = self._dep(status='cancelled')
        status_map = {'20': 'cancelled'}
        tasks_by_id = {'20': dep}
        assert (
            scheduler._deps_satisfied(
                task, status_map, tasks_by_id, delivered_check_cache={'20': True}
            )
            is True
        )

    # --- dep without delivered_checks metadata: never consulted ------------

    def test_done_dep_without_checks_not_consulted(self, scheduler: Scheduler):
        task = self._dependent()
        dep = self._dep(status='done', with_checks=False)
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep}
        # Cache is non-empty but doesn't even mention '20' — since the dep
        # carries no delivered_checks metadata it must never be consulted.
        assert (
            scheduler._deps_satisfied(
                task, status_map, tasks_by_id, delivered_check_cache={'other': False}
            )
            is True
        )

    # --- tasks_by_id=None → arm disabled (byte-identical) -------------------

    def test_tasks_by_id_none_arm_disabled(self, scheduler: Scheduler):
        task = self._dependent()
        status_map = {'20': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, status_map, None, delivered_check_cache={'20': False}
            )
            is True
        )


# ---------------------------------------------------------------------------
# TestEligibilityForwardsDeliveredCache (task 2580 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestEligibilityForwardsDeliveredCache:
    """``Scheduler._eligible_for_dispatch`` must forward a
    ``delivered_check_cache`` kwarg into ``_deps_satisfied`` — mirrors how
    ``external_status_cache`` is already forwarded there.  Default (unset)
    reproduces byte-identical legacy behaviour.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler.finish_startup()
        return scheduler

    _CHECKS = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]

    def _dependent(self, dep_id: str = '20') -> dict:
        return {
            'id': '10',
            'status': 'pending',
            'dependencies': [{'id': dep_id}],
            'metadata': {},
        }

    def _dep(self, dep_id: str = '20') -> dict:
        return {
            'id': dep_id,
            'status': 'done',
            'dependencies': [],
            'metadata': {'delivered_checks': self._CHECKS},
        }

    def test_ineligible_when_cache_maps_dep_to_false(self, scheduler: Scheduler):
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = scheduler._eligible_for_dispatch(
            task, '10', status_map, tasks_by_id,
            delivered_check_cache={'20': False},
        )

        assert result == (False, None)

    def test_eligible_when_cache_maps_dep_to_true(self, scheduler: Scheduler):
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = scheduler._eligible_for_dispatch(
            task, '10', status_map, tasks_by_id,
            delivered_check_cache={'20': True},
        )

        assert result == (True, None)

    def test_default_unset_byte_identical(self, scheduler: Scheduler):
        """Without delivered_check_cache, a done dep carrying checks is
        eligible purely off status — legacy behaviour, byte-identical."""
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = scheduler._eligible_for_dispatch(task, '10', status_map, tasks_by_id)

        assert result == (True, None)


# ---------------------------------------------------------------------------
# TestTickContextField (task 2580 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestTickContextField:
    """``TickContext`` must carry a ``delivered_check_cache`` field
    defaulting to an empty dict, mirroring ``external_cache``."""

    def test_delivered_check_cache_field_defaults_empty_dict(self):
        ctx = TickContext(tasks=[], status_map={}, tasks_by_id={})
        assert ctx.delivered_check_cache == {}

    def test_delivered_check_cache_field_accepts_constructor_kwarg(self):
        ctx = TickContext(
            tasks=[], status_map={}, tasks_by_id={},
            delivered_check_cache={'20': True},
        )
        assert ctx.delivered_check_cache == {'20': True}

    def test_delivered_check_cache_default_not_shared_across_instances(self):
        """Default factory must not share a mutable default across instances."""
        ctx1 = TickContext(tasks=[], status_map={}, tasks_by_id={})
        ctx2 = TickContext(tasks=[], status_map={}, tasks_by_id={})
        ctx1.delivered_check_cache['x'] = True
        assert ctx2.delivered_check_cache == {}


# ---------------------------------------------------------------------------
# TestDeliveredChecksConfig (task 2580 — step-9 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


class TestDeliveredChecksConfig:
    """``DeliveredChecksConfig`` sub-model + its green-tier RELOADABLE_FIELDS
    leaf.  Task 2580 (delta) owns only ``max_checks_per_tick`` — the sweep's
    per-tick evaluation budget; task 2583 (epsilon) layers the remaining
    knobs (enabled/grace_cycles/check_timeout_secs) onto this same
    sub-model.
    """

    def test_default_max_checks_per_tick_is_50(self):
        config = OrchestratorConfig()
        assert config.delivered_checks.max_checks_per_tick == 50

    def test_max_checks_per_tick_rejects_zero(self):
        with pytest.raises(ValidationError):
            DeliveredChecksConfig(max_checks_per_tick=0)

    def test_max_checks_per_tick_rejects_negative(self):
        with pytest.raises(ValidationError):
            DeliveredChecksConfig(max_checks_per_tick=-1)

    def test_max_checks_per_tick_is_hot_reloadable(self):
        """The scheduler-tuning green tier: an operator may retune the sweep
        budget via mcp__escalation__reload_config without a restart."""
        assert 'delivered_checks.max_checks_per_tick' in RELOADABLE_FIELDS


# ---------------------------------------------------------------------------
# TestNoteDeliveredHold (task 2580 — step-11 RED / step-12 GREEN)
# ---------------------------------------------------------------------------


class TestNoteDeliveredHold:
    """``Scheduler._note_delivered_hold`` — hold-visibility event on a
    dedicated ``_streak_delivered_hold`` counter.  Mirrors
    ``_note_external_hold``'s bump-then-emit shape, but WITHOUT threshold
    gating: task 2580 (delta) has no grace_cycles config yet (task 2583/
    epsilon layers threshold-gated escalation on top with its own counter),
    so every held tick emits ``EventType.delivered_check_gate_held`` with
    the running streak count.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config, event_store=_RecordingEventStore())  # type: ignore[arg-type]
        scheduler.finish_startup()
        return scheduler

    def _held_events(self, scheduler: Scheduler) -> list[tuple[str, dict]]:
        _event_store = scheduler.event_store
        assert _event_store is not None
        return [
            (evt, data)
            for evt, data in _event_store.events  # type: ignore[attr-defined]
            if evt == str(EventType.delivered_check_gate_held)
        ]

    def test_first_call_bumps_streak_to_one_and_emits_event(self, scheduler: Scheduler):
        detail = {'name': 'cap-one', 'dep_id': '20', 'main_sha': 'abc123', 'kind': 'grep'}

        scheduler._note_delivered_hold('T', detail=detail)

        assert scheduler._streak_delivered_hold.value('T') == 1
        events = self._held_events(scheduler)
        assert len(events) == 1
        _evt_type, evt_data = events[0]
        assert evt_data['task_id'] == 'T'
        assert evt_data['data'].get('ticks') == 1
        assert evt_data['data'].get('detail') == detail

    def test_repeated_calls_increment_streak_and_emit_every_tick(self, scheduler: Scheduler):
        """Unlike _note_external_hold, delta has no threshold — every call
        both bumps the streak AND emits (pure per-tick visibility)."""
        detail = {'name': 'cap-one', 'dep_id': '20', 'main_sha': 'abc123', 'kind': 'grep'}

        for _ in range(3):
            scheduler._note_delivered_hold('T', detail=detail)

        assert scheduler._streak_delivered_hold.value('T') == 3
        events = self._held_events(scheduler)
        assert [data['data'].get('ticks') for _evt, data in events] == [1, 2, 3]

    def test_detail_names_the_failed_check(self, scheduler: Scheduler):
        """The payload detail dict must name the failed check (name/dep_id/
        main_sha/kind) for epsilon (task 2583) to consume when layering
        grace-streak escalation on top."""
        detail = {'name': 'cap-two', 'dep_id': '30', 'main_sha': 'deadbeef', 'kind': 'script'}

        scheduler._note_delivered_hold('T2', detail=detail)

        events = self._held_events(scheduler)
        assert events[0][1]['data']['detail'] == detail

    def test_detail_defaults_to_none(self, scheduler: Scheduler):
        """detail is optional — omitting it still bumps the streak and emits."""
        scheduler._note_delivered_hold('T3')

        assert scheduler._streak_delivered_hold.value('T3') == 1
        events = self._held_events(scheduler)
        assert events[0][1]['data'].get('detail') is None

    def test_independent_task_ids_have_independent_streaks(self, scheduler: Scheduler):
        scheduler._note_delivered_hold('A')
        scheduler._note_delivered_hold('A')
        scheduler._note_delivered_hold('B')

        assert scheduler._streak_delivered_hold.value('A') == 2
        assert scheduler._streak_delivered_hold.value('B') == 1
