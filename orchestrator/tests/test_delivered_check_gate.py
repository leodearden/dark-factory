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
from unittest.mock import AsyncMock

import pytest
from _recording_event_store import _RecordingEventStore
from pydantic import ValidationError

from orchestrator.config import RELOADABLE_FIELDS, DeliveredChecksConfig, OrchestratorConfig
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

    def test_first_call_warns_then_subsequent_calls_log_at_debug(
        self, scheduler: Scheduler, caplog
    ):
        """reviewer_comprehensive amendment: the event fires on EVERY held
        tick (see test_repeated_calls_increment_streak_and_emit_every_tick
        above — unaffected by this change), but the WARNING *log* line is
        bounded to the first tick of a hold episode. Without this, a
        steady-state hold (a failing check that stays failed until a fix
        lands on main) would WARNING-log once per tick forever."""
        import logging

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            scheduler._note_delivered_hold('T', detail={'name': 'cap-one'})
            after_first = len(caplog.records)
            scheduler._note_delivered_hold('T', detail={'name': 'cap-one'})
            scheduler._note_delivered_hold('T', detail={'name': 'cap-one'})

        assert after_first == 1, (
            f'expected exactly one WARNING on the first held tick; got '
            f'{[r.getMessage() for r in caplog.records[:after_first]]!r}'
        )
        assert len(caplog.records) == 1, (
            'subsequent held ticks must NOT log at WARNING (steady-state '
            f'log spam); got {[r.getMessage() for r in caplog.records]!r}'
        )
        # The event, unlike the log, is unaffected — still fires every tick.
        assert scheduler._streak_delivered_hold.value('T') == 3
        assert len(self._held_events(scheduler)) == 3


# ---------------------------------------------------------------------------
# TestComputeDeliveredCheckCache (task 2580 — step-13 RED / step-14 GREEN)
# ---------------------------------------------------------------------------


class TestComputeDeliveredCheckCache:
    """``Scheduler._compute_delivered_check_cache`` — the per-tick sweep
    evaluating every distinct terminal local dep carrying
    ``metadata.delivered_checks`` against the committed ``main`` tree, and
    building the ``{dep_task_id: passed}`` projection ``_deps_satisfied``
    consumes. Injects a fake ``_resolve_main_sha`` (instance override — no
    real git repo needed) and monkeypatches the module-level
    ``orchestrator.scheduler.run_delivered_check`` import to record calls
    without shelling out.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config, event_store=_RecordingEventStore())  # type: ignore[arg-type]
        scheduler.finish_startup()
        return scheduler

    _ONE_CHECK = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]
    _TWO_CHECKS = [
        {'name': 'cap-a', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'},
        {'name': 'cap-b', 'kind': 'grep', 'pattern': 'bar', 'expect': 'present'},
    ]

    def _dependent(self, task_id: str = '10', dep_id: str = '20') -> dict:
        return {
            'id': task_id,
            'status': 'pending',
            'dependencies': [{'id': dep_id}],
            'metadata': {},
        }

    def _dep(self, dep_id: str = '20', status: str = 'done', checks: list | None = None) -> dict:
        checks = self._ONE_CHECK if checks is None else checks
        return {
            'id': dep_id,
            'status': status,
            'dependencies': [],
            'metadata': {'delivered_checks': checks},
        }

    def _fake_sha(self, sha: str = 'sha1'):
        """Fake ``_resolve_main_sha`` recording how many times it's called."""
        calls = {'n': 0}

        async def _resolve():
            calls['n'] += 1
            return sha

        return _resolve, calls

    def _fake_runner(self, results: dict):
        """Fake ``run_delivered_check`` keyed by ``check['name']`` ->
        ``DeliveredCheckResult`` (or an ``Exception`` instance to raise).
        Records every check name it's invoked with, in call order."""
        calls: list[str] = []

        async def _fake(check, *, project_root, ref='main'):
            calls.append(check['name'])
            outcome = results[check['name']]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return _fake, calls

    def _held_events(self, scheduler: Scheduler) -> list[tuple[str, dict]]:
        _event_store = scheduler.event_store
        assert _event_store is not None
        return [
            (evt, data)
            for evt, data in _event_store.events  # type: ignore[attr-defined]
            if evt == str(EventType.delivered_check_gate_held)
        ]

    # --- (a) no checked deps -> {} and the SHA resolver is NEVER called ----

    @pytest.mark.asyncio
    async def test_no_checked_deps_returns_empty_and_never_resolves_sha(
        self, scheduler: Scheduler
    ):
        fake_resolve, sha_calls = self._fake_sha()
        scheduler._resolve_main_sha = fake_resolve
        task = self._dependent()
        dep = {'id': '20', 'status': 'done', 'dependencies': [], 'metadata': {}}
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert result == {}
        assert sha_calls['n'] == 0

    # --- (b) done dep, all checks DELIVERED -> {dep: True} -----------------

    @pytest.mark.asyncio
    async def test_done_dep_all_checks_delivered_projects_true(
        self, scheduler: Scheduler, monkeypatch
    ):
        fake_resolve, sha_calls = self._fake_sha('sha1')
        scheduler._resolve_main_sha = fake_resolve
        fake_runner, runner_calls = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert result == {'20': True}
        assert sha_calls['n'] == 1
        assert runner_calls == ['cap-one']

    # --- (c) done dep, one check FAILED -> {dep: False} + hold (row 4) -----

    @pytest.mark.asyncio
    async def test_done_dep_one_check_failed_projects_false_and_holds(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, _calls = self._fake_runner({'cap-one': DeliveredCheckResult.FAILED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert result == {'20': False}
        assert scheduler._streak_delivered_hold.value('10') == 1
        held = self._held_events(scheduler)
        assert len(held) == 1
        _evt, data = held[0]
        assert data['task_id'] == '10'
        assert data['data']['detail'] == {
            'name': 'cap-one', 'dep_id': '20', 'main_sha': 'sha1', 'kind': 'grep',
        }

    # --- (d) runner ERRORED -> absent, uncached, no hold, no streak (row 7) -

    @pytest.mark.asyncio
    async def test_runner_errored_leaves_dep_absent_no_cache_no_hold(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, _calls = self._fake_runner({'cap-one': DeliveredCheckResult.ERRORED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert result == {}
        assert ('20', 'sha1') not in scheduler._delivered_check_cache
        assert scheduler._streak_delivered_hold.value('10') == 0
        assert self._held_events(scheduler) == []

    # --- (e) budget: max_checks_per_tick=1, 2-check dep -> deferred --------

    @pytest.mark.asyncio
    async def test_budget_defers_second_check_of_a_two_check_dep(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler.config.delivered_checks.max_checks_per_tick = 1
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({
            'cap-a': DeliveredCheckResult.DELIVERED,
            'cap-b': DeliveredCheckResult.DELIVERED,
        })
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep(checks=self._TWO_CHECKS)
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert result == {}
        assert calls == ['cap-a'], 'the second check must NOT run once the budget is spent'
        assert ('20', 'sha1') not in scheduler._delivered_check_cache

    # --- (f) cache hit: same-sha re-sweep does NOT re-invoke the runner ----

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_reinvoke_runner(self, scheduler: Scheduler, monkeypatch):
        scheduler._resolve_main_sha, sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert first == {'20': True}
        assert second == {'20': True}
        assert calls == ['cap-one'], 'only ONE invocation total across both sweeps'
        assert sha_calls['n'] == 2, 'the SHA resolver itself still runs every tick'

    # --- (g) self-heal: SHA advance prunes stale cache + clears streak -----

    @pytest.mark.asyncio
    async def test_sha_advance_self_heals_and_clears_hold_streak(
        self, scheduler: Scheduler, monkeypatch
    ):
        sha_box = {'value': 'sha1'}

        async def _resolve():
            return sha_box['value']

        scheduler._resolve_main_sha = _resolve
        fake_runner, _calls = self._fake_runner({'cap-one': DeliveredCheckResult.FAILED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        assert first == {'20': False}
        assert scheduler._streak_delivered_hold.value('10') == 1

        # Main advances; the check now resolves DELIVERED.
        sha_box['value'] = 'sha2'
        fake_runner2, calls2 = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner2)

        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert second == {'20': True}
        assert calls2 == ['cap-one'], 'stale-sha cache entry must be pruned, forcing re-invoke'
        assert scheduler._streak_delivered_hold.value('10') == 0
        assert ('20', 'sha1') not in scheduler._delivered_check_cache
        assert ('20', 'sha2') in scheduler._delivered_check_cache

    # --- (h) cached-False dep still holds + emits detail on every sweep ----

    @pytest.mark.asyncio
    async def test_cached_false_dep_emits_hold_on_every_sweep(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Mirrors (f) ``test_cache_hit_does_not_reinvoke_runner`` but with a
        FAILED check: a dep that failed on a prior tick and is now served
        from the ``(dep, sha)`` cache-hit branch must STILL emit a hold
        event with a meaningful (non-``None``) detail on every held tick —
        not just the first, uncached tick."""
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({'cap-one': DeliveredCheckResult.FAILED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert first == {'20': False}
        assert second == {'20': False}
        assert calls == ['cap-one'], 'only ONE invocation total — the 2nd sweep is a cache hit'
        held = self._held_events(scheduler)
        assert len(held) == 2, 'delivered_check_gate_held must fire on EVERY held tick'
        assert scheduler._streak_delivered_hold.value('10') == 2
        for _evt, data in held:
            assert data['task_id'] == '10'
            assert data['data']['detail'] == {
                'name': 'cap-one', 'dep_id': '20', 'main_sha': 'sha1', 'kind': 'grep',
            }

    # --- (i) cached-False detail replays the check that ACTUALLY failed ----

    @pytest.mark.asyncio
    async def test_cached_false_detail_names_actual_failed_check_not_first(
        self, scheduler: Scheduler, monkeypatch
    ):
        """reviewer_comprehensive amendment: a dep with TWO checks where the
        FIRST DELIVERS and the SECOND FAILS must have its cached-False hold
        detail name the check that actually failed (cap-b) — never the
        dep's first ``delivered_checks`` entry (cap-a, which passed) — on
        BOTH the uncached (first) sweep and a cache-hit (second) sweep at
        the same main SHA. Guards against reconstructing the detail from
        ``checks[0]``, which would misname a passing check as the failure.
        """
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({
            'cap-a': DeliveredCheckResult.DELIVERED,
            'cap-b': DeliveredCheckResult.FAILED,
        })
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep(checks=self._TWO_CHECKS)  # [cap-a, cap-b] — cap-a is checks[0]
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert first == {'20': False}
        assert second == {'20': False}
        assert calls == ['cap-a', 'cap-b'], 'both checks ran, but only on the uncached sweep'
        held = self._held_events(scheduler)
        assert len(held) == 2
        expected_detail = {'name': 'cap-b', 'dep_id': '20', 'main_sha': 'sha1', 'kind': 'grep'}
        for _evt, data in held:
            assert data['data']['detail'] == expected_detail, (
                f"must name the check that actually FAILED (cap-b), not checks[0] "
                f"(cap-a, which passed); got {data['data']['detail']!r}"
            )

    # --- (j) persistently-ERRORED dep: bounded diagnostic, never backed off -

    @pytest.mark.asyncio
    async def test_errored_dep_logs_bounded_diagnostic_and_never_backs_off(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """reviewer_comprehensive amendment: a dep whose check stays
        ERRORED across many consecutive sweeps must not be COMPLETELY
        silent (liveness) — but the diagnostic WARNING log is bounded (not
        per-tick), and the runner is NEVER backed off: it must keep
        re-invoking on every single sweep so a fix landing on main
        self-heals immediately (row 7 — see
        ``test_row7_runner_errored_withheld_then_dispatches_once_healthy``
        in ``TestAcquireNextDeliveredGate``, which relies on exactly this).
        """
        import logging

        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({'cap-one': DeliveredCheckResult.ERRORED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            for _ in range(19):
                await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
            assert not caplog.records, (
                'must not WARNING-log before the bounded threshold is reached; '
                f'got {[r.getMessage() for r in caplog.records]!r}'
            )
            await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert calls == ['cap-one'] * 20, (
            f'no backoff: the runner must re-invoke on EVERY sweep even past '
            f'the log threshold; got {len(calls)} invocations'
        )
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            f'expected a bounded diagnostic WARNING by the 20th consecutive '
            f'ERRORED sweep; got records={caplog.records!r}'
        )
        # Row 7's dispatch-gate contract is untouched by the diagnostic log:
        # still no hold event / no _streak_delivered_hold bump for errored.
        assert self._held_events(scheduler) == []
        assert scheduler._streak_delivered_hold.value('10') == 0


# ---------------------------------------------------------------------------
# TestAcquireNextDeliveredGate (task 2580 — step-17 RED / step-18 GREEN)
# ---------------------------------------------------------------------------


class TestAcquireNextDeliveredGate:
    """``acquire_next`` wires the delivered-check dep-gate end-to-end: one
    per-tick sweep, correct dispatch decisions.  Mirrors
    ``TestAcquireNextExternalDepGate`` (test_scheduler.py) but for a LOCAL
    dep carrying ``metadata.delivered_checks`` instead of a cross-project
    external dep.

    Boundary rows under test (plans/capability-delivered-checks-prd.md
    §Boundary):
    - row 3 (transparent): done dep, check resolves DELIVERED → dispatched.
    - row 4 (headline): done dep, check resolves FAILED → NOT dispatched,
      a ``delivered_check_gate_held`` event is recorded, streak bumps.
    - row 6 (headline): a withheld dep self-heals once main advances to a
      SHA where the check now resolves DELIVERED → dispatched next tick,
      hold streak clears.
    - row 7: a runner ERROR withholds fail-safe with NO hold event/streak
      bump that tick; once the runner recovers, the dependent dispatches.

    Task delta (2580) has no grace-streak escalation counter or callback
    yet — task epsilon (2583) layers that on top — so there is no
    ``on_*_block``-style callback to assert on here; the only per-tick
    contract is withhold + ``_note_delivered_hold`` visibility.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=2)
        scheduler = Scheduler(config, event_store=_RecordingEventStore())  # type: ignore[arg-type]
        scheduler.finish_startup()
        return scheduler

    _CHECKS = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]

    def _dep(self, dep_id: str = '20', status: str = 'done') -> dict:
        return {
            'id': dep_id,
            'status': status,
            'dependencies': [],
            'metadata': {'delivered_checks': self._CHECKS},
        }

    def _dependent(self, tid: str = '10', dep_id: str = '20') -> dict:
        return {
            'id': tid,
            'title': f'Task {tid}',
            'status': 'pending',
            'dependencies': [{'id': dep_id}],
            'metadata': {'files': ['backend']},
        }

    def _held_events(self, scheduler: Scheduler) -> list[tuple[str, dict]]:
        _event_store = scheduler.event_store
        assert _event_store is not None
        return [
            (evt, data)
            for evt, data in _event_store.events  # type: ignore[attr-defined]
            if evt == str(EventType.delivered_check_gate_held)
        ]

    def _fake_sha(self, sha: str = 'sha1'):
        async def _resolve():
            return sha

        return _resolve

    def _fake_runner(self, outcome):
        """Fake ``run_delivered_check`` returning (or raising) *outcome* for
        every check it's invoked with."""

        async def _fake(check, *, project_root, ref='main'):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return _fake

    # --- row 3 (transparent): check DELIVERED -> dispatched, no hold -------

    @pytest.mark.asyncio
    async def test_row3_check_delivered_dispatches(self, scheduler: Scheduler, monkeypatch):
        scheduler.get_tasks = AsyncMock(return_value=[self._dep(), self._dependent()])
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.DELIVERED),
        )

        result = await scheduler.acquire_next()

        assert result is not None and result.task_id == '10', (
            f'Delivered check → should dispatch; got {result!r}'
        )
        assert self._held_events(scheduler) == []

    # --- row 4 (headline): check FAILED -> withheld + hold event -----------

    @pytest.mark.asyncio
    async def test_row4_check_failed_not_dispatched_holds(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler.get_tasks = AsyncMock(return_value=[self._dep(), self._dependent()])
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.FAILED),
        )

        result = await scheduler.acquire_next()

        assert result is None, f'Failed check → must NOT dispatch; got {result!r}'
        assert scheduler._streak_delivered_hold.value('10') == 1
        held = self._held_events(scheduler)
        assert len(held) == 1
        _evt, data = held[0]
        assert data['task_id'] == '10'
        assert data['data']['detail'] == {
            'name': 'cap-one', 'dep_id': '20', 'main_sha': 'sha1', 'kind': 'grep',
        }

    # --- row 6 (headline): SHA advance self-heals ---------------------------

    @pytest.mark.asyncio
    async def test_row6_sha_advance_self_heals_dispatches_next_tick(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler.get_tasks = AsyncMock(return_value=[self._dep(), self._dependent()])
        sha_box = {'value': 'sha1'}

        async def _resolve():
            return sha_box['value']

        scheduler._resolve_main_sha = _resolve
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.FAILED),
        )

        first = await scheduler.acquire_next()
        assert first is None
        assert scheduler._streak_delivered_hold.value('10') == 1
        assert len(self._held_events(scheduler)) == 1

        # Main advances; a new commit makes the check now resolve DELIVERED.
        sha_box['value'] = 'sha2'
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.DELIVERED),
        )

        second = await scheduler.acquire_next()

        assert second is not None and second.task_id == '10', (
            f'Self-heal on new main SHA → should dispatch; got {second!r}'
        )
        assert scheduler._streak_delivered_hold.value('10') == 0

    # --- row 7: runner ERRORED -> fail-safe withhold, no hold/streak -------

    @pytest.mark.asyncio
    async def test_row7_runner_errored_withheld_then_dispatches_once_healthy(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler.get_tasks = AsyncMock(return_value=[self._dep(), self._dependent()])
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.ERRORED),
        )

        first = await scheduler.acquire_next()

        assert first is None, f'Runner error → must NOT dispatch; got {first!r}'
        assert self._held_events(scheduler) == []
        assert scheduler._streak_delivered_hold.value('10') == 0

        # The runner recovers on the next tick.
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.DELIVERED),
        )

        second = await scheduler.acquire_next()

        assert second is not None and second.task_id == '10', (
            f'Runner recovered → should dispatch; got {second!r}'
        )
