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

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _recording_event_store import _RecordingEventStore
from pydantic import ValidationError

from orchestrator.config import RELOADABLE_FIELDS, DeliveredChecksConfig, OrchestratorConfig
from orchestrator.delivered_checks import (
    DeliveredCheckResult,
    DeliveredChecksBlock,
    DeliveredChecksVerdict,
    gate_mark_done_on_delivered_checks,
    run_delivered_check,
    verify_delivered_checks_on_main,
)
from orchestrator.event_store import EventType
from orchestrator.scheduler import (
    _CONTINUE,
    Scheduler,
    SchedulerCallbacks,
    TickContext,
    _build_delivered_check_escalation,
    _delivered_checks_descriptor_digest,
)

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
            ['git', '-C', '/proj', 'grep', '-E', '-e', 'FooBar', 'main', '--', 'src/a.py', 'src/b.py']
        ]

    @pytest.mark.asyncio
    async def test_argv_omits_dashdash_when_paths_empty(self):
        runner, calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'FooBar', 'expect': 'present'}

        await run_delivered_check(check, project_root='/proj', ref='main', runner=runner)

        assert calls == [['git', '-C', '/proj', 'grep', '-E', '-e', 'FooBar', 'main']]

    @pytest.mark.asyncio
    async def test_default_ref_is_main(self):
        runner, calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'FooBar', 'expect': 'present'}

        # ref= omitted entirely — default must be 'main'.
        await run_delivered_check(check, project_root='/proj', runner=runner)

        assert calls == [['git', '-C', '/proj', 'grep', '-E', '-e', 'FooBar', 'main']]

    @pytest.mark.asyncio
    async def test_pattern_starting_with_dash_is_not_parsed_as_an_option(self):
        """reviewer_comprehensive amendment: a pattern beginning with '-'
        must be passed as the literal search pattern (via the ``-e``
        separator), never mistaken by ``git grep`` for another option."""
        runner, calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': '-foo', 'expect': 'present'}

        await run_delivered_check(check, project_root='/proj', runner=runner)

        assert calls == [['git', '-C', '/proj', 'grep', '-E', '-e', '-foo', 'main']]

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

    # --- terminal_dep_records fallback (task 2692 — step-3 RED / step-4 GREEN)
    #
    # The active-only get_tasks fetch that seeds tasks_by_id excludes
    # DONE/CANCELLED producers, so a just-completed dep carrying
    # metadata.delivered_checks is genuinely ABSENT from tasks_by_id (not
    # smuggled in like the fixtures above). terminal_dep_records is the
    # dedicated fallback the scheduler backfills for exactly this case.

    def test_terminal_dep_records_fallback_consulted_when_absent_from_tasks_by_id_cached_false(
        self, scheduler: Scheduler
    ):
        """Dep genuinely absent from tasks_by_id (active-only fetch excluded
        it) but present in terminal_dep_records — cache=False must still
        withhold, exactly as if the record had been found in tasks_by_id."""
        task = self._dependent()
        dep = self._dep(status='done')
        status_map = {'20': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, status_map, {},
                delivered_check_cache={'20': False},
                terminal_dep_records={'20': dep},
            )
            is False
        )

    def test_terminal_dep_records_fallback_consulted_when_absent_from_tasks_by_id_cached_true(
        self, scheduler: Scheduler
    ):
        task = self._dependent()
        dep = self._dep(status='done')
        status_map = {'20': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, status_map, {},
                delivered_check_cache={'20': True},
                terminal_dep_records={'20': dep},
            )
            is True
        )

    def test_terminal_dep_records_fallback_absent_from_cache_not_satisfied(
        self, scheduler: Scheduler
    ):
        """Dep absent from BOTH tasks_by_id and the cache, but present in
        terminal_dep_records — fail-safe wait (not satisfied), mirroring
        test_done_dep_checks_absent_from_cache_not_satisfied above."""
        task = self._dependent()
        dep = self._dep(status='done')
        status_map = {'20': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, status_map, {},
                delivered_check_cache={},
                terminal_dep_records={'20': dep},
            )
            is False
        )

    def test_terminal_dep_records_unset_byte_identical(self, scheduler: Scheduler):
        """terminal_dep_records omitted (default None): a dep absent from
        tasks_by_id is simply skipped by the arm — byte-identical to
        legacy behaviour (no fallback source is consulted at all)."""
        task = self._dependent()
        status_map = {'20': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, status_map, {}, delivered_check_cache={'20': False}
            )
            is True
        )

    def test_terminal_dep_records_explicit_none_byte_identical(self, scheduler: Scheduler):
        """Passing terminal_dep_records=None explicitly is identical to
        omitting it — the fallback stays inert."""
        task = self._dependent()
        status_map = {'20': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, status_map, {},
                delivered_check_cache={'20': False},
                terminal_dep_records=None,
            )
            is True
        )

    def test_terminal_dep_records_fallback_not_consulted_when_dep_already_in_tasks_by_id(
        self, scheduler: Scheduler
    ):
        """When the dep record IS already present in tasks_by_id,
        terminal_dep_records must never be consulted — tasks_by_id wins.
        Proven behaviourally: tasks_by_id's record carries checks (so the
        arm applies and cache=False blocks); terminal_dep_records carries a
        record for the SAME id with NO checks, which would make the arm a
        silent no-op (returns True) if it were consulted instead."""
        task = self._dependent()
        dep_in_tasks_by_id = self._dep(status='done', with_checks=True)
        dep_in_terminal_records = self._dep(status='done', with_checks=False)
        status_map = {'20': 'done'}
        assert (
            scheduler._deps_satisfied(
                task, status_map, {'20': dep_in_tasks_by_id},
                delivered_check_cache={'20': False},
                terminal_dep_records={'20': dep_in_terminal_records},
            )
            is False
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

    # --- terminal_dep_records forwarding (task 2692 — step-5 RED / step-6 GREEN)
    #
    # The dep is genuinely absent from tasks_by_id here (unlike the fixtures
    # above, which smuggle it in directly) — mirroring the real active-only
    # get_tasks fetch that excludes a done/cancelled producer.

    def test_ineligible_when_terminal_dep_records_fallback_maps_dep_to_false(
        self, scheduler: Scheduler
    ):
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'10': task}

        result = scheduler._eligible_for_dispatch(
            task, '10', status_map, tasks_by_id,
            delivered_check_cache={'20': False},
            terminal_dep_records={'20': dep},
        )

        assert result == (False, None)

    def test_eligible_when_terminal_dep_records_fallback_maps_dep_to_true(
        self, scheduler: Scheduler
    ):
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'10': task}

        result = scheduler._eligible_for_dispatch(
            task, '10', status_map, tasks_by_id,
            delivered_check_cache={'20': True},
            terminal_dep_records={'20': dep},
        )

        assert result == (True, None)

    def test_terminal_dep_records_default_unset_byte_identical(self, scheduler: Scheduler):
        """Without terminal_dep_records, a dep absent from tasks_by_id is
        simply skipped by the delivered-check arm — eligible purely off
        status, byte-identical to legacy behaviour."""
        task = self._dependent()
        status_map = {'20': 'done'}
        tasks_by_id = {'10': task}

        result = scheduler._eligible_for_dispatch(
            task, '10', status_map, tasks_by_id,
            delivered_check_cache={'20': False},
        )

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
        assert ctx1.delivered_check_cache is not None
        ctx1.delivered_check_cache['x'] = True
        assert ctx2.delivered_check_cache == {}

    def test_delivered_check_cache_accepts_none(self):
        """task 2583 (epsilon): the annotation widens to dict[str, bool] |
        None so the kill switch (delivered_checks.enabled=False) can signal
        'gate off' distinctly from 'sweep ran, found nothing checked' ({})."""
        ctx = TickContext(
            tasks=[], status_map={}, tasks_by_id={}, delivered_check_cache=None,
        )
        assert ctx.delivered_check_cache is None


# ---------------------------------------------------------------------------
# TestTerminalDepRecordsField (task 2692 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestTerminalDepRecordsField:
    """``TickContext`` must carry a ``terminal_dep_records`` field defaulting
    to an empty dict, mirroring ``external_cache`` / ``delivered_check_cache``.

    Populated by ``_phase_backfill_terminal_dep_records`` with fetched
    TERMINAL dep records that are missing from ``tasks_by_id`` (the
    active-only ``get_tasks`` fetch excludes done/cancelled producers) —
    see task 2692's root-cause analysis. Consulted ONLY by the two
    delivered-check consumers (``_deps_satisfied``,
    ``_compute_delivered_check_cache``) as a purely additive fallback —
    never merged into ``tasks_by_id`` itself.
    """

    def test_terminal_dep_records_field_defaults_empty_dict(self):
        ctx = TickContext(tasks=[], status_map={}, tasks_by_id={})
        assert ctx.terminal_dep_records == {}

    def test_terminal_dep_records_field_accepts_constructor_kwarg(self):
        ctx = TickContext(
            tasks=[], status_map={}, tasks_by_id={},
            terminal_dep_records={'20': {'id': '20', 'status': 'done'}},
        )
        assert ctx.terminal_dep_records == {'20': {'id': '20', 'status': 'done'}}

    def test_terminal_dep_records_default_not_shared_across_instances(self):
        """Default factory must not share a mutable default across instances."""
        ctx1 = TickContext(tasks=[], status_map={}, tasks_by_id={})
        ctx2 = TickContext(tasks=[], status_map={}, tasks_by_id={})
        ctx1.terminal_dep_records['x'] = {'id': 'x'}
        assert ctx2.terminal_dep_records == {}


# ---------------------------------------------------------------------------
# TestPhaseBackfillTerminalDepRecords (task 2692 — step-9 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


class TestPhaseBackfillTerminalDepRecords:
    """``Scheduler._phase_backfill_terminal_dep_records`` — the new per-tick
    phase that fetches TERMINAL dep records missing from ``ctx.tasks_by_id``
    (the active-only ``get_tasks`` fetch excludes done/cancelled producers)
    into the dedicated ``ctx.terminal_dep_records`` fallback, via the lean
    per-id ``get_task`` primitive. Only PENDING tasks' deps are considered,
    and only deps whose ``status_map`` entry is TERMINAL — everything else
    must never trigger a fetch (cost containment: the common case pays zero
    extra cost).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler.finish_startup()
        return scheduler

    def _dependent(
        self, task_id: str = '10', dep_id: str = '20', status: str = 'pending'
    ) -> dict:
        return {
            'id': task_id,
            'status': status,
            'dependencies': [{'id': dep_id}],
            'metadata': {},
        }

    # --- (a) missing terminal dep -> fetched and recorded ------------------

    @pytest.mark.asyncio
    async def test_missing_terminal_dep_is_fetched_and_recorded(self, scheduler: Scheduler):
        dependent = self._dependent()
        ctx = TickContext(
            tasks=[dependent],
            status_map={'20': 'done'},
            tasks_by_id={'10': dependent},  # dep '20' genuinely absent
        )
        fetched = {'id': '20', 'status': 'done', 'metadata': {'delivered_checks': []}}
        scheduler.get_task = AsyncMock(return_value=fetched)

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_awaited_once_with('20')
        assert ctx.terminal_dep_records == {'20': fetched}

    # --- (b) kill switch: delivered_checks.enabled=False -> no fetch -------

    @pytest.mark.asyncio
    async def test_disabled_kill_switch_skips_fetch(self, scheduler: Scheduler):
        scheduler.config.delivered_checks.enabled = False
        dependent = self._dependent()
        ctx = TickContext(
            tasks=[dependent], status_map={'20': 'done'}, tasks_by_id={'10': dependent}
        )
        scheduler.get_task = AsyncMock(return_value={'id': '20'})

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_not_awaited()
        assert ctx.terminal_dep_records == {}

    # --- (c) dep already in tasks_by_id -> not "missing", no fetch ---------

    @pytest.mark.asyncio
    async def test_dep_already_in_tasks_by_id_not_fetched(self, scheduler: Scheduler):
        dependent = self._dependent()
        dep = {'id': '20', 'status': 'done', 'metadata': {}}
        ctx = TickContext(
            tasks=[dependent],
            status_map={'20': 'done'},
            tasks_by_id={'10': dependent, '20': dep},
        )
        scheduler.get_task = AsyncMock(return_value={'id': '20', 'should': 'not-be-used'})

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_not_awaited()
        assert ctx.terminal_dep_records == {}

    # --- (d) no PENDING task references the missing dep -> no fetch --------

    @pytest.mark.asyncio
    async def test_non_pending_task_missing_terminal_dep_not_fetched(self, scheduler: Scheduler):
        """Only PENDING tasks' deps are considered — an in-progress task
        referencing a missing terminal dep must not trigger a fetch."""
        task = self._dependent(status='in-progress')
        ctx = TickContext(
            tasks=[task], status_map={'20': 'done'}, tasks_by_id={'10': task}
        )
        scheduler.get_task = AsyncMock(return_value={'id': '20'})

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_not_awaited()
        assert ctx.terminal_dep_records == {}

    # --- (e) NON-terminal missing dep -> not fetched ------------------------

    @pytest.mark.asyncio
    async def test_non_terminal_missing_dep_not_fetched(self, scheduler: Scheduler):
        dependent = self._dependent()
        ctx = TickContext(
            tasks=[dependent],
            status_map={'20': 'pending'},  # missing from tasks_by_id but NOT terminal
            tasks_by_id={'10': dependent},
        )
        scheduler.get_task = AsyncMock(return_value={'id': '20'})

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_not_awaited()
        assert ctx.terminal_dep_records == {}

    # --- (f) get_task returns None -> dep absent from records, no crash ----

    @pytest.mark.asyncio
    async def test_get_task_returns_none_dep_absent_no_crash(self, scheduler: Scheduler):
        dependent = self._dependent()
        ctx = TickContext(
            tasks=[dependent], status_map={'20': 'done'}, tasks_by_id={'10': dependent}
        )
        scheduler.get_task = AsyncMock(return_value=None)

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_awaited_once_with('20')
        assert ctx.terminal_dep_records == {}

    # --- (g) multiple missing terminal deps -> all fetched, keyed by id ----

    @pytest.mark.asyncio
    async def test_multiple_missing_terminal_deps_all_fetched(self, scheduler: Scheduler):
        task_a = self._dependent(task_id='10', dep_id='20')
        task_b = self._dependent(task_id='11', dep_id='21')
        ctx = TickContext(
            tasks=[task_a, task_b],
            status_map={'20': 'done', '21': 'cancelled'},
            tasks_by_id={'10': task_a, '11': task_b},
        )
        records = {
            '20': {'id': '20', 'status': 'done', 'metadata': {}},
            '21': {'id': '21', 'status': 'cancelled', 'metadata': {}},
        }

        async def _fake_get_task(task_id):
            return records[task_id]

        scheduler.get_task = AsyncMock(side_effect=_fake_get_task)

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        assert scheduler.get_task.await_count == 2
        assert ctx.terminal_dep_records == records

    # --- (h) warmed record carrying delivered_checks -> re-fetched, not stale --

    @pytest.mark.asyncio
    async def test_warmed_record_carrying_delivered_checks_is_refetched_not_served_stale(
        self, scheduler: Scheduler
    ):
        """Task 2977: a cached TERMINAL dep record that carries a non-empty
        ``metadata.delivered_checks`` must be treated as a cache MISS (and
        re-fetched) every sweep, not served straight from
        ``_terminal_dep_record_cache`` — otherwise an operator's in-place
        correction of a DONE dep's check descriptor at a fixed main SHA is
        never observed, and task 2975's descriptor-digest self-heal (which
        is computed from whatever this phase serves into
        ``ctx.terminal_dep_records``) can never fire."""
        stale_record = {
            'id': '20',
            'status': 'done',
            'metadata': {
                'delivered_checks': [
                    {'kind': 'grep', 'name': 'c', 'pattern': 'OLD_BAD_PATTERN'}
                ]
            },
        }
        scheduler._terminal_dep_record_cache['20'] = stale_record
        corrected_record = {
            'id': '20',
            'status': 'done',
            'metadata': {
                'delivered_checks': [
                    {'kind': 'grep', 'name': 'c', 'pattern': 'FIXED_PATTERN'}
                ]
            },
        }
        dependent = self._dependent()
        ctx = TickContext(
            tasks=[dependent],
            status_map={'20': 'done'},
            tasks_by_id={'10': dependent},  # dep '20' genuinely absent
        )
        scheduler.get_task = AsyncMock(return_value=corrected_record)

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_awaited_once_with('20')
        assert ctx.terminal_dep_records['20'] == corrected_record
        assert scheduler._terminal_dep_record_cache['20'] == corrected_record

    # --- (i) warmed CHECKLESS record -> still served from cache, no re-fetch --

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'cached_metadata',
        [
            {'delivered_checks': []},
            {},  # no 'delivered_checks' key at all
        ],
        ids=['empty-list', 'key-absent'],
    )
    async def test_warmed_record_without_delivered_checks_still_served_from_cache_no_refetch(
        self, scheduler: Scheduler, cached_metadata: dict
    ):
        """Boundary/regression guard pinning the cache's entire performance
        rationale: a warmed record that carries NO delivered_checks (the
        overwhelming common case) must still be served for free, with no
        get_task call — this must keep passing after the task 2977 fix, or
        that fix has become an over-broad 'always re-fetch'."""
        cached_record = {'id': '20', 'status': 'done', 'metadata': cached_metadata}
        scheduler._terminal_dep_record_cache['20'] = cached_record
        dependent = self._dependent()
        ctx = TickContext(
            tasks=[dependent],
            status_map={'20': 'done'},
            tasks_by_id={'10': dependent},  # dep '20' genuinely absent
        )
        scheduler.get_task = AsyncMock(return_value={'id': '20', 'should': 'not-be-used'})

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_not_awaited()
        assert ctx.terminal_dep_records['20'] is cached_record

    # --- (j) budget contention: new dep_ids win over re-validation churn ---

    @pytest.mark.asyncio
    async def test_budget_prioritizes_new_dep_over_revalidation_dep(
        self, scheduler: Scheduler
    ):
        """Task 2977 (reviewer_comprehensive performance amendment): when
        the per-tick fetch budget can't cover both a genuinely-NEW dep_id
        (never cached) and an already-warmed checks-carrying dep_id up for
        re-validation, the new dep_id must win the budget — otherwise
        steady-state re-validation churn of already-known checks-carrying
        deps could chronically starve a brand-new dep out of the budget
        tick after tick. Dep ids are chosen so plain alphabetical sort
        ('21' < '29') would pick the WRONG (re-validation) dep first
        without the fix, proving this isn't an accident of sort order."""
        scheduler.config.delivered_checks.max_checks_per_tick = 1
        stale_record = {
            'id': '21',
            'status': 'done',
            'metadata': {
                'delivered_checks': [{'kind': 'grep', 'name': 'c', 'pattern': 'OLD'}]
            },
        }
        scheduler._terminal_dep_record_cache['21'] = stale_record
        task_a = self._dependent(task_id='10', dep_id='21')  # re-validation candidate
        task_b = self._dependent(task_id='11', dep_id='29')  # genuinely new candidate
        ctx = TickContext(
            tasks=[task_a, task_b],
            status_map={'21': 'done', '29': 'done'},
            tasks_by_id={'10': task_a, '11': task_b},  # both deps absent
        )
        new_record = {'id': '29', 'status': 'done', 'metadata': {}}
        scheduler.get_task = AsyncMock(return_value=new_record)

        result = await scheduler._phase_backfill_terminal_dep_records(ctx)

        assert result is _CONTINUE
        scheduler.get_task.assert_awaited_once_with('29')
        assert ctx.terminal_dep_records == {'29': new_record}
        assert '21' not in ctx.terminal_dep_records, (
            'the deferred re-validation dep is not served this tick — '
            'unchanged fail-open-on-deferral semantics, same as before task 2977'
        )
        # the stale cache entry is untouched (not overwritten, not evicted)
        assert scheduler._terminal_dep_record_cache['21'] == stale_record


# ---------------------------------------------------------------------------
# TestSchedulerCallbacksOnDeliveredCheckBlock (task 2583 — step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------


class TestSchedulerCallbacksOnDeliveredCheckBlock:
    """``SchedulerCallbacks`` must carry an ``on_delivered_check_block`` hook,
    defaulting to None, mirroring ``on_external_dep_block`` — the structural
    plumbing the grace-streak escalation (task 2583, epsilon) invokes."""

    def test_defaults_to_none(self):
        assert SchedulerCallbacks().on_delivered_check_block is None

    def test_accepts_constructor_kwarg(self):
        async def _fn(task_id, *, summary, detail, category):
            pass

        callbacks = SchedulerCallbacks(on_delivered_check_block=_fn)

        assert callbacks.on_delivered_check_block is _fn


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

    # --- task 2583 (epsilon) knobs: enabled / grace_cycles / check_timeout_secs -

    def test_default_enabled_is_true(self):
        """The gate is inert without metadata.delivered_checks on any dep, so
        it defaults ON — an operator must opt OUT explicitly."""
        config = OrchestratorConfig()
        assert config.delivered_checks.enabled is True

    def test_default_grace_cycles_is_3(self):
        config = OrchestratorConfig()
        assert config.delivered_checks.grace_cycles == 3

    def test_grace_cycles_rejects_zero(self):
        with pytest.raises(ValidationError):
            DeliveredChecksConfig(grace_cycles=0)

    def test_grace_cycles_rejects_negative(self):
        with pytest.raises(ValidationError):
            DeliveredChecksConfig(grace_cycles=-1)

    def test_default_check_timeout_secs_is_120(self):
        config = OrchestratorConfig()
        assert config.delivered_checks.check_timeout_secs == 120

    def test_check_timeout_secs_rejects_zero(self):
        with pytest.raises(ValidationError):
            DeliveredChecksConfig(check_timeout_secs=0)

    def test_check_timeout_secs_rejects_negative(self):
        with pytest.raises(ValidationError):
            DeliveredChecksConfig(check_timeout_secs=-1)

    def test_epsilon_knobs_are_hot_reloadable(self):
        """Same green tier as max_checks_per_tick — no restart needed to
        retune the grace window, flip the kill switch, or adjust the
        per-check wall-clock timeout."""
        assert 'delivered_checks.enabled' in RELOADABLE_FIELDS
        assert 'delivered_checks.grace_cycles' in RELOADABLE_FIELDS
        assert 'delivered_checks.check_timeout_secs' in RELOADABLE_FIELDS


# ---------------------------------------------------------------------------
# TestDeliveredCheckGateEnabledSwitch (task 2583 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestDeliveredCheckGateEnabledSwitch:
    """``Scheduler._phase_delivered_check_gate`` kill switch: with
    ``delivered_checks.enabled=False``, the phase must short-circuit to
    ``ctx.delivered_check_cache = None`` WITHOUT running the sweep at all —
    no git, no streaks, no escalation. Must be ``None``, not ``{}``:
    ``_deps_satisfied`` only disables its delivered-check arm when the cache
    is exactly ``None`` (an empty dict still activates the arm and withholds
    every checked dep)."""

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config, event_store=_RecordingEventStore())  # type: ignore[arg-type]
        scheduler.finish_startup()
        return scheduler

    @pytest.mark.asyncio
    async def test_disabled_short_circuits_to_none_cache_and_skips_sweep(
        self, scheduler: Scheduler
    ):
        scheduler.config.delivered_checks.enabled = False
        sweep = AsyncMock(side_effect=AssertionError('sweep must not run when disabled'))
        scheduler._compute_delivered_check_cache = sweep
        ctx = TickContext(tasks=[], status_map={}, tasks_by_id={})

        await scheduler._phase_delivered_check_gate(ctx)

        assert ctx.delivered_check_cache is None
        sweep.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_default_runs_sweep_and_sets_dict(self, scheduler: Scheduler):
        sweep = AsyncMock(return_value={'20': True})
        scheduler._compute_delivered_check_cache = sweep
        ctx = TickContext(tasks=[], status_map={}, tasks_by_id={})

        await scheduler._phase_delivered_check_gate(ctx)

        assert ctx.delivered_check_cache == {'20': True}
        sweep.assert_called_once()


# ---------------------------------------------------------------------------
# TestBuildDeliveredCheckEscalation (task 2583 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestBuildDeliveredCheckEscalation:
    """``_build_delivered_check_escalation`` — the PRD Resolved-6 summary/
    detail renderer the grace-streak escalation (task 2583, epsilon)
    consumes.  Pure rendering: no scheduler state, no side effects."""

    def test_grep_kind_summary_and_detail(self):
        check = {
            'name': 'cap-one',
            'kind': 'grep',
            'pattern': 'foo',
            'paths': ['src/'],
            'expect': 'present',
        }

        summary, detail = _build_delivered_check_escalation(
            task_id='10',
            dep_id='20',
            dep_status='done',
            check=check,
            main_sha='abcdef0123456789',
        )

        assert summary == (
            "DEP_CAPABILITY_NOT_DELIVERED: task 10 — dep 20 done but check "
            "'cap-one' fails on main@abcdef012345"
        ), f'unexpected summary: {summary!r}'
        assert 'grep' in detail
        assert 'foo' in detail
        assert 'src/' in detail
        assert 'present' in detail
        assert 'FAILED' in detail
        assert 'set task 10 back to pending' in detail

    def test_script_kind_detail_names_script_and_args_not_pattern(self):
        check = {
            'name': 'cap-two',
            'kind': 'script',
            'script': 'scripts/check_thing.sh',
            'args': ['--foo', 'bar'],
            'expect': 'exit_zero',
        }

        summary, detail = _build_delivered_check_escalation(
            task_id='11',
            dep_id='21',
            dep_status='done',
            check=check,
            main_sha='0123456789abcdef',
        )

        assert summary == (
            "DEP_CAPABILITY_NOT_DELIVERED: task 11 — dep 21 done but check "
            "'cap-two' fails on main@0123456789ab"
        ), f'unexpected summary: {summary!r}'
        assert 'scripts/check_thing.sh' in detail
        assert '--foo' in detail
        assert 'bar' in detail
        assert 'exit_zero' in detail
        assert 'FAILED' in detail
        assert 'pattern' not in detail, 'script-kind detail must not mention pattern'
        assert 'set task 11 back to pending' in detail


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
# TestDeliveredChecksDescriptorDigest (task 2975 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestDeliveredChecksDescriptorDigest:
    """``_delivered_checks_descriptor_digest`` — the pure canonical-JSON
    sha256 helper that lets :meth:`Scheduler._compute_delivered_check_cache`
    fold a descriptor digest into its cache key (task 2975), so correcting a
    dep's ``metadata.delivered_checks`` at a FIXED main SHA is a cache MISS
    (re-evaluate) instead of continuing to serve a stale cached verdict.
    Pure function: no scheduler state, no side effects.
    """

    _ONE_CHECK = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]

    # --- (a) determinism: identical input -> identical, stable digest ------

    def test_identical_lists_produce_same_digest_deterministically(self):
        checks = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]
        other = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]

        digest1 = _delivered_checks_descriptor_digest(checks)
        digest2 = _delivered_checks_descriptor_digest(other)
        digest3 = _delivered_checks_descriptor_digest(checks)

        assert isinstance(digest1, str)
        assert len(digest1) == 64, 'expected a sha256 hex digest (64 hex chars)'
        assert all(c in '0123456789abcdef' for c in digest1)
        assert digest1 == digest2 == digest3, 'same descriptor content must hash identically'

    # --- (b) any descriptor field change -> a DIFFERENT digest -------------

    def test_changed_grep_pattern_produces_different_digest(self):
        original = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]
        changed = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'bar', 'expect': 'present'}]

        assert _delivered_checks_descriptor_digest(original) != (
            _delivered_checks_descriptor_digest(changed)
        )

    def test_changed_paths_produces_different_digest(self):
        original = [{
            'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo',
            'paths': ['src/'], 'expect': 'present',
        }]
        changed = [{
            'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo',
            'paths': ['src/', 'lib/'], 'expect': 'present',
        }]

        assert _delivered_checks_descriptor_digest(original) != (
            _delivered_checks_descriptor_digest(changed)
        )

    def test_changed_script_args_produces_different_digest(self):
        original = [{
            'name': 'cap-two', 'kind': 'script', 'script': 'scripts/check_thing.sh',
            'args': ['--foo', 'bar'], 'expect': 'exit_zero',
        }]
        changed = [{
            'name': 'cap-two', 'kind': 'script', 'script': 'scripts/check_thing.sh',
            'args': ['--foo', 'baz'], 'expect': 'exit_zero',
        }]

        assert _delivered_checks_descriptor_digest(original) != (
            _delivered_checks_descriptor_digest(changed)
        )

    def test_changed_script_name_produces_different_digest(self):
        original = [{
            'name': 'cap-two', 'kind': 'script', 'script': 'scripts/check_thing.sh',
            'args': [], 'expect': 'exit_zero',
        }]
        changed = [{
            'name': 'cap-two', 'kind': 'script', 'script': 'scripts/check_other.sh',
            'args': [], 'expect': 'exit_zero',
        }]

        assert _delivered_checks_descriptor_digest(original) != (
            _delivered_checks_descriptor_digest(changed)
        )

    # --- (c) key reordering within a check dict -> the SAME digest ---------

    def test_key_reorder_within_check_dict_produces_same_digest(self):
        forward = [{'name': 'cap-one', 'pattern': 'foo', 'expect': 'present'}]
        reordered = [{'expect': 'present', 'pattern': 'foo', 'name': 'cap-one'}]

        assert _delivered_checks_descriptor_digest(forward) == (
            _delivered_checks_descriptor_digest(reordered)
        )

    # --- (d) empty list and None both hash the same, distinct from 1 check -

    def test_empty_list_and_none_produce_same_stable_digest_distinct_from_one_check(self):
        empty_digest1 = _delivered_checks_descriptor_digest([])
        empty_digest2 = _delivered_checks_descriptor_digest([])
        none_digest1 = _delivered_checks_descriptor_digest(None)
        none_digest2 = _delivered_checks_descriptor_digest(None)
        one_check_digest = _delivered_checks_descriptor_digest(self._ONE_CHECK)

        assert empty_digest1 == empty_digest2, 'empty-list digest must be stable'
        assert none_digest1 == none_digest2, 'None digest must be stable'
        assert empty_digest1 == none_digest1, '[] and None must normalize to the same digest'
        assert empty_digest1 != one_check_digest


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

    def _dcc_key(self, dep_id: str, sha: str, checks: list | None) -> tuple[str, str, str]:
        """Build the 3-tuple ``_delivered_check_cache`` key (task 2975),
        using the same digest helper the scheduler uses, for assertions
        that need to name a specific cache entry below."""
        return (dep_id, sha, _delivered_checks_descriptor_digest(checks))

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
        assert not any(
            k[0] == '20' and k[1] == 'sha1' for k in scheduler._delivered_check_cache
        ), 'ERRORED must not leave ANY digest-variant cache entry for this dep/sha'
        assert scheduler._streak_delivered_hold.value('10') == 0
        assert self._held_events(scheduler) == []

    # --- (e) budget: the first dep this sweep is guaranteed to resolve -----

    @pytest.mark.asyncio
    async def test_first_dep_exceeding_budget_runs_to_completion_and_warns(
        self, scheduler: Scheduler, monkeypatch, caplog
    ):
        """reviewer_comprehensive amendment: without a forward-progress
        guarantee, a dep whose OWN check count exceeds
        max_checks_per_tick would hit `over_budget` at the same relative
        position every single sweep (since `used` resets to 0 each tick)
        and could never resolve — permanently starving its dependent. The
        first dep a sweep actually evaluates (not served from cache) is
        therefore always run to completion regardless of budget, and
        exceeding the budget this way logs a bounded WARNING so an
        under-sized max_checks_per_tick is operator-visible."""
        import logging

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

        with caplog.at_level(logging.WARNING, logger='orchestrator.scheduler'):
            result = await scheduler._compute_delivered_check_cache(
                [task], status_map, tasks_by_id
            )

        assert result == {'20': True}, (
            'the first (and only) dep this sweep must fully resolve despite exceeding budget'
        )
        assert calls == ['cap-a', 'cap-b'], 'both checks must run — forward progress is guaranteed'
        key = self._dcc_key('20', 'sha1', self._TWO_CHECKS)
        assert key in scheduler._delivered_check_cache
        assert scheduler._delivered_check_cache[key] is True
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            'exceeding the per-tick budget for the guaranteed-progress dep must log a WARNING'
        )

    # --- (e2) budget still defers a later, non-first dep in the sweep ------

    @pytest.mark.asyncio
    async def test_budget_still_defers_a_later_non_first_dep(
        self, scheduler: Scheduler, monkeypatch
    ):
        """The forward-progress guarantee applies ONLY to the first dep a
        sweep actually evaluates; a second dep sharing the same tick's
        budget is still deferred exactly as before once the budget is
        spent, preserving the sweep's per-tick worst-case cost bound."""
        scheduler.config.delivered_checks.max_checks_per_tick = 1
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({
            'cap-a': DeliveredCheckResult.DELIVERED,
            'cap-one': DeliveredCheckResult.DELIVERED,
        })
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task_a = self._dependent(task_id='10', dep_id='20')
        dep_a = self._dep(
            dep_id='20',
            checks=[{'name': 'cap-a', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}],
        )
        task_b = self._dependent(task_id='11', dep_id='21')
        dep_b = self._dep(dep_id='21')  # default _ONE_CHECK, named 'cap-one'
        status_map = {'20': 'done', '21': 'done'}
        tasks_by_id = {'20': dep_a, '21': dep_b, '10': task_a, '11': task_b}

        result = await scheduler._compute_delivered_check_cache(
            [task_a, task_b], status_map, tasks_by_id
        )

        assert result == {'20': True}
        assert calls == ['cap-a'], 'the second dep must NOT be evaluated once the budget is spent'
        assert not any(
            k[0] == '21' and k[1] == 'sha1' for k in scheduler._delivered_check_cache
        ), 'the deferred dep must not leave ANY digest-variant cache entry'

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
        assert not any(
            k[0] == '20' and k[1] == 'sha1' for k in scheduler._delivered_check_cache
        ), 'every sha1-keyed digest variant must be pruned once main advances'
        assert self._dcc_key('20', 'sha2', self._ONE_CHECK) in scheduler._delivered_check_cache

    # --- (h) cached-False dep still holds + emits detail on every sweep ----

    @pytest.mark.asyncio
    async def test_cached_false_dep_emits_hold_on_every_sweep(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Mirrors (f) ``test_cache_hit_does_not_reinvoke_runner`` but with a
        FAILED check: a dep that failed on a prior tick re-runs every sweep
        (FAILED is never cached — REFILE of task 2782/2783) and must STILL
        emit a hold event with a meaningful (non-``None``) detail on every
        held tick — not just the first tick."""
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
        assert calls == ['cap-one', 'cap-one'], 'FAILED re-runs every sweep, never cached'
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
        FIRST DELIVERS and the SECOND FAILS must have its FAILED hold
        detail name the check that actually failed (cap-b) — never the
        dep's first ``delivered_checks`` entry (cap-a, which passed) — on
        BOTH the first sweep and a second sweep at the same main SHA
        (FAILED is never cached — REFILE of task 2782/2783 — so both
        checks actually re-run on both sweeps). Guards against
        reconstructing the detail from ``checks[0]``, which would misname
        a passing check as the failure.
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
        assert calls == ['cap-a', 'cap-b', 'cap-a', 'cap-b'], (
            'both checks re-run on EVERY sweep — FAILED is never cached'
        )
        held = self._held_events(scheduler)
        assert len(held) == 2
        expected_detail = {'name': 'cap-b', 'dep_id': '20', 'main_sha': 'sha1', 'kind': 'grep'}
        for _evt, data in held:
            assert data['data']['detail'] == expected_detail, (
                f"must name the check that actually FAILED (cap-b), not checks[0] "
                f"(cap-a, which passed); got {data['data']['detail']!r}"
            )

    # --- (i2) REFILE 2783: a transient FAILED must not wedge the gate at the
    #     SAME main sha — it must be able to flip to DELIVERED on a later
    #     sweep once the capability is actually observed present -----------

    @pytest.mark.asyncio
    async def test_failed_then_delivered_same_sha_flips_and_unwedges(
        self, scheduler: Scheduler, monkeypatch
    ):
        """REFILE of task 2782: a FAILED delivered-check result must NOT be
        cached sticky, because — unlike DELIVERED — FAILED is not
        monotone-safe against the OBSERVATION: a ``git grep`` can
        transiently miss (e.g. a race with a concurrent write during the
        merge that produced the SHA, or a read microseconds before the
        pattern becomes visible at that ref) even though the capability is
        actually present at that SAME main SHA. Reproduces the exact
        recurrence shape: main does NOT advance between sweeps, and the
        SAME dep/check pair resolves FAILED on sweep 1 and then DELIVERED
        on sweep 2 at the identical SHA. Before this fix, sweep 2 was
        served straight from a sticky-False cache entry (the runner never
        re-invoked) and stayed wedged at ``{'20': False}`` forever at this
        SHA — self-heal only ever happened once main ADVANCED. After the
        fix, a FAILED result re-runs every sweep (symmetric with the
        already-proven ERRORED fail-safe path below), so it can flip to
        DELIVERED at the SAME SHA instead of staying wedged.
        """
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')

        # The class `_fake_runner` maps check-name -> a FIXED outcome and
        # cannot vary the outcome across calls, so this test defines its
        # own tiny sequence-based runner: 'cap-one' resolves FAILED on the
        # first invocation and DELIVERED on the second.
        outcomes = [DeliveredCheckResult.FAILED, DeliveredCheckResult.DELIVERED]
        calls: list[str] = []

        async def _sequence_runner(check, *, project_root, ref='main'):
            calls.append(check['name'])
            return outcomes[len(calls) - 1]

        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', _sequence_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        assert first == {'20': False}
        assert not any(
            k[0] == '20' and k[1] == 'sha1' for k in scheduler._delivered_check_cache
        ), 'a FAILED result must NOT be cached sticky (any digest variant)'

        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert second == {'20': True}, (
            'the gate must flip to DELIVERED, not stay wedged FAILED at the same SHA'
        )
        assert calls == ['cap-one', 'cap-one'], (
            'the check must re-run on sweep 2 — it must NOT be served from a sticky-False cache'
        )
        key = self._dcc_key('20', 'sha1', self._ONE_CHECK)
        assert scheduler._delivered_check_cache[key] is True, (
            'the monotone-safe DELIVERED result IS now cached'
        )
        assert scheduler._streak_delivered_hold.value('10') == 0, (
            'hold streak must clear once the dep resolves DELIVERED'
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

    # --- (k) check_timeout_secs: a hung check is treated as ERRORED --------

    @pytest.mark.asyncio
    async def test_check_exceeding_timeout_is_treated_as_errored(
        self, scheduler: Scheduler, monkeypatch
    ):
        """task 2583 (epsilon): check_timeout_secs is an outer asyncio.wait_for
        backstop around each run_delivered_check call. A check that hangs
        past the timeout maps to ERRORED — the same fail-safe outcome as a
        runner exception (row 7): dep left uncached, no hold event, no
        streak bump (neither _streak_delivered_hold nor the epsilon
        _streak_delivered_fail)."""
        scheduler.config.delivered_checks.check_timeout_secs = 0.01
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')

        async def _slow_runner(check, *, project_root, ref='main'):
            await asyncio.sleep(1)
            return DeliveredCheckResult.DELIVERED

        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', _slow_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        result = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert result == {}
        assert not any(
            k[0] == '20' and k[1] == 'sha1' for k in scheduler._delivered_check_cache
        ), 'a timed-out (ERRORED) check must not leave ANY digest-variant cache entry'
        assert self._held_events(scheduler) == []
        assert scheduler._streak_delivered_hold.value('10') == 0
        assert scheduler._streak_delivered_fail.value(('10', '20')) == 0

    # --- terminal_dep_records fallback (task 2692 — step-7 RED / step-8 GREEN)
    #
    # Mirrors TestDepsSatisfiedDeliveredGate's terminal_dep_records tests: a
    # dep genuinely absent from tasks_by_id (the active-only get_tasks fetch
    # excluded it) but present in terminal_dep_records must still be swept —
    # checked_deps collects it, the runner evaluates its checks, and the
    # projection reflects the real outcome.

    @pytest.mark.asyncio
    async def test_terminal_dep_records_fallback_dep_delivered_projects_true(
        self, scheduler: Scheduler, monkeypatch
    ):
        fake_resolve, sha_calls = self._fake_sha('sha1')
        scheduler._resolve_main_sha = fake_resolve
        fake_runner, runner_calls = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'10': task}  # dep genuinely absent — active-only fetch excluded it

        result = await scheduler._compute_delivered_check_cache(
            [task], status_map, tasks_by_id, terminal_dep_records={'20': dep}
        )

        assert result == {'20': True}
        assert sha_calls['n'] == 1
        assert runner_calls == ['cap-one']

    @pytest.mark.asyncio
    async def test_terminal_dep_records_fallback_dep_failed_projects_false(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, runner_calls = self._fake_runner({'cap-one': DeliveredCheckResult.FAILED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'10': task}

        result = await scheduler._compute_delivered_check_cache(
            [task], status_map, tasks_by_id, terminal_dep_records={'20': dep}
        )

        assert result == {'20': False}
        assert runner_calls == ['cap-one']

    @pytest.mark.asyncio
    async def test_without_terminal_dep_records_dep_absent_from_both_returns_empty(
        self, scheduler: Scheduler
    ):
        """terminal_dep_records omitted (default None): a dep absent from
        tasks_by_id is simply skipped by the collection loop — byte-identical
        no-op, exactly like
        test_no_checked_deps_returns_empty_and_never_resolves_sha."""
        fake_resolve, sha_calls = self._fake_sha('sha1')
        scheduler._resolve_main_sha = fake_resolve
        task = self._dependent()
        status_map = {'20': 'done'}
        tasks_by_id = {'10': task}

        result = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert result == {}
        assert sha_calls['n'] == 0

    # --- (l) descriptor change at a FIXED sha invalidates a cached
    #     DELIVERED result and forces re-evaluation (task 2975 — the
    #     esc-2911-1/2 recurrence: a corrected descriptor must self-heal
    #     WITHOUT waiting for main to advance) --------------------------

    @pytest.mark.asyncio
    async def test_descriptor_change_same_sha_invalidates_cached_delivered_and_reevaluates(
        self, scheduler: Scheduler, monkeypatch
    ):
        """esc-2911-1/2: an operator correcting a dep's
        ``metadata.delivered_checks`` descriptor (pattern/paths/script) at a
        FIXED main SHA must be picked up on the very next sweep — NOT
        wedged behind a stale cached DELIVERED verdict until an unrelated
        commit advances main and prunes it. Before this fix, the cache was
        keyed ``(dep_id, main_sha)`` only, so a descriptor change at a fixed
        SHA was entirely invisible to it.

        Sweeps 3-4 additionally fold in the *unchanged*-descriptor
        efficiency guarantee (mirrors (f)
        ``test_cache_hit_does_not_reinvoke_runner``): a cache hit is only
        possible once a descriptor has actually resolved DELIVERED, since
        FAILED is (by design — REFILE of task 2782/2783, see (i2)
        ``test_failed_then_delivered_same_sha_flips_and_unwedges``) never
        cached. So sweep 3 mirrors that same transient-FAILED-then-
        DELIVERED flip for the CHANGED descriptor (still unchanged from
        sweep 2) before sweep 4 observes a genuine cache hit — the runner
        is not re-invoked a further time once the changed descriptor
        itself has stabilized DELIVERED and been cached.
        """
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep(checks=[
            {'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'},
        ])
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        assert first == {'20': True}
        assert calls == ['cap-one']

        # Operator edits the descriptor at the SAME sha (pattern 'foo' ->
        # 'bar', same check name) — a fresh runner now resolves it FAILED.
        dep['metadata'] = {'delivered_checks': [
            {'name': 'cap-one', 'kind': 'grep', 'pattern': 'bar', 'expect': 'present'},
        ]}
        fake_runner2, calls2 = self._fake_runner({'cap-one': DeliveredCheckResult.FAILED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner2)

        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert second == {'20': False}, (
            'the changed descriptor must be RE-EVALUATED, not served from the '
            'stale cached True left over from the OLD descriptor'
        )
        assert calls2 == ['cap-one'], (
            'the changed descriptor must MISS the cache and invoke the runner exactly once'
        )

        # SAME (still-changed) descriptor observed again: like (i2), a
        # FAILED result is never cached, so it can flip to DELIVERED on a
        # later sweep at the identical descriptor/SHA instead of staying
        # wedged.
        fake_runner3, calls3 = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner3)

        third = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert third == {'20': True}
        assert calls3 == ['cap-one'], 'FAILED is never cached, so this must be a fresh invocation'

        # A FOURTH sweep at the SAME (now-DELIVERED, still-unchanged)
        # descriptor is a genuine cache hit — the unchanged-descriptor
        # efficiency guarantee also holds for a descriptor that changed
        # mid-flight, not just one that was stable from the first sweep.
        fourth = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert fourth == {'20': True}
        assert calls3 == ['cap-one'], (
            'the runner must NOT be re-invoked once the changed descriptor is cached'
        )

    # --- (m) FAILED descriptor corrected at the SAME sha -> DELIVERED next
    #     sweep (task's literal acceptance criterion). Already-green
    #     regression lock, NOT the RED driver — FAILED was never cached to
    #     begin with (REFILE of task 2782/2783), so this already worked
    #     before this task's fix; see (l) above for the genuinely-RED
    #     scenario (a cached DELIVERED must be INVALIDATED when the
    #     descriptor changes). ------------------------------------------

    @pytest.mark.asyncio
    async def test_failed_descriptor_corrected_same_sha_reevaluates_to_delivered(
        self, scheduler: Scheduler, monkeypatch
    ):
        """The task's literal acceptance criterion: a broken delivered-check
        pattern that FAILS, corrected in ``metadata.delivered_checks`` at
        the SAME main sha, must resolve DELIVERED on the very next sweep.
        """
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({'cap-one': DeliveredCheckResult.FAILED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep(checks=[
            {'name': 'cap-one', 'kind': 'grep', 'pattern': 'fooo', 'expect': 'present'},
        ])
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        assert first == {'20': False}
        assert calls == ['cap-one']

        # Operator corrects the broken pattern at the SAME sha.
        dep['metadata'] = {'delivered_checks': [
            {'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'},
        ]}
        fake_runner2, calls2 = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner2)

        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert second == {'20': True}
        assert calls2 == ['cap-one']

    # --- (n) descriptor-variant pruning bounds cache growth to ONE entry
    #     per (dep, sha) — without this, repeated descriptor edits at a
    #     fixed SHA would accumulate one stale variant per edit until main
    #     advances ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_descriptor_change_prunes_stale_digest_variant_same_sha(
        self, scheduler: Scheduler, monkeypatch
    ):
        """A descriptor change at a fixed main SHA must not just MISS the
        cache (see (l) above) — the STALE ``(dep, sha, old-digest)`` entry
        left behind by the prior descriptor must be actively PRUNED, so
        cache growth stays bounded to one variant per ``(dep, sha)`` rather
        than accumulating one entry per historical descriptor edit.
        """
        scheduler._resolve_main_sha, _sha_calls = self._fake_sha('sha1')
        fake_runner, calls = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        checks_d1 = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]
        dep = self._dep(checks=checks_d1)
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        assert first == {'20': True}
        key_d1 = self._dcc_key('20', 'sha1', checks_d1)
        assert key_d1 in scheduler._delivered_check_cache

        # Operator edits the descriptor at the SAME sha; the new descriptor
        # also resolves DELIVERED (so its own variant gets cached, letting
        # us assert exactly one variant survives).
        checks_d2 = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'baz', 'expect': 'present'}]
        dep['metadata'] = {'delivered_checks': checks_d2}
        fake_runner2, calls2 = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner2)

        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert second == {'20': True}
        assert calls2 == ['cap-one']
        key_d2 = self._dcc_key('20', 'sha1', checks_d2)
        assert key_d1 not in scheduler._delivered_check_cache, (
            'the stale (dep, sha, old-digest) variant must be pruned once the '
            'descriptor changes, not left to linger alongside the new variant'
        )
        same_dep_sha_keys = [
            k for k in scheduler._delivered_check_cache if k[0] == '20' and k[1] == 'sha1'
        ]
        assert same_dep_sha_keys == [key_d2], (
            f'cache growth must stay bounded to ONE variant per (dep, sha); '
            f'got {same_dep_sha_keys!r}'
        )

    # --- (o) regression: SHA advance still prunes EVERY digest variant, not
    #     just the one matching the current descriptor — the digest-variant
    #     prune above is ADDITIVE to, not a replacement for, the existing
    #     SHA-based prune ------------------------------------------------

    @pytest.mark.asyncio
    async def test_sha_advance_prunes_across_digest_variants(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Regression lock for the retained SHA-prune (~3098): once a
        ``(dep, sha, digest)`` entry is cached, advancing main to a new SHA
        must still prune EVERY sha1-keyed variant (regardless of its
        digest), not just whichever digest happens to match the current
        descriptor. Guards against a future digest-variant-scoped prune
        accidentally narrowing the existing whole-SHA prune.
        """
        sha_box = {'value': 'sha1'}

        async def _resolve():
            return sha_box['value']

        scheduler._resolve_main_sha = _resolve
        fake_runner, _calls = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner)
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        first = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        assert first == {'20': True}
        assert self._dcc_key('20', 'sha1', self._ONE_CHECK) in scheduler._delivered_check_cache

        sha_box['value'] = 'sha2'
        fake_runner2, calls2 = self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED})
        monkeypatch.setattr('orchestrator.scheduler.run_delivered_check', fake_runner2)

        second = await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert second == {'20': True}
        assert calls2 == ['cap-one']
        assert not any(
            k[1] == 'sha1' for k in scheduler._delivered_check_cache
        ), 'every sha1-keyed variant must be pruned once main advances, regardless of digest'
        assert self._dcc_key('20', 'sha2', self._ONE_CHECK) in scheduler._delivered_check_cache


# ---------------------------------------------------------------------------
# TestDeliveredCheckGraceEscalation (task 2583 — step-9 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


class TestDeliveredCheckGraceEscalation:
    """Grace-streak escalation (task 2583, epsilon): after
    ``delivered_checks.grace_cycles`` consecutive ran-and-FAILED sweep
    ticks for a given (dependent, dep) pair, ``_compute_delivered_check_cache``
    invokes ``on_delivered_check_block`` INSTEAD OF (not in addition to)
    the per-tick ``_note_delivered_hold`` visibility event on that tick, and
    clears the fail-streak so a persistent failure can re-fire after
    ``grace_cycles`` more ticks. A DELIVERED tick clears the fail-streak
    (streak-reset-on-pass); an ERRORED tick leaves it untouched (delta's
    fail-safe contract, PRD row 7 — never a bump on error).
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        config.delivered_checks.grace_cycles = 3
        scheduler = Scheduler(
            config,
            event_store=_RecordingEventStore(),  # type: ignore[arg-type]
            callbacks=SchedulerCallbacks(on_delivered_check_block=AsyncMock()),
        )
        scheduler.finish_startup()
        return scheduler

    _ONE_CHECK = [{'name': 'cap-one', 'kind': 'grep', 'pattern': 'foo', 'expect': 'present'}]

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
        async def _resolve():
            return sha

        return _resolve

    def _fake_runner(self, results: dict):
        async def _fake(check, *, project_root, ref='main'):
            outcome = results[check['name']]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return _fake

    def _held_events(self, scheduler: Scheduler) -> list[tuple[str, dict]]:
        _event_store = scheduler.event_store
        assert _event_store is not None
        return [
            (evt, data)
            for evt, data in _event_store.events  # type: ignore[attr-defined]
            if evt == str(EventType.delivered_check_gate_held)
        ]

    @pytest.mark.asyncio
    async def test_escalates_on_grace_cycles_consecutive_failures(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner({'cap-one': DeliveredCheckResult.FAILED}),
        )
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}
        callback = scheduler._callbacks.on_delivered_check_block
        assert callback is not None

        await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        callback.assert_not_called()

        await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        callback.assert_called_once()
        call_args, call_kwargs = callback.call_args
        assert call_args[0] == '10'
        assert call_kwargs['category'] == 'dependency_capability'
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in call_kwargs['summary']
        assert 'cap-one' in call_kwargs['summary']
        assert '20' in call_kwargs['summary']
        assert 'sha1' in call_kwargs['summary']
        assert scheduler._streak_delivered_fail.value(('10', '20')) == 0
        # No held event on the escalation (3rd) tick — only ticks 1 and 2.
        assert len(self._held_events(scheduler)) == 2

    @pytest.mark.asyncio
    async def test_escalation_survives_sha_advances_and_transient_absence(
        self, scheduler: Scheduler, monkeypatch
    ):
        """End-to-end regression (task 2743, reproduces the reported incident).

        Drives the REAL per-tick phase order — ``_phase_stale_sweep`` then
        ``_phase_delivered_check_gate`` — across ticks where ``main`` advances
        EVERY tick and the still-pending gate-held dependent transiently drops
        out of the active-only fetch on one tick (the merge/reconcile churn
        that coincides with a main advance).

        Pre-fix, tick3's ``_phase_stale_sweep`` GC'd the grace streak on that
        transient absence, so it never reached ``grace_cycles`` and the
        born-at-L2 ``dependency_capability`` escalation never fired (RED).
        Post-fix the streak survives the one-tick disappearance — and the
        intervening SHA advances, which never keyed it — so the escalation
        fires exactly once at ``grace_cycles`` (GREEN).
        """
        sha_box = {'value': 'sha1'}

        async def _resolve():
            return sha_box['value']

        scheduler._resolve_main_sha = _resolve
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner({'cap-one': DeliveredCheckResult.FAILED}),
        )
        task = self._dependent()
        dep = self._dep()
        callback = scheduler._callbacks.on_delivered_check_block
        assert callback is not None

        async def _tick(*, present: bool, sha: str) -> None:
            sha_box['value'] = sha
            if present:
                # The dependent is pending and gate-held — the real tick's
                # _update_age_anchors stamps its pending-age anchor each tick
                # it appears in the fetch; model that so tick3's absence
                # exercises the stale_sweep absence-GC path under test.
                scheduler._pending_anchor['10'] = 10
                ctx = TickContext(
                    tasks=[task],
                    status_map={'20': 'done'},
                    tasks_by_id={'20': dep, '10': task},
                )
            else:
                # '10' transiently drops out of the active-only fetch: absent
                # from ctx.tasks AND ctx.tasks_by_id, still tracked via the
                # anchor stamped on the prior tick, and non-terminal.
                ctx = TickContext(
                    tasks=[],
                    status_map={'20': 'done'},
                    tasks_by_id={'20': dep},
                )
            await scheduler._phase_stale_sweep(ctx)
            await scheduler._phase_delivered_check_gate(ctx)

        await _tick(present=True, sha='sha1')   # FAIL -> streak 1
        await _tick(present=True, sha='sha2')   # FAIL -> streak 2
        callback.assert_not_called()
        assert scheduler._streak_delivered_fail.value(('10', '20')) == 2

        # Transient absence coinciding with a main advance: stale_sweep runs,
        # the gate can't bump an absent task. The grace streak must FREEZE at
        # 2 (survive), not reset to 0.
        await _tick(present=False, sha='sha3')
        assert scheduler._streak_delivered_fail.value(('10', '20')) == 2
        callback.assert_not_called()

        await _tick(present=True, sha='sha4')   # FAIL -> streak 3 -> ESCALATE

        callback.assert_called_once()
        call_args, call_kwargs = callback.call_args
        assert call_args[0] == '10'
        assert call_kwargs['category'] == 'dependency_capability'
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in call_kwargs['summary']

    @pytest.mark.asyncio
    async def test_streak_resets_on_pass(self, scheduler: Scheduler, monkeypatch):
        sha_box = {'value': 'sha1'}

        async def _resolve():
            return sha_box['value']

        scheduler._resolve_main_sha = _resolve
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner({'cap-one': DeliveredCheckResult.FAILED}),
        )
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        assert scheduler._streak_delivered_fail.value(('10', '20')) == 1

        sha_box['value'] = 'sha2'
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner({'cap-one': DeliveredCheckResult.DELIVERED}),
        )

        await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert scheduler._streak_delivered_fail.value(('10', '20')) == 0
        assert scheduler._callbacks.on_delivered_check_block is not None
        scheduler._callbacks.on_delivered_check_block.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_bump_on_error(self, scheduler: Scheduler, monkeypatch):
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner({'cap-one': DeliveredCheckResult.ERRORED}),
        )
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}

        await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert scheduler._streak_delivered_fail.value(('10', '20')) == 0
        assert scheduler._callbacks.on_delivered_check_block is not None
        scheduler._callbacks.on_delivered_check_block.assert_not_called()

    @pytest.mark.asyncio
    async def test_re_escalates_after_grace_cycles_more_failures(
        self, scheduler: Scheduler, monkeypatch
    ):
        """Clear-then-fire contract: once the fail-streak is cleared by a
        first escalation, ``grace_cycles`` MORE consecutive FAILED ticks
        (e.g. an operator re-pends the task but the underlying capability
        is still broken) re-fires a SECOND escalation rather than staying
        silent forever."""
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner({'cap-one': DeliveredCheckResult.FAILED}),
        )
        task = self._dependent()
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task}
        callback = scheduler._callbacks.on_delivered_check_block
        assert callback is not None

        # First 3 consecutive FAILED ticks -> streak hits grace_cycles (3)
        # -> first escalation, streak cleared.
        for _ in range(3):
            await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)
        callback.assert_called_once()
        assert scheduler._streak_delivered_fail.value(('10', '20')) == 0

        # 3 MORE consecutive FAILED ticks (still the same SHA, still
        # failing) -> the cleared streak climbs back to grace_cycles and
        # re-fires a second, independent escalation.
        for _ in range(3):
            await scheduler._compute_delivered_check_cache([task], status_map, tasks_by_id)

        assert callback.call_count == 2
        assert scheduler._streak_delivered_fail.value(('10', '20')) == 0
        second_call_args, second_call_kwargs = callback.call_args_list[1]
        assert second_call_args[0] == '10'
        assert 'DEP_CAPABILITY_NOT_DELIVERED' in second_call_kwargs['summary']
        assert 'cap-one' in second_call_kwargs['summary']

    @pytest.mark.asyncio
    async def test_per_dependent_dep_keying_escalates_independently(
        self, scheduler: Scheduler, monkeypatch
    ):
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner({'cap-one': DeliveredCheckResult.FAILED}),
        )
        task_a = self._dependent(task_id='10', dep_id='20')
        task_b = self._dependent(task_id='11', dep_id='20')
        dep = self._dep()
        status_map = {'20': 'done'}
        tasks_by_id = {'20': dep, '10': task_a, '11': task_b}
        callback = scheduler._callbacks.on_delivered_check_block
        assert callback is not None

        for _ in range(3):
            await scheduler._compute_delivered_check_cache(
                [task_a, task_b], status_map, tasks_by_id
            )

        assert callback.call_count == 2
        called_task_ids = {c.args[0] for c in callback.call_args_list}
        assert called_task_ids == {'10', '11'}


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


# ---------------------------------------------------------------------------
# TestAcquireNextDeliveredGateRealFilteredFetch (task 2692 — step-11 RED / step-12 GREEN)
# ---------------------------------------------------------------------------


class TestAcquireNextDeliveredGateRealFilteredFetch:
    """The regression the delta/epsilon follow-up (task 2692) exists to
    close: ``TestAcquireNextDeliveredGate`` above drives the gate via
    ``scheduler.get_tasks = AsyncMock(return_value=[dep, dependent])`` —
    this SMUGGLES the done dep past the real ``ACTIVE_TASK_STATUSES``
    filter, so it lands in ``tasks_by_id`` regardless of whether the
    backfill plumbing (``ctx.terminal_dep_records``) actually works.

    These tests instead drive ``acquire_next()`` through a STATUS-HONORING
    ``get_tasks`` fake that filters its return by the ``statuses=`` kwarg —
    exactly like the real fused-memory tool. The active fetch then yields
    ONLY the pending dependent; the done dep is genuinely excluded. Its
    status arrives via a mocked ``get_statuses`` (the existing
    ``backfill_dep_status`` phase) and its record via a mocked ``get_task``
    (the new ``backfill_terminal_dep_records`` phase) — exactly the two
    real fetches production takes. On baseline (before step-12 wires the
    new phase into ``_TICK_PHASE_ORDER`` and threads
    ``terminal_dep_records`` through both consumers) the FAILED-check test
    below FAILS: the dependent dispatches anyway because the gate is a
    silent no-op through this path.
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

    def _status_honoring_get_tasks(self, *records: dict):
        """A ``get_tasks`` fake that honors the ``statuses=`` filter kwarg —
        unlike a plain ``AsyncMock(return_value=...)``, which returns every
        record regardless of the filter and so smuggles terminal deps past
        the real active-only fetch.

        Accepts (and ignores) ``distinguish_failure`` for signature
        compatibility with the real ``Scheduler.get_tasks`` overloads —
        ``acquire_next`` now calls with ``distinguish_failure=True``, and
        this fake never simulates a read failure, so the flag never changes
        its (always-successful) behaviour."""

        async def _gt(*, statuses=None, distinguish_failure=False):
            xs = list(records)
            return xs if statuses is None else [t for t in xs if t['status'] in statuses]

        return _gt

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

    # --- (1) phase-order invariant ------------------------------------------

    def test_backfill_terminal_dep_records_placed_between_dep_status_and_gate(self):
        order = Scheduler._TICK_PHASE_ORDER

        assert 'backfill_terminal_dep_records' in order
        idx_dep_status = order.index('backfill_dep_status')
        idx_terminal = order.index('backfill_terminal_dep_records')
        idx_gate = order.index('delivered_check_gate')
        idx_build = order.index('build_candidates')
        idx_pins = order.index('select_pins')

        assert idx_dep_status < idx_terminal < idx_gate
        assert idx_terminal < idx_build
        assert idx_terminal < idx_pins

    # --- (2) FAILED withhold: the exact gap δ/ε left ------------------------

    @pytest.mark.asyncio
    async def test_failed_check_on_genuinely_excluded_dep_withholds(
        self, scheduler: Scheduler, monkeypatch
    ):
        dep = self._dep()
        dependent = self._dependent()
        scheduler.get_tasks = self._status_honoring_get_tasks(dep, dependent)
        scheduler.get_statuses = AsyncMock(return_value=({'20': 'done'}, None))
        scheduler.get_task = AsyncMock(return_value=dep)
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.FAILED),
        )

        result = await scheduler.acquire_next()

        assert result is None, (
            f'A FAILED delivered check on a dep genuinely excluded from the '
            f'active-only fetch must still withhold its dependent — the '
            f'gate must not be a silent no-op through the real dispatch '
            f'path; got {result!r}'
        )
        assert scheduler._streak_delivered_hold.value('10') == 1
        held = self._held_events(scheduler)
        assert len(held) == 1
        _evt, data = held[0]
        assert data['task_id'] == '10'
        assert data['data']['detail'] == {
            'name': 'cap-one', 'dep_id': '20', 'main_sha': 'sha1', 'kind': 'grep',
        }

    # --- (3) DELIVERED transparency -----------------------------------------

    @pytest.mark.asyncio
    async def test_delivered_check_on_genuinely_excluded_dep_dispatches(
        self, scheduler: Scheduler, monkeypatch
    ):
        dep = self._dep()
        dependent = self._dependent()
        scheduler.get_tasks = self._status_honoring_get_tasks(dep, dependent)
        scheduler.get_statuses = AsyncMock(return_value=({'20': 'done'}, None))
        scheduler.get_task = AsyncMock(return_value=dep)
        scheduler._resolve_main_sha = self._fake_sha('sha1')
        monkeypatch.setattr(
            'orchestrator.scheduler.run_delivered_check',
            self._fake_runner(DeliveredCheckResult.DELIVERED),
        )

        result = await scheduler.acquire_next()

        assert result is not None and result.task_id == '10', (
            f'A DELIVERED check must dispatch — the gate must stay '
            f'transparent when the check actually passes; got {result!r}'
        )
        assert self._held_events(scheduler) == []


# ---------------------------------------------------------------------------
# TestVerifyDeliveredChecksOnMain (task 2794 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestVerifyDeliveredChecksOnMain:
    """``verify_delivered_checks_on_main``'s pure aggregation/precedence.

    Drives the shared reconcile-side capability guard (task 2794) directly
    with an injected fake runner — no Harness, no real git repo. Pins the
    SAME precedence as ``Scheduler._compute_delivered_check_cache``
    (scheduler.py 3150-3211): all-DELIVERED > any-FAILED > ERRORED, with a
    hung check (``asyncio.wait_for`` past ``check_timeout_secs``) mapped to
    ERRORED (fail-safe). The function must NEVER raise.
    """

    def _grep(self, name: str, pattern: str) -> dict:
        """A minimal valid grep descriptor (expect=present is required)."""
        return {'name': name, 'kind': 'grep', 'pattern': pattern, 'expect': 'present'}

    def _rc_by_pattern_runner(self, rc_by_pattern: dict[str, int]):
        """Fake runner returning a per-pattern rc.

        ``run_delivered_check``'s grep branch builds argv
        ``['git','-C',<proj>,'grep','-E','-e',<pattern>,<ref>, ...]`` — the
        pattern sits at index 6, so keying the returned rc off it lets each
        check in a multi-check list resolve to a distinct outcome.
        """

        async def _runner(argv, **kwargs):
            pattern = argv[6]
            return (rc_by_pattern[pattern], '', '')

        return _runner

    @pytest.mark.asyncio
    async def test_all_delivered_carries_passed_main_sha(self):
        """(a) two grep checks both rc==0 (expect=present) -> 'all_delivered',
        and the verdict echoes the passed main_sha."""
        checks = [self._grep('a', 'PatA'), self._grep('b', 'PatB')]
        runner = self._rc_by_pattern_runner({'PatA': 0, 'PatB': 0})

        verdict = await verify_delivered_checks_on_main(
            checks,
            project_root='/proj',
            main_sha='deadbeef',
            check_timeout_secs=5.0,
            runner=runner,
        )

        assert isinstance(verdict, DeliveredChecksVerdict)
        assert verdict.outcome == 'all_delivered'
        assert verdict.main_sha == 'deadbeef'
        assert verdict.failed_check is None

    @pytest.mark.asyncio
    async def test_single_failed_carries_that_descriptor(self):
        """(b) one rc==1 -> 'failed', and failed_check is that descriptor."""
        failing = self._grep('gone', 'PatGone')
        checks = [self._grep('ok', 'PatOk'), failing]
        runner = self._rc_by_pattern_runner({'PatOk': 0, 'PatGone': 1})

        verdict = await verify_delivered_checks_on_main(
            checks,
            project_root='/proj',
            main_sha='sha1',
            check_timeout_secs=5.0,
            runner=runner,
        )

        assert verdict.outcome == 'failed'
        assert verdict.failed_check is failing

    @pytest.mark.asyncio
    async def test_failed_beats_errored(self):
        """(c) precedence: one rc==1 (FAILED) + one rc==2 (ERRORED) ->
        'failed'. A definitive absence must drive re-dispatch, never be
        masked into a fail-safe no-op by an unrelated errored check —
        even when the ERRORED check is encountered first."""
        failing = self._grep('gone', 'PatGone')
        checks = [self._grep('boom', 'PatBoom'), failing]
        runner = self._rc_by_pattern_runner({'PatBoom': 2, 'PatGone': 1})

        verdict = await verify_delivered_checks_on_main(
            checks,
            project_root='/proj',
            main_sha='sha1',
            check_timeout_secs=5.0,
            runner=runner,
        )

        assert verdict.outcome == 'failed'
        assert verdict.failed_check is failing

    @pytest.mark.asyncio
    async def test_errored_without_failed_is_fail_safe(self):
        """(d) one rc>=2 (git error) with the rest DELIVERED and none FAILED
        -> 'errored' (fail-safe wait; no failed_check)."""
        checks = [self._grep('ok', 'PatOk'), self._grep('boom', 'PatBoom')]
        runner = self._rc_by_pattern_runner({'PatOk': 0, 'PatBoom': 128})

        verdict = await verify_delivered_checks_on_main(
            checks,
            project_root='/proj',
            main_sha='sha1',
            check_timeout_secs=5.0,
            runner=runner,
        )

        assert verdict.outcome == 'errored'
        assert verdict.failed_check is None

    @pytest.mark.asyncio
    async def test_hung_check_times_out_to_errored(self):
        """(e) a runner that awaits longer than check_timeout_secs ->
        TimeoutError mapped to ERRORED -> 'errored'. Reaching the assert
        (no exception propagated) is itself the never-raises guarantee."""

        async def _slow_runner(argv, **kwargs):
            await asyncio.sleep(1.0)
            return (0, '', '')

        checks = [self._grep('slow', 'PatSlow')]

        verdict = await verify_delivered_checks_on_main(
            checks,
            project_root='/proj',
            main_sha='sha1',
            check_timeout_secs=0.01,
            runner=_slow_runner,
        )

        assert verdict.outcome == 'errored'


# ---------------------------------------------------------------------------
# TestGateMarkDoneOnDeliveredChecks (task 3057 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------

#: The caller-supplied logger every row below passes as ``log=``. The helper
#: MUST log on THIS logger (not ``orchestrator.delivered_checks``) so each
#: adopting seam keeps its own module logger and its existing
#: caplog-addressable name — that is the contract task 2794's stranded-arm
#: caplog assertions (logger='orchestrator.harness') depend on surviving the
#: step-20 refactor onto this helper.
_SEAM_LOGGER = logging.getLogger('test.orchestrator.gate_seam')

_VERIFY_TARGET = 'orchestrator.delivered_checks.verify_delivered_checks_on_main'


class TestGateMarkDoneOnDeliveredChecks:
    """``gate_mark_done_on_delivered_checks`` — the ONE shared mark-done
    decision routed through by all eleven attribution-shaped stamp seams
    (task 3057).

    ``None`` <=> "mark-done may proceed"; a :class:`DeliveredChecksBlock`
    <=> "do NOT stamp done here". Pins task 2794's six-row acceptance matrix
    AT SOURCE, so the ten adopting seams only have to pin their own
    delegation + recovery rather than re-deriving the decision.

    ``verify_delivered_checks_on_main`` is patched throughout: this class
    pins the DECISION layer. The check-runner semantics it delegates to are
    pinned by ``TestVerifyDeliveredChecksOnMain`` above — the helper must
    reuse that function VERBATIM and never fork a second check runner.
    """

    _SHA = 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2'
    _SITE = 'unit-test-seam'

    # -- fixtures ----------------------------------------------------------

    def _grep_check(self, name: str = 'cap-x', pattern: str = 'SomePattern') -> dict:
        return {'name': name, 'kind': 'grep', 'pattern': pattern, 'expect': 'present'}

    def _script_check(
        self, name: str = 'cap-s', script: str = 'scripts/verify_cap.sh'
    ) -> dict:
        return {'name': name, 'kind': 'script', 'script': script, 'timeout_secs': 5.0}

    def _meta(self, checks: list[dict] | None = None) -> dict:
        return {'delivered_checks': checks if checks is not None else [self._grep_check()]}

    def _git_ops(self, sha: str | None = None, raises: bool = False):
        """Stub git_ops exposing ONLY ``get_main_sha`` (the sole attribute the
        helper is permitted to touch — a wider stub would let an accidental
        second git call pass unnoticed)."""
        stub = AsyncMock()
        if raises:
            stub.get_main_sha = AsyncMock(side_effect=RuntimeError('git exploded'))
        else:
            stub.get_main_sha = AsyncMock(
                return_value=self._SHA if sha is None else sha
            )
        return stub

    def _warnings(self, caplog) -> list[logging.LogRecord]:
        return [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == _SEAM_LOGGER.name
        ]

    async def _call(self, metadata, *, verify_mock, **kwargs):
        """Invoke the helper with ``verify_delivered_checks_on_main`` patched."""
        kwargs.setdefault('project_root', '/proj')
        kwargs.setdefault('check_timeout_secs', 5.0)
        kwargs.setdefault('site', self._SITE)
        kwargs.setdefault('log', _SEAM_LOGGER)
        with patch(_VERIFY_TARGET, verify_mock):
            return await gate_mark_done_on_delivered_checks('t-1', metadata, **kwargs)

    # -- Row 1: hollow-done regression / FAILED ----------------------------

    @pytest.mark.asyncio
    async def test_failed_verdict_blocks_and_warns(self, caplog):
        """Row 1 — the hollow-done regression this task exists to close.

        A FAILED verdict must yield a ``reason='failed'`` block carrying the
        SHA and the offending descriptor, and emit exactly ONE WARNING on the
        CALLER's logger naming task id / check name / pattern / main SHA /
        site — everything an operator needs to see WHY a done was withheld.
        """
        failed_check = self._grep_check()
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='failed', main_sha=self._SHA, failed_check=failed_check,
        ))
        git_ops = self._git_ops()

        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            block = await self._call(
                self._meta(), verify_mock=verify, git_ops=git_ops,
            )

        assert isinstance(block, DeliveredChecksBlock)
        assert block.reason == 'failed'
        assert block.main_sha == self._SHA
        assert block.failed_check == failed_check

        warnings = self._warnings(caplog)
        assert len(warnings) == 1, f'expected exactly one WARNING, got {warnings}'
        text = warnings[0].getMessage()
        assert 't-1' in text
        assert 'cap-x' in text
        assert 'SomePattern' in text
        assert self._SHA in text
        assert self._SITE in text

    # -- Row 2: FAILED, script kind ----------------------------------------

    @pytest.mark.asyncio
    async def test_failed_script_check_warning_names_the_script(self, caplog):
        """Row 2 — pins the is_grep branch of the WARNING: a script-kind
        descriptor has no ``pattern``, so the message must name the SCRIPT
        (naming a ``None`` pattern would be useless to an operator)."""
        failed_check = self._script_check()
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='failed', main_sha=self._SHA, failed_check=failed_check,
        ))

        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            block = await self._call(
                self._meta([failed_check]),
                verify_mock=verify,
                git_ops=self._git_ops(),
            )

        assert block is not None and block.reason == 'failed'
        text = self._warnings(caplog)[0].getMessage()
        assert 'scripts/verify_cap.sh' in text
        assert 'cap-s' in text

    # -- Row 3: all_delivered ----------------------------------------------

    @pytest.mark.asyncio
    async def test_all_delivered_returns_none_and_is_silent(self, caplog):
        """Row 3 — the capability IS on main: mark-done proceeds (``None``),
        and nothing is logged at WARNING (a verified done is not an event
        worth warning about)."""
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='all_delivered', main_sha=self._SHA,
        ))

        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            block = await self._call(
                self._meta(), verify_mock=verify, git_ops=self._git_ops(),
            )

        assert block is None
        assert self._warnings(caplog) == []

    # -- Row 4: no delivered_checks (inertness) ----------------------------

    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param({}, id='empty-metadata'),
            pytest.param(None, id='none-metadata'),
            pytest.param({'delivered_checks': []}, id='empty-checks-list'),
        ],
    )
    @pytest.mark.asyncio
    async def test_no_delivered_checks_is_inert_with_zero_io(self, metadata):
        """Row 4 — check-less tasks (the overwhelmingly common case) keep
        their exact pre-guard behavior with ZERO I/O: no git call, no check
        run. This inertness lives HERE, in one place, which is why every
        adopting seam is allowed to delegate unconditionally."""
        verify = AsyncMock()
        git_ops = self._git_ops()

        block = await self._call(metadata, verify_mock=verify, git_ops=git_ops)

        assert block is None
        verify.assert_not_awaited()
        git_ops.get_main_sha.assert_not_awaited()

    # -- Row 5: check-runner ERROR/timeout ---------------------------------

    @pytest.mark.asyncio
    async def test_errored_verdict_blocks_fail_safe(self, caplog):
        """Row 5 — the checks could not be evaluated: make no claim either
        way. Still a BLOCK (never stamp on unknown capability state), but
        ``reason='errored'`` so callers can tell it apart from a definitive
        absence."""
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='errored', main_sha=self._SHA,
        ))

        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            block = await self._call(
                self._meta(), verify_mock=verify, git_ops=self._git_ops(),
            )

        assert block is not None
        assert block.reason == 'errored'
        assert block.main_sha == self._SHA
        assert block.failed_check is None

        text = self._warnings(caplog)[0].getMessage()
        assert 't-1' in text
        assert self._SHA in text
        assert self._SITE in text

    # -- Row 6: kill switch ------------------------------------------------

    @pytest.mark.asyncio
    async def test_enabled_false_is_the_single_kill_switch(self):
        """Row 6 — ``enabled=False`` disarms the guard even with a FAILED
        verdict staged, and costs ZERO I/O. Every seam FORWARDS the config
        flag here rather than short-circuiting locally, so one hot-reload of
        ``delivered_checks.enabled`` disarms all eleven guards at once."""
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='failed', main_sha=self._SHA, failed_check=self._grep_check(),
        ))
        git_ops = self._git_ops()

        block = await self._call(
            self._meta(), verify_mock=verify, git_ops=git_ops, enabled=False,
        )

        assert block is None
        verify.assert_not_awaited()
        git_ops.get_main_sha.assert_not_awaited()

    # -- main-SHA resolution fail-safe arms (kept DISTINCT) ----------------

    @pytest.mark.asyncio
    async def test_get_main_sha_raising_is_fail_safe(self, caplog):
        """``get_main_sha()`` raising (git error) -> ``main_sha_unresolved``
        with a ``None`` SHA, and the checks are NEVER run (there is no ref to
        run them against)."""
        verify = AsyncMock()

        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            block = await self._call(
                self._meta(), verify_mock=verify, git_ops=self._git_ops(raises=True),
            )

        assert block is not None
        assert block.reason == 'main_sha_unresolved'
        assert block.main_sha is None
        verify.assert_not_awaited()
        assert 't-1' in self._warnings(caplog)[0].getMessage()

    @pytest.mark.asyncio
    async def test_get_main_sha_empty_is_fail_safe_with_distinct_warning(self, caplog):
        """``get_main_sha()`` returning ``''`` -> the same block class, but a
        DISTINCT warning text. Kept separate from the raising arm so a
        regression that drops either one is caught: an empty SHA is a silent
        git-state anomaly, not an exception."""
        verify = AsyncMock()

        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            block = await self._call(
                self._meta(), verify_mock=verify, git_ops=self._git_ops(sha=''),
            )

        assert block is not None
        assert block.reason == 'main_sha_unresolved'
        assert block.main_sha is None
        verify.assert_not_awaited()

        empty_text = self._warnings(caplog)[0].getMessage()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            await self._call(
                self._meta(), verify_mock=AsyncMock(),
                git_ops=self._git_ops(raises=True),
            )
        raising_text = self._warnings(caplog)[0].getMessage()
        assert empty_text != raising_text

    # -- never-raises -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_verify_raising_degrades_to_errored_block(self, caplog):
        """Defence in depth: ``verify_delivered_checks_on_main`` documents
        "Never raises", but if that contract is ever broken the guard must
        still not be able to CRASH a mark-done path. It degrades to a
        fail-safe ``errored`` block instead of propagating."""
        verify = AsyncMock(side_effect=RuntimeError('verify exploded'))

        with caplog.at_level(logging.WARNING, logger=_SEAM_LOGGER.name):
            block = await self._call(
                self._meta(), verify_mock=verify, git_ops=self._git_ops(),
            )

        assert block is not None
        assert block.reason == 'errored'
        assert block.main_sha == self._SHA

    # -- pre-resolved main_sha mode ----------------------------------------

    @pytest.mark.asyncio
    async def test_supplied_main_sha_short_circuits_git(self):
        """The two seams that already hold a main SHA (``reconcile_landed_row``
        RC-2 and ``_handle_already_done_report``) pass it through.

        Not merely an optimization — a CORRECTNESS requirement: both have
        already made an ancestry decision against that exact SHA, and
        re-resolving could audit a DIFFERENT (newer) main than the decision
        being gated.
        """
        supplied = 'f' * 40
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='all_delivered', main_sha=supplied,
        ))
        git_ops = self._git_ops()

        block = await self._call(
            self._meta(), verify_mock=verify, git_ops=git_ops, main_sha=supplied,
        )

        assert block is None
        git_ops.get_main_sha.assert_not_awaited()
        assert verify.await_args.kwargs['main_sha'] == supplied

    @pytest.mark.asyncio
    async def test_supplied_main_sha_permits_no_git_ops(self):
        """In pre-resolved mode ``git_ops=None`` is acceptable — merge_queue's
        module-level ``reconcile_landed_row`` has no git_ops handle at all."""
        supplied = 'e' * 40
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='all_delivered', main_sha=supplied,
        ))

        block = await self._call(
            self._meta(), verify_mock=verify, git_ops=None, main_sha=supplied,
        )

        assert block is None
        assert verify.await_args.kwargs['main_sha'] == supplied

    @pytest.mark.asyncio
    async def test_no_git_ops_and_no_main_sha_is_fail_safe(self):
        """Conversely: no git_ops AND no pre-resolved SHA means the guard
        cannot resolve a ref at all -> fail-safe block, never a silent pass."""
        verify = AsyncMock()

        block = await self._call(self._meta(), verify_mock=verify, git_ops=None)

        assert block is not None
        assert block.reason == 'main_sha_unresolved'
        verify.assert_not_awaited()

    # -- delegation contract ------------------------------------------------

    @pytest.mark.asyncio
    async def test_forwards_project_root_timeout_and_runner_verbatim(self):
        """The helper is a DECISION layer, not a second check runner: the
        checks list, project_root, check_timeout_secs and the injected runner
        all reach ``verify_delivered_checks_on_main`` verbatim."""
        checks = [self._grep_check('a', 'PatA'), self._grep_check('b', 'PatB')]
        verify = AsyncMock(return_value=DeliveredChecksVerdict(
            outcome='all_delivered', main_sha=self._SHA,
        ))

        async def _custom_runner(argv, **kwargs):
            return (0, '', '')

        block = await self._call(
            self._meta(checks),
            verify_mock=verify,
            git_ops=self._git_ops(),
            project_root='/some/main/checkout',
            check_timeout_secs=17.5,
            runner=_custom_runner,
        )

        assert block is None
        verify.assert_awaited_once()
        assert verify.await_args.args[0] == checks
        assert verify.await_args.kwargs['project_root'] == '/some/main/checkout'
        assert verify.await_args.kwargs['check_timeout_secs'] == 17.5
        assert verify.await_args.kwargs['main_sha'] == self._SHA
        assert verify.await_args.kwargs['runner'] is _custom_runner
