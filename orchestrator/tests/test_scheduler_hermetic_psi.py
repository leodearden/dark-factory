"""Tests for the suite-wide PSI-hermeticity seam (task 2418).

Two independent host-state dependencies made ~11 scheduler / scheduler-state
tests non-deterministic under full parallel ``pytest -n auto --dist
loadgroup``: the dispatch-admission gate reads real ``/proc/pressure/*`` by
default (``psi_admission.enabled`` defaults True), and under host load
``mem_some_avg10`` can cross the configured threshold and defer dispatch of
non-deterministic candidates. This file pins the fix for that half of the
flake (the snapshot-path half is covered by
``test_scheduler_state.py::TestWriteStateSnapshot``):

  step-3 (RED)  An idle PsiSample neutralizes the dispatch-admission gate;
                a saturated one (injected, never host-sampled) still defers
                a non-deterministic candidate exactly as designed. Proves
                ``idle_psi_sample()`` is a safe, deterministic default for
                every scheduler test — saturation is injected here, never
                sampled from the host.
  step-5 (RED)  The autouse conftest fixture defaults every bare Scheduler's
                ``_read_psi_sample`` to ``idle_psi_sample``, so the suite is
                host-independent by default without per-test injection.

Mirrors test_scheduler_dispatch_admission.py's conventions: module-local
task-dict helper, ``scheduler.get_tasks = AsyncMock(return_value=[...])``,
driving ``acquire_next()`` directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _orch_helpers import idle_psi_sample
from shared.psi import PsiSample

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler


def _pending_task(
    task_id: str,
    *,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal non-deterministic (no task_kind metadata) pending-task dict.

    Absence of ``metadata.task_kind`` makes ``Scheduler.is_deterministic``
    return False (see scheduler.py), so this candidate is fully subject to
    the dispatch-admission gate — deterministic-kind tasks are exempt and
    would not exercise psi_hold at all.
    """
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'dependencies': [],
        'metadata': {'files': files or [f'{task_id}/src']},
        'priority': priority,
    }


@pytest.mark.asyncio
class TestIdlePsiSampleNeutralizesAdmissionGate:
    """idle_psi_sample() is a safe, deterministic non-saturating default;
    an injected saturated sample still defers as designed."""

    async def test_idle_psi_sample_neutralizes_admission_gate(self) -> None:
        task = _pending_task('X', priority='high')

        # Phase A: a saturated PSI sample — injected directly, never
        # sampled from the host — defers the non-deterministic candidate.
        # mem_some10=99.0 is over the default mem_some_avg10=15.0 threshold.
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler.finish_startup()
        scheduler._dispatched.add('inflight')  # meets default min_inflight_floor=1
        scheduler._read_psi_sample = lambda: PsiSample(
            cpu_some10=0.0, mem_some10=99.0, mem_full10=0.0, io_some10=0.0,
            read_ok=True,
        )
        scheduler.get_tasks = AsyncMock(return_value=[task])

        assert await scheduler.acquire_next() is None, (
            'a saturated PSI sample must defer the non-deterministic candidate'
        )

        # Phase B: idle_psi_sample() neutralizes the gate — the identical
        # candidate now dispatches on a fresh scheduler. ``_dispatched`` is
        # seeded identically to Phase A so min_inflight_floor is still met
        # and the gate stays *engaged* (psi_in_flight >= floor); otherwise
        # dispatch would succeed merely because the floor wasn't met, and
        # this phase would prove nothing about idle_psi_sample specifically.
        scheduler2 = Scheduler(OrchestratorConfig(max_per_module=1))
        scheduler2.finish_startup()
        scheduler2._dispatched.add('inflight')  # meets default min_inflight_floor=1
        scheduler2._read_psi_sample = idle_psi_sample
        scheduler2.get_tasks = AsyncMock(return_value=[task])

        result = await scheduler2.acquire_next()
        assert result is not None
        assert result.task_id == 'X'


class TestAutouseFixtureDefaultsToIdlePsi:
    """Guard test for conftest's autouse ``_hermetic_psi_reader`` fixture."""

    def test_autouse_fixture_defaults_scheduler_to_idle_psi(self) -> None:
        """Every bare Scheduler defaults ``_read_psi_sample`` to an idle sample.

        Without the autouse ``_hermetic_psi_reader`` fixture, the constructor
        default (scheduler.py) is the real ``shared.psi.read_psi_sample``,
        which reads live ``/proc/pressure/*`` — the behavioral assertions
        below would then reflect actual host pressure instead of the fixed
        idle values asserted here. Mirrors
        ``test_scheduler_state.py::test_bare_config_project_root_isolated_to_tmp``,
        the guard test for the sibling autouse ``_isolate_orch_config`` fixture.
        """
        scheduler = Scheduler(OrchestratorConfig(max_per_module=1))

        # Behavior-level contract only: the bound reader returns a
        # non-saturating, successful sample regardless of host load. This is
        # the meaningful guarantee callers of _read_psi_sample rely on, and
        # it already fully covers the contract this guard test exists to
        # pin. Deliberately NOT asserting `scheduler._read_psi_sample is
        # idle_psi_sample` (implementation-coupled identity) — a harmless
        # refactor of the fixture (e.g. wrapping idle_psi_sample in a
        # lambda/partial) would break an identity check with no behavioral
        # regression, so it would add fragility without proportional
        # coverage.
        sample = scheduler._read_psi_sample()
        assert sample.read_ok is True
        assert sample.cpu_some10 == 0.0
        assert sample.mem_some10 == 0.0
        assert sample.mem_full10 == 0.0
        assert sample.io_some10 == 0.0
