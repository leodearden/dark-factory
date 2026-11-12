"""verify.py's admission seams that a host race reads (plans/remote-task-leg-verify-prd.md decision 14).

``run_scoped_verification`` takes a caller's ``AdmissionPlan`` so a verify moved
between hosts keeps its arrival, and a test leg marks its plan's latch once it
holds a turn (or runs without the gate), before its command spawns. Driven
through the public ``run_scoped_verification``, ``admission_active`` and
``admission_policy`` with real shell commands on a scratch directory. Every
config points ``verify_admission_slots_dir`` under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from shared.verify_admission import acquire_task_slot

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.verify import admission_active, admission_policy, run_scoped_verification
from orchestrator.verify_admission_gate import AdmissionPlan, AdmissionPolicy, gate_for


def _config(tmp_path: Path, **overrides: Any) -> OrchestratorConfig:
    fields: dict[str, Any] = {
        'project_root': tmp_path,
        'verify_admission_slots_dir': str(tmp_path / 'slots'),
        'verify_admission_task_slots': 1,
        'verify_clock_stop_enabled': False,
        'verify_env_probes': [],
        'verify_admission_nice_task': 'nice -n 10',
        'verify_admission_nice_background': 'nice -n 10',
    }
    fields.update(overrides)
    return OrchestratorConfig(**fields)


class _Scratch:
    """A scratch worktree with one module whose test command touches a marker."""

    def __init__(self, tmp_path: Path) -> None:
        self.worktree = tmp_path / 'wt'
        (self.worktree / 'pkg').mkdir(parents=True)
        self.marker = tmp_path / 'test-leg-ran'

    def module(self, *, with_test: bool = True) -> ModuleConfig:
        return ModuleConfig(
            prefix='pkg',
            test_command=f'touch {shlex.quote(str(self.marker))}' if with_test else None,
            lint_command='true',
            type_check_command=None,
        )

    async def verify(self, config: OrchestratorConfig, **kwargs: Any):
        with_test = kwargs.pop('with_test', True)
        async with asyncio.timeout(30):
            return await run_scoped_verification(
                self.worktree, config, [self.module(with_test=with_test)],
                task_files=[], task_id=kwargs.pop('task_id', '7'), role=kwargs.pop('role', 'task'),
                **kwargs,
            )


def _plan_ids(store: EventStore) -> list[str]:
    return [row['data']['plan_id'] for row in store.fetch_events_by_type(EventType.verify_admission_plan)]


async def _until(
    predicate: Callable[[], bool], *, watching: asyncio.Task | None = None, timeout: float = 10.0,
) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            if watching is not None and watching.done():
                watching.result()
                raise AssertionError('the verify finished before reaching the awaited state')
            await asyncio.sleep(0.01)


@pytest.mark.real_verify_admission
@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
class TestCallerPlan:
    async def test_a_passed_plan_is_the_plan_the_call_records(self, tmp_path):
        scratch = _Scratch(tmp_path)
        store = EventStore(tmp_path / 'runs.db', run_id='t')
        plan = AdmissionPlan.begin(role='task', task_id='7')

        result = await scratch.verify(_config(tmp_path), admission_plan=plan, event_store=store)

        assert result.passed
        assert _plan_ids(store) == [plan.plan_id]

    async def test_calls_without_a_plan_record_a_plan_each(self, tmp_path):
        scratch = _Scratch(tmp_path)
        store = EventStore(tmp_path / 'runs.db', run_id='t')

        await scratch.verify(_config(tmp_path), event_store=store)
        await scratch.verify(_config(tmp_path), event_store=store)

        first, second = _plan_ids(store)
        assert first != second


@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
class TestCallerPlanMustMatchTheCall:
    async def test_a_plan_of_another_role_is_refused_before_any_leg(self, tmp_path):
        scratch = _Scratch(tmp_path)
        plan = AdmissionPlan.begin(role='background', task_id='7')

        with pytest.raises(ValueError, match=r"'background'.*'task'|'task'.*'background'"):
            await scratch.verify(_config(tmp_path), admission_plan=plan)

        assert not scratch.marker.exists()

    async def test_a_merge_call_refuses_a_task_plan(self, tmp_path):
        scratch = _Scratch(tmp_path)
        plan = AdmissionPlan.begin(role='task', task_id='7')

        with pytest.raises(ValueError, match=r"'task'.*'merge'|'merge'.*'task'"):
            await scratch.verify(_config(tmp_path), admission_plan=plan, role='merge')

        assert not scratch.marker.exists()

    async def test_a_plan_of_another_task_is_refused_before_any_leg(self, tmp_path):
        scratch = _Scratch(tmp_path)
        plan = AdmissionPlan.begin(role='task', task_id='8')

        with pytest.raises(ValueError, match=r"'8'.*'7'|'7'.*'8'"):
            await scratch.verify(_config(tmp_path), admission_plan=plan, task_id='7')

        assert not scratch.marker.exists()


@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
class TestLatchWithoutTheGate:
    async def test_a_test_leg_run_ungated_marks_its_plan(self, tmp_path):
        scratch = _Scratch(tmp_path)
        plan = AdmissionPlan.begin(role='task', task_id='7')

        result = await scratch.verify(_config(tmp_path), admission_plan=plan)

        assert result.passed
        assert scratch.marker.exists()
        assert plan.has_been_granted() is True

    async def test_a_plan_with_no_test_leg_stays_unmarked(self, tmp_path):
        scratch = _Scratch(tmp_path)
        plan = AdmissionPlan.begin(role='task', task_id='7')

        result = await scratch.verify(_config(tmp_path), admission_plan=plan, with_test=False)

        assert result.passed
        assert plan.has_been_granted() is False


@pytest.mark.real_verify_admission
@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
class TestLatchThroughTheGate:
    async def test_a_gated_test_leg_marks_its_plan(self, tmp_path):
        scratch = _Scratch(tmp_path)
        plan = AdmissionPlan.begin(role='task', task_id='7')

        result = await scratch.verify(_config(tmp_path), admission_plan=plan)

        assert result.passed
        assert result.admission_legs is not None
        assert plan.has_been_granted() is True

    async def test_a_granted_leg_waiting_for_its_flock_already_reads_granted(self, tmp_path):
        scratch = _Scratch(tmp_path)
        config = _config(tmp_path)
        slots_dir = Path(config.verify_admission_slots_dir)
        slots_dir.mkdir()
        plan = AdmissionPlan.begin(role='task', task_id='7')

        with acquire_task_slot('task', slots_dir=slots_dir, n=1, wait=False) as held:
            assert held is True
            run = asyncio.create_task(scratch.verify(config, admission_plan=plan))
            await _until(plan.has_been_granted, watching=run)
            await asyncio.sleep(0.2)

            assert not scratch.marker.exists()
            assert len(gate_for(slots_dir).snapshot().holders) == 1
        async with asyncio.timeout(30):
            result = await run

        assert result.passed
        assert scratch.marker.exists()


@pytest.mark.usefixtures('code_default_config')
class TestAdmissionStateReads:
    def test_the_conftest_seam_governs_admission_active(self, tmp_path):
        assert admission_active(_config(tmp_path, verify_admission_enabled=True)) is False

    @pytest.mark.real_verify_admission
    @pytest.mark.parametrize('enabled', [True, False])
    def test_admission_active_follows_config(self, tmp_path, enabled):
        assert admission_active(_config(tmp_path, verify_admission_enabled=enabled)) is enabled

    def test_admission_policy_states_the_config(self, tmp_path):
        config = _config(
            tmp_path,
            verify_admission_task_slots=3,
            verify_admission_order='arrival',
            verify_admission_background_promote_secs=42.0,
        )

        assert admission_policy(config) == AdmissionPolicy(
            slots=3, order='arrival', promote_after_secs=42.0,
        )
