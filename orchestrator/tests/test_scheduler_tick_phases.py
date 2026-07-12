"""Phase-level tests for Scheduler.acquire_next's tick-phase decomposition.

These construct a ``Scheduler`` + hand-build a ``TickContext`` and call the
``_phase_*`` methods directly — deliberately WITHOUT a full-tick harness
(no ``get_tasks``/``acquire_next`` orchestration is driven here). The
existing ``test_scheduler*.py`` files remain the untouched behaviour-parity
suite that exercises full ticks end-to-end.
"""

import dataclasses

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler, TickContext


@pytest.fixture
def scheduler() -> Scheduler:
    config = OrchestratorConfig(max_per_module=1)
    sched = Scheduler(config)
    sched.finish_startup()
    return sched


class TestTickContextShape:
    def test_tick_context_shape(self):
        assert dataclasses.is_dataclass(TickContext)
        field_names = {f.name for f in dataclasses.fields(TickContext)}
        task_named_fields = {
            'tasks', 'status_map', 'tasks_by_id',
            'external_cache', 'overrides', 'candidates',
        }
        assert task_named_fields <= field_names

        # Real construction using all six task-named fields — proves each
        # is a valid constructor kwarg.
        ctx = TickContext(
            tasks=[{'id': '1'}],
            status_map={'1': 'pending'},
            tasks_by_id={'1': {'id': '1'}},
            external_cache={'other:1': 'done'},
            overrides={},
            candidates=[{'id': '1'}],
        )
        assert ctx.tasks == [{'id': '1'}]
        assert ctx.status_map == {'1': 'pending'}
        assert ctx.tasks_by_id == {'1': {'id': '1'}}
        assert ctx.external_cache == {'other:1': 'done'}
        assert ctx.overrides == {}
        assert ctx.candidates == [{'id': '1'}]

        # Minimal construction (only the three fields with no default)
        # spot-checks the threading-field defaults.
        default_ctx = TickContext(tasks=[], status_map={}, tasks_by_id={})
        assert default_ctx.max_id == 0
        assert default_ctx.external_resolver_failed is False
        assert default_ctx.candidates == []
        assert default_ctx.gated_ids == set()
        assert default_ctx.stale_ids == set()
        assert default_ctx.psi_hold is False
        assert default_ctx.dispatch_deferred_emitted is False
