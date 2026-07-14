"""Tests for the W9-θ ``CancellationScope`` supervisor and ``WorkflowCancelled``.

PRD: plans/workflow-state-machine-prd.md task θ (§8.1 ``WorkflowCancelled``,
§8.2 CX-1, §9 boundary rows 14-15). Replaces three coupled cancellation
mechanisms (harness hard-cancel `task.cancel()` + the ``sys.exc_info()`` B1
sniff / harness B2 dual-guard; the soft ``_cancel_event`` opt-in awaits) with
ONE typed ``WorkflowCancelled(kind)`` caught at exactly one place in
``TaskWorkflow.run()``, and an ordered kind-aware ``on_terminal`` cleanup list.

Test coverage:
  step-01: pure-unit ``WorkflowCancelled`` construct/raise/catch contract
  step-03: ``CancellationScope.supervise`` soft-cancel + normal-return paths
  step-05: ``CancellationScope.supervise`` hard-cancel + repeated-cancel
    cleanup-survival paths
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.workflow_types import WorkflowCancelled

# ---------------------------------------------------------------------------
# step-01: WorkflowCancelled construct/raise/catch/read contract
# ---------------------------------------------------------------------------


class TestWorkflowCancelledType:
    """Pins the frozen-dataclass-Exception construct/raise/catch/read contract
    — the one asyncio-adjacent gotcha to smoke out before anything (the
    ``CancellationScope`` supervisor, ``run()``'s single catch site) depends
    on it.
    """

    def test_constructs_with_hard_kind(self):
        wc = WorkflowCancelled(kind='hard')
        assert isinstance(wc, Exception)
        assert wc.kind == 'hard'

    def test_constructs_with_soft_kind(self):
        wc = WorkflowCancelled(kind='soft')
        assert isinstance(wc, Exception)
        assert wc.kind == 'soft'

    def test_raise_and_catch_hard(self):
        with pytest.raises(WorkflowCancelled) as excinfo:
            raise WorkflowCancelled(kind='hard')
        assert excinfo.value.kind == 'hard'

    def test_raise_and_catch_soft(self):
        with pytest.raises(WorkflowCancelled) as excinfo:
            raise WorkflowCancelled(kind='soft')
        assert excinfo.value.kind == 'soft'

    def test_raise_and_catch_via_except_as_reads_kind(self):
        # The exact idiom run() uses: `except WorkflowCancelled as wc: ... wc.kind`.
        caught_kind = None
        try:
            raise WorkflowCancelled(kind='soft')
        except WorkflowCancelled as wc:
            caught_kind = wc.kind
        assert caught_kind == 'soft'

    @pytest.mark.asyncio
    async def test_raise_and_catch_across_an_await_boundary(self):
        # WorkflowCancelled must survive propagation out of a coroutine —
        # this is how it will actually travel (out of CancellationScope.supervise,
        # an `await`, up to run()'s single catch site).
        async def _raises() -> None:
            await asyncio.sleep(0)
            raise WorkflowCancelled(kind='hard')

        with pytest.raises(WorkflowCancelled) as excinfo:
            await _raises()
        assert excinfo.value.kind == 'hard'
