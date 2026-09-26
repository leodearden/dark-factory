"""Shared scaffolding for the task-4427 campaign-gate pins.

Three test files pin the same contract from different entry points —
``test_eval_driver.py`` (the four stage fan-outs and the two cell executors),
``test_runner_matrix.py`` (``run_eval_matrix``) and ``test_eval_architect.py``
(``run_architect_eval`` plus the CM itself) — so the probe, the sentinels, the
assertion bodies and the base-config builder live here rather than being cloned
into each. They were cloned once already (reviewer: duplication), which is how
the in-cell teardown guard came to be wrong in two places at once.

Not ``_orch_helpers.py``: this is one task's scaffolding for one contract, and a
module of its own keeps that scope legible (heuristic 6) without growing the
2200-line general-purpose helper further.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.evals.runner import EvalResult

# Distinguishes "this case passed no usage_gate at all" (the OWNED path, where
# the executor builds and tears down its own) from "this case passed
# usage_gate=None" (a campaign owner that is deliberately ungated). The
# production _GATE_UNSET sentinel exists for exactly this distinction; the
# tests need their own because a plain None default could not express it.
NO_INJECTED_GATE = object()

# The same distinction read from the other end: what a recording fake saw when
# its caller passed no ``usage_gate=`` argument at all. A cell reaching this
# value is the bug — it would build a gate of its own.
GATE_ARG_MISSING = object()


def eval_base_config(tmp_path: Path):
    """A deterministic pure-code-default base config via the REAL load_config().

    Writes a minimal YAML setting only ``project_root`` so ``load_config``
    layers it over the packaged ``defaults.yaml`` — every leaf resolves to its
    code default through the real production config-load entry point, never a
    hand-built ``OrchestratorConfig``.

    Must be non-None for any gate case: ``campaign_usage_gate(None)`` is
    deliberately ungated, so a ``None`` base config never builds a gate to
    observe.
    """
    from orchestrator.config import load_config

    cfg_path = tmp_path / 'orchestrator.yaml'
    cfg_path.write_text(f'project_root: {tmp_path}\n')
    return load_config(cfg_path)


def stage_task_loader(path: Path) -> dict:
    """Minimal task dict for a fan-out whose executors are all faked.

    Carries the union of what the campaign loops read before dispatch:
    ``run_eval_matrix`` reads ``task['id']`` (only for ``_result_exists``,
    which ``force=True`` skips) and the stage thunks read nothing else. The
    executors themselves are monkeypatched, so it is never deeply inspected.
    """
    return {'id': path.stem, 'project_root': '/fake', 'pre_task_commit': 'x'}


class CampaignGateProbe:
    """Records what each cell of a campaign fan-out was handed.

    ``_build_eval_usage_gate`` MUST be patched for any of this: the base config
    is a REAL config whose packaged default is ``usage_cap.enabled=true``, so
    the unpatched builder would construct a live ``UsageGate`` (probe dirs,
    account state, a SIGHUP handler).

    OBSERVE IN THE FAKE, ASSERT IN THE HELPER. The fake runs as a CELL, and
    every campaign loop here log-and-continues on non-cancel exceptions, so an
    ``AssertionError`` raised inside it is swallowed and the cell is silently
    dropped from the results — an in-cell assert cannot fail the test it exists
    to strengthen. (Measured: an ``assert False`` in the fake left every
    campaign-gate case green.) So the fake only RECORDS, and
    :func:`assert_one_gate_serves_every_cell` checks the records afterwards,
    including that no cell was dropped.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, gate=None):
        from orchestrator.evals import runner

        self._runner = runner
        self._monkeypatch = monkeypatch
        self.gate = gate
        # The ``usage_gate=`` argument each cell received, in call order.
        self.seen: list = []
        # The gate's ``shutdown`` await count AT THE MOMENT each cell ran. All
        # zeros is the only evidence that distinguishes "torn down once after
        # the fan-out" (correct) from "torn down between cells" (which leaves
        # later cells running against a dead gate) — an after-the-fact
        # ``assert_awaited_once`` cannot tell those apart.
        self.shutdowns_at_call: list[int] = []
        # Cells that raised deliberately (``fail_on`` / ``cancel_on``), so a
        # helper can tell an expected drop from a swallowed accident.
        self.raised: list[str] = []
        self.build = AsyncMock(return_value=gate)
        monkeypatch.setattr(runner, '_build_eval_usage_gate', self.build)
        monkeypatch.setattr(runner, 'load_task', stage_task_loader)

    def install(self, *executor_names: str, fail_on: str | None = None,
                cancel_on: str | None = None):
        """Patch each named executor with the recording fake."""
        async def fake(task_path, *args, trial=1,
                       usage_gate=GATE_ARG_MISSING, **_kwargs):
            self.seen.append(usage_gate)
            self.shutdowns_at_call.append(
                self.gate.shutdown.await_count if self.gate is not None else 0
            )
            if cancel_on is not None and cancel_on in task_path.stem:
                self.raised.append(task_path.stem)
                raise asyncio.CancelledError()
            if fail_on is not None and fail_on in task_path.stem:
                self.raised.append(task_path.stem)
                raise RuntimeError('boom in one cell')
            label = getattr(args[0], 'name', 'cell') if args else 'cell'
            return EvalResult(task_path.stem, label, 'done', {}, '/tmp/wt',
                              trial=trial)

        for name in executor_names:
            self._monkeypatch.setattr(self._runner, name, fake)


def distinct_gates(probe: CampaignGateProbe) -> set[int]:
    return {id(g) for g in probe.seen}


def _assert_no_cell_was_dropped(probe: CampaignGateProbe, results) -> None:
    """Every cell that ran is accounted for in *results*.

    The campaign loops drop a cell whose coroutine raised, so without this a
    swallowed exception inside the fake looks exactly like a healthy fan-out to
    every other assertion here.
    """
    assert len(results) == len(probe.seen) - len(probe.raised), (
        f'{len(probe.seen)} cells ran, {len(probe.raised)} raised deliberately, '
        f'but only {len(results)} results came back — a cell was dropped'
    )


async def assert_one_gate_serves_every_cell(probe: CampaignGateProbe, stage):
    results = await stage()

    # ONE build for the whole fan-out, however many cells it expanded to.
    probe.build.assert_awaited_once()
    assert len(probe.seen) > 1, 'the case must fan out to several cells'
    assert not probe.raised, 'the healthy case must not have a failing cell'
    _assert_no_cell_was_dropped(probe, results)
    # …and literally the same object in every cell: one cap-state view.
    assert distinct_gates(probe) == {id(probe.gate)}
    # No cell ran against an already-shut-down gate…
    assert probe.shutdowns_at_call == [0] * len(probe.seen), (
        'the campaign gate was torn down while cells were still running — '
        'later cells would be left without failover'
    )
    # …and it came down exactly once, after the last of them returned.
    assert probe.gate is not None
    probe.gate.shutdown.assert_awaited_once()
    return results


async def assert_teardown_survives_a_failing_cell(probe: CampaignGateProbe, stage):
    results = await stage()

    # The surviving cells still completed (the pre-existing continue-on-failure
    # contract is untouched) and the gate is still torn down exactly once.
    assert results, 'the non-failing cells must still return their results'
    assert probe.raised, 'the case must actually have failed a cell'
    _assert_no_cell_was_dropped(probe, results)
    assert probe.gate is not None
    probe.gate.shutdown.assert_awaited_once()
    return results


async def assert_teardown_survives_cancellation(probe: CampaignGateProbe, stage):
    with pytest.raises(asyncio.CancelledError):
        await stage()

    # Cancellation still propagates AND the gate is torn down — no leaked probe
    # loop on the SIGINT path, which is the one an operator actually takes.
    assert probe.raised, 'the case must actually have cancelled a cell'
    assert probe.gate is not None
    probe.gate.shutdown.assert_awaited_once()


async def assert_a_degraded_campaign_stays_ungated(probe: CampaignGateProbe, stage):
    results = await stage()

    # Built once, degraded to None — and every cell is told so EXPLICITLY. A
    # cell that received no argument would build its own gate, restoring the
    # per-cell construction the hoist removes.
    probe.build.assert_awaited_once()
    assert len(probe.seen) > 1
    assert not probe.raised
    _assert_no_cell_was_dropped(probe, results)
    assert probe.seen == [None] * len(probe.seen)
    return results
