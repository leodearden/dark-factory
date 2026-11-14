"""One offline-lane pass: ``offline_lane_pass.run_pass`` over injected seams, a real run-record store
and either a recording engine or the real one over a fake task client (PRD scenario 23)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from _lane_episode_fakes import (
    COMMAND,
    HEAD1,
    LANE_COMMAND,
    A,
    Clock,
    FakeAncestry,
    FakeLedgerClient,
    _run,
)

from orchestrator.config import OrchestratorConfig
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    ConfirmTarget,
    PassOutcome,
    Unconfirmable,
    UnconfirmableReason,
    outcome_to_json,
)
from orchestrator.offline_lane_confirm import confirm as lane_confirm
from orchestrator.offline_lane_episode_state import PassEffect, PassObservation
from orchestrator.offline_lane_episodes import LaneEngine
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_pass import (
    ConfirmSeam,
    DetectSeam,
    LaneLeg,
    PassObserver,
    PassOutcomeHook,
    lane_pass_outcome_hooks,
    run_pass,
)
from orchestrator.offline_lane_records import (
    LEGACY_INFRA_COMMAND,
    LEGACY_NUMERIC_COMMAND,
    Leg,
    Phase,
    RunRecord,
    RunRecordStore,
)
from orchestrator.offline_lane_runs import ConfirmRunner

GENERIC = LaneLeg.generic(LANE_COMMAND)
REPRODUCED = Confirmed(ConfirmMode.NARROWED, (A,), (), ())
NOTE = EvidenceNote('hook', None, 'suite-wide leak')
EFFECT = PassEffect(
    tests_counted=(A,), episodes_started=(A,), episodes_extended=(), episodes_ended=(),
    threshold_reached=(), stale_heads=(), no_decision=(), tasks_filed=(), escalations_raised=(),
)


class RecordingEngine:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.observations: list[PassObservation] = []

    async def observe(self, observation: PassObservation) -> PassEffect:
        self.log.append('observe')
        self.observations.append(observation)
        return EFFECT


@dataclass
class FakeConfirm:
    """Answers *outcome*, leaving a confirm record on the pass the way the lane's confirm does."""

    store: RunRecordStore
    log: list[str]
    outcome: Confirmed | Unconfirmable = REPRODUCED
    calls: list[tuple[ConfirmTarget, RunRecord | None]] = field(default_factory=list)

    async def __call__(
        self, target: ConfirmTarget, detect: RunRecord | None, runner: ConfirmRunner,
    ) -> Confirmed | Unconfirmable:
        self.log.append('confirm')
        self.calls.append((target, detect))
        lane_pass = self.store.current_pass(COMMAND)
        assert lane_pass is not None
        self.store.write_run(lane_pass, _run('confirm', lane_pass.head, self.store.now(), failing=(A,)))
        return self.outcome


@dataclass
class RecordingHook:
    log: list[str]
    notes: tuple[EvidenceNote, ...] = (NOTE,)
    calls: list[tuple[str, str, PassOutcome, tuple[RunRecord, ...]]] = field(default_factory=list)

    async def __call__(
        self, command: str, head: str, outcome: PassOutcome, run_records: Sequence[RunRecord],
    ) -> Sequence[EvidenceNote]:
        self.log.append('hook')
        self.calls.append((command, head, outcome, tuple(run_records)))
        return self.notes


async def _raising_hook(
    command: str, head: str, outcome: PassOutcome, run_records: Sequence[RunRecord],
) -> Sequence[EvidenceNote]:
    raise RuntimeError('hook broke')


def _detect(store: RunRecordStore, *, failing: tuple[str, ...] = ()) -> DetectSeam:
    """A generic command's detect seam that leaves a real detect record, red when *failing* names tests."""

    async def detect() -> tuple[int, str]:
        lane_pass = store.current_pass(COMMAND)
        assert lane_pass is not None
        record = _run('detect', lane_pass.head, store.now(), failing=failing)
        store.write_run(lane_pass, record)
        return record.returncode, ''

    return detect


def _injected(returncode: int) -> DetectSeam:
    async def detect() -> tuple[int, str]:
        return returncode, ''

    return detect


@dataclass
class Pass:
    tmp_path: Path
    clock: Clock = field(default_factory=Clock)
    log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.store = RunRecordStore(self.tmp_path, clock=self.clock)
        self.engine = RecordingEngine(self.log)

    async def run(
        self,
        detect: DetectSeam,
        *,
        target: LaneLeg = GENERIC,
        confirm: ConfirmSeam | None = None,
        hooks: Sequence[PassOutcomeHook] = (),
        engine: PassObserver | None = None,
    ) -> bool:
        return await run_pass(
            target,
            worktree=self.tmp_path,
            head=HEAD1,
            detect=detect,
            confirm=confirm or FakeConfirm(self.store, self.log),
            store=self.store,
            engine=engine or self.engine,
            hooks=hooks,
        )

    def observed(self) -> PassObservation:
        (observation,) = self.engine.observations
        return observation

    def pass_blocks(self, command: str = COMMAND) -> tuple[object, object]:
        lane_pass = self.store.current_pass(command)
        assert lane_pass is not None
        record = self.store.load_pass(command, lane_pass.pass_id)
        assert record is not None
        return record.outcome, record.effect


@pytest.mark.asyncio
async def test_a_green_pass_is_observed_with_no_outcome_and_asks_neither_confirm_nor_hooks(tmp_path):
    lane = Pass(tmp_path)
    confirm_seam = FakeConfirm(lane.store, lane.log)
    hook = RecordingHook(lane.log)

    green = await lane.run(_detect(lane.store), confirm=confirm_seam, hooks=(hook,))

    observation = lane.observed()
    assert green is True
    assert lane.log == ['observe']
    assert observation.outcome is None
    assert observation.notes == ()
    assert observation.detect == lane.store.last_record(COMMAND, Phase.DETECT)
    assert observation.confirm is None
    assert lane.pass_blocks() == (None, EFFECT.to_json())


@pytest.mark.asyncio
async def test_a_red_pass_is_confirmed_then_shown_to_the_hooks_then_observed_with_their_notes(tmp_path):
    lane = Pass(tmp_path)
    confirm_seam = FakeConfirm(lane.store, lane.log)
    hook = RecordingHook(lane.log)

    green = await lane.run(_detect(lane.store, failing=(A,)), confirm=confirm_seam, hooks=(hook,))

    detect = lane.store.last_record(COMMAND, Phase.DETECT)
    confirmed = lane.store.last_record(COMMAND, Phase.CONFIRM)
    observation = lane.observed()
    assert green is False
    assert lane.log == ['confirm', 'hook', 'observe']
    assert confirm_seam.calls == [(ConfirmTarget.generic(LANE_COMMAND, tmp_path), detect)]
    assert hook.calls == [(COMMAND, HEAD1, REPRODUCED, (detect, confirmed))]
    assert (observation.outcome, observation.notes) == (REPRODUCED, (NOTE,))
    assert (observation.detect, observation.confirm) == (detect, confirmed)
    assert lane.pass_blocks() == (outcome_to_json(REPRODUCED), EFFECT.to_json())


@pytest.mark.asyncio
async def test_a_raising_hook_loses_only_its_own_notes(tmp_path, caplog):
    lane = Pass(tmp_path)
    survivor = RecordingHook(lane.log)

    with caplog.at_level(logging.ERROR, logger='orchestrator.offline_lane_pass'):
        await lane.run(_detect(lane.store, failing=(A,)), hooks=(_raising_hook, survivor))

    assert lane.observed().notes == (NOTE,)
    assert len(survivor.calls) == 1
    assert any('hook' in r.getMessage() and r.exc_info for r in caplog.records)


@pytest.mark.asyncio
async def test_a_green_injected_detect_seam_is_a_green_pass_with_no_detect_record(tmp_path):
    lane = Pass(tmp_path)

    green = await lane.run(_injected(0), confirm=lane_confirm)

    observation = lane.observed()
    assert green is True
    assert (observation.detect, observation.outcome) == (None, None)
    assert lane.pass_blocks() == (None, EFFECT.to_json())


@pytest.mark.asyncio
async def test_a_red_injected_detect_seam_is_unconfirmable_for_want_of_a_detect_record(tmp_path):
    lane = Pass(tmp_path)
    hook = RecordingHook(lane.log)

    green = await lane.run(_injected(1), confirm=lane_confirm, hooks=(hook,))

    unconfirmable = Unconfirmable(UnconfirmableReason.NO_DETECT_RECORD, '')
    observation = lane.observed()
    assert green is False
    assert (observation.detect, observation.outcome, observation.notes) == (None, unconfirmable, (NOTE,))
    assert hook.calls == [(COMMAND, HEAD1, unconfirmable, ())]
    assert lane.pass_blocks() == (outcome_to_json(unconfirmable), EFFECT.to_json())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('target', 'command', 'leg', 'label'),
    [
        (LaneLeg.numeric(), LEGACY_NUMERIC_COMMAND, Leg.NUMERIC, 'numeric'),
        (LaneLeg.infra(), LEGACY_INFRA_COMMAND, Leg.INFRA, 'infra'),
    ],
)
async def test_a_legacy_leg_passes_through_the_same_pipeline_at_high_priority(
    tmp_path, caplog, target: LaneLeg, command: str, leg: Leg, label: str,
):
    lane = Pass(tmp_path)

    with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane_pass'):
        await lane.run(_injected(0), target=target)

    observation = lane.observed()
    assert (observation.command, observation.leg, observation.lane_command) == (command, leg.value, None)
    assert observation.fix_task_priority == 'high'
    assert [r.getMessage().split(' head=')[0] for r in caplog.records] == [f'offline-lane: {label} sub-run']
    assert lane.pass_blocks(command) == (None, EFFECT.to_json())


@pytest.mark.asyncio
async def test_a_generic_command_passes_at_its_own_priority_and_logs_its_sub_run_under_its_name(tmp_path, caplog):
    lane = Pass(tmp_path)

    with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane_pass'):
        await lane.run(_detect(lane.store, failing=(A,)))

    (line,) = [r.getMessage() for r in caplog.records]
    assert lane.observed().fix_task_priority == LANE_COMMAND.fix_task_priority
    assert lane.observed().lane_command == LANE_COMMAND
    assert line.startswith(f'offline-lane: {COMMAND} sub-run head={HEAD1[:12]} status=FAIL ended_by=exit duration=')


def test_production_has_no_pass_outcome_hooks_yet(tmp_path):
    assert lane_pass_outcome_hooks(OrchestratorConfig(project_root=tmp_path), None) == ()


@pytest.mark.asyncio
async def test_scenario_23_a_hook_note_reaches_the_filed_task_digest(tmp_path):
    lane = Pass(tmp_path)
    client = FakeLedgerClient(lane.log)
    engine = LaneEngine(
        OrchestratorConfig(project_root=tmp_path), FakeAncestry(),
        task_client=client, escalation_queue=None, now=lane.clock,
    )
    hook = RecordingHook(lane.log)

    await lane.run(_detect(lane.store, failing=(A,)), hooks=(hook,), engine=engine)
    lane.clock.advance(minutes=30)
    await lane.run(_detect(lane.store, failing=(A,)), hooks=(hook,), engine=engine)

    (filed,) = client.submitted
    assert '  [hook] (all reproduced tests): suite-wide leak' in filed['details']
    _, effect = lane.pass_blocks()
    assert isinstance(effect, dict)
    assert len(effect['tasks_filed']) == 1
