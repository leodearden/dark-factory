"""One offline-lane pass: ``offline_lane_pass.run_pass`` over injected seams, a real run-record store
and either a recording engine or the real one over a fake task client (PRD scenario 23)."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable, Sequence
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
from escalation.queue import EscalationQueue
from shared.project_layout import escalation_queue_dir

from orchestrator.config import LaneCommand, OrchestratorConfig
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
    never_stopping,
    run_pass,
)
from orchestrator.offline_lane_records import (
    LEGACY_INFRA_COMMAND,
    LEGACY_NUMERIC_COMMAND,
    Leg,
    PassRecord,
    Phase,
    ProbeKind,
    ProbeResult,
    RunRecord,
    RunRecordStore,
)
from orchestrator.offline_lane_runs import ConfirmRunner, ProbeSeam, run_probe
from orchestrator.verify_cache import keys, registry
from orchestrator.verify_cache.audit_hook import AUDIT_ESCALATION_SENTINEL, VerifyCacheAuditHook
from orchestrator.verify_cache.audit_run import AUDIT_COMMAND, AuditNode
from orchestrator.verify_evidence import UnitTier

GENERIC = LaneLeg.generic(LANE_COMMAND)
REPRODUCED = Confirmed(ConfirmMode.NARROWED, (A,), (), ())
NOTE = EvidenceNote('hook', None, 'suite-wide leak')
EFFECT = PassEffect(
    tests_counted=(A,), episodes_started=(A,), episodes_extended=(), episodes_ended=(),
    threshold_reached=(), stale_heads=(), no_decision=(), tasks_filed=(), escalations_raised=(),
)
SIGTERM_ENDED = Unconfirmable(UnconfirmableReason.CONFIRM_SIGNAL_ENDED, '-15')
PROBED = LaneLeg.generic(LANE_COMMAND.model_copy(update={'outage_probe': 'python probe.py'}))
SIGTERMED_PROBE = ProbeResult(ProbeKind.BROKEN, -15, None, '', 0.5)


def _exited(record: RunRecord) -> RunRecord:
    return record


def _sigtermed(record: RunRecord) -> RunRecord:
    """*record* as a sub-run SIGTERM ended, the way a unit stop kills the lane's child."""
    return dataclasses.replace(record, ended_by='signal', returncode=-15)


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
    ended: Callable[[RunRecord], RunRecord] = _exited
    calls: list[tuple[ConfirmTarget, RunRecord | None]] = field(default_factory=list)

    async def __call__(
        self, target: ConfirmTarget, detect: RunRecord | None, runner: ConfirmRunner,
    ) -> Confirmed | Unconfirmable:
        self.log.append('confirm')
        self.calls.append((target, detect))
        lane_pass = self.store.current_pass(COMMAND)
        assert lane_pass is not None
        self.store.write_run(lane_pass, self.ended(_run('confirm', lane_pass.head, self.store.now(), failing=(A,))))
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


def _detect(
    store: RunRecordStore,
    *,
    failing: tuple[str, ...] = (),
    ended: Callable[[RunRecord], RunRecord] = _exited,
) -> DetectSeam:
    """A generic command's detect seam that leaves a real detect record, red when *failing* names tests."""

    async def detect() -> tuple[int, str]:
        lane_pass = store.current_pass(COMMAND)
        assert lane_pass is not None
        record = ended(_run('detect', lane_pass.head, store.now(), failing=failing))
        store.write_run(lane_pass, record)
        return record.returncode, ''

    return detect


def _probe(log: list[str], answer: ProbeResult) -> ProbeSeam:
    async def probe(command: LaneCommand, worktree: Path) -> ProbeResult:
        log.append('probe')
        return answer

    return probe


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
        probe_runner: ProbeSeam = run_probe,
        stopping: Callable[[], bool] = never_stopping,
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
            probe_runner=probe_runner,
            stopping=stopping,
        )

    def observed(self) -> PassObservation:
        (observation,) = self.engine.observations
        return observation

    def stored(self, command: str = COMMAND) -> PassRecord:
        lane_pass = self.store.current_pass(command)
        assert lane_pass is not None
        record = self.store.load_pass(command, lane_pass.pass_id)
        assert record is not None
        return record

    def pass_blocks(self, command: str = COMMAND) -> tuple[object, object]:
        record = self.stored(command)
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


@pytest.mark.asyncio
async def test_a_detect_the_lanes_own_stop_killed_leaves_its_pass_with_no_effect(tmp_path, caplog):
    lane = Pass(tmp_path)
    confirm_seam = FakeConfirm(lane.store, lane.log)
    hook = RecordingHook(lane.log)

    with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane_pass'):
        green = await lane.run(
            _detect(lane.store, failing=(A,), ended=_sigtermed),
            confirm=confirm_seam, hooks=(hook,), stopping=lambda: True,
        )

    killed = lane.store.last_record(COMMAND, Phase.DETECT)
    assert killed is not None and (killed.ended_by, killed.returncode) == ('signal', -15)
    assert green is False
    assert lane.log == []
    assert (confirm_seam.calls, hook.calls, lane.engine.observations) == ([], [], [])
    assert (lane.stored().detect, lane.stored().confirm) == (killed, None)
    assert lane.pass_blocks() == (None, None)
    assert any('while the lane is stopping' in r.getMessage() for r in caplog.records), caplog.text


@pytest.mark.asyncio
async def test_a_confirm_the_lanes_own_stop_killed_leaves_its_pass_with_no_effect(tmp_path):
    lane = Pass(tmp_path)
    confirm_seam = FakeConfirm(lane.store, lane.log, outcome=SIGTERM_ENDED, ended=_sigtermed)
    hook = RecordingHook(lane.log)

    green = await lane.run(
        _detect(lane.store, failing=(A,)), confirm=confirm_seam, hooks=(hook,), stopping=lambda: True,
    )

    killed = lane.store.last_record(COMMAND, Phase.CONFIRM)
    assert killed is not None and killed.ended_by == 'signal'
    assert green is False
    assert len(confirm_seam.calls) == 1
    assert lane.log == ['confirm']
    assert (hook.calls, lane.engine.observations) == ([], [])
    assert lane.stored().confirm == killed
    assert lane.pass_blocks() == (None, None)


@pytest.mark.asyncio
async def test_an_outage_probe_the_lanes_own_stop_killed_leaves_its_pass_with_no_effect(tmp_path, caplog):
    lane = Pass(tmp_path)
    confirm_seam = FakeConfirm(lane.store, lane.log)
    hook = RecordingHook(lane.log)

    with caplog.at_level(logging.INFO, logger='orchestrator.offline_lane_pass'):
        green = await lane.run(
            _detect(lane.store, failing=(A,)), target=PROBED, confirm=confirm_seam, hooks=(hook,),
            probe_runner=_probe(lane.log, SIGTERMED_PROBE), stopping=lambda: True,
        )

    detect = lane.stored().detect
    assert green is False
    assert lane.log == ['probe']
    assert (confirm_seam.calls, hook.calls, lane.engine.observations) == ([], [], [])
    assert detect is not None and detect.probe == SIGTERMED_PROBE
    assert lane.stored().confirm is None
    assert lane.pass_blocks() == (None, None)
    assert any('its outage probe' in r.getMessage() for r in caplog.records), caplog.text


@pytest.mark.asyncio
async def test_a_signal_ended_outage_probe_outside_a_stop_is_still_broken(tmp_path):
    lane = Pass(tmp_path)
    hook = RecordingHook(lane.log)

    await lane.run(
        _detect(lane.store, failing=(A,)), target=PROBED, hooks=(hook,),
        probe_runner=_probe(lane.log, SIGTERMED_PROBE), stopping=lambda: False,
    )

    assert lane.log == ['probe', 'confirm', 'hook', 'observe']
    assert lane.observed().probe == SIGTERMED_PROBE
    assert lane.pass_blocks() == (outcome_to_json(REPRODUCED), EFFECT.to_json())


@pytest.mark.asyncio
async def test_a_signal_ended_confirm_outside_a_stop_is_still_unconfirmable(tmp_path):
    lane = Pass(tmp_path)
    confirm_seam = FakeConfirm(lane.store, lane.log, outcome=SIGTERM_ENDED, ended=_sigtermed)
    hook = RecordingHook(lane.log)

    await lane.run(_detect(lane.store, failing=(A,)), confirm=confirm_seam, hooks=(hook,), stopping=lambda: False)

    assert lane.log == ['confirm', 'hook', 'observe']
    assert lane.observed().outcome == SIGTERM_ENDED
    assert lane.pass_blocks() == (outcome_to_json(SIGTERM_ENDED), EFFECT.to_json())


@pytest.mark.asyncio
async def test_the_stop_alone_never_drops_an_exit_ended_pass(tmp_path):
    lane = Pass(tmp_path)
    hook = RecordingHook(lane.log)

    green = await lane.run(_detect(lane.store, failing=(A,)), hooks=(hook,), stopping=lambda: True)

    assert green is False
    assert lane.log == ['confirm', 'hook', 'observe']
    assert lane.observed().outcome == REPRODUCED
    assert lane.pass_blocks() == (outcome_to_json(REPRODUCED), EFFECT.to_json())


def test_production_registers_the_verify_cache_audit_hook(tmp_path):
    (hook,) = lane_pass_outcome_hooks(OrchestratorConfig(project_root=tmp_path), None)

    assert isinstance(hook, VerifyCacheAuditHook)
    assert asyncio.run(hook(COMMAND, HEAD1, REPRODUCED, ())) == ()
    assert not registry.registry_db_path(tmp_path).exists()


AUDIT_LEG = LaneLeg.generic(LaneCommand(
    name=AUDIT_COMMAND, command='orchestrator verify-cache audit-run --config cfg.yaml', fix_task_priority='high',
))
ALPHA_NODE = str(AuditNode('alpha', 'tests/test_a.py::t1'))
ALPHA_KEY = keys.UnitKey.of('module:alpha:test', UnitTier.MODULE, 's', (('spec', 's'), ('tree:alpha', 'a' * 40)))
BETA_NODE = str(AuditNode('beta', 'tests/test_b.py::t1'))
BETA_KEY = keys.UnitKey.of('module:beta:test', UnitTier.MODULE, 's', (('spec', 's'), ('tree:beta', 'b' * 40)))


def _audit_record(store: RunRecordStore, phase: str) -> RunRecord:
    lane_pass = store.current_pass(AUDIT_COMMAND)
    assert lane_pass is not None
    record = dataclasses.replace(_run(phase, lane_pass.head, store.now(), failing=(ALPHA_NODE,)), command=AUDIT_COMMAND)
    store.write_run(lane_pass, record)
    return record


def _audit_detect(store: RunRecordStore) -> DetectSeam:
    async def detect() -> tuple[int, str]:
        return _audit_record(store, 'detect').returncode, ''

    return detect


def _audit_confirm(store: RunRecordStore) -> ConfirmSeam:
    async def confirm(target: ConfirmTarget, detect: RunRecord | None, runner: ConfirmRunner) -> Confirmed:
        _audit_record(store, 'confirm')
        return Confirmed(ConfirmMode.INTERSECTED, (ALPHA_NODE,), (), ())

    return confirm


class _Tee:
    """The real engine, with every observation it was shown kept for the test."""

    def __init__(self, engine: PassObserver) -> None:
        self.engine = engine
        self.observations: list[PassObservation] = []

    async def observe(self, observation: PassObservation) -> PassEffect:
        self.observations.append(observation)
        return await self.engine.observe(observation)


def _seed_leaks(project_root: Path, *unit_keys: keys.UnitKey) -> tuple[registry.Registry, tuple[registry.Entry, ...]]:
    """The project's registry with a green for each of *unit_keys*, every one consumed by HEAD1's landing."""
    store = registry.Registry(registry.registry_db_path(project_root))
    entries = tuple(
        asyncio.run(store.record_green(key, evidence_event_id=7, run_id='r1', sha='abc')) for key in unit_keys
    )
    asyncio.run(store.consume(HEAD1, entries))
    return store, entries


def test_a_verify_cache_leak_note_reaches_the_task_the_lane_files(tmp_path):
    store, (entry,) = _seed_leaks(tmp_path, ALPHA_KEY)
    config = OrchestratorConfig(project_root=tmp_path)
    hooks = lane_pass_outcome_hooks(config, EscalationQueue(tmp_path / 'escalations'))
    lane = Pass(tmp_path)
    client = FakeLedgerClient(lane.log)
    engine = _Tee(LaneEngine(config, FakeAncestry(), task_client=client, escalation_queue=None, now=lane.clock))

    async def one_pass() -> None:
        await lane.run(
            _audit_detect(lane.store), target=AUDIT_LEG, confirm=_audit_confirm(lane.store), hooks=hooks, engine=engine,
        )

    asyncio.run(one_pass())

    (note,) = engine.observations[-1].notes
    assert (note.source, note.node) == ('verify-cache-audit', ALPHA_NODE)
    assert not asyncio.run(store.view([ALPHA_KEY])).entries
    assert asyncio.run(store.policy_state()).streak == 1
    assert client.submitted == []

    lane.clock.advance(minutes=30)
    asyncio.run(one_pass())

    (filed,) = client.submitted
    for named in ('module:alpha:test', ALPHA_KEY.key, f'entry {entry.entry_id}', HEAD1):
        assert named in filed['details'], filed['details']
    assert asyncio.run(store.policy_state()).streak == 1


def test_without_a_wired_queue_the_audit_files_its_kill_switch_blocker_on_the_projects_queue(tmp_path):
    _seed_leaks(tmp_path, ALPHA_KEY, BETA_KEY)
    (hook,) = lane_pass_outcome_hooks(OrchestratorConfig(project_root=tmp_path), None)
    assert isinstance(hook, VerifyCacheAuditHook)

    asyncio.run(hook(AUDIT_COMMAND, HEAD1, Confirmed(ConfirmMode.INTERSECTED, (ALPHA_NODE, BETA_NODE), (), ()), ()))

    (blocker,) = EscalationQueue(escalation_queue_dir(tmp_path)).get_by_task(
        AUDIT_ESCALATION_SENTINEL, status='pending',
    )
    assert blocker.level == 2


def test_a_wired_queue_is_the_one_the_audit_files_its_kill_switch_blocker_on(tmp_path):
    _seed_leaks(tmp_path, ALPHA_KEY, BETA_KEY)
    wired = EscalationQueue(tmp_path / 'other-escalations')
    (hook,) = lane_pass_outcome_hooks(OrchestratorConfig(project_root=tmp_path), wired)
    assert isinstance(hook, VerifyCacheAuditHook)

    asyncio.run(hook(AUDIT_COMMAND, HEAD1, Confirmed(ConfirmMode.INTERSECTED, (ALPHA_NODE, BETA_NODE), (), ()), ()))

    (blocker,) = wired.get_by_task(AUDIT_ESCALATION_SENTINEL, status='pending')
    assert blocker.level == 2
    assert EscalationQueue(escalation_queue_dir(tmp_path)).get_by_task(AUDIT_ESCALATION_SENTINEL) == []


@pytest.mark.asyncio
async def test_an_unreadable_registry_costs_the_audit_pass_only_its_notes(tmp_path, caplog):
    db_path = registry.registry_db_path(tmp_path)
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b'this is not a sqlite database' * 64)
    lane = Pass(tmp_path)
    hooks = lane_pass_outcome_hooks(OrchestratorConfig(project_root=tmp_path), None)

    with caplog.at_level(logging.ERROR, logger='orchestrator.offline_lane_pass'):
        await lane.run(_audit_detect(lane.store), target=AUDIT_LEG, confirm=_audit_confirm(lane.store), hooks=hooks)

    assert lane.observed().notes == ()
    assert any('pass-outcome hook' in record.getMessage() for record in caplog.records), caplog.text


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
