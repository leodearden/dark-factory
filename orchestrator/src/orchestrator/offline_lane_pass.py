"""One pass of one offline-lane command: detect, confirm a red, ask the outcome hooks, then the engine.

PRD ``plans/offline-lane-red-path-prd.md`` D4, D13 and section 5.7.  The engine's
:meth:`~orchestrator.offline_lane_episodes.LaneEngine.observe` sees every pass, green
included, and is the only point at which a pass has an effect; the pass record then
keeps the outcome and that effect.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmTarget,
    PassOutcome,
    Unconfirmable,
    outcome_to_json,
)
from orchestrator.offline_lane_episode_state import PassEffect, PassObservation
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_records import (
    LEGACY_INFRA_COMMAND,
    LEGACY_NUMERIC_COMMAND,
    Leg,
    Phase,
    RunRecord,
    RunRecordStore,
    classify_exit,
)
from orchestrator.offline_lane_runs import ConfirmRunner, confirm_runner

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.config import LaneCommand, OrchestratorConfig

logger = logging.getLogger(__name__)

LEGACY_FIX_TASK_PRIORITY = 'high'

PassOutcomeHook = Callable[[str, str, PassOutcome, Sequence[RunRecord]], Awaitable[Sequence[EvidenceNote]]]
DetectSeam = Callable[[], Awaitable[tuple[int, str]]]
ConfirmOutcome = Confirmed | Unconfirmable
ConfirmSeam = Callable[[ConfirmTarget, RunRecord | None, ConfirmRunner], Awaitable[ConfirmOutcome]]


def lane_pass_outcome_hooks(
    config: OrchestratorConfig, escalation_queue: EscalationQueue | None,
) -> tuple[PassOutcomeHook, ...]:
    """The production section 5.7 hooks, each asked once per red pass that reached an outcome."""
    return ()


class PassObserver(Protocol):
    async def observe(self, observation: PassObservation) -> PassEffect: ...


@dataclass(frozen=True)
class LaneLeg:
    """A command the lane runs: a generic command carries its config, the two legacy legs none."""

    leg: Leg
    command: str
    lane_command: LaneCommand | None
    priority: str

    @classmethod
    def numeric(cls) -> LaneLeg:
        return cls(Leg.NUMERIC, LEGACY_NUMERIC_COMMAND, None, LEGACY_FIX_TASK_PRIORITY)

    @classmethod
    def infra(cls) -> LaneLeg:
        return cls(Leg.INFRA, LEGACY_INFRA_COMMAND, None, LEGACY_FIX_TASK_PRIORITY)

    @classmethod
    def generic(cls, command: LaneCommand) -> LaneLeg:
        return cls(Leg.GENERIC, command.name, command, command.fix_task_priority)

    @property
    def label(self) -> str:
        return self.command if self.lane_command is not None else self.leg.value


def _log_sub_run(target: LaneLeg, head: str, returncode: int, duration: float) -> None:
    logger.info(
        'offline-lane: %s sub-run head=%s status=%s ended_by=%s duration=%.1fs',
        target.label, head[:12], 'PASS' if returncode == 0 else 'FAIL', classify_exit(returncode), duration,
    )


def _run_records(store: RunRecordStore, command: str) -> tuple[RunRecord, ...]:
    records = (store.last_record(command, Phase.DETECT), store.last_record(command, Phase.CONFIRM))
    return tuple(record for record in records if record is not None)


async def _hook_notes(
    hooks: Sequence[PassOutcomeHook],
    command: str,
    head: str,
    outcome: PassOutcome,
    run_records: Sequence[RunRecord],
) -> tuple[EvidenceNote, ...]:
    notes: list[EvidenceNote] = []
    for hook in hooks:
        try:
            notes.extend(await hook(command, head, outcome, run_records))
        except Exception:
            logger.exception('offline-lane: %s: pass-outcome hook %r raised; its notes are lost', command, hook)
    return tuple(notes)


async def run_pass(
    target: LaneLeg,
    *,
    worktree: Path,
    head: str,
    detect: DetectSeam,
    confirm: ConfirmSeam,
    store: RunRecordStore,
    engine: PassObserver,
    hooks: Sequence[PassOutcomeHook],
) -> bool:
    """Run one pass of *target* at *head*; true when its detection was green.

    A red detection that left no run record (an injected seam) is handed to *confirm*
    with no record, which the lane's confirm answers ``Unconfirmable(NO_DETECT_RECORD)``.
    """
    lane_pass = await asyncio.to_thread(store.allocate_pass, target.command, target.leg, head)
    started = time.monotonic()
    returncode, _tail = await detect()
    _log_sub_run(target, head, returncode, time.monotonic() - started)
    detect_record = store.last_record(target.command, Phase.DETECT)
    outcome: ConfirmOutcome | None = None
    notes: tuple[EvidenceNote, ...] = ()
    if returncode != 0:
        confirm_target = ConfirmTarget(target.leg, worktree, target.lane_command)
        outcome = await confirm(confirm_target, detect_record, confirm_runner(store, lane_pass))
        notes = await _hook_notes(hooks, target.command, head, outcome, _run_records(store, target.command))
    effect = await engine.observe(PassObservation(
        command=target.command,
        leg=target.leg.value,
        head=head,
        pass_id=lane_pass.pass_id,
        started_at=lane_pass.started_at,
        detect=detect_record,
        confirm=store.last_record(target.command, Phase.CONFIRM),
        outcome=outcome,
        fix_task_priority=target.priority,
        notes=notes,
        lane_command=target.lane_command,
    ))
    await asyncio.to_thread(
        store.set_pass_blocks,
        lane_pass,
        outcome=None if outcome is None else outcome_to_json(outcome),
        effect=effect.to_json(),
    )
    return returncode == 0
