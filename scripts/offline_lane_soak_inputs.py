"""Everything the offline-lane soak report reads, read through each store's owner, as one value (task 6867).

The stores are the lane's run records, its episode records, the flake ledger, the
escalation queue and the task store (PRD ``plans/offline-lane-red-path-prd.md`` §9 ζ).
:func:`load_inputs` reads them into one frozen :class:`SoakInputs`, or says why there is
nothing to judge (:class:`Unjudgeable`).  It never writes: several owner readers
provision what they open, so each store's existence is tested before its reader runs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import _checkout as _checkout
from orchestrator.flake_ledger import ledger_db_path, read_call_sites_written
from orchestrator.guard_state import guard_path
from orchestrator.offline_lane_confirm import PassOutcome, outcome_from_json
from orchestrator.offline_lane_episode_state import (
    EPISODES_FILE,
    CommandRecord,
    EpisodeStore,
    PassEffect,
)
from orchestrator.offline_lane_records import (
    Phase,
    RunRecordStore,
    StoredPass,
    command_key,
    records_root,
)
from shared.project_layout import escalation_queue_dir

_BEGINNING = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class Unjudgeable:
    reason: str


class TaskReader(Protocol):
    """``flake_ledger.FlakeLedgerTaskClient.get_task``'s convention: ``(task, None)`` read,
    ``(None, None)`` a corroborated absence, ``(None, error)`` a failed read."""

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]: ...


@dataclass(frozen=True)
class StoreLocations:
    records: Path
    ledger: Path
    escalations: Path
    episodes: Path

    @classmethod
    def of(cls, project_root: Path) -> StoreLocations:
        episodes = guard_path(project_root, EPISODES_FILE)
        if episodes is None:
            raise ValueError(f'{project_root!r} is not a project root')
        return cls(
            records=records_root(project_root),
            ledger=ledger_db_path(project_root),
            escalations=escalation_queue_dir(project_root),
            episodes=episodes,
        )


@dataclass(frozen=True)
class WindowPass:
    """One stored pass in the window, its blocks parsed by their owners' readers.

    *problems* holds why the record or a block did not parse; *missing_outputs* the
    phases whose finished sub-run's output file does not exist.
    """

    stored: StoredPass
    outcome: PassOutcome | None
    effect: PassEffect | None
    problems: tuple[str, ...]
    missing_outputs: tuple[str, ...]

    @property
    def pass_id(self) -> str:
        return self.stored.pass_id

    @property
    def started_at(self) -> datetime:
        return self.stored.started_at


@dataclass(frozen=True)
class SoakInputs:
    since: datetime
    now: datetime
    commands: tuple[str, ...]
    passes: tuple[WindowPass, ...]
    episodes: Mapping[str, CommandRecord]


def _outcome(stored: StoredPass) -> tuple[PassOutcome | None, str | None]:
    block = None if stored.record is None else stored.record.outcome
    if block is None:
        return None, None
    if not isinstance(block, Mapping):
        return None, f'outcome block is not an object: {type(block).__name__}'
    try:
        return outcome_from_json(block), None
    except ValueError as exc:
        return None, f'outcome block: {exc}'


def _effect(stored: StoredPass) -> tuple[PassEffect | None, str | None]:
    block = None if stored.record is None else stored.record.effect
    if block is None:
        return None, None
    try:
        return PassEffect.from_json(block), None
    except ValueError as exc:
        return None, f'effect block: {exc}'


def _missing_outputs(stored: StoredPass) -> tuple[str, ...]:
    record = stored.record
    if record is None:
        return ()
    runs = ((Phase.DETECT, record.detect), (Phase.CONFIRM, record.confirm))
    return tuple(phase.value for phase, run in runs if run is not None and not Path(run.output_path).exists())


def _window_pass(stored: StoredPass) -> WindowPass:
    outcome, outcome_problem = _outcome(stored)
    effect, effect_problem = _effect(stored)
    problems = tuple(p for p in (stored.problem, outcome_problem, effect_problem) if p is not None)
    return WindowPass(stored, outcome, effect, problems, _missing_outputs(stored))


def _missing_store(stores: StoreLocations) -> Unjudgeable | None:
    if not stores.records.is_dir():
        return Unjudgeable(f'run records missing: {stores.records}')
    if not stores.ledger.is_file():
        return Unjudgeable(f'flake ledger missing: {stores.ledger}')
    if read_call_sites_written(stores.ledger) is None:
        return Unjudgeable(f'flake ledger unreadable: {stores.ledger}')
    if not stores.escalations.is_dir():
        return Unjudgeable(f'escalation queue missing: {stores.escalations}')
    if not stores.episodes.is_file():
        return Unjudgeable(f'episode store missing: {stores.episodes}')
    return None


def _cut_over(every_pass: Sequence[StoredPass]) -> datetime | None:
    """The earliest pass-id start of a pass carrying an effect block, over every command."""
    return min(
        (p.started_at for p in every_pass if p.record is not None and p.record.effect is not None),
        default=None,
    )


def _episodes(project_root: Path, now: datetime, commands: Sequence[str]) -> dict[str, CommandRecord]:
    store = EpisodeStore(project_root, clock=lambda: now)
    return {name: store.load(name) for name in store.commands() if command_key(name) in commands}


async def load_inputs(
    project_root: Path,
    *,
    since: datetime | None,
    commands: Sequence[str],
    now: datetime,
    task_reader: TaskReader,
) -> SoakInputs | Unjudgeable:
    """The soak inputs since *since* (default: the cut-over) for *commands* (names; default every command with passes)."""
    stores = StoreLocations.of(project_root)
    missing = _missing_store(stores)
    if missing is not None:
        return missing
    record_store = RunRecordStore(project_root)
    try:
        every_pass = list(record_store.stored_passes_since(_BEGINNING))
    except OSError as exc:
        return Unjudgeable(f'run records unreadable: {stores.records}: {exc}')
    if since is None:
        since = _cut_over(every_pass)
        if since is None:
            return Unjudgeable('no pass record carries an effect block, so the cut-over is unknown; pass --since')
    in_window = [p for p in every_pass if p.started_at >= since]
    selected = {command_key(name) for name in commands} or {p.command_key for p in in_window}
    window = tuple(_window_pass(p) for p in in_window if p.command_key in selected)
    if not window:
        return Unjudgeable(f'no pass record since {since.isoformat()}')
    return SoakInputs(
        since=since,
        now=now,
        commands=tuple(sorted(selected)),
        passes=window,
        episodes=_episodes(project_root, now, tuple(selected)),
    )
