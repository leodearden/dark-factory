"""Everything the offline-lane soak report reads, read through each store's owner, as one value (task 6867).

The stores are the lane's run records, its episode records, the flake ledger, the
escalation queue and the task store (PRD ``plans/offline-lane-red-path-prd.md`` §9 ζ).
:func:`load_inputs` reads them into one frozen :class:`SoakInputs`, or says why there is
nothing to judge (:class:`Unjudgeable`).  It never writes: several owner readers
provision what they open, so each store's existence is tested before its reader runs.
"""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import _checkout as _checkout
import httpx
from escalation.models import Escalation
from escalation.queue import EscalationQueue, iter_all_escalation_paths, read_escalation_for_scan
from migrate_metadata_modules_to_files import FusedMemoryClient, write_failure_reason
from orchestrator.flake_debt_drain import REGRESSED_AFTER_RESOLUTION, call_site_evidence
from orchestrator.flake_ledger import (
    DUE_ITEM_STALE_AFTER,
    DebtRow,
    DueRow,
    DueSummary,
    FlakeCallSite,
    FlakeOccurrenceRow,
    ledger_db_path,
    read_call_sites_written,
    read_debt_many,
    read_due_items,
    read_occurrences,
    summarize_due,
)
from orchestrator.guard_state import guard_path
from orchestrator.offline_lane_confirm import Outage, PassOutcome, outcome_from_json
from orchestrator.offline_lane_episode_state import (
    EPISODES_FILE,
    CommandRecord,
    EpisodeStore,
    PassEffect,
)
from orchestrator.offline_lane_escalations import escalation_state
from orchestrator.offline_lane_records import (
    FailureRecord,
    Phase,
    RunRecordStore,
    StoredPass,
    command_key,
    records_root,
)
from shared.project_layout import escalation_queue_dir

_BEGINNING = datetime.min.replace(tzinfo=UTC)
_LATEST = datetime.max.replace(tzinfo=UTC)
_LANE_REENTRY_REF = call_site_evidence(FlakeCallSite.offline_lane, observed_at='')['ref']
_SCAN_PARSE_ERRORS = (KeyError, TypeError, ValueError)


@dataclass(frozen=True)
class Unjudgeable:
    reason: str


class TaskReader(Protocol):
    """``flake_ledger.FlakeLedgerTaskClient.get_task``'s convention: ``(task, None)`` read,
    ``(None, None)`` a corroborated absence, ``(None, error)`` a failed read."""

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]: ...


class ToolClient(Protocol):
    """The slice of ``migrate_metadata_modules_to_files.FusedMemoryClient`` a task read uses."""

    async def call_tool(self, name: str, arguments: dict, *, timeout: float | None = None) -> dict: ...


_TASK_NOT_FOUND = 'TaskNotFoundError'


class FusedMemoryTaskReader:
    """:class:`TaskReader` over the fused-memory ``get_task`` tool.

    A ``TaskNotFoundError`` reply is a corroborated absence.  Any other reply that
    ``write_failure_reason`` reads as a failure, and any exception the call raises,
    is a failed read: the reader never raises.
    """

    def __init__(self, client: ToolClient, project_root: str) -> None:
        self._client = client
        self._project_root = project_root

    async def get_task(self, task_id: str) -> tuple[dict | None, Exception | None]:
        try:
            reply: Any = await self._client.call_tool('get_task', {'id': task_id, 'project_root': self._project_root})
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            return None, exc
        if isinstance(reply, dict) and reply.get('error') and reply.get('error_type') == _TASK_NOT_FOUND:
            return None, None
        problem = write_failure_reason(reply)
        if problem is not None:
            return None, RuntimeError(f'get_task {task_id}: {problem}')
        return reply, None


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
class LedgerReads:
    """The flake ledger as the window sees it: occurrence rows since its start, the debt rows of the
    tests the windowed passes offered, and the whole due list."""

    occurrences: tuple[FlakeOccurrenceRow, ...]
    debts: Mapping[str, DebtRow]
    due_summary: DueSummary
    due_items: tuple[DueRow, ...]


@dataclass(frozen=True)
class EscalationReads:
    """The escalation queue as the window sees it.

    *stuck_filing_states* maps each command's stuck-filing record id to its status, and
    *outages* each record id an outage pass raised to that record, None when the queue holds
    no such record.  *regressions* are the offline_lane re-entries of a resolved flake debt
    filed in the window; *unread* the queue files the scan could not read, with the reason.
    """

    stuck_filing_states: Mapping[str, str | None]
    outages: Mapping[str, Escalation | None]
    regressions: tuple[Escalation, ...]
    unread: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SoakInputs:
    """*tasks* maps each task read to its body, or None for a corroborated absence; *failures* maps
    (command, test) to that test's failure records over every stored pass of the selected commands;
    *record_bytes* is the size of every file under the run-record root."""

    since: datetime
    now: datetime
    commands: tuple[str, ...]
    passes: tuple[WindowPass, ...]
    episodes: Mapping[str, CommandRecord]
    tasks: Mapping[str, Mapping[str, Any] | None]
    ledger: LedgerReads
    failures: Mapping[tuple[str, str], tuple[FailureRecord, ...]]
    escalations: EscalationReads
    record_bytes: int

    def effects(self) -> Iterator[tuple[WindowPass, PassEffect]]:
        """Each windowed pass that carries a readable effect block, with that block."""
        for window_pass in self.passes:
            if window_pass.effect is not None:
                yield window_pass, window_pass.effect


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


def _failures(every_pass: Sequence[StoredPass], selected: Collection[str]) -> dict[tuple[str, str], tuple[FailureRecord, ...]]:
    found: dict[tuple[str, str], list[FailureRecord]] = defaultdict(list)
    for stored in every_pass:
        record = stored.record
        if record is None or stored.command_key not in selected:
            continue
        for run in (record.detect, record.confirm):
            for failure in () if run is None else run.failures:
                found[(record.command, failure.node)].append(failure)
    return {key: tuple(failures) for key, failures in found.items()}


def _read_ledger(
    db: Path, *, since: datetime, now: datetime, window: Sequence[WindowPass],
) -> LedgerReads | Unjudgeable:
    offered = dict.fromkeys(debt.node for p in window if p.effect is not None for debt in p.effect.debts_offered)
    summary = summarize_due(db, stale_before=now - DUE_ITEM_STALE_AFTER)
    due_items = None if summary is None else read_due_items(db, now=_LATEST, limit=summary.size)
    if summary is None or due_items is None:
        return Unjudgeable(f'flake ledger unreadable: {db}')
    return LedgerReads(
        occurrences=tuple(read_occurrences(db, since=since.isoformat())),
        debts=read_debt_many(db, offered),
        due_summary=summary,
        due_items=tuple(due_items),
    )


def episode_bindings(episodes: Mapping[str, CommandRecord]) -> dict[tuple[str, str], str]:
    """(command name, node) to the task each stored episode, live or ended, is bound to now."""
    return {
        (record.command, node): episode.covering_task_id
        for record in episodes.values()
        for node, episode in record.episodes.items()
        if episode.covering_task_id is not None
    }


def _task_ids(window: Sequence[WindowPass], episodes: Mapping[str, CommandRecord], ledger: LedgerReads) -> list[str]:
    """The ids the windowed effect blocks filed, the tasks the stored episodes are bound to now,
    and the owners of the offered tests' debts."""
    filed = [task_id for p in window if p.effect is not None for task_id in p.effect.tasks_filed]
    bound = episode_bindings(episodes).values()
    owners = [row.owner_task_id for row in ledger.debts.values() if row.owner_task_id is not None]
    return list(dict.fromkeys((*filed, *bound, *owners)))


def _is_lane_regression(escalation: Escalation, since: datetime) -> bool:
    if escalation.category != REGRESSED_AFTER_RESOLUTION:
        return False
    if not any(entry.get('ref') == _LANE_REENTRY_REF for entry in escalation.evidence):
        return False
    filed_at = datetime.fromisoformat(escalation.timestamp)
    return (filed_at if filed_at.tzinfo else filed_at.replace(tzinfo=UTC)) >= since


def _scan_regressions(queue_dir: Path, since: datetime) -> tuple[list[Escalation], list[tuple[str, str]]]:
    """The offline_lane regression records filed since *since*, and every file the scan could not read.

    A file archived between the listing and its read ('vanished') is routine, not unread.
    """
    found: list[Escalation] = []
    unread: list[tuple[str, str]] = []
    for path in iter_all_escalation_paths(queue_dir):
        escalation, reason = read_escalation_for_scan(
            path, context='offline_lane_soak_inputs', parse_errors=_SCAN_PARSE_ERRORS,
        )
        if escalation is None:
            if reason != 'vanished':
                unread.append((str(path), reason))
            continue
        try:
            if _is_lane_regression(escalation, since):
                found.append(escalation)
        except ValueError as exc:
            unread.append((str(path), f'timestamp: {exc}'))
    return found, unread


def _read_escalations(
    queue_dir: Path, *, since: datetime, episodes: Mapping[str, CommandRecord], window: Sequence[WindowPass],
) -> EscalationReads:
    """*queue_dir* exists, so the queue built over it makes nothing."""
    queue = EscalationQueue(queue_dir)
    stuck = [r.stuck_filing_escalation_id for r in episodes.values() if r.stuck_filing_escalation_id is not None]
    raised = [
        escalation_id
        for p in window
        if isinstance(p.outcome, Outage) and p.effect is not None
        for escalation_id in p.effect.escalations_raised
    ]
    regressions, unread = _scan_regressions(queue_dir, since)
    return EscalationReads(
        stuck_filing_states={escalation_id: escalation_state(queue, escalation_id) for escalation_id in stuck},
        outages={escalation_id: queue.get(escalation_id) for escalation_id in raised},
        regressions=tuple(regressions),
        unread=tuple(unread),
    )


def _directory_bytes(root: Path) -> int:
    """The size of every file under *root*; a file pruned during the walk counts nothing."""
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                total += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total


async def _read_tasks(
    task_ids: Sequence[str], task_reader: TaskReader,
) -> dict[str, Mapping[str, Any] | None] | Unjudgeable:
    """Each task as read, None for a corroborated absence; a failed read leaves nothing to judge."""
    tasks: dict[str, Mapping[str, Any] | None] = {}
    for task_id in task_ids:
        task, error = await task_reader.get_task(task_id)
        if error is not None:
            return Unjudgeable(f'task store unreadable: {task_id}: {error}')
        tasks[task_id] = task
    return tasks


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
    ledger = _read_ledger(stores.ledger, since=since, now=now, window=window)
    if isinstance(ledger, Unjudgeable):
        return ledger
    episodes = _episodes(project_root, now, tuple(selected))
    tasks = await _read_tasks(_task_ids(window, episodes, ledger), task_reader)
    if isinstance(tasks, Unjudgeable):
        return tasks
    return SoakInputs(
        since=since,
        now=now,
        commands=tuple(sorted(selected)),
        passes=window,
        episodes=episodes,
        tasks=tasks,
        ledger=ledger,
        failures=_failures(every_pass, selected),
        escalations=_read_escalations(stores.escalations, since=since, episodes=episodes, window=window),
        record_bytes=_directory_bytes(stores.records),
    )


async def load_inputs_from_server(
    project_root: Path,
    *,
    server_url: str,
    since: datetime | None,
    commands: Sequence[str],
    now: datetime,
) -> SoakInputs | Unjudgeable:
    """:func:`load_inputs` reading tasks from the fused-memory server at *server_url*.

    The reader returns every failed read, so an HTTP error escaping the client's
    block came from opening or closing the client: the task store is unreachable.
    """
    try:
        async with FusedMemoryClient(server_url) as client:
            return await load_inputs(
                project_root, since=since, commands=commands, now=now,
                task_reader=FusedMemoryTaskReader(client, str(project_root)),
            )
    except httpx.HTTPError as exc:
        return Unjudgeable(f'task store unreachable: {server_url}: {exc!r}')
