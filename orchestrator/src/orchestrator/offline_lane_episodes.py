"""The offline lane's red-path engine: the reads around a pass, the fix-task filing, the escalations.

PRD ``plans/offline-lane-red-path-prd.md`` D7, D8, D10-D12 and sections 5.4 and 5.10.  The records and
the pure effect of a pass live in
``orchestrator/src/orchestrator/offline_lane_episode_state.py``; :class:`LaneEngine`
reads what :func:`apply_pass` needs, saves its result, files the fix tasks it names and
raises the lane's escalation records.  :meth:`LaneEngine.observe` and
:meth:`LaneEngine.tick` are its interface, with one read beside them,
:meth:`LaneEngine.pinned_pass_ids`.

Rules the engine and its pure rules keep beyond the PRD's text:

- R1 a new episode's ``last_green_head`` is the previous pass's head when that pass's
  detect record shows the test passed, else the command's ``last_green_head``.
- R2 the digest reads every counted pass's run record; an unreadable one keeps its id
  with an unknown head, start and mode, and no failure lines.
- R3 the reproduce line is fixed when a filing first falls due and kept on the due
  episodes; ``fix_task_priority`` is kept on the command record for retries.
- R4 a cancelled or absent covering task whose escalation record is closed (resolved
  or dismissed) reads ``CLOSED_RELEASED`` and is treated as ``DONE_CONTAINED``.
- R5 :func:`apply_pass` takes the pass's reads as one :class:`PassReads`.
- R6 a failed filing keeps its reason on the due episodes; due groups are the
  episodes sharing a covering task, else sharing ``filing_due_since``.
- R7 a stale head appends no reproduced head; every live episode expires
  ``red_unlanded_hours`` after its last counted pass; the unlanded escalation fires for
  an open, deferred or cancelled covering task; the command pseudo-test's ``kind`` is
  the unconfirmable reason; with no task client nothing falls due.
- R8 a filing combined into a live task binds it, and commits planning only when that
  task is deferred.
- R9 a filing combined into a terminal task binds it as swallowed, and a swallowed
  binding reads ``CLOSED`` whatever the task's live status.
- R10 :meth:`LaneEngine.pinned_pass_ids` is the one read beside ``observe`` and ``tick``,
  answered from the engine's own store for the run-record prune.
- R11 a due filing with no covering task whose every episode has ended is cancelled,
  not retried; one live episode keeps the whole group's filing going.
- R12 each probe answer, in a pass or on a tick, maintains both probe records: the one
  of the answer's kind is filed, or amended when the answer changed; the other is
  released with the level-2s that hold it.  A record resolved by hand is replaced only
  by a pass.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from shared.task_statuses import OwnerLiveness, owner_liveness
from shared.task_submit import SubmitFailed, SubmitOutcome, SubmittedTask, SubmittedTicket

from orchestrator.flake_ledger import fix_contained, resolving_commit
from orchestrator.offline_lane_confirm import Confirmed, Outage, Unconfirmable, outcome_from_json
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    CoveringVerdict,
    Episode,
    EpisodeStore,
    LanePolicy,
    PassEffect,
    PassObservation,
    PassReads,
    apply_outage,
    apply_pass,
    recheck_due,
    reproduce_for,
    reproduced_nodes,
    unconfirmable_node,
    with_probe,
)
from orchestrator.offline_lane_escalations import (
    PENDING_ESCALATION,
    ProbeReport,
    StuckFiling,
    UnlandedRed,
    amend_outage,
    amend_probe_broken,
    amend_stuck_filing,
    escalation_state,
    file_outage,
    file_probe_broken,
    file_stuck_filing,
    file_unlanded_red,
    release_lane_record,
    resolve_stuck_filing,
)
from orchestrator.offline_lane_filing import (
    UNKNOWN,
    CountedPassEvidence,
    TestEvidence,
    build_lane_command_task_arguments,
    build_lane_fix_task_arguments,
)
from orchestrator.offline_lane_records import (
    PassRecord,
    ProbeKind,
    ProbeResult,
    RunRecord,
    RunRecordStore,
)
from orchestrator.offline_lane_runs import ProbeSeam, command_cwd, run_probe

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.config import LaneCommand, OrchestratorConfig
    from orchestrator.flake_ledger import FlakeLedgerTaskClient

logger = logging.getLogger(__name__)

UNKNOWN_MODE = 'unknown'
UNKNOWN_STATUS = 'unknown'
CLOSED_ESCALATION_STATES = frozenset({'resolved', 'dismissed'})
DEFERRED_STATUS = 'deferred'


def _utc_now() -> datetime:
    return datetime.now(UTC)


def lane_policy(config: OrchestratorConfig) -> LanePolicy:
    """The lane's persistence policy, read from the live ``config.git`` knobs."""
    return LanePolicy(
        persist_passes=config.git.offline_lane_persist_passes,
        recheck_secs=config.git.offline_lane_recheck_secs,
        red_unlanded_hours=config.git.offline_lane_red_unlanded_hours,
    )


class LaneGit(Protocol):
    """What the engine asks of git: one ancestry read, and where the lane's worktree is."""

    async def is_ancestor_checked(self, ancestor: str, descendant: str) -> bool | None: ...

    @property
    def persistent_offline_deep_worktree_path(self) -> Path: ...


@dataclass(frozen=True)
class TickResult:
    commands_due: tuple[str, ...]
    tasks_filed: tuple[str, ...]


@dataclass(frozen=True)
class _TaskRead:
    """One live read of a covering task; ``contained`` is D10's done question, asked only of a done task."""

    task: Mapping[str, Any] | None
    error: str | None
    contained: bool | None = None

    @property
    def status(self) -> str | None:
        return None if self.task is None else self.task.get('status')


def _with_episodes(record: CommandRecord, nodes: Iterable[str], **changes: Any) -> CommandRecord:
    episodes = dict(record.episodes)
    for node in nodes:
        episodes[node] = replace(episodes[node], **changes)
    return replace(record, episodes=episodes)


def _due_groups(record: CommandRecord) -> list[tuple[str, ...]]:
    groups: dict[tuple[str, object], list[str]] = {}
    for node, episode in sorted(record.episodes.items()):
        if episode.filing_due:
            if episode.covering_task_id is not None:
                key: tuple[str, object] = ('task', episode.covering_task_id)
            else:
                key = ('due_since', episode.filing_due_since)
            groups.setdefault(key, []).append(node)
    return [tuple(nodes) for nodes in groups.values()]


def _covering(record: CommandRecord, nodes: Sequence[str]) -> str | None:
    return record.episodes[nodes[0]].covering_task_id


def _persisted_reproduce(record: CommandRecord, nodes: Sequence[str]) -> str:
    reproduce = record.episodes[nodes[0]].reproduce
    if reproduce is None:
        raise ValueError(f'{record.command}: due filing for {list(nodes)} has no reproduce line')
    return reproduce


def _marked_due(record: CommandRecord, nodes: Sequence[str], reproduce: str, now: datetime) -> CommandRecord:
    undue = [node for node in nodes if not record.episodes[node].filing_due]
    return _with_episodes(record, undue, filing_due=True, filing_due_since=now, reproduce=reproduce)


def _after_submit(
    record: CommandRecord, nodes: Sequence[str], outcome: SubmitOutcome, now: datetime,
) -> CommandRecord:
    """γ's arms (PRD 5.4; the ledger's ``_owner_from_submission`` for a combined task)."""
    match outcome:
        case SubmittedTask(task_id=task_id, combined=combined, status=status):
            bound = _with_episodes(record, nodes, covering_task_id=task_id, filed_at=now, filing_failure=None)
            liveness = owner_liveness(status)
            if not combined or liveness is OwnerLiveness.DEFERRED:
                return bound
            return _with_episodes(
                bound, nodes, planning_committed=True, filing_due=False, filing_due_since=None,
                reproduce=None, binding_swallowed=liveness is not OwnerLiveness.OPEN,
            )
        case SubmittedTicket(ticket_id=ticket_id):
            failure = f'submit_task returned ticket {ticket_id!r}, not a task: planning mode did not take effect'
        case SubmitFailed(reason=reason):
            failure = reason
    return _with_episodes(record, nodes, filing_failure=failure)


def _after_commit(record: CommandRecord, nodes: Sequence[str], task_id: str, read: _TaskRead) -> CommandRecord:
    if read.error is not None:
        failure = f'task {task_id} unreadable after commit_planning: {read.error}'
    elif read.task is None:
        failure = f'task {task_id} absent after commit_planning'
    elif read.status == DEFERRED_STATUS:
        failure = f'task {task_id} still deferred after commit_planning'
    else:
        return _with_episodes(record, nodes, planning_committed=True, filing_failure=None)
    return _with_episodes(record, nodes, filing_failure=failure)


def _cleared(record: CommandRecord, nodes: Sequence[str]) -> CommandRecord:
    if not all(record.episodes[node].planning_committed for node in nodes):
        return record
    return _with_episodes(record, nodes, filing_due=False, filing_due_since=None, reproduce=None)


def _outcome_mode(outcome: object) -> str:
    if isinstance(outcome, Confirmed):
        return outcome.mode.value
    if isinstance(outcome, Unconfirmable):
        return 'unconfirmable'
    return UNKNOWN_MODE


def _recorded_mode(record: PassRecord) -> str:
    if not isinstance(record.outcome, Mapping):
        return UNKNOWN_MODE
    try:
        return _outcome_mode(outcome_from_json(record.outcome))
    except ValueError:
        return UNKNOWN_MODE


def _first_line(run: RunRecord | None, node: str, *, any_node: bool) -> str | None:
    if run is None:
        return None
    return next((f.first_line for f in run.failures if any_node or f.node == node), None)


def _counted_pass(
    command: str, episode: Episode, pass_id: str, record: PassRecord | None, record_dir: Path, mode: str | None,
) -> CountedPassEvidence:
    if record is None:
        return CountedPassEvidence(pass_id, UNKNOWN, None, mode or UNKNOWN_MODE, None, None, str(record_dir))
    any_node = episode.node == unconfirmable_node(command)
    return CountedPassEvidence(
        pass_id=pass_id,
        head=record.head,
        started_at=record.started_at,
        mode=mode or _recorded_mode(record),
        detect_first_line=_first_line(record.detect, episode.node, any_node=any_node),
        confirm_first_line=_first_line(record.confirm, episode.node, any_node=any_node),
        record_dir=str(record_dir),
    )


def _abandoned(record: CommandRecord, nodes: Sequence[str]) -> bool:
    episodes = [record.episodes[node] for node in nodes]
    return episodes[0].covering_task_id is None and not any(episode.live for episode in episodes)


def _cancelled(record: CommandRecord, nodes: Sequence[str]) -> CommandRecord:
    return _with_episodes(
        record, nodes, filing_due=False, filing_due_since=None, reproduce=None, filing_failure=None,
    )


def _stuck_reason(stuck: Sequence[Episode]) -> str:
    reasons = dict.fromkeys(episode.filing_failure for episode in stuck if episode.filing_failure)
    return '; '.join(reasons) or 'no filing attempt has completed'


class _ProbeSource(StrEnum):
    """Who ran a probe: a red pass, or the tick's probe loop during an outage."""

    PASS = 'pass'
    TICK = 'tick'


@dataclass(frozen=True)
class _ProbeRecord:
    """One of the two probe records a command keeps: where its id is stored, and how it is written."""

    kind: ProbeKind
    stored: Callable[[CommandRecord], str | None]
    with_id: Callable[[CommandRecord, str | None], CommandRecord]
    file: Callable[[EscalationQueue, ProbeReport], str]
    amend: Callable[[EscalationQueue, str, ProbeReport], bool]


_PROBE_RECORDS = (
    _ProbeRecord(
        ProbeKind.OUTAGE,
        lambda record: record.outage_escalation_id,
        lambda record, escalation_id: replace(record, outage_escalation_id=escalation_id),
        file_outage,
        amend_outage,
    ),
    _ProbeRecord(
        ProbeKind.BROKEN,
        lambda record: record.probe_broken_escalation_id,
        lambda record, escalation_id: replace(record, probe_broken_escalation_id=escalation_id),
        file_probe_broken,
        amend_probe_broken,
    ),
)

_RELEASED_BECAUSE = {
    ProbeKind.UP: 'The outage probe passed; the lane closed its own record.',
    ProbeKind.BROKEN: 'The outage probe is now broken; the lane closed this record and opened a broken-probe record.',
    ProbeKind.OUTAGE: 'The outage probe now names a failed dependency; the lane closed this record and opened an outage record.',
}
_NO_PROBE_DECLARED = 'This command no longer declares an outage probe; the lane closed its own record.'
_NOT_CONFIGURED = 'This command is no longer configured or enabled; the lane closed its own record.'


def _answer(probe: ProbeResult | None) -> tuple[str, int | None, str | None, str] | None:
    """What a probe record states of a probe: everything but its duration."""
    if probe is None:
        return None
    return (probe.result, probe.returncode, probe.failed_dependency, probe.first_line)


def _release_reason(answer: ProbeResult | None) -> str:
    return _NO_PROBE_DECLARED if answer is None else _RELEASED_BECAUSE[ProbeKind(answer.result)]


def _no_effect(escalations_raised: tuple[str, ...]) -> PassEffect:
    return PassEffect(
        tests_counted=(), episodes_started=(), episodes_extended=(), episodes_ended=(),
        threshold_reached=(), stale_heads=(), no_decision=(), tasks_filed=(),
        escalations_raised=escalations_raised,
    )


class LaneEngine:
    """The red path of the offline lane: :meth:`observe` once per pass, :meth:`tick` on the worker's idle poll."""

    def __init__(
        self,
        config: OrchestratorConfig,
        git: LaneGit,
        *,
        task_client: FlakeLedgerTaskClient | None,
        escalation_queue: EscalationQueue | None,
        now: Callable[[], datetime] | None = None,
        probe_runner: ProbeSeam = run_probe,
    ) -> None:
        self._config = config
        self._git = git
        self._tasks = task_client
        self._queue = escalation_queue
        self._now = now or _utc_now
        self._probe_runner = probe_runner
        self._store = EpisodeStore(config.project_root, clock=self._now)
        self._records = RunRecordStore(config.project_root, clock=self._now)

    async def observe(self, observation: PassObservation) -> PassEffect:
        if isinstance(observation.outcome, Outage):
            return await self._observed_outage(observation, observation.outcome.probe)
        policy = lane_policy(self._config)
        record = await asyncio.to_thread(self._store.load, observation.command)
        record, filed = await self._retry_due_filings(record)
        reads, statuses = await self._pass_reads(record, observation)
        applied = apply_pass(record, observation, reads, policy, self._now())
        record = applied.record
        await self._save(record)
        unfinished = {
            task_id for node, verdict in reads.verdicts.items()
            if verdict is CoveringVerdict.DEFERRED_UNFINISHED
            and (task_id := record.episodes[node].covering_task_id) is not None
        }
        groups = [*applied.to_file, *(g for g in _due_groups(record) if _covering(record, g) in unfinished)]
        record, opened = await self._open_filings(record, groups, observation)
        saved = record
        record, raised = await self._raise_unlanded(record, applied.unlanded, statuses)
        record = await self._maintained_stuck_record(record, policy)
        probed: tuple[str, ...] = ()
        if observation.probe is not None:
            record, probed = await self._probe_answered(record, observation, observation.probe)
        if record != saved:
            await self._save(record)
        return replace(applied.effect, tasks_filed=(*filed, *opened), escalations_raised=(*raised, *probed))

    async def _observed_outage(self, observation: PassObservation, probe: ProbeResult) -> PassEffect:
        """D5's outage row: held episodes are marked, the probe records kept, and nothing is filed."""
        record = await asyncio.to_thread(self._store.load, observation.command)
        record = apply_outage(record, observation)
        record, raised = await self._probe_answered(record, observation, probe)
        await self._save(record)
        return _no_effect(raised)

    async def _probe_answered(
        self, record: CommandRecord, observation: PassObservation, probe: ProbeResult,
    ) -> tuple[CommandRecord, tuple[str, ...]]:
        cmd = observation.lane_command
        if cmd is None:
            raise ValueError(f'{observation.command}: a probe answered a pass with no lane command')
        record_dir = str(self._records.pass_directory(observation.command, observation.pass_id))
        return await self._probe_records_kept(record, cmd, probe, _ProbeSource.PASS, record_dir)

    async def tick(self, *, head: str) -> TickResult:
        policy = lane_policy(self._config)
        due: list[str] = []
        filed: list[str] = []
        for command in await asyncio.to_thread(self._store.commands):
            record = await asyncio.to_thread(self._store.load, command)
            record, tasks = await self._retry_due_filings(record)
            filed.extend(tasks)
            saved = record
            record = await self._maintained_stuck_record(record, policy)
            record, probe_due = await self._probe_loop(record, self._lane_command(command), policy)
            rechecked = None if probe_due else recheck_due(record, head, policy, self._now())
            if rechecked is not None:
                record = rechecked
            if probe_due or rechecked is not None:
                due.append(command)
            if record != saved:
                await self._save(record)
        return TickResult(tuple(due), tuple(filed))

    def _lane_command(self, command: str) -> LaneCommand | None:
        """*command*'s configuration as of now, read on every tick so a reload takes effect."""
        return next((cmd for cmd in self._config.git.offline_lane_commands if cmd.name == command), None)

    async def _probe_loop(
        self, record: CommandRecord, cmd: LaneCommand | None, policy: LanePolicy,
    ) -> tuple[CommandRecord, bool]:
        """D7's probe loop and its two hold exits; true when the command is due a pass."""
        probe_ids = [kind.stored(record) for kind in _PROBE_RECORDS]
        if cmd is None or not cmd.enabled or not cmd.outage_probe:
            if not record.in_outage and not any(probe_ids):
                return record, False
            reason = _NO_PROBE_DECLARED if cmd is not None and cmd.enabled else _NOT_CONFIGURED
            record = replace(record, last_probe_result=None)
            for kind in _PROBE_RECORDS:
                record, _ = await self._released(record, kind, reason)
            return record, False
        if not record.in_outage:
            return await self._stale_records_released(record), False
        now = self._now()
        if record.last_probe_at is not None and now - record.last_probe_at < policy.recheck:
            return record, False
        worktree = self._git.persistent_offline_deep_worktree_path
        if not await asyncio.to_thread(command_cwd(worktree, cmd).is_dir):
            return record, True
        answer = await self._probe_runner(cmd, worktree)
        record, _ = await self._probe_records_kept(record, cmd, answer, _ProbeSource.TICK, None)
        return record, answer.result != ProbeKind.OUTAGE

    async def _stale_records_released(self, record: CommandRecord) -> CommandRecord:
        """The record with every stored probe record that no longer matches the last answer released.

        The flag follows the releases: it stays set while any of them fails, and clears once all succeed.
        """
        answer = record.last_probe_result
        stale = [
            kind for kind in _PROBE_RECORDS
            if kind.stored(record) is not None and (answer is None or kind.kind != answer.result)
        ]
        if not stale:
            return record
        behind = False
        for kind in stale:
            record, failed = await self._released(record, kind, _release_reason(answer))
            behind = behind or failed
        return replace(record, probe_record_behind=behind)

    async def _probe_records_kept(
        self,
        record: CommandRecord,
        cmd: LaneCommand,
        answer: ProbeResult,
        source: _ProbeSource,
        record_dir: str | None,
    ) -> tuple[CommandRecord, tuple[str, ...]]:
        """R12: the record after *answer*, with both probe records maintained; the ids this call filed."""
        previous = record.last_probe_result
        record = with_probe(record, answer, self._now())
        if self._queue is None:
            logger.error('offline-lane: %s: no escalation queue; %s probe record not written', record.command, answer.result)
            return record, ()
        report = ProbeReport(
            command=record.command,
            probe_command=cmd.outage_probe or '',
            cwd=str(command_cwd(self._git.persistent_offline_deep_worktree_path, cmd)),
            timeout_secs=cmd.outage_probe_timeout_secs,
            probe=answer,
            record_dir=record_dir,
        )
        changed = _answer(answer) != _answer(previous) or record.probe_record_behind
        record = await self._stale_records_released(replace(record, probe_record_behind=False))
        filed: list[str] = []
        behind = record.probe_record_behind
        for kind in _PROBE_RECORDS:
            if kind.kind == answer.result:
                record, opened, failed = await self._kept(self._queue, record, kind, report, source, changed)
                filed.extend([opened] if opened else [])
                behind = behind or failed
        return replace(record, probe_record_behind=behind), tuple(filed)

    async def _kept(
        self,
        queue: EscalationQueue,
        record: CommandRecord,
        kind: _ProbeRecord,
        report: ProbeReport,
        source: _ProbeSource,
        changed: bool,
    ) -> tuple[CommandRecord, str | None, bool]:
        """The record of the answer's kind: amended while pending, else filed (by a pass, or when none is stored)."""
        stored = kind.stored(record)
        try:
            if stored is not None:
                pending = await asyncio.to_thread(escalation_state, queue, stored) == PENDING_ESCALATION
                if pending and (not changed or await asyncio.to_thread(kind.amend, queue, stored, report)):
                    return record, None, False
                if source is _ProbeSource.TICK:
                    return record, None, False
            opened = await asyncio.to_thread(kind.file, queue, report)
        except Exception:
            logger.exception('offline-lane: %s: %s probe record not written; retried at the next probe',
                             record.command, kind.kind)
            return record, None, True
        return kind.with_id(record, opened), opened, False

    async def _released(self, record: CommandRecord, kind: _ProbeRecord, reason: str) -> tuple[CommandRecord, bool]:
        """The record with *kind*'s stored record released and its id cleared; true when the release failed."""
        stored = kind.stored(record)
        if stored is None or self._queue is None:
            return record, False
        try:
            await asyncio.to_thread(release_lane_record, self._queue, stored, reason)
        except Exception:
            logger.exception('offline-lane: %s: %s probe record not written; retried at the next probe',
                             record.command, kind.kind)
            return record, True
        return kind.with_id(record, None), False

    def pinned_pass_ids(self) -> frozenset[str]:
        """The counted pass ids the held episodes refer to; synchronous, for the run-record prune off the loop."""
        return self._store.pinned_pass_ids()

    async def _save(self, record: CommandRecord) -> None:
        await asyncio.to_thread(self._store.save, record)

    async def _saved(self, before: CommandRecord, after: CommandRecord) -> CommandRecord:
        if after != before:
            await self._save(after)
        return after

    async def _pass_reads(
        self, record: CommandRecord, observation: PassObservation,
    ) -> tuple[PassReads, dict[str, str]]:
        """D10's reads for the red-again bound tests, the batched refresh read, and the previous pass's detect record."""
        terminal = await self._terminal_task_ids(record)
        reproduced = reproduced_nodes(observation)
        bound = {
            node: episode for node in reproduced
            if (episode := record.episodes.get(node)) is not None and episode.covering_task_id is not None
        }
        task_ids = sorted({episode.covering_task_id for episode in bound.values() if episode.covering_task_id})
        task_reads = {task_id: await self._read_covering_task(task_id, observation.head) for task_id in task_ids}
        verdicts = {
            node: await self._covering_verdict(record.command, episode, task_reads[episode.covering_task_id])
            for node, episode in bound.items()
            if episode.covering_task_id is not None
        }
        statuses = {task_id: read.status or UNKNOWN_STATUS for task_id, read in task_reads.items()}
        starts = any(node not in record.episodes or not record.episodes[node].live for node in reproduced)
        previous = await self._previous_detect(record) if starts else None
        return PassReads(verdicts, terminal, previous), statuses

    async def _terminal_task_ids(self, record: CommandRecord) -> frozenset[str]:
        bound = sorted({e.covering_task_id for e in record.episodes.values() if e.covering_task_id is not None})
        if not bound or self._tasks is None:
            return frozenset()
        try:
            statuses, error = await self._tasks.get_statuses(bound)
        except Exception as exc:
            statuses, error = {}, exc
        if error is not None:
            logger.error(
                'offline-lane: %s: covering task statuses unreadable, every binding refreshed: %s',
                record.command, error,
            )
            return frozenset()
        closed = {OwnerLiveness.DONE, OwnerLiveness.CLOSED}
        return frozenset(t for t in bound if t in statuses and owner_liveness(statuses[t]) in closed)

    async def _read_task(self, task_id: str) -> _TaskRead:
        if self._tasks is None:
            return _TaskRead(None, 'no task client')
        try:
            task, error = await self._tasks.get_task(task_id)
        except Exception as exc:
            task, error = None, exc
        if error is not None:
            return _TaskRead(None, str(error))
        return _TaskRead(task, None)

    async def _read_covering_task(self, task_id: str, head: str) -> _TaskRead:
        read = await self._read_task(task_id)
        if read.task is None or owner_liveness(read.status) is not OwnerLiveness.DONE:
            return read
        commit = resolving_commit(read.task)
        ancestry = None if commit is None else await self._ancestry(commit, head)
        return replace(read, contained=fix_contained(read.task, ancestry))

    async def _ancestry(self, commit: str, head: str) -> bool | None:
        try:
            return await self._git.is_ancestor_checked(commit, head)
        except Exception:
            logger.exception('offline-lane: ancestry read %s..%s failed', commit, head)
            return None

    async def _covering_verdict(self, command: str, episode: Episode, read: _TaskRead) -> CoveringVerdict:
        task_id = episode.covering_task_id
        if read.error is not None:
            logger.error('offline-lane: %s: covering task %s of %s unreadable, no decision: %s',
                         command, task_id, episode.node, read.error)
            return CoveringVerdict.UNREADABLE
        liveness = OwnerLiveness.CLOSED if episode.binding_swallowed else owner_liveness(read.status)
        if liveness is OwnerLiveness.OPEN:
            return CoveringVerdict.OPEN
        if liveness is OwnerLiveness.DEFERRED:
            return CoveringVerdict.DEFERRED if episode.planning_committed else CoveringVerdict.DEFERRED_UNFINISHED
        if liveness is OwnerLiveness.CLOSED:
            released = await self._escalation_resolved(episode.escalation_id)
            return CoveringVerdict.CLOSED_RELEASED if released else CoveringVerdict.CLOSED
        if read.contained is None:
            logger.error('offline-lane: %s: whether head contains the fix of task %s is unknown, no decision for %s',
                         command, task_id, episode.node)
            return CoveringVerdict.UNREADABLE
        return CoveringVerdict.DONE_CONTAINED if read.contained else CoveringVerdict.DONE_STALE_HEAD

    async def _escalation_resolved(self, escalation_id: str | None) -> bool:
        if escalation_id is None or self._queue is None:
            return False
        try:
            state = await asyncio.to_thread(escalation_state, self._queue, escalation_id)
        except Exception:
            logger.exception('offline-lane: state of escalation %s unreadable', escalation_id)
            return False
        return state in CLOSED_ESCALATION_STATES

    async def _previous_detect(self, record: CommandRecord) -> RunRecord | None:
        if record.last_pass_id is None:
            return None
        previous = await asyncio.to_thread(self._records.load_pass, record.command, record.last_pass_id)
        return None if previous is None else previous.detect

    async def _retry_due_filings(self, record: CommandRecord) -> tuple[CommandRecord, list[str]]:
        if self._tasks is None:
            return record, []
        filed: list[str] = []
        for group in _due_groups(record):
            if _abandoned(record, group):
                logger.warning('offline-lane: %s: filing cancelled: no live episode in its group %s',
                               record.command, list(group))
                record = await self._saved(record, _cancelled(record, group))
                continue
            record, task_id = await self._file(self._tasks, record, group, opening=None)
            filed.extend([task_id] if task_id else [])
        return record, filed

    async def _open_filings(
        self, record: CommandRecord, groups: Sequence[tuple[str, ...]], observation: PassObservation,
    ) -> tuple[CommandRecord, list[str]]:
        if self._tasks is None:
            for group in groups:
                logger.warning('offline-lane: %s: no task client, not filing for %s', record.command, list(group))
            return record, []
        filed: list[str] = []
        for group in groups:
            record, task_id = await self._file(self._tasks, record, group, opening=observation)
            filed.extend([task_id] if task_id else [])
        return record, filed

    async def _file(
        self,
        client: FlakeLedgerTaskClient,
        record: CommandRecord,
        nodes: tuple[str, ...],
        *,
        opening: PassObservation | None,
    ) -> tuple[CommandRecord, str | None]:
        """PRD 5.4, each step its own write; a retry resumes at the first step not yet recorded.

        *opening* is the pass that made this filing due, None for a retry.  Returns the
        task id this call bound by a submission, None when it submitted nothing or was swallowed.
        """
        reproduce = reproduce_for(opening, nodes) if opening is not None else _persisted_reproduce(record, nodes)
        filed = None
        record = await self._saved(record, _marked_due(record, nodes, reproduce, self._now()))
        if _covering(record, nodes) is None:
            outcome = await self._submit(client, await self._fix_task_arguments(record, nodes, reproduce, opening))
            record = await self._saved(record, _after_submit(record, nodes, outcome, self._now()))
            filed = self._submitted(record, nodes)
        task_id = _covering(record, nodes)
        if task_id is None or not record.episodes[nodes[0]].filing_due:
            return record, filed
        await self._commit_planning(client, task_id)
        read = await self._read_task(task_id)
        record = await self._saved(record, _after_commit(record, nodes, task_id, read))
        record = await self._saved(record, _cleared(record, nodes))
        return record, filed

    def _submitted(self, record: CommandRecord, nodes: Sequence[str]) -> str | None:
        episode = record.episodes[nodes[0]]
        if episode.covering_task_id is None:
            logger.warning('offline-lane: %s: filing for %s failed, still due: %s',
                           record.command, list(nodes), episode.filing_failure)
            return None
        if episode.binding_swallowed:
            logger.error('offline-lane: %s: filing for %s was swallowed by terminal task %s, treated as cancelled',
                         record.command, list(nodes), episode.covering_task_id)
            return None
        return episode.covering_task_id

    async def _submit(self, client: FlakeLedgerTaskClient, arguments: dict[str, Any]) -> SubmitOutcome:
        try:
            return await client.submit_task(arguments)
        except Exception as exc:
            logger.exception('offline-lane: submit_task raised')
            return SubmitFailed(f'submit_task raised {exc!r}')

    async def _commit_planning(self, client: FlakeLedgerTaskClient, task_id: str) -> None:
        try:
            await client.commit_planning([task_id])
        except Exception:
            logger.exception('offline-lane: commit_planning raised for task %s', task_id)

    async def _fix_task_arguments(
        self, record: CommandRecord, nodes: tuple[str, ...], reproduce: str, opening: PassObservation | None,
    ) -> dict[str, Any]:
        leg, priority = record.filing_identity()
        modes = {} if opening is None else {opening.pass_id: _outcome_mode(opening.outcome)}
        episodes = [record.episodes[node] for node in nodes]
        if nodes == (unconfirmable_node(record.command),):
            episode = episodes[0]
            return build_lane_command_task_arguments(
                command=record.command, leg=leg, priority=priority, head=episode.first_red_head,
                node=episode.node, passes=await self._counted_evidence(record.command, episode, modes),
                reason=episode.kind, reproduce=reproduce, prior_task_ids=episode.prior_task_ids,
                notes=episode.notes,
            )
        tests = [
            TestEvidence(
                node=episode.node,
                kind=episode.kind,
                passes=await self._counted_evidence(record.command, episode, modes),
                first_red_head=episode.first_red_head,
                last_green_head=episode.last_green_head,
                notes=episode.notes,
                prior_task_ids=episode.prior_task_ids,
            )
            for episode in episodes
        ]
        return build_lane_fix_task_arguments(
            command=record.command, leg=leg, priority=priority, tests=tests, reproduce=reproduce,
        )

    async def _counted_evidence(
        self, command: str, episode: Episode, modes: Mapping[str, str],
    ) -> tuple[CountedPassEvidence, ...]:
        evidence = []
        for pass_id in episode.counted_passes:
            record = await asyncio.to_thread(self._records.load_pass, command, pass_id)
            record_dir = self._records.pass_directory(command, pass_id)
            evidence.append(_counted_pass(command, episode, pass_id, record, record_dir, modes.get(pass_id)))
        return tuple(evidence)

    async def _raise_unlanded(
        self, record: CommandRecord, task_ids: Sequence[str], statuses: Mapping[str, str],
    ) -> tuple[CommandRecord, tuple[str, ...]]:
        """D11: one record per covering task; a failed write is retried at the next qualifying pass."""
        if task_ids and self._queue is None:
            logger.error('offline-lane: %s: no escalation queue; unlanded red not raised for tasks %s',
                         record.command, list(task_ids))
        if self._queue is None:
            return record, ()
        raised: list[str] = []
        for task_id in task_ids:
            nodes = tuple(sorted(n for n, e in record.episodes.items() if e.covering_task_id == task_id))
            bound = [record.episodes[node] for node in nodes]
            red = UnlandedRed(
                command=record.command,
                covering_task_id=task_id,
                task_status=statuses.get(task_id, UNKNOWN_STATUS),
                nodes=tuple(episode.node for episode in bound if episode.live),
                filed_at=min(e.filed_at for e in bound if e.filed_at is not None),
                record_dirs=tuple(
                    str(self._records.pass_directory(record.command, pass_id))
                    for pass_id in dict.fromkeys(p for e in bound for p in e.counted_passes)
                ),
            )
            try:
                escalation_id = await asyncio.to_thread(file_unlanded_red, self._queue, red)
            except Exception:
                logger.exception('offline-lane: %s: unlanded-red escalation for task %s not written; retried next pass',
                                 record.command, task_id)
                continue
            record = _with_episodes(record, nodes, escalation_id=escalation_id)
            raised.append(escalation_id)
        return record, tuple(raised)

    async def _maintained_stuck_record(self, record: CommandRecord, policy: LanePolicy) -> CommandRecord:
        """PRD 14 item 16: one record while a filing has stayed due ``recheck_secs``; resolved by the lane after."""
        if self._queue is None:
            return record
        now = self._now()
        stuck = [
            e for e in record.episodes.values()
            if e.filing_due and e.filing_due_since is not None and now - e.filing_due_since >= policy.recheck
        ]
        current = record.stuck_filing_escalation_id
        try:
            state = None if current is None else await asyncio.to_thread(escalation_state, self._queue, current)
            if not stuck:
                if state == PENDING_ESCALATION and current is not None:
                    await asyncio.to_thread(resolve_stuck_filing, self._queue, current)
                return replace(record, stuck_filing_escalation_id=None)
            filing = StuckFiling(
                command=record.command,
                nodes=tuple(sorted(e.node for e in stuck)),
                due_since=min(e.filing_due_since for e in stuck if e.filing_due_since is not None),
                reason=_stuck_reason(stuck),
            )
            if state == PENDING_ESCALATION and current is not None:
                await asyncio.to_thread(amend_stuck_filing, self._queue, current, filing)
                return record
            escalation_id = await asyncio.to_thread(file_stuck_filing, self._queue, filing)
        except Exception:
            logger.exception('offline-lane: %s: stuck-filing escalation not maintained; retried next time',
                             record.command)
            return record
        return replace(record, stuck_filing_escalation_id=escalation_id)

