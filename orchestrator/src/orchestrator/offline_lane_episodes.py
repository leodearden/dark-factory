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
- R13 before submitting for a group with no covering task, the engine looks up the
  project's open lane fix tasks (``spawn_context`` ``offline_lane_red``, each read live),
  at most once per pass or tick, and binds each test to the first that covers it (its
  ``x_offline_lane`` command and nodes, or a pre-cutover task's ``failing_tests``),
  committed and named in ``prior_task_ids``.  Only the rest are submitted.  A failed
  lookup, or one open lane task that cannot be read, files nothing for any command in
  that pass or tick: every such filing stays due with the reason, which names that task
  and reaches the stuck-filing record.
- R14 a command record that has observed no pass takes its last pass and last green
  head from the command's run records before the pass, so a cutover or a lost store
  keeps a green pass on record.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from shared.psi import PsiSample, cpu_some10_or_none, read_psi_sample
from shared.task_statuses import OwnerLiveness, TaskStatus, owner_liveness
from shared.task_submit import SubmitFailed, SubmitOutcome, SubmittedTask, SubmittedTicket

from orchestrator.flake_ledger import (
    DebtContext,
    DebtTrigger,
    FlakeCallSite,
    FlakeLedgerTaskClient,
    FlakeSuppression,
    fix_contained,
    resolving_commit,
)
from orchestrator.flake_recorder import record_call_site_observations
from orchestrator.offline_lane_confirm import Confirmed, Outage, Unconfirmable, outcome_from_json
from orchestrator.offline_lane_episode_state import (
    Applied,
    CommandRecord,
    CoveringStatuses,
    CoveringVerdict,
    Episode,
    EpisodeStore,
    LanePolicy,
    OfferedDebt,
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
    with_recorded_history,
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
    SPAWN_CONTEXT,
    SPAWN_CONTEXT_KEY,
    UNKNOWN,
    CountedPassEvidence,
    TestEvidence,
    build_lane_command_task_arguments,
    build_lane_fix_task_arguments,
    covering_lane_task,
)
from orchestrator.offline_lane_ledger import (
    DebtDecision,
    debt_context,
    debt_decision,
    pass_suppressions,
    recent_occurrences,
)
from orchestrator.offline_lane_records import (
    RECORD_RETENTION,
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

logger = logging.getLogger(__name__)

UNKNOWN_MODE = 'unknown'
UNKNOWN_STATUS = 'unknown'
CLOSED_ESCALATION_STATES = frozenset({'resolved', 'dismissed'})
DEFERRED_STATUS = 'deferred'
_OPEN_STATUSES = tuple(sorted(s.value for s in TaskStatus if owner_liveness(s) is OwnerLiveness.OPEN))


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


class LaneTaskClient(FlakeLedgerTaskClient, Protocol):
    """The lane's task seam: the flake ledger's, and the metadata lookup R13 adopts open fix tasks through."""

    async def find_tasks_by_metadata(
        self, key: str, value: str, statuses: Sequence[str],
    ) -> tuple[list[dict], Exception | None]: ...


@dataclass(frozen=True)
class TickResult:
    commands_due: tuple[str, ...]
    tasks_filed: tuple[str, ...]


_OpenLaneTasks = tuple[dict[str, Mapping[str, Any]], str | None]


class _OpenLaneTaskLookup:
    """R13's read of the open lane fix tasks, made at most once per pass or tick and shared by its filings.

    A task submitted later in the same pass or tick names only its own group's nodes, which no
    other group shares, so no filing needs to see it.
    """

    def __init__(self, read: Callable[[], Awaitable[_OpenLaneTasks]]) -> None:
        self._read = read
        self._answer: _OpenLaneTasks | None = None

    async def answer(self) -> _OpenLaneTasks:
        if self._answer is None:
            self._answer = await self._read()
        return self._answer


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
            committed = _with_episodes(
                bound, nodes, planning_committed=True, binding_swallowed=liveness is not OwnerLiveness.OPEN,
            )
            return _cleared(committed, nodes)
        case SubmittedTicket(ticket_id=ticket_id):
            failure = f'submit_task returned ticket {ticket_id!r}, not a task: planning mode did not take effect'
        case SubmitFailed(reason=reason):
            failure = reason
    return _with_episodes(record, nodes, filing_failure=failure)


def _adopted(record: CommandRecord, adopted: Mapping[str, str], now: datetime) -> CommandRecord:
    """R13: each node of *adopted* bound to its open task, as a submission combined into an open task binds."""
    for node, task_id in adopted.items():
        prior = record.episodes[node].prior_task_ids
        record = _with_episodes(
            record, (node,), covering_task_id=task_id, planning_committed=True, filed_at=now,
            filing_failure=None, binding_swallowed=False,
            prior_task_ids=prior if task_id in prior else (*prior, task_id),
        )
    return _cleared(record, tuple(adopted))


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
        task_client: LaneTaskClient | None,
        escalation_queue: EscalationQueue | None,
        now: Callable[[], datetime] | None = None,
        probe_runner: ProbeSeam = run_probe,
        read_psi: Callable[[], PsiSample] = read_psi_sample,
    ) -> None:
        self._config = config
        self._git = git
        self._tasks = task_client
        self._queue = escalation_queue
        self._now = now or _utc_now
        self._probe_runner = probe_runner
        self._read_psi = read_psi
        self._store = EpisodeStore(config.project_root, clock=self._now)
        self._records = RunRecordStore(config.project_root, clock=self._now)

    async def observe(self, observation: PassObservation) -> PassEffect:
        if isinstance(observation.outcome, Outage):
            return await self._observed_outage(observation, observation.outcome.probe)
        policy = lane_policy(self._config)
        record = await asyncio.to_thread(self._store.load, observation.command)
        open_tasks = _OpenLaneTaskLookup(self._open_lane_tasks)
        record, filed = await self._retry_due_filings(record, open_tasks)
        record = await self._with_recorded_history(record, observation)
        covering = await self._covering_statuses(record)
        reads, statuses = await self._pass_reads(record, observation, covering)
        applied = apply_pass(record, observation, reads, policy, self._now())
        offered, undecided = await self._recorded_in_ledger(record, applied, observation, covering)
        record = applied.record
        await self._save(record)
        unfinished = {
            task_id for node, verdict in reads.verdicts.items()
            if verdict is CoveringVerdict.DEFERRED_UNFINISHED
            and (task_id := record.episodes[node].covering_task_id) is not None
        }
        groups = [*applied.to_file, *(g for g in _due_groups(record) if _covering(record, g) in unfinished)]
        record, opened = await self._open_filings(record, groups, observation, open_tasks)
        saved = record
        record, raised = await self._raise_unlanded(record, applied.unlanded, statuses)
        record = await self._maintained_stuck_record(record, policy)
        probed: tuple[str, ...] = ()
        if observation.probe is not None:
            record, probed = await self._probe_answered(record, observation, observation.probe)
        if record != saved:
            await self._save(record)
        return replace(
            applied.effect,
            tasks_filed=(*filed, *opened),
            escalations_raised=(*raised, *probed),
            debts_offered=offered,
            debts_undecided=undecided,
        )

    async def _recorded_in_ledger(
        self, before: CommandRecord, applied: Applied, observation: PassObservation, covering: CoveringStatuses,
    ) -> tuple[tuple[OfferedDebt, ...], tuple[str, ...]]:
        """D6: what this pass tells the flake ledger, written before the episodes' new state and any filing.

        Returns the debts offered and the tests left undecided by an unreadable covering status.
        """
        if not isinstance(observation.outcome, Confirmed | Unconfirmable):
            return (), ()
        return await self._red_recorded(before, observation, covering)

    async def _red_recorded(
        self, before: CommandRecord, observation: PassObservation, covering: CoveringStatuses,
    ) -> tuple[tuple[OfferedDebt, ...], tuple[str, ...]]:
        """The red pass's one recorder call: its suppressions, and its cleared tests that owe debt."""
        eligible, undecided = await self._cleared_debts(before, observation, covering)
        record_dir = str(self._records.pass_directory(observation.command, observation.pass_id))
        suppressions = pass_suppressions(observation, record_dir=record_dir, psi_cpu_some10=self._pressure())
        await self._record(suppressions, eligible, head=observation.head, observation=observation)
        offered = tuple(OfferedDebt(node, DebtTrigger.cleared, observation.head) for node in eligible)
        return offered, undecided

    async def _record(
        self,
        suppressions: Sequence[FlakeSuppression],
        eligible: Mapping[str, DebtContext],
        *,
        head: str,
        observation: PassObservation,
    ) -> None:
        """One recorder call at *head*, at *observation*'s start: *suppressions* and the due items of *eligible*."""
        await record_call_site_observations(
            suppressions,
            call_site=FlakeCallSite.offline_lane,
            project_root=self._config.project_root,
            project_id=self._config.fused_memory.project_id,
            head_sha=head,
            observed_at=observation.started_at.isoformat(),
            observing_task_id=None,
            debt_eligible=eligible,
            event_store=None,
            escalation_queue=self._queue,
        )

    async def _cleared_debts(
        self, before: CommandRecord, observation: PassObservation, covering: CoveringStatuses,
    ) -> tuple[dict[str, DebtContext], tuple[str, ...]]:
        """The contexts of the cleared tests that owe flake debt, and the cleared tests left undecided."""
        outcome = observation.outcome
        if not isinstance(outcome, Confirmed):
            return {}, ()
        eligible: dict[str, DebtContext] = {}
        undecided: list[str] = []
        for node in outcome.cleared:
            episode = before.episodes.get(node)
            match debt_decision(node, observation.command, episode, covering):
                case DebtDecision.ELIGIBLE:
                    earlier = await self._counted_records(observation.command, episode)
                    occurrences = recent_occurrences(node, earlier, observation)
                    eligible[node] = debt_context(node, DebtTrigger.cleared, observation, episode, occurrences)
                case DebtDecision.COVERED:
                    logger.info('offline-lane: %s: %s cleared under its open fix task; owes no flake debt',
                                observation.command, node)
                case DebtDecision.UNDECIDED:
                    logger.error(
                        'offline-lane: %s: cleared test %s not offered as flake debt: status of its covering '
                        'task %s unreadable: %s',
                        observation.command, node, episode.covering_task_id if episode else None, covering.error,
                    )
                    undecided.append(node)
                case DebtDecision.NEVER:
                    pass
        return eligible, tuple(undecided)

    async def _counted_records(self, command: str, episode: Episode | None) -> list[PassRecord]:
        """The run records of *episode*'s counted passes, oldest first, read off the loop; unreadable ones skipped."""
        if episode is None:
            return []
        loaded = [await asyncio.to_thread(self._records.load_pass, command, pass_id) for pass_id in episode.counted_passes]
        return [record for record in loaded if record is not None]

    def _pressure(self) -> float | None:
        """Host CPU pressure for the ledger's rows; a failed read is ``None`` and never changes an outcome."""
        try:
            return cpu_some10_or_none(self._read_psi())
        except Exception:
            logger.warning('offline-lane: host pressure read failed; recording psi_cpu_some10=None', exc_info=True)
            return None

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
        open_tasks = _OpenLaneTaskLookup(self._open_lane_tasks)
        for command in await asyncio.to_thread(self._store.commands):
            record = await asyncio.to_thread(self._store.load, command)
            record, tasks = await self._retry_due_filings(record, open_tasks)
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
        self, record: CommandRecord, observation: PassObservation, covering: CoveringStatuses,
    ) -> tuple[PassReads, dict[str, str]]:
        """D10's reads for the red-again bound tests, the batched refresh read, and the previous pass's detect record."""
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
        return PassReads(verdicts, covering.terminal_task_ids, previous), statuses

    async def _covering_statuses(self, record: CommandRecord) -> CoveringStatuses:
        """The pass's one batched status read of every bound covering task."""
        bound = sorted({e.covering_task_id for e in record.episodes.values() if e.covering_task_id is not None})
        if not bound:
            return CoveringStatuses({})
        if self._tasks is None:
            return CoveringStatuses({}, 'no task client')
        try:
            statuses, error = await self._tasks.get_statuses(bound)
        except Exception as exc:
            statuses, error = {}, exc
        if error is not None:
            logger.error(
                'offline-lane: %s: covering task statuses unreadable, every binding refreshed: %s',
                record.command, error,
            )
            return CoveringStatuses({}, str(error))
        return CoveringStatuses({task_id: statuses[task_id] for task_id in bound if task_id in statuses})

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

    async def _with_recorded_history(self, record: CommandRecord, observation: PassObservation) -> CommandRecord:
        """R14; the run records are evidence, never a gate, so an unreadable listing seeds nothing."""
        if record.last_pass_id is not None:
            return record
        try:
            recorded = await asyncio.to_thread(self._recorded_before, observation)
        except (OSError, ValueError):
            logger.exception('offline-lane: %s: run records unreadable, no history seeded', record.command)
            return record
        return with_recorded_history(record, recorded)

    def _recorded_before(self, observation: PassObservation) -> list[PassRecord]:
        recorded = self._records.passes_since(observation.command, observation.started_at - RECORD_RETENTION)
        return [earlier for earlier in recorded if earlier.started_at < observation.started_at]

    async def _previous_detect(self, record: CommandRecord) -> RunRecord | None:
        if record.last_pass_id is None:
            return None
        previous = await asyncio.to_thread(self._records.load_pass, record.command, record.last_pass_id)
        return None if previous is None else previous.detect

    async def _retry_due_filings(
        self, record: CommandRecord, open_tasks: _OpenLaneTaskLookup,
    ) -> tuple[CommandRecord, list[str]]:
        if self._tasks is None:
            return record, []
        filed: list[str] = []
        for group in _due_groups(record):
            if _abandoned(record, group):
                logger.warning('offline-lane: %s: filing cancelled: no live episode in its group %s',
                               record.command, list(group))
                record = await self._saved(record, _cancelled(record, group))
                continue
            record, task_id = await self._file(self._tasks, record, group, opening=None, open_tasks=open_tasks)
            filed.extend([task_id] if task_id else [])
        return record, filed

    async def _open_filings(
        self,
        record: CommandRecord,
        groups: Sequence[tuple[str, ...]],
        observation: PassObservation,
        open_tasks: _OpenLaneTaskLookup,
    ) -> tuple[CommandRecord, list[str]]:
        if self._tasks is None:
            for group in groups:
                logger.warning('offline-lane: %s: no task client, not filing for %s', record.command, list(group))
            return record, []
        filed: list[str] = []
        for group in groups:
            record, task_id = await self._file(self._tasks, record, group, opening=observation, open_tasks=open_tasks)
            filed.extend([task_id] if task_id else [])
        return record, filed

    async def _file(
        self,
        client: LaneTaskClient,
        record: CommandRecord,
        nodes: tuple[str, ...],
        *,
        opening: PassObservation | None,
        open_tasks: _OpenLaneTaskLookup,
    ) -> tuple[CommandRecord, str | None]:
        """PRD 5.4, each step its own write; a retry resumes at the first step not yet recorded.

        *opening* is the pass that made this filing due, None for a retry.  Returns the
        task id this call bound by a submission, None when it submitted nothing or was swallowed.
        """
        reproduce = reproduce_for(opening, nodes) if opening is not None else _persisted_reproduce(record, nodes)
        filed = None
        record = await self._saved(record, _marked_due(record, nodes, reproduce, self._now()))
        if _covering(record, nodes) is None:
            record, nodes = await self._adopt_open_tasks(open_tasks, record, nodes)
            if not nodes:
                return record, None
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

    async def _adopt_open_tasks(
        self, lookup: _OpenLaneTaskLookup, record: CommandRecord, nodes: tuple[str, ...],
    ) -> tuple[CommandRecord, tuple[str, ...]]:
        """R13: the saved record with every node an open lane fix task covers bound to it; the nodes left to submit."""
        open_tasks, failure = await lookup.answer()
        if failure is not None:
            logger.warning('offline-lane: %s: filing for %s not attempted, still due: %s',
                           record.command, list(nodes), failure)
            return await self._saved(record, _with_episodes(record, nodes, filing_failure=failure)), ()
        adopted = {
            node: task_id for node in nodes
            if (task_id := covering_lane_task(open_tasks, record.command, node)) is not None
        }
        if adopted:
            logger.warning('offline-lane: %s: adopted the open lane fix tasks covering %s, not filing for them',
                           record.command, adopted)
        record = await self._saved(record, _adopted(record, adopted, self._now()))
        return record, tuple(node for node in nodes if node not in adopted)

    async def _open_lane_tasks(self) -> _OpenLaneTasks:
        """The project's open lane fix tasks, read live, in lookup order; or why they could not be read."""
        if self._tasks is None:
            return {}, 'no task client'
        try:
            rows, error = await self._tasks.find_tasks_by_metadata(SPAWN_CONTEXT_KEY, SPAWN_CONTEXT, _OPEN_STATUSES)
        except Exception as exc:
            rows, error = [], exc
        if error is not None:
            return {}, f'open lane task lookup failed: {error}'
        open_tasks: dict[str, Mapping[str, Any]] = {}
        for task_id in (str(row['task_id']) for row in rows):
            read = await self._read_task(task_id)
            if read.error is not None:
                return {}, f'open lane task {task_id} unreadable: {read.error}'
            if read.task is not None and owner_liveness(read.status) is OwnerLiveness.OPEN:
                open_tasks[task_id] = read.task
        return open_tasks, None

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

