"""What the offline lane remembers about its red tests, and what one pass does to it.

PRD ``plans/offline-lane-red-path-prd.md`` D5, D10, D12 and section 5.3: one record per
lane command, its per-test episodes, the store that keeps it, and the pure rules over
it (:func:`apply_pass` for a pass, :func:`apply_outage` for an outage pass,
:func:`with_probe` for a probe's answer, :func:`recheck_due` for an idle tick).  The I/O
around them is ``orchestrator/src/orchestrator/offline_lane_episodes.py::LaneEngine``.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from orchestrator.guard_state import PersistentMap, guard_path
from orchestrator.offline_lane_confirm import Confirmed, Outage, Unconfirmable, reproduce_line
from orchestrator.offline_lane_filing import EvidenceNote
from orchestrator.offline_lane_records import Leg, ProbeKind, ProbeResult, RunRecord
from orchestrator.offline_lane_runs import (
    INFRA_SCOPE,
    RUN_ALL_INFRA_SCRIPT,
    RUN_OFFLINE_DEEP_SCRIPT,
)

if TYPE_CHECKING:
    from orchestrator.config import LaneCommand

logger = logging.getLogger(__name__)

EPISODES_FILE = 'offline_lane_episodes.json'
EPISODE_RETENTION = timedelta(days=14)
UNCONFIRMABLE_SUFFIX = '::unconfirmable'
DEFAULT_FAILURE_KIND = 'failed'

_LEGACY_REPRODUCE = {
    Leg.NUMERIC: RUN_OFFLINE_DEEP_SCRIPT,
    Leg.INFRA: f'{RUN_ALL_INFRA_SCRIPT} --scope {INFRA_SCOPE}',
}

_T = TypeVar('_T')
_REQUIRED: Any = object()


def unconfirmable_node(command: str) -> str:
    return f'{command}{UNCONFIRMABLE_SUFFIX}'


@dataclass(frozen=True)
class LanePolicy:
    persist_passes: int = 2
    recheck_secs: float = 1800.0
    red_unlanded_hours: float = 24.0

    @property
    def recheck(self) -> timedelta:
        return timedelta(seconds=self.recheck_secs)

    @property
    def red_unlanded(self) -> timedelta:
        return timedelta(hours=self.red_unlanded_hours)



class _Fields:
    """Typed reads of one JSON object; every miss is a ``ValueError`` unless the read names its *absent* value."""

    def __init__(self, payload: object, what: str) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError(f'{what} must be an object, got {type(payload).__name__}')
        self._payload: Mapping[str, Any] = payload
        self._what = what

    def raw(self, key: str, *, absent: Any = _REQUIRED) -> Any:
        if key not in self._payload:
            if absent is not _REQUIRED:
                return absent
            raise ValueError(f'{self._what}: missing {key!r}')
        return self._payload[key]

    def typed(self, key: str, expected: type[_T], *, absent: Any = _REQUIRED) -> _T:
        value = self.raw(key, absent=absent)
        if not isinstance(value, expected) or (expected is not bool and isinstance(value, bool)):
            raise ValueError(f'{self._what}: {key!r} must be {expected.__name__}, got {value!r}')
        return value

    def optional(self, key: str, expected: type[_T], *, absent: Any = _REQUIRED) -> _T | None:
        return None if self.raw(key, absent=absent) is None else self.typed(key, expected)

    def instant(self, key: str) -> datetime:
        try:
            instant = datetime.fromisoformat(self.typed(key, str))
        except ValueError as exc:
            raise ValueError(f'{self._what}: {key!r} is not an instant: {exc}') from exc
        if instant.tzinfo is None:
            raise ValueError(f'{self._what}: {key!r} must carry a UTC offset')
        return instant

    def optional_instant(self, key: str, *, absent: Any = _REQUIRED) -> datetime | None:
        return None if self.raw(key, absent=absent) is None else self.instant(key)

    def strings(self, key: str) -> tuple[str, ...]:
        values = self.typed(key, list)
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f'{self._what}: {key!r} must list strings')
        return tuple(values)

    def objects(self, key: str) -> list[object]:
        return self.typed(key, list)


def _iso(instant: datetime | None) -> str | None:
    return None if instant is None else instant.isoformat()


def _note_to_json(note: EvidenceNote) -> dict[str, Any]:
    return {'source': note.source, 'node': note.node, 'text': note.text}


def _note_from_json(payload: object) -> EvidenceNote:
    fields = _Fields(payload, 'note')
    return EvidenceNote(fields.typed('source', str), fields.optional('node', str), fields.typed('text', str))


@dataclass(frozen=True)
class Episode:
    """One (command, node) red episode, and the binding to its covering task that outlives it."""

    node: str
    kind: str
    counted_passes: tuple[str, ...]
    last_counted_started_at: datetime | None
    first_red_head: str
    reproduced_heads: tuple[str, ...]
    last_green_head: str | None
    had_outage_pass: bool
    covering_task_id: str | None
    planning_committed: bool
    filed_at: datetime | None
    filing_due: bool
    filing_due_since: datetime | None
    escalation_id: str | None
    notes: tuple[EvidenceNote, ...]
    prior_task_ids: tuple[str, ...]
    ended_at: datetime | None
    last_refreshed_at: datetime
    filing_failure: str | None = None
    reproduce: str | None = None
    binding_swallowed: bool = False

    def __post_init__(self) -> None:
        if self.filing_due and (self.filing_due_since is None or self.reproduce is None):
            raise ValueError(f'{self.node}: a due filing needs its due time and reproduce line')

    @property
    def live(self) -> bool:
        return self.ended_at is None

    def to_json(self) -> dict[str, Any]:
        return {
            'node': self.node,
            'kind': self.kind,
            'counted_passes': list(self.counted_passes),
            'last_counted_started_at': _iso(self.last_counted_started_at),
            'first_red_head': self.first_red_head,
            'reproduced_heads': list(self.reproduced_heads),
            'last_green_head': self.last_green_head,
            'had_outage_pass': self.had_outage_pass,
            'covering_task_id': self.covering_task_id,
            'planning_committed': self.planning_committed,
            'filed_at': _iso(self.filed_at),
            'filing_due': self.filing_due,
            'filing_due_since': _iso(self.filing_due_since),
            'escalation_id': self.escalation_id,
            'notes': [_note_to_json(note) for note in self.notes],
            'prior_task_ids': list(self.prior_task_ids),
            'ended_at': _iso(self.ended_at),
            'last_refreshed_at': self.last_refreshed_at.isoformat(),
            'filing_failure': self.filing_failure,
            'reproduce': self.reproduce,
            'binding_swallowed': self.binding_swallowed,
        }

    @classmethod
    def from_json(cls, payload: object) -> Episode:
        fields = _Fields(payload, 'episode')
        return cls(
            node=fields.typed('node', str),
            kind=fields.typed('kind', str),
            counted_passes=fields.strings('counted_passes'),
            last_counted_started_at=fields.optional_instant('last_counted_started_at'),
            first_red_head=fields.typed('first_red_head', str),
            reproduced_heads=fields.strings('reproduced_heads'),
            last_green_head=fields.optional('last_green_head', str),
            had_outage_pass=fields.typed('had_outage_pass', bool),
            covering_task_id=fields.optional('covering_task_id', str),
            planning_committed=fields.typed('planning_committed', bool),
            filed_at=fields.optional_instant('filed_at'),
            filing_due=fields.typed('filing_due', bool),
            filing_due_since=fields.optional_instant('filing_due_since'),
            escalation_id=fields.optional('escalation_id', str),
            notes=tuple(_note_from_json(note) for note in fields.objects('notes')),
            prior_task_ids=fields.strings('prior_task_ids'),
            ended_at=fields.optional_instant('ended_at'),
            last_refreshed_at=fields.instant('last_refreshed_at'),
            filing_failure=fields.optional('filing_failure', str),
            reproduce=fields.optional('reproduce', str),
            binding_swallowed=fields.typed('binding_swallowed', bool),
        )


@dataclass(frozen=True)
class CommandRecord:
    """Everything the lane remembers about one command; ``leg`` and ``fix_task_priority`` are None until its first pass.

    ``probe_record_behind`` is true while an escalation write for the latest probe
    answer has failed, so the lane's open probe record does not yet state it.
    """

    command: str
    leg: str | None
    episodes: Mapping[str, Episode]
    rechecks: Mapping[str, int]
    outage_escalation_id: str | None
    last_probe_result: ProbeResult | None
    last_pass_started_at: datetime | None
    last_pass_head: str | None
    last_pass_id: str | None
    last_green_head: str | None
    stuck_filing_escalation_id: str | None
    fix_task_priority: str | None
    probe_broken_escalation_id: str | None
    last_probe_at: datetime | None
    probe_record_behind: bool

    @classmethod
    def empty(cls, command: str) -> CommandRecord:
        return cls(
            command=command,
            leg=None,
            episodes={},
            rechecks={},
            outage_escalation_id=None,
            last_probe_result=None,
            last_pass_started_at=None,
            last_pass_head=None,
            last_pass_id=None,
            last_green_head=None,
            stuck_filing_escalation_id=None,
            fix_task_priority=None,
            probe_broken_escalation_id=None,
            last_probe_at=None,
            probe_record_behind=False,
        )

    @property
    def in_outage(self) -> bool:
        """D12's hold: the command's last probe answered an outage."""
        return self.last_probe_result is not None and self.last_probe_result.result == ProbeKind.OUTAGE

    def filing_identity(self) -> tuple[str, str]:
        """``(leg, fix_task_priority)``; a filing exists only after a pass set both."""
        if self.leg is None or self.fix_task_priority is None:
            raise ValueError(f'{self.command}: no pass has been observed to file from')
        return self.leg, self.fix_task_priority

    def to_json(self) -> dict[str, Any]:
        return {
            'command': self.command,
            'leg': self.leg,
            'episodes': [episode.to_json() for episode in self.episodes.values()],
            'rechecks': dict(self.rechecks),
            'outage_escalation_id': self.outage_escalation_id,
            'last_probe_result': None if self.last_probe_result is None else asdict(self.last_probe_result),
            'last_pass_started_at': _iso(self.last_pass_started_at),
            'last_pass_head': self.last_pass_head,
            'last_pass_id': self.last_pass_id,
            'last_green_head': self.last_green_head,
            'stuck_filing_escalation_id': self.stuck_filing_escalation_id,
            'fix_task_priority': self.fix_task_priority,
            'probe_broken_escalation_id': self.probe_broken_escalation_id,
            'last_probe_at': _iso(self.last_probe_at),
            'probe_record_behind': self.probe_record_behind,
        }

    @classmethod
    def from_json(cls, payload: object) -> CommandRecord:
        """Strict on every key but the three task 6863 added, which a record written before it lacks."""
        fields = _Fields(payload, 'command record')
        episodes = [Episode.from_json(item) for item in fields.objects('episodes')]
        rechecks = fields.typed('rechecks', dict)
        if not all(
            isinstance(head, str) and isinstance(count, int) and not isinstance(count, bool)
            for head, count in rechecks.items()
        ):
            raise ValueError("command record: 'rechecks' must map heads to counts")
        probe = fields.optional('last_probe_result', dict)
        return cls(
            command=fields.typed('command', str),
            leg=fields.optional('leg', str),
            episodes={episode.node: episode for episode in episodes},
            rechecks=rechecks,
            outage_escalation_id=fields.optional('outage_escalation_id', str),
            last_probe_result=None if probe is None else ProbeResult.from_json(probe),
            last_pass_started_at=fields.optional_instant('last_pass_started_at'),
            last_pass_head=fields.optional('last_pass_head', str),
            last_pass_id=fields.optional('last_pass_id', str),
            last_green_head=fields.optional('last_green_head', str),
            stuck_filing_escalation_id=fields.optional('stuck_filing_escalation_id', str),
            fix_task_priority=fields.optional('fix_task_priority', str),
            probe_broken_escalation_id=fields.optional('probe_broken_escalation_id', str, absent=None),
            last_probe_at=fields.optional_instant('last_probe_at', absent=None),
            probe_record_behind=fields.typed('probe_record_behind', bool, absent=False),
        )


class EpisodeStore:
    """The lane's episode records, one whole record per command.  Synchronous: callers on the loop use ``asyncio.to_thread``."""

    def __init__(self, project_root: str | Path | None, *, clock: Callable[[], datetime]) -> None:
        self._map = PersistentMap(guard_path(project_root, EPISODES_FILE), ttl=EPISODE_RETENTION, now=clock)

    def load(self, command: str) -> CommandRecord:
        payload = self._map.get(command)
        if payload is None:
            return CommandRecord.empty(command)
        try:
            record = CommandRecord.from_json(payload)
        except ValueError as exc:
            logger.error('offline-lane: episode record of %r is malformed, treated as empty: %s', command, exc)
            return CommandRecord.empty(command)
        if record.command != command:
            logger.error('offline-lane: episode record under %r names %r, treated as empty', command, record.command)
            return CommandRecord.empty(command)
        return record

    def save(self, record: CommandRecord) -> None:
        self._map[record.command] = record.to_json()

    def commands(self) -> list[str]:
        return list(self._map)

    def pinned_pass_ids(self) -> frozenset[str]:
        """Every counted pass id an episode still held, live or ended, refers to (PRD 5.1's retention exception)."""
        return frozenset(
            pass_id
            for command in self.commands()
            for episode in self.load(command).episodes.values()
            for pass_id in episode.counted_passes
        )


class CoveringVerdict(StrEnum):
    """D10: what a red-again test's covering task means for this pass."""

    OPEN = 'open'
    DEFERRED_UNFINISHED = 'deferred_unfinished'
    DEFERRED = 'deferred'
    DONE_CONTAINED = 'done_contained'
    DONE_STALE_HEAD = 'done_stale_head'
    CLOSED = 'closed'
    CLOSED_RELEASED = 'closed_released'
    UNREADABLE = 'unreadable'


_BINDING_DROPPED = frozenset({CoveringVerdict.DONE_CONTAINED, CoveringVerdict.CLOSED_RELEASED})
_BINDING_KEPT = frozenset({CoveringVerdict.OPEN, CoveringVerdict.DEFERRED, CoveringVerdict.DEFERRED_UNFINISHED})
_UNLANDED = _BINDING_KEPT | {CoveringVerdict.CLOSED}


@dataclass(frozen=True)
class PassObservation:
    """One pass of one command; ``lane_command`` is the generic command's config, None on the legacy legs.

    ``probe`` is the outage probe this pass ran, None when it ran none.
    """

    command: str
    leg: str
    head: str
    pass_id: str
    started_at: datetime
    detect: RunRecord | None
    confirm: RunRecord | None
    outcome: Confirmed | Unconfirmable | Outage | None
    fix_task_priority: str
    notes: tuple[EvidenceNote, ...]
    lane_command: LaneCommand | None
    probe: ProbeResult | None = None

    def __post_init__(self) -> None:
        generic = Leg(self.leg) is Leg.GENERIC
        if generic != (self.lane_command is not None):
            raise ValueError(f'a {self.leg} pass carries a lane command exactly when it is generic')
        if self.probe is not None and not generic:
            raise ValueError(f'a {self.leg} pass runs no outage probe')
        if isinstance(self.outcome, Outage) and self.outcome.probe != self.probe:
            raise ValueError(f'{self.command}: an outage pass carries the probe that answered it')


@dataclass(frozen=True)
class PassReads:
    """What the engine read before applying a pass: D10 verdicts for the red-again bound tests,
    the bound task ids the batched status read showed terminal, and the previous pass's detect record."""

    verdicts: Mapping[str, CoveringVerdict]
    terminal_task_ids: frozenset[str]
    previous_detect: RunRecord | None


class EndReason(StrEnum):
    CLEARED = 'cleared'
    PASSED = 'passed'
    EXPIRED = 'expired'
    REPLACED = 'replaced'


@dataclass(frozen=True)
class EndedEpisode:
    node: str
    reason: str
    counted_passes: int
    reproduced_heads: tuple[str, ...]


@dataclass(frozen=True)
class PassEffect:
    tests_counted: tuple[str, ...]
    episodes_started: tuple[str, ...]
    episodes_extended: tuple[str, ...]
    episodes_ended: tuple[EndedEpisode, ...]
    threshold_reached: tuple[str, ...]
    stale_heads: tuple[str, ...]
    no_decision: tuple[str, ...]
    tasks_filed: tuple[str, ...]
    escalations_raised: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            'tests_counted': list(self.tests_counted),
            'episodes_started': list(self.episodes_started),
            'episodes_extended': list(self.episodes_extended),
            'episodes_ended': [
                {
                    'node': ended.node,
                    'reason': ended.reason,
                    'counted_passes': ended.counted_passes,
                    'reproduced_heads': list(ended.reproduced_heads),
                }
                for ended in self.episodes_ended
            ],
            'threshold_reached': list(self.threshold_reached),
            'stale_heads': list(self.stale_heads),
            'no_decision': list(self.no_decision),
            'tasks_filed': list(self.tasks_filed),
            'escalations_raised': list(self.escalations_raised),
        }


@dataclass(frozen=True)
class Applied:
    record: CommandRecord
    effect: PassEffect
    to_file: tuple[tuple[str, ...], ...]
    unlanded: tuple[str, ...]


class _Shows(StrEnum):
    """D5's rows for one test, before D10 and spacing are applied."""

    REPRODUCED = 'reproduced'
    CLEARED = 'cleared'
    PASSED = 'passed'
    NOTHING = 'nothing'


def _passed(detect: RunRecord | None, node: str) -> bool:
    if detect is None:
        return False
    if detect.outcomes is not None and node in detect.outcomes:
        return detect.outcomes[node] == 'passed'
    return detect.returncode == 0


def _shows(observation: PassObservation, node: str) -> _Shows:
    outcome = observation.outcome
    if isinstance(outcome, Unconfirmable):
        return _Shows.REPRODUCED if node == unconfirmable_node(observation.command) else _Shows.NOTHING
    if isinstance(outcome, Confirmed):
        if node in outcome.reproduced:
            return _Shows.REPRODUCED
        if node in outcome.cleared:
            return _Shows.CLEARED
        if node in outcome.inconclusive:
            return _Shows.NOTHING
    return _Shows.PASSED if _passed(observation.detect, node) else _Shows.NOTHING


def reproduced_nodes(observation: PassObservation) -> tuple[str, ...]:
    outcome = observation.outcome
    if isinstance(outcome, Unconfirmable):
        return (unconfirmable_node(observation.command),)
    if isinstance(outcome, Confirmed):
        return outcome.reproduced
    return ()


def _failure_kind(observation: PassObservation, node: str) -> str:
    if isinstance(observation.outcome, Unconfirmable):
        return observation.outcome.reason.value
    for run in (observation.detect, observation.confirm):
        if run is not None:
            for failure in run.failures:
                if failure.node == node:
                    return failure.kind
    return DEFAULT_FAILURE_KIND


def _whole_command(observation: PassObservation) -> str:
    cmd = observation.lane_command
    if cmd is None:
        return _LEGACY_REPRODUCE[Leg(observation.leg)]
    return f'cd {shlex.quote(cmd.cwd)} && {cmd.command}'


def reproduce_for(observation: PassObservation, nodes: Sequence[str]) -> str:
    """A runnable line for *nodes*: the narrowed command where it narrows, else the whole command."""
    cmd = observation.lane_command
    if cmd is None or tuple(nodes) == (unconfirmable_node(observation.command),):
        return _whole_command(observation)
    return reproduce_line(cmd, nodes) or _whole_command(observation)


def _with_head(heads: tuple[str, ...], head: str) -> tuple[str, ...]:
    return heads if head in heads else (*heads, head)


def _expired(episode: Episode, at: datetime, policy: LanePolicy) -> bool:
    return (
        episode.live
        and episode.last_counted_started_at is not None
        and at - episode.last_counted_started_at >= policy.red_unlanded
    )


def _unbound(episode: Episode) -> Episode:
    prior = episode.prior_task_ids
    if episode.covering_task_id is not None and episode.covering_task_id not in prior:
        prior = (*prior, episode.covering_task_id)
    return replace(
        episode,
        covering_task_id=None,
        planning_committed=False,
        filed_at=None,
        filing_due=False,
        filing_due_since=None,
        escalation_id=None,
        filing_failure=None,
        reproduce=None,
        binding_swallowed=False,
        prior_task_ids=prior,
    )


@dataclass(frozen=True)
class _Step:
    """What one pass did to one test's episode."""

    episode: Episode | None
    counted: bool = False
    started: bool = False
    extended: bool = False
    ended: EndedEpisode | None = None
    stale: bool = False
    no_decision: bool = False


class _PassApplier:
    """The D5 and D10 rows for one pass; pure, built and discarded by :func:`apply_pass`."""

    def __init__(self, record: CommandRecord, observation: PassObservation, reads: PassReads,
                 policy: LanePolicy, now: datetime) -> None:
        self.record = record
        self.obs = observation
        self.reads = reads
        self.policy = policy
        self.now = now

    def ending(self, episode: Episode, reason: EndReason) -> tuple[Episode, EndedEpisode]:
        return (
            replace(episode, ended_at=self.now, last_refreshed_at=self.now),
            EndedEpisode(episode.node, reason.value, len(episode.counted_passes), episode.reproduced_heads),
        )

    def ended(self, episode: Episode, reason: EndReason) -> _Step:
        closed, record = self.ending(episode, reason)
        return _Step(closed, ended=record)

    def green_head_before(self, node: str) -> str | None:
        previous = self.reads.previous_detect
        if previous is not None and _passed(previous, node):
            return previous.head
        return self.record.last_green_head

    def started(self, node: str, base: Episode | None) -> Episode:
        """A new episode at one counted pass; *base* lends its binding and prior tasks."""
        obs = self.obs
        fresh = {
            'kind': _failure_kind(obs, node),
            'counted_passes': (obs.pass_id,),
            'last_counted_started_at': obs.started_at,
            'first_red_head': obs.head,
            'reproduced_heads': (obs.head,),
            'last_green_head': self.green_head_before(node),
            'had_outage_pass': False,
            'notes': (),
            'ended_at': None,
            'last_refreshed_at': self.now,
        }
        if base is not None:
            return replace(base, **fresh)
        return Episode(
            node=node, covering_task_id=None, planning_committed=False, filed_at=None,
            filing_due=False, filing_due_since=None, escalation_id=None, prior_task_ids=(), **fresh,
        )

    def start(self, node: str, base: Episode | None, replaced: Episode | None = None) -> _Step:
        ended = None
        if replaced is not None and replaced.live:
            ended = self.ending(replaced, EndReason.REPLACED)[1]
        return _Step(self.started(node, base), counted=True, started=True, ended=ended)

    def reproduced_on(self, episode: Episode) -> _Step:
        obs = self.obs
        heads = _with_head(episode.reproduced_heads, obs.head)
        last = episode.last_counted_started_at
        if last is not None and obs.started_at - last < self.policy.recheck:
            return _Step(replace(episode, reproduced_heads=heads))
        counted = replace(
            episode,
            counted_passes=(*episode.counted_passes, obs.pass_id),
            last_counted_started_at=obs.started_at,
            reproduced_heads=heads,
        )
        return _Step(counted, counted=True, extended=True)

    def red_again_covered(self, node: str, episode: Episode) -> _Step:
        verdict = self.reads.verdicts.get(node, CoveringVerdict.UNREADABLE)
        if verdict is CoveringVerdict.UNREADABLE:
            return _Step(episode, no_decision=True)
        if verdict is CoveringVerdict.DONE_STALE_HEAD:
            return _Step(episode, stale=True)
        if verdict in _BINDING_DROPPED or (verdict is CoveringVerdict.CLOSED and not episode.live):
            return self.start(node, _unbound(episode), replaced=episode)
        step = self.reproduced_on(episode) if episode.live else self.start(node, episode)
        if verdict is CoveringVerdict.DEFERRED_UNFINISHED and step.episode is not None:
            return replace(step, episode=self.due(step.episode))
        return step

    def due(self, episode: Episode) -> Episode:
        if episode.filing_due:
            return episode
        return replace(
            episode, filing_due=True, filing_due_since=self.now,
            reproduce=episode.reproduce or reproduce_for(self.obs, (episode.node,)),
        )

    def step(self, node: str, episode: Episode | None) -> _Step:
        shows = _shows(self.obs, node)
        if shows is _Shows.NOTHING:
            return _Step(episode)
        if shows is not _Shows.REPRODUCED:
            if episode is None or not episode.live:
                return _Step(episode)
            return self.ended(episode, EndReason(shows.value))
        if episode is None:
            return self.start(node, None)
        if episode.covering_task_id is not None:
            return self.red_again_covered(node, episode)
        if episode.live:
            return self.reproduced_on(episode)
        return self.start(node, episode)

    def crossed(self, before: Episode | None, step: _Step) -> bool:
        threshold = self.policy.persist_passes
        prior_count = 0 if step.started or before is None else len(before.counted_passes)
        return (
            step.counted
            and step.episode is not None
            and prior_count < threshold <= len(step.episode.counted_passes)
        )

    def unlanded(self, node: str, step: _Step) -> bool:
        episode = step.episode
        return (
            step.counted
            and episode is not None
            and episode.covering_task_id is not None
            and episode.escalation_id is None
            and episode.filed_at is not None
            and self.obs.started_at - episode.filed_at >= self.policy.red_unlanded
            and self.reads.verdicts.get(node) in _UNLANDED
        )

    def with_notes(self, episodes: dict[str, Episode], reproduced: Iterable[str]) -> None:
        reproduced = set(reproduced)
        for note in self.obs.notes:
            targets = reproduced if note.node is None else {note.node}
            for node in targets:
                episode = episodes.get(node)
                if episode is not None and episode.live:
                    episodes[node] = replace(episode, notes=(*episode.notes, note))

    def kept(self, episode: Episode) -> Episode | None:
        """§5.3's refresh rule, applied after the pass."""
        bound_live = (
            episode.covering_task_id is not None
            and episode.covering_task_id not in self.reads.terminal_task_ids
        )
        if episode.live or bound_live:
            return replace(episode, last_refreshed_at=self.now)
        if self.now - episode.last_refreshed_at >= EPISODE_RETENTION:
            return None
        return episode

    def command_fields(self, episodes: Mapping[str, Episode]) -> CommandRecord:
        obs = self.obs
        detect_green = obs.detect is not None and obs.detect.returncode == 0
        return replace(
            self.record,
            leg=obs.leg,
            episodes=episodes,
            rechecks={head: count for head, count in self.record.rechecks.items() if head == obs.head},
            last_pass_started_at=obs.started_at,
            last_pass_head=obs.head,
            last_pass_id=obs.pass_id,
            last_green_head=obs.head if detect_green else self.record.last_green_head,
            fix_task_priority=obs.fix_task_priority,
        )


def apply_pass(
    record: CommandRecord,
    observation: PassObservation,
    reads: PassReads,
    policy: LanePolicy,
    now: datetime,
) -> Applied:
    """The effect of one pass on one command's record: D5, D10, expiry, notes, refresh.  Pure."""
    if isinstance(observation.outcome, Outage):
        raise ValueError(f'{observation.command}: an outage pass goes through apply_outage, not apply_pass')
    applier = _PassApplier(record, observation, reads, policy, now)
    ended: list[EndedEpisode] = []
    episodes: dict[str, Episode] = {}
    for node, episode in record.episodes.items():
        if _expired(episode, observation.started_at, policy):
            episode, expiry = applier.ending(episode, EndReason.EXPIRED)
            ended.append(expiry)
        episodes[node] = episode
    steps: dict[str, _Step] = {}
    for node in sorted(set(episodes) | set(reproduced_nodes(observation))):
        before = episodes.get(node)
        step = applier.step(node, before)
        steps[node] = step
        if step.episode is not None:
            episodes[node] = step.episode
        if step.ended is not None:
            ended.append(step.ended)
    counted = [node for node, step in steps.items() if step.counted]
    applier.with_notes(episodes, reproduced_nodes(observation))
    crossing = [node for node, step in steps.items() if applier.crossed(record.episodes.get(node), step)]
    to_file = tuple(
        node for node in crossing
        if episodes[node].covering_task_id is None and not episodes[node].filing_due
    )
    unlanded = sorted({
        task_id
        for node, step in steps.items()
        if applier.unlanded(node, step) and (task_id := episodes[node].covering_task_id) is not None
    })
    kept = {node: refreshed for node, episode in episodes.items() if (refreshed := applier.kept(episode))}
    effect = PassEffect(
        tests_counted=tuple(counted),
        episodes_started=tuple(node for node, step in steps.items() if step.started),
        episodes_extended=tuple(node for node, step in steps.items() if step.extended),
        episodes_ended=tuple(ended),
        threshold_reached=tuple(crossing),
        stale_heads=tuple(node for node, step in steps.items() if step.stale),
        no_decision=tuple(node for node, step in steps.items() if step.no_decision),
        tasks_filed=(),
        escalations_raised=(),
    )
    return Applied(applier.command_fields(kept), effect, (to_file,) if to_file else (), tuple(unlanded))


def apply_outage(record: CommandRecord, observation: PassObservation) -> CommandRecord:
    """D5's outage row: every live episode is marked as having seen an outage pass, and nothing else in it changes.

    The pass fields stay at the last non-outage pass, and ``leg`` and
    ``fix_task_priority`` are set, being the command's identity rather than evidence.
    """
    if not isinstance(observation.outcome, Outage):
        raise ValueError(f'{observation.command}: apply_outage takes an outage pass only')
    episodes = {
        node: replace(episode, had_outage_pass=True) if episode.live else episode
        for node, episode in record.episodes.items()
    }
    return replace(record, leg=observation.leg, episodes=episodes, fix_task_priority=observation.fix_task_priority)


def with_probe(record: CommandRecord, probe: ProbeResult, at: datetime) -> CommandRecord:
    """The record after a probe answered *probe* at *at*, in a pass or on a tick."""
    return replace(record, last_probe_result=probe, last_probe_at=at)


def _recheck_fell_due(episode: Episode, record: CommandRecord, policy: LanePolicy, now: datetime) -> bool:
    last = episode.last_counted_started_at
    if not episode.live or last is None or len(episode.counted_passes) >= policy.persist_passes:
        return False
    if _expired(episode, now, policy):
        return False
    due_at = last + policy.recheck
    return due_at <= now and (record.last_pass_started_at is None or record.last_pass_started_at < due_at)


def recheck_due(record: CommandRecord, head: str, policy: LanePolicy, now: datetime) -> CommandRecord | None:
    """D12: the record with this head's recheck counted, or None when no recheck is due or the command is in an outage."""
    if record.in_outage:
        return None
    done = record.rechecks.get(head, 0)
    if record.leg != Leg.GENERIC.value or done >= policy.persist_passes - 1:
        return None
    if not any(_recheck_fell_due(episode, record, policy, now) for episode in record.episodes.values()):
        return None
    return replace(record, rechecks={**record.rechecks, head: done + 1})

