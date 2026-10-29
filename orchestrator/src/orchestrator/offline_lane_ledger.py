"""What one offline-lane pass tells the flake ledger.

PRD ``plans/offline-lane-red-path-prd.md`` D6 and section 5.8: the flake suppressions a
pass's outcome amounts to, at call site ``offline_lane``, the tests it saw recover under
the one-head rule, whether a cleared or recovered test owes flake debt, and the context
its de-flake task is written from.  Pure; the I/O around it (the
host-pressure read, the run-record loads and the recorder calls) is
``orchestrator/src/orchestrator/offline_lane_episodes.py::LaneEngine``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

from orchestrator.flake_ledger import (
    DEBT_CONTEXT_EVIDENCE_LIMIT,
    UNKNOWN_TEST_ID,
    DebtContext,
    DebtTrigger,
    EvidenceKind,
    FailureEvidence,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    RecentOccurrence,
)
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Unconfirmable,
    inconclusive_reason,
)
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    CoveringStatuses,
    EndedEpisode,
    EndReason,
    Episode,
    PassObservation,
    is_live_owner,
    reproduce_for,
    unconfirmable_node,
)
from orchestrator.offline_lane_records import FailureRecord, PassRecord, RunRecord

LANE_RUNNER = 'local'
NUMERIC_UNNAMED_CLEAR = 'numeric_leg_cleared_unnamed'
INCONCLUSIVE_UNEXPLAINED = 'inconclusive_unexplained'
LEGACY_WORKING_DIRECTORY = '.'


def _failure(node: str, detect: RunRecord | None, confirm: RunRecord | None) -> FailureRecord | None:
    """*node*'s failure as the detect run recorded it, else as the confirm run did."""
    for run in (detect, confirm):
        if run is not None:
            for failure in run.failures:
                if failure.node == node:
                    return failure
    return None


def _names_no_test(node: str, observation: PassObservation) -> bool:
    """The command's pseudo-test and the ledger's sentinel stand for a command, not a test."""
    return node in (unconfirmable_node(observation.command), UNKNOWN_TEST_ID)


def _named(nodes: Sequence[str], observation: PassObservation) -> tuple[str, ...]:
    return tuple(node for node in nodes if not _names_no_test(node, observation))


@dataclass(frozen=True)
class _PassStamp:
    """What every suppression of one pass shares."""

    observation: PassObservation
    record_dir: str
    psi_cpu_some10: float | None

    def evidence(self, nodes: Sequence[str]) -> dict[str, FailureEvidence] | None:
        observation = self.observation
        failures = {node: _failure(node, observation.detect, observation.confirm) for node in nodes}
        evidence = {
            node: FailureEvidence(EvidenceKind(failure.kind), failure.first_line, self.record_dir)
            for node, failure in failures.items()
            if failure is not None
        }
        return evidence or None

    def suppression(
        self, verdict: FlakeVerdict, nodes: Sequence[str], reason: str | None = None,
    ) -> FlakeSuppression:
        return FlakeSuppression(
            verdict=verdict,
            test_ids=tuple(nodes),
            observed_at=self.observation.started_at.isoformat(),
            call_site=FlakeCallSite.offline_lane,
            runner=LANE_RUNNER,
            psi_cpu_some10=self.psi_cpu_some10,
            unconfirmable_reason=reason,
            evidence=self.evidence(nodes),
        )


def _inconclusive_reason(node: str, observation: PassObservation) -> str:
    confirm = observation.confirm
    reason = None if confirm is None else inconclusive_reason(node, confirm)
    return INCONCLUSIVE_UNEXPLAINED if reason is None else reason.value


def _by_reason(nodes: Sequence[str], observation: PassObservation) -> dict[str, tuple[str, ...]]:
    """*nodes* grouped by why each is inconclusive, reasons in first-seen order."""
    groups: dict[str, tuple[str, ...]] = {}
    for node in nodes:
        reason = _inconclusive_reason(node, observation)
        groups[reason] = (*groups.get(reason, ()), node)
    return groups


def _confirmed(outcome: Confirmed, stamp: _PassStamp) -> Iterator[FlakeSuppression]:
    observation = stamp.observation
    reproduced = _named(outcome.reproduced, observation)
    cleared = _named(outcome.cleared, observation)
    if reproduced:
        yield stamp.suppression(FlakeVerdict.fails_in_isolation, reproduced)
    if cleared:
        yield stamp.suppression(FlakeVerdict.passes_in_isolation, cleared)
    for reason, nodes in _by_reason(_named(outcome.inconclusive, observation), observation).items():
        yield stamp.suppression(FlakeVerdict.unconfirmable, nodes, reason)
    if outcome.mode is ConfirmMode.LEG_NATIVE and not reproduced:
        yield stamp.suppression(FlakeVerdict.unconfirmable, (), NUMERIC_UNNAMED_CLEAR)


def pass_suppressions(
    observation: PassObservation, *, record_dir: str, psi_cpu_some10: float | None,
) -> tuple[FlakeSuppression, ...]:
    """The flake suppressions *observation*'s outcome amounts to; ``()`` for a green or outage pass."""
    stamp = _PassStamp(observation, record_dir, psi_cpu_some10)
    outcome = observation.outcome
    if isinstance(outcome, Unconfirmable):
        return (stamp.suppression(FlakeVerdict.unconfirmable, (), outcome.reason.value),)
    if isinstance(outcome, Confirmed):
        return tuple(_confirmed(outcome, stamp))
    return ()


@dataclass(frozen=True)
class Recovery:
    """A test that failed and passed at one head with no fix filed; *episode* is as it was before the pass ended it."""

    node: str
    evidence_head: str
    episode: Episode


def _evidence_head(episode: Episode, observation: PassObservation) -> str | None:
    """The head the episode both failed and passed at: this pass's, else the green head before it."""
    if observation.head in episode.reproduced_heads:
        return observation.head
    if episode.last_green_head in episode.reproduced_heads:
        return episode.last_green_head
    return None


def _recovery(episode: Episode, observation: PassObservation) -> Recovery | None:
    if _names_no_test(episode.node, observation):
        return None
    if episode.covering_task_id is not None or episode.had_outage_pass:
        return None
    head = _evidence_head(episode, observation)
    return None if head is None else Recovery(episode.node, head, episode)


def recoveries(
    before: CommandRecord, ended: Sequence[EndedEpisode], observation: PassObservation,
) -> tuple[Recovery, ...]:
    """The episodes *observation* ended by a pass that recover under the one-head rule, in *ended*'s order."""
    found = (
        _recovery(before.episodes[episode.node], observation)
        for episode in ended
        if episode.reason == EndReason.PASSED.value
    )
    return tuple(recovery for recovery in found if recovery is not None)


class DebtDecision(StrEnum):
    """Whether a cleared or recovered test is offered to the ledger as owing flake debt."""

    ELIGIBLE = 'eligible'
    COVERED = 'covered'
    UNDECIDED = 'undecided'
    NEVER = 'never'


def debt_decision(
    node: str, command: str, episode: Episode | None, covering: CoveringStatuses,
) -> DebtDecision:
    """ELIGIBLE unless a live fix task covers *node*: COVERED when its read status is open or deferred, UNDECIDED when unreadable."""
    if node in (unconfirmable_node(command), UNKNOWN_TEST_ID):
        return DebtDecision.NEVER
    task_id = None if episode is None else episode.covering_task_id
    if episode is None or task_id is None or episode.binding_swallowed:
        return DebtDecision.ELIGIBLE
    if covering.error is not None:
        return DebtDecision.UNDECIDED
    if is_live_owner(covering.statuses.get(task_id)):
        return DebtDecision.COVERED
    return DebtDecision.ELIGIBLE


def _distinct(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def debt_context(
    node: str,
    trigger: DebtTrigger,
    observation: PassObservation,
    episode: Episode | None,
    occurrences: Sequence[RecentOccurrence],
) -> DebtContext:
    """Why *node* owes debt after *observation*: the episode's history, then this pass."""
    cmd = observation.lane_command
    heads = () if episode is None else episode.reproduced_heads
    pass_ids = () if episode is None else episode.counted_passes
    return DebtContext(
        call_site=FlakeCallSite.offline_lane,
        trigger=trigger,
        command=observation.command,
        working_directory=LEGACY_WORKING_DIRECTORY if cmd is None else cmd.cwd,
        reproduce=reproduce_for(observation, (node,)),
        heads=_distinct((*heads, observation.head)),
        pass_ids=_distinct((*pass_ids, observation.pass_id)),
        evidence=tuple(occurrences[-DEBT_CONTEXT_EVIDENCE_LIMIT:]),
    )


def _occurrence(node: str, seen: PassRecord | PassObservation) -> RecentOccurrence | None:
    failure = _failure(node, seen.detect, seen.confirm)
    if failure is None:
        return None
    return RecentOccurrence(seen.pass_id, seen.head, EvidenceKind(failure.kind), failure.first_line)


def recent_occurrences(
    node: str, earlier: Sequence[PassRecord], current: PassObservation | None,
) -> tuple[RecentOccurrence, ...]:
    """*node*'s failures in *earlier*, oldest first, then *current*'s; the most recent ``DEBT_CONTEXT_EVIDENCE_LIMIT``."""
    passes: list[PassRecord | PassObservation] = [*earlier, *(() if current is None else (current,))]
    found = [occurrence for seen in passes if (occurrence := _occurrence(node, seen)) is not None]
    return tuple(found[-DEBT_CONTEXT_EVIDENCE_LIMIT:])
