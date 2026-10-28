"""What one offline-lane pass tells the flake ledger.

PRD ``plans/offline-lane-red-path-prd.md`` D6 and section 5.8: the flake suppressions a
pass's outcome amounts to, at call site ``offline_lane``.  Pure; the I/O around it (the
host-pressure read, the run-record loads and the recorder calls) is
``orchestrator/src/orchestrator/offline_lane_episodes.py::LaneEngine``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    EvidenceKind,
    FailureEvidence,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
)
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Unconfirmable,
    inconclusive_reason,
)
from orchestrator.offline_lane_episode_state import PassObservation, unconfirmable_node
from orchestrator.offline_lane_records import FailureRecord, RunRecord

LANE_RUNNER = 'local'
NUMERIC_UNNAMED_CLEAR = 'numeric_leg_cleared_unnamed'
INCONCLUSIVE_UNEXPLAINED = 'inconclusive_unexplained'


def _failure(node: str, detect: RunRecord | None, confirm: RunRecord | None) -> FailureRecord | None:
    """*node*'s failure as the detect run recorded it, else as the confirm run did."""
    for run in (detect, confirm):
        if run is not None:
            for failure in run.failures:
                if failure.node == node:
                    return failure
    return None


def _named(nodes: Sequence[str], observation: PassObservation) -> tuple[str, ...]:
    """*nodes* without the command's pseudo-test and the ledger's sentinel, which name no test."""
    unnamed = {unconfirmable_node(observation.command), UNKNOWN_TEST_ID}
    return tuple(node for node in nodes if node not in unnamed)


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
