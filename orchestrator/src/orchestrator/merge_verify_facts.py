"""The typed dispatch facts a ``merge_verify`` row carries.

Rationale: ``plans/merge-lane-policy-prd.md`` D10.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Collection
from enum import StrEnum


class RunnerReason(StrEnum):
    """Why a merge verify ran on the host it ran on."""

    #: Policy ``prefer_remote`` took a free remote.
    PREFER_REMOTE_FREE = 'prefer_remote_free'
    #: Fell to local because a remote was busy.
    REMOTE_BUSY = 'remote_busy'
    #: Fell to local because every remote was quarantined or PARKED.
    REMOTE_QUARANTINED = 'remote_quarantined'
    #: Local by rule: no remote configured, or a local-only caller
    #: (shadow, history leg, leaseless gate).
    LOCAL_ONLY_RULE = 'local_only_rule'
    #: A gate re-verify on the head's held host.  Historical: produced
    #: before task 8525 made that re-verify a dispatch of the rebased head.
    GATE_ON_HEAD_HOST = 'gate_on_head_host'
    #: An in-place retry on the host attempt 0 held.
    RETRY_HELD_HOST = 'retry_held_host'
    #: Policy ``prefer_local`` took the free local host.
    PREFER_LOCAL_FREE = 'prefer_local_free'
    #: Policy ``prefer_local`` overflowed to a remote.
    LOCAL_BUSY = 'local_busy'


class BaseRelation(StrEnum):
    """What the verified merge commit was built on."""

    #: The current tip of main.
    MAIN = 'main'
    #: An in-flight predecessor's merge commit.
    STACKED = 'stacked'
    #: An in-flight head's gate-rebased commit.
    REBASED_HEAD = 'rebased_head'
    #: Neither main nor anything on the in-flight frontier.
    STALE = 'stale'


class ChainDepthBasis(StrEnum):
    """What the deep policy chose a verify's chain depth from."""

    #: The first depth-table row the rolling per-item pass estimate meets.
    PASS_ESTIMATE = 'pass_estimate'
    #: The window held no verdicts, so the first row applied.
    NO_ESTIMATE = 'no_estimate'


@dataclasses.dataclass(frozen=True)
class ChainDepthReason:
    """Why one dispatch verified the chain depth it did.

    *row* is the ``(min_pass_estimate, depth)`` row that applied, if any;
    *queue_depth* counts the dispatching request and the chainable ones
    queued behind it.
    """

    basis: ChainDepthBasis
    pass_estimate: float | None
    row: tuple[float, int] | None
    queue_depth: int

    def wire(self) -> dict[str, object]:
        return {
            'basis': self.basis.value,
            'pass_estimate': self.pass_estimate,
            'row': None if self.row is None else list(self.row),
            'queue_depth': self.queue_depth,
        }


@dataclasses.dataclass(frozen=True)
class ChainMembers:
    """The requests one verify covers, as built: the dispatching one, then its chain links."""

    request_ids: tuple[str, ...]
    task_ids: tuple[str, ...]

    def event_fields(self) -> dict[str, list[str]]:
        return {
            'chain_request_ids': list(self.request_ids),
            'chain_task_ids': list(self.task_ids),
        }


@dataclasses.dataclass(frozen=True)
class MergeVerifyDispatchFacts:
    """The allocation decision and base a single verify dispatch ran under,
    the requests it covers, and why the deep policy chose its depth.

    The two ``chain_*`` member keys are written only when *members* is known.
    ``chain_depth_reason`` is always written, null when the dispatch never
    consulted the deep policy.
    """

    runner_reason: RunnerReason | None = None
    base_relation: BaseRelation | None = None
    members: ChainMembers | None = None
    chain_depth_reason: ChainDepthReason | None = None

    def event_fields(self) -> dict[str, object]:
        return {
            'runner_reason': _wire(self.runner_reason),
            'base_relation': _wire(self.base_relation),
            'chain_depth_reason': (
                None if self.chain_depth_reason is None else self.chain_depth_reason.wire()
            ),
            **_member_fields(self.members),
        }

    def covering(self, members: ChainMembers) -> MergeVerifyDispatchFacts:
        return dataclasses.replace(self, members=members)

    def with_chain_depth_reason(self, reason: ChainDepthReason | None) -> MergeVerifyDispatchFacts:
        return dataclasses.replace(self, chain_depth_reason=reason)

    def as_retry(self) -> MergeVerifyDispatchFacts:
        return dataclasses.replace(self, runner_reason=RunnerReason.RETRY_HELD_HOST)

    @classmethod
    def local_only(cls) -> MergeVerifyDispatchFacts:
        return cls(runner_reason=RunnerReason.LOCAL_ONLY_RULE)


def _wire(member: StrEnum | None) -> str | None:
    return None if member is None else member.value


def _member_fields(members: ChainMembers | None) -> dict[str, list[str]]:
    return {} if members is None else members.event_fields()


def classify_base_relation(
    base_sha: str,
    *,
    main_sha: str | None,
    stacked_on: Collection[str],
    rebased_heads: Collection[str],
) -> BaseRelation | None:
    """Relate *base_sha* to main and the in-flight frontier.

    ``None`` when main could not be read and the base is on no frontier
    commit, so the relation is unknown rather than stale.
    """
    base = base_sha.strip()
    if main_sha is not None and base == main_sha.strip():
        return BaseRelation.MAIN
    if base in {sha.strip() for sha in rebased_heads}:
        return BaseRelation.REBASED_HEAD
    if base in {sha.strip() for sha in stacked_on}:
        return BaseRelation.STACKED
    return None if main_sha is None else BaseRelation.STALE
