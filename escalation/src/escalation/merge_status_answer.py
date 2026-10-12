"""The merge_status response model and the D2 provenance invariants it cannot be built without.

Contract: ``plans/merge-status-durable-non-landed-prd.md`` D1 (outcome vs
epistemic states), D2 (history never sets an outcome state; invariants I1 and
I3), D4 (``last_outcome`` is an enrichment, not a tier) and D8 (``hint`` is
derived from ``reason`` through one table).  The tier walk that produces these
answers is ``escalation/src/escalation/server.py::merge_status``.

I2 holds in its intent, not its letter: no ``state`` is ever derived from a
prior-run row, which I1 already guarantees.  A git-proven ``done`` beside a
prior-run ``done`` in ``last_outcome`` is corroboration, not leakage
(``escalation/tests/test_merge_status_tiers.py::TestGitOutranksCheapTiers``).
"""
from __future__ import annotations

import dataclasses
import enum
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from shared.merge_state import EPISTEMIC_STATES, TERMINAL_STATES, MergeReason, MergeState

__all__ = [
    'MergeStatusSource',
    'PRESENT_TRUTH_SOURCES',
    'map_terminal_state',
    'hint_for',
    'LastOutcome',
    'MergeStatusAnswer',
]


class MergeStatusSource(enum.StrEnum):
    """Which tier produced a merge_status answer — the response's ``source``."""

    live = 'live'
    event_store = 'event_store'
    git = 'git'
    journal = 'journal'
    event_store_cross_run = 'event_store_cross_run'
    none = 'none'


#: The only tiers whose evidence may set an OUTCOME state (I1).
PRESENT_TRUTH_SOURCES = frozenset(
    {MergeStatusSource.live, MergeStatusSource.event_store, MergeStatusSource.git}
)


def map_terminal_state(raw: str) -> MergeState:
    """Collapse a raw finalize status (a ``MergeSubmitStatus`` value, or ``abandoned``) to the poll vocabulary."""
    if raw in ('done', 'done_wip_recovery', 'already_merged'):
        return MergeState.done
    if raw == 'conflict':
        return MergeState.conflict
    if raw == 'abandoned':
        return MergeState.abandoned
    if raw == 'superseded':
        return MergeState.superseded
    return MergeState.blocked


_HINTS: Mapping[MergeReason, str] = types.MappingProxyType({
    MergeReason.never_submitted: (
        'No live, journal or event-store record exists and git shows no landing '
        'on main: this merge was never submitted. Submit it with merge_request.'
    ),
    MergeReason.prior_run_terminal: (
        "Only a previous orchestrator run's record exists (see last_outcome); it "
        "is history, not this run's outcome. Resubmit with merge_request if the "
        'work is still not on main.'
    ),
    MergeReason.journaled_not_live: (
        'The durable merge journal holds this request, but the live worker '
        'snapshot does not show it. Possibly no merge worker is reachable from '
        'this server, or the request is between journal and queue, for example '
        'awaiting startup recovery. Do not resubmit; the journal still owns this '
        'request.'
    ),
    MergeReason.coalesce_train_id: (
        'This id names a coalesced train and has no outcome row of its own; '
        "last_outcome is its newest member's record. Poll each member by its "
        'branch or task_id for that member\'s outcome.'
    ),
    MergeReason.landing_unattributable: (
        "Git shows this branch's work on main but could not confirm it as this "
        'task\'s landing (attribution or effect unverified). Do not resubmit; '
        'verify the landing on main.'
    ),
    MergeReason.probe_failed: (
        'A probe raised while resolving this merge (source names the tier); the '
        'answer is unknown, not negative. Retry merge_status; do not resubmit '
        'on this answer.'
    ),
})


def hint_for(reason: MergeReason) -> str:
    return _HINTS[reason]


_EPISTEMIC_SOURCE: Mapping[MergeState, MergeStatusSource] = types.MappingProxyType({
    MergeState.no_record: MergeStatusSource.none,
    MergeState.stale_record: MergeStatusSource.event_store_cross_run,
    MergeState.journaled: MergeStatusSource.journal,
})

_ANSWER_OWNED_KEYS = frozenset({'state', 'request_id', 'generation', 'source', 'hint', 'last_outcome'})

#: Every answer's ``generation`` until request generations exist (PRD P4).
_RENDERED_GENERATION = 1


@dataclass(frozen=True)
class LastOutcome:
    """A durable finalize row, labelled with the run that wrote it (D4)."""

    state: MergeState
    run_id: str
    finished_at: str
    is_current_run: bool
    reason: str | None
    merge_sha: str | None
    superseded_by: str | None
    request_id: str | None
    branch: str | None

    def __post_init__(self) -> None:
        if self.state not in TERMINAL_STATES:
            raise ValueError(
                f'LastOutcome.state must be a TERMINAL_STATES member, got {self.state!r}'
            )

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> LastOutcome:
        return cls(
            state=map_terminal_state(row['state']),
            run_id=row['run_id'],
            finished_at=row['finished_at'],
            is_current_run=row['is_current_run'],
            reason=row.get('reason'),
            merge_sha=row.get('merge_sha'),
            superseded_by=row.get('superseded_by'),
            request_id=row.get('request_id'),
            branch=row.get('branch'),
        )

    def to_wire(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class MergeStatusAnswer:
    """One merge_status answer; construction is where every D2 invariant is checked."""

    state: MergeState | str
    source: MergeStatusSource
    request_id: str | None
    reason: MergeReason | None
    fields: Mapping[str, Any]
    last_outcome: LastOutcome | None = None

    @classmethod
    def outcome(cls, source: MergeStatusSource, fields: Mapping[str, Any]) -> MergeStatusAnswer:
        extras = dict(fields)
        state = extras.pop('state')
        request_id = extras.pop('request_id', None)
        generation = extras.pop('generation', _RENDERED_GENERATION)
        if generation != _RENDERED_GENERATION:
            raise ValueError(
                f'merge_status renders generation {_RENDERED_GENERATION} only; '
                f'a {source.value!r} tier reported generation {generation!r}'
            )
        return cls(
            state=state, source=source, request_id=request_id, reason=None,
            fields=types.MappingProxyType(extras),
        )

    @classmethod
    def epistemic(
        cls,
        state: MergeState,
        reason: MergeReason | None,
        request_id: str | None,
        source: MergeStatusSource | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> MergeStatusAnswer:
        resolved_source = source if source is not None else _EPISTEMIC_SOURCE.get(state)
        if resolved_source is None:
            raise ValueError(
                f'epistemic state {state!r} has no paired source; pass the failing tier'
            )
        return cls(
            state=state, source=resolved_source,
            request_id=request_id, reason=reason,
            fields=types.MappingProxyType(dict(fields or {})),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.source, MergeStatusSource):
            raise ValueError(
                f'merge_status answer for state {self.state!r} needs a MergeStatusSource, '
                f'got {self.source!r}'
            )
        epistemic = self.state in EPISTEMIC_STATES
        forbidden = _ANSWER_OWNED_KEYS | ({'reason'} if epistemic else frozenset())
        shadowed = sorted(forbidden & self.fields.keys())
        if shadowed:
            raise ValueError(f'merge_status answer fields may not shadow rendered keys: {shadowed}')
        if not epistemic:
            if self.source not in PRESENT_TRUTH_SOURCES:
                raise ValueError(
                    f'I1: outcome state {self.state!r} requires a present-truth source, '
                    f'got source {self.source.value!r}'
                )
            if self.reason is not None:
                raise ValueError(
                    f'outcome state {self.state!r} carries no MergeReason, got {self.reason!r}'
                )
            return
        if self.reason is None:
            raise ValueError(f'epistemic state {self.state!r} requires a MergeReason')
        if (self.state == MergeState.unknown) != (self.reason is MergeReason.probe_failed):
            raise ValueError(
                f'I3: state unknown iff reason probe_failed, got state {self.state!r} '
                f'with reason {self.reason!r}'
            )
        paired = _EPISTEMIC_SOURCE.get(MergeState(self.state))
        if paired is not None and self.source is not paired:
            raise ValueError(
                f'epistemic state {self.state!r} pairs with source {paired.value!r}, '
                f'got source {self.source.value!r}'
            )

    def with_last_outcome(self, last_outcome: LastOutcome | None) -> MergeStatusAnswer:
        return dataclasses.replace(self, last_outcome=last_outcome)

    def to_response(self) -> dict[str, Any]:
        response: dict[str, Any] = {
            'state': self.state,
            'request_id': self.request_id,
            'generation': _RENDERED_GENERATION,
            'source': self.source.value,
            **self.fields,
        }
        if self.reason is not None:
            response['reason'] = self.reason
            response['hint'] = hint_for(self.reason)
        response['last_outcome'] = (
            self.last_outcome.to_wire() if self.last_outcome is not None else None
        )
        return response
