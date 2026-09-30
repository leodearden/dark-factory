"""Which merge lane an MCP-submitted merge request lands in (task 4888).

The whole lane decision for ``escalation/src/escalation/server.py::
merge_request``: validate a caller-supplied lane, apply the precedence
``lane argument > metadata.merge_lane > 'normal'``, and report WHICH source
won.  Pure — no I/O, no awaits, no server capture — so the server does only
wiring and the rule is testable without building a server, a queue, a
registry and a fake worker.

**The asymmetry is the point, and this paragraph is its SINGLE in-repo
statement.**  An INHERITED ``metadata.merge_lane`` fails OPEN: it was written
by a different actor at a different time, possibly by a machine writer, so an
unrecognised value normalises silently to ``'normal'`` and a lane resolution
can never fail a merge submission.  A CALLER-SUPPLIED lane fails LOUD: it is
live operator intent, the caller is present to be told, and silently
discarding it is the defect this module exists to fix — ``lane='higgh'`` must
not quietly downgrade a main-health hotfix.

Every other surface carrying this behaviour — ``escalation/src/escalation/
server.py::merge_request``'s docstring, ``docs/task-authoring.md`` §8, the
``'merge_lane'`` entry in ``shared/src/shared/task_metadata.py``, and both
test modules — states the CONTRACT and cites this paragraph rather than
restating the reason, on the same one-place rule that frozenset entry already
applies to its carrier census.  Eight copies of an argument drift; one does
not.

Both halves key on the SAME vocabulary.  ``MERGE_LANES`` and
``lane_for_task_metadata`` are imported from ``orchestrator.merge_queue``,
never copied: ``lane_for_task_metadata`` is the same function the
orchestrator's own submit path uses (``orchestrator/src/orchestrator/
workflow.py::TaskWorkflow::_submit_to_merge_queue``), so the two submit paths
cannot drift apart, and the loud-reject path tests membership against the
exact tuple the silent-normalise path keys on.  Deliberately NOT routed
through ``_normalize_lane``, whose defining behaviour — map anything
unrecognised to ``'normal'`` — is right for an inherited value and would
reproduce this task's defect one level up for a caller-supplied one.

Those imports are RUNTIME-ONLY, inside the function bodies, for the reason
``escalation/src/escalation/git_authority.py`` states for its own:
``escalation/pyproject.toml`` declares no orchestrator dependency (the
dependency runs the other way), so they resolve only because the escalation
server is hosted inside the orchestrator process.  Hoisting either to module
level would turn a caller's graceful degradation into an import-time crash
of the whole escalation package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

# The lane vocabulary spelled as a TYPE.  ``MERGE_LANES`` stays the single
# runtime source of truth — every check in this module keys on it — but it is
# ``tuple[str, ...]``, so it carries no static information.  Orchestrator
# spells the same Literal inline at ``merge_types.py::MergeRequest.lane`` and
# ``merge_queue.py::lane_for_task_metadata``; matching it here is what lets a
# resolved lane reach ``MergeRequest(lane=...)`` without a cast at the call
# site.
MergeLane = Literal['normal', 'high']


class InvalidMergeLane(ValueError):
    """A caller supplied a lane that is not in the merge-queue vocabulary.

    Carries the offending value and the vocabulary as STRUCTURED attributes
    rather than only as prose, so ``merge_request`` renders its
    ``{error, code, hint}`` reject envelope from them and the valid-lane list
    in the hint cannot drift from the tuple the check keyed on.

    A ``ValueError`` subclass so a caller that does not know this type still
    catches it.
    """

    def __init__(self, value: Any, valid_lanes: tuple[str, ...]) -> None:
        self.value = value
        self.valid_lanes = valid_lanes
        super().__init__(
            f'invalid merge lane {value!r}; expected one of {list(valid_lanes)}'
        )


@dataclass(frozen=True)
class LaneChoice:
    """A resolved lane PLUS which source it came from.

    ``'normal'`` alone cannot distinguish "the task asked for normal" from
    "nothing asked at all" — so the lane is never handed on without its
    provenance.  Copies the shape, and the reason for it, of
    ``escalation/src/escalation/git_authority.py::TaskMetadataResult``, where
    ``metadata == {}`` alone equally cannot distinguish a fact from a fault.

    ``source`` is one of ``'argument'`` (the caller passed ``lane=``),
    ``'task_metadata'`` (the task's ``merge_lane`` key was present), or
    ``'default'`` (neither).  It is echoed to the submitter as
    ``lane_source`` on the queued and attached responses, which is the audit
    trail for a ``'high'`` submission.
    """

    lane: MergeLane
    source: str


def validate_requested_lane(requested: Any) -> MergeLane | None:
    """Check a CALLER-supplied lane, or pass ``None`` through unchanged.

    ``None`` means "no argument supplied" and is not an error — it is what
    hands the decision to :func:`resolve_merge_lane`'s metadata arm.  Any
    other value must be a member of ``MERGE_LANES`` exactly; anything else
    raises :class:`InvalidMergeLane`.

    Pure and cheap by design, so ``merge_request`` can call it before paying
    for any git or task-metadata work: a typo costs the caller nothing but
    the round-trip that told them about it.  Non-``str`` values are reachable
    — the argument arrives off the MCP wire — and are rejected by the same
    membership test rather than coerced.
    """
    if requested is None:
        return None
    from orchestrator.merge_queue import MERGE_LANES  # type: ignore[reportMissingImports]

    if not isinstance(requested, str) or requested not in MERGE_LANES:
        raise InvalidMergeLane(requested, tuple(MERGE_LANES))
    # A membership test against a ``tuple[str, ...]`` cannot narrow to the
    # Literal, so the cast RECORDS the invariant the line above just
    # established rather than asserting an unchecked one.
    return cast(MergeLane, requested)


def resolve_merge_lane(
    *, requested: MergeLane | None, task_metadata: dict | None
) -> LaneChoice:
    """Apply ``requested > task_metadata['merge_lane'] > 'normal'``.

    *requested* must already have passed :func:`validate_requested_lane`; it
    wins whenever it is not ``None``, INCLUDING when it equals the default
    ``'normal'``.  That case is the one a truthiness test gets wrong, and it
    is the case that lets an operator hold a task with
    ``metadata.merge_lane='high'`` back to the normal lane.

    The metadata arm delegates to ``orchestrator.merge_queue::
    lane_for_task_metadata`` — see the module docstring for why that, and not
    ``_normalize_lane``, is the right normaliser here.  ``source`` is
    ``'task_metadata'`` iff the ``'merge_lane'`` key is PRESENT — presence,
    not truthiness, and not "the result differs from ``'normal'``" — because
    a task that asked for ``'normal'`` and a task that asked for nothing are
    different facts even though they get the same lane.
    """
    if requested is not None:
        return LaneChoice(lane=requested, source='argument')

    from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
        lane_for_task_metadata,
    )

    metadata = task_metadata or {}
    source = 'task_metadata' if 'merge_lane' in metadata else 'default'
    return LaneChoice(lane=lane_for_task_metadata(metadata), source=source)
