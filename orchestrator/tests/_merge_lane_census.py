"""One reading of the merge lane's public placement surface, shared by its tests.

``SpeculativeMergeWorker.snapshot()`` is the worker's own answer to "what is
queued, where, and in what order".  It enumerates the lane buffers in priority
order (``high`` before ``normal``) and then anything still undrained on the
outer queue, stamping every entry with its ``task_id``, ``request_id``,
``state``, ``lane`` and ``position`` — so it reads WIDER than any single lane
buffer: a red tip that the round it failed requeued is visible here, at the
tail, which the buffer alone could not report at all.

The readers live here rather than once per test file because the claim they
make is one claim.  Three near-copies of it had already appeared across the
γ5 group, each restating the same semantics in its own docstring, which is the
SPOT smell (`docs/code-quality.md` heuristic 11) and leaves the family free to
drift apart one file at a time.

Not every lane test is on this module yet: a file that still reads placement
off ``worker._lane_buffers`` directly is making the same claim a second way,
and converging it is for whichever group owns that file — this module is the
destination, not a completed migration.
"""
from __future__ import annotations

from orchestrator.merge_queue import SpeculativeMergeWorker

#: The wire state ``snapshot()`` stamps on anything still waiting to be merged.
#: ``ItemLifecycleState.QUEUED`` and ``LANE_BUFFERED`` share this one string
#: (``merge_queue.py::_REGISTRY_STATE_TO_WIRE``), so a test that must tell
#: those two apart still reads the lifecycle registry directly.
QUEUED = 'queued'

#: Lane names in the priority order ``snapshot()`` enumerates them.
LANES = ('high', 'normal')


def queued_entries(
    worker: SpeculativeMergeWorker, lane: str = 'normal',
) -> list[dict]:
    """Every ``snapshot()`` entry still queued in *lane*, head-of-line first."""
    return [
        entry for entry in worker.snapshot()['entries']
        if entry['state'] == QUEUED and entry['lane'] == lane
    ]


def queued_in_lane(
    worker: SpeculativeMergeWorker, lane: str = 'normal',
) -> list[str]:
    """Task ids queued in *lane*, in queue order.

    ORDER, not just membership: submission order inside a lane is the next
    round's ``chain_snapshot``, so a membership-only check cannot see a
    reorder.  Task ids are the readable form and the right key for an ORDER
    claim; when the point is that the very same request OBJECTS are still
    queued, use :func:`queued_request_ids` — a task id cannot tell a request
    apart from a fresh one carrying the same id.
    """
    return [entry['task_id'] for entry in queued_entries(worker, lane)]


def queued_request_ids(
    worker: SpeculativeMergeWorker, lane: str = 'normal',
) -> list[str]:
    """Request ids queued in *lane*, in queue order — the IDENTITY reading.

    ``request_id`` is minted per ``MergeRequest`` object, so this says "these
    exact requests, still queued, still in this order" where
    :func:`queued_in_lane` can only say "requests for these tasks".  That gap
    is the coalesce/duplicate-submission failure mode: a request object
    silently replaced by a fresh one for the same task is invisible to a
    task-id comparison and visible here.

    The one place the two readings genuinely diverge is a test that FORCES two
    distinct objects to share a request_id (the journal-recovery twin) — there
    identity is the thing under test, not the background assumption.
    """
    return [entry['request_id'] for entry in queued_entries(worker, lane)]


def lanes_by_task(worker: SpeculativeMergeWorker) -> dict[str, str]:
    """``task_id -> lane`` for everything queued, from ONE census.

    The membership reading — the weaker claim that still needs saying: the
    item is on the queue at all, in the lane it was submitted to, rather than
    having been silently promoted or dropped.  Built once so a per-task loop
    pays for one ``snapshot()`` rather than one (or two) per task id;
    ``snapshot()`` walks the in-flight set, the redispatch set, the verifier
    queue, both lane buffers, the outer queue and the lifecycle registry, so
    that difference is a whole pipeline census per lookup.

    First entry wins, and ``snapshot()`` lists ``high`` before ``normal``, so a
    task somehow queued in both reports ``high`` — the same answer a
    priority-ordered lane scan gives.
    """
    lanes: dict[str, str] = {}
    for entry in worker.snapshot()['entries']:
        if entry['state'] == QUEUED:
            lanes.setdefault(entry['task_id'], entry['lane'])
    return lanes


def census_for(
    worker: SpeculativeMergeWorker, request_id: str,
) -> list[tuple[str, str]]:
    """``(state, lane)`` for every ``snapshot()`` entry carrying *request_id*.

    One call pins both facts a duplicate-submission test asserts: a request
    wrongly appended to a lane buffer is sourced from snapshot's CONTAINER
    section and therefore dropped from its registry section
    (``merge_queue.py::SpeculativeMergeWorker.snapshot``), surfacing as
    ``queued`` rather than its registry state; and a divergent second pipeline
    item under the same request_id would surface as a second tuple.
    """
    return [
        (entry['state'], entry['lane'])
        for entry in worker.snapshot()['entries']
        if entry['request_id'] == request_id
    ]
