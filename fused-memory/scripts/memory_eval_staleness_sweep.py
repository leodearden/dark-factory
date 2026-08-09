#!/usr/bin/env python3
"""E4 staleness sweep — is the corpus still surfacing things it superseded?

``docs/prds/memory-eval-program.md`` §5 leaf γ. Where leaf β
(``memory_eval_retrieval_probe.py``, E1) asks whether retrieval returns the
RIGHT thing for a committed registry of topics, this leaf asks the corpus
about itself: which of the supersession/parent/correction pointers it carries
still resolve, whether a superseded entry still outranks the entry that
replaced it, and whether entries assert live task state for tasks that have
since gone terminal.

**What it measures** (four metrics across three families, all owned by this
leaf):

======================================  ==========  ==================
metric_id                               kind        direction
======================================  ==========  ==================
``superseded-still-surfacing``          count       higher_is_worse
``dangling-pointers``                   count       higher_is_worse
``successor-pointer-present``           tripwire    (rule (a) is already
                                                    directional)
``task-terminal-staleness``             count       higher_is_worse
======================================  ==========  ==================

``dangling-pointers`` and ``successor-pointer-present`` are the two spellings
leaf β's docstring and leaf α's committed exemplars reserve for THIS leaf, and
they are not redundant with each other: the count feeds α's Poisson
count-shift trend (is the corpus accumulating dangling pointers?), the
tripwire feeds α's grandfathered structural rule with the ratchet (did THIS
supersession edge newly break, or did a previously-broken one get fixed?). One
aggregate count cannot express per-edge grandfathering; one tripwire cannot
express a trend over ``parent_id``/``corrects`` targets that have no stable
per-item identity.

β's ``superseded-above-successor`` is NOT reused here. That metric is
registry-declared-pair shaped and lives under β's ``e1-retrieval-health``;
this leaf's family (1) is corpus-DISCOVERED from live ``supersedes`` metadata.
Different populations, different exposure, different eval_id.

**This script never writes to the live corpus and never evaluates a limit.**

Both halves are load-bearing:

- *Never writes.* There is no ``--apply`` band, no delete/add/update call and
  no write path anywhere in this module. D8's read-only runner pattern
  (``audit_duplicate_memories.py`` ``_run``) is copied minus every mutation.
  The guarantee is asserted as BEHAVIOUR in the tests — the sweep is driven
  against a service double whose every write method raises — rather than
  merely claimed in this docstring.
- *Never evaluates a limit.* Per G6/M2 every threshold, tolerance,
  grandfather set and alarm lives in leaf α's limits evaluator. No pass rate,
  bound or verdict appears in this script or in any of its tests.
  ``--scan-limit`` is a resource cap whose firing is DISCLOSED into the
  artifact, not a tuned bound.

**Schema home (D2).** Every artifact model, validator, path helper, stamp
format and atomic writer comes from :mod:`shared.memory_eval_metrics`. Nothing
in that contract is re-declared here.

**Pointer parsing (D7 / INV-5).** All three keys in :data:`POINTER_KEYS` are
parsed through the ONE imported
:func:`fused_memory.memory_metadata.normalize_supersedes`. ``parent_id`` and
``corrects`` carry the identical ``None``/scalar/list-of-UUIDs ambiguity
``supersedes`` does, and the failure mode is the one task 3112 recorded: a
bare ``for target in value`` over a 36-character UUID *string* iterates it into
36 single characters, none of which resolve, manufacturing a systematic false
dangling-pointer report. A second local normalizer for the other two keys
would be exactly the INV-5 violation D7 forbids while re-introducing the bug
that helper was written to prevent.

**Zero exposure is ABSENT, not zero (D1).** A family that measured nothing
emits NO metric rather than a ``value=0 / n=0`` datapoint — a fabricated
"nothing wrong here" entering leaf α's baseline window is worse than a gap in
it. ``parent_id`` makes this live rather than hypothetical: it is in
``RESERVED_VOCABULARY_KEYS`` but the metadata census measures zero live
records carrying it. The report names every family explicitly so an absence
can never be misread as health.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import logging
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger('memory_eval_staleness_sweep')

# ---------------------------------------------------------------------------
# Pinned M1 contract vocabulary
#
# These spellings are COPIED from leaf α's committed consumer-side exemplars
# and from leaf β's docstring, which reserves `dangling-pointers` and
# `successor-pointer-present` for this leaf by name. The limits evaluator
# joins a run to its baseline window BY metric_id, so a different spelling
# would not fail loudly — it would make the metric invisible to the evaluator
# and to the dashboard, which is strictly worse than a crash.
# ---------------------------------------------------------------------------

EVAL_ID = 'e4-staleness-sweep'
"""Also the artifact directory segment (``<root>/<eval_id>/metrics-<STAMP>.json``).

This leaf's OWN eval_id, deliberately not β's ``e1-retrieval-health``, even
though the PRD describes E1+E4 as "one scheduled retrieval-health runner".
That is a SCHEDULING statement (leaf ε invokes both), not a one-artifact one:
``write_metric_series`` atomically OVERWRITES
``<root>/<eval_id>/metrics-<STAMP>.json``, and ``RUN_STAMP_ENV_VAR`` exists
precisely so several runners in one logical run share a stamp — so a shared
eval_id would make this leaf silently clobber β's artifact on every scheduled
run, and vice versa. Separate ids also keep α's baseline windows independent,
since the evaluator joins a window BY metric_id WITHIN one eval_id series.
Overridable via ``--eval-id`` (δ's precedent).
"""

METRIC_SUPERSEDED_STILL_SURFACING = 'superseded-still-surfacing'
"""Count of superseded entries that outranked their successor. n = comparable pairs."""

METRIC_DANGLING_POINTERS = 'dangling-pointers'
"""Count of pointer targets that do not resolve. n = pointers examined."""

METRIC_SUCCESSOR_POINTER_PRESENT = 'successor-pointer-present'
"""Tripwire (M2 rule a). One item per ``supersedes`` edge, keyed by content."""

METRIC_TASK_TERMINAL_STALENESS = 'task-terminal-staleness'
"""Count of entries asserting live state for a terminal task. n = entries
referencing a terminal task at all."""

POINTER_KEYS: tuple[str, ...] = ('supersedes', 'parent_id', 'corrects')
"""The metadata keys whose values are memory-id pointers.

All three go through :func:`normalize_supersedes` — see the module docstring
for why a second parser for the other two would re-introduce 3112's bug.
``parent_id`` is reserved vocabulary with zero live population today; it is
swept anyway so the first genuine use is measured rather than discovered.
"""

TRIPWIRE_ITEM_PREFIX = 's-'
"""``TripwireItem.item_key`` shape. A STORED key (α's grandfather set persists
it), not a display string."""

_DEFAULT_METRICS_ROOT = str(Path(__file__).resolve().parent.parent / 'data' / 'memory-evals')
"""``fused-memory/data/memory-evals`` (M1 §3), resolved off THIS file.

Not off the cwd: a scheduled run's working directory is not guaranteed, and a
relative default would scatter artifacts wherever the scheduler happened to
start — invisible to the limits evaluator, which scans one root.
"""

_WHITESPACE_RE = re.compile(r'\s+')


def content_key(text: str) -> str:
    """A stable 16-hex-char identity for a memory's *text*.

    ``sha256(...).hexdigest()[:16]`` — the ``shared/task_metadata.py``
    convention, matching leaf β's ``content_key`` so the two runners describe
    the same entry the same way.

    Whitespace is normalised (surrounding stripped, internal runs collapsed to
    one space) BEFORE hashing so that re-indentation, a wrapped line or a
    trailing newline picked up in transit does not read as a different entry.
    Nothing else is normalised: case and punctuation are content, and folding
    them would make two genuinely different claims collide.
    """
    normalized = _WHITESPACE_RE.sub(' ', text).strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)
"""What a Mem0 point id looks like. Used ONLY to classify a pointer member as
well-formed for the disclosure — never to drop one (see
:func:`malformed_pointer_refs`)."""


# ---------------------------------------------------------------------------
# Family (2a) — pointer extraction
#
# THE only place this runner reads a pointer key's raw metadata value, and it
# reads all three through the one imported normalize_supersedes(). See the
# module docstring for why a second parser here would re-introduce 3112's
# char-iteration bug.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointerRef:
    """One declared edge: *source_id* ``--[key]-->`` *target*.

    *target* is deliberately typed ``Any``: ``normalize_supersedes`` never
    drops or coerces a member, so a malformed one (short hex, an int, a
    ``None`` inside a list) arrives here intact and is reported by name rather
    than silently discarded — a dropped member is a silently discarded
    supersession edge.

    *source_content* rides along because the ``successor-pointer-present``
    tripwire keys its items by CONTENT hash, not by the source UUID (D5,
    "content-hash item keys (UUID rot)"): leaf α grandfathers by item_key, so
    a UUID that rotated under re-consolidation would read as a brand-new
    failure and fire a false alarm. Carrying it on the ref is what lets
    :func:`successor_pointer_items` stay pure over the refs alone.
    """

    source_id: str
    key: str
    target: Any
    source_content: str = ''


def pointer_targets(record: dict) -> list[PointerRef]:
    """Every pointer edge *record*'s metadata declares, in :data:`POINTER_KEYS` order.

    All three keys are parsed by the ONE imported
    :func:`fused_memory.memory_metadata.normalize_supersedes`. D7 mandates the
    helper for ``supersedes``; ``parent_id`` and ``corrects`` carry the
    identical ``None``/scalar/list ambiguity, so writing a second local parse
    for them would be exactly the INV-5 violation D7 forbids *while*
    re-introducing the bug the helper was written to prevent — iterating a
    36-character UUID string into 36 single characters, none of which resolve,
    manufacturing a systematic false dangling-pointer report (task 3112).

    Iteration follows :data:`POINTER_KEYS`, not the metadata dict's insertion
    order: the latter is an artifact of however the record happened to be
    written, and letting it leak through would make the emitted ordering — and
    therefore the report an operator diffs between runs — depend on nothing
    meaningful.

    Malformed members are RETAINED, matching the helper's no-drop contract;
    :func:`malformed_pointer_refs` is how they are surfaced.
    """
    from fused_memory.memory_metadata import normalize_supersedes  # noqa: PLC0415

    metadata = record.get('metadata') or {}
    source_id = str(record.get('id') or '')
    source_content = record.get('content') or ''
    refs: list[PointerRef] = []
    for key in POINTER_KEYS:
        for target in normalize_supersedes(metadata.get(key)):
            refs.append(PointerRef(
                source_id=source_id,
                key=key,
                target=target,
                source_content=source_content,
            ))
    return refs


def is_readable_target(target: Any) -> bool:
    """True when *target* can be handed to a Qdrant point-id read.

    ONE predicate behind both the disclosure (:func:`malformed_pointer_refs`)
    and the read plan (:func:`unique_pointer_targets`), because they are the
    same question and two spellings of it drift. The drift is not theoretical:
    a looser read plan hands Qdrant's ``retrieve`` an id it rejects outright,
    and since ``retrieve`` takes a LIST of ids, one malformed pointer anywhere
    in the corpus would fail the read and take down a sweep that exists to
    report on exactly that kind of damage.
    """
    return isinstance(target, str) and bool(_UUID_RE.match(target))


def malformed_pointer_refs(refs: list[PointerRef]) -> list[PointerRef]:
    """The refs whose target is not a memory-id-shaped string.

    Reported, never dropped. The census still counts a malformed member as an
    unresolved pointer — a value that cannot name a memory is a broken edge,
    not a missing measurement — but an operator reading a dangling count needs
    to know how much of it is "the target is gone" versus "the pointer was
    never writable in the first place", because the two have different fixes.
    """
    return [ref for ref in refs if not is_readable_target(ref.target)]


def unique_pointer_targets(refs: list[PointerRef]) -> list[str]:
    """The distinct memory-id-shaped targets in *refs*, in first-seen order.

    The read plan for :func:`resolve_pointer_targets`: a target cited by
    several sources costs ONE live read, not one per citation. Order is
    first-seen rather than sorted so a run's store-access sequence is
    reproducible from the scan alone.

    Targets failing :func:`is_readable_target` are excluded — they cannot be
    handed to a point-id read at all. They are not thereby forgiven:
    :func:`dangling_census` still counts them unresolved off their absence
    from the resolution map, and :func:`malformed_pointer_refs` names them.
    """
    seen: dict[str, None] = {}
    for ref in refs:
        if is_readable_target(ref.target):
            seen.setdefault(ref.target, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Family (2b) — the dangling census
#
# PURE. The live `get_memory_by_id` reads happen in the async band and arrive
# here as an already-fetched resolution map, which is what keeps every
# assertion about this family in the merge lane.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DanglingCensus:
    """What the pointer sweep found, with the unresolved edges named.

    ``by_key`` carries a row ONLY for a key with live pointers. A key with no
    population (``parent_id`` measures zero live records today) gets no row
    rather than a zero one, for the same reason ``build_series`` omits a
    zero-exposure metric: a fabricated "nothing wrong here" is worse than a
    gap, and the report names every family so the absence cannot read as
    health.
    """

    examined: int
    resolved: int
    unresolved: int
    by_key: dict[str, dict[str, int]]
    unresolved_refs: list[PointerRef]


def dangling_census(
    refs: list[PointerRef], resolution: dict[str, bool],
) -> DanglingCensus:
    """Count *refs* against *resolution* (``target id -> does it exist``).

    *resolution* is supplied by the caller because corroboration has to happen
    against the LIVE store (INV-3), not against the scrolled snapshot already
    in hand: the scroll is capped and category-scoped, so a target outside it
    would read as dangling when it exists, and a census that confirmed itself
    from its own scan would be measuring its own scan depth.

    A target the map does not carry counts as UNRESOLVED. That is the loud
    direction: a pointer the resolver never reached is exactly the case a
    permissive default would paper over.

    ``examined`` counts POINTERS, not distinct targets — the same target cited
    by three sources is three edges that could each be broken, and each keeps
    its own attribution in ``unresolved_refs``. The de-duplication that saves
    the redundant store READ lives in :func:`unique_pointer_targets`, where it
    costs no accounting.
    """
    by_key: dict[str, dict[str, int]] = {}
    unresolved_refs: list[PointerRef] = []
    resolved = 0
    for ref in refs:
        row = by_key.setdefault(ref.key, {'examined': 0, 'resolved': 0, 'unresolved': 0})
        row['examined'] += 1
        target = ref.target if isinstance(ref.target, str) else None
        if target is not None and resolution.get(target, False):
            row['resolved'] += 1
            resolved += 1
        else:
            row['unresolved'] += 1
            unresolved_refs.append(ref)
    return DanglingCensus(
        examined=len(refs),
        resolved=resolved,
        unresolved=len(unresolved_refs),
        by_key=by_key,
        unresolved_refs=unresolved_refs,
    )


# ---------------------------------------------------------------------------
# Family (2c) — the successor-pointer tripwire (supersedes edges ONLY)
#
# The same population the count above covers, sliced to the one key with a
# stable per-item identity. alpha's rule (a) grandfathers by item_key and
# ratchets: a newly-broken edge alarms, a repaired one is released silently.
# That is expressible only per edge, which is why this metric exists ALONGSIDE
# the aggregate count rather than instead of it.
# ---------------------------------------------------------------------------

def _tripwire_item_key(ref: PointerRef) -> str:
    """``s-<source content hash>-<target discriminator>``.

    Content-derived, never the raw source UUID (D5, "content-hash item keys
    (UUID rot)"). Leaf α persists this key in its grandfather set, so a source
    id that rotated under re-consolidation would present a previously-known
    failure as a brand-new one and fire a false alarm; the content survives
    that rotation.

    The target discriminator is what keeps two edges out of ONE source
    distinct. It is a hash rather than the target id itself for the same
    reason the source half is: an item_key is a stored contract, and burying a
    raw UUID in it would make the key rot with the target too.
    """
    target_digest = hashlib.sha256(repr(ref.target).encode('utf-8')).hexdigest()[:8]
    return f'{TRIPWIRE_ITEM_PREFIX}{content_key(ref.source_content)}-{target_digest}'


def successor_pointer_items(refs: list[PointerRef], resolution: dict[str, bool]) -> list:
    """One :class:`shared.memory_eval_metrics.TripwireItem` per ``supersedes`` edge.

    ``parent_id``/``corrects`` refs are deliberately excluded: they are
    measured by the ``dangling-pointers`` COUNT, whose Poisson trend needs no
    per-item identity. A tripwire over them could not be grandfathered, since
    those targets have no stable item key to ratchet on.

    Returns the shared model directly so the tripwire's ``n == len(items)``
    and ``value == failing count`` invariants are satisfied by construction
    when :func:`build_series` assembles the metric.

    Sorted by item_key and de-duplicated. A duplicate key means two edges with
    identical source CONTENT pointing at the same target — the same claim
    written twice, which is one corpus fact, not two — and the schema rejects
    a tripwire whose ``n`` disagrees with its item count, so the collapse has
    to happen here rather than being discovered at emit time.
    """
    from shared.memory_eval_metrics import TripwireItem  # noqa: PLC0415

    by_key: dict[str, bool] = {}
    for ref in refs:
        if ref.key != 'supersedes':
            continue
        target = ref.target if isinstance(ref.target, str) else None
        passed = bool(target is not None and resolution.get(target, False))
        item_key = _tripwire_item_key(ref)
        # A repeated key is the same edge; AND rather than overwrite, so a
        # collision can only ever make the item stricter, never quietly
        # upgrade a broken edge to passing.
        by_key[item_key] = by_key.get(item_key, True) and passed
    return [
        TripwireItem(item_key=key, passed=passed)
        for key, passed in sorted(by_key.items())
    ]


# ---------------------------------------------------------------------------
# Family (1) — is a superseded entry still outranking its successor?
#
# Corpus-DISCOVERED pairs (from live `supersedes` metadata), not registry
# declared ones. beta's `superseded-above-successor` measures the registry
# population under a different eval_id; this measures whatever the corpus
# actually claims about itself today.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfacingRecord:
    """One both-present pair and where each member landed.

    Both ids and both ranks: a bare count tells an operator that something
    inverted but not which pair to go and look at.
    """

    successor_id: str
    superseded_id: str
    successor_rank: int
    superseded_rank: int

    @property
    def inverted(self) -> bool:
        return self.superseded_rank < self.successor_rank


@dataclass(frozen=True)
class SurfacingObservation:
    """Family (1)'s exposure, event count and detail.

    ``pairs_comparable`` is the metric's ``n`` and ``still_surfacing`` is its
    value; ``records`` holds every both-present pair (the superseded entry
    coming back AT ALL is worth reading even when it came back below its
    successor) and ``inversions`` the subset that counted.
    """

    pairs_comparable: int
    still_surfacing: int
    records: tuple[SurfacingRecord, ...]
    inversions: tuple[SurfacingRecord, ...]


EMPTY_SURFACING = SurfacingObservation(
    pairs_comparable=0, still_surfacing=0, records=(), inversions=(),
)
"""The zero-exposure observation. ``build_series`` emits no metric for it."""


def rank_index(ranked_ids: list[str]) -> dict[str, int]:
    """``{memory id: 1-based first position}`` over the list the store returned.

    Position, not score: equal ``relevance_score`` values resolve by the order
    the store handed back rather than by an unstable re-sort, so two runs over
    the same list produce the same count. A tie that flapped would look to the
    evaluator like a real regression.

    ``setdefault`` keeps the FIRST occurrence: a store that returned the same
    id twice has still returned it at its best rank, and taking the later one
    would report a ranking problem that is really a duplication one.
    """
    first_rank: dict[str, int] = {}
    for index, memory_id in enumerate(ranked_ids, start=1):
        if memory_id:
            first_rank.setdefault(memory_id, index)
    return first_rank


def superseded_surfacing(
    pairs: list[tuple[str, str]],
    ranked_ids: list[str],
    *,
    ranks: dict[str, int] | None = None,
) -> SurfacingObservation:
    """Score ``(successor_id, superseded_id)`` *pairs* against one ranked list.

    **Both-present-only exposure.** A pair with just one member in
    *ranked_ids* contributes nothing to the count AND nothing to
    ``pairs_comparable``. An inversion can only ever fire on a both-present
    pair, so a half-present one carries no possibility of an event; charging
    it to the exposure anyway makes the rate move for the wrong reason. It is
    also already measured elsewhere — an absent successor is a findability
    question — and counting it here as well would charge one defect against
    two metrics, inflating any downstream trend by double-weighting a single
    fix.

    That is why the metric's ``n`` is the comparable-pair count rather than
    the number of ``supersedes`` edges discovered: if retrieval improves so
    that forty pairs come back both-present instead of four, the event count
    can rise while a discovered-pair ``n`` stays pinned, and leaf α's Poisson
    tail test would read a retrieval IMPROVEMENT as a rate regression.

    Pass *ranks* (a :func:`rank_index` of the same list) when the caller
    already has it.
    """
    index = rank_index(ranked_ids) if ranks is None else ranks
    records: list[SurfacingRecord] = []
    for successor_id, superseded_id in pairs:
        successor_rank = index.get(successor_id)
        superseded_rank = index.get(superseded_id)
        if successor_rank is None or superseded_rank is None:
            continue
        records.append(SurfacingRecord(
            successor_id=successor_id,
            superseded_id=superseded_id,
            successor_rank=successor_rank,
            superseded_rank=superseded_rank,
        ))
    inversions = tuple(record for record in records if record.inverted)
    return SurfacingObservation(
        pairs_comparable=len(records),
        still_surfacing=len(inversions),
        records=tuple(records),
        inversions=inversions,
    )


def combine_surfacing(observations: list[SurfacingObservation]) -> SurfacingObservation:
    """Fold per-query observations into the one the artifact reports.

    Each pair is scored against the ranked list of ITS OWN query (the search
    is derived from the successor's content), so a run produces one
    observation per pair rather than one for the whole sweep. Exposure and
    events are additive across those queries; concatenating the details keeps
    every pair nameable in the report.
    """
    records = tuple(r for obs in observations for r in obs.records)
    inversions = tuple(r for obs in observations for r in obs.inversions)
    return SurfacingObservation(
        pairs_comparable=sum(obs.pairs_comparable for obs in observations),
        still_surfacing=sum(obs.still_surfacing for obs in observations),
        records=records,
        inversions=inversions,
    )


# ---------------------------------------------------------------------------
# Family (3) — entries asserting LIVE task state for a terminal task
#
# Every predicate this family needs already has a single home in
# fused_memory.reconciliation.task_filter. Nothing here re-derives one:
# TASK_REF_RE decides what a task reference is, and
# frames_live_task_status_as_current_fact decides what counts as asserting
# live state. See referenced_task_ids/terminal_staleness for why that matters
# beyond INV-5 tidiness.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StalenessRecord:
    """One entry framing a terminal task's state as a current fact."""

    record_id: str
    task_id: str
    status: str


@dataclass(frozen=True)
class StalenessObservation:
    """Family (3)'s exposure, event count and detail.

    ``entries_referencing_terminal`` is the metric's ``n`` — entries that
    reference a terminal task AT ALL, not the whole corpus. An entry that
    never mentions a terminal task cannot make this claim, so charging it to
    the exposure would make the rate track corpus growth instead of staleness.
    """

    entries_referencing_terminal: int
    stale: int
    records: tuple[StalenessRecord, ...]


def referenced_task_ids(record: dict) -> set[str]:
    """Task ids *record* references, from its content AND its ``task_id`` metadata.

    ``TASK_REF_RE`` (``task N`` / ``df N`` / ``#N``, never a bare number) is
    the reconciliation filter's own definition of a task reference, imported
    rather than restated: a second spelling here would drift from the one the
    recon stages enforce, and the two would disagree about the same sentence.

    The metadata half matters at corpus scale — the census measures 18,850
    live records carrying ``task_id`` — and is normalised to ``str`` because
    it is written as both an int and a string.
    """
    from fused_memory.reconciliation.task_filter import TASK_REF_RE  # noqa: PLC0415

    ids = set(TASK_REF_RE.findall(record.get('content') or ''))
    raw = (record.get('metadata') or {}).get('task_id')
    if isinstance(raw, (str, int)) and not isinstance(raw, bool) and str(raw).strip():
        ids.add(str(raw).strip())
    return ids


def terminal_staleness(records: list[dict], terminal_task_ids) -> StalenessObservation:
    """Entries asserting live state for a task in *terminal_task_ids*.

    *terminal_task_ids* accepts either a ``Mapping`` of ``id -> status`` (what
    :func:`fetch_terminal_task_ids` returns, and what carries the status into
    the detail) or a bare set of ids, which reports ``'terminal'``.

    The live-state judgement is delegated ENTIRELY to
    ``task_filter.frames_live_task_status_as_current_fact``. That is INV-5,
    but it is also a performance requirement: ``POINT_IN_TIME_CHECK_RE``'s two
    ``(?=.*...)`` lookaheads under ``re.DOTALL`` are quadratic in content
    length (measured 2.2 ms at 424 characters against 794 ms at 4240), and the
    helper prefilters with the lookahead-free ``LIVE_TASK_STATUS_RE`` so the
    expensive pattern only ever runs on the rare entry that already looks
    live. Calling the two regexes directly here — in either order — would
    re-derive the predicate AND silently drop the ordering that keeps a
    corpus-scale scan tractable.

    Exposure counts ENTRIES, not (entry, task) pairs: one entry naming three
    terminal tasks is one chance to be stale. The detail still names every
    (entry, task) pair, because that is what an operator has to go and fix.

    Terminal is ``shared.task_statuses.TERMINAL`` — ``done`` and ``cancelled``
    only. ``deferred`` is NOT terminal, and an entry asserting live state for
    a deferred task is not making a false claim.
    """
    from fused_memory.reconciliation import task_filter  # noqa: PLC0415

    statuses = terminal_task_ids if isinstance(terminal_task_ids, dict) else {}
    terminal = set(terminal_task_ids)
    if not terminal:
        return StalenessObservation(entries_referencing_terminal=0, stale=0, records=())

    exposed = 0
    stale = 0
    found: list[StalenessRecord] = []
    for record in records:
        hits = sorted(referenced_task_ids(record) & terminal)
        if not hits:
            continue
        exposed += 1
        if not task_filter.frames_live_task_status_as_current_fact(record.get('content') or ''):
            continue
        stale += 1
        record_id = str(record.get('id') or '')
        found.extend(
            StalenessRecord(
                record_id=record_id,
                task_id=task_id,
                status=str(statuses.get(task_id, 'terminal')),
            )
            for task_id in hits
        )
    return StalenessObservation(
        entries_referencing_terminal=exposed, stale=stale, records=tuple(found),
    )


# ---------------------------------------------------------------------------
# M1 series assembly
#
# Every model, validator and path helper comes from shared.memory_eval_metrics
# (D2). Nothing about the artifact shape is re-declared here.
# ---------------------------------------------------------------------------

def pinned_metric_ids() -> tuple[str, ...]:
    """Every metric_id a run of this leaf is expected to emit.

    THE list this leaf owns, in one place, so "which family went unmeasured"
    is answerable by comparing against it rather than by remembering what
    :func:`build_series` can emit.
    """
    return (
        METRIC_SUPERSEDED_STILL_SURFACING,
        METRIC_DANGLING_POINTERS,
        METRIC_SUCCESSOR_POINTER_PRESENT,
        METRIC_TASK_TERMINAL_STALENESS,
    )


def _count(metric_id: str, value: int, exposure: int, *, details_path: str | None = None):
    """A count Metric, or ``None`` when the family measured nothing.

    ``None`` rather than a ``value=0 / n=0`` datapoint: a count kind has no
    more claim to a zero-trial measurement than a proportion does, and
    emitting one would put a fabricated "nothing wrong here" into the baseline
    window leaf α computes limits from. Absent is the honest signal, and
    :func:`metric_families_not_measured` puts the absence in the report — a
    metric that vanishes without explanation reads as healthy.

    Every count in this leaf is ``higher_is_worse``: more dangling pointers,
    more superseded entries outranking their successors and more stale
    liveness claims are each unambiguously the regression direction.
    """
    from shared.memory_eval_metrics import Metric  # noqa: PLC0415

    if exposure <= 0:
        return None
    return Metric(
        metric_id=metric_id,
        kind='count',
        value=float(value),
        n=exposure,
        direction='higher_is_worse',
        details_path=details_path,
    )


def _disclosure_counts(
    census: DanglingCensus,
    surfacing: SurfacingObservation,
    staleness: StalenessObservation,
    malformed: int,
    refs: list[PointerRef],
) -> dict[str, int]:
    """The narrowings that must ride along INSIDE the machine-readable artifact.

    Reporting them only in prose would be a silent cap for every consumer that
    reads the JSON — which is all of them. ``corpus.counts`` is free-form
    category -> size by design (its docstring: the bucket vocabulary is not
    that schema's to own), so it is where a per-run disclosure belongs.

    The per-key rows matter because ``dangling-pointers`` aggregates three
    keys into one number: without them a consumer cannot tell a corpus that
    grew its ``corrects`` population from one that started breaking
    ``supersedes`` edges, and those have different fixes. ``malformed`` splits
    "the target is gone" from "the pointer was never writable", which the
    aggregate also cannot express.

    ``pointer_targets_unique_reads`` counts DISTINCT READABLE targets — one
    live ``get_memory_by_id`` per unique target, which is exactly the read plan
    :func:`resolve_pointer_targets` executes. It is deliberately not derived
    from ``census.examined``/``census.resolved``: those count EDGES (see
    :func:`dangling_census`, "``examined`` counts POINTERS, not distinct
    targets"), so a target cited by three sources contributes three to them and
    one read to the store. An operator reading a read-cost field wants reads,
    and a field that instead grew with citation density would drift upward as
    the corpus cross-references itself — a trend leaf α would read off this
    artifact as real. Unreadable targets are excluded for the same reason
    :func:`unique_pointer_targets` excludes them: no read is ever issued for
    one. They stay disclosed under ``pointer_refs_malformed``, so nothing that
    was visible becomes invisible — it simply stops being counted as a read.
    """
    counts: dict[str, int] = {
        'pointer_refs_malformed': malformed,
        'pointer_targets_unique_reads': len(unique_pointer_targets(refs)),
        'surfacing_pairs_observed': len(surfacing.records),
        'task_terminal_entry_task_pairs': len(staleness.records),
    }
    for key, row in sorted(census.by_key.items()):
        for field_name, value in sorted(row.items()):
            counts[f'pointers_{key}_{field_name}'] = value
    return counts


def build_series(
    census: DanglingCensus,
    tripwire_items: list,
    surfacing: SurfacingObservation,
    staleness: StalenessObservation,
    corpus_counts: dict[str, int],
    project_id: str,
    stamp: str,
    *,
    refs: list[PointerRef],
    eval_id: str = EVAL_ID,
):
    """Assemble the M1 metric series for one sweep run.

    Emits at most the four metrics this leaf owns, in the pinned vocabulary.
    β's ``superseded-above-successor`` and its topic metrics are that leaf's
    and never appear here.

    *refs* is the scan's full ref list, threaded through for the read-cost
    disclosure alone — no metric is computed from it. It is required rather
    than defaulted because a caller that forgot it would silently disclose a
    read plan of zero, and this runner exists to disclose.

    The result is validated before it is returned, so an aggregation bug
    surfaces in this runner rather than in leaf α's evaluator — the M1
    "rejected at emit time, not read time" guarantee applied to the producer's
    own arithmetic.
    """
    from shared.memory_eval_metrics import (  # noqa: PLC0415
        SCHEMA_VERSION,
        Corpus,
        Metric,
        MetricSeries,
        report_artifact_path,
        validate_metric_series,
    )

    # The report FILENAME, not its absolute path: the artifact directory gets
    # copied and served (the dashboard reads it as plain files), and an
    # absolute path from this machine would be a dangling pointer there.
    details_path = report_artifact_path('.', eval_id, stamp).name

    metrics: list[Any] = []
    for metric in (
        _count(
            METRIC_SUPERSEDED_STILL_SURFACING,
            surfacing.still_surfacing, surfacing.pairs_comparable,
            details_path=details_path,
        ),
        _count(
            METRIC_DANGLING_POINTERS,
            census.unresolved, census.examined,
            details_path=details_path,
        ),
    ):
        if metric is not None:
            metrics.append(metric)

    # A tripwire with no items is not a passing tripwire — the schema rejects
    # an empty items list outright, and rightly: "no supersedes edge in this
    # corpus" is an absence of evidence, not a clean structural check.
    if tripwire_items:
        metrics.append(Metric(
            metric_id=METRIC_SUCCESSOR_POINTER_PRESENT,
            kind='tripwire',
            value=float(sum(1 for item in tripwire_items if not item.passed)),
            n=len(tripwire_items),
            items=list(tripwire_items),
            details_path=details_path,
        ))

    terminal = _count(
        METRIC_TASK_TERMINAL_STALENESS,
        staleness.stale, staleness.entries_referencing_terminal,
        details_path=details_path,
    )
    if terminal is not None:
        metrics.append(terminal)

    counts = dict(corpus_counts)
    disclosures = _disclosure_counts(
        census, surfacing, staleness,
        len(malformed_pointer_refs(census.unresolved_refs)), refs,
    )
    for key, value in disclosures.items():
        if key in counts:
            raise ValueError(
                f'corpus_counts key {key!r} collides with a run disclosure this '
                'runner computes. Rename the caller-supplied key: silently '
                'overwriting either one would hide a narrowing.'
            )
        counts[key] = value

    series = MetricSeries(
        schema_version=SCHEMA_VERSION,
        eval_id=eval_id,
        run_stamp=stamp,
        corpus=Corpus(project_id=project_id, counts=counts),
        metrics=metrics,
    )
    validate_metric_series(series)
    return series


def metric_families_not_measured(series) -> list[str]:
    """Pinned metric ids *series* does not carry, in the pinned order.

    Every family here declines to emit rather than emit a zero-exposure
    datapoint, because a fabricated trial in leaf α's baseline window is worse
    than a gap in it. But an absent metric is not an error to the evaluator
    either — it joins by metric_id and simply stops trending what is missing.
    So the absence is named HERE, in the run's own report, where the
    alternative is a metric that silently stops existing and reads to a human
    as one that had nothing to report.
    """
    present = {metric.metric_id for metric in series.metrics}
    return [metric_id for metric_id in pinned_metric_ids() if metric_id not in present]


# ---------------------------------------------------------------------------
# The async I/O band
#
# Deliberately thin: fetch, normalise, hand off to the pure functions above.
# Every judgement this sweep makes lives up there, which is what keeps the
# whole of it assertable in the merge lane without a store. Nothing here
# calls a mutating method — the read-only guarantee is a property of this
# band's call list, and the test double enforces it by raising on writes.
# ---------------------------------------------------------------------------

SWEEP_CATEGORIES: tuple[str, ...] = (
    'procedural_knowledge',
    'preferences_and_norms',
    'observations_and_summaries',
)
"""The Mem0-primary categories a sweep enumerates, δ's ``_ALL_CATEGORIES``.

Mem0-primary only: ``scroll_by_metadata`` is a Qdrant payload scroll, so the
three Graphiti-primary categories are not reachable through it at all. The
report names that scope rather than leaving a near-zero count to be misread
as an empty graph.
"""

_CONTENT_KEYS: tuple[str, ...] = ('data', 'memory', 'content')
"""Payload keys tried in order for a scrolled record's text.

``data`` is the canonical scroll-payload key and is tried first; ``memory``
is a search-layer key that can appear stale on a scroll payload; ``content``
is a defensive third. Same order and same reasoning as δ's ``fetch_memories``
— the two runners read the same payloads and must agree about which string is
the memory.
"""


async def fetch_pointer_records(
    memory: Any,
    project_id: str,
    *,
    categories: Iterable[str] = SWEEP_CATEGORIES,
    scan_limit: int = 5000,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Enumerate *project_id*'s Mem0 records, with the scan's own limits disclosed.

    Issues ONE ``scroll_by_metadata`` per category. That is the enumeration,
    not a missed optimisation: the primitive builds an AND-equality filter,
    has no OR, and REJECTS an empty filter dict outright ("use get_all()"), so
    there is no single whole-corpus call to widen into.

    Each raw record is normalised to the ``{'id', 'content', 'metadata'}``
    shape every pure function above takes, so a live run drives exactly the
    code path the merge-lane tests cover.

    Args:
        memory: Live (or mock) MemoryService. Only ``mem0.scroll_by_metadata``
            is touched.
        project_id: Project scope to scan.
        categories: Categories to enumerate (default: :data:`SWEEP_CATEGORIES`).
        scan_limit: Max points PER CATEGORY.

    Returns:
        ``(records, scan_stats)``. *scan_stats* maps each category to
        ``{'scanned': n, 'truncated': 0|1}``, counted PER CATEGORY so a cap
        firing on one is never reported as a clean scan of the whole corpus.
        ``truncated`` is an int so it can be summed into ``corpus.counts``
        directly. This disclosure is the price of not paginating: the census
        below is honest about being a capped sample, rather than silently
        presenting one as a census.

    Raises:
        TimeoutError: A Qdrant read timeout propagates rather than degrading
            to an empty list, so a timed-out scan is never mistaken for an
            empty corpus — the same no-silent-fail posture the primitive
            itself adopted.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    scope = Scope(project_id=project_id)
    records: list[dict[str, Any]] = []
    scan_stats: dict[str, dict[str, int]] = {}

    for category in categories:
        # A repeated category (δ hit this too) would append a second copy of
        # every record under the SAME id, doubling every pointer edge the
        # census then counts. scan_stats already holds that category's
        # numbers, so skipping loses nothing from the disclosure.
        if category in scan_stats:
            continue
        raw_records = await memory.mem0.scroll_by_metadata(
            scope, {'category': category}, limit=scan_limit,
        ) or []
        scan_stats[category] = {
            'scanned': len(raw_records),
            'truncated': int(len(raw_records) >= scan_limit),
        }
        for raw in raw_records:
            payload = raw.get('metadata') or {}
            content = ''
            for key in _CONTENT_KEYS:
                value = payload.get(key)
                if isinstance(value, str) and value:
                    content = value
                    break
            records.append({
                'id': str(raw.get('id') or ''),
                'content': content,
                'metadata': payload,
            })
    return records, scan_stats


async def resolve_pointer_targets(
    memory: Any,
    project_id: str,
    refs: list[PointerRef],
) -> dict[str, bool]:
    """Corroborate every pointer target in *refs* against the live store.

    One ``get_memory_by_id`` per UNIQUE target (see
    :func:`unique_pointer_targets`): a target cited by several sources costs
    one read, but the read itself is never skipped. Marking a target resolved
    because it happened to appear in the scroll already in hand would make the
    census self-confirming — the scroll is capped and category-scoped, so a
    target outside it exists yet would read as dangling. INV-3: corroborate
    against ground truth, never against the snapshot that raised the question.

    Malformed targets are absent from the result rather than mapped to
    ``False``: they were never handed to a read, so this map says nothing
    about them. :func:`dangling_census` treats that absence as unresolved and
    :func:`malformed_pointer_refs` names them, which keeps "the target is
    gone" and "the pointer was never writable" separable in the report.

    Raises:
        TimeoutError: Propagated, never folded into the unresolved count. The
            primitive distinguishes "genuinely absent" from "backend timed
            out" precisely so this caller can too; reporting a blip as a
            dangling pointer would fabricate a defect and fire an α alarm on
            an infrastructure hiccup.
    """
    resolution: dict[str, bool] = {}
    for target in unique_pointer_targets(refs):
        resolution[target] = await memory.get_memory_by_id(project_id, target) is not None
    return resolution


async def fetch_surfacing_ranks(
    memory: Any,
    project_id: str,
    refs: list[PointerRef],
    *,
    limit: int = 10,
) -> SurfacingObservation:
    """Score every ``supersedes`` edge in *refs* against its own search.

    One search per edge, using the SUCCESSOR's content as the query: the
    question this family asks is "when the corpus is asked about what the
    successor says, does the entry it replaced come back above it?", and only
    a query derived from the successor's own text poses it. A shared or
    synthetic query would measure a different thing for every pair.

    Each edge is therefore scored against the ranked list of its own query and
    the per-edge observations are folded by :func:`combine_surfacing`. Rank is
    list POSITION, so equal scores resolve by returned order and two runs over
    the same list produce the same count — a tie that flapped would read to
    leaf α as a real regression.

    An edge whose successor content is empty is skipped rather than searched
    with an empty query: an empty query's ranking is arbitrary, and scoring
    against it would manufacture inversions from noise.
    """
    observations: list[SurfacingObservation] = []
    for ref in refs:
        if ref.key != 'supersedes' or not isinstance(ref.target, str):
            continue
        query = (ref.source_content or '').strip()
        if not query:
            continue
        results = await memory.search(query, project_id=project_id, limit=limit)
        ranked_ids = [str(getattr(r, 'id', '') or '') for r in (results or [])]
        observations.append(
            superseded_surfacing([(ref.source_id, ref.target)], ranked_ids),
        )
    return combine_surfacing(observations)


@dataclass(frozen=True)
class TerminalTaskJoin:
    """Terminal task statuses for family (3), or a NAMED reason there are none.

    The two are not the same fact and must not collapse into one empty
    mapping. A run that could not reach the task backend omits the metric for
    lack of exposure — and so does a run that reached it and found no terminal
    task referenced anywhere. Only one of those means the family was measured
    and came back clean; ``skipped_reason`` is what lets the report say which.
    """

    statuses: dict[str, str]
    """Terminal task id → its terminal status, for the detail records."""

    skipped_reason: str | None = None

    @property
    def available(self) -> bool:
        """True when the join actually ran, whatever it found."""
        return self.skipped_reason is None


async def fetch_terminal_task_ids(config: Any) -> TerminalTaskJoin:
    """Read terminal task statuses through the SQLite task backend.

    Terminal is ``shared.task_statuses.TERMINAL`` — ``done`` and ``cancelled``
    only. ``deferred`` is deliberately NOT terminal: a deferred task can still
    be picked up, so an entry asserting live state about one is not stale.

    Follows ``sweep_orphan_flag_markers._resolve_terminal_task_ids``'s
    start/get_statuses/close-in-``finally`` shape, and its status-only read —
    ``get_statuses`` never decodes a metadata column, which matters at a task
    tree this size.

    Ids are normalised to ``str`` because :func:`referenced_task_ids` yields
    strings (``TASK_REF_RE`` matches text): an int key surviving here would
    silently never intersect, and the family would report clean.

    A missing or failing backend degrades to a NAMED skip rather than an
    exception: this is a read-only reporting sweep and the other three
    families are still measurable without the task tree. It does not degrade
    to a silent empty set — see :class:`TerminalTaskJoin`.
    """
    from shared.task_statuses import TERMINAL  # noqa: PLC0415

    taskmaster = getattr(config, 'taskmaster', None)
    if taskmaster is None:
        return TerminalTaskJoin(
            statuses={},
            skipped_reason=(
                'taskmaster is not configured, so no task statuses could be read. '
                'The task-terminal-staleness family was NOT measured — its absence '
                'from the metrics is a gap, not a clean result.'
            ),
        )

    from fused_memory.backends.sqlite_task_backend import (  # noqa: PLC0415
        SqliteTaskBackend,
    )

    try:
        backend = SqliteTaskBackend(taskmaster)
        await backend.start()
        try:
            statuses = await backend.get_statuses(taskmaster.project_root)
        finally:
            await backend.close()
    except Exception as exc:
        # exc_info so a genuine wiring failure stays distinguishable in the
        # logs from the unconfigured no-op above, which never reaches here.
        logger.warning(
            'memory_eval_staleness_sweep: terminal task status read failed; '
            'the task-terminal-staleness family will be reported as skipped.',
            exc_info=True,
        )
        return TerminalTaskJoin(
            statuses={},
            skipped_reason=(
                f'reading task statuses failed ({type(exc).__name__}: {exc}). '
                'The task-terminal-staleness family was NOT measured — its absence '
                'from the metrics is a gap, not a clean result.'
            ),
        )

    return TerminalTaskJoin(statuses={
        str(task_id): str(status)
        for task_id, status in (statuses or {}).items()
        if status in TERMINAL
    })


# ---------------------------------------------------------------------------
# The human-readable report
#
# The shared module renders the metric TABLE. Everything here is what only
# this runner knows: which families measured nothing, which scans were capped,
# and which pointers were never writable in the first place.
#
# Nothing here adjudicates. No bound, no ratchet and no pass/fail verdict
# appears in this output — all of that belongs to leaf α's evaluator (G6), and
# a second home for it would drift from the first without anyone noticing.
# ---------------------------------------------------------------------------

_MAX_NAMED = 20
"""Detail rows printed per section before the remainder is counted instead.

The count of what was elided is always printed, so a long tail is visible as a
number even when it is not enumerated. A section that silently stopped at
twenty would read as a complete list of twenty.
"""


@dataclass(frozen=True)
class ReportSection:
    """One block of the report, under a STABLE machine key.

    The key is never rendered — it exists so a caller (and a test) can ask
    WHICH blocks a report carries, and in what order, without matching on the
    English inside them. Prose is the part of this module expected to be
    reworded; a check that keys on prose constrains wording rather than
    behaviour, while keying on structure keeps the disclosure guarantees
    falsifiable: a section that stops being emitted, or is emitted on the
    wrong run, fails — and a copy edit does not. β's convention, for the same
    reason.
    """

    key: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)


def _elided(rows: list[str], total: int) -> list[str]:
    """*rows* plus a line naming how many more there were."""
    if total > len(rows):
        return [*rows, f'    ... and {total - len(rows)} more']
    return rows


def sweep_report_sections(
    series,
    *,
    census: DanglingCensus,
    tripwire_items: list,
    surfacing: SurfacingObservation,
    staleness: StalenessObservation,
    scan_stats: dict[str, dict[str, int]],
    terminal_join: TerminalTaskJoin,
) -> tuple[ReportSection, ...]:
    """The report, decomposed — one block per family, then the disclosures.

    Every family gets a section whether or not it emitted a metric. That is
    the point: a family that measured nothing omits its metric (a fabricated
    zero-exposure datapoint in leaf α's baseline window is worse than a gap),
    and without a section saying so the absence would read to a human as a
    clean result.
    """
    from shared.memory_eval_metrics import render_report  # noqa: PLC0415

    sections: list[ReportSection] = [
        ReportSection('header', (
            f'E4 staleness sweep — {series.eval_id} @ {series.run_stamp}',
            f'corpus: {series.corpus.project_id}',
            '',
            'This report MEASURES. It evaluates no limit and reaches no',
            'verdict; every threshold lives in the limits evaluator (leaf α).',
        )),
        ReportSection('metrics', ('', render_report(series).rstrip('\n'))),
    ]

    unmeasured = metric_families_not_measured(series)
    if unmeasured:
        sections.append(ReportSection('not_measured', (
            '',
            'NOT MEASURED this run (metric omitted, not zero):',
            *(f'  {metric_id}' for metric_id in unmeasured),
            '  An omitted metric is a gap in the trend, not a clean result.',
        )))

    sections.append(ReportSection('superseded_surfacing', (
        '',
        'Family 1 — superseded entries still surfacing',
        f'  comparable pairs (both returned): {surfacing.pairs_comparable}',
        f'  superseded above its successor:   {surfacing.still_surfacing}',
        *_elided(
            [
                f'    {record.superseded_id} (rank {record.superseded_rank}) '
                f'above {record.successor_id} (rank {record.successor_rank})'
                for record in surfacing.inversions[:_MAX_NAMED]
            ],
            len(surfacing.inversions),
        ),
    )))

    unresolved_rows = [
        f'    {ref.key}: {ref.source_id} -> {ref.target!r}'
        for ref in census.unresolved_refs[:_MAX_NAMED]
    ]
    sections.append(ReportSection('dangling_pointers', (
        '',
        'Family 2 — dangling pointer census',
        f'  pointers examined: {census.examined}',
        f'  resolved:          {census.resolved}',
        f'  unresolved:        {census.unresolved}',
        *(f'  {key}: {row}' for key, row in sorted(census.by_key.items())),
        # Named, not just counted: a bare total tells an operator that
        # something dangles but not which pointer to go and look at.
        *_elided(unresolved_rows, len(census.unresolved_refs)),
    )))

    failing = [item for item in tripwire_items if not item.passed]
    sections.append(ReportSection('successor_pointer_tripwire', (
        '',
        'Family 2b — successor pointer present (per supersedes edge)',
        f'  edges checked: {len(tripwire_items)}',
        f'  edges whose predecessor is gone: {len(failing)}',
        *_elided([f'    {item.item_key}' for item in failing[:_MAX_NAMED]], len(failing)),
    )))

    sections.append(ReportSection('task_terminal_staleness', (
        '',
        'Family 3 — entries asserting live state for a terminal task',
        f'  entries referencing a terminal task: {staleness.entries_referencing_terminal}',
        f'  of those, asserting live state:      {staleness.stale}',
        *_elided(
            [
                f'    {record.record_id} claims task {record.task_id} is live '
                f'(actually {record.status})'
                for record in staleness.records[:_MAX_NAMED]
            ],
            len(staleness.records),
        ),
    )))

    if not terminal_join.available:
        sections.append(ReportSection('task_backend_skipped', (
            '',
            'DISCLOSURE — the task join did not run:',
            f'  {terminal_join.skipped_reason}',
        )))

    truncated = sorted(
        category for category, row in scan_stats.items() if row.get('truncated')
    )
    if truncated:
        sections.append(ReportSection('scan_truncation', (
            '',
            'DISCLOSURE — the scan hit its per-category cap:',
            *(
                f'  {category}: scanned {scan_stats[category]["scanned"]} '
                '(the cap; there may be more)'
                for category in truncated
            ),
            '  These families describe a CAPPED SAMPLE, not the whole corpus.',
        )))

    malformed = malformed_pointer_refs(census.unresolved_refs)
    if malformed:
        sections.append(ReportSection('malformed_pointers', (
            '',
            'DISCLOSURE — pointer members that could never name a memory:',
            # Separated from genuinely-missing targets on purpose: "the target
            # is gone" and "the pointer was never writable" have different
            # fixes, and the aggregate count cannot express which this is.
            *_elided(
                [
                    f'  {ref.key}: {ref.source_id} -> {ref.target!r}'
                    for ref in malformed[:_MAX_NAMED]
                ],
                len(malformed),
            ),
        )))

    return tuple(sections)


def join_report_sections(sections: tuple[ReportSection, ...]) -> str:
    """Render *sections* to the text an operator reads.

    Each section carries its own leading blank line, so this is a plain
    concatenation — there is no separator policy here that could disagree with
    what a section believes its own shape is.
    """
    return '\n'.join(line for section in sections for line in section.lines) + '\n'


def write_report_text(path: str | Path, text: str) -> None:
    """Replace the report at *path* with *text*, atomically.

    A plain ``write_text`` would be a truncate-and-write over the file
    ``write_metric_series`` had just written atomically — and that module's
    ``_atomic_write_text`` exists precisely because the memory-eval leaves
    share one artifact root that the dashboard reads as plain files. A crash
    or a concurrent reader mid-write would leave a truncated report beside a
    valid metrics artifact, the one state that atomicity was designed to
    exclude. Widening the report must not reopen the hole.

    The mechanism is copied rather than imported (β does the same):
    ``_atomic_write_text`` is module-private in ``shared`` and this leaf holds
    no lock on that package. ``mkstemp`` gives an OS-guaranteed exclusively
    created sibling — not a pid-derived name, which concurrent writers under
    the shared root could collide on.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        suffix='.tmp', prefix=f'{path.name}.', dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# The read-only run band
#
# D8's runner pattern minus every mutation: CONFIG_PATH from --config,
# FusedMemoryConfig(), MemoryService(), initialize(), try/finally close().
#
# There is no --apply band and no write call anywhere below. The guarantee is
# asserted as BEHAVIOUR: the tests drive this band against a service double
# whose every write method raises, so a run that completes never wrote.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepOutcome:
    """Everything one run produced, for a caller that wants more than an exit code."""

    series: Any
    census: DanglingCensus
    tripwire_items: list
    surfacing: SurfacingObservation
    staleness: StalenessObservation
    scan_stats: dict[str, dict[str, int]]
    terminal_join: TerminalTaskJoin
    report: str
    sections: tuple[ReportSection, ...]
    """The report's blocks under their machine keys — ``report`` is their join.

    Carried so a caller can ask what this run DISCLOSED without pattern
    matching English out of the rendered text.
    """
    metrics_path: Path | None
    report_path: Path | None


def corpus_project_id(project_ids: tuple[str, ...]) -> str:
    """The single ``Corpus.project_id`` for a run covering *project_ids*.

    M1's Corpus carries one project id and this runner emits one artifact per
    stamp, so a multi-project run needs a stable joined identifier. Single
    project in, that project out — β's shape, so the two leaves' artifacts
    stay comparable.
    """
    return '+'.join(project_ids)


async def run_sweep(
    memory: Any,
    *,
    project_ids: tuple[str, ...],
    scan_limit: int,
    out_root: str | Path,
    stamp: str | None = None,
    config: Any = None,
    eval_id: str = EVAL_ID,
    write_metrics: bool = True,
) -> SweepOutcome:
    """Sweep *project_ids* against *memory* and assemble the run's artifacts.

    *memory* is injected rather than constructed here, which is what lets the
    read-only guarantee be tested: the whole band runs against a double whose
    write methods raise.

    The series is built even when *write_metrics* is False, because the report
    names which families went unmeasured and that answer comes from the
    assembled series — a ``--no-metrics`` run must not be a less honest one.
    """
    from shared.memory_eval_metrics import run_stamp, write_metric_series  # noqa: PLC0415

    effective_stamp = stamp or run_stamp()

    records: list[dict[str, Any]] = []
    scan_stats: dict[str, dict[str, int]] = {}
    refs: list[PointerRef] = []
    resolution: dict[str, bool] = {}
    surfacings: list[SurfacingObservation] = []

    for project_id in dict.fromkeys(project_ids):
        project_records, project_stats = await fetch_pointer_records(
            memory, project_id, scan_limit=scan_limit,
        )
        records.extend(project_records)
        for category, row in project_stats.items():
            merged = scan_stats.setdefault(category, {'scanned': 0, 'truncated': 0})
            merged['scanned'] += row['scanned']
            merged['truncated'] = max(merged['truncated'], row['truncated'])

        project_refs = [ref for record in project_records for ref in pointer_targets(record)]
        refs.extend(project_refs)
        resolution.update(
            await resolve_pointer_targets(memory, project_id, project_refs),
        )
        surfacings.append(await fetch_surfacing_ranks(memory, project_id, project_refs))

    terminal_join = (
        await fetch_terminal_task_ids(config) if config is not None
        else TerminalTaskJoin(statuses={}, skipped_reason=(
            'no config was supplied to this run, so no task statuses could be '
            'read. The task-terminal-staleness family was NOT measured — its '
            'absence from the metrics is a gap, not a clean result.'
        ))
    )

    census = dangling_census(refs, resolution)
    tripwire_items = successor_pointer_items(refs, resolution)
    surfacing = combine_surfacing(surfacings)
    staleness = terminal_staleness(records, terminal_join.statuses)

    corpus_counts = {
        f'scanned_{category}': row['scanned'] for category, row in sorted(scan_stats.items())
    }
    corpus_counts.update({
        f'truncated_{category}': row['truncated']
        for category, row in sorted(scan_stats.items())
    })

    series = build_series(
        census=census,
        tripwire_items=tripwire_items,
        surfacing=surfacing,
        staleness=staleness,
        corpus_counts=corpus_counts,
        project_id=corpus_project_id(tuple(project_ids)),
        stamp=effective_stamp,
        refs=refs,
        eval_id=eval_id,
    )

    sections = sweep_report_sections(
        series,
        census=census,
        tripwire_items=tripwire_items,
        surfacing=surfacing,
        staleness=staleness,
        scan_stats=scan_stats,
        terminal_join=terminal_join,
    )
    report = join_report_sections(sections)

    metrics_path: Path | None = None
    report_path: Path | None = None
    if write_metrics:
        metrics_path, report_path = write_metric_series(
            series, out_root, stamp=effective_stamp,
        )
        # The shared writer lays down its own metric table; replace it with
        # this runner's fuller text, atomically, so the two artifacts under
        # one stamp never disagree about what the run found.
        write_report_text(report_path, report)

    return SweepOutcome(
        series=series,
        census=census,
        tripwire_items=tripwire_items,
        surfacing=surfacing,
        staleness=staleness,
        scan_stats=scan_stats,
        terminal_join=terminal_join,
        report=report,
        sections=sections,
        metrics_path=metrics_path,
        report_path=report_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface — δ's flag vocabulary, minus every mutation flag.

    There is deliberately no ``--apply``, ``--fix`` or ``--prune``: this sweep
    reports, and a reporting runner with a mutation flag is one refactor away
    from being a repair tool nobody reviewed as one. The absence is asserted
    by equality in the tests, so adding one is a decision, not a slip.
    """
    parser = argparse.ArgumentParser(description=(
        'Read-only E4 staleness sweep: superseded surfacing, dangling-pointer '
        'census, and task-terminal staleness. Reports; never repairs.'
    ))
    parser.add_argument(
        '--project-id', dest='project_id', required=True,
        help='Project id to sweep',
    )
    parser.add_argument(
        '--scan-limit', dest='scan_limit', type=int, default=5000,
        help='Maximum number of memories to scan PER CATEGORY (default: 5000). '
             'A cap that fires is disclosed in the report and in corpus.counts.',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    parser.add_argument(
        '--metrics-root', dest='metrics_root', default=_DEFAULT_METRICS_ROOT,
        help=f'Root for M1 metric artifacts (default: {_DEFAULT_METRICS_ROOT})',
    )
    parser.add_argument(
        '--eval-id', dest='eval_id', default=EVAL_ID,
        help=f'Artifact directory name under --metrics-root (default: {EVAL_ID})',
    )
    parser.add_argument(
        '--no-metrics', dest='no_metrics', action='store_true',
        help='Skip the metrics artifact (report to stdout only)',
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    """Build a live MemoryService, sweep, print the report."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # Imported as MODULE ATTRIBUTES, function-locally (D8): that indirection
    # is the seam the read-only double patches, so the guarantee is testable
    # end to end without a test-only hook in this script.
    import fused_memory.config.schema as _schema  # noqa: PLC0415
    import fused_memory.services.memory_service as _service  # noqa: PLC0415

    config = _schema.FusedMemoryConfig()
    memory = _service.MemoryService(config)
    await memory.initialize()
    try:
        outcome = await run_sweep(
            memory,
            project_ids=(args.project_id,),
            scan_limit=args.scan_limit,
            out_root=args.metrics_root,
            config=config,
            eval_id=args.eval_id,
            write_metrics=not args.no_metrics,
        )
    finally:
        await memory.close()

    print(outcome.report, end='')
    if outcome.metrics_path is not None:
        logger.info('metrics: %s', outcome.metrics_path)
        logger.info('report:  %s', outcome.report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
