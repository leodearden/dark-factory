#!/usr/bin/env python3
"""LME replay corpus — a stratified, outcome-blind sample of real dark_factory episodes.

PRD ``plans/local-memory-models-eval-prd.md`` task δ.

Strictly READ-ONLY. It reads the ``Episodic`` population out of a FalkorDB
graph over ``GRAPH.RO_QUERY`` — a server-enforced read-only command path, not
a client-side promise — and writes one committed artifact next to this script.
It never writes a node, never creates an index, and never constructs a
graphiti driver.

.. warning::

   **Never import or construct ``graphiti_core.driver.falkordb_driver.FalkorDriver``
   here.** Its ``__init__`` fire-and-forgets ``build_indices_and_constraints()``,
   which would create indices on ``dark_factory`` and destroy the protected
   no-index evidence owned by ``docs/prds/falkordb-index-provisioning.md``.
   ``falkordb.FalkorDB`` — the DB client used below — is a categorically
   different object and does no such thing. The no-driver ``ro_query`` pattern
   is already in-tree at ``fused-memory/tests/test_list_indices_integration.py``.

The binding hazard: no conditioning on the incumbent's outcome
--------------------------------------------------------------
Corpus membership must never depend on how well the INCUMBENT extraction
pipeline did on an episode. ``e.entity_edges`` is the per-episode record of
what that pipeline produced, so it is the exact outcome proxy that must stay
untouched. The guarantee is mechanized three ways, each of them tested:

1. The Cypher projection names only the six fields below — no ``entity_edges``,
   no filtering predicate of any kind.
2. :class:`EpisodeRecord` has no outcome attribute, so no downstream sampling
   rule *can* condition on one.
3. The manifest records the machine-checkable facts (the projected field list,
   the dimensions used) beside the prose statement, so a reviewer can check
   the claim rather than trust it.

Measured read-only on 2026-08-05: exactly one episode of 2770
(``e622a9bf-f1c8-431b-ad36-92762d69436d``) has ``size(entity_edges) == 0`` —
the one the incumbent extracted nothing from. It is the eligibility anchor:
it must remain selectable.

Record schema
-------------
One entry per selected episode; the projection is closed, not open-ended::

    {
      "uuid":               "<episode uuid>",   # ─┐ the six projected
      "name":               "<episode name>",   #  │ fields — and NOTHING
      "group_id":           "dark_factory",     #  │ else. Adding an
      "source_description": "add_memory:temporal_facts",  # │ outcome field
      "created_at":         "2026-05-16T…+00:00",         # │ here is the
      "content":            "<the replayed bytes>"        # ─┘ hazard.
    }

Stratification
--------------
The cross-product (month bucket of ``created_at``) × (payload kind from
``source_description``). ``e.source`` is NOT usable as a payload axis: measured
read-only, it is uniformly ``'text'`` for all 2770 rows.

Artifacts
---------
``corpus_manifest.json``, committed beside this script. It carries the literal
marker line below, as does this module::

    PRD-MARKER:local-memory-models-eval corpus-manifest

Declared consumers: ε (replay engine input), ζ (control replays), θ (full arm
replays).

Failure semantics
-----------------
Every failure is loud and typed. Nothing is absorbed and continued.

============================  ============================================
Condition                     Behaviour
============================  ============================================
Malformed ``created_at``      :class:`CorpusBuildError` — never an
                              ``'unknown'`` bucket, which would silently
                              become an extra stratum.
Unusable                      :class:`CorpusBuildError` naming the
``source_description``        offending value.
``n`` > population            :class:`CorpusBuildError` — never a quietly
                              shorter corpus.
Non-``GRAPH.RO_QUERY``        :class:`CorpusBuildError` from the reader's
command attempted             client-side guard.
============================  ============================================

Verification (``--verify``) instead reports a status and exits on its code.
Each failure has its own code because each has a different remedy, and a CI
job reads the code without ever opening the report:

====================  ====  ==================================================
Status                Exit  Meaning and remedy
====================  ====  ==================================================
``ok``                   0  Every episode re-derived from the recorded
                            criteria and every content hash matched.
``id_mismatch``          2  Re-deriving produced a DIFFERENT corpus than the
                            manifest records. The manifest no longer describes
                            what its own criteria produce — re-derive, or
                            investigate tampering.
``hash_drift``           3  The selection is still correct, but episode
                            content changed underneath it. Re-hash, and treat
                            ε/ζ/θ results computed against the old bytes as
                            suspect.
``missing_episodes``     4  An id the manifest names is absent from the
                            store. Reported, never silently skipped.
``bad_manifest``         5  The artifact is structurally unusable — reported
                            as a status rather than raised as a traceback, so
                            a reader can tell a broken artifact from a broken
                            checker.
====================  ====  ==================================================

Exit ``1`` (:data:`EXIT_RUN_FAILED`) is reserved for a run that could not
complete at all — the store was unreachable, or the requested corpus is
unsatisfiable. It is deliberately outside the table above so a caller can tell
"the corpus is wrong" from "the check never ran".
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUILDER_VERSION = 1
"""Stamped into the manifest's criteria; bump on any change to a sampling rule.

A manifest built by an older builder is not re-derivable by a newer one if a
rule moved, and a reviewer needs to be able to see that from the artifact.
"""

PROJECTED_FIELDS: tuple[str, ...] = (
    'uuid',
    'name',
    'group_id',
    'source_description',
    'created_at',
    'content',
)
"""The closed set of episode fields this builder ever reads.

Load-bearing, not documentation: the reader builds its Cypher RETURN list from
this tuple, the manifest records it as the machine-checkable half of the
no-outcome-filter statement, and a test asserts it contains no outcome signal.
Adding ``entity_edges`` here would breach the binding hazard.
"""

_CATEGORY_PREFIXES: tuple[str, ...] = ('add_memory:', 'replay_from_mem0:')
"""The two in-tree episode writers that encode the category in the description.

``MemoryService`` writes ``add_memory:<category>`` on the classified-write path
and ``replay_from_mem0:<category>`` on the Mem0 replay path. Anything else is a
caller-supplied ``add_episode`` description.
"""

_TEMPORAL_PREFIX_RE = re.compile(r'^\[temporal:[^\]]*\]\s*')
"""``graphiti_client`` prepends ``[temporal:<ctx>] `` when temporal_context is set.

Verified absent from dark_factory today — all rows are bare ``add_memory:*`` —
but stripping it here keeps a future episode from landing in its own bogus
payload-kind stratum.
"""

STRATIFICATION_DIMENSIONS: tuple[str, ...] = (
    'month(created_at)',
    'payload_kind(source_description)',
)
"""The two stratification axes, named as the manifest records them.

``source_description``, NOT ``source``: measured read-only, ``e.source`` is
uniformly ``'text'`` across every episode in the store and discriminates
nothing. Neither axis is derived from anything the incumbent pipeline
produced — that is the point, and the manifest records this tuple beside the
no-outcome-filter statement so the claim is checkable.
"""

ADD_EPISODE_KIND = 'add_episode'
"""Payload kind for a caller-supplied ``add_episode`` description.

One explicit bucket rather than one stratum per caller string: the payload axis
must stay a small closed set, or the cross-product fragments into hundreds of
singleton cells and stratification stops meaning anything.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CorpusBuildError(RuntimeError):
    """A corpus build or verification could not proceed on the data it was given.

    Always raised in preference to degrading silently. Every condition in the
    module docstring's failure-semantics table surfaces as one of these.
    """


# ---------------------------------------------------------------------------
# The episode record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode, projected to the six fields this builder is allowed to read.

    Deliberately carries **no** ``entity_edges`` and no other outcome field.
    That absence is the structural half of the no-outcome-filter guarantee: a
    sampling rule cannot condition on a signal the record does not hold. A test
    pins the field set exactly, so re-adding one is a red test, not a silent
    regression.
    """

    uuid: str
    name: str
    group_id: str
    source_description: str
    created_at: str
    content: str


# ---------------------------------------------------------------------------
# Stratum derivation
# ---------------------------------------------------------------------------


def parse_timestamp(value: object, *, field: str = 'created_at') -> datetime:
    """Parse one of the store's ISO-8601 spellings into an aware datetime.

    The single normalization point for every timestamp this module reads.
    ``dark_factory`` has accumulated four spellings of the same instant —
    ``+00:00`` offset, ``Z`` suffix, naive, and a space separator — and they
    must all land on one value, or a single month splits across strata purely
    by which code path persisted the row.

    A **naive** value is read as UTC. Every writer in the tree stamps UTC, and
    the alternative is worse: a naive value that stays naive cannot be compared
    against an aware one at all, so :func:`verify_manifest`'s window bound
    would raise ``TypeError`` on exactly the rows it is meant to filter.

    An aware value is returned with its own offset intact, **not** converted to
    UTC. :func:`month_bucket` reads ``.year``/``.month`` straight off this
    result, so converting here would silently re-bucket an episode written at,
    say, ``2026-05-01T00:30+05:00`` into the previous month.

    *field* names the caller's field in the error text, so a bad manifest
    criterion and a bad store row are distinguishable in the message.

    Raises :class:`CorpusBuildError` on anything unparseable rather than
    returning a sentinel — see :func:`month_bucket` for why a fallback bucket
    would make the stratification report lie.
    """
    if not isinstance(value, str) or not value.strip():
        raise CorpusBuildError(f'{field} must be a non-empty ISO-8601 string, got {value!r}')
    text = value.strip().replace(' ', 'T', 1)
    if text.endswith(('Z', 'z')):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CorpusBuildError(
            f'{field} is not a parseable ISO-8601 timestamp: {value!r} ({exc})'
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def month_bucket(created_at: object) -> str:
    """Return the ``YYYY-MM`` time stratum for *created_at*.

    Normalization lives in :func:`parse_timestamp`; this is the bucketing on
    top of it.

    Raises :class:`CorpusBuildError` on anything unparseable, rather than
    returning an ``'unknown'`` bucket. An ``'unknown'`` bucket would quietly
    become an extra stratum, and the stratification report would look complete
    while describing a corpus whose time axis had partially collapsed.
    """
    parsed = parse_timestamp(created_at)
    return f'{parsed.year:04d}-{parsed.month:02d}'


def payload_kind(source_description: object) -> str:
    """Return the payload stratum for *source_description*.

    ``source_description`` — not ``source`` — is the payload-kind signal:
    measured read-only, ``e.source`` is uniformly ``'text'`` across all 2770
    dark_factory episodes and discriminates nothing.

    Recognized shapes, in order: an optional ``[temporal:<ctx>] `` prefix is
    stripped first, then ``add_memory:<category>`` and
    ``replay_from_mem0:<category>`` yield ``<category>``, and any other
    non-empty description is a caller-supplied ``add_episode`` string and
    yields :data:`ADD_EPISODE_KIND`.

    Raises :class:`CorpusBuildError` naming the offending value when the
    description is missing or blank — there is no defensible bucket for an
    episode whose writer cannot be identified at all.
    """
    if not isinstance(source_description, str) or not source_description.strip():
        raise CorpusBuildError(
            f'source_description must be a non-empty string, got {source_description!r}'
        )
    text = _TEMPORAL_PREFIX_RE.sub('', source_description.strip()).strip()
    if not text:
        raise CorpusBuildError(
            f'source_description carries only a temporal prefix: {source_description!r}'
        )
    for prefix in _CATEGORY_PREFIXES:
        if text.startswith(prefix):
            category = text[len(prefix) :].strip()
            if not category:
                raise CorpusBuildError(
                    f'source_description has an empty category: {source_description!r}'
                )
            return category
    return ADD_EPISODE_KIND


def content_hash(content: str) -> str:
    """Return the full 64-char SHA-256 hex digest of *content*'s UTF-8 bytes.

    Deliberately NOT ``memory_eval_retrieval_probe.content_key()``, which is
    ``sha256(whitespace-normalized)[:16]``. That digest's job is stable
    *identity matching* of memory text across re-indentation and rewrapping;
    this one's job is the opposite — **drift detection** on replay inputs. For
    ε/ζ/θ the bytes fed to the extraction pipeline are the experiment's
    independent variable, so a whitespace change IS a change and must fail
    verification, and truncating to 16 hex chars would weaken a collision
    guarantee that costs nothing to keep. Same primitive, different semantics.
    """
    if not isinstance(content, str):
        raise CorpusBuildError(f'content must be a string, got {type(content).__name__}')
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def stratum_key(record: EpisodeRecord) -> tuple[str, str]:
    """Return *record*'s ``(month, payload_kind)`` stratum key.

    Propagates :class:`CorpusBuildError` from either axis: a record that cannot
    be placed in a cell must abort the build, not land in a fallback cell that
    the stratification report would then misreport as real coverage.
    """
    return (month_bucket(record.created_at), payload_kind(record.source_description))


ALLOCATION_RULE = 'min-1-floor + largest-remainder proportional, capped at cell size'
"""Recorded verbatim in the manifest's criteria so a reviewer can re-derive.

The string is the contract; :func:`allocate` is its implementation. A manifest
that names a different rule than the builder applies is not re-derivable, so
the two travel together.
"""


def allocate(cell_counts: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    """Split *n* seats across the cells of *cell_counts*, summing to exactly *n*.

    The rule, in this order:

    1. Reject an unsatisfiable request loudly — non-positive *n*, an empty
       cell, ``n`` below the cell count, or ``n`` above the population.
    2. Seat 1 in every cell (the floor).
    3. Distribute the residual by largest remainder over each cell's
       proportional share, capping at cell size and redistributing any capped
       overflow until the total lands on *n* exactly.
    4. Break remainder ties on the sorted cell key, so the result is fully
       deterministic with no RNG and no dependence on dict iteration order.

    **Why a floor at all.** Measured on 2026-08-05, the
    ``('2026-04', 'procedural_knowledge')`` cell holds exactly ONE of the
    store's 2770 episodes. Its proportional share at N=200 is 0.07 seats, so
    pure proportional allocation rounds it to zero and silently deletes an
    entire payload kind from the corpus — and therefore from every downstream
    arm comparison in ε/ζ/θ. Nothing in the artifact would show the loss: the
    stratification report would still reconcile, over a payload axis that had
    quietly lost a value. The floor costs 16 of 200 seats and makes that
    impossible.

    **Why loud rejection rather than a best effort.** Shaving cells to fit a
    too-small *n*, or returning fewer than *n* for a too-large one, would
    reintroduce exactly the invisible coverage loss the floor prevents — only
    now with the artifact claiming a corpus size it does not have.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise CorpusBuildError(f'n must be an int, got {n!r}')
    if n <= 0:
        raise CorpusBuildError(f'n must be positive, got {n}')
    if not cell_counts:
        raise CorpusBuildError('cannot allocate over an empty stratification')

    empty = sorted(cell for cell, count in cell_counts.items() if count <= 0)
    if empty:
        raise CorpusBuildError(
            f'cell_counts contains {len(empty)} empty cell(s) {empty}; exclude them '
            f'before allocating — a min-1 seat in an empty cell can never be filled '
            f'and the allocation would not sum to n'
        )

    population = sum(cell_counts.values())
    if n > population:
        raise CorpusBuildError(
            f'cannot draw n={n} episodes from a population of {population}'
        )
    if n < len(cell_counts):
        raise CorpusBuildError(
            f'n={n} is below the {len(cell_counts)} non-empty cells, so the min-1 '
            f'floor is unsatisfiable; raise n to at least {len(cell_counts)} rather '
            f'than dropping cells, which would silently lose coverage'
        )

    cells = sorted(cell_counts)
    alloc = dict.fromkeys(cells, 1)
    residual = n - len(cells)

    # Largest remainder over the proportional share of the residual. Ranking
    # on (-remainder, cell) rather than (-remainder,) alone is what makes ties
    # deterministic; without it, equal remainders would resolve by whatever
    # order the dict happened to have.
    if residual:
        shares = {c: cell_counts[c] * residual / population for c in cells}
        ranked = sorted(cells, key=lambda c: (-(shares[c] % 1), c))
        for cell in cells:
            alloc[cell] += int(shares[cell])
        leftover = residual - sum(int(shares[c]) for c in cells)
        for cell in ranked[:leftover]:
            alloc[cell] += 1

    # Cap at cell size and redistribute the overflow. Loops because a
    # redistribution can itself overfill another cell; it terminates because
    # every pass either places all overflow or strictly shrinks the set of
    # cells with room, and n <= population guarantees room exists.
    while True:
        overflow = 0
        for cell in cells:
            if alloc[cell] > cell_counts[cell]:
                overflow += alloc[cell] - cell_counts[cell]
                alloc[cell] = cell_counts[cell]
        if not overflow:
            break
        room = [c for c in cells if alloc[c] < cell_counts[c]]
        if not room:  # pragma: no cover — unreachable while n <= population
            raise CorpusBuildError(
                f'cannot place {overflow} seat(s): every cell is at capacity but '
                f'the allocation totals {sum(alloc.values())}, not {n}'
            )
        ranked = sorted(room, key=lambda c: (-cell_counts[c], c))
        for cell in (ranked * (overflow // len(ranked) + 1))[:overflow]:
            alloc[cell] += 1

    total = sum(alloc.values())
    if total != n:  # pragma: no cover — an allocator bug, not a data condition
        raise CorpusBuildError(f'allocation totals {total}, expected {n}')
    return alloc


# ---------------------------------------------------------------------------
# Selection — a prefix of a per-cell seeded permutation
# ---------------------------------------------------------------------------

SELECTION_RULE = (
    'per cell: canonical uuid sort -> seeded permutation -> take the first '
    'allocate() entries (a PREFIX, so growing N only appends)'
)
"""Recorded verbatim in the manifest's criteria; see :func:`select`."""

DISPOSITIONS: tuple[str, ...] = ('selected', 'not_drawn')
"""The closed vocabulary of exits from :func:`select`.

Every record handed to :func:`select` leaves by exactly ONE of these — the
``SampleResult`` convention from ``scripts/legibility/sampling.py``. Only two
doors exist here *because* the sampler applies no eligibility filter at all:
there is deliberately no ``ineligible`` or ``low_quality`` exit, since either
would be a place for outcome conditioning to hide. A record is either drawn or
it is not, and nothing else can happen to it.
"""


@dataclass(frozen=True)
class SelectionResult:
    """The selection plus its exhaustive accounting.

    ``selected`` and ``not_selected`` partition the input population exactly:
    no episode disappears unexplained. Without that, a corpus that quietly lost
    records would still report a clean total.
    """

    selected: list[EpisodeRecord]
    not_selected: dict[str, str]
    allocation: dict[tuple[str, str], int]
    cell_counts: dict[tuple[str, str], int]


def _group_by_cell(
    population: list[EpisodeRecord],
) -> dict[tuple[str, str], list[EpisodeRecord]]:
    """Group *population* by stratum key, rejecting duplicate uuids.

    A duplicate uuid would break the disposition accounting (one id could be
    both selected and not-selected), so it is a loud error rather than a
    silently deduplicated row.
    """
    if not population:
        raise CorpusBuildError('cannot select from an empty population')
    seen: set[str] = set()
    cells: dict[tuple[str, str], list[EpisodeRecord]] = {}
    for record in population:
        if record.uuid in seen:
            raise CorpusBuildError(
                f'duplicate episode uuid in population: {record.uuid!r} — '
                f'disposition accounting requires unique ids'
            )
        seen.add(record.uuid)
        cells.setdefault(stratum_key(record), []).append(record)
    return cells


def cell_permutation(
    population: list[EpisodeRecord], cell: tuple[str, str], *, seed: str
) -> list[str]:
    """Return *cell*'s full seeded permutation of uuids, for any N.

    Exposed as its own function so the prefix property is directly checkable:
    a reviewer (and the test suite) can confirm that what :func:`select`
    returned for a cell really is the head of this list, rather than taking the
    claim on faith.

    The **canonical uuid sort before seeding** is what makes the result
    independent of the order FalkorDB returns rows — which is not guaranteed
    stable between runs. Without it, a re-run could silently produce a
    different corpus while still claiming the same seed.
    """
    cells = _group_by_cell(population)
    if cell not in cells:
        raise CorpusBuildError(f'no such stratum cell in population: {cell!r}')
    uuids = sorted(record.uuid for record in cells[cell])
    # Per-cell RNG derived from the seed — never a module-global `random`
    # draw, which would make the result depend on whatever else ran first in
    # the process (the keyword-only injected-RNG convention at
    # scripts/legibility/census.py). Per cell rather than one stream for the
    # whole run so a cell's permutation does not shift when a DIFFERENT cell
    # gains or loses episodes.
    random.Random(f'{seed}:{cell[0]}:{cell[1]}').shuffle(uuids)
    return uuids


def select(population: list[EpisodeRecord], n: int, *, seed: str) -> SelectionResult:
    """Draw *n* records from *population*, stratified, deterministically.

    Within each cell the draw is a **prefix of a seeded permutation**: sort the
    cell canonically by uuid, permute it under ``Random(f'{seed}:{month}:{kind}')``,
    and take the first ``allocate()[cell]`` entries.

    Two properties fall out of that choice, and both are load-bearing:

    *Order-independence* — the canonical sort happens before any seeding, so
    the sample does not depend on the order the store returned rows in.

    *Nestedness under an N re-tune* — PRD Open Q4 leaves the final corpus size
    (~150–300) to be settled jointly with ζ from measured control variance and
    wall-clock. Because growing N only ever *appends* to a cell's take, ζ can
    re-run this builder at a different N without invalidating replays ε has
    already completed at the smaller one. Note this is a PER-CELL guarantee,
    not a global one: largest-remainder allocation can move a single seat
    between cells as N changes, so a cell whose allocation *shrank* is the one
    case where an earlier pick is dropped.

    Records are returned ordered by ``(cell key, position in that cell's
    permutation)``, so the output ordering is fully determined rather than
    incidental.

    No eligibility filter is applied — see :data:`DISPOSITIONS`. Every episode
    in *population* competes, including one the incumbent pipeline extracted
    nothing from.
    """
    cells = _group_by_cell(population)
    cell_counts = {cell: len(records) for cell, records in cells.items()}
    allocation = allocate(cell_counts, n)

    by_uuid = {record.uuid: record for record in population}
    selected: list[EpisodeRecord] = []
    not_selected: dict[str, str] = {}
    for cell in sorted(cells):
        permutation = cell_permutation(population, cell, seed=seed)
        take = allocation[cell]
        selected.extend(by_uuid[uuid] for uuid in permutation[:take])
        for uuid in permutation[take:]:
            not_selected[uuid] = 'not_drawn'

    if len(selected) != n:  # pragma: no cover — a sampler bug, not a data condition
        raise CorpusBuildError(f'selected {len(selected)} records, expected {n}')
    if len(selected) + len(not_selected) != len(population):  # pragma: no cover
        raise CorpusBuildError(
            f'disposition accounting lost records: {len(selected)} selected + '
            f'{len(not_selected)} not selected != {len(population)} in population'
        )
    return SelectionResult(
        selected=selected,
        not_selected=not_selected,
        allocation=allocation,
        cell_counts=cell_counts,
    )


# ---------------------------------------------------------------------------
# The reader seam — GRAPH.RO_QUERY, no graphiti driver
# ---------------------------------------------------------------------------

DEFAULT_GRAPH = 'dark_factory'
"""The graph this corpus is drawn from."""

RO_COMMAND = 'GRAPH.RO_QUERY'
"""The only FalkorDB command this builder is permitted to issue.

Read-only here is **server-enforced**, not client-promised: a ``CREATE`` issued
through this command path was refused in a live probe and materialized no
graph. That is a strictly stronger guarantee than a client-side convention,
and it is why this builder reads over Cypher rather than through the product
read path.
"""

POPULATION_CYPHER = 'MATCH (e:Episodic) RETURN ' + ', '.join(
    f'e.{field}' for field in PROJECTED_FIELDS
)
"""The single query this builder issues. Built from :data:`PROJECTED_FIELDS`.

Note what is deliberately absent:

* **No ``entity_edges``** and no other incumbent-success signal — the binding
  hazard. The projection is derived from one constant that a test pins, so the
  query and the manifest's claim about the query cannot drift apart.
* **No ``WHERE``** — no filter of any kind. The population IS the population.
* **No ``LIMIT``** — the full population is required or the stratum
  denominators are wrong, and every proportional share computed from them
  would be wrong with it.

Why not the product ``get_episodes`` read path: ``MemoryService.get_episodes``
projects six fields that do **not** include ``source_description`` — which is
the payload-kind stratification dimension here — and the MCP tool clamps
``last_n`` to 1000, well below the ~2770-row population.
"""


class EpisodeReader:
    """Reads the ``Episodic`` population over ``GRAPH.RO_QUERY``.

    Constructed either with an explicit *graph* handle (the tests pass a
    double) or with host/port, in which case it opens a ``falkordb.asyncio``
    client lazily.

    ``falkordb.FalkorDB`` — the DB client — is categorically distinct from
    ``graphiti_core.driver.falkordb_driver.FalkorDriver``, the graphiti driver
    whose ``__init__`` fire-and-forgets ``build_indices_and_constraints()``.
    This class must never construct the latter; see the module docstring's
    warning and the in-tree precedent at
    ``fused-memory/tests/test_list_indices_integration.py``.
    """

    def __init__(
        self,
        *,
        graph: object | None = None,
        graph_name: str = DEFAULT_GRAPH,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._graph = graph
        self.graph_name = graph_name
        self.host = host if host is not None else _falkor_host_from_env()
        self.port = port if port is not None else _falkor_port_from_env()

    @staticmethod
    def assert_read_only_command(command: str) -> None:
        """Raise unless *command* is :data:`RO_COMMAND`.

        A client-side guard layered on top of the server-side one. The server
        would refuse a write anyway; this makes the violation a typed
        :class:`CorpusBuildError` at the seam that owns the guarantee, rather
        than a redis error surfacing from three layers down where a reader
        would not connect it to this module's read-only claim.
        """
        if command != RO_COMMAND:
            raise CorpusBuildError(
                f'build_corpus may only issue {RO_COMMAND}, refused {command!r} — '
                f'this script is strictly read-only'
            )

    def _resolve_graph(self) -> object:
        """Return the graph handle, opening a client on first use.

        The import is local so that merely importing this module — which the
        tests do, with a double in hand — never drags in the falkordb client.
        """
        if self._graph is None:
            from falkordb.asyncio import FalkorDB  # noqa: PLC0415

            self._graph = FalkorDB(host=self.host, port=self.port).select_graph(
                self.graph_name
            )
        return self._graph

    async def fetch_population(self) -> list[EpisodeRecord]:
        """Return every ``Episodic`` node, projected to :class:`EpisodeRecord`.

        One query, no filter, no limit. A row whose ``created_at`` or
        ``source_description`` cannot be parsed aborts the build via
        :class:`CorpusBuildError` rather than being dropped — a silently
        dropped row would shrink a stratum's denominator and skew every
        proportional share computed from it.
        """
        self.assert_read_only_command(RO_COMMAND)
        graph = self._resolve_graph()
        response = await graph.ro_query(POPULATION_CYPHER)  # type: ignore[attr-defined]
        rows = getattr(response, 'result_set', response)
        population: list[EpisodeRecord] = []
        for index, row in enumerate(rows):
            if len(row) != len(PROJECTED_FIELDS):
                raise CorpusBuildError(
                    f'row {index} has {len(row)} columns, expected '
                    f'{len(PROJECTED_FIELDS)} {PROJECTED_FIELDS}'
                )
            record = EpisodeRecord(**dict(zip(PROJECTED_FIELDS, row, strict=True)))
            stratum_key(record)  # fail fast and loudly on an unplaceable row
            population.append(record)
        if not population:
            raise CorpusBuildError(
                f'graph {self.graph_name!r} returned no Episodic nodes — refusing to '
                f'build a corpus from an empty population'
            )
        return population


def _falkor_host_from_env() -> str:
    """FalkorDB host from ``FALKOR_HOST``, defaulting to localhost.

    Same env vars and defaults as ``fused-memory/tests/_fm_helpers.py``, so an
    operator who has pointed the test suite at a FalkorDB has pointed this
    script at the same one.
    """
    return os.environ.get('FALKOR_HOST', 'localhost')


def _falkor_port_from_env() -> int:
    """FalkorDB port from ``FALKOR_PORT``, defaulting to 6379.

    The ``int()`` is load-bearing: ``os.environ`` yields str and
    ``FalkorDB(port='6379')`` misbehaves. Unset *or empty* takes the default —
    empty is what a CI template like ``FALKOR_PORT: ${FALKOR_PORT}`` produces
    when the variable is not set. A non-empty non-numeric value is an operator
    error and says so, rather than reading as "FalkorDB not reachable".
    """
    raw = os.environ.get('FALKOR_PORT', '').strip()
    if not raw:
        return 6379
    try:
        return int(raw)
    except ValueError:
        raise CorpusBuildError(
            f'FALKOR_PORT must be an integer, got {raw!r}. Unset it to use the '
            f'default (6379), or set it to the port your FalkorDB listens on.'
        ) from None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

PRD_MARKER = 'PRD-MARKER:local-memory-models-eval corpus-manifest'
"""The literal the delivered_check greps for on the COMMITTED tree.

Carried as a top-level manifest field so it survives JSON serialization
intact, and present in this module's docstring so the script satisfies the
check too. See ``plans/local-memory-models-eval-prd.capability-manifest.yaml``.
"""

DEFAULT_N = 200
"""Provisional corpus size — the midpoint of the PRD's 150–300 band.

Checked against the measured census: at ~7.2% sampling every one of the 16
non-empty cells receives at least one seat, and the min-1 floor consumes only
16 of the 200. PRD Open Q4 defers the FINAL figure to ζ, so this is a starting
point recorded as a criterion, not a baked-in constant — see
:data:`N_RATIONALE`.
"""

N_RATIONALE = (
    "N=200 is delta (δ)'s provisional choice: the midpoint of the PRD's 150-300 "
    'band, checked against the measured census so that all 16 non-empty cells '
    'receive at least one seat. PRD Open Q4 defers the FINAL corpus size to '
    'zeta (ζ), to be settled from measured control variance and wall-clock. '
    'Because selection is a prefix of a per-cell seeded permutation, zeta can '
    're-run this builder at a different N without invalidating replays '
    'epsilon (ε) has already completed.'
)
"""Recorded in the manifest so a later reader does not mistake 200 for settled.

Task labels are spelled BOTH ways — ``zeta (ζ)`` — deliberately. The PRD names
its tasks with Greek letters, but this string lands in a JSON artifact that
reviewers and scripts will grep; a bare ``ζ`` is not something anyone types at
a search prompt. The ASCII spelling is what makes the criterion findable, the
Greek is what ties it back to the PRD row.
"""

DEFAULT_SEED = 'local-memory-models-eval:corpus:v1'
"""The default sampling seed — fixed, never drawn from the clock or ``urandom``.

A random default would make the committed artifact irreproducible from the CLI
alone: a reviewer running the documented command would get a different corpus
every time and could not tell a tampered manifest from a re-seeded one. The
seed is recorded in the manifest, and :func:`verify_manifest` re-derives from
the RECORDED seed rather than from this constant, so changing the default here
never invalidates an existing artifact.
"""

NO_OUTCOME_FILTER_STATEMENT = (
    'Corpus membership is NEVER conditioned on the incumbent extraction '
    "pipeline's outcome. The population query projects only the six fields "
    'listed in projected_fields and applies no WHERE clause, no LIMIT and no '
    'ranking of any kind; e.entity_edges — the per-episode record of what the '
    'incumbent produced — is never read, never filtered on, and never used as a '
    'stratification dimension. Within each stratum the draw is an unconditioned '
    'seeded permutation, not a score-ranked take. An episode the incumbent '
    'extracted zero edges from is exactly as eligible as any other.'
)
"""The prose half of the guarantee. The checkable half sits beside it."""

DEFAULT_MANIFEST_PATH = str(Path(__file__).resolve().parent / 'corpus_manifest.json')
"""Where the committed manifest lives.

Deliberately beside this script rather than in the conventional
``fused-memory/data/memory-evals/`` eval-artifact directory:
``fused-memory/.gitignore`` ignores ``data/``, so a manifest written there
would be uncommittable — satisfying neither the "committed corpus manifest"
deliverable nor the delivered_check, which greps the COMMITTED tree. The
stamped-run convention is right for gitignored per-run output; this artifact is
a committed, reviewable input fixture, so it takes one stable path with no run
stamp.
"""


def _cell_label(cell: tuple[str, str]) -> str:
    """Render a cell key as a JSON-safe ``month|kind`` string.

    JSON object keys must be strings, and a tuple would serialize as an
    unparseable ``"('2026-04', 'temporal_facts')"``. The pipe is safe because
    neither a month bucket nor a category ever contains one.
    """
    return f'{cell[0]}|{cell[1]}'


def build_manifest(
    result: SelectionResult,
    population: list[EpisodeRecord],
    *,
    n: int,
    seed: str,
    graph: str = DEFAULT_GRAPH,
) -> dict:
    """Assemble the committed corpus manifest.

    Carries four things: the selected ids with their content hashes, the
    criteria needed to re-derive them, a stratification report that reconciles
    to the whole population, and the no-outcome-filter guarantee in both its
    prose and machine-checkable forms.

    Episode **content is deliberately not copied** into the manifest. The
    replay engines read content from the store by uuid; a second copy here
    could silently disagree with the first, and hash drift — the exact thing
    :func:`verify_manifest` exists to catch — would then be undetectable.
    """
    # Both bounds are taken on the NORMALIZED instant, never on the raw string.
    # The store spells one instant four ways (see :func:`parse_timestamp`) and
    # those spellings do not sort consistently as text: ``' '`` (0x20) < ``'T'``,
    # ``'Z'`` > ``'+'``, and a non-UTC offset sorts by wall clock rather than by
    # instant. :func:`verify_manifest` consumes ``max_created_at`` through
    # ``parse_timestamp``, so a lexicographic bound here would leave the producer
    # and the consumer of this field disagreeing about which episodes were in
    # frame — and a freshly built, untampered manifest could fail its own
    # ``--verify``, the exact false alarm the window bound exists to prevent.
    #
    # The recorded VALUE keeps the store's own spelling rather than a normalized
    # one, so the artifact still shows what the store literally holds and a
    # reviewer can find the bounding row by the text in front of them.
    def _instant(record: EpisodeRecord) -> datetime:
        return parse_timestamp(record.created_at)

    earliest = min(population, key=_instant).created_at
    latest = max(population, key=_instant).created_at
    cells = {
        _cell_label(cell): {
            'population': count,
            'allocated': result.allocation[cell],
            'selected': sum(
                1 for r in result.selected if stratum_key(r) == cell
            ),
        }
        for cell, count in sorted(result.cell_counts.items())
    }
    return {
        'prd_marker': PRD_MARKER,
        'criteria': {
            'allocation_rule': ALLOCATION_RULE,
            'builder_version': BUILDER_VERSION,
            'graph': graph,
            'n': n,
            'n_rationale': N_RATIONALE,
            'population_size': len(population),
            'seed': seed,
            'selection_rule': SELECTION_RULE,
            'stratification_dimensions': list(STRATIFICATION_DIMENSIONS),
            'window': {'min_created_at': earliest, 'max_created_at': latest},
        },
        'episodes': [
            {
                'uuid': record.uuid,
                'content_hash': content_hash(record.content),
                'month': stratum_key(record)[0],
                'payload_kind': stratum_key(record)[1],
            }
            for record in result.selected
        ],
        'no_outcome_filter': {
            'entity_edges_queried': False,
            'entity_edges_used_as_filter': False,
            'entity_edges_used_as_stratum': False,
            'population_query': POPULATION_CYPHER,
            'projected_fields': list(PROJECTED_FIELDS),
            'statement': NO_OUTCOME_FILTER_STATEMENT,
            'stratification_dimensions': list(STRATIFICATION_DIMENSIONS),
        },
        'stratification_report': {
            'cells': cells,
            'totals': {
                'allocated': sum(c['allocated'] for c in cells.values()),
                'not_selected': len(result.not_selected),
                'population': sum(c['population'] for c in cells.values()),
                'selected': sum(c['selected'] for c in cells.values()),
            },
        },
    }


def serialize_manifest(manifest: dict) -> str:
    """Render *manifest* canonically, so two builds diff cleanly.

    ``sort_keys`` and the fixed indent are what make "two builds over an
    unchanged store are byte-identical" true; without them a re-run would diff
    against itself on incidental key ordering and a real change would be
    unreadable inside the noise. ``ensure_ascii=False`` keeps UTF-8 verbatim
    rather than escaping it into unreviewable ``\\uXXXX`` runs.
    """
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def write_manifest(manifest: dict, path: str | Path = DEFAULT_MANIFEST_PATH) -> Path:
    """Write *manifest* to *path* canonically and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_manifest(manifest), encoding='utf-8')
    return target


# ---------------------------------------------------------------------------
# Verification — the INV-2 / INV-4 seam
# ---------------------------------------------------------------------------

EXIT_CODES = {
    'ok': 0,
    'id_mismatch': 2,
    'hash_drift': 3,
    'missing_episodes': 4,
    'bad_manifest': 5,
}
"""Process exit code per verification status; see the module docstring's table.

Each failure gets its own code because each has a different remedy, and a
wrapper or CI job reads the code without ever opening the report. Collapsing
them into one non-zero would send a reader down the wrong path — see
:class:`VerifyReport`.
"""

EXIT_RUN_FAILED = 1
"""The run could not complete at all — no verdict was reached.

Deliberately outside :data:`EXIT_CODES`, whose values are 0 and 2-5, leaving 1
free for this. A store that could not be read and a corpus that verified
badly are opposite situations: one says "re-run me", the other says "the
artifact is wrong". A single non-zero for both would conflate them.
"""


@dataclass(frozen=True)
class VerifyReport:
    """The outcome of :func:`verify_manifest`, structured rather than boolean.

    A bare ``False`` tells a reviewer nothing about what to fix, so every
    failure carries its diff: which ids are on which side, which contents
    drifted from what to what, which ids vanished.
    """

    status: str
    detail: str
    checked: int = 0
    rederived: list[str] = dataclasses.field(default_factory=list)
    # The value type is a union because the diff carries two kinds of entry:
    # the id lists naming each side, and ``order_differs`` — the boolean that
    # separates "the same episodes in a different order" from "different
    # episodes", which are different diagnoses with different remedies.
    id_mismatch: dict[str, list[str] | bool] = dataclasses.field(default_factory=dict)
    hash_drift: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    missing: list[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == 'ok'

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


def _read_criteria(manifest: object) -> tuple[dict, list[dict]]:
    """Extract and structurally validate the parts verification depends on.

    Raises :class:`CorpusBuildError` rather than letting a malformed manifest
    surface as a ``KeyError`` or ``TypeError`` traceback three frames down,
    where a reader could not tell a broken artifact from a broken checker.
    """
    if not isinstance(manifest, dict):
        raise CorpusBuildError(f'manifest must be a JSON object, got {type(manifest).__name__}')
    criteria = manifest.get('criteria')
    if not isinstance(criteria, dict):
        raise CorpusBuildError('manifest has no "criteria" object to re-derive from')
    for key in ('seed', 'n'):
        if key not in criteria:
            raise CorpusBuildError(f'manifest criteria is missing {key!r}')
    if not isinstance(criteria['n'], int) or isinstance(criteria['n'], bool):
        raise CorpusBuildError(f'manifest criteria n must be an int, got {criteria["n"]!r}')
    if not isinstance(criteria['seed'], str):
        raise CorpusBuildError(f'manifest criteria seed must be a str, got {criteria["seed"]!r}')
    window = criteria.get('window')
    if not isinstance(window, dict) or 'max_created_at' not in window:
        raise CorpusBuildError(
            'manifest criteria is missing window.max_created_at — verification cannot '
            're-derive against an unbounded population, and a manifest that cannot '
            'state its own sampling frame must not be verified as if it had one'
        )
    if not isinstance(window['max_created_at'], str):
        raise CorpusBuildError(
            f'manifest criteria window.max_created_at must be a str, '
            f'got {window["max_created_at"]!r}'
        )
    # Parse for effect: an unparseable bound is a bad manifest, reported here
    # with the other criteria checks rather than surfacing later as a
    # re-derivation failure, which would name the wrong culprit.
    parse_timestamp(window['max_created_at'], field='criteria.window.max_created_at')
    episodes = manifest.get('episodes')
    if not isinstance(episodes, list) or not episodes:
        raise CorpusBuildError('manifest has no non-empty "episodes" list')
    for index, entry in enumerate(episodes):
        if not isinstance(entry, dict) or 'uuid' not in entry or 'content_hash' not in entry:
            raise CorpusBuildError(
                f'episodes[{index}] must be an object with uuid and content_hash, '
                f'got {entry!r}'
            )
    return criteria, episodes


def verify_manifest(manifest: object, population: list[EpisodeRecord]) -> VerifyReport:
    """Re-derive *manifest*'s sample from its OWN recorded criteria and compare.

    This is the deliverable's user-observable promise, mechanized: a reviewer
    holding the artifact and a live store can confirm the corpus is what it
    says it is.

    .. important::

       Verification re-runs :func:`select` from the recorded ``seed`` and ``n``
       and compares the RESULT to the recorded ``episodes``. It must never read
       ``episodes`` as the source of truth for what to select — that check
       would be vacuous, passing on any manifest including a tampered one, and
       a check that always says yes is worse than no check because it launders
       the artifact.

    **Re-derivation runs against the population as of the recorded window.**
    ``dark_factory`` is written continuously by the running fleet, so the live
    population is strictly larger every time anyone checks. Sampling from that
    grown population changes the cell denominators, which moves a
    largest-remainder seat, which changes the corpus — so an unwindowed
    re-derivation reports ``id_mismatch`` on a perfectly good artifact within
    minutes of building it, and the check degrades into noise a reviewer learns
    to ignore. ``criteria.window.max_created_at`` is therefore a **bound**, not
    a description: only episodes at or before it were ever in the sampling
    frame, and only they are fed to :func:`select` here.

    The window is deliberately applied to the re-derivation ONLY. ``missing``
    and the hash-drift lookup run against the FULL population, because "the
    store still holds what the corpus names, with the same bytes" is a
    different question from "what was the sampling frame" — windowing those
    would hide a real deletion behind a bound the manifest itself supplies.

    Failures are reported in priority order — a broken manifest before a
    missing episode, a missing episode before an id mismatch, an id mismatch
    before hash drift — because each is a precondition for the next being
    meaningful. Hashes cannot be compared against episodes that are not there,
    and drift in a selection that is already wrong is not the reader's problem
    yet.
    """
    try:
        criteria, episodes = _read_criteria(manifest)
    except CorpusBuildError as exc:
        return VerifyReport(status='bad_manifest', detail=str(exc))

    recorded_ids = [entry['uuid'] for entry in episodes]
    present = {record.uuid: record for record in population}
    missing = [uuid for uuid in recorded_ids if uuid not in present]
    if missing:
        return VerifyReport(
            status='missing_episodes',
            detail=(
                f'{len(missing)} of {len(recorded_ids)} manifest episodes are absent '
                f'from the population — the store no longer holds what the corpus '
                f'names'
            ),
            checked=len(recorded_ids),
            missing=missing,
        )

    try:
        cutoff = parse_timestamp(
            criteria['window']['max_created_at'], field='criteria.window.max_created_at'
        )
        windowed = [
            record for record in population if parse_timestamp(record.created_at) <= cutoff
        ]
        rederived = [
            record.uuid
            for record in select(windowed, criteria['n'], seed=criteria['seed']).selected
        ]
    except CorpusBuildError as exc:
        return VerifyReport(
            status='bad_manifest',
            detail=f'cannot re-derive from the recorded criteria: {exc}',
            checked=len(recorded_ids),
        )

    if rederived != recorded_ids:
        recorded_size = criteria.get('population_size')
        frame = ''
        if isinstance(recorded_size, int) and recorded_size != len(windowed):
            frame = (
                f'; the sampling frame does not reconcile either — the window holds '
                f'{len(windowed)} episodes now but the manifest recorded '
                f'{recorded_size}'
            )
        return VerifyReport(
            status='id_mismatch',
            detail=(
                f're-deriving from the recorded criteria (seed={criteria["seed"]!r}, '
                f'n={criteria["n"]}) produced a different corpus than the manifest '
                f'records{frame}'
            ),
            checked=len(recorded_ids),
            rederived=rederived,
            id_mismatch={
                'only_in_manifest': sorted(set(recorded_ids) - set(rederived)),
                'only_in_rederivation': sorted(set(rederived) - set(recorded_ids)),
                'order_differs': sorted(set(recorded_ids)) == sorted(set(rederived)),
            },
        )

    drift = {
        entry['uuid']: {
            'expected': entry['content_hash'],
            'actual': content_hash(present[entry['uuid']].content),
        }
        for entry in episodes
        if content_hash(present[entry['uuid']].content) != entry['content_hash']
    }
    if drift:
        return VerifyReport(
            status='hash_drift',
            detail=(
                f'{len(drift)} of {len(episodes)} episodes have different content than '
                f'when the manifest was built — the SELECTION is still correct, but '
                f'replay results computed against the old bytes are suspect'
            ),
            checked=len(recorded_ids),
            rederived=rederived,
            hash_drift=drift,
        )

    return VerifyReport(
        status='ok',
        detail=(
            f'{len(recorded_ids)} episodes re-derived from the recorded criteria and '
            f'matched, with every content hash intact'
        ),
        checked=len(recorded_ids),
        rederived=rederived,
    )


def record_field_names() -> tuple[str, ...]:
    """Return :class:`EpisodeRecord`'s field names, in declaration order.

    Used by the manifest to record what was actually projected, so the
    no-outcome-filter claim is checkable against the code rather than asserted
    in prose alone.
    """
    return tuple(f.name for f in dataclasses.fields(EpisodeRecord))


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(manifest: dict) -> str:
    """Render the stratification report as a terminal table.

    Rendered from the manifest rather than from the in-memory
    :class:`SelectionResult`, so what an operator reads on stdout and what a
    reviewer reads in the committed artifact are the same numbers by
    construction. A report computed independently could disagree with the file
    it describes, and the disagreement would be invisible.
    """
    criteria = manifest['criteria']
    report = manifest['stratification_report']
    cells: dict = report['cells']
    totals = report['totals']
    width = max([len('stratum'), *(len(label) for label in cells)])
    window = criteria['window']
    lines = [
        f'LME replay corpus — graph={criteria["graph"]} n={criteria["n"]} '
        f'seed={criteria["seed"]!r} builder_version={criteria["builder_version"]}',
        f'population: {criteria["population_size"]} episodes in {len(cells)} non-empty '
        f'cells, {window["min_created_at"]} .. {window["max_created_at"]}',
        '',
        f'{"stratum".ljust(width)}  population  allocated  selected',
        f'{"-" * width}  ----------  ---------  --------',
    ]
    lines += [
        f'{label.ljust(width)}  {cell["population"]:>10}  {cell["allocated"]:>9}  '
        f'{cell["selected"]:>8}'
        for label, cell in sorted(cells.items())
    ]
    lines.append(
        f'{"TOTAL".ljust(width)}  {totals["population"]:>10}  {totals["allocated"]:>9}  '
        f'{totals["selected"]:>8}'
    )
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """The CLI: flat mode flags, no subcommands.

    Every flag is a read or an output-location parameter. There is deliberately
    no flag that mutates the store — no ``--prune``, no ``--reindex``, no
    ``--apply`` — and a test pins this flag set by EQUALITY so one cannot be
    added without a test saying so out loud.
    """
    parser = argparse.ArgumentParser(
        # The module docstring carries RST tables that argparse's formatter
        # reflows into rubble (the memory_eval_retrieval_probe treatment); the
        # first line plus the two guarantees is what an operator needs here.
        description=(
            f'{(__doc__ or "LME replay corpus builder").splitlines()[0]}\n\n'
            'Strictly read-only: one GRAPH.RO_QUERY against the graph, one\n'
            'committed manifest written beside this script. Corpus membership\n'
            "is never conditioned on the incumbent pipeline's outcome."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help=(
            'Re-derive the manifest at --out from its OWN recorded criteria and '
            'compare. Catches a tampered seed, a drifted episode and a vanished '
            f'one, each on its own exit code {sorted(EXIT_CODES.values())}. '
            'Default mode is build.'
        ),
    )
    parser.add_argument(
        '--n',
        type=int,
        default=DEFAULT_N,
        help=(
            'Corpus size (default: %(default)s, the midpoint of the PRD 150-300 '
            'band). An n that cannot satisfy the min-1 floor, or that exceeds '
            f'the population, exits {EXIT_RUN_FAILED} rather than quietly '
            'producing a shorter corpus. Ignored under --verify, which uses the '
            'n the manifest recorded.'
        ),
    )
    parser.add_argument(
        '--seed',
        default=DEFAULT_SEED,
        help=(
            'Sampling seed (default: %(default)s). Fixed, never clock-derived: a '
            'random default would make the committed artifact irreproducible and '
            'a reviewer could not tell a re-seed from tampering. Ignored under '
            '--verify, which re-derives from the recorded seed.'
        ),
    )
    parser.add_argument(
        '--graph',
        default=DEFAULT_GRAPH,
        help=(
            'FalkorDB graph to read (default: %(default)s). Under --verify this '
            'is only a fallback: re-derivation reads the graph the manifest '
            'records, since verifying against a different graph would compare a '
            'corpus to a population it was never drawn from.'
        ),
    )
    parser.add_argument(
        '--out',
        default=DEFAULT_MANIFEST_PATH,
        help=(
            'Manifest path — written in build mode, read in --verify mode '
            '(default: %(default)s). The default is under scripts/ and NOT under '
            'data/, which fused-memory/.gitignore ignores; a manifest written '
            'there would be uncommittable and would fail the delivered_check.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help=(
            'Print the stratification report and write NOTHING. Use it to check '
            'an --n before committing an artifact, so a re-tune never leaves a '
            'half-considered manifest in the tree.'
        ),
    )
    return parser


def _fetch_population(graph_name: str) -> list[EpisodeRecord]:
    """Read the whole ``Episodic`` population from *graph_name*.

    ``EpisodeReader`` is looked up on the module at call time rather than
    captured, so a test can substitute an offline double for the whole CLI.
    """
    return asyncio.run(EpisodeReader(graph_name=graph_name).fetch_population())


def _run_build(args: argparse.Namespace) -> int:
    """Build mode: sample, report, and (unless ``--dry-run``) write."""
    population = _fetch_population(args.graph)
    result = select(population, args.n, seed=args.seed)
    manifest = build_manifest(
        result, population, n=args.n, seed=args.seed, graph=args.graph
    )
    print(render_report(manifest), end='')
    if args.dry_run:
        print(f'dry-run: nothing written (would have written {args.out})')
        return 0
    print(f'manifest: {write_manifest(manifest, args.out)}')
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    """Verify mode: re-derive the artifact at ``--out`` and exit on its status.

    The manifest is read BEFORE the store, so an unreadable artifact costs no
    query — and reports ``bad_manifest`` rather than surfacing whatever the
    store said, which would send the reader after the wrong problem.
    """
    path = Path(args.out)
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        return _emit_verdict(
            VerifyReport(status='bad_manifest', detail=f'cannot read {path}: {exc}')
        )
    except ValueError as exc:
        return _emit_verdict(
            VerifyReport(status='bad_manifest', detail=f'{path} is not valid JSON: {exc}')
        )

    # Re-derive against the graph the MANIFEST names — that is what "re-derive
    # from the recorded criteria" means. --graph is the fallback for a manifest
    # that records none.
    recorded_graph = None
    if isinstance(manifest, dict) and isinstance(manifest.get('criteria'), dict):
        candidate = manifest['criteria'].get('graph')
        recorded_graph = candidate if isinstance(candidate, str) and candidate else None
    population = _fetch_population(recorded_graph or args.graph)
    return _emit_verdict(verify_manifest(manifest, population))


def _emit_verdict(report: VerifyReport) -> int:
    """Print *report* and return its exit code.

    A clean verdict goes to stdout; every failure goes to stderr WITH its
    structured diff, so a caller that only reads stderr still sees which ids or
    hashes are at issue rather than a bare status word.
    """
    line = f'verify: {report.status} — {report.detail}'
    if report.ok:
        print(line)
        return report.exit_code
    print(line, file=sys.stderr)
    diff = {
        key: value
        for key, value in (
            ('id_mismatch', report.id_mismatch),
            ('hash_drift', report.hash_drift),
            ('missing', report.missing),
        )
        if value
    }
    if diff:
        print(json.dumps(diff, indent=2, sort_keys=True), file=sys.stderr)
    return report.exit_code


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        return _run_verify(args) if args.verify else _run_build(args)
    except CorpusBuildError as exc:
        # A typed build/read failure, reported as a message plus a documented
        # code rather than a traceback plus exit 1-by-accident. Distinct from
        # every EXIT_CODES verification status: nothing was verified.
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_RUN_FAILED


if __name__ == '__main__':
    sys.exit(main())
