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
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

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


def month_bucket(created_at: object) -> str:
    """Return the ``YYYY-MM`` time stratum for *created_at*.

    Accepts the ISO-8601 spellings the store has accumulated — ``+00:00``
    offset, ``Z`` suffix, naive, and a space separator — and normalizes them to
    one bucket per calendar month. Without that normalization a single month
    would split across strata purely by which code path persisted the row.

    Raises :class:`CorpusBuildError` on anything unparseable, rather than
    returning an ``'unknown'`` bucket. An ``'unknown'`` bucket would quietly
    become an extra stratum, and the stratification report would look complete
    while describing a corpus whose time axis had partially collapsed.
    """
    if not isinstance(created_at, str) or not created_at.strip():
        raise CorpusBuildError(
            f'created_at must be a non-empty ISO-8601 string, got {created_at!r}'
        )
    text = created_at.strip().replace(' ', 'T', 1)
    if text.endswith(('Z', 'z')):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CorpusBuildError(
            f'created_at is not a parseable ISO-8601 timestamp: {created_at!r} ({exc})'
        ) from exc
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


def record_field_names() -> tuple[str, ...]:
    """Return :class:`EpisodeRecord`'s field names, in declaration order.

    Used by the manifest to record what was actually projected, so the
    no-outcome-filter claim is checkable against the code rather than asserted
    in prose alone.
    """
    return tuple(f.name for f in dataclasses.fields(EpisodeRecord))
