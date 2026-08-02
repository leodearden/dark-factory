#!/usr/bin/env python3
"""Retro topic/canonical stamping sweep over KNOWN consolidated clusters.

Leaf θ of ``docs/prds/memory-metadata-vocabulary.md`` (task 3201).  Back-fills
``metadata.topic`` — and, where a cluster has one undisputed canonical,
``metadata.canonical: true`` — onto memories that were consolidated before
those vocabulary keys existed, so the retrieval layer can finally see the
clusters the curator already adjudicated.

What "bounded" means here (D11)
-------------------------------
This is NOT a corpus sweep.  Every target is addressed by **memory id**,
drawn from one of three enumerated, checkable sources:

1. the live ``canonical: true`` scroll (6 records at authoring time — 1 in
   ``dark_factory``, 5 in ``reify``), whose own ``consolidates`` /
   ``retires`` / ``replaces`` / ``supersedes`` uuid lists name their members;
2. curator-gate clusters — the reify half via the committed E1 topic
   registry (``tests/fixtures/memory_eval_topic_registry.json``) joined to
   3130's labeled fixture on ``cluster_id``; the dark_factory half via the
   committed :data:`DF_CURATOR_GATE_CLUSTERS` manifest in this module;
3. 3130's labeled calibration fixture
   (``tests/fixtures/write_triage_calibration.jsonl``), whose rows carry a
   ``memory_id`` and a curator ``label`` directly.

No content-hash-to-live-id resolution happens anywhere.  Resolving the E1
registry's 16-hex ``members`` would mean scrolling whole Mem0 categories and
hashing every record — precisely the unbounded sweep D11 and this task's
brief forbid.  Staying id-addressed is what makes boundedness *checkable*
rather than merely asserted.

Why this script enforces ε's rules itself
-----------------------------------------
Measured, not assumed: ``MemoryService.update_memory`` never calls
``_apply_memory_metadata_validation`` — that seam runs only from
``add_memory`` and ``add_system_record``.  So a metadata-only patch reaches
Qdrant with **no** slug-shape check and **no** canonical-uniqueness probe.

θ is the one caller stamping ``canonical: true`` at scale, and it is exactly
the corpus ε's ``memory_metadata.enforce`` flag cannot yet be flipped
against.  Leaning on a seam this script does not traverse would let it write
a second canonical for a topic, or a non-conforming slug, with equal
silence — re-creating the defect ε shipped warn-mode to avoid.  So the two
probes are re-expressed here against the injected service.  That is a
deliberate, narrow INV-5 exception: the alternative — routing θ through
``add_memory`` — would re-embed the content and change point ids, breaking
the whole in-place-update contract this script depends on
(``plans/mem0-in-place-update-decision.md`` §3).

The *rules themselves* are still single-homed.  The slug shape and its cap
come from :mod:`fused_memory.topic_slug` by import (INV-5); only the
*probe order* is duplicated.

``supersedes`` normalization (PRD D2)
-------------------------------------
D2 says the scalar->list fold "rides θ's sweep where it touches entries
anyway".  That is a convenience, not a mandate to manufacture a validation
failure: two of the six live canonical records carry ``supersedes`` as an
English *sentence*, and blindly wrapping it would produce a one-member list
that fails ``_is_full_uuid``.  So the fold is applied only when the scalar
parses as a full UUID; prose is left byte-identical and reported.

Usage
-----
  # Dry run (default): plan everything, print the report, touch nothing.
  python scripts/retro_stamp_topics.py

  # Commit the stamps.
  python scripts/retro_stamp_topics.py --apply
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fused_memory.memory_metadata import _is_full_uuid, normalize_supersedes
from fused_memory.topic_slug import TOPIC_SLUG_MAX_LEN, is_valid_topic_slug

__all__ = [
    'CLUSTER_MEMBER_KEYS',
    'TOPIC_SLUG_MAX_LEN',
    'ClusterPlan',
    'PatchDecision',
    'compute_patch',
    'derive_topic_slug',
    'is_valid_topic_slug',
    'normalize_supersedes',
    'plan_canonical_clusters',
]


# ---------------------------------------------------------------------------
# Pure core — derivation
# ---------------------------------------------------------------------------

#: Any run of characters that cannot appear inside a slug segment.  Note the
#: complement class is ``[a-z0-9]`` only: ``_`` is NOT preserved, which is the
#: whole point of the fold (98 of 352 live topic values are snake_case).
#:
#: This is deliberately NOT the anchored slug validator — that lives once, in
#: :mod:`fused_memory.topic_slug`, and is called below.  Two different
#: patterns doing two different jobs; the *verdict* has one home.
_NON_SLUG_RUN_RE = re.compile(r'[^a-z0-9]+')


def derive_topic_slug(value: object) -> str | None:
    """Fold *value* into ε's topic-slug shape, or ``None`` if it cannot be.

    The fold: lowercase, strip, collapse every run of non-``[a-z0-9]``
    characters (which includes ``_``, so snake_case becomes hyphen-case) to a
    single ``-``, then strip leading/trailing hyphens.  The result is returned
    **only** if :func:`fused_memory.topic_slug.is_valid_topic_slug` accepts
    it — which is also where the ``TOPIC_SLUG_MAX_LEN`` cap is enforced.

    Returning ``None`` rather than a repaired value is load-bearing.  An
    over-long topic truncated to 100 chars, or ``'!!!'`` turned into
    ``'unnamed-topic'``, would file a record under a topic no human chose;
    the caller instead reports it and moves on (loud over silent).

    NOT a copy of ``memory_eval_retrieval_probe._slugify``, and the two must
    not be "unified": that one preserves ``_`` and falls back to
    ``'unnamed-topic'``, so it emits slugs ε *rejects*.  It is right for its
    own job (naming derivation candidates for human review) and wrong for
    this one (writing a validated vocabulary key to the corpus).

    Args:
        value: Any object.  A non-``str`` is a ``None`` verdict, matching
            ``is_valid_topic_slug``'s "non-str is False" convention — both
            are handed untrusted values off live records and fixtures.

    Returns:
        The conforming slug, or ``None`` when no honest fold exists.
    """
    if not isinstance(value, str):
        return None
    folded = _NON_SLUG_RUN_RE.sub('-', value.strip().lower()).strip('-')
    return folded if is_valid_topic_slug(folded) else None


@dataclass(frozen=True)
class PatchDecision:
    """What to write to one record, and why.

    Attributes:
        patch: The metadata patch to hand to ``update_memory``.  **Empty
            means issue no call at all** — see :func:`compute_patch`.
        dispositions: Ordered, deduplicated reasons.  Every key considered
            contributes exactly one, whether or not it produced a write, so
            the report can distinguish "already correct" from "never looked
            at" — the two an absent line would otherwise conflate.
    """

    patch: dict = field(default_factory=dict)
    dispositions: tuple[str, ...] = ()


def compute_patch(
    existing_metadata: dict,
    *,
    target_topic: str,
    make_canonical: bool,
) -> PatchDecision:
    """Decide the minimal metadata patch for one record.

    The idempotence heart of the sweep.  A patch is emitted only for keys
    that would actually change; when nothing would, ``patch`` is empty and
    the caller must issue **no** ``update_memory`` — not an update that
    happens to be a no-op.  That is what makes a second run cost zero writes
    rather than zero net effect, and it is why ``run``'s "stamped" count is
    an honest measure of what the corpus gained.

    Pure: takes a plain dict, touches no service, mutates nothing.

    Topic rules
    -----------
    * absent -> stamp *target_topic* (``topic_stamped``);
    * present and folding to *target_topic* already in conforming shape ->
      no write (``topic_already_present``);
    * present in a shape that folds TO *target_topic* (the snake_case twin)
      -> rewrite in place (``topic_normalized``).  This is the normalization
      ε's enforcement note delegates here, and the precondition for flipping
      ``memory_metadata.enforce``;
    * present and folding to something else, or unfoldable -> **refuse**
      (``conflicting_existing_topic``).  A retro sweep must not be able to
      destroy a topic a human set.  An unfoldable existing value means no
      honest comparison is available, which is a reason to refuse rather
      than a licence to overwrite.

    Args:
        existing_metadata: The record's current metadata, as read live.
        target_topic: The slug this cluster resolved to.  Already validated
            by the planner that produced it.
        make_canonical: Whether this record is its cluster's undisputed
            canonical.  Never demotes: ``False`` emits nothing.

    Returns:
        A :class:`PatchDecision`; ``patch`` may be empty.
    """
    patch: dict = {}
    dispositions: list[str] = []

    existing_topic = existing_metadata.get('topic')
    if existing_topic is None:
        patch['topic'] = target_topic
        dispositions.append('topic_stamped')
    elif existing_topic == target_topic:
        dispositions.append('topic_already_present')
    elif derive_topic_slug(existing_topic) == target_topic:
        # Same topic, legacy shape (e.g. eval_worktree_plan_tools_missing ->
        # eval-worktree-plan-tools-missing). Rewriting is normalization, not
        # reassignment — the fold is what proves they are the same fact.
        patch['topic'] = target_topic
        dispositions.append('topic_normalized')
    else:
        dispositions.append('conflicting_existing_topic')

    # Canonical: stamping only. `make_canonical=False` emits NOTHING — not a
    # `False` (which would be a positive claim θ has no basis for) and not a
    # deletion. A cluster whose canonical is disputed or plural leaves the
    # sweep with a topic on every member and an opinion about none.
    if make_canonical:
        if existing_metadata.get('canonical') is True:
            dispositions.append('canonical_already_present')
        else:
            patch['canonical'] = True
            dispositions.append('canonical_stamped')

    # supersedes (PRD D2): fold scalar -> list, but ONLY when every member
    # would still be a resolvable id afterwards. `normalize_supersedes`
    # faithfully wraps any scalar — including the English sentences two live
    # canonical records carry — so folding unconditionally would write a
    # one-member list that fails `_is_full_uuid`, converting a merely-legacy
    # shape into an outright validation failure. Both helpers are imported
    # rather than re-expressed (INV-5).
    if 'supersedes' in existing_metadata:
        raw = existing_metadata['supersedes']
        members = normalize_supersedes(raw)
        if not all(_is_full_uuid(m) for m in members):
            dispositions.append('supersedes_not_normalizable')
        elif isinstance(raw, list):
            # Already the target shape — no write, so run two stays empty.
            dispositions.append('supersedes_already_normalized')
        else:
            patch['supersedes'] = members
            dispositions.append('supersedes_normalized')

    return PatchDecision(patch=patch, dispositions=tuple(dispositions))


# ---------------------------------------------------------------------------
# Pure core — planning source (1): the live `canonical: true` scroll
# ---------------------------------------------------------------------------

#: The metadata keys a canonical record uses to name the records it absorbed.
#:
#: Measured across the six live ``canonical: true`` records (2026-08-02), not
#: guessed: ``consolidates`` (dark_factory 0929cff6, reify dbc478b8),
#: ``retires`` + SCALAR ``replaces`` (reify 28cbf1c4), list ``supersedes``
#: (reify bbc063a7, dbc478b8).  A list-only harvester would silently drop the
#: scalar ``replaces`` edge.
#:
#: A named, ORDERED constant rather than a literal inside the loop: the order
#: fixes member ordering, so two dry-runs of the same corpus diff cleanly.
#:
#: Deliberately excluded: ``replaces_verbatim`` (reify bbc063a7) names the one
#: record this entry reissues rather than a cluster member, and is already
#: covered by that record's ``supersedes``.
CLUSTER_MEMBER_KEYS: tuple[str, ...] = (
    'consolidates',
    'retires',
    'replaces',
    'supersedes',
)


@dataclass(frozen=True)
class ClusterPlan:
    """One cluster to stamp: a topic, its members, and maybe its canonical.

    Attributes:
        topic: The slug every member (and the canonical) gets.  Always
            already validated by the planner that built it.
        project_id: Which corpus the ids live in.  Sources span two
            projects, so this cannot be a run-level constant.
        canonical_memory_id: The one record to also stamp ``canonical:
            true``, or ``None`` when the cluster has no undisputed
            canonical.  ``None`` is a normal outcome, not a defect.
        member_memory_ids: Deduplicated, order-stable, never containing
            ``canonical_memory_id``.
        source: Which of the three bounded sources produced this plan.
            Carried into the report so a stamped record is traceable to the
            enumeration that justified it.
        canonical_plural: The cluster's end-state has SEVERAL canonical
            entries (DF gate 3036), so no single record may be stamped.
        provenance: Free-form audit context (gate task id, cluster id, …).
    """

    topic: str
    project_id: str
    canonical_memory_id: str | None
    member_memory_ids: tuple[str, ...]
    source: str
    canonical_plural: bool = False
    provenance: dict = field(default_factory=dict)


def _harvest_member_ids(
    metadata: dict, *, memory_id: str, skips: list[dict]
) -> tuple[str, ...]:
    """Collect cluster member ids from *metadata*'s member-bearing keys.

    Every value is shaped by :func:`normalize_supersedes` (which tolerates
    both the scalar and list spellings without coercing members) and then
    filtered by ``_is_full_uuid``.  A value that survives neither is appended
    to *skips* rather than dropped: a member the sweep could not resolve is a
    fact about the cluster, and a report that omitted it would claim a clean
    pass over membership it never actually determined.

    The record's own id is excluded — a self-reference would otherwise make
    ``stamp_one`` visit it twice with contradictory ``make_canonical``.
    """
    seen: dict[str, None] = {}
    for key in CLUSTER_MEMBER_KEYS:
        if key not in metadata:
            continue
        for member in normalize_supersedes(metadata[key]):
            if not _is_full_uuid(member):
                skips.append({
                    'reason': 'member_not_a_uuid',
                    'memory_id': memory_id,
                    'key': key,
                    'value': member,
                })
                continue
            if member == memory_id:
                continue
            seen.setdefault(member, None)
    return tuple(seen)


def plan_canonical_clusters(
    records: list[dict], *, project_id: str
) -> tuple[list[ClusterPlan], list[dict]]:
    """Plan one cluster per live ``canonical: true`` record.

    Source (1) of the bounded set.  Each record already declares both halves
    of its cluster: its own id is the canonical, and its member-bearing keys
    name the records it absorbed.  Nothing here is inferred from content.

    Pure: *records* in, ``(plans, skips)`` out.  No service, no I/O, and
    *records* is not mutated.

    A record whose ``topic`` cannot be folded to a conforming slug produces
    NO plan and a ``topic_underivable`` skip.  Not a guessed slug (that would
    file a whole cluster under a topic nobody chose) and not an empty plan
    (that would read as a cluster the sweep handled).

    Args:
        records: Scroll-shaped ``{'id', 'created_at', 'metadata'}`` dicts,
            exactly as ``get_memories_by_metadata`` returns them.
        project_id: The corpus *records* were scrolled from.

    Returns:
        ``(plans, skips)`` — skips are report-ready dicts each carrying a
        ``reason`` and the ``memory_id`` it concerns.
    """
    plans: list[ClusterPlan] = []
    skips: list[dict] = []
    for record in records:
        memory_id = record['id']
        metadata = record.get('metadata') or {}
        topic = derive_topic_slug(metadata.get('topic'))
        if topic is None:
            skips.append({
                'reason': 'topic_underivable',
                'memory_id': memory_id,
                'project_id': project_id,
                'raw_topic': metadata.get('topic'),
            })
            continue
        members = _harvest_member_ids(metadata, memory_id=memory_id, skips=skips)
        plans.append(ClusterPlan(
            topic=topic,
            project_id=project_id,
            canonical_memory_id=memory_id,
            member_memory_ids=members,
            source='canonical_scroll',
            provenance={'raw_topic': metadata.get('topic')},
        ))
    return plans, skips
