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

import argparse
import asyncio
import functools
import importlib.util
import json
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from fused_memory.memory_metadata import _is_full_uuid, normalize_supersedes
from fused_memory.topic_slug import TOPIC_SLUG_MAX_LEN, is_valid_topic_slug

__all__ = [
    'CALIBRATION_FIXTURE_PATH',
    'CLUSTER_MEMBER_KEYS',
    'DF_CURATOR_GATE_CLUSTERS',
    'DEFAULT_JSON_OUT',
    'DEFAULT_MD_OUT',
    'DEFAULT_PROJECTS',
    'ERROR_OUTCOMES',
    'GateCluster',
    'SKIP_BUCKETS',
    'SOURCE_DESCRIPTIONS',
    'WRITE_REASON',
    'WRITE_SOURCE',
    'main',
    'render_json',
    'render_markdown',
    'resolve_exit_code',
    'resolve_projects',
    'run',
    'stamp_one',
    'TOPIC_REGISTRY_PATH',
    'TOPIC_SLUG_MAX_LEN',
    'ClusterPlan',
    'PatchDecision',
    'compute_patch',
    'derive_topic_slug',
    'filter_plans_to_projects',
    'is_valid_topic_slug',
    'load_calibration_rows',
    'load_topic_registry',
    'normalize_supersedes',
    'StampTarget',
    'merge_plans',
    'plan_calibration_clusters',
    'plan_canonical_clusters',
    'plan_gate_clusters',
]

#: This script lives at ``<repo>/fused-memory/scripts/``.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent

#: 3130's labeled curator dataset — source (3).  104 rows over 20 clusters,
#: each row carrying ``memory_id``, ``cluster_id`` and a curator ``label``.
CALIBRATION_FIXTURE_PATH = str(
    _PACKAGE_ROOT / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'
)

#: The committed E1 topic registry — the hand-authored topic slugs, keyed by
#: ``provenance.cluster_id``.  Its 20 ``derived_from: curator_gate`` entries
#: are 3112's reify gate census already materialized and human-completed,
#: which is what makes it the reify half of source (2) as well.
TOPIC_REGISTRY_PATH = str(
    _PACKAGE_ROOT / 'tests' / 'fixtures' / 'memory_eval_topic_registry.json'
)

_PROBE_SCRIPT_PATH = _PACKAGE_ROOT / 'scripts' / 'memory_eval_retrieval_probe.py'


@functools.cache
def _load_probe_module() -> types.ModuleType:
    """Load ``memory_eval_retrieval_probe`` by path.

    ``scripts/`` is not a package, so a plain import cannot reach it — the
    same importlib idiom this script's own tests use to reach this script.

    Memoized on THIS function, not merely in ``sys.modules``: repeat calls
    must return the same module object the aliases below were bound from,
    and ``sys.modules`` is a shared slot any other by-path loader of the
    same script may overwrite with a freshly executed module (the probe's
    own test module does exactly that).  Reading the slot on every call
    would then hand back a module whose ``RegistryEntry`` is a *different
    class object* than the one this module re-exports — silently breaking
    any ``isinstance`` check across the seam.  ``sys.modules`` is still
    consulted for the FIRST resolution so an already-imported probe is not
    executed twice.
    """
    mod_name = 'memory_eval_retrieval_probe'
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, _PROBE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {_PROBE_SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_probe = _load_probe_module()

#: The registry loader and its model, REUSED rather than re-parsed.  It is
#: required-strict (a malformed entry fails by name) and additive-tolerant —
#: ``RegistryEntry.extra`` exists precisely so this consumer can load a
#: richer entry without forcing a fixture rewrite; its docstring names 3201.
load_topic_registry = _probe.load_topic_registry
RegistryEntry = _probe.RegistryEntry
TopicRegistry = _probe.TopicRegistry
Canonical = _probe.Canonical
Phrasing = _probe.Phrasing


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


# ---------------------------------------------------------------------------
# Pure core — planning sources (2)-reify and (3): ONE join on cluster_id
# ---------------------------------------------------------------------------

#: The curator label that means "this row is a member of the cluster".
#:
#: ONLY ``duplicate``.  ``distinct`` and ``pseudo_contradiction`` rows were
#: adjudicated as separate claims that merely READ as contradictory — the
#: rule ``_derive_curator_gate_candidates`` documents.  θ reuses it rather
#: than inventing a second, looser notion of "cluster", because two
#: definitions of the same word are how they drift apart.
_MEMBER_LABEL = 'duplicate'
_CANONICAL_LABEL = 'canonical'


def load_calibration_rows(path: str | Path) -> list[dict]:
    """Read 3130's labeled calibration fixture (JSONL, one row per line).

    Blank lines are tolerated; a malformed line raises, because a fixture
    this sweep reads ids out of is not a place to guess.
    """
    rows: list[dict] = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plan_calibration_clusters(
    rows: list[dict], registry: object
) -> tuple[list[ClusterPlan], list[dict]]:
    """Join 3130's labeled rows to the E1 registry on ``cluster_id``.

    One join resolves TWO of the bounded sources: source (3) outright, and
    the reify half of source (2) — the registry's ``derived_from:
    curator_gate`` entries are 3112's reify gate census, already materialized
    and topic-completed by a human.

    The join direction matters.  The rows carry ids and labels but NO topic;
    the registry carries the hand-authored slug, which is the part a machine
    cannot regenerate.  Re-deriving instead would send
    ``_derive_curator_gate_candidates`` down its ``_slugify(cluster_id)``
    fallback and mint a slugified UUID as a topic — so a cluster with no
    registry entry is skipped rather than stamped.

    Pure: no I/O (the caller loads both fixtures), no mutation of *rows*.

    Args:
        rows: Calibration rows, as :func:`load_calibration_rows` returns.
        registry: A ``TopicRegistry`` — anything exposing ``.entries``.

    Returns:
        ``(plans, skips)``.
    """
    by_cluster: dict[str, object] = {}
    for entry in getattr(registry, 'entries', ()):
        cluster_id = (getattr(entry, 'provenance', None) or {}).get('cluster_id')
        if cluster_id:
            by_cluster.setdefault(str(cluster_id), entry)

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        cluster_id = row.get('cluster_id')
        if cluster_id:
            grouped.setdefault(str(cluster_id), []).append(row)

    plans: list[ClusterPlan] = []
    skips: list[dict] = []
    for cluster_id, group in sorted(grouped.items()):
        entry = by_cluster.get(cluster_id)
        if entry is None:
            skips.append({
                'reason': 'no_registry_topic',
                'cluster_id': cluster_id,
                'row_count': len(group),
            })
            continue
        topic = derive_topic_slug(getattr(entry, 'topic', None))
        if topic is None:
            skips.append({
                'reason': 'topic_underivable',
                'cluster_id': cluster_id,
                'raw_topic': getattr(entry, 'topic', None),
            })
            continue
        project_id = getattr(entry, 'project_id', None) or 'reify'

        canonical_id: str | None = None
        members: dict[str, None] = {}
        for row in group:
            label = row.get('label')
            if label not in (_CANONICAL_LABEL, _MEMBER_LABEL):
                continue
            memory_id = row.get('memory_id')
            # The isinstance arm is redundant at runtime (`_is_full_uuid`
            # rejects a non-str) but not to a reader or a type checker: the
            # row came off a JSONL fixture, so `memory_id` is genuinely
            # `Any | None` here and the narrowing has to be visible.
            if not isinstance(memory_id, str) or not _is_full_uuid(memory_id):
                skips.append({
                    'reason': 'member_not_a_uuid',
                    'cluster_id': cluster_id,
                    'key': 'memory_id',
                    'value': memory_id,
                })
                continue
            if label == _CANONICAL_LABEL:
                if canonical_id is None:
                    canonical_id = memory_id
                continue
            members.setdefault(memory_id, None)

        if canonical_id is None:
            # Not a blocker: the members are still a real cluster and still
            # benefit from a topic. What the sweep must not do is PICK one.
            skips.append({
                'reason': 'cluster_without_canonical',
                'cluster_id': cluster_id,
                'topic': topic,
            })
        members.pop(canonical_id, None)  # type: ignore[arg-type]

        plans.append(ClusterPlan(
            topic=topic,
            project_id=project_id,
            canonical_memory_id=canonical_id,
            member_memory_ids=tuple(members),
            source='calibration_registry_join',
            provenance={
                'cluster_id': cluster_id,
                'gate_ids': (getattr(entry, 'provenance', None) or {}).get('gate_ids'),
            },
        ))
    return plans, skips


# ---------------------------------------------------------------------------
# Committed manifest — source (2), dark_factory half
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCluster:
    """One dark_factory curator gate, transcribed for review.

    Attributes:
        gate_task_id: The gate task these ids were read from.
        topic: Hand-authored slug.  A machine cannot invent this — the gate
            tasks carry no topic field.
        member_memory_ids: Transcribed ids, or ``None`` when the gate carries
            NO id enumeration at all.  ``None`` and ``()`` mean different
            things: the former is "we could not enumerate", the latter would
            be "a gate with no members", which is false.
        canonical_memory_id: The one undisputed canonical, if the gate names
            one.
        canonical_plural: The gate's own end-state has SEVERAL canonicals.
        source_key: The gate-task metadata key(s) the ids came from, so an
            auditor can re-derive the transcription instead of trusting it.
            Empty exactly when ``member_memory_ids is None``.
        no_enumeration_reason: Why no ids could be transcribed.  Required
            when ``member_memory_ids is None``; this is what turns "covered
            4 of 7 gates" from silence into a report line.
        note: Anything else a reader needs to judge the row — most often
            that the transcription is narrower than the gate's title claims.
    """

    gate_task_id: str
    topic: str
    project_id: str = 'dark_factory'
    member_memory_ids: tuple[str, ...] | None = None
    canonical_memory_id: str | None = None
    canonical_plural: bool = False
    source_key: tuple[str, ...] = ()
    no_enumeration_reason: str | None = None
    note: str | None = None


#: The seven dark_factory curator-gate clusters PRD D11 enumerates.
#:
#: A committed, reviewable table rather than a live task-store read — the
#: same human-reviewable-constants convention ``consolidate_namespace_families
#: .py`` documents for its alias maps.  Three reasons it is not read live:
#: the seven gates' enumeration keys are genuinely heterogeneous, a live read
#: would need a task-backend dependency this script otherwise does not have,
#: and it would still hit the same three-gate hole.
#:
#: Ids transcribed 2026-08-02 from ``.taskmaster/tasks/tasks.db``.  A
#: transcription is not a corroboration: ``stamp_one`` re-reads every record
#: before writing and reports ``memory_not_found`` for ids consolidated away
#: since the gate ran (INV-3).  Measured example — gate 3036's
#: ``19705df4`` no longer resolves.
DF_CURATOR_GATE_CLUSTERS: tuple[GateCluster, ...] = (
    # 2969 [done] "Human-gated merge: consolidate duplicate systemctl-fixture
    # procedural_knowledge memories (e3ca4270 + 93720d2b)".
    # The two ids appear ONLY in the title, as 8-hex prefixes — not resolvable
    # ids, and not in metadata at all.
    GateCluster(
        gate_task_id='2969',
        topic='systemctl-fixture-duplicates',
        no_enumeration_reason=(
            'gate task carries no id-enumeration metadata key; the two members '
            'appear only in the title as 8-hex prefixes (e3ca4270, 93720d2b), '
            'which are not resolvable memory ids'
        ),
    ),
    # 2973 [done] "Human gate: consolidate ~9-10 duplicate 'architect
    # plan-revalidation after requeue/lock' procedural_knowledge entries".
    GateCluster(
        gate_task_id='2973',
        topic='architect-plan-revalidation-after-requeue',
        no_enumeration_reason=(
            'gate task carries no id-enumeration metadata key, and the title '
            'names no ids at all — only an approximate count ("~9-10")'
        ),
    ),
    # 3011 [done] "Human gate: consolidate 12-entry duplicate 'architect
    # report_task_already_done requires main-reachable commit'
    # procedural_knowledge cluster (subcase-tagged)".
    GateCluster(
        gate_task_id='3011',
        topic='report-task-already-done-main-reachable-commit',
        source_key=('cited_memories',),
        member_memory_ids=(
            '742651f0-eccd-4cb7-bc4f-9db2d980d9c3',
            'e584c016-8914-4c86-8296-97e48e1c3502',
            'b99c3c8b-c5b4-4f4d-810d-2096ff31b270',
        ),
        note=(
            'PARTIAL: the title states a 12-entry cluster but metadata '
            'enumerates 3. Only the enumerated ids are stamped; the other 9 '
            'are not silently claimed as covered'
        ),
    ),
    # 3016 [done] "Human gate: consolidate 3-entry duplicate
    # pooled-worktree-lane .venv EXDEV procedural_knowledge cluster".
    GateCluster(
        gate_task_id='3016',
        topic='pooled-worktree-lane-venv-exdev',
        no_enumeration_reason=(
            'gate task carries no id-enumeration metadata key, and the title '
            'names no ids — only a count ("3-entry")'
        ),
    ),
    # 3036 [done] "Human gate: consolidate 7-entry duplicate
    # pyright-worktree-import-resolution procedural_knowledge cluster into
    # package/precondition-tagged canonical ENTRIES" — plural in the gate's
    # own words, so topic on all and canonical on none.
    GateCluster(
        gate_task_id='3036',
        topic='pyright-worktree-import-resolution',
        source_key=('memory_ids',),
        canonical_plural=True,
        member_memory_ids=(
            '19705df4-3f01-41cb-8eab-41624a160a00',
            '8ef8cf8d-4bd7-45ff-b99a-4f2e0d1d4855',
            '9cf1e664-8a27-4e1e-9d3d-c6c210476142',
            'ae4a759a-2405-4f9d-875d-9ac2433c87c7',
            'b3d4d44e-1906-45dc-b156-f894264f3b25',
            'adcfa20b-7d2a-4eb1-9d83-8f6a5c871f25',
            '8fcdafbc-6776-4875-bc29-3ce3724dc794',
            '4a74ed20-6076-41a2-b967-7e339883e800',
        ),
        note=(
            'the gate resolved to SEVERAL package/precondition-tagged canonical '
            'entries, so no single record may be stamped canonical'
        ),
    ),
    # 3063 [cancelled] "Human gate: confirm task 3053 cancellation was
    # deliberate + decide fate of stale merge_request branch-arg Mem0 cluster
    # (d1e1b490 + ~10 legacy repeats)".
    # Members are the VALUES of `legacy_cluster_candidates_resolved` (a
    # short-prefix -> full-uuid map); `legacy_cluster_candidates_unverified`
    # is empty. `mem0_entry` is the 8-hex prefix 'd1e1b490' and is deliberately
    # NOT transcribed — a truncated prefix is not an id, and stamping one
    # would produce a memory_not_found line indistinguishable from a genuinely
    # consolidated-away record.
    # The topic matches reify canonical dbc478b8's: same claim, two corpora,
    # one namespace (D4).
    GateCluster(
        gate_task_id='3063',
        topic='merge-request-bare-task-id-branch-arg',
        source_key=('legacy_cluster_candidates_resolved', 'canonical_corrected_entry'),
        canonical_memory_id='6e7b0075-f50a-490e-9ab5-9fae07bedb18',
        member_memory_ids=(
            '9e36e654-6bff-4a2a-aa5b-8bc41db52f3b',
            'bcd24aae-923b-4215-b4d8-502d3acf3784',
            'fe97780b-c1ee-487e-b0dd-dbf417dae039',
            '70f59809-874e-4cd3-a104-c6e7aaa24ba0',
            '46f68b53-21f1-4e22-a446-7aca40ce653f',
            'd0fb45c7-3f6d-46d8-a159-f5d0cebf7e03',
            'c7be9c72-ab31-4b72-974e-9f542069eafe',
            '20d3db17-12cb-4e3c-b5a8-bda5cc6d1563',
            '12b1f775-ac45-48f9-9ca1-25cdfc55d7f7',
            '90cdfe97-adc8-4785-8518-7a56183565e3',
        ),
        note=(
            'gate task is cancelled, but its cluster and its corrected '
            "canonical were both identified; the 8-hex 'mem0_entry' prefix "
            'is deliberately not transcribed'
        ),
    ),
    # 3092 [blocked] "Human gate: consolidate 7-entry stale
    # pre-commit-pyright-timeout procedural_knowledge cluster (superseded by
    # task 2551)".
    # THREE ids carry the canonical fact, so — same rule as 3036 — every id
    # gets the topic and none gets `canonical`.
    GateCluster(
        gate_task_id='3092',
        topic='pre-commit-pyright-timeout',
        source_key=('stale_cluster_memory_ids', 'canonical_fact_memory_ids'),
        canonical_plural=True,
        member_memory_ids=(
            '39b84e1d-0849-46f9-a8d5-dfbae412ec80',
            'f63e5a50-324f-48d9-81d6-a256d437c9ea',
            '56c00b29-0a18-4a74-ab11-149eaae0c91e',
            'ae49326e-9e62-4356-be03-8dfd94ca6b4b',
            'edf1a3e9-87e9-4315-98d6-718cbd2a6852',
            '3efcef36-88cd-4479-ba91-cc7087daba67',
            '5fbc3cc5-22d5-4a4c-84e2-1569d0ad4694',
            'e559d60f-c325-40da-83ac-6f56c41dc7ad',
            '175daec6-a4e5-4b54-89bd-ceb134f25334',
            '1109af2a-9b99-449f-a842-592ce65b9f40',
        ),
        note=(
            'the first 7 are the stale cluster, the last 3 carry the canonical '
            'fact — three canonicals means none may be stamped'
        ),
    ),
)


def plan_gate_clusters(
    manifest: tuple[GateCluster, ...],
) -> tuple[list[ClusterPlan], list[dict]]:
    """Turn the committed gate manifest into cluster plans.

    Pure.  A gate declaring ``member_memory_ids is None`` produces NO plan and
    a ``skipped_no_enumeration`` entry naming the gate and the reason — the
    no-silent-caps rule.  An empty plan would look like a gate the sweep
    handled; an absent line would look like a gate that did not exist.
    """
    plans: list[ClusterPlan] = []
    skips: list[dict] = []
    for entry in manifest:
        if entry.member_memory_ids is None:
            skips.append({
                'reason': 'skipped_no_enumeration',
                'gate_task_id': entry.gate_task_id,
                'topic': entry.topic,
                'detail': entry.no_enumeration_reason or '',
            })
            continue
        plans.append(ClusterPlan(
            topic=entry.topic,
            project_id=entry.project_id,
            canonical_memory_id=entry.canonical_memory_id,
            member_memory_ids=entry.member_memory_ids,
            source='df_curator_gate_manifest',
            canonical_plural=entry.canonical_plural,
            provenance={
                'gate_task_id': entry.gate_task_id,
                'source_key': list(entry.source_key),
                'note': entry.note,
            },
        ))
    return plans, skips


def filter_plans_to_projects(
    plans: list[ClusterPlan],
    projects: tuple[str, ...],
) -> tuple[list[ClusterPlan], list[dict]]:
    """Narrow *plans* to the selected corpora, naming everything dropped.

    Pure.  ``--project`` is the operator's blast-radius bound, so it has to
    bound EVERY source rather than only the live scroll it happens to be
    threaded through: a planner that reads its ``project_id`` from a
    registry entry or a committed manifest is otherwise unaffected by the
    narrowing, and a run scoped to one corpus writes into the other.

    Two properties this function exists to guarantee:

    * **Named, never silent.** Each drop returns a
      ``out_of_scope_project`` skip carrying the project, topic and source,
      so the artifact still states what the narrowing excluded.  A sweep
      that quietly covers less than its own report implies is the failure
      mode the whole no-silent-caps rule guards against.
    * **Not an error.** Deliberate narrowing is a scope statement, so
      ``out_of_scope_project`` stays OUT of :data:`ERROR_OUTCOMES` and the
      run still exits 0.

    Order is preserved: this call narrows, it does not re-rank.

    Args:
        plans: Any planner's output.
        projects: The selected corpora.  Empty drops everything — which is
            what an empty selection means — rather than raising.

    Returns:
        ``(kept, skips)``.
    """
    kept: list[ClusterPlan] = []
    skips: list[dict] = []
    for plan in plans:
        if plan.project_id in projects:
            kept.append(plan)
            continue
        skips.append({
            'reason': 'out_of_scope_project',
            'project_id': plan.project_id,
            'topic': plan.topic,
            'source': plan.source,
        })
    return kept, skips


# ---------------------------------------------------------------------------
# Pure core — merging three plan lists into one per-record work list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StampTarget:
    """One record to stamp, and what to stamp on it.

    The unit ``run`` iterates.  Everything ambiguous has already been
    resolved into a skip by :func:`merge_plans`, so a target is unconditional
    intent: this id gets this topic, and possibly ``canonical``.
    """

    project_id: str
    memory_id: str
    topic: str
    make_canonical: bool
    source: str


def merge_plans(*plan_lists: list[ClusterPlan]) -> tuple[list[StampTarget], list[dict]]:
    """Flatten the three planners into a deduplicated, ordered work list.

    Overlap between sources is EXPECTED — the live canonical scroll and the
    E1 registry genuinely cover some of the same topics — so two plans for
    one ``(project_id, topic)`` merge: members union, sources combine.

    Genuine ambiguity is never resolved here, only reported:

    * two sources naming DIFFERENT canonicals for one topic ->
      ``canonical_disagreement``: stamp the topic on every member, canonical
      on none.  Picking one risks stamping the wrong record, and since the
      loser keeps no marker it would also erase the evidence that anything
      was in doubt;
    * a ``canonical_plural`` cluster -> ``canonical_plural``, same treatment;
    * one memory id claimed by two topics -> ``conflicting_topic_claim``, and
      that id is stamped by neither.  Letting iteration order decide would
      make the corpus depend on which planner ran first, invisibly.

    The result upholds ε's <=1-canonical-per-``(project_id, topic)`` rule
    BEFORE any write — necessary because ``update_memory`` never reaches the
    service-side uniqueness probe, so a violating plan would reach Qdrant
    unchallenged.

    Output is sorted by ``(project_id, topic, memory_id)`` so two dry-runs of
    an unchanged corpus produce byte-comparable reports.
    """
    grouped: dict[tuple[str, str], dict] = {}
    for plans in plan_lists:
        for plan in plans:
            key = (plan.project_id, plan.topic)
            group = grouped.setdefault(key, {
                'members': {},
                'canonicals': {},
                'sources': set(),
                'plural': False,
                'notes': [],
            })
            group['sources'].add(plan.source)
            group['plural'] = group['plural'] or plan.canonical_plural
            note = (plan.provenance or {}).get('note')
            if note:
                group['notes'].append(note)
            if plan.canonical_memory_id:
                group['canonicals'].setdefault(plan.canonical_memory_id, None)
            for member in plan.member_memory_ids:
                group['members'].setdefault(member, None)

    skips: list[dict] = []
    # (project_id, topic) -> resolved canonical id (or None)
    resolved: dict[tuple[str, str], str | None] = {}
    for (project_id, topic), group in sorted(grouped.items()):
        canonicals = list(group['canonicals'])
        if group['plural']:
            skips.append({
                'reason': 'canonical_plural',
                'project_id': project_id,
                'topic': topic,
                'note': '; '.join(group['notes']) or None,
            })
            resolved[(project_id, topic)] = None
        elif len(canonicals) > 1:
            skips.append({
                'reason': 'canonical_disagreement',
                'project_id': project_id,
                'topic': topic,
                'canonical_memory_ids': sorted(canonicals),
            })
            resolved[(project_id, topic)] = None
        else:
            resolved[(project_id, topic)] = canonicals[0] if canonicals else None

    # Which topics claim each id, per project. A canonical is also a claim,
    # so a record cannot be one topic's canonical and another's member.
    claims: dict[tuple[str, str], set[str]] = {}
    for (project_id, topic), group in grouped.items():
        ids = set(group['members']) | set(group['canonicals'])
        for memory_id in ids:
            claims.setdefault((project_id, memory_id), set()).add(topic)

    targets: list[StampTarget] = []
    for (project_id, topic), group in grouped.items():
        source = '+'.join(sorted(group['sources']))
        canonical_id = resolved[(project_id, topic)]
        ids = set(group['members']) | set(group['canonicals'])
        for memory_id in ids:
            if len(claims[(project_id, memory_id)]) > 1:
                continue  # reported once, below
            targets.append(StampTarget(
                project_id=project_id,
                memory_id=memory_id,
                topic=topic,
                make_canonical=memory_id == canonical_id,
                source=source,
            ))

    for (project_id, memory_id), topics in sorted(claims.items()):
        if len(topics) > 1:
            skips.append({
                'reason': 'conflicting_topic_claim',
                'project_id': project_id,
                'memory_id': memory_id,
                'topics': sorted(topics),
            })

    targets.sort(key=lambda t: (t.project_id, t.topic, t.memory_id))
    return targets, skips


# ---------------------------------------------------------------------------
# The single I/O boundary
# ---------------------------------------------------------------------------

#: Written to every ``update_memory`` so the write journal attributes each
#: stamp to this sweep rather than to a generic ``mcp_tool``.  The amendment
#: storm alarm reads this field; a bulk run under the default source would
#: look exactly like the runaway rewrite that alarm exists to catch.
WRITE_SOURCE = 'retro_stamp_topics'

#: Recorded on the write journal row beside the patch.
WRITE_REASON = 'PRD leaf θ retro topic/canonical stamping (task 3201)'

#: Outcomes that mean the corpus did not get what the plan intended.  Named
#: once, here, so :func:`resolve_exit_code` and the report agree on what
#: "clean" means instead of each keeping its own list.
ERROR_OUTCOMES: frozenset[str] = frozenset({
    'memory_not_found',
    'update_failed',
    'canonical_uniqueness_blocked',
    'canonical_probe_failed',
    'conflicting_existing_topic',
})


async def stamp_one(memory_service, target: StampTarget, *, apply: bool) -> dict:
    """Stamp one record, or report precisely why it was not stamped.

    The only function here that touches a store.  Order is load-bearing:

    1. **live re-read** (``get_memory_by_id``).  The plan's ids come from a
       committed manifest and fixtures transcribed weeks earlier, so the
       record may have been consolidated away since (measured: gate 3036's
       ``19705df4`` no longer resolves).  Reading first is also what turns a
       vanished id into a report line instead of a Qdrant ``set_payload``
       that acknowledges a write to nothing.
    2. **decide** (:func:`compute_patch`).  An empty patch ends the call: a
       no-op ``update_memory`` would still journal a write op and still
       count toward the content-amendment storm alarm, and would inflate the
       report's stamped count with records that gained nothing.
    3. **canonical-uniqueness probe**, only when the patch would actually set
       ``canonical`` — one ``count_memories_by_metadata`` and, only on a
       non-zero count, one bounded ``get_memories_by_metadata(limit=1)`` to
       name the incumbent.  The same two-probe shape as
       ``services.memory_service._check_canonical_uniqueness``, re-expressed
       here for the measured reason that ``update_memory`` never reaches
       that seam (see the module docstring's INV-5 note).
    4. **write**, metadata-only and merge-mode.

    FAIL CLOSED on a probe error, inverting the service seam's warn-mode
    default.  There the calculus is that failing a valid interactive write
    because the complaint machinery broke is worse than the duplicate it
    guards; here this script *is* the enforcement layer, it writes canonicals
    in bulk unattended, and a re-run costs nothing — so an unverifiable
    canonical is refused.  It gets its own outcome rather than being folded
    into ``canonical_uniqueness_blocked`` because "the store was unreachable"
    and "a duplicate exists" are different facts an operator must be able to
    tell apart.

    Args:
        memory_service: Anything with the four ``MemoryService`` coroutines
            used below.  Injected rather than constructed so the tests never
            stand up a backend.
        target: One :class:`StampTarget` from :func:`merge_plans`.
        apply: ``False`` (the default posture) performs every read and the
            full decision, then stops short of the write — so a dry-run
            report states what an apply would really do, probe verdict
            included, rather than what the plan hoped.

    Returns:
        A report-ready dict always carrying ``outcome``, plus whatever that
        outcome needs to be actionable without a follow-up query
        (``incumbent_id``, ``error``, ``existing_topic``, …).
    """
    base = {
        'project_id': target.project_id,
        'memory_id': target.memory_id,
        'topic': target.topic,
        'source': target.source,
        'make_canonical': target.make_canonical,
    }

    record = await memory_service.get_memory_by_id(
        project_id=target.project_id, memory_id=target.memory_id,
    )
    if record is None:
        return {**base, 'outcome': 'memory_not_found', 'patch': {}, 'dispositions': ()}

    metadata = dict(record.get('metadata') or {})
    decision = compute_patch(
        metadata, target_topic=target.topic, make_canonical=target.make_canonical,
    )
    result = {**base, 'patch': dict(decision.patch), 'dispositions': decision.dispositions}

    # An existing topic that is neither ours nor a legacy spelling of ours
    # means the plan put this record in the wrong cluster. Refuse the WHOLE
    # write — including any `canonical`, which is scoped by topic and would
    # otherwise head a cluster this record does not belong to.
    if 'conflicting_existing_topic' in decision.dispositions:
        return {
            **result,
            'outcome': 'conflicting_existing_topic',
            'existing_topic': metadata.get('topic'),
            'patch': {},
        }

    if not decision.patch:
        return {**result, 'outcome': 'already_stamped'}

    if decision.patch.get('canonical') is True:
        filters = {'topic': target.topic, 'canonical': True}
        try:
            count = await memory_service.count_memories_by_metadata(
                target.project_id, filters,
            )
            incumbents = (
                await memory_service.get_memories_by_metadata(
                    target.project_id, filters, limit=1,
                )
                if count
                else []
            )
        except Exception as exc:
            return {
                **result,
                'outcome': 'canonical_probe_failed',
                'error': f'{type(exc).__name__}: {exc}',
            }
        incumbent_id = incumbents[0].get('id') if incumbents else None
        # Finding OURSELVES is not a conflict: the probe cannot exclude the
        # record being written, so a re-run over a partially-stamped corpus
        # sees its own row. Treating that as a violation would leave the
        # sweep unable to finish what it started.
        if count and incumbent_id != target.memory_id:
            return {
                **result,
                'outcome': 'canonical_uniqueness_blocked',
                'incumbent_id': incumbent_id,
                'patch': {},
            }

    if not apply:
        return {**result, 'outcome': 'would_stamp'}

    response = await memory_service.update_memory(
        memory_id=target.memory_id,
        project_id=target.project_id,
        metadata_patch=dict(decision.patch),
        metadata_mode='merge',
        reason=WRITE_REASON,
        _source=WRITE_SOURCE,
    )
    # `update_memory` reports a not-found by RETURNING a structured envelope
    # rather than raising, so a caller that only guarded against exceptions
    # would score a refused write as a stamp.
    if isinstance(response, dict) and response.get('error_type'):
        return {
            **result,
            'outcome': 'update_failed',
            'error_type': response.get('error_type'),
            'error': response.get('error'),
            'response': response,
        }
    return {**result, 'outcome': 'stamped', 'response': response}


# ---------------------------------------------------------------------------
# The run — three sources in, one auditable report out
# ---------------------------------------------------------------------------

#: The two corpora the six live ``canonical: true`` records straddle (1 in
#: ``dark_factory``, 5 in ``reify``).  A single-project default would silently
#: cover a sixth of source (1).
DEFAULT_PROJECTS: tuple[str, ...] = ('dark_factory', 'reify')

#: Every bucket the report carries, PRE-SEEDED TO EMPTY.  Seeding is the
#: point: an absent bucket reads as "nothing was skipped", which is a
#: different claim from "we looked and found nothing".  A hole in coverage
#: that never reaches the artifact is a hole nobody closes.
SKIP_BUCKETS: tuple[str, ...] = (
    # from the planners
    'skipped_no_enumeration',
    'no_registry_topic',
    'topic_underivable',
    'member_not_a_uuid',
    'cluster_without_canonical',
    'out_of_scope_project',
    # from merge_plans
    'conflicting_topic_claim',
    'canonical_disagreement',
    'canonical_plural',
    # from stamp_one
    'conflicting_existing_topic',
    'canonical_uniqueness_blocked',
    'canonical_probe_failed',
    'memory_not_found',
    'update_failed',
    'supersedes_not_normalizable',
)

#: What each source IS, in the artifact rather than only in this docstring.
#: The boundedness argument is this leaf's whole safety case; stating it
#: only in the source puts it where a reader of the output cannot check it.
SOURCE_DESCRIPTIONS: dict[str, str] = {
    'canonical_scroll': (
        'source (1): the live `canonical: true` scroll, per project — each '
        "record's own consolidates/retires/replaces/supersedes uuid lists "
        'name its members'
    ),
    'calibration_registry_join': (
        "source (3) and the reify half of source (2): 3130's labeled "
        'calibration fixture joined to the committed E1 topic registry on '
        'cluster_id (the registry supplies the hand-authored slug a machine '
        'cannot regenerate)'
    ),
    'df_curator_gate_manifest': (
        'source (2), dark_factory half: the committed DF_CURATOR_GATE_CLUSTERS '
        'manifest, transcribed from the seven PRD-D11 curator gates'
    ),
}


async def run(
    memory_service,
    *,
    projects: tuple[str, ...] = DEFAULT_PROJECTS,
    apply: bool = False,
    calibration_rows: list[dict] | None = None,
    registry: object | None = None,
    gate_manifest: tuple[GateCluster, ...] = DF_CURATOR_GATE_CLUSTERS,
) -> dict:
    """Plan all three bounded sources, stamp them, and report what happened.

    Sequential awaits throughout.  The target count is bounded in the low
    hundreds, so concurrency would buy little — and it would complicate the
    one ordering that matters: two concurrent canonical stamps for the same
    topic could both observe a zero count and both write, re-opening exactly
    the TOCTOU window ``_check_canonical_uniqueness`` documents as its
    residual risk.  Bounded work does not need a lock it cannot have.

    Args:
        memory_service: Injected; see :func:`stamp_one`.
        projects: Which corpora to scroll for source (1).
        apply: ``False`` (default) rehearses every read and decision and
            withholds only the writes.
        calibration_rows: 3130's rows.  ``None`` loads the committed fixture;
            tests inject their own so they never depend on fixture drift.
        registry: The E1 topic registry.  ``None`` loads the committed one
            through the probe's own strict loader.
        gate_manifest: The DF gate table.  Defaults to the committed one.

    Returns:
        The report dict — rendered by :func:`render_markdown` /
        :func:`render_json` and graded by :func:`resolve_exit_code`.
    """
    if calibration_rows is None:
        calibration_rows = load_calibration_rows(CALIBRATION_FIXTURE_PATH)
    if registry is None:
        registry = load_topic_registry(TOPIC_REGISTRY_PATH)

    skips: dict[str, list[dict]] = {bucket: [] for bucket in SKIP_BUCKETS}

    def _record_skips(entries: list[dict]) -> None:
        """File each skip under its own reason, loudly if it is a new one.

        An unknown reason is appended to a bucket created on the spot rather
        than dropped: a reason this function has not heard of is precisely
        the kind of thing that must not vanish between the planner that
        raised it and the artifact.
        """
        for entry in entries:
            skips.setdefault(entry.get('reason', 'unclassified'), []).append(entry)

    # --- source (1): the live canonical scroll, per project ----------------
    canonical_plans: list[ClusterPlan] = []
    scrolled_records = 0
    for project_id in projects:
        records = await memory_service.get_memories_by_metadata(
            project_id, {'canonical': True},
        )
        scrolled_records += len(records)
        plans, source_skips = plan_canonical_clusters(records, project_id=project_id)
        canonical_plans.extend(plans)
        _record_skips(source_skips)

    # --- sources (3) + (2)-reify: one join on cluster_id --------------------
    calibration_plans, calibration_skips = plan_calibration_clusters(
        calibration_rows, registry,
    )
    _record_skips(calibration_skips)

    # --- source (2), dark_factory half: the committed manifest --------------
    gate_plans, gate_skips = plan_gate_clusters(gate_manifest)
    _record_skips(gate_skips)

    # --- the scope bound, applied to ALL THREE sources ----------------------
    # Before `merge_plans`, not after: once merged, an out-of-scope plan has
    # already contributed its members to an in-scope topic's union, and
    # filtering targets at that point would let the excluded corpus decide
    # what the included one gets stamped with.
    #
    # `canonical_plans` goes through too, even though the scroll loop above
    # already built it per project and it can therefore drop nothing today.
    # That makes this a chokepoint rather than a coincidence: a fourth
    # planner wired in ahead of it inherits the bound instead of quietly
    # re-opening the leak this filter was added to close.
    scoped: list[list[ClusterPlan]] = []
    for plan_list in (canonical_plans, calibration_plans, gate_plans):
        in_scope, out_of_scope = filter_plans_to_projects(plan_list, projects)
        scoped.append(in_scope)
        _record_skips(out_of_scope)

    targets, merge_skips = merge_plans(*scoped)
    _record_skips(merge_skips)

    results: list[dict] = []
    stamped_by_topic: dict[str, int] = {}
    would_stamp_by_topic: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    for target in targets:
        result = await stamp_one(memory_service, target, apply=apply)
        results.append(result)
        outcome = result['outcome']
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == 'stamped':
            stamped_by_topic[target.topic] = stamped_by_topic.get(target.topic, 0) + 1
        elif outcome == 'would_stamp':
            would_stamp_by_topic[target.topic] = (
                would_stamp_by_topic.get(target.topic, 0) + 1
            )
        if outcome in skips:
            skips[outcome].append(result)
        # A record can be stamped AND carry a note — a legacy `supersedes`
        # this sweep declined to fold is a fact about the record whether or
        # not its topic landed, so it rides alongside the outcome rather
        # than competing with it.
        if 'supersedes_not_normalizable' in result.get('dispositions', ()):
            skips['supersedes_not_normalizable'].append({
                'reason': 'supersedes_not_normalizable',
                'memory_id': result['memory_id'],
                'project_id': result['project_id'],
                'topic': result['topic'],
            })

    return {
        'apply': apply,
        # The safety claim, stated where the artifact's reader can check it.
        'bounded': True,
        'projects': list(projects),
        'sources': {
            'canonical_scroll': {
                'description': SOURCE_DESCRIPTIONS['canonical_scroll'],
                'record_count': scrolled_records,
                'plan_count': len(canonical_plans),
            },
            'calibration_registry_join': {
                'description': SOURCE_DESCRIPTIONS['calibration_registry_join'],
                'row_count': len(calibration_rows),
                'cluster_count': len({
                    row.get('cluster_id') for row in calibration_rows
                    if row.get('cluster_id')
                }),
                'plan_count': len(calibration_plans),
            },
            'df_curator_gate_manifest': {
                'description': SOURCE_DESCRIPTIONS['df_curator_gate_manifest'],
                'gate_count': len(gate_manifest),
                'plan_count': len(gate_plans),
            },
        },
        'target_count': len(targets),
        'stamped_total': sum(stamped_by_topic.values()),
        'stamped_by_topic': stamped_by_topic,
        'would_stamp_total': sum(would_stamp_by_topic.values()),
        'would_stamp_by_topic': would_stamp_by_topic,
        'outcomes': outcomes,
        'results': results,
        'skips': skips,
    }


# ---------------------------------------------------------------------------
# Report rendering, exit code, CLI
# ---------------------------------------------------------------------------

#: Default artifact paths, beside the census report leaf β already writes
#: (``census_memory_metadata.py``) — the conventional place, so an operator's
#: run lands somewhere reviewable and committable rather than in /tmp.
#:
#: Whether to commit the result is the operator's call, and the PRD's θ bullet
#: deliberately does NOT cite one as existing: a stamping report snapshots
#: live corpus state that rots immediately, so a committed copy invites
#: trusting a stale count.  Contrast leaf α's census, a one-shot measurement
#: worth pinning.
DEFAULT_JSON_OUT = str(_REPO_ROOT / 'plans' / 'retro-topic-stamping-report.json')
DEFAULT_MD_OUT = str(_REPO_ROOT / 'plans' / 'retro-topic-stamping-report.md')

#: The skip keys worth printing inline in a bucket entry, in reading order.
#: A whitelist rather than a dump of the whole dict: the stamp results that
#: land in error buckets carry a full ``response`` envelope, which would bury
#: the four fields that actually identify the record.
_SKIP_DETAIL_KEYS: tuple[str, ...] = (
    'gate_task_id', 'cluster_id', 'project_id', 'memory_id', 'topic',
    'topics', 'canonical_memory_ids', 'incumbent_id', 'raw_topic', 'key',
    'value', 'error', 'error_type', 'detail', 'note', 'existing_topic',
    'row_count',
)


def render_json(report: dict) -> str:
    """The machine-readable artifact.

    ``default=str`` because a report can carry a tuple of dispositions or a
    datetime that wandered in off a scroll row; a renderer that raised on one
    would lose the whole run's evidence over a formatting detail.
    """
    return json.dumps(report, indent=2, default=str, sort_keys=True)


def _render_skip_entry(entry: dict) -> str:
    parts = [
        f'{key}={entry[key]!r}'
        for key in _SKIP_DETAIL_KEYS
        if entry.get(key) is not None
    ]
    # Fall back to the whole dict when no whitelisted key matched, rather
    # than emitting a bare bullet: an entry the renderer does not recognise
    # is exactly the one worth showing verbatim.
    return '- ' + (', '.join(parts) if parts else repr(entry))


def render_markdown(report: dict) -> str:
    """The human-readable artifact.

    Every EMPTY bucket is stated as an explicit ``: 0`` heading rather than
    omitted.  This is the renderer's most load-bearing rule: an artifact that
    silently drops empty buckets reads identically whether the sweep skipped
    nothing or never computed the bucket at all — and the second case is a
    hole in coverage that, being invisible, nobody ever closes.
    """
    mode = 'APPLY' if report.get('apply') else 'DRY RUN'
    lines: list[str] = [
        '# Retro topic/canonical stamping — PRD leaf θ (task 3201)',
        '',
        f'**Mode:** {mode}',
        f'**Projects:** {", ".join(report.get("projects") or [])}',
        (
            '**Scope:** bounded — every target is addressed by memory id from '
            'an enumerated source below, never by a corpus scroll.'
        ),
        '',
        '## Sources',
        '',
        '| source | plans | counts | description |',
        '| --- | --- | --- | --- |',
    ]
    for name, block in sorted((report.get('sources') or {}).items()):
        counts = ', '.join(
            f'{key}={value}' for key, value in sorted(block.items())
            if key not in ('description', 'plan_count')
        )
        lines.append(
            f'| {name} | {block.get("plan_count", 0)} | {counts} | '
            f'{block.get("description", "")} |'
        )

    stamped = report.get('stamped_by_topic') or {}
    would = report.get('would_stamp_by_topic') or {}
    lines += [
        '',
        '## Stamped by topic',
        '',
        '| topic | records stamped | would stamp |',
        '| --- | --- | --- |',
    ]
    for topic in sorted(set(stamped) | set(would)):
        lines.append(f'| {topic} | {stamped.get(topic, 0)} | {would.get(topic, 0)} |')
    if not stamped and not would:
        lines.append('| _none_ | 0 | 0 |')
    lines += [
        '',
        f'**Total stamped:** {report.get("stamped_total", 0)}  ',
        f'**Total would-stamp:** {report.get("would_stamp_total", 0)}  ',
        f'**Targets planned:** {report.get("target_count", 0)}',
        '',
        '## Outcomes',
        '',
    ]
    outcomes = report.get('outcomes') or {}
    if outcomes:
        lines.extend(f'- `{name}`: {count}' for name, count in sorted(outcomes.items()))
    else:
        lines.append('- _no targets_')

    lines += ['', '## Skips', '']
    skips = report.get('skips') or {}
    # Iterate the CANONICAL bucket order first so a bucket run() forgot to
    # emit shows up as a 0 line here rather than as an absence, then any
    # extra bucket a planner invented at runtime.
    ordered = list(SKIP_BUCKETS) + sorted(set(skips) - set(SKIP_BUCKETS))
    for bucket in ordered:
        entries = skips.get(bucket) or []
        lines += ['', f'### {bucket}: {len(entries)}', '']
        if not entries:
            lines.append('_none_')
            continue
        lines.extend(_render_skip_entry(entry) for entry in entries)
    return '\n'.join(lines) + '\n'


def resolve_exit_code(report: dict) -> int:
    """0 on a clean run, 1 when anything did not land as planned.

    Graded off :data:`ERROR_OUTCOMES` — the SAME set ``stamp_one`` and the
    report use — so the exit code and the artifact can never disagree about
    whether a run was clean.

    Partial failure is the normal shape of this sweep: the manifest's ids
    were transcribed weeks before any run and the corpus moved underneath.
    That makes an honest non-zero the one signal an automated caller can act
    on; a bare 0 would read as success (the
    ``strip_leaked_control_keys.py`` convention).
    """
    outcomes = report.get('outcomes') or {}
    return 1 if any(outcomes.get(name) for name in ERROR_OUTCOMES) else 0


def resolve_projects(args) -> tuple[str, ...]:
    """``--project`` REPLACES the default list; it does not extend it.

    ``action='append'`` appends to whatever default argparse holds, so a
    plain ``default=DEFAULT_PROJECTS`` would turn ``--project reify`` into
    all three — silently widening a sweep whose entire safety argument is
    that it is narrow.  Hence ``default=None`` here and the fallback in one
    named place.

    The selection bounds EVERY source, not only the live canonical scroll
    it is passed to: ``run`` funnels all three plan lists through
    :func:`filter_plans_to_projects` before merging them, and names each
    drop under ``out_of_scope_project``.  Without that, a run narrowed to
    one corpus still wrote into the other, since the calibration join takes
    its project from the registry entry and the gate manifest hard-codes
    ``dark_factory``.
    """
    return tuple(args.projects) if args.projects else DEFAULT_PROJECTS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Bounded retro topic/canonical stamping sweep over known '
            'consolidated clusters (PRD leaf θ, task 3201).'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit the stamps. Without it the run is a full rehearsal that '
             'reads and decides everything but writes nothing.',
    )
    parser.add_argument(
        '--project', dest='projects', action='append', default=None,
        help=f'Project to sweep; repeatable. Replaces the default '
             f'({", ".join(DEFAULT_PROJECTS)}) rather than extending it. '
             f'Bounds every source — the live scroll, the calibration join '
             f'and the gate manifest alike — and the report names each '
             f'plan it excluded under out_of_scope_project.',
    )
    parser.add_argument(
        '--json-out', dest='json_out', default=DEFAULT_JSON_OUT,
        help=f'Machine-readable report path (default: {DEFAULT_JSON_OUT}).',
    )
    parser.add_argument(
        '--md-out', dest='md_out', default=DEFAULT_MD_OUT,
        help=f'Human-readable report path (default: {DEFAULT_MD_OUT}).',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build a live service, run the sweep, write both artifacts, exit graded."""
    args = _build_parser().parse_args(argv)

    if args.config:
        import os  # noqa: PLC0415

        os.environ['CONFIG_PATH'] = str(args.config)

    async def _run_live() -> dict:
        # Deferred so importing this module — which the tests do, by path —
        # never constructs a backend.
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        try:
            await memory.initialize()
            return await run(
                memory, projects=resolve_projects(args), apply=args.apply,
            )
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

    report = asyncio.run(_run_live())

    Path(args.json_out).write_text(render_json(report), encoding='utf-8')
    Path(args.md_out).write_text(render_markdown(report), encoding='utf-8')

    print(render_markdown(report))
    print(f'wrote {args.json_out}')
    print(f'wrote {args.md_out}')
    if not args.apply:
        print('DRY RUN — nothing was modified. Re-run with --apply to commit.')
    return resolve_exit_code(report)


if __name__ == '__main__':
    sys.exit(main())
