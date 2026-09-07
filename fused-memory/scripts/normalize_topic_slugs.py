#!/usr/bin/env python3
"""Corpus-wide ``metadata.topic`` slug normalization (task 4878).

Folds every non-conforming live ``topic`` value into the shape
:mod:`fused_memory.topic_slug` defines — in practice snake_case to
hyphen-case, which is what all of the measured non-conforming values are —
so that exact-match retrieval (``get_memories_by_metadata({'topic': T})``,
which is how consolidation clusters and the canonical-uniqueness probe
address a topic) stops depending on which spelling a writer happened to use.

THIS IS A CORPUS SWEEP — the deliberate contrast with the sibling
-----------------------------------------------------------------
``scripts/retro_stamp_topics.py`` (PRD leaf θ, task 3201) states in its own
docstring that it is *not* a corpus sweep: every one of its targets is
addressed by memory id, drawn from three enumerated bounded sources, and it
reports ``bounded: True``.  That boundedness is its entire safety argument.

This script is the opposite and says so up front: it walks the whole
collection.  They are separate scripts precisely so that claim stays true
where it is made — bolting an unbounded scroll into the bounded sweep would
falsify its docstring, its report field and its rendered header in place.
The jobs differ too: that one STAMPS a topic onto records carrying none;
this one REWRITES the value of one that already exists.

What is shared is the *fold*.  :func:`fused_memory.topic_slug.derive_topic_slug`
has one home (INV-5) and both scripts import it, so the two can never disagree
about what a legacy value becomes.

The migration spans TWO stores
------------------------------
Most of the work patches Mem0 record metadata.  But
``metadata.x_recon_consolidation_gate.topic`` is a Tier-C block on **task**
metadata — read by ``reconciliation/consolidation_gate.py`` and enforced on
the ``done`` transition by ``middleware/task_interceptor.py`` — a genuinely
different store, reached through the MCP server rather than a backend call.

Renaming the memories without moving that block would leave a gate task's
closure check scrolling a slug no record carries any more: the uncloseable-gate
trap re-created at a NEW address, which is strictly worse than leaving the
snake_case slug alone because the operator would then also have to discover
that this migration caused it.  So the two halves are planned as a pair and
refused as a pair.

Dry run is the default
----------------------
Nothing is written without ``--apply``.  See the operator runbook below.
"""
from __future__ import annotations

from dataclasses import dataclass

from fused_memory.topic_slug import derive_topic_slug, is_valid_topic_slug
from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)

# Convention inherited from ``retro_stamp_topics.py``: this list carries every
# imported helper the tests reach through the module object, not only the names
# defined locally.  This script is loaded by path via ``importlib`` and cannot
# be imported from, so its ``__all__`` grants the shared rule no second home.
__all__ = [
    'ERROR_OUTCOMES',
    'Rename',
    'SKIP_BUCKETS',
    'StoreMutationUnavailable',
    'assert_store_mutation_allowed',
    'derive_topic_slug',
    'is_valid_topic_slug',
    'plan_renames',
]


#: Outcomes that mean the corpus did not get what the plan intended.  Named
#: once, here, so :func:`resolve_exit_code` and the report agree on what
#: "clean" means instead of each keeping its own list.
ERROR_OUTCOMES: frozenset[str] = frozenset({
    'slug_collision',
    'canonical_collision',
})

#: Every bucket the report carries, PRE-SEEDED TO EMPTY.  Seeding is the
#: point: an absent bucket reads as "nothing was skipped", which is a
#: different claim from "we looked and found nothing".  A hole in coverage
#: that never reaches the artifact is a hole nobody closes.
SKIP_BUCKETS: tuple[str, ...] = (
    # from plan_renames
    'topic_unfoldable',
    'record_without_id',
    'slug_collision',
    'canonical_collision',
)


# ---------------------------------------------------------------------------
# The plan row
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rename:
    """One record's topic move, decided before anything is written.

    Frozen because the artifact and the writes must agree: the whole reason
    the corpus is planned before it is touched is so an operator can read the
    rehearsal and know what ``--apply`` will do.  A mutable row lets the two
    diverge between the report and the write loop.

    Attributes:
        project_id: The corpus this record lives in.  Carried on the row
            rather than inferred at the write, because collisions are scoped
            per project and the report interleaves both defaults.
        memory_id: The Mem0 record id — how :func:`rename_one` addresses it.
        old_topic: The raw non-conforming value, verbatim.  Needed to detect
            a record whose topic moved between the plan and the write, and to
            probe the legacy slug for residue afterwards.
        new_topic: The folded, validated slug.  Always a value
            :func:`is_valid_topic_slug` accepts — the planner never emits a
            row it could not stand behind.
    """

    project_id: str
    memory_id: str
    old_topic: str
    new_topic: str


# ---------------------------------------------------------------------------
# Pure core — planning
# ---------------------------------------------------------------------------

def plan_renames(records, *, project_id: str) -> tuple[list[Rename], list[dict]]:
    """Decide what to rename in one project's records, and what to refuse.

    Pure: takes scroll-shaped ``{'id', 'created_at', 'metadata'}`` dicts and no
    service, so every verdict is reproducible from the input alone.

    Takes the project's FULL record set rather than one record at a time.  It
    has to: deciding whether ``foo_bar`` may become ``foo-bar`` requires
    knowing whether anything already carries ``foo-bar``, which is not a
    property of the record being examined.

    Four dispositions, and the distinctions between them are load-bearing:

    * **rename** — the topic fails :func:`is_valid_topic_slug` and folds
      cleanly.  This is the work.
    * **ignored, silently** — the topic already conforms, or there is no
      ``topic`` key, or it is ``None``.  Deliberately NOT a skip line: most
      of the ~49k-record corpus conforms, and filing each conforming record
      would bury the ~141 that need attention.  A ``None`` topic is an
      *absent* topic, not a malformed one; stamping an absent topic is
      ``retro_stamp_topics.py``'s job, not this script's.
    * **skipped, reported** — no honest fold exists (``topic_unfoldable``), or
      the record has no id to write to (``record_without_id``).  The entry
      carries the RAW value and never a guessed slug: an operator resolving
      it by hand needs to see what is actually on the record, and the reason
      it is here at all is that this script refused to name a replacement.
    * **refused, per SLUG** — the fold lands on an occupied slug
      (``slug_collision``), possibly severely (``canonical_collision``).  See
      the guard below; the refusal covers every record carrying that legacy
      value, and is filed once for the slug rather than once per record.

    Output is sorted by ``(project_id, new_topic, memory_id)`` — and the skips
    correspondingly — because scroll order is not stable, and two rehearsals
    over an unchanged corpus must produce byte-comparable artifacts for a diff
    to mean anything.

    Args:
        records: Scroll-shaped record dicts for ONE project.
        project_id: The corpus they came from; stamped onto every row.

    Returns:
        ``(renames, skips)`` — frozen :class:`Rename` rows, and report-ready
        skip dicts each carrying a ``reason``.
    """
    renames: list[Rename] = []
    skips: list[dict] = []

    # --- pass 1: what does the corpus already look like? -------------------
    # Built BEFORE any verdict, from the whole record set, so no decision can
    # depend on the order the backend happened to page records back in.
    #
    #   observed_topics       every topic value present, conforming or not
    #   canonical_ids_by_topic  which records under a topic claim canonical
    #   sources_by_target     which legacy values want to fold onto a target
    observed_topics: set[str] = set()
    canonical_ids_by_topic: dict[str, list[str]] = {}
    sources_by_target: dict[str, set[str]] = {}
    for record in records:
        metadata = record.get('metadata') or {}
        topic = metadata.get('topic')
        if not isinstance(topic, str):
            continue
        observed_topics.add(topic)
        # ``is True``, not truthiness: metadata comes off a live store and is
        # not schema-enforced at this seam, so ``canonical: 'false'`` — a
        # non-empty string — would otherwise read as a canonical claim and
        # escalate a mild refusal into the severe bucket.
        if metadata.get('canonical') is True:
            memory_id = record.get('id')
            if isinstance(memory_id, str) and memory_id:
                canonical_ids_by_topic.setdefault(topic, []).append(memory_id)
        if not is_valid_topic_slug(topic):
            folded = derive_topic_slug(topic)
            if folded is not None:
                sources_by_target.setdefault(folded, set()).add(topic)

    # --- pass 2: classify, then emit --------------------------------------
    # WHY REFUSE RATHER THAN MERGE.  Folding ``foo_bar`` onto ``foo-bar`` when
    # something already carries ``foo-bar`` is not a normalization: it merges
    # two topic namespaces, changing cluster membership for records this sweep
    # never examined.  When both sides hold a ``canonical: true`` it is worse
    # than that — ``topic`` is what SCOPES canonical uniqueness
    # (``memory_service._check_canonical_uniqueness`` probes
    # ``{'topic': T, 'canonical': True}``), so the merged cluster holds exactly
    # the second canonical that invariant exists to prevent.  Under the shipped
    # ``memory_metadata.enforce = false`` that seam WARN-fails open, so it would
    # land silently with this script as its author.
    #
    # WHY THE SERVICE-SIDE PROBE CANNOT SUBSTITUTE FOR THIS ONE.  The seam
    # adjudicates a SINGLE write against live state, one at a time.  It would
    # let the first member of a colliding group through and refuse only the
    # second — leaving the corpus in a state determined by iteration order, and
    # half-merged.  Deciding here, against the whole record set, makes the
    # verdict a property of the CORPUS instead.  Same posture, same reason, as
    # ``retro_stamp_topics.stamp_one``'s ``stray_canonical_on_member``.
    collisions: dict[str, dict] = {}
    for record in records:
        metadata = record.get('metadata') or {}
        if 'topic' not in metadata:
            continue
        topic = metadata['topic']
        if topic is None:
            continue
        if is_valid_topic_slug(topic):
            continue

        memory_id = record.get('id')
        folded = derive_topic_slug(topic)
        if folded is None:
            skips.append({
                'reason': 'topic_unfoldable',
                'project_id': project_id,
                'memory_id': memory_id,
                'topic': topic,
                'note': (
                    'no honest fold exists for this value (it collapses to '
                    'empty, exceeds TOPIC_SLUG_MAX_LEN once folded, or is not '
                    'a string) — resolve it by hand rather than letting the '
                    'sweep invent a slug no human chose'
                ),
            })
            continue

        # Occupied by a record that already carries the target value, or
        # contested by a SECOND legacy value folding onto the same target.
        # The second case is refused for BOTH sources: letting one through
        # would make the outcome depend on scroll order, and would refuse the
        # other for an occupancy this script had just manufactured.
        incumbent = folded in observed_topics
        contested = len(sources_by_target.get(folded, ())) > 1
        if incumbent or contested:
            entry = collisions.get(topic)
            if entry is None:
                old_canonicals = sorted(canonical_ids_by_topic.get(topic, ()))
                new_canonicals = sorted(
                    memory_id
                    for other in (
                        {folded} | (sources_by_target.get(folded, set()) - {topic})
                    )
                    for memory_id in canonical_ids_by_topic.get(other, ())
                )
                severe = bool(old_canonicals) and bool(new_canonicals)
                entry = {
                    'reason': (
                        'canonical_collision' if severe else 'slug_collision'
                    ),
                    'project_id': project_id,
                    'old_topic': topic,
                    'new_topic': folded,
                    'colliding_count': 0,
                    'memory_ids': [],
                    'old_canonical_ids': old_canonicals,
                    'new_canonical_ids': new_canonicals,
                    'note': (
                        'the fold would merge two topic namespaces AND carry a '
                        'canonical from each side into one cluster, '
                        'manufacturing the second canonical '
                        '_check_canonical_uniqueness exists to prevent — '
                        'resolve by hand before re-running'
                        if severe else
                        'the target slug is already occupied (by conforming '
                        'records, or by a second legacy value folding onto it), '
                        'so the rename would MERGE two topic namespaces rather '
                        'than normalize one — resolve by hand before re-running'
                    ),
                }
                collisions[topic] = entry
            entry['colliding_count'] += 1
            if isinstance(memory_id, str) and memory_id:
                entry['memory_ids'].append(memory_id)
            continue

        if not memory_id or not isinstance(memory_id, str):
            # ``update_memory`` is addressed by memory id.  Refusing here keeps
            # an unaddressable row out of the planned-work count instead of
            # letting it fail one at a time at the write boundary, inside a
            # report that already claimed it.
            skips.append({
                'reason': 'record_without_id',
                'project_id': project_id,
                'memory_id': memory_id,
                'topic': topic,
                'new_topic': folded,
                'note': 'record carries no usable id, so it cannot be written to',
            })
            continue

        renames.append(Rename(
            project_id=project_id,
            memory_id=memory_id,
            old_topic=topic,
            new_topic=folded,
        ))

    for entry in collisions.values():
        entry['memory_ids'].sort()
        skips.append(entry)

    renames.sort(key=lambda r: (r.project_id, r.new_topic, r.memory_id))
    skips.sort(key=lambda s: (
        str(s.get('reason')),
        str(s.get('project_id')),
        str(s.get('old_topic', '')),
        str(s.get('memory_id')),
    ))
    return renames, skips
