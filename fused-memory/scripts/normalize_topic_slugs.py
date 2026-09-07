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
    'Rename',
    'StoreMutationUnavailable',
    'assert_store_mutation_allowed',
    'derive_topic_slug',
    'is_valid_topic_slug',
    'plan_renames',
]


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

    renames.sort(key=lambda r: (r.project_id, r.new_topic, r.memory_id))
    skips.sort(key=lambda s: (
        str(s.get('reason')), str(s.get('project_id')), str(s.get('memory_id')),
    ))
    return renames, skips
