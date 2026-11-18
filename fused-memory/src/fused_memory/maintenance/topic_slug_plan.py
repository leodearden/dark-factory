"""The pure planner behind ``scripts/normalize_topic_slugs.py`` (task 4878).

Decides, from scroll-shaped records alone, which non-conforming ``topic``
values fold onto which slug and which are refused (no honest fold, no id to
write to, or a fold that would merge two topic namespaces), with no service:
every verdict is reproducible from the input.
"""
from __future__ import annotations

from dataclasses import dataclass

from fused_memory.canonical_marker import is_canonical
from fused_memory.topic_slug import derive_topic_slug, is_valid_topic_slug


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


def is_non_conforming_topic(topic: object) -> bool:
    """A PRESENT topic the slug rule rejects, whatever its type.

    ``None`` is an absent topic (``retro_stamp_topics.py``'s job), never a
    non-conforming one.  Both the corpus boundary and the planner ask this, so
    the two cannot disagree about which records need attention.
    """
    return topic is not None and not is_valid_topic_slug(topic)


def in_selection(legacy_topic: object, topics: frozenset[str] | None) -> bool:
    """Whether a ``--topic`` selection covers *legacy_topic*.

    ``None`` is no selection and covers every value.  A selection holds only
    strings, so a non-string topic (a ``topic_unfoldable`` entry) is never in one.
    """
    return topics is None or (isinstance(legacy_topic, str) and legacy_topic in topics)


def _selection_defect(topic: object) -> str | None:
    """Why *topic* cannot be a selected legacy value, or ``None`` if it can."""
    if not isinstance(topic, str):
        return 'not a string'
    if is_valid_topic_slug(topic):
        return 'already conforms; name the legacy value that folds onto it'
    if derive_topic_slug(topic) is None:
        return 'has no fold'
    return None


def refuse_invalid_selection(topics: frozenset[str] | None) -> None:
    """Raise ``ValueError`` unless *topics* is ``None`` or names foldable legacy values.

    Passing the target slug instead of its source is the likely typo.  Left
    unrefused, it would plan nothing and exit 0, which reads as a clean run.
    A legacy value no record carries is NOT refused: a re-run after a
    successful ``--apply`` legitimately finds the value drained.
    """
    if topics is None:
        return
    if not topics:
        raise ValueError(
            'empty topic selection: name at least one legacy topic value, or '
            'omit --topic to migrate every non-conforming value')
    defects = sorted(
        f'{topic!r} ({defect})'
        for topic in topics
        if (defect := _selection_defect(topic)) is not None
    )
    if defects:
        raise ValueError(
            'a topic selection names LEGACY source values, never target slugs; '
            'refused: ' + ', '.join(defects))


def plan_renames(
    records,
    *,
    project_id: str,
    occupied_topics=None,
    canonical_ids_by_topic=None,
    topics: frozenset[str] | None = None,
) -> tuple[list[Rename], list[dict]]:
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
        topics: The legacy values whose verdicts to return; ``None`` returns
            every verdict.  It narrows the verdicts only: a selected value
            that an unselected spelling contests is still refused.

    Returns:
        ``(renames, skips)`` — frozen :class:`Rename` rows, and report-ready
        skip dicts each carrying a ``reason``.

    Raises:
        ValueError: *topics* names no foldable legacy value
            (:func:`refuse_invalid_selection`).
    """
    refuse_invalid_selection(topics)
    renames: list[Rename] = []
    skips: list[dict] = []

    # --- pass 1: what does the corpus already look like? -------------------
    # Built BEFORE any verdict, from the whole record set, so no decision can
    # depend on the order the backend happened to page records back in.  It
    # ignores *topics* for the same reason: the planner must see every legacy
    # value to know which targets are contested.
    #
    #   observed_topics       every topic value present, conforming or not
    #   canonical_ids_by_topic  which records under a topic claim canonical
    #   sources_by_target     which legacy values want to fold onto a target
    #
    # *occupied_topics* / *canonical_ids_by_topic* carry the incumbents that
    # are NOT in *records*.  ``enumerate_topic_bearing`` discards conforming
    # records to keep a corpus sweep affordable, so on the live path the
    # incumbent a fold would collide with has already been thrown away by the
    # time this function runs — and a collision guard that cannot see the
    # incumbent silently degrades into a namespace merger.  Tests that pass a
    # complete record set (conforming rows included) need neither argument.
    observed_topics: set[str] = set(occupied_topics or ())
    canonical_ids_by_topic = {
        topic: list(ids) for topic, ids in (canonical_ids_by_topic or {}).items()
    }
    sources_by_target: dict[str, set[str]] = {}
    for record in records:
        metadata = record.get('metadata') or {}
        topic = metadata.get('topic')
        if not isinstance(topic, str):
            continue
        observed_topics.add(topic)
        if is_canonical(metadata):
            memory_id = record.get('id')
            if isinstance(memory_id, str) and memory_id and memory_id not in (
                canonical_ids_by_topic.get(topic) or ()
            ):
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
        topic = metadata.get('topic')
        if not is_non_conforming_topic(topic) or not in_selection(topic, topics):
            continue

        memory_id = record.get('id')
        folded = derive_topic_slug(topic)
        if not isinstance(topic, str) or folded is None:
            skips.append({
                'reason': 'topic_unfoldable',
                'project_id': project_id,
                'memory_id': memory_id,
                'topic': topic,
                'topic_type': type(topic).__name__,
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
