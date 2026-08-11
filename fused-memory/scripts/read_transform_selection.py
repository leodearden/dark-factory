#!/usr/bin/env python3
"""Read-transform selection over the ratified C write shape (task 4004).

WHAT THIS IS
------------
``bake_off_storage_shape.py`` arbitrated the WRITE shape (E2, leaf ζ) and
its gate ratified **C-peers**.  That settles where the bytes go; it does
not settle what a reader is handed back.  This script measures the READ
side over that already-ratified write shape, so task **3111** (the topic
pin) can be landed against a measurement instead of an intuition.

Three candidate transforms, each a pure function over an already-fetched
hit list, each scored against the flat read as the baseline:

  1. **promoting topic pin** — the landed ``apply_topic_anchor``'s firing
     rule, but the canonical is promoted INTO the window instead of
     appended past its edge.
  2. **topic-keyed grouped read** — grouping keyed on ``topic`` rather
     than on ``parent_id``, so C's *peers* group without a parent-link
     writer, plus an explicit sighting-crediting knob.
  3. **topic-diversity cap** — at most *n* records per topic in the
     window (MMR-shaped), the cheapest family member.

WHY A SIBLING SCRIPT, NOT MORE ARMS IN THE BAKE-OFF
---------------------------------------------------
``bake_off_storage_shape.py``'s ``ARM_VARIANTS`` / ``_REQUIRED_ARM_METRICS``
/ ``_check_arms`` and its committed-artifact tests assert the E2 arm set and
protocol block exactly, because a partial table must never render as a
complete one.  Injecting read-side arms there would break those guards and
would rewrite an artifact that is the record of a DIFFERENT experiment.
This script therefore emits a SIBLING artifact pair,
``plans/read-transform-selection-report.{json,md}``, and takes the bake-off
module as a library.

WHAT IS REUSED, AND WHY NOTHING IS REIMPLEMENTED
-------------------------------------------------
Everything measurable already exists next door and is imported unchanged:
``ArmRecord``, ``ScoredHit``, ``build_canonical_by_topic``,
``claim_recall_at_k``, ``tokens_returned``, ``resolve_token_estimator``,
``_mean``, ``_rate``, the ``_atomic_write_text`` / ``_cell`` /
``_NO_MEASUREMENT`` artifact discipline, and the fetch cache.  The bake-off
carries an explicit INV-5 note (:1066-1083) about why there are not two
``recall@k`` implementations in this repo; the same argument forbids a
third here.  It also keeps this table's token column directly comparable
with the committed E2 table, since both resolve the identical estimator.

MEASUREMENT DISCIPLINE
----------------------
Inherited verbatim from the bake-off: rank/set-based metrics only, a
``None`` is *no measurement* and never a measured zero, and a missing
metric raises rather than rendering a partial row.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_script(name: str) -> Any:
    """Load a sibling script in ``scripts/`` by path.

    The same mechanism as ``bake_off_storage_shape._load_sibling_script``
    (:1513), and registered in ``sys.modules`` for the same reason:
    ``@dataclass`` and other reflection-based decorators look the defining
    module up there, so a module loaded without registering breaks on the
    way in.

    It is spelled out here rather than imported from the bake-off only
    because of the chicken-and-egg — this is the helper that loads the
    bake-off in the first place.  Once :func:`bake_off` has returned, any
    FURTHER sibling (``memory_eval_transcript_corpus``, the calibration
    fixture loader) is loaded through the bake-off's own helper, so the
    duplication stops at this one bootstrap.
    """
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if name in sys.modules:
        return sys.modules[name]
    path = _SCRIPTS_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load sibling script {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def bake_off() -> Any:
    """``scripts/bake_off_storage_shape.py`` — the measurement library."""
    return _load_script('bake_off_storage_shape')


# ---------------------------------------------------------------------------
# Arm (1): the PROMOTING topic pin
# ---------------------------------------------------------------------------


def apply_promoting_topic_anchor(
    hits: list[Any],
    canonical_by_topic: dict[str, Any],
) -> list[Any]:
    """Pin each present topic's canonical to the FRONT of the ranking.

    The firing rule is the landed ``apply_topic_anchor``'s, unchanged: the
    pin fires when a hit carries ``metadata['topic'] == T``, and the record
    it pins is ``canonical_by_topic[T]``.  The two therefore select the
    identical SET of records and differ in exactly one dimension — WHERE
    the canonical lands — which is what makes this a controlled variant of
    the shipped transform rather than a second, differently-triggered rule.

    WHY A PROMOTING VARIANT EXISTS AT ALL
    -------------------------------------
    ``apply_topic_anchor`` APPENDS, deliberately, so that a measured
    improvement is attributable to the canonical becoming *reachable at
    all* rather than to the transform having hand-placed it high.  But
    ``read_path`` truncates AFTER the transforms (``records[:k]``, :3243),
    so at an already-FULL window the appended canonical is truncated
    straight back off and the pin cannot change the reader's window at all.
    The E2 report's ``pin changed window = 0.00`` under c_peers is that
    arithmetic — it is forced, not a verdict on anchoring.  Only a
    transform that places the canonical INSIDE the budget can be measured
    at a full window, which is what this one does.

    THE COST, STATED
    ----------------
    At ``len(hits) == k`` promotion evicts the k-th ranked record from the
    reader's window.  This function itself drops nothing — it reorders, and
    returns every input record — but downstream truncation at the reader's
    budget makes the pair *lossy at the window edge*.  That is a real cost
    of arm (1) and the report states it rather than leaving 3111 to
    discover it: the pin buys canonical reachability by spending the last
    slot.  It is also why an additive pin remains the right default where
    the window has headroom, where the two transforms agree exactly.

    ORDERING
    --------
    Promoted canonicals lead, ordered by their topic's FIRST APPEARANCE in
    the ranking, and everything else follows in its original relative
    order.  First-appearance is the store's own ordering signal and the
    only one available here; a set-derived order would make the answer
    depend on hash iteration, and a topic-name sort would impose an
    alphabetical prior on the reader's window.  A canonical that was
    ITSELF ranked is MOVED, never copied, so no record is ever duplicated
    and the window size is unchanged.

    ``contested`` is not read — not as an argument, not as a metadata key.
    It is a hand-labelled bake-off FIXTURE field, absent from the live
    ``RESERVED_VOCABULARY_KEYS`` (``fused_memory/memory_metadata.py``:601)
    with no writer and no adjudication surface, so an arm that needed it
    would be unimplementable today.  Arm (1) needs it not at all.

    ``canonical is True`` is NOT re-checked here, for the same reason
    ``apply_topic_anchor`` does not re-check it: β's
    ``invalid_canonical_type`` rule is bool identity and is enforced once,
    at index-construction time, by ``build_canonical_by_topic``.

    Returns the input list unchanged — same object, same elements — when
    nothing is pinnable.  Pure: never mutates *hits*, *canonical_by_topic*,
    or any record in either.
    """
    # dict-not-set: insertion order makes the promoted prefix deterministic.
    present_topics: dict[str, None] = {}
    for hit in hits:
        topic = hit.metadata.get('topic')
        if topic is not None:
            present_topics.setdefault(topic, None)

    promoted: list[Any] = []
    promoted_ids: set[str] = set()
    for topic in present_topics:
        canonical = canonical_by_topic.get(topic)
        if canonical is None:
            continue  # no canonical for this topic — never synthesize one
        if canonical.record_id in promoted_ids:
            continue  # two topics, one canonical: promote it once
        promoted.append(canonical)
        promoted_ids.add(canonical.record_id)

    if not promoted:
        return hits
    # An already-ranked canonical is MOVED to the front, not duplicated;
    # every other hit keeps its relative order.
    return [*promoted, *(hit for hit in hits if hit.record_id not in promoted_ids)]
