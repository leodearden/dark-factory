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

from fused_memory.topic_slug import TOPIC_SLUG_MAX_LEN, is_valid_topic_slug

__all__ = [
    'TOPIC_SLUG_MAX_LEN',
    'derive_topic_slug',
    'is_valid_topic_slug',
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
