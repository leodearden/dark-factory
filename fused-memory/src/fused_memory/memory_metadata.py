"""Normative Mem0 metadata vocabulary registry (task 3195, leaf β).

This module is the **single normative home** for the Mem0 metadata
vocabulary defined by ``docs/prds/memory-metadata-vocabulary.md`` (V1).
Per INV-5 and PRD §6, consumers **import** from here — they never restate
the vocabulary.  A second copy of any constant in this module is a bug.

Contents
--------
* ``TOPIC_SLUG_RE`` / ``TOPIC_SLUG_MAX_LEN`` — the shared ``topic`` slug
  shape (PRD D4: ``ProceduralTopicCluster.topic_id`` and
  ``metadata.topic`` are one namespace with one regex).
* ``normalize_supersedes`` — PRD D2's scalar/list/None normalizer.

Measured basis for the slug shape
---------------------------------
Derived from leaf α's census
(``plans/memory-metadata-census-report.json`` @ ``b5af3e4b03``,
``coverage.complete = true``) rather than guessed:

* accepts all **5** seeded ``ProceduralTopicCluster.topic_id`` values
  (PRD §10's one hard requirement); longest is 52 chars;
* accepts **254 of 352** distinct live ``topic`` values (355 of 491
  records); the longest conforming live value is 69 chars, so the
  100-char cap bounds the key while rejecting nothing observed;
* the 98 non-conforming live values are all snake_case.  Under the
  warn-mode default (``memory_metadata.enforce = False``) these emit a
  census line and the write proceeds — leaf θ's bounded retro-stamping
  sweep is the intended normalizer.  This is why the warn default is
  load-bearing rather than merely cautious.

``kind`` is deliberately **NOT** slug-validated: 321 of the 329 live
``kind`` values are snake_case, so applying this regex to ``kind`` would
reject essentially the entire live population.  ``kind`` is
registry-membership-validated instead, exactly as V1 specifies.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    'TOPIC_SLUG_MAX_LEN',
    'TOPIC_SLUG_RE',
    'normalize_supersedes',
]


#: Shared ``topic`` slug shape (PRD D4 — one namespace for
#: ``ProceduralTopicCluster.topic_id`` and ``metadata.topic``).
#:
#: Lowercase alphanumeric segments joined by single hyphens.  Anchored at
#: both ends with ``\Z`` rather than ``$`` so a trailing newline cannot
#: sneak past (``$`` matches before a final ``\n``).
TOPIC_SLUG_RE = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*\Z')

#: Maximum ``topic`` slug length.  See the module docstring for the
#: measured basis (longest conforming live topic 69, longest seeded
#: cluster id 52 — 100 has headroom and rejects nothing observed).
TOPIC_SLUG_MAX_LEN = 100


def normalize_supersedes(value: Any) -> list[Any]:
    """Normalize a ``supersedes`` metadata value to a list (PRD D2).

    ``supersedes`` is a list in V1, but the corpus carries 81 records with
    a **scalar** value and 65 with a list.  The live scalar writer is
    ``reconciliation/harness.py:1167``; the readers are
    ``reconciliation/targeted.py:1464`` (truthiness discriminator) and
    leaf 3112's closure predicate.  Both go through this helper so the
    legacy scalar shape stays tolerated on read.

    Accepts ``None`` (→ ``[]``), a scalar (→ single-element list), or any
    non-``str`` sequence (→ list copy).  The returned list is always a
    fresh object, never an alias of the caller's list.

    This function **never drops or coerces members**.  A malformed member
    (short hex, non-string — the census counts 3 and 8 live respectively)
    survives normalization intact so that
    :func:`validate_memory_metadata` can reject it *by name*.  Silently
    dropping it here would be a silent-fail-soft: the write would succeed
    having quietly discarded a supersession edge.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    # Any other scalar (int, dict, ...) is wrapped rather than rejected —
    # the shape validator owns rejection, this function owns shape only.
    return [value]
