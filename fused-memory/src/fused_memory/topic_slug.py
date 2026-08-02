"""The ONE topic-slug namespace (task 3198, leaf ε).

Single normative home for the shared ``topic`` slug shape defined by
``docs/prds/memory-metadata-vocabulary.md`` D4: ``metadata.topic`` (the
Mem0 vocabulary key) and :attr:`ProceduralTopicCluster.topic_id` (the
config-side write-time topic guard) are **one namespace with one regex**.
Both consumers import these objects; per INV-5, a second copy of the
pattern or the cap anywhere in the tree is a bug.

Why this is a separate leaf rather than living in the registry
--------------------------------------------------------------
:mod:`fused_memory.memory_metadata` is the advertised vocabulary home and
still re-exports these names, so nothing about the registry's contract
changed.  But ``config/schema.py`` **cannot** import from it.  Measured,
not theorised: ``memory_metadata`` imports
:mod:`fused_memory.backends.mem0_client`, which imports
``fused_memory.config.schema`` at module scope.  Loading a patched
``config/schema.py`` carrying ``from fused_memory.memory_metadata import
TOPIC_SLUG_RE`` under the real module name raises::

    ImportError: cannot import name 'FusedMemoryConfig'
                 from 'fused_memory.config.schema'

A function-local deferred import would dodge that ImportError but is
worse: ``_default_topic_guard_clusters()`` builds clusters at
config-default time, so the validator can fire while ``mem0_client`` is
mid-import.  It would also make config loading pull the ``mem0`` SDK —
measurably not required today (``import fused_memory.config.schema``
leaves ``mem0`` out of ``sys.modules``), and
``orchestrator/tests/test_reopen_sticks_e2e.py:58`` imports
``FusedMemoryConfig`` without it.

Hence this module imports **only the standard library**, and must keep
doing so.  ``tests/test_topic_slug_namespace.py`` pins that in a fresh
interpreter; if you add a ``fused_memory`` import here, the extraction
buys nothing and the cycle above comes back.

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
"""

from __future__ import annotations

import re

__all__ = [
    'TOPIC_SLUG_MAX_LEN',
    'TOPIC_SLUG_RE',
    'is_valid_topic_slug',
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


def is_valid_topic_slug(value: object) -> bool:
    """Return whether *value* is a well-formed topic slug.

    THE shared predicate — both consumers call this rather than
    re-expressing the regex or the cap inline, so "is this a valid topic"
    has exactly one answer across the memory side and the config side
    (D4).  ``tests/test_topic_slug_namespace.py`` asserts that identity by
    ``is``, so an inlined copy fails by design.

    Takes ``object`` and never raises: both call sites hand it untrusted
    values, and ``TOPIC_SLUG_RE.match(None)`` would turn a rejection into
    a ``TypeError``.  A non-``str`` is a ``False`` verdict.

    Returns a real ``bool``, not the truthy ``re.Match`` — callers store
    the result on pydantic models and in log payloads.
    """
    return (
        isinstance(value, str)
        and bool(TOPIC_SLUG_RE.match(value))
        and len(value) <= TOPIC_SLUG_MAX_LEN
    )
