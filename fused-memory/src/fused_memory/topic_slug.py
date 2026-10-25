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
    'derive_topic_slug',
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

    THE shared predicate — every consumer calls this rather than
    re-expressing the regex or the cap inline, so "is this a valid topic"
    has exactly one answer across the memory side and the config side
    (D4).  ``tests/test_topic_slug_namespace.py`` asserts that identity by
    ``is``, so an inlined copy of the *constants* fails by design.

    The call sites, named so the claim above stays checkable (the ``is``
    tests pin the constants but cannot see an inlined re-expression of the
    RULE, which is how one crept back in once already — 3198 amendment):

    * ``memory_metadata.validate_memory_metadata`` §2a — the ``topic``
      shape check;
    * ``config.schema.ProceduralTopicCluster._validate_topic_id_slug`` —
      the config-side ``topic_id`` check;
    * ``services.memory_service._check_canonical_uniqueness`` guard 2 —
      refuses to build a store query on a malformed key.

    Add a step to this predicate (a normalization pass, a reserved-prefix
    rule) and all three move together; inline the rule at any of them and
    they silently diverge.

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


#: Any run of characters that cannot appear inside a slug segment.  Note the
#: complement class is ``[a-z0-9]`` only: ``_`` is NOT preserved, which is the
#: whole point of the fold (98 of 352 live topic values are snake_case).
#:
#: This is deliberately NOT the anchored slug validator :data:`TOPIC_SLUG_RE`
#: above — two different patterns doing two different jobs.  The *verdict*
#: still has one home: :func:`derive_topic_slug` never decides validity
#: itself, it asks :func:`is_valid_topic_slug`.
_NON_SLUG_RUN_RE = re.compile(r'[^a-z0-9]+')


def derive_topic_slug(value: object) -> str | None:
    """Fold *value* into this module's topic-slug shape, or ``None``.

    The fold: lowercase, strip, collapse every run of non-``[a-z0-9]``
    characters (which includes ``_``, so snake_case becomes hyphen-case) to a
    single ``-``, then strip leading/trailing hyphens.  The result is returned
    **only** if :func:`is_valid_topic_slug` accepts it — which is also where
    the :data:`TOPIC_SLUG_MAX_LEN` cap is enforced.

    Returning ``None`` rather than a repaired value is load-bearing.  An
    over-long topic truncated to 100 chars, or ``'!!!'`` turned into
    ``'unnamed-topic'``, would file a record under a topic no human chose;
    the caller instead reports it and moves on (loud over silent).

    NOT a copy of ``memory_eval_retrieval_probe._slugify``, and the two must
    not be "unified": that one preserves ``_`` and falls back to
    ``'unnamed-topic'``, so it emits slugs this module *rejects*.  It is right
    for its own job (naming derivation candidates for human review) and wrong
    for this one (writing a validated vocabulary key to the corpus).

    Lives here rather than in a script because it has two consumers and INV-5
    gives a shared rule one home (task 4878): ``scripts/retro_stamp_topics.py``
    (the bounded, id-addressed stamping sweep, which defined it first) and
    ``scripts/normalize_topic_slugs.py`` (the corpus-wide normalization
    migration).  Both import it; neither re-expresses it.

    Args:
        value: Any object.  A non-``str`` is a ``None`` verdict, matching
            :func:`is_valid_topic_slug`'s "non-str is False" convention — both
            are handed untrusted values off live records and fixtures.

    Returns:
        The conforming slug, or ``None`` when no honest fold exists.
    """
    if not isinstance(value, str):
        return None
    folded = _NON_SLUG_RUN_RE.sub('-', value.strip().lower()).strip('-')
    return folded if is_valid_topic_slug(folded) else None
