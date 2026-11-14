"""The one strict reading of the ``canonical`` metadata marker (task 6215).

Every reader that asks "does this record claim canonical?" calls
:func:`is_canonical`; none re-types the comparison.

* Identity, not truthiness or equality: ``1 == True`` and ``1.0 == True``,
  and a string ``'true'`` is truthy, so only the bool ``True`` claims.
* A non-``Mapping`` is not canonical: raw store payloads are not
  schema-enforced at read time, so a reader degrades rather than raises.
* A separate stdlib-only leaf, not part of :mod:`fused_memory.memory_metadata`
  (which re-exports it as the same object), because
  ``reconciliation/consolidation_auto.py`` is a stdlib-only import leaf and
  cannot import the registry.  :mod:`fused_memory.topic_slug` is the
  precedent and states the measured import cost.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ['is_canonical']


def is_canonical(meta: object) -> bool:
    return isinstance(meta, Mapping) and meta.get('canonical') is True
