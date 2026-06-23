"""Overlap-footprint seam for the two-layer merge queue (PRD §5.1, task γ).

This module defines the cross-project pluggable interface used to determine
whether two changesets' footprints overlap, triggering re-verification in the
merge queue.  It mirrors the 1595 disjoint-delta relation already present in
merge_queue.py (_rebase_delta_touched_overlap, :1778-1907) and
(_reverify_rebased_tree, :1910-1991): a NON-EMPTY intersection of the
branch-touched path sets ⇒ overlap ⇒ re-verify.

The seam lives in a dedicated module so that reify (task κ, external dep
dark_factory:1888) can import only this module — without pulling in the 11 k-line
merge_queue.py and its heavy git/verify dependency tree.

Public surface (also re-exported from orchestrator.merge_queue for in-subsystem
consumers):
  Footprint
  OverlapFootprintDetector
  DefaultPathOverlapDetector
  DEFAULT_OVERLAP_DETECTOR
  register_overlap_detector
  get_overlap_detector
  changesets_overlap
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Footprint — opaque, value-comparable, hashable set of strings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """An opaque, value-comparable, hashable representation of a changeset footprint.

    Both the default path detector and reify's crate-graph detector produce
    string sets, so one concrete Footprint type keeps the Protocol signature
    simple.  Callers should compare via ``detector.overlaps(a, b)`` rather
    than inspecting ``members`` directly.
    """

    members: frozenset[str]


# ---------------------------------------------------------------------------
# OverlapFootprintDetector Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OverlapFootprintDetector(Protocol):
    """Pluggable interface for computing changeset overlap footprints.

    Implementations must be stateless / pure: footprint() and overlaps()
    must not swallow exceptions.  Fail-open logic lives in the
    ``changesets_overlap`` convenience wrapper, not here.
    """

    def footprint(self, changed_paths: Sequence[str]) -> Footprint:
        """Return the overlap footprint for a set of changed paths."""
        ...  # pragma: no cover

    def overlaps(self, a: Footprint, b: Footprint) -> bool:
        """Return True iff footprints a and b overlap (re-verify required)."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# DefaultPathOverlapDetector
# ---------------------------------------------------------------------------


class DefaultPathOverlapDetector:
    """Default detector: footprint = frozenset of changed file paths;
    overlaps = non-empty path intersection.

    Mirrors the 1595 disjoint-delta relation from merge_queue.py:
      * branch_touched & intervening_delta ≠ ∅  ⟹  overlap
    """

    def footprint(self, changed_paths: Sequence[str]) -> Footprint:
        return Footprint(members=frozenset(changed_paths))

    def overlaps(self, a: Footprint, b: Footprint) -> bool:
        return bool(a.members & b.members)


# Module-level singleton — importers can use this directly.
DEFAULT_OVERLAP_DETECTOR: OverlapFootprintDetector = DefaultPathOverlapDetector()


# ---------------------------------------------------------------------------
# Project-id-keyed registry
# ---------------------------------------------------------------------------

_DETECTORS: dict[str, OverlapFootprintDetector] = {}


def register_overlap_detector(project_id: str, detector: OverlapFootprintDetector) -> None:
    """Register a custom detector for *project_id* (last-registration-wins).

    dark-factory does NOT pre-register itself — the default is reached via
    the fallback in ``get_overlap_detector``.  Only overrides (e.g. reify,
    task κ) call this function.
    """
    _DETECTORS[project_id] = detector


def get_overlap_detector(project_id: str | None = None) -> OverlapFootprintDetector:
    """Return the detector for *project_id*, or ``DEFAULT_OVERLAP_DETECTOR``.

    Absence (``None`` or unregistered id) ⟹ default, per §5.1.
    """
    if not project_id:
        return DEFAULT_OVERLAP_DETECTOR
    return _DETECTORS.get(project_id, DEFAULT_OVERLAP_DETECTOR)


# ---------------------------------------------------------------------------
# Fail-open convenience wrapper
# ---------------------------------------------------------------------------


def changesets_overlap(
    project_id: str | None,
    a_paths: Sequence[str],
    b_paths: Sequence[str],
) -> bool:
    """Return True iff the two changesets' footprints overlap.

    Selects the detector by *project_id* (absence ⟹ default), builds both
    footprints, and returns ``detector.overlaps(fa, fb)``.

    FAIL-OPEN per §5.1: any exception raised during detector selection,
    footprint construction, or the overlaps check is caught, logged at
    WARNING (with traceback), and the function returns ``True`` — never
    skip a re-verify on detector error.
    """
    try:
        detector = get_overlap_detector(project_id)
        fa = detector.footprint(a_paths)
        fb = detector.footprint(b_paths)
        return detector.overlaps(fa, fb)
    except Exception:
        logger.warning(
            "changesets_overlap: detector raised; returning True (fail-open). "
            "project_id=%r",
            project_id,
            exc_info=True,
        )
        return True
