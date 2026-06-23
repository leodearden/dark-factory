"""Tests for the OverlapFootprintDetector seam (PRD §5.1, task γ).

Covers:
  - Footprint / DefaultPathOverlapDetector contract (steps 1-2)
  - Registration + default fallback (steps 3-4)
  - Fail-open convenience wrapper changesets_overlap (steps 5-6)
  - merge_queue facade re-export surface (steps 7-8)
"""

from __future__ import annotations

import pytest

import orchestrator.overlap_footprint as _ov_mod
from orchestrator.overlap_footprint import (
    DEFAULT_OVERLAP_DETECTOR,
    DefaultPathOverlapDetector,
    Footprint,
    OverlapFootprintDetector,
    get_overlap_detector,
    register_overlap_detector,
)


# ---------------------------------------------------------------------------
# Step 1: DefaultPathOverlapDetector contract tests
# ---------------------------------------------------------------------------


class TestFootprintAndDefaultDetector:
    """Contract tests for Footprint and DefaultPathOverlapDetector."""

    def test_footprint_returns_footprint_instance(self) -> None:
        det = DefaultPathOverlapDetector()
        fp = det.footprint(["a.py", "b.py"])
        assert isinstance(fp, Footprint)

    def test_footprint_equal_for_same_paths(self) -> None:
        det = DefaultPathOverlapDetector()
        fp1 = det.footprint(["a.py", "b.py"])
        fp2 = det.footprint(["b.py", "a.py"])  # order irrelevant
        assert fp1 == fp2

    def test_footprint_is_hashable(self) -> None:
        det = DefaultPathOverlapDetector()
        fp = det.footprint(["a.py", "b.py"])
        # Putting in a set exercises __hash__
        s = {fp}
        assert fp in s

    def test_overlaps_true_for_shared_path(self) -> None:
        det = DefaultPathOverlapDetector()
        fa = det.footprint(["a.py", "b.py"])
        fb = det.footprint(["b.py", "c.py"])
        assert det.overlaps(fa, fb) is True

    def test_overlaps_false_for_disjoint_paths(self) -> None:
        det = DefaultPathOverlapDetector()
        fa = det.footprint(["a.py"])
        fb = det.footprint(["b.py"])
        assert det.overlaps(fa, fb) is False

    def test_overlaps_symmetry(self) -> None:
        det = DefaultPathOverlapDetector()
        fa = det.footprint(["a.py", "b.py"])
        fb = det.footprint(["b.py", "c.py"])
        assert det.overlaps(fa, fb) == det.overlaps(fb, fa)

    def test_overlaps_reflexivity_nonempty(self) -> None:
        det = DefaultPathOverlapDetector()
        fa = det.footprint(["a.py", "b.py"])
        assert det.overlaps(fa, fa) is True

    def test_empty_footprint_disjoint_from_everything(self) -> None:
        det = DefaultPathOverlapDetector()
        empty = det.footprint([])
        nonempty = det.footprint(["a.py"])
        assert det.overlaps(empty, nonempty) is False
        assert det.overlaps(nonempty, empty) is False
        assert det.overlaps(empty, empty) is False

    def test_textual_conflict_implies_overlap(self) -> None:
        """§5.1 superset property: two changesets editing the same path => overlaps True."""
        det = DefaultPathOverlapDetector()
        shared = "src/main.py"
        fa = det.footprint([shared, "src/utils.py"])
        fb = det.footprint([shared, "src/other.py"])
        assert det.overlaps(fa, fb) is True

    def test_default_overlap_detector_conforms_to_protocol(self) -> None:
        """isinstance check verifies @runtime_checkable Protocol conformance."""
        assert isinstance(DEFAULT_OVERLAP_DETECTOR, OverlapFootprintDetector)


# ---------------------------------------------------------------------------
# Step 3: Registration + default fallback tests
# ---------------------------------------------------------------------------


class _StubDetector:
    """Minimal detector stub for registry tests."""

    def footprint(self, changed_paths: object) -> Footprint:
        return Footprint(members=frozenset())

    def overlaps(self, a: Footprint, b: Footprint) -> bool:
        return False


@pytest.fixture()
def clean_registry() -> object:
    """Snapshot and restore _DETECTORS so tests don't leak global state."""
    snapshot = dict(_ov_mod._DETECTORS)
    yield
    _ov_mod._DETECTORS.clear()
    _ov_mod._DETECTORS.update(snapshot)


class TestRegistry:
    """Registration + default fallback."""

    def test_none_returns_default(self, clean_registry: object) -> None:
        assert get_overlap_detector(None) is DEFAULT_OVERLAP_DETECTOR

    def test_unregistered_project_returns_default(self, clean_registry: object) -> None:
        assert get_overlap_detector("some_unregistered_project") is DEFAULT_OVERLAP_DETECTOR

    def test_registered_project_returns_custom(self, clean_registry: object) -> None:
        stub = _StubDetector()
        register_overlap_detector("reify", stub)
        assert get_overlap_detector("reify") is stub

    def test_other_project_still_gets_default_after_registration(
        self, clean_registry: object
    ) -> None:
        stub = _StubDetector()
        register_overlap_detector("reify", stub)
        assert get_overlap_detector("dark_factory") is DEFAULT_OVERLAP_DETECTOR

    def test_reregistration_overwrites(self, clean_registry: object) -> None:
        stub1 = _StubDetector()
        stub2 = _StubDetector()
        register_overlap_detector("reify", stub1)
        register_overlap_detector("reify", stub2)
        assert get_overlap_detector("reify") is stub2
