"""Doc-content and doc/code-consistency tests for the two-layer merge queue.

This file encodes the required content of the operator-facing and
developer-facing two-layer merge-queue documentation as executable
assertions (TDD-for-docs pattern, mirroring test_skill_prompt.py).

Each ``test_skill_*`` test asserts that stable phrases are present in
``skills/merge-queue/SKILL.md``; ``test_design_doc_*`` tests target
``skills/merge-queue/references/two-layer-model.md``.  A doc/code-
consistency guard checks that every code symbol the docs cite exists in
``orchestrator/src/orchestrator/merge_queue.py`` (MERGE_QUEUE_SRC).

Authoritative source for required content: the λ=1895 integration-test
header (test_merge_queue_two_layer_integration.py:1-55) and the real
public symbols in merge_queue.py.
"""

from __future__ import annotations

import pathlib


def _repo_root() -> pathlib.Path:
    """Return the repository root (two levels above this test file)."""
    # __file__ → orchestrator/tests/test_merge_queue_docs.py
    # parents[0] → orchestrator/tests/
    # parents[1] → orchestrator/
    # parents[2] → <repo root>
    return pathlib.Path(__file__).parents[2]


def _read(relpath: str) -> str:
    """Return the text of a repo-root-relative file."""
    return (_repo_root() / relpath).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Doc/code consistency guard — read once; all tests share this constant.
# Every code symbol the docs cite must exist in this source text.
# ---------------------------------------------------------------------------
MERGE_QUEUE_SRC: str = _read("orchestrator/src/orchestrator/merge_queue.py")


class TestSkillDocumentsNeedsRebaseBounce:
    """SKILL.md must describe the needs_rebase bounce and two-layer structure."""

    _skill = _read("skills/merge-queue/SKILL.md")

    def test_two_layer_section_heading(self) -> None:
        """SKILL.md must contain a two-layer section heading."""
        assert "two-layer merge queue" in self._skill.lower(), (
            "skills/merge-queue/SKILL.md must contain a heading that "
            "introduces the two-layer merge queue"
        )

    def test_needs_rebase_outcome_documented(self) -> None:
        """SKILL.md must document the needs_rebase outcome."""
        assert "needs_rebase" in self._skill, (
            "skills/merge-queue/SKILL.md must document the needs_rebase "
            "merge outcome"
        )

    def test_graph_time_disk_free_bounce(self) -> None:
        """SKILL.md must describe the graph-time, disk-free bounce."""
        skill_lower = self._skill.lower()
        # The bounce happens at conflict-graph time, before any verify slot is consumed
        graph_time = (
            "graph-time" in skill_lower
            or "graph time" in skill_lower
            or "conflict graph" in skill_lower
        )
        disk_free = (
            "disk-free" in skill_lower
            or "disk free" in skill_lower
            or "no verify slot" in skill_lower
            or "before it consumes a verify slot" in skill_lower
            or "without consuming a verify slot" in skill_lower
        )
        assert graph_time, (
            "skills/merge-queue/SKILL.md must describe the graph-time bounce "
            "(bounce happens at conflict-graph computation time)"
        )
        assert disk_free, (
            "skills/merge-queue/SKILL.md must describe the disk-free nature of "
            "the bounce (no verify slot consumed)"
        )

    # Doc/code consistency: cited constants must exist in merge_queue.py

    def test_needs_rebase_reason_prefix_exists_in_code(self) -> None:
        """NEEDS_REBASE_REASON_PREFIX must exist in merge_queue.py."""
        assert "NEEDS_REBASE_REASON_PREFIX" in MERGE_QUEUE_SRC, (
            "NEEDS_REBASE_REASON_PREFIX not found in merge_queue.py — "
            "doc cites a symbol that doesn't exist in the code"
        )

    def test_merge_bounce_cap_exists_in_code(self) -> None:
        """MERGE_BOUNCE_CAP must exist in merge_queue.py."""
        assert "MERGE_BOUNCE_CAP" in MERGE_QUEUE_SRC, (
            "MERGE_BOUNCE_CAP not found in merge_queue.py — "
            "doc cites a symbol that doesn't exist in the code"
        )


class TestSkillDocumentsAgingOrderAndCrossrefs:
    """SKILL.md must describe aging order, frozen-prefix invariant, and cross-refs."""

    _skill = _read("skills/merge-queue/SKILL.md")
    _skill_lower = _skill.lower()

    def test_age_of_first_submission_phrase(self) -> None:
        """SKILL.md must describe the age-of-first-submission landing order."""
        has_phrase = (
            "age of first submission" in self._skill_lower
            or "first submission" in self._skill_lower
        )
        assert has_phrase, (
            "skills/merge-queue/SKILL.md must describe the age-of-first-submission "
            "landing order (conflict-clique-scoped)"
        )

    def test_conflict_clique_phrase(self) -> None:
        """SKILL.md must mention conflict cliques for the aging order."""
        assert "clique" in self._skill_lower, (
            "skills/merge-queue/SKILL.md must mention 'clique' (conflict-clique-scoped "
            "aging order)"
        )

    def test_disjoint_throughput_bypass_phrase(self) -> None:
        """SKILL.md must describe the disjoint throughput bypass."""
        has_phrase = (
            "disjoint" in self._skill_lower
            or "throughput bypass" in self._skill_lower
        )
        assert has_phrase, (
            "skills/merge-queue/SKILL.md must describe the disjoint-item throughput bypass"
        )

    def test_frozen_prefix_immutability_invariant(self) -> None:
        """SKILL.md must describe the frozen-prefix/verify-frontier immutability invariant."""
        has_frozen = (
            "frozen prefix" in self._skill_lower
            or "frozen-prefix" in self._skill_lower
        )
        has_immutable = (
            "immutable" in self._skill_lower
            or "immutability" in self._skill_lower
            or "never reordered" in self._skill_lower
        )
        assert has_frozen, (
            "skills/merge-queue/SKILL.md must describe the frozen-prefix concept"
        )
        assert has_immutable, (
            "skills/merge-queue/SKILL.md must describe the immutability of "
            "items in the frozen verify frontier"
        )

    def test_crossref_warm_lane_dp_batch(self) -> None:
        """SKILL.md must reference the complementary warm-lane Δp space-safety batch (1859)."""
        assert "1859" in self._skill, (
            "skills/merge-queue/SKILL.md must cross-reference the warm-lane "
            "Δp space-safety batch (task 1859)"
        )

    def test_crossref_enospc_failsoft_out_of_scope(self) -> None:
        """SKILL.md must reference the ENOSPC fail-soft and mark it out of scope."""
        assert "out of scope" in self._skill_lower, (
            "skills/merge-queue/SKILL.md must reference the merge-verify ENOSPC "
            "fail-soft and mark it as out of scope"
        )
