"""Tests for the operational-ask registry — the filing-policy gate that routes
operational live-data/live-mutation asks to a deterministic PURE-GATE instead
of the TDD architect pipeline.

Mirrors tests/test_task_curator.py's TestCancelledPremiseBlocklistLoader /
TestCancelledPremiseBlocklistMatcher and test_recon_code_fix_premise_guard.py's
TestLoadPremiseRegistry / TestMatchCandidate — same shape, new module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from fused_memory.middleware.task_curator import CandidateTask

# ──────────────────────────────────────────────────────────────────────────────
# step-1 RED: TestLoadOperationalRegistry
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadOperationalRegistry:
    """Tests for load_operational_registry() and OperationalAskEntry from
    fused_memory.middleware.operational_ask_registry.
    """

    def test_load_valid_yaml_returns_entries(self, tmp_path):
        """(a) Valid YAML returns a list of OperationalAskEntry dataclasses."""
        from fused_memory.middleware.operational_ask_registry import (
            OperationalAskEntry,
            load_operational_registry,
        )

        yaml_content = """
- name: test_entry
  reason: A test reason for routing to a deterministic gate
  title_substrings:
    - "merge_entities"
    - "live"
  description_substrings:
    - "FalkorDB"
    - "--apply"
"""
        p = tmp_path / "registry.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        entries = load_operational_registry(p)

        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, OperationalAskEntry)
        assert e.name == "test_entry"
        assert e.reason == "A test reason for routing to a deterministic gate"
        assert e.title_substrings == ["merge_entities", "live"]
        assert e.description_substrings == ["FalkorDB", "--apply"]

    def test_load_none_path_returns_empty_no_warning(self, caplog):
        """(b) path=None returns [] without any warnings."""
        from fused_memory.middleware.operational_ask_registry import load_operational_registry

        with caplog.at_level("WARNING"):
            entries = load_operational_registry(None)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 0

    def test_load_missing_path_returns_empty_and_warns(self, tmp_path, caplog):
        """(b) Missing file returns [] and emits exactly one WARNING."""
        from fused_memory.middleware.operational_ask_registry import load_operational_registry

        missing = tmp_path / "does_not_exist.yaml"
        with caplog.at_level("WARNING"):
            entries = load_operational_registry(missing)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    @pytest.mark.skipif(
        sys.platform == "win32" or getattr(os, "getuid", lambda: -1)() == 0,
        reason="chmod not reliable on Windows or when running as root",
    )
    def test_load_unreadable_file_returns_empty_and_warns(self, tmp_path, caplog):
        """(b) Unreadable file returns [] and emits exactly one WARNING."""
        from fused_memory.middleware.operational_ask_registry import load_operational_registry

        locked = tmp_path / "unreadable.yaml"
        locked.write_text("- name: x\n", encoding="utf-8")
        locked.chmod(0o000)
        try:
            with caplog.at_level("WARNING"):
                entries = load_operational_registry(locked)
        finally:
            locked.chmod(0o644)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_load_malformed_yaml_returns_empty_and_warns(self, tmp_path, caplog):
        """(b) Malformed YAML returns [] and emits exactly one WARNING."""
        from fused_memory.middleware.operational_ask_registry import load_operational_registry

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unclosed bracket\n: invalid\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            entries = load_operational_registry(bad_yaml)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_load_non_list_top_level_returns_empty_and_warns(self, tmp_path, caplog):
        """(b) A non-list top-level YAML document returns [] and emits one WARNING."""
        from fused_memory.middleware.operational_ask_registry import load_operational_registry

        p = tmp_path / "not_a_list.yaml"
        p.write_text("just_a_key: just_a_value\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            entries = load_operational_registry(p)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_load_entry_missing_required_field_skips_and_warns(self, tmp_path, caplog):
        """(c) Entry missing title_substrings is skipped with WARNING; well-formed entries returned."""
        from fused_memory.middleware.operational_ask_registry import (
            OperationalAskEntry,
            load_operational_registry,
        )

        yaml_content = """
- name: good_entry
  reason: A good entry
  title_substrings:
    - "pattern"
  description_substrings:
    - "description match"
- name: bad_entry
  reason: Missing title_substrings
  description_substrings:
    - "description match"
"""
        p = tmp_path / "mixed.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        with caplog.at_level("WARNING"):
            entries = load_operational_registry(p)

        assert len(entries) == 1
        assert entries[0].name == "good_entry"
        assert isinstance(entries[0], OperationalAskEntry)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1

    def test_load_entry_with_non_list_substrings_skips_and_warns(self, tmp_path, caplog):
        """(c) Entry with title_substrings/description_substrings not a list is skipped with WARNING."""
        from fused_memory.middleware.operational_ask_registry import load_operational_registry

        yaml_content = """
- name: bad_entry
  reason: title_substrings is a string, not a list
  title_substrings: "not-a-list"
  description_substrings:
    - "description match"
"""
        p = tmp_path / "bad_type.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        with caplog.at_level("WARNING"):
            entries = load_operational_registry(p)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# step-1 RED: TestMatchCandidate
# ──────────────────────────────────────────────────────────────────────────────


class TestMatchCandidate:
    """Tests for match_candidate() from fused_memory.middleware.operational_ask_registry."""

    def _make_entry(
        self,
        name: str = "test_entry",
        title_subs: list | None = None,
        desc_subs: list | None = None,
        reason: str = "test reason",
    ):
        from fused_memory.middleware.operational_ask_registry import OperationalAskEntry
        return OperationalAskEntry(
            name=name,
            reason=reason,
            title_substrings=title_subs or ["merge_entities", "live"],
            description_substrings=desc_subs or ["FalkorDB"],
        )

    def _make_candidate(self, title: str, description: str = "", details: str = ""):
        return CandidateTask(title=title, description=description, details=details)

    def test_match_all_title_subs_and_one_desc_sub(self):
        """(d) Matches when ALL title_substrings AND at least one description_substring hit."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entry = self._make_entry(
            title_subs=["merge_entities", "live"],
            desc_subs=["FalkorDB", "Neo4j"],
        )
        candidate = self._make_candidate(
            title="Run merge_entities against the live graph",
            description="Needs to merge_entities directly on the FalkorDB backend.",
        )
        result = match_candidate(candidate, [entry])
        assert result is entry

    def test_no_match_when_title_substring_missing(self):
        """(d) Returns None when any title_substring is absent."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entry = self._make_entry(
            title_subs=["merge_entities", "live"],
            desc_subs=["FalkorDB"],
        )
        candidate = self._make_candidate(
            title="Run merge_entities in a test fixture",
            description="Uses the FalkorDB test container.",
        )
        result = match_candidate(candidate, [entry])
        assert result is None

    def test_no_match_when_no_description_substring_hits(self):
        """(d) Returns None when none of description_substrings appear."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entry = self._make_entry(
            title_subs=["merge_entities", "live"],
            desc_subs=["FalkorDB", "Neo4j"],
        )
        candidate = self._make_candidate(
            title="Run merge_entities against the live graph",
            description="Completely unrelated description about something else.",
        )
        result = match_candidate(candidate, [entry])
        assert result is None

    def test_no_match_when_entries_empty(self):
        """(d) Returns None when entries list is empty."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        candidate = self._make_candidate(
            title="Run merge_entities against the live graph",
            description="Needs FalkorDB access.",
        )
        result = match_candidate(candidate, [])
        assert result is None

    def test_returns_first_match_in_list_order(self):
        """(d) When multiple entries match, returns the first one."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entry_a = self._make_entry(
            name="entry_a",
            title_subs=["merge_entities", "live"],
            desc_subs=["FalkorDB"],
        )
        entry_b = self._make_entry(
            name="entry_b",
            title_subs=["merge_entities", "live"],
            desc_subs=["Neo4j"],
        )
        candidate = self._make_candidate(
            title="Run merge_entities against the live graph",
            description="FalkorDB and Neo4j both mentioned.",
        )
        result = match_candidate(candidate, [entry_a, entry_b])
        assert result is entry_a

    def test_case_insensitive_matching(self):
        """Match is case-insensitive for both title and description."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entry = self._make_entry(
            title_subs=["MERGE_ENTITIES", "LIVE"],
            desc_subs=["FALKORDB"],
        )
        candidate = self._make_candidate(
            title="run merge_entities against the live graph",
            description="uses the falkordb backend directly.",
        )
        result = match_candidate(candidate, [entry])
        assert result is entry

    def test_desc_substring_can_appear_in_details(self):
        """Description_substrings are matched against description + details combined."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entry = self._make_entry(
            title_subs=["merge_entities", "live"],
            desc_subs=["FalkorDB"],
        )
        candidate = self._make_candidate(
            title="Run merge_entities against the live graph",
            description="Unrelated description.",
            details="Talks directly to FalkorDB in production.",
        )
        result = match_candidate(candidate, [entry])
        assert result is entry


# ──────────────────────────────────────────────────────────────────────────────
# step-5 RED: TestSeedOperationalAskRegistry
# ──────────────────────────────────────────────────────────────────────────────


class TestSeedOperationalAskRegistry:
    """Loads the SHIPPED fused-memory/config/operational_ask_registry.yaml and
    asserts runtime matching behaviour (not YAML wording) against
    representative fixtures for each recurring operational-ask signature.

    Real recurring instances that motivated these entries: tasks 1945/2082
    (prune_recon_cycle_summaries.py, esc-1942-5/esc-1942-10/esc-2082-12/13),
    1946 (rebuild_entity_summaries project-wide, finding f92d7fb3), 1939
    (knowlive namespace purge --apply, esc-1939-4), and 2081 (merge_entities
    against a live FalkorDB graph). See task 2085's analysis for the full
    precedent trail.
    """

    # tests/test_operational_ask_registry.py -> tests/ -> fused-memory/
    SOURCE_ROOT = Path(__file__).resolve().parent.parent
    REGISTRY_PATH = SOURCE_ROOT / "config" / "operational_ask_registry.yaml"

    def _load_entries(self):
        from fused_memory.middleware.operational_ask_registry import load_operational_registry
        return load_operational_registry(self.REGISTRY_PATH)

    def test_shipped_registry_has_at_least_four_well_formed_entries(self):
        """(a) The seed registry has >= 4 entries, each with non-empty substrings."""
        entries = self._load_entries()
        assert len(entries) >= 4, (
            f"Expected >= 4 seed entries, got {len(entries)}: {[e.name for e in entries]}"
        )
        for e in entries:
            assert e.title_substrings, f"entry {e.name!r} has empty title_substrings"
            assert e.description_substrings, f"entry {e.name!r} has empty description_substrings"

    def test_prune_recon_cycle_summaries_matches(self):
        """(b) A prune_recon_cycle_summaries.py --apply ask matches (tasks 1945/2082)."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entries = self._load_entries()
        candidate = CandidateTask(
            title=(
                "Operational: rerun prune_recon_cycle_summaries.py --project-id "
                "know_live (dry-run then --apply)"
            ),
            description=(
                "Collapse the pre-existing untagged Stage 1 cycle-summary pile: "
                "dry-run first, review, then --apply against live Mem0/Qdrant."
            ),
        )
        entry = match_candidate(candidate, entries)
        assert entry is not None, (
            f"Expected a match for prune_recon_cycle_summaries fixture, got None. "
            f"Entries: {[e.name for e in entries]}"
        )
        assert entry.name == "prune_recon_cycle_summaries_apply"

    def test_merge_entities_live_graph_matches(self):
        """(b) A merge_entities-against-live-FalkorDB ask matches (task 2081)."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entries = self._load_entries()
        candidate = CandidateTask(
            title="Merge duplicate Graphiti entity nodes via merge_entities against the live FalkorDB graph",
            description=(
                "Confirm the duplicate via direct query, then call merge_entities "
                "against the production FalkorDB graph to collapse the nodes."
            ),
        )
        entry = match_candidate(candidate, entries)
        assert entry is not None, (
            f"Expected a match for merge_entities fixture, got None. "
            f"Entries: {[e.name for e in entries]}"
        )
        assert entry.name == "merge_entities_live_graph"

    def test_rebuild_entity_summaries_project_wide_matches(self):
        """(b) A project-wide rebuild_entity_summaries ask matches (task 1946)."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entries = self._load_entries()
        candidate = CandidateTask(
            title="Operational: rebuild_entity_summaries project-wide for dark_factory and know_live",
            description="Backfill post-invalidation stale entity summaries across all affected projects.",
        )
        entry = match_candidate(candidate, entries)
        assert entry is not None, (
            f"Expected a match for rebuild_entity_summaries fixture, got None. "
            f"Entries: {[e.name for e in entries]}"
        )
        assert entry.name == "rebuild_entity_summaries_project_wide"

    def test_knowlive_namespace_purge_apply_matches(self):
        """(b) A knowlive namespace purge --apply ask matches (task 1939)."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entries = self._load_entries()
        candidate = CandidateTask(
            title="Operator action: run knowlive namespace purge --apply against production",
            description=(
                "Purge and re-key the legacy knowlive namespace in FalkorDB and Mem0 "
                "via purge_knowlive_namespace.py --apply."
            ),
        )
        entry = match_candidate(candidate, entries)
        assert entry is not None, (
            f"Expected a match for knowlive namespace purge fixture, got None. "
            f"Entries: {[e.name for e in entries]}"
        )
        assert entry.name == "knowlive_namespace_purge_apply"

    def test_unrelated_code_fix_candidate_matches_nothing(self):
        """(c) A clearly-unrelated normal code-fix candidate matches NOTHING."""
        from fused_memory.middleware.operational_ask_registry import match_candidate

        entries = self._load_entries()
        candidate = CandidateTask(
            title="Fix null pointer in task_curator batch dedup when candidates list is empty",
            description=(
                "Add a guard so curate_batch_prepared does not crash on an empty "
                "candidates list; add a regression test."
            ),
        )
        entry = match_candidate(candidate, entries)
        assert entry is None, f"Expected no match for unrelated candidate, got {entry.name!r}"
