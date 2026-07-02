"""Tests for the recon code-fix premise-verification guard.

Mirrors tests/test_task_curator.py's TestCancelledPremiseBlocklistLoader /
TestCancelledPremiseBlocklistMatcher, extended with source_assertions and
live source/test re-verification (verify_premise_refuted).
"""

from __future__ import annotations

from fused_memory.middleware.task_curator import CandidateTask

# ──────────────────────────────────────────────────────────────────────────────
# task-1972 step-03 RED: TestLoadPremiseRegistry
# ──────────────────────────────────────────────────────────────────────────────


class TestLoadPremiseRegistry:
    """Tests for load_premise_registry() and PremiseEntry/SourceAssertion from
    fused_memory.middleware.recon_code_fix_premise_guard.
    """

    def test_load_valid_yaml_returns_entries(self, tmp_path):
        """(a) Valid YAML returns a list of PremiseEntry dataclasses with source_assertions."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            PremiseEntry,
            SourceAssertion,
            load_premise_registry,
        )

        yaml_content = """
- name: test_entry
  reason: A test reason for the premise being refuted
  title_substrings:
    - "entity-summary rebuild"
  description_substrings:
    - "invalid_at filter"
  source_assertions:
    - file: src/fused_memory/services/memory_service.py
      must_contain:
        - "invalid_at"
      must_not_contain: []
"""
        p = tmp_path / "registry.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        entries = load_premise_registry(p)

        assert len(entries) == 1
        e = entries[0]
        assert isinstance(e, PremiseEntry)
        assert e.name == "test_entry"
        assert e.reason == "A test reason for the premise being refuted"
        assert e.title_substrings == ["entity-summary rebuild"]
        assert e.description_substrings == ["invalid_at filter"]
        assert len(e.source_assertions) == 1
        sa = e.source_assertions[0]
        assert isinstance(sa, SourceAssertion)
        assert sa.file == "src/fused_memory/services/memory_service.py"
        assert sa.must_contain == ["invalid_at"]
        assert sa.must_not_contain == []

    def test_load_missing_path_returns_empty_and_warns(self, tmp_path, caplog):
        """(b) Missing file returns [] and emits exactly one WARNING."""
        from fused_memory.middleware.recon_code_fix_premise_guard import load_premise_registry

        missing = tmp_path / "does_not_exist.yaml"
        with caplog.at_level("WARNING"):
            entries = load_premise_registry(missing)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_load_malformed_yaml_returns_empty_and_warns(self, tmp_path, caplog):
        """(c) Malformed YAML returns [] and emits exactly one WARNING."""
        from fused_memory.middleware.recon_code_fix_premise_guard import load_premise_registry

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unclosed bracket\n: invalid\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            entries = load_premise_registry(bad_yaml)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_load_entry_missing_required_field_skips_and_warns(self, tmp_path, caplog):
        """(d) Entry missing title_substrings is skipped with WARNING; well-formed entries returned."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            PremiseEntry,
            load_premise_registry,
        )

        yaml_content = """
- name: good_entry
  reason: A good entry
  title_substrings:
    - "pattern"
  description_substrings:
    - "description match"
  source_assertions:
    - file: some/file.py
      must_contain: ["token"]
- name: bad_entry
  reason: Missing title_substrings
  description_substrings:
    - "description match"
  source_assertions:
    - file: some/other.py
      must_contain: ["token"]
"""
        p = tmp_path / "mixed.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        with caplog.at_level("WARNING"):
            entries = load_premise_registry(p)

        assert len(entries) == 1
        assert entries[0].name == "good_entry"
        assert isinstance(entries[0], PremiseEntry)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1

    def test_load_entry_with_non_list_source_assertions_skips_and_warns(self, tmp_path, caplog):
        """(d2) Entry whose source_assertions is not a list is skipped with WARNING."""
        from fused_memory.middleware.recon_code_fix_premise_guard import load_premise_registry

        yaml_content = """
- name: good_entry
  reason: A good entry
  title_substrings:
    - "pattern"
  description_substrings:
    - "description match"
  source_assertions:
    - file: some/file.py
      must_contain: ["token"]
- name: bad_entry
  reason: source_assertions is a dict, not a list
  title_substrings:
    - "other pattern"
  description_substrings:
    - "other match"
  source_assertions:
    file: some/other.py
"""
        p = tmp_path / "mixed2.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        with caplog.at_level("WARNING"):
            entries = load_premise_registry(p)

        assert len(entries) == 1
        assert entries[0].name == "good_entry"

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1

    def test_load_none_path_returns_empty_no_warning(self, caplog):
        """(bonus) path=None returns [] without any warnings."""
        from fused_memory.middleware.recon_code_fix_premise_guard import load_premise_registry

        with caplog.at_level("WARNING"):
            entries = load_premise_registry(None)

        assert entries == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 0

    def test_load_entry_missing_must_contain_defaults_to_empty_list(self, tmp_path):
        """must_contain/must_not_contain default to [] when omitted from a source_assertion."""
        from fused_memory.middleware.recon_code_fix_premise_guard import load_premise_registry

        yaml_content = """
- name: test_entry
  reason: test reason
  title_substrings:
    - "pattern"
  description_substrings:
    - "match"
  source_assertions:
    - file: some/file.py
"""
        p = tmp_path / "defaults.yaml"
        p.write_text(yaml_content, encoding="utf-8")

        entries = load_premise_registry(p)

        assert len(entries) == 1
        sa = entries[0].source_assertions[0]
        assert sa.must_contain == []
        assert sa.must_not_contain == []


# ──────────────────────────────────────────────────────────────────────────────
# task-1972 step-05 RED: TestMatchCandidate
# ──────────────────────────────────────────────────────────────────────────────


class TestMatchCandidate:
    """Tests for match_candidate() from fused_memory.middleware.recon_code_fix_premise_guard.

    Same matching shape as cancelled_premise_blocklist.match_candidate — see
    TestCancelledPremiseBlocklistMatcher in test_task_curator.py.
    """

    def _make_entry(
        self,
        name: str = "test_entry",
        title_subs: list | None = None,
        desc_subs: list | None = None,
        reason: str = "test reason",
        source_assertions: list | None = None,
    ):
        from fused_memory.middleware.recon_code_fix_premise_guard import PremiseEntry
        return PremiseEntry(
            name=name,
            reason=reason,
            title_substrings=title_subs or ["entity-summary rebuild", "invalid_at"],
            description_substrings=desc_subs or ["missing invalid_at filter"],
            source_assertions=source_assertions or [],
        )

    def _make_candidate(self, title: str, description: str = "", details: str = ""):
        return CandidateTask(title=title, description=description, details=details)

    def test_match_all_title_subs_and_one_desc_sub(self):
        """(a) Matches when ALL title_substrings AND at least one description_substring hit."""
        from fused_memory.middleware.recon_code_fix_premise_guard import match_candidate

        entry = self._make_entry(
            title_subs=["entity-summary rebuild", "invalid_at"],
            desc_subs=["missing invalid_at filter", "stale entities"],
        )
        candidate = self._make_candidate(
            title="Fix entity-summary rebuild missing invalid_at filter",
            description="Rebuild does not check missing invalid_at filter before writing.",
        )
        result = match_candidate(candidate, [entry])
        assert result is entry

    def test_no_match_when_title_substring_missing(self):
        """(b) Returns None when any title_substring is absent."""
        from fused_memory.middleware.recon_code_fix_premise_guard import match_candidate

        entry = self._make_entry(
            title_subs=["entity-summary rebuild", "invalid_at"],
            desc_subs=["missing invalid_at filter"],
        )
        # Title has 'invalid_at' but NOT 'entity-summary rebuild'
        candidate = self._make_candidate(
            title="Fix invalid_at handling in memory service",
            description="Includes missing invalid_at filter check.",
        )
        result = match_candidate(candidate, [entry])
        assert result is None

    def test_no_match_when_no_description_substring_hits(self):
        """(c) Returns None when none of description_substrings appear."""
        from fused_memory.middleware.recon_code_fix_premise_guard import match_candidate

        entry = self._make_entry(
            title_subs=["entity-summary rebuild", "invalid_at"],
            desc_subs=["missing invalid_at filter", "stale entities"],
        )
        # Title matches but description has neither substring
        candidate = self._make_candidate(
            title="Fix entity-summary rebuild missing invalid_at filter",
            description="Completely unrelated description about something else.",
        )
        result = match_candidate(candidate, [entry])
        assert result is None

    def test_no_match_when_entries_empty(self):
        """(d) Returns None when entries list is empty."""
        from fused_memory.middleware.recon_code_fix_premise_guard import match_candidate

        candidate = self._make_candidate(
            title="Fix entity-summary rebuild missing invalid_at filter",
            description="Rebuild does not check missing invalid_at filter.",
        )
        result = match_candidate(candidate, [])
        assert result is None

    def test_returns_first_match_in_list_order(self):
        """(e) When multiple entries match, returns the first one."""
        from fused_memory.middleware.recon_code_fix_premise_guard import match_candidate

        entry_a = self._make_entry(
            name="entry_a",
            title_subs=["entity-summary rebuild", "invalid_at"],
            desc_subs=["missing invalid_at filter"],
        )
        entry_b = self._make_entry(
            name="entry_b",
            title_subs=["entity-summary rebuild", "invalid_at"],
            desc_subs=["stale entities"],
        )
        # Candidate matches both
        candidate = self._make_candidate(
            title="Fix entity-summary rebuild missing invalid_at filter",
            description="missing invalid_at filter causes stale entities to linger.",
        )
        result = match_candidate(candidate, [entry_a, entry_b])
        assert result is entry_a

    def test_case_insensitive_matching(self):
        """Match is case-insensitive for both title and description."""
        from fused_memory.middleware.recon_code_fix_premise_guard import match_candidate

        entry = self._make_entry(
            title_subs=["ENTITY-SUMMARY REBUILD", "INVALID_AT"],
            desc_subs=["MISSING INVALID_AT FILTER"],
        )
        candidate = self._make_candidate(
            title="fix entity-summary rebuild missing invalid_at filter",
            description="missing invalid_at filter in the rebuild path.",
        )
        result = match_candidate(candidate, [entry])
        assert result is entry

    def test_desc_substring_can_appear_in_details(self):
        """Description_substrings are matched against description + details combined."""
        from fused_memory.middleware.recon_code_fix_premise_guard import match_candidate

        entry = self._make_entry(
            title_subs=["entity-summary rebuild", "invalid_at"],
            desc_subs=["stale entities"],
        )
        # Match is in details, not description
        candidate = self._make_candidate(
            title="Fix entity-summary rebuild missing invalid_at filter",
            description="Unrelated description.",
            details="Leaves stale entities visible after tombstoning.",
        )
        result = match_candidate(candidate, [entry])
        assert result is entry


# ──────────────────────────────────────────────────────────────────────────────
# task-1972 step-07 RED: TestVerifyPremiseRefuted
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifyPremiseRefuted:
    """Tests for verify_premise_refuted() from
    fused_memory.middleware.recon_code_fix_premise_guard.

    source_root is a tmp_path standing in for the fused-memory server cwd;
    each SourceAssertion.file is relative to it.
    """

    def _make_entry(self, source_assertions):
        from fused_memory.middleware.recon_code_fix_premise_guard import PremiseEntry
        return PremiseEntry(
            name="test_entry",
            reason="test reason",
            title_substrings=["irrelevant"],
            description_substrings=["irrelevant"],
            source_assertions=source_assertions,
        )

    def test_true_when_single_assertion_holds(self, tmp_path):
        """(a) True when the only assertion's must_contain/must_not_contain both hold."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            SourceAssertion,
            verify_premise_refuted,
        )

        target = tmp_path / "memory_service.py"
        target.write_text("def rebuild():\n    filter_by(invalid_at=None)\n", encoding="utf-8")

        entry = self._make_entry([
            SourceAssertion(
                file="memory_service.py",
                must_contain=["invalid_at"],
                must_not_contain=["deprecated_marker"],
            )
        ])

        assert verify_premise_refuted(entry, tmp_path) is True

    def test_false_when_must_contain_token_absent(self, tmp_path):
        """(b) False when a must_contain token is absent from the cited file."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            SourceAssertion,
            verify_premise_refuted,
        )

        target = tmp_path / "memory_service.py"
        target.write_text("def rebuild():\n    pass\n", encoding="utf-8")

        entry = self._make_entry([
            SourceAssertion(file="memory_service.py", must_contain=["invalid_at"], must_not_contain=[])
        ])

        assert verify_premise_refuted(entry, tmp_path) is False

    def test_false_when_must_not_contain_token_present(self, tmp_path):
        """(c) False when a must_not_contain token IS present in the cited file."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            SourceAssertion,
            verify_premise_refuted,
        )

        target = tmp_path / "recon_report.py"
        target.write_text("description: str = Field(max_length=500)\n", encoding="utf-8")

        entry = self._make_entry([
            SourceAssertion(file="recon_report.py", must_contain=[], must_not_contain=["max_length"])
        ])

        assert verify_premise_refuted(entry, tmp_path) is False

    def test_false_when_cited_file_missing(self, tmp_path):
        """(d) Fail-open: False when the cited file does not exist under source_root."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            SourceAssertion,
            verify_premise_refuted,
        )

        entry = self._make_entry([
            SourceAssertion(file="does_not_exist.py", must_contain=["token"], must_not_contain=[])
        ])

        assert verify_premise_refuted(entry, tmp_path) is False

    def test_false_when_any_one_of_multiple_assertions_fails(self, tmp_path):
        """(e) AND semantics: one failing assertion among several makes the whole entry False."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            SourceAssertion,
            verify_premise_refuted,
        )

        good = tmp_path / "good.py"
        good.write_text("has_token = True\n", encoding="utf-8")
        bad = tmp_path / "bad.py"
        bad.write_text("nothing relevant here\n", encoding="utf-8")

        entry = self._make_entry([
            SourceAssertion(file="good.py", must_contain=["has_token"], must_not_contain=[]),
            SourceAssertion(file="bad.py", must_contain=["missing_token"], must_not_contain=[]),
        ])

        assert verify_premise_refuted(entry, tmp_path) is False

    def test_true_when_all_of_multiple_assertions_hold(self, tmp_path):
        """(f) True only when ALL assertions across multiple files hold."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            SourceAssertion,
            verify_premise_refuted,
        )

        a = tmp_path / "a.py"
        a.write_text("alpha_token\n", encoding="utf-8")
        b = tmp_path / "b.py"
        b.write_text("beta_token\n", encoding="utf-8")

        entry = self._make_entry([
            SourceAssertion(file="a.py", must_contain=["alpha_token"], must_not_contain=["forbidden"]),
            SourceAssertion(file="b.py", must_contain=["beta_token"], must_not_contain=["forbidden"]),
        ])

        assert verify_premise_refuted(entry, tmp_path) is True

    def test_reads_file_fresh_not_cached(self, tmp_path):
        """(g) Self-correcting: re-reads the file fresh each call, no caching across calls."""
        from fused_memory.middleware.recon_code_fix_premise_guard import (
            SourceAssertion,
            verify_premise_refuted,
        )

        target = tmp_path / "memory_service.py"
        target.write_text("no marker here\n", encoding="utf-8")

        entry = self._make_entry([
            SourceAssertion(file="memory_service.py", must_contain=["invalid_at"], must_not_contain=[])
        ])

        assert verify_premise_refuted(entry, tmp_path) is False

        # Premise becomes refuted once the source is edited to add the marker —
        # proves there is no stale caching between calls.
        target.write_text("invalid_at filter added\n", encoding="utf-8")
        assert verify_premise_refuted(entry, tmp_path) is True

    def test_true_when_no_assertions_at_all(self, tmp_path):
        """(bonus) Vacuously True when source_assertions is an empty list (ALL of none holds)."""
        from fused_memory.middleware.recon_code_fix_premise_guard import verify_premise_refuted

        entry = self._make_entry([])

        assert verify_premise_refuted(entry, tmp_path) is True
