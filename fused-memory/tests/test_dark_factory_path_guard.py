"""Tests for the dark-factory path-scope guard.

Structured as three test classes, one per public artifact:
  - TestFindDarkFactoryPaths  (step-1/2)
  - TestPathGuardVerdict       (step-3/4)
  - TestCheckCandidate          (step-5/6)
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# TestFindDarkFactoryPaths  (step-1)
# ---------------------------------------------------------------------------


class TestFindDarkFactoryPaths:
    """Unit tests for find_dark_factory_paths()."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from fused_memory.middleware.dark_factory_path_guard import find_dark_factory_paths  # noqa: PLC0415
        self.find = find_dark_factory_paths

    def test_empty_text_returns_empty_list(self):
        assert self.find('') == []

    def test_text_with_no_matches_returns_empty_list(self):
        assert self.find('nothing to see here, just some/other/path and foo.py') == []

    def test_single_match_orchestrator(self):
        result = self.find('Investigate orchestrator/harness.py deadlock')
        assert result == ['orchestrator/']

    def test_fused_memory_prefix_matches(self):
        result = self.find('fused-memory/src/fused_memory/middleware/task_curator.py')
        assert result == ['fused-memory/']

    def test_fused_memory_underscore_prefix_matches(self):
        result = self.find('fused_memory/src/something.py')
        assert result == ['fused_memory/']

    def test_multiple_distinct_prefixes_deduped_and_ordered(self):
        text = 'orchestrator/x and fused-memory/y and orchestrator/z again'
        result = self.find(text)
        # orchestrator/ appears first in text, fused-memory/ second; deduped
        assert result == ['orchestrator/', 'fused-memory/']

    def test_word_boundary_foo_orchestrator_does_not_match(self):
        """foo-orchestrator/ must NOT match the `orchestrator/` prefix."""
        # The regex uses (?:^|[^A-Za-z0-9_-]) as the boundary character class,
        # so a leading hyphen-letter chain prevents the match.
        result = self.find('foo-orchestrator/something.py')
        assert result == []

    def test_non_prefix_substring_does_not_match(self):
        """nonorchestrator/ must not match orchestrator/."""
        result = self.find('nonorchestrator/something.py')
        assert result == []

    def test_prefix_at_start_of_string_matches(self):
        """orchestrator/ at position 0 (no leading character) must match."""
        result = self.find('orchestrator/harness.py')
        assert result == ['orchestrator/']

    def test_prefix_after_newline_matches(self):
        text = 'description:\nreview fused-memory/middleware.py for issues'
        result = self.find(text)
        assert 'fused-memory/' in result

    def test_prefix_after_space_matches(self):
        text = 'edit the file graphiti/core/something.py'
        result = self.find(text)
        assert result == ['graphiti/']

    def test_prefix_after_comma_matches(self):
        text = 'files: a.py, orchestrator/harness.py'
        result = self.find(text)
        assert result == ['orchestrator/']

    def test_all_known_prefixes_are_matched(self):
        """Every canonical dark-factory prefix is detected when present."""
        from fused_memory.middleware.dark_factory_path_guard import DARK_FACTORY_PATH_PREFIXES  # noqa: PLC0415
        for prefix in DARK_FACTORY_PATH_PREFIXES:
            text = f'look at {prefix}some_file.py'
            result = self.find(text)
            assert prefix in result, f'prefix {prefix!r} was not matched in {text!r}'

    def test_order_follows_first_occurrence(self):
        """Prefixes are returned in the order of their first match in the text."""
        text = 'see escalation/foo.py and also mem0/bar.py and escalation/baz.py'
        result = self.find(text)
        assert result[0] == 'escalation/'
        assert result[1] == 'mem0/'
        assert len(result) == 2  # no duplicates


# ---------------------------------------------------------------------------
# TestPathGuardVerdict  (step-3)
# ---------------------------------------------------------------------------


class TestPathGuardVerdict:
    """Unit tests for PathGuardVerdict."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from fused_memory.middleware.dark_factory_path_guard import PathGuardVerdict  # noqa: PLC0415
        self.Verdict = PathGuardVerdict

    def test_ok_outcome_is_not_rejection(self):
        v = self.Verdict(outcome='ok')
        assert v.is_rejection is False

    def test_ok_to_error_dict_is_empty(self):
        v = self.Verdict(outcome='ok')
        assert v.to_error_dict() == {}

    def test_rejection_is_rejection(self):
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/',),
        )
        assert v.is_rejection is True

    def test_rejection_to_error_dict_has_required_keys(self):
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/', 'fused-memory/'),
        )
        d = v.to_error_dict()
        assert 'error' in d
        assert 'error_type' in d
        assert 'project_id' in d
        assert 'matched_paths' in d

    def test_rejection_to_error_dict_error_type(self):
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/',),
        )
        assert v.to_error_dict()['error_type'] == 'DarkFactoryPathScopeViolation'

    def test_rejection_to_error_dict_project_id(self):
        v = self.Verdict(
            outcome='rejection',
            project_id='some_project',
            matched_paths=('mem0/',),
        )
        assert v.to_error_dict()['project_id'] == 'some_project'

    def test_rejection_to_error_dict_matched_paths_is_list(self):
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/', 'graphiti/'),
        )
        d = v.to_error_dict()
        assert isinstance(d['matched_paths'], list)
        assert 'orchestrator/' in d['matched_paths']
        assert 'graphiti/' in d['matched_paths']

    def test_rejection_error_message_mentions_matched_paths(self):
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/',),
        )
        error_msg = v.to_error_dict()['error']
        assert 'orchestrator/' in error_msg

    def test_rejection_error_message_mentions_project_id(self):
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/',),
        )
        error_msg = v.to_error_dict()['error']
        assert 'reify' in error_msg

    def test_rejection_error_message_mentions_dark_factory(self):
        """The error message should tell the caller to file under dark_factory."""
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/',),
        )
        error_msg = v.to_error_dict()['error']
        assert 'dark_factory' in error_msg

    def test_verdict_is_frozen(self):
        """Attempting to set an attribute on a frozen dataclass raises."""
        v = self.Verdict(outcome='ok')
        with pytest.raises((AttributeError, TypeError)):
            v.outcome = 'rejection'  # type: ignore[misc]

    def test_verdict_is_hashable(self):
        """A frozen dataclass with hashable fields must be hashable."""
        v = self.Verdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('orchestrator/',),
        )
        # Should not raise
        _ = hash(v)


# ---------------------------------------------------------------------------
# TestCheckCandidate  (step-5)
# ---------------------------------------------------------------------------


class TestCheckCandidate:
    """Unit tests for check_candidate_for_dark_factory_paths()."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from fused_memory.middleware.dark_factory_path_guard import check_candidate_for_dark_factory_paths  # noqa: PLC0415
        from fused_memory.middleware.task_curator import CandidateTask  # noqa: PLC0415
        self.check = check_candidate_for_dark_factory_paths
        self.CandidateTask = CandidateTask

    def _make_candidate(
        self,
        title: str = 'A task',
        description: str = '',
        details: str = '',
        files_to_modify: list[str] | None = None,
    ):
        return self.CandidateTask(
            title=title,
            description=description,
            details=details,
            files_to_modify=files_to_modify or [],
        )

    def test_dark_factory_project_id_is_always_ok(self):
        """Even if the task has dark-factory paths, filing under dark_factory is fine."""
        candidate = self._make_candidate(
            title='Fix orchestrator/harness.py',
            description='See fused-memory/src/ for context',
            details='Also check graphiti/something.py',
            files_to_modify=['orchestrator/harness.py', 'fused-memory/src/x.py'],
        )
        result = self.check(candidate, 'dark_factory')
        assert result.outcome == 'ok'
        assert not result.is_rejection

    def test_clean_fields_non_dark_factory_project_is_ok(self):
        candidate = self._make_candidate(
            title='Refactor routing logic',
            description='Generic refactor of foo/bar.py',
            details='Update tests in tests/unit/',
            files_to_modify=['src/routes.py'],
        )
        result = self.check(candidate, 'reify')
        assert result.outcome == 'ok'
        assert not result.is_rejection

    def test_match_in_title_only_is_rejection(self):
        candidate = self._make_candidate(
            title='Investigate orchestrator/harness.py deadlock',
        )
        result = self.check(candidate, 'reify')
        assert result.is_rejection
        assert 'orchestrator/' in result.matched_paths

    def test_match_in_description_only_is_rejection(self):
        candidate = self._make_candidate(
            description='See fused-memory/src/fused_memory/middleware/task_curator.py',
        )
        result = self.check(candidate, 'reify')
        assert result.is_rejection
        assert 'fused-memory/' in result.matched_paths

    def test_match_in_details_only_is_rejection(self):
        candidate = self._make_candidate(
            details='Fix bug in graphiti/core/utils.py',
        )
        result = self.check(candidate, 'reify')
        assert result.is_rejection
        assert 'graphiti/' in result.matched_paths

    def test_match_in_files_to_modify_only_is_rejection(self):
        candidate = self._make_candidate(
            files_to_modify=['fused_memory/middleware/task_interceptor.py'],
        )
        result = self.check(candidate, 'reify')
        assert result.is_rejection
        assert 'fused_memory/' in result.matched_paths

    def test_matched_paths_is_deduped_and_ordered(self):
        """orchestrator/ mentioned in description and also in files_to_modify → appears once."""
        candidate = self._make_candidate(
            description='See orchestrator/x.py',
            files_to_modify=['orchestrator/y.py'],
        )
        result = self.check(candidate, 'reify')
        assert result.is_rejection
        assert result.matched_paths == ('orchestrator/',)

    def test_multiple_matches_are_deduped(self):
        """Two different prefixes → both in matched_paths, no duplicates."""
        candidate = self._make_candidate(
            description='orchestrator/x.py and fused-memory/y.py both need changes',
        )
        result = self.check(candidate, 'reify')
        assert result.is_rejection
        assert 'orchestrator/' in result.matched_paths
        assert 'fused-memory/' in result.matched_paths
        # No duplicate entries
        assert len(result.matched_paths) == len(set(result.matched_paths))

    def test_rejection_project_id_is_the_input_project_id(self):
        candidate = self._make_candidate(title='Fix mem0/storage.py')
        result = self.check(candidate, 'some_other_project')
        assert result.is_rejection
        assert result.project_id == 'some_other_project'
