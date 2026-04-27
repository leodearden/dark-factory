"""Tests for the multi-project path-scope guard.

Covers behaviour beyond what the back-compat shim
``test_dark_factory_path_guard.py`` exercises: per-project prefix
registries, suggested_project derivation, multi-project mismatches,
empty-registry short-circuit, and the prompt-only ``check_text_for_scope``
branch.
"""

from __future__ import annotations

from pathlib import Path

from fused_memory.middleware.path_scope_guard import (
    PathGuardVerdict,
    check_candidate_for_scope,
    check_text_for_scope,
    find_paths,
)
from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry
from fused_memory.middleware.task_curator import CandidateTask


def _mkproj(parent: Path, name: str, dirs: list[str]) -> Path:
    root = parent / name
    root.mkdir()
    for d in dirs:
        (root / d).mkdir()
    return root


def _two_project_registry(tmp_path: Path) -> ProjectPrefixRegistry:
    """Reify (crates/, gui/) + dark-factory (fused-memory/, orchestrator/)."""
    a = _mkproj(tmp_path, 'reify', ['crates', 'gui'])
    b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator'])
    return ProjectPrefixRegistry.from_roots([str(a), str(b)])


def _candidate(
    title: str = '',
    description: str = '',
    details: str = '',
    files_to_modify: list[str] | None = None,
) -> CandidateTask:
    """Build a real CandidateTask with only the fields the guard reads."""
    return CandidateTask(
        title=title,
        description=description,
        details=details,
        files_to_modify=files_to_modify or [],
        priority='medium',
    )


# ---------------------------------------------------------------------------
# find_paths — generalised regex
# ---------------------------------------------------------------------------


class TestFindPaths:
    def test_empty_text_returns_empty(self):
        assert find_paths('', ('foo/',)) == []

    def test_empty_prefixes_returns_empty(self):
        assert find_paths('lots of text', ()) == []

    def test_single_match(self):
        assert find_paths('see crates/x.rs', ('crates/',)) == ['crates/']

    def test_dedup_and_order(self):
        result = find_paths('crates/a then gui/b then crates/c', ('crates/', 'gui/'))
        assert result == ['crates/', 'gui/']

    def test_word_boundary_no_match_on_suffix(self):
        # "supercrates/" must not match "crates/"
        assert find_paths('supercrates/x.rs', ('crates/',)) == []


# ---------------------------------------------------------------------------
# check_candidate_for_scope
# ---------------------------------------------------------------------------


class TestCheckCandidateForScope:
    def test_empty_registry_always_ok(self):
        # No registry → guard is a no-op.
        empty = ProjectPrefixRegistry.from_roots([])
        c = _candidate(title='Edit fused-memory/X')
        v = check_candidate_for_scope(c, 'reify', empty)
        assert v.outcome == 'ok'
        assert v.matched_paths == ()

    def test_own_project_paths_are_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Refactor crates/foo')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'ok'

    def test_other_project_paths_rejected(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Edit fused-memory/X')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.matched_paths == ('fused-memory/',)
        assert v.suggested_project == 'dark_factory'
        assert v.error_type == 'DarkFactoryPathScopeViolation'

    def test_reverse_direction_caught(self, tmp_path):
        """The reify→dark-factory direction the original guard missed."""
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Update crates/widget.rs')
        v = check_candidate_for_scope(c, 'dark_factory', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'reify'

    def test_multiple_other_projects_no_single_suggestion(self, tmp_path):
        """When mismatches span >1 project, suggested_project is None."""
        # Add a third project so a candidate can mention paths from two others.
        c_root = _mkproj(tmp_path, 'cthird', ['cthird_dir'])
        registry = ProjectPrefixRegistry.from_roots([
            str(_mkproj(tmp_path, 'reify', ['crates'])),
            str(_mkproj(tmp_path, 'dark-factory', ['fused-memory'])),
            str(c_root),
        ])
        c = _candidate(
            title='wat',
            description='Edit fused-memory/X and crates/Y',
        )
        v = check_candidate_for_scope(c, 'cthird', registry)
        assert v.outcome == 'rejection'
        assert set(v.matched_paths) == {'fused-memory/', 'crates/'}
        assert v.suggested_project is None  # ambiguous

    def test_files_to_modify_scanned(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        c = _candidate(
            title='generic title',
            files_to_modify=['fused-memory/src/x.py'],
        )
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'dark_factory'

    def test_unknown_prefix_is_silently_ignored(self, tmp_path):
        """A path prefix that's not in the registry doesn't trigger rejection."""
        registry = _two_project_registry(tmp_path)
        c = _candidate(title='Look at random/path/here.py')
        v = check_candidate_for_scope(c, 'reify', registry)
        assert v.outcome == 'ok'


# ---------------------------------------------------------------------------
# check_text_for_scope (prompt-only path)
# ---------------------------------------------------------------------------


class TestCheckTextForScope:
    def test_none_text_is_ok(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_text_for_scope(None, 'reify', registry)
        assert v.outcome == 'ok'

    def test_text_with_other_project_path_rejected(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        v = check_text_for_scope('please patch fused-memory/foo', 'reify', registry)
        assert v.outcome == 'rejection'
        assert v.suggested_project == 'dark_factory'

    def test_empty_registry_short_circuits(self):
        empty = ProjectPrefixRegistry.from_roots([])
        v = check_text_for_scope('fused-memory/X', 'reify', empty)
        assert v.outcome == 'ok'


# ---------------------------------------------------------------------------
# Verdict.to_error_dict
# ---------------------------------------------------------------------------


class TestVerdictErrorDict:
    def test_ok_verdict_yields_empty(self):
        v = PathGuardVerdict(outcome='ok')
        assert v.to_error_dict() == {}

    def test_rejection_with_suggested_project(self):
        v = PathGuardVerdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        d = v.to_error_dict()
        assert d['error_type'] == 'DarkFactoryPathScopeViolation'
        assert d['project_id'] == 'reify'
        assert d['matched_paths'] == ['fused-memory/']
        assert d['suggested_project'] == 'dark_factory'
        assert 'dark_factory' in d['error']

    def test_rejection_without_suggested_project(self):
        v = PathGuardVerdict(
            outcome='rejection',
            project_id='reify',
            matched_paths=('fused-memory/', 'crates/'),
            suggested_project=None,
        )
        d = v.to_error_dict()
        assert d['suggested_project'] is None
        # Error message should hint at manual routing.
        assert 'manually' in d['error'] or 'manual' in d['error']
