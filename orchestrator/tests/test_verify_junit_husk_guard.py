"""Unit tests for verify._prepare_junit_report_path — the shape-2 junit-husk
guard (task 2922).

The merge-role verify junit writer used to do
``(worktree / '.df-verify-junit').mkdir(parents=True, exist_ok=True)``. A late
merge-role verify writer firing AFTER the worktree was torn down would
re-create the entire ``<worktree>`` path as an empty ~8KB husk that the merge-
worktree ledger audit then flags as an unregistered ``_merge-*`` directory
(shape-2). ``_prepare_junit_report_path`` guards that path: it returns None
WITHOUT creating anything when the worktree no longer exists (or is not a
directory), and only builds the report path — creating just the
``.df-verify-junit`` child, never the worktree ancestor — when the worktree is
a live directory.

The symbol is imported LOCALLY inside each test so a not-yet-implemented
``_prepare_junit_report_path`` fails these tests (RED) without breaking
collection of the rest of the suite.
"""
from __future__ import annotations

from pathlib import Path


def test_existing_worktree_returns_report_path_and_creates_dir(
    tmp_path: Path,
) -> None:
    """Happy path: an existing worktree dir yields a resolved
    ``.df-verify-junit/report.xml`` and actually creates the child dir."""
    from orchestrator.verify import _prepare_junit_report_path

    result = _prepare_junit_report_path(tmp_path, None)

    assert result is not None
    assert result == (tmp_path / '.df-verify-junit' / 'report.xml').resolve()
    assert result.name == 'report.xml'
    assert result.parent.name == '.df-verify-junit'
    # The .df-verify-junit dir was actually created.
    assert (tmp_path / '.df-verify-junit').is_dir()


def test_module_prefix_carries_sanitized_infix(tmp_path: Path) -> None:
    """module_prefix='pkg/sub' yields ``report.pkg_sub.xml`` (matching
    _make_infix semantics: ``/`` and spaces -> ``_``)."""
    from orchestrator.verify import _prepare_junit_report_path

    result = _prepare_junit_report_path(tmp_path, 'pkg/sub')

    assert result is not None
    assert result.name == 'report.pkg_sub.xml'
    assert (tmp_path / '.df-verify-junit').is_dir()


def test_missing_worktree_returns_none_without_creating_husk(
    tmp_path: Path,
) -> None:
    """HUSK GUARD (shape-2 regression): when the worktree does NOT exist
    (post-teardown late write), return None and DO NOT recreate the worktree
    path — no ``.df-verify-junit`` husk may materialize."""
    from orchestrator.verify import _prepare_junit_report_path

    worktree = tmp_path / 'torn-down-worktree'
    assert not worktree.exists()

    result = _prepare_junit_report_path(worktree, None)

    assert result is None
    assert not worktree.exists(), (
        'the torn-down worktree path must not be recreated as a husk'
    )


def test_worktree_is_a_file_returns_none(tmp_path: Path) -> None:
    """A worktree path that is a file (not a directory) returns None and the
    file is left untouched."""
    from orchestrator.verify import _prepare_junit_report_path

    worktree = tmp_path / 'not-a-dir'
    worktree.write_text('i am a file\n')

    result = _prepare_junit_report_path(worktree, None)

    assert result is None
    assert worktree.is_file(), 'the file must be left untouched'
    assert not (worktree / '.df-verify-junit').exists()
