"""Tests for orchestrator.module_charter — the sole orchestrator-side
composition of Lock-charter Contract 1 (metadata.files is ALWAYS file-level;
coarsen at READ only).

derive_modules and sanitize_files_for_persist replace 4+ per-site
re-implementations of the strip_directory_locks -> files_to_modules(depth)
pipeline (see scheduler.py._get_modules / handle_blast_radius_expansion and
harness.py._tag_task_modules for the call sites this module consolidates).
"""

from __future__ import annotations

import logging

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.module_charter import derive_modules, sanitize_files_for_persist
from orchestrator.scheduler import Scheduler


class TestDeriveModules:
    """derive_modules(files, depth, *, task_id='') -> list[str]."""

    def test_mixed_file_and_dir_strips_dir_and_coarsens(self):
        """Directory entry stripped; surviving file coarsened to its module."""
        files = ['src/config/schema.py', 'crates/reify-eval/src']
        assert derive_modules(files, depth=2) == ['src/config']

    def test_all_directory_input_returns_empty(self):
        """No file-level survivors after the strip -> no derived modules."""
        files = ['crates/reify-eval/src', 'crates/reify-eval/tests']
        assert derive_modules(files, depth=2) == []

    def test_multiple_files_in_one_module_dedup_to_single_sorted_entry(self):
        """Two files under the same depth-N prefix collapse to one entry."""
        files = [
            'crates/reify-compiler/src/foo.rs',
            'crates/reify-compiler/src/bar.rs',
        ]
        assert derive_modules(files, depth=3) == ['crates/reify-compiler/src']

    def test_logs_info_naming_stripped_directories_when_task_id_set(self, caplog):
        """Rejected directory charter entries are named in an INFO diagnostic."""
        files = ['src/config/schema.py', 'crates/reify-eval/src']
        with caplog.at_level(logging.INFO, logger='orchestrator.module_charter'):
            derive_modules(files, depth=2, task_id='42')
        assert any(
            record.levelno == logging.INFO
            and 'crates/reify-eval/src' in record.getMessage()
            for record in caplog.records
        ), (
            f'Expected an INFO log naming the stripped directory entry; got '
            f'{[record.getMessage() for record in caplog.records]!r}'
        )


class TestSanitizeFilesForPersist:
    """sanitize_files_for_persist(files) -> list[str]."""

    def test_strips_directory_shaped_entries(self):
        files = ['crates/reify-eval/src', 'a/b.py', 'crates/reify-eval/tests', 'c.rs']
        assert sanitize_files_for_persist(files) == ['a/b.py', 'c.rs']

    def test_drops_non_string_and_blank_and_whitespace_tokens(self):
        files = [None, 42, '', '   ', 'src/foo.py']
        assert sanitize_files_for_persist(files) == ['src/foo.py']

    def test_preserves_order_of_surviving_file_level_entries(self):
        files = ['z/mod/foo.py', 'crates/reify-eval/src', 'a/mod/bar.py']
        assert sanitize_files_for_persist(files) == ['z/mod/foo.py', 'a/mod/bar.py']


# ---------------------------------------------------------------------------
# Extension-less tracked files (dark_factory #3248) — Face 2 of the
# classification defect: the SILENT under-locking one.
#
# The concrete hazard, in one line: two tasks editing ``hooks/project-checks``
# concurrently can both hold NO lock on it, because a charter that strips to
# empty degrades to a per-task ``task-<id>`` synthetic lock that conflicts with
# nothing.
#
# Face 1 (loud) is the submit-time LockCharterViolation raised by the γ guard,
# pinned in fused-memory/tests/test_lock_charter_guard.py.  Face 2 is this one,
# and it is worse precisely because it is quiet: derive_modules strips the entry
# with nothing louder than an INFO diagnostic, files_to_modules then returns
# [], and Scheduler._get_modules falls through to the task-<id> fallback.
# Nothing in that chain reports a problem.
# ---------------------------------------------------------------------------


class TestDeriveModulesRetainsExtensionlessFiles:
    """Extension-less tracked FILES must survive the α strip.

    Before #3248 the α strip's criterion was "no recognised file EXTENSION",
    which silently discarded real tracked files such as ``hooks/project-checks``,
    ``hooks/pre-commit``, ``LICENSE`` and ``Dockerfile``.
    """

    def test_extensionless_tracked_file_is_retained(self):
        """derive_modules must NOT strip hooks/project-checks to an empty charter.

        On base this returns ``[]``: ``strip_directory_locks`` drops the entry
        and only an INFO log is emitted, so the task ends up holding a
        ``task-<id>`` lock that conflicts with no other task's charter.
        """
        result = derive_modules(['hooks/project-checks'], depth=4)

        assert result, (
            'derive_modules stripped hooks/project-checks to an empty charter. '
            'It is a real tracked FILE, so it must derive a real lock key — '
            'otherwise the caller falls through to a task-<id> synthetic lock '
            'and two tasks editing this file hold no lock against each other.'
        )
        assert any('hooks' in key for key in result), (
            f'expected a lock key derived from hooks/project-checks, got {result!r}'
        )

    @pytest.mark.parametrize(
        'path',
        [
            'hooks/project-checks',
            'hooks/pre-commit',
            'hooks/pre-merge-commit',
            'LICENSE',
            'fused-memory/docker/Dockerfile',
        ],
    )
    def test_every_extensionless_charter_derives_a_lock(self, path: str):
        """Each real tracked extension-less file derives a non-empty charter."""
        assert derive_modules([path], depth=4), (
            f'{path!r} is a real tracked file but derived an empty lock charter'
        )

    def test_mixed_charter_keeps_both_file_kinds(self):
        """A charter mixing extension-less and dotted files retains both.

        This is the shape the originating incident actually had: a task
        declaring ``hooks/project-checks`` alongside an ordinary ``.py`` file.
        """
        result = derive_modules(
            ['hooks/project-checks', 'orchestrator/src/orchestrator/scheduler.py'],
            depth=4,
        )
        assert any('hooks' in key for key in result), (
            f'the extension-less entry was dropped from the charter: {result!r}'
        )
        assert any('orchestrator' in key for key in result), (
            f'the dotted entry was dropped from the charter: {result!r}'
        )


class TestGetModulesDoesNotFallThroughToSyntheticLock:
    """A charter of only extension-less files must not reach the task-<id> fallback.

    ``Scheduler._get_modules`` derives modules from ``metadata.files`` and, when
    the derived list is falsy, returns ``[f'task-{task_id}']``.  That synthetic
    lock is per-task and therefore conflicts with nothing — it is the mechanism
    by which the strip turns into silent under-locking rather than a visible
    error.
    """

    @pytest.fixture
    def scheduler(self) -> Scheduler:
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(config)
        scheduler.finish_startup()
        return scheduler

    def test_extensionless_only_charter_does_not_get_synthetic_lock(
        self, scheduler: Scheduler
    ):
        task = {
            'id': '3248',
            'metadata': {'files': ['hooks/project-checks']},
        }

        result = scheduler._get_modules(task)

        assert result != ['task-3248'], (
            'a task declaring only hooks/project-checks fell through to the '
            'task-3248 synthetic lock, so it holds NO lock on the file it edits'
        )
        assert 'task-3248' not in result, (
            f'synthetic fallback lock leaked into the derived charter: {result!r}'
        )
        assert result, f'expected a real derived lock charter, got {result!r}'

    def test_two_tasks_on_the_same_extensionless_file_conflict(
        self, scheduler: Scheduler
    ):
        """The point of the fix, stated as the property that was violated.

        Two tasks whose only declared file is the SAME extension-less path must
        derive overlapping charters.  Under the synthetic fallback they derived
        ``['task-A']`` and ``['task-B']`` — disjoint — so the scheduler would
        happily run both against the same file at once.
        """
        task_a = {'id': 'A', 'metadata': {'files': ['hooks/project-checks']}}
        task_b = {'id': 'B', 'metadata': {'files': ['hooks/project-checks']}}

        modules_a = scheduler._get_modules(task_a)
        modules_b = scheduler._get_modules(task_b)

        assert set(modules_a) & set(modules_b), (
            f'two tasks editing hooks/project-checks derived DISJOINT charters '
            f'({modules_a!r} vs {modules_b!r}), so neither blocks the other'
        )


class TestSanitizeFilesForPersistKeepsExtensionlessFiles:
    """The WRITE-path half of the same defect.

    Every ``metadata.files`` write path calls ``sanitize_files_for_persist``.
    While extension-less files classified as directories, this erased them from
    ``metadata.files`` on EVERY persist — so even a correctly-declared charter
    decayed to empty behind the task's back.
    """

    def test_extensionless_entry_is_preserved(self):
        assert sanitize_files_for_persist(['hooks/project-checks']) == [
            'hooks/project-checks'
        ], (
            'sanitize_files_for_persist erased a real tracked file from '
            'metadata.files; the charter would decay to empty on next persist'
        )

    def test_mixed_list_preserves_both(self):
        result = sanitize_files_for_persist(
            ['hooks/pre-commit', 'shared/src/shared/locking.py']
        )
        assert result == ['hooks/pre-commit', 'shared/src/shared/locking.py']


class TestAlphaStripStillRejectsRealDirectories:
    """Regression pin: the fix must not be over-broad.

    These assertions ALREADY PASS on base and must keep passing.  They are what
    distinguishes "recognise extension-less FILES" from "stop stripping
    directories", which would reintroduce the subtree-wide prefix locks of
    reify-3468 that the α strip exists to prevent.
    """

    @pytest.mark.parametrize('path', ['hooks', 'crates/reify-eval/src', 'backend'])
    def test_real_directories_still_strip(self, path: str):
        assert derive_modules([path], depth=4) == [], (
            f'{path!r} is a real DIRECTORY and must still strip to an empty '
            f'charter — retaining it would derive a subtree-wide prefix lock'
        )

    def test_all_directory_charter_still_strips_to_empty(self):
        assert derive_modules(['hooks', 'crates/reify-eval/src'], depth=4) == []

    @pytest.mark.parametrize('path', ['graphiti', 'mem0'])
    def test_gitlink_submodule_roots_stay_directories(self, path: str):
        """Submodule mount points (mode 160000) stay directories BY DESIGN.

        They are extension-less and would otherwise be candidates for the
        allowlist.  Admitting them would let a task declare an entire vendored
        submodule as its lock charter — strictly worse than the bug being fixed.
        """
        assert derive_modules([path], depth=4) == [], (
            f'{path!r} is a submodule mount point and must stay a DIRECTORY; '
            f'admitting it would make a whole vendored submodule declarable'
        )
