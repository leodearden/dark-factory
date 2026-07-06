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

from orchestrator.module_charter import derive_modules, sanitize_files_for_persist


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
