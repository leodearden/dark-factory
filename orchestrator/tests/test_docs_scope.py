"""Tests for orchestrator.agents.docs_scope — the docs-only classifier that
gates the conditional project_root grant in
``orchestrator/src/orchestrator/agents/write_set.py::compute_write_set``.
"""
from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from orchestrator.agents.docs_scope import (
    DOCS_SUFFIXES,
    DOCS_TOP_LEVEL_DIRS,
    docs_only_paths,
    is_docs_path,
)


class TestVocabulary:
    def test_docs_suffixes(self) -> None:
        assert DOCS_SUFFIXES == frozenset({'.md', '.yaml', '.yml'})

    def test_docs_top_level_dirs(self) -> None:
        assert DOCS_TOP_LEVEL_DIRS == frozenset({'docs'})


class TestIsDocsPath:
    @pytest.mark.parametrize('rel', [
        'CLAUDE.md',
        'docs/prds/foo.md',
        'plans/x.capability-manifest.yaml',
        'config/settings.yml',
        'deeply/nested/dir/README.md',
    ])
    def test_docs_suffix_at_any_depth(self, rel: str) -> None:
        assert is_docs_path(PurePosixPath(rel)) is True

    @pytest.mark.parametrize('rel', ['docs/img/diagram.png', 'docs'])
    def test_top_level_docs_dir_without_docs_suffix(self, rel: str) -> None:
        assert is_docs_path(PurePosixPath(rel)) is True

    def test_nested_docs_component_is_not_docs(self) -> None:
        assert is_docs_path(PurePosixPath('orchestrator/docs/x.py')) is False

    @pytest.mark.parametrize('rel', [
        'src/x.py', 'crates/a/src/lib.rs', 'pyproject.toml', 'package.json',
        'scripts/run.sh',
    ])
    def test_non_docs(self, rel: str) -> None:
        assert is_docs_path(PurePosixPath(rel)) is False


class TestDocsOnlyPaths:
    def test_all_docs_returns_normalized_paths_in_order(self) -> None:
        assert docs_only_paths(['docs/prds/foo.md', 'CLAUDE.md', 'docs/img/a.png']) == (
            PurePosixPath('docs/prds/foo.md'),
            PurePosixPath('CLAUDE.md'),
            PurePosixPath('docs/img/a.png'),
        )

    def test_tuple_input_accepted(self) -> None:
        assert docs_only_paths(('CLAUDE.md',)) == (PurePosixPath('CLAUDE.md'),)

    def test_duplicates_removed_after_normalization(self) -> None:
        assert docs_only_paths(['docs/a.md', './docs/a.md', 'CLAUDE.md']) == (
            PurePosixPath('docs/a.md'),
            PurePosixPath('CLAUDE.md'),
        )

    @pytest.mark.parametrize('non_docs', [
        'src/x.py', 'crates/a/src/lib.rs', 'pyproject.toml', 'package.json',
    ])
    def test_any_non_docs_entry_returns_empty(self, non_docs: str) -> None:
        assert docs_only_paths(['docs/a.md', non_docs, 'CLAUDE.md']) == ()

    def test_none(self) -> None:
        assert docs_only_paths(None) == ()

    def test_empty_list(self) -> None:
        assert docs_only_paths([]) == ()

    def test_bare_str(self) -> None:
        assert docs_only_paths('docs/a.md') == ()

    def test_dict(self) -> None:
        assert docs_only_paths({'docs/a.md': True}) == ()

    def test_non_str_entry(self) -> None:
        assert docs_only_paths(['docs/a.md', 3]) == ()

    @pytest.mark.parametrize('blank', ['', '   ', '\t'])
    def test_blank_entry(self, blank: str) -> None:
        assert docs_only_paths(['docs/a.md', blank]) == ()

    def test_absolute_entry(self) -> None:
        assert docs_only_paths(['/home/x/docs/a.md']) == ()

    def test_parent_component_entry(self) -> None:
        assert docs_only_paths(['docs/../src/x.md']) == ()

    def test_pure_no_filesystem_needed(self) -> None:
        missing = 'docs/definitely/not/on/disk/anywhere-4627.md'
        assert docs_only_paths([missing]) == (PurePosixPath(missing),)
