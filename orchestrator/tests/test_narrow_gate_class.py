"""Which merge changes qualify for the narrow merge gate: path globs and the classifiers (task 5411)."""
from __future__ import annotations

import pytest

from orchestrator.narrow_gate_class import (
    ChangedPath,
    GateClassification,
    GateMiss,
    MergeGateResolution,
    MissReason,
    classify_docs_change,
    matches_path_glob,
)

DOCS_PATHS = ['*.md', 'docs/**', 'plans/*.md', 'README*', 'skills/**/*.md']


def _modified(path: str, *, old_mode: str = '100644', new_mode: str = '100644') -> ChangedPath:
    return ChangedPath(path=path, status='M', old_mode=old_mode, new_mode=new_mode)


def _added(path: str) -> ChangedPath:
    return ChangedPath(path=path, status='A', old_mode='000000', new_mode='100644')


def _deleted(path: str) -> ChangedPath:
    return ChangedPath(path=path, status='D', old_mode='100644', new_mode='000000')


def _full(reason: MissReason, path: str | None) -> GateClassification:
    return GateClassification(MergeGateResolution.FULL, GateMiss(reason, path))


class TestMatchesPathGlob:

    @pytest.mark.parametrize(('pattern', 'path', 'expected'), [
        ('*.md', 'README.md', True),
        ('*.md', 'docs/a.md', False),
        ('*.md', 'fused-memory/src/fused_memory/reconciliation/prompts/stage1.md', False),
        ('docs/**', 'docs/a.md', True),
        ('docs/**', 'docs/x/y/z.json', True),
        ('docs/**', 'docs/a', True),
        ('plans/*.md', 'plans/p.md', True),
        ('plans/*.md', 'plans/p.evidence/s.md', False),
        ('README*', 'README', True),
        ('README*', 'README.md', True),
        ('skills/**/*.md', 'skills/a/SKILL.md', True),
        ('skills/**/*.md', 'skills/a/b/c.md', True),
        ('skills/**/*.md', 'skills/spawn/spawn-claude.sh', False),
        ('plans/*.capability-manifest.yaml', 'plans/x-prd.capability-manifest.yaml', True),
        ('plans/*.capability-manifest.yaml', 'plans/x.yaml', False),
    ])
    def test_globs(self, pattern: str, path: str, expected: bool) -> None:
        assert matches_path_glob(path, pattern) is expected

    def test_question_mark_stays_within_a_segment(self) -> None:
        assert matches_path_glob('a/b', 'a?b') is False
        assert matches_path_glob('axb', 'a?b') is True

    def test_regex_metacharacters_in_a_pattern_are_literal(self) -> None:
        assert matches_path_glob('a+b.md', 'a+b.md') is True
        assert matches_path_glob('aab.md', 'a+b.md') is False


class TestResolutionVocabulary:

    def test_values(self) -> None:
        assert [r.value for r in MergeGateResolution] == ['full_gate', 'docs_gate', 'comments_gate']

    def test_only_docs_and_comments_are_narrow(self) -> None:
        assert [r for r in MergeGateResolution if r.is_narrow] == [
            MergeGateResolution.DOCS, MergeGateResolution.COMMENTS,
        ]

    def test_a_narrow_classification_with_a_miss_is_refused(self) -> None:
        with pytest.raises(ValueError, match='docs_gate') as raised:
            GateClassification(MergeGateResolution.DOCS, GateMiss(MissReason.DELETED, 'a.md'))
        assert 'deleted' in str(raised.value).lower()

    def test_a_full_classification_without_a_miss_is_refused(self) -> None:
        with pytest.raises(ValueError, match='full_gate') as raised:
            GateClassification(MergeGateResolution.FULL, None)
        assert 'None' in str(raised.value)


class TestClassifyDocsChange:

    def test_allowlisted_present_paths_take_the_docs_gate(self) -> None:
        changes = [_modified('README.md'), _added('docs/new.md'), _modified('plans/p.md')]

        assert classify_docs_change(changes, DOCS_PATHS) == GateClassification(MergeGateResolution.DOCS, None)

    def test_an_empty_change_set_takes_the_full_gate(self) -> None:
        assert classify_docs_change([], DOCS_PATHS) == _full(MissReason.EMPTY_DELTA, None)

    @pytest.mark.parametrize('path', [
        'dark-factory-orchestrator.yaml',
        'skills/spawn/spawn-claude.sh',
        'fused-memory/src/fused_memory/reconciliation/prompts/stage1.md',
        'pyproject.toml',
    ])
    def test_a_path_outside_the_allowlist_takes_the_full_gate_naming_it(self, path: str) -> None:
        changes = [_modified('README.md'), _modified(path)]

        assert classify_docs_change(changes, DOCS_PATHS) == _full(MissReason.NOT_ALLOWLISTED, path)

    def test_a_deleted_allowlisted_path_takes_the_full_gate(self) -> None:
        assert classify_docs_change([_deleted('docs/a.md')], DOCS_PATHS) == _full(MissReason.DELETED, 'docs/a.md')

    def test_a_mode_change_takes_the_full_gate(self) -> None:
        changes = [_modified('docs/a.md', old_mode='100644', new_mode='100755')]

        assert classify_docs_change(changes, DOCS_PATHS) == _full(MissReason.MODE_CHANGED, 'docs/a.md')

    @pytest.mark.parametrize('mode', ['120000', '160000'])
    def test_a_non_regular_file_mode_takes_the_full_gate(self, mode: str) -> None:
        changes = [_modified('docs/a.md', old_mode=mode, new_mode=mode)]

        assert classify_docs_change(changes, DOCS_PATHS) == _full(MissReason.MODE_CHANGED, 'docs/a.md')

    def test_an_unchanged_executable_mode_is_allowed(self) -> None:
        changes = [_modified('docs/a.md', old_mode='100755', new_mode='100755')]

        assert classify_docs_change(changes, DOCS_PATHS).resolution is MergeGateResolution.DOCS

    @pytest.mark.parametrize('path', ['/etc/passwd.md', 'docs/../src/x.md', '../README.md'])
    def test_an_unnormalised_path_takes_the_full_gate(self, path: str) -> None:
        assert classify_docs_change([_modified(path)], ['**']) == _full(MissReason.NOT_ALLOWLISTED, path)
