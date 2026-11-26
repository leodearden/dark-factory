"""Which merge changes qualify for the narrow merge gate: path globs and the classifiers (task 5411)."""
from __future__ import annotations

import textwrap
from collections.abc import Callable

import pytest

from orchestrator.narrow_gate_class import (
    ChangedPath,
    GateClassification,
    GateMiss,
    MergeGateResolution,
    MissReason,
    classify_change,
    classify_docs_change,
    matches_path_glob,
    python_comments_only,
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


def _src(text: str) -> str:
    return textwrap.dedent(text).lstrip('\n')


_BASE = _src('''
    """Module docstring."""
    import os


    class Thing:
        """Class docstring."""

        def method(self) -> str:
            """Method docstring."""
            label = 'a label'
            return label + os.sep


    async def fetch() -> int:
        """Async docstring."""
        count = 1  # noqa: F841
        total = 2
        return total
''')


class TestPythonCommentsOnly:

    @pytest.mark.parametrize(('old', 'new'), [
        ("label = 'a label'\n", "label = 'a label'  # a new comment\n"),
        ('count = 1  # noqa: F841\n', 'count = 1  # noqa: F841  # reworded\n'),
        ('"""Module docstring."""', '"""A reworded module docstring."""'),
        ('"""Class docstring."""', '"""A reworded class docstring."""'),
        ('"""Method docstring."""', '"""A reworded method docstring."""'),
        ('"""Async docstring."""', '"""A reworded async docstring."""'),
        ('    """Async docstring."""\n', ''),
        ('count = 1  # noqa: F841\n', 'count = 1  # type: ignore[assignment]\n'),
    ])
    def test_a_comment_or_docstring_edit_is_comments_only(self, old: str, new: str) -> None:
        assert _BASE.count(old) == 1
        assert python_comments_only(_BASE, _BASE.replace(old, new)) is True

    def test_adding_a_docstring_where_none_existed_is_comments_only(self) -> None:
        base = _src('''
            def f() -> int:
                return 1
        ''')
        head = _src('''
            def f() -> int:
                """Now documented."""
                return 1
        ''')

        assert python_comments_only(base, head) is True

    @pytest.mark.parametrize(('old', 'new'), [
        ("label = 'a label'", "label = 'another label'"),
        ('total = 2\n    return total', 'subtotal = 2\n    return subtotal'),
        ('count = 1  # noqa: F841\n    total = 2', 'total = 2\n    count = 1  # noqa: F841'),
    ])
    def test_a_code_edit_is_not_comments_only(self, old: str, new: str) -> None:
        assert _BASE.count(old) == 1
        assert python_comments_only(_BASE, _BASE.replace(old, new)) is False

    def test_a_changed_shebang_is_not_comments_only(self) -> None:
        body = 'print(1)\n'

        assert python_comments_only('#!/usr/bin/env python3\n' + body, '#!/usr/bin/python\n' + body) is False

    def test_adding_a_shebang_is_not_comments_only(self) -> None:
        assert python_comments_only('print(1)\n', '#!/usr/bin/env python3\nprint(1)\n') is False

    def test_an_unchanged_shebang_with_a_comment_edit_is_comments_only(self) -> None:
        base = '#!/usr/bin/env python3\nprint(1)\n'

        assert python_comments_only(base, base + '# trailing\n') is True

    def test_a_changed_coding_cookie_is_not_comments_only(self) -> None:
        body = "label = 'a'\n"

        assert python_comments_only(body, '# -*- coding: latin-1 -*-\n' + body) is False

    @pytest.mark.parametrize(('base', 'head'), [
        ('def broken(:\n', 'def broken(:\n# comment\n'),
        ('x = 1\n', 'x = (\n'),
        ('x = (\n', 'x = 1\n'),
    ])
    def test_a_syntax_error_on_either_side_is_not_comments_only(self, base: str, head: str) -> None:
        assert python_comments_only(base, head) is False

    def test_the_inputs_are_not_mutated_between_calls(self) -> None:
        head = _BASE.replace('"""Module docstring."""', '"""Reworded."""')

        assert python_comments_only(_BASE, head) is True
        assert python_comments_only(_BASE, head.replace("'a label'", "'x'")) is False


_COMMENT_ONLY_PY = ("x = 1\n", "x = 1  # explained\n")
_CODE_CHANGED_PY = ("x = 1\n", "x = 2\n")


def _reader(blobs: dict[str, tuple[str | None, str | None]]) -> Callable[[str], tuple[str | None, str | None]]:
    return lambda path: blobs[path]


class TestClassifyChange:

    def test_only_allowlisted_docs_take_the_docs_gate(self) -> None:
        changes = [_modified('README.md'), _added('docs/a.md')]

        assert classify_change(changes, DOCS_PATHS, _reader({})) == GateClassification(MergeGateResolution.DOCS)

    def test_docs_plus_a_comment_only_py_take_the_comments_gate(self) -> None:
        changes = [_modified('README.md'), _modified('pkg/mod.py')]

        classification = classify_change(changes, DOCS_PATHS, _reader({'pkg/mod.py': _COMMENT_ONLY_PY}))

        assert classification == GateClassification(MergeGateResolution.COMMENTS)

    def test_a_comment_only_py_alone_takes_the_comments_gate(self) -> None:
        classification = classify_change(
            [_modified('pkg/mod.py')], DOCS_PATHS, _reader({'pkg/mod.py': _COMMENT_ONLY_PY}),
        )

        assert classification == GateClassification(MergeGateResolution.COMMENTS)

    def test_a_py_whose_ast_changed_takes_the_full_gate_naming_it(self) -> None:
        changes = [_modified('README.md'), _modified('pkg/mod.py')]

        classification = classify_change(changes, DOCS_PATHS, _reader({'pkg/mod.py': _CODE_CHANGED_PY}))

        assert classification == _full(MissReason.CODE_CHANGED, 'pkg/mod.py')

    @pytest.mark.parametrize('path', ['crates/a/src/lib.rs', 'dashboard/src/app.ts'])
    def test_a_language_with_no_comment_predicate_takes_the_full_gate(self, path: str) -> None:
        classification = classify_change([_modified(path)], DOCS_PATHS, _reader({path: ('// a\n', '// b\n')}))

        assert classification == _full(MissReason.NO_COMMENT_PREDICATE, path)

    def test_a_newly_added_py_takes_the_full_gate(self) -> None:
        classification = classify_change(
            [_added('pkg/new.py')], DOCS_PATHS, _reader({'pkg/new.py': (None, '# only a comment\n')}),
        )

        assert classification == _full(MissReason.CODE_CHANGED, 'pkg/new.py')

    @pytest.mark.parametrize('pair', [(None, 'x = 1\n'), ('x = 1\n', None), (None, None)])
    def test_an_unreadable_side_takes_the_full_gate(self, pair: tuple[str | None, str | None]) -> None:
        classification = classify_change([_modified('pkg/mod.py')], DOCS_PATHS, _reader({'pkg/mod.py': pair}))

        assert classification == _full(MissReason.UNDECODABLE, 'pkg/mod.py')

    def test_a_deleted_py_takes_the_full_gate(self) -> None:
        classification = classify_change([_deleted('pkg/mod.py')], DOCS_PATHS, _reader({}))

        assert classification == _full(MissReason.DELETED, 'pkg/mod.py')

    def test_a_mode_changed_py_takes_the_full_gate(self) -> None:
        changes = [_modified('pkg/mod.py', old_mode='100644', new_mode='100755')]

        classification = classify_change(changes, DOCS_PATHS, _reader({'pkg/mod.py': _COMMENT_ONLY_PY}))

        assert classification == _full(MissReason.MODE_CHANGED, 'pkg/mod.py')

    def test_an_empty_change_set_takes_the_full_gate(self) -> None:
        assert classify_change([], DOCS_PATHS, _reader({})) == _full(MissReason.EMPTY_DELTA, None)

    def test_the_miss_is_the_first_in_sorted_path_order(self) -> None:
        changes = [_modified('z/lib.rs'), _modified('b/mod.py'), _modified('a/mod.py')]
        blobs = {'z/lib.rs': ('', ''), 'b/mod.py': _CODE_CHANGED_PY, 'a/mod.py': _CODE_CHANGED_PY}

        assert classify_change(changes, DOCS_PATHS, _reader(blobs)) == _full(MissReason.CODE_CHANGED, 'a/mod.py')

    def test_a_docs_path_is_never_read(self) -> None:
        changes = [_modified('README.md'), _modified('pkg/mod.py')]

        classification = classify_change(changes, DOCS_PATHS, _reader({'pkg/mod.py': _COMMENT_ONLY_PY}))

        assert classification.resolution is MergeGateResolution.COMMENTS
