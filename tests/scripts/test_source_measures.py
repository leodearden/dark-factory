"""Unit tests of ``scripts/source_measures.py``, the per-file source measures.

Every snippet here is INLINE and synthetic, written with neutral module names
(``pkg.mod``), so these tests pin what each measure computes and know nothing
of any consumer's configuration. The few real-tree assertions are explicit
anti-vacuity anchors, not definitions.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest
import source_measures
from git_listing import git

REPO_ROOT = Path(__file__).parents[2]


def _indexed_repo(root: Path, files: dict[str, str]) -> Path:
    """A fresh repo at *root* whose index holds *files*.

    ``git ls-files`` reads the index, so nothing needs committing.
    """
    root.mkdir(parents=True)
    git(root, 'init', '-q')
    for relpath, text in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    git(root, 'add', '--', *files)
    return root


# ---------------------------------------------------------------------------
# file_size_measures -- lines and prose lines.


class TestFileSizeMeasures:
    def test_lines_counts_physical_lines(self) -> None:
        source = 'a = 1\nb = 2\nc = 3\n'
        assert source_measures.file_size_measures(source, path='t.py').lines == 3

    def test_lines_counts_a_final_line_without_a_trailing_newline(self) -> None:
        assert source_measures.file_size_measures('a = 1\nb = 2', path='t.py').lines == 2

    def test_blank_lines_are_lines_but_not_prose(self) -> None:
        measures = source_measures.file_size_measures('a = 1\n\n\nb = 2\n', path='t.py')
        assert measures.lines == 4
        assert measures.prose_lines == 0

    def test_module_docstring_counts_as_prose(self) -> None:
        measures = source_measures.file_size_measures('"""Doc."""\na = 1\n', path='t.py')
        assert measures.prose_lines == 1

    def test_function_and_class_docstrings_count_as_prose(self) -> None:
        source = (
            'class C:\n'
            '    """Class doc."""\n'
            '\n'
            '    def m(self):\n'
            '        """Method doc."""\n'
            '        return 1\n'
        )
        assert source_measures.file_size_measures(source, path='t.py').prose_lines == 2

    def test_async_function_docstring_counts_as_prose(self) -> None:
        source = 'async def f():\n    """Doc."""\n    return 1\n'
        assert source_measures.file_size_measures(source, path='t.py').prose_lines == 1

    def test_multiline_docstring_counts_once_per_line_it_spans(self) -> None:
        source = '"""Line one.\n\nLine three.\n"""\na = 1\n'
        measures = source_measures.file_size_measures(source, path='t.py')
        assert measures.lines == 5
        assert measures.prose_lines == 4

    def test_standalone_comment_lines_count_as_prose(self) -> None:
        source = '# one\n# two\na = 1\n'
        assert source_measures.file_size_measures(source, path='t.py').prose_lines == 2

    def test_trailing_inline_comment_makes_its_code_line_prose(self) -> None:
        source = 'a = 1  # why\nb = 2\n'
        assert source_measures.file_size_measures(source, path='t.py').prose_lines == 1

    def test_a_line_that_is_both_docstring_and_comment_counts_once(self) -> None:
        # The two line-number sets are unioned, not summed.
        source = '"""Doc."""  # trailing\na = 1\n'
        assert source_measures.file_size_measures(source, path='t.py').prose_lines == 1

    def test_a_string_literal_mentioning_hash_is_not_a_comment(self) -> None:
        # THE tokenize-vs-regex discriminator: no regex over source text gets
        # this right, and getting it wrong would inflate prose_lines on any file
        # that formats a '#'-bearing string.
        source = "url = 'http://x/#frag'\nheading = '# not a comment'\n"
        assert source_measures.file_size_measures(source, path='t.py').prose_lines == 0

    def test_a_non_docstring_string_expression_is_not_prose(self) -> None:
        # Only the FIRST body element of a module/class/function is a docstring.
        source = '"""Doc."""\n"""Not a docstring."""\na = 1\n'
        assert source_measures.file_size_measures(source, path='t.py').prose_lines == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.file_size_measures('def (:\n', path='broken.py')
        message = str(excinfo.value)
        assert 'broken.py' in message
        assert 'SyntaxError' in message

    def test_unparseable_source_never_returns_a_zero_measure(self) -> None:
        # INV-11: an unmeasurable file is the finding, not a 0 that silently
        # satisfies every comparison made against it.
        with pytest.raises(source_measures.MetricsError):
            source_measures.file_size_measures('class ???:\n', path='broken.py')

    def test_measures_are_frozen(self) -> None:
        measures = source_measures.file_size_measures('a = 1\n', path='t.py')
        with pytest.raises(dataclasses.FrozenInstanceError):
            measures.lines = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structural import measures: function-local (reach-back) imports, and
# re-export shim names.


class TestFunctionLocalImports:
    def test_import_from_inside_a_def_is_counted(self) -> None:
        source = 'def f():\n    from x import y\n    return y\n'
        assert source_measures.function_local_imports(source, path='t.py') == 1

    def test_plain_import_inside_a_def_is_counted(self) -> None:
        source = 'def f():\n    import x\n    return x\n'
        assert source_measures.function_local_imports(source, path='t.py') == 1

    def test_import_inside_an_async_def_is_counted(self) -> None:
        source = 'async def f():\n    from x import y\n    return y\n'
        assert source_measures.function_local_imports(source, path='t.py') == 1

    def test_import_inside_a_nested_def_is_counted_once(self) -> None:
        # Walking from each function node would visit the inner import twice
        # (once from the outer function, once from the inner); dedupe by node
        # identity keeps this at 1.
        source = (
            'def outer():\n'
            '    def inner():\n'
            '        from x import y\n'
            '        return y\n'
            '    return inner\n'
        )
        assert source_measures.function_local_imports(source, path='t.py') == 1

    def test_module_level_import_is_not_counted(self) -> None:
        source = 'from x import y\nimport z\n\n\ndef f():\n    return y\n'
        assert source_measures.function_local_imports(source, path='t.py') == 0

    def test_import_in_a_class_body_outside_any_function_is_not_counted(self) -> None:
        source = 'class C:\n    from x import y\n'
        assert source_measures.function_local_imports(source, path='t.py') == 0

    def test_import_in_a_method_body_is_counted(self) -> None:
        source = 'class C:\n    def m(self):\n        from x import y\n        return y\n'
        assert source_measures.function_local_imports(source, path='t.py') == 1

    def test_a_docstring_quoting_an_import_is_not_counted(self) -> None:
        # AST, not regex: prose quoting an import statement is not one.
        source = (
            'def f():\n'
            '    """Resolves via ``from pkg.mod import X``."""\n'
            '    return 1\n'
        )
        assert source_measures.function_local_imports(source, path='t.py') == 0

    def test_each_import_statement_counts_once_regardless_of_names(self) -> None:
        source = 'def f():\n    from x import a, b, c\n    return a\n'
        assert source_measures.function_local_imports(source, path='t.py') == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.function_local_imports('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)


class TestReexportNames:
    def test_unreferenced_module_level_import_from_names_are_reexports(self) -> None:
        source = 'from a import (B, C)\n'
        assert source_measures.reexport_names(source, path='t.py') == ['B', 'C']

    def test_a_referenced_binding_is_not_a_reexport(self) -> None:
        source = 'from a import B\n\nx = B()\n'
        assert source_measures.reexport_names(source, path='t.py') == []

    def test_an_alias_is_reported_under_the_bound_name(self) -> None:
        source = 'from a import B as C\n'
        assert source_measures.reexport_names(source, path='t.py') == ['C']

    def test_a_used_alias_is_not_a_reexport(self) -> None:
        source = 'from a import B as C\n\nx = C()\n'
        assert source_measures.reexport_names(source, path='t.py') == []

    def test_a_name_referenced_only_in_a_docstring_still_counts(self) -> None:
        # AST, not regex: prose mentioning the name does not make it used.
        source = '"""Re-exports B for consumers."""\nfrom a import B\n# B lives here\n'
        assert source_measures.reexport_names(source, path='t.py') == ['B']

    def test_a_name_referenced_inside_a_nested_function_counts_as_used(self) -> None:
        source = (
            'from a import B\n'
            '\n'
            '\n'
            'def outer():\n'
            '    def inner():\n'
            '        return B\n'
            '    return inner\n'
        )
        assert source_measures.reexport_names(source, path='t.py') == []

    def test_a_function_local_import_from_is_not_a_reexport(self) -> None:
        # Only MODULE-LEVEL ImportFrom bindings form the module's public
        # surface; a deferred import inside a function is the reach-back
        # measure's business, not this one's.
        source = 'def f():\n    from a import B\n    return 1\n'
        assert source_measures.reexport_names(source, path='t.py') == []

    def test_a_plain_import_is_not_counted(self) -> None:
        # The measure is the `from X import (...)` shim block; a bare
        # `import x` binds a module, not a re-exported name.
        source = 'import a\n'
        assert source_measures.reexport_names(source, path='t.py') == []

    def test_a_future_import_is_not_a_reexport(self) -> None:
        # `from __future__ import annotations` is a compiler directive, not a
        # name a downstream module could import, and ruff's F401 -- the
        # predicate this measure claims to agree with -- explicitly never flags
        # it. Counting it would REWARD deleting a future import, which silently
        # changes runtime annotation semantics (esc-5021-6).
        source = 'from __future__ import annotations\nimport os\nx = 1\n'
        assert source_measures.reexport_names(source, path='t.py') == []

    def test_a_future_import_does_not_mask_a_real_shim(self) -> None:
        # Excluding __future__ must not swallow the genuine shims beside it.
        source = 'from __future__ import annotations\nfrom a import B\n'
        assert source_measures.reexport_names(source, path='t.py') == ['B']

    def test_star_import_is_not_reported_as_a_name(self) -> None:
        source = 'from a import *\n'
        assert source_measures.reexport_names(source, path='t.py') == []

    def test_names_are_returned_sorted_and_deduped(self) -> None:
        source = 'from a import Z\nfrom b import A\n'
        assert source_measures.reexport_names(source, path='t.py') == ['A', 'Z']

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.reexport_names('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)

    def test_deleting_the_noqa_comment_does_not_change_the_measure(self) -> None:
        # THE ungameable property, pinned directly. A comment-scanning detector
        # would zero out on this purely cosmetic edit; the structural one cannot
        # see comments at all.
        source = 'from a import (  # noqa: F401  re-export shim\n    B,\n    C,\n)\n'
        stripped = source.replace('  # noqa: F401  re-export shim', '')
        assert source_measures.reexport_names(source, path='t.py') == ['B', 'C']
        assert source_measures.reexport_names(stripped, path='t.py') == ['B', 'C']


# ---------------------------------------------------------------------------
# The complexipy adapter, and its version contract (INV-11 for tools).

_TINY_SOURCE = (
    'def f(a):\n'
    '    if a:\n'
    '        for i in range(3):\n'
    '            if i:\n'
    '                return i\n'
    '    return 0\n'
    '\n'
    '\n'
    'class C:\n'
    '    def m(self):\n'
    '        return 1\n'
)


class TestCognitiveComplexity:
    def test_keyed_by_complexipy_qualname(self, tmp_path: Path) -> None:
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        scores = source_measures.file_cognitive_measures(target).per_function
        assert scores == {'f': 6, 'C::m': 0}

    def test_module_with_no_functions_returns_an_empty_map(self, tmp_path: Path) -> None:
        target = tmp_path / 'empty.py'
        target.write_text('X = 1\n', encoding='utf-8')
        assert source_measures.file_cognitive_measures(target).per_function == {}

    def test_file_total_is_reported_separately(self, tmp_path: Path) -> None:
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        assert source_measures.file_cognitive_measures(target).total == 6

    def test_both_projections_come_from_one_complexipy_measurement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The file total and the per-function map are two VIEWS of one
        # complexipy result, not two measurements; this counter forbids a
        # second measurement per file from growing back.
        #
        # The counter RECORDS AND DELEGATES rather than stubbing, so the
        # equalities below are still checked against a real measurement.
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        raw = source_measures._file_complexity(target)
        expected_total = int(raw.complexity)
        expected_per_function = {f.name: f.complexity for f in raw.functions}
        # Anti-vacuity: a module measuring 0 with no functions would satisfy
        # the equalities below while witnessing nothing.
        assert expected_total > 0
        assert expected_per_function

        calls: list[Path] = []
        original = source_measures._file_complexity

        def counting(path: Path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(source_measures, '_file_complexity', counting)
        measures = source_measures.file_cognitive_measures(target)

        assert calls == [target], calls
        assert measures.total == expected_total
        assert measures.per_function == expected_per_function

    def test_missing_complexipy_raises_naming_the_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Never a skipped measure and never a 0: a tool the instrument cannot
        # run is an instrument failure with a named cause (INV-11).
        monkeypatch.setitem(sys.modules, 'complexipy', None)
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.file_cognitive_measures(target)
        message = str(excinfo.value)
        assert 'complexipy' in message
        assert 'dev' in message

    def test_unparseable_file_raises_naming_the_path(self, tmp_path: Path) -> None:
        # INV-11's polarity: a file the instrument cannot measure is the
        # FINDING, never a silent 0 and never a skip.
        target = tmp_path / 'broken.py'
        target.write_text('def (:\n', encoding='utf-8')
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.file_cognitive_measures(target)
        assert 'broken.py' in str(excinfo.value)


class TestComplexipyVersionContract:
    def test_required_specifier_is_the_measured_range(self) -> None:
        assert source_measures.COMPLEXIPY_REQUIRED == '>=6.2,<7'

    def test_installed_version_satisfies_the_requirement(self) -> None:
        # The executable form of the measured pin.
        source_measures.require_complexipy()
        assert source_measures.satisfies_complexipy_requirement(
            source_measures.complexipy_version()
        )

    def test_pyproject_pin_matches_the_scripts_requirement(self) -> None:
        # Goes RED the moment someone relaxes the dependency to 7.x. The pin and
        # the runtime check are two halves of one contract; letting them drift
        # would leave every measurement silently taken with the wrong engine.
        pyproject = (REPO_ROOT / 'orchestrator/pyproject.toml').read_text(encoding='utf-8')
        assert f'"complexipy{source_measures.COMPLEXIPY_REQUIRED}"' in pyproject

    def test_version_out_of_range_raises_naming_both_versions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(source_measures, 'complexipy_version', lambda: '7.0.1')
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.require_complexipy()
        message = str(excinfo.value)
        assert '7.0.1' in message
        assert source_measures.COMPLEXIPY_REQUIRED in message

    def test_version_below_the_floor_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(source_measures, 'complexipy_version', lambda: '6.1.0')
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.require_complexipy()
        assert '6.1.0' in str(excinfo.value)

    @pytest.mark.parametrize(
        ('version', 'ok'),
        [
            ('6.2.0', True),
            ('6.2', True),
            ('6.9.9', True),
            ('6.1.9', False),
            ('6.0.0', False),
            ('5.0.0', False),
            ('7.0.0', False),
            ('7.0.1', False),
            ('8.0.0', False),
        ],
    )
    def test_specifier_boundaries(self, version: str, ok: bool) -> None:
        assert source_measures.satisfies_complexipy_requirement(version) is ok

    def test_missing_complexipy_distribution_raises_naming_the_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.metadata

        def _boom(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, 'version', _boom)
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.complexipy_version()
        assert 'complexipy' in str(excinfo.value)


class TestMaintainabilityIndex:
    def test_returns_a_float_for_a_tiny_module(self) -> None:
        value = source_measures.maintainability_index(_TINY_SOURCE, path='tiny.py')
        assert isinstance(value, float)
        assert 0.0 <= value <= 100.0

    def test_missing_radon_raises_naming_the_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, 'radon.metrics', None)
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.maintainability_index(_TINY_SOURCE, path='tiny.py')
        assert 'radon' in str(excinfo.value)

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.maintainability_index('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Patch targets: (module, leaf) pairs patched through a module in a given set.

MODULES = ('pkg.mod', 'pkg.lane')


class TestPatchTargets:
    def test_string_path_patch_is_detected(self) -> None:
        source = "patch('pkg.mod.run_scoped', x)\n"
        assert source_measures.patch_targets(source, MODULES) == {('pkg.mod', 'run_scoped')}

    def test_dotted_patch_is_detected(self) -> None:
        source = "mock.patch('pkg.mod.advance', x)\n"
        assert source_measures.patch_targets(source, MODULES) == {('pkg.mod', 'advance')}

    def test_string_path_setattr_is_detected(self) -> None:
        source = "monkeypatch.setattr('pkg.mod.foo', x)\n"
        assert source_measures.patch_targets(source, MODULES) == {('pkg.mod', 'foo')}

    def test_a_submodule_path_keeps_its_dotted_leaf(self) -> None:
        source = "patch('pkg.lane.sub.spin', x)\n"
        assert source_measures.patch_targets(source, MODULES) == {('pkg.lane', 'sub.spin')}

    def test_object_path_setattr_via_from_import(self) -> None:
        source = (
            'from pkg import mod\n'
            '\n'
            '\n'
            'def t(monkeypatch):\n'
            "    monkeypatch.setattr(mod, 'x', 1)\n"
        )
        assert source_measures.patch_targets(source, MODULES) == {('pkg.mod', 'x')}

    def test_object_path_setattr_via_import_as_alias(self) -> None:
        source = (
            'import pkg.mod as m\n'
            '\n'
            '\n'
            'def t(monkeypatch):\n'
            "    monkeypatch.setattr(m, 'y', 1)\n"
        )
        assert source_measures.patch_targets(source, MODULES) == {('pkg.mod', 'y')}

    def test_patch_object_on_the_bare_attribute_chain(self) -> None:
        source = "patch.object(pkg.mod, 'x', 1)\n"
        assert source_measures.patch_targets(source, MODULES) == {('pkg.mod', 'x')}

    def test_patch_object_on_a_module_outside_the_set_is_not_counted(self) -> None:
        source = (
            'from pkg import other\n'
            '\n'
            '\n'
            "p = patch.object(other, 'x', 1)\n"
        )
        assert source_measures.patch_targets(source, MODULES) == set()

    def test_an_unrelated_attribute_sharing_the_leaf_is_not_counted(self) -> None:
        source = "patch.object(workflow.mod, 'x', 1)\n"
        assert source_measures.patch_targets(source, MODULES) == set()

    def test_a_docstring_quoting_the_dotted_path_is_not_counted(self) -> None:
        source = '"""Patches pkg.mod.foo at the lookup site."""\n'
        assert source_measures.patch_targets(source, MODULES) == set()

    def test_a_comment_quoting_the_dotted_path_is_not_counted(self) -> None:
        source = "# patch('pkg.mod.foo')\nx = 1\n"
        assert source_measures.patch_targets(source, MODULES) == set()

    def test_distinct_names_not_call_sites(self) -> None:
        source = (
            "patch('pkg.mod.foo', a)\n"
            "patch('pkg.mod.foo', b)\n"
            "patch('pkg.mod.bar', c)\n"
        )
        assert source_measures.patch_targets(source, MODULES) == {
            ('pkg.mod', 'foo'),
            ('pkg.mod', 'bar'),
        }

    def test_a_bare_module_patch_with_no_leaf_is_not_counted(self) -> None:
        source = "patch('pkg.mod', x)\n"
        assert source_measures.patch_targets(source, MODULES) == set()

    def test_a_non_string_second_arg_is_not_counted(self) -> None:
        source = (
            'from pkg import mod\n'
            '\n'
            '\n'
            'p = patch.object(mod, NAME, 1)\n'
        )
        assert source_measures.patch_targets(source, MODULES) == set()

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.patch_targets('def (:\n', MODULES, path='broken.py')
        assert 'broken.py' in str(excinfo.value)

    def test_each_target_is_paired_with_its_module(self) -> None:
        source = "patch('pkg.mod._x', a)\npatch('pkg.lane._y', b)\n"
        targets = source_measures.patch_targets(source, MODULES)
        assert targets == {('pkg.mod', '_x'), ('pkg.lane', '_y')}
        assert {(target.module, target.leaf) for target in targets} == targets

    @pytest.mark.parametrize('modules', [('pkg', 'pkg.mod'), ('pkg.mod', 'pkg')])
    def test_a_string_target_maps_to_the_longest_module_prefix(
        self, modules: tuple[str, ...]
    ) -> None:
        source = "patch('pkg.mod._x', a)\npatch('pkg.other', b)\n"
        assert source_measures.patch_targets(source, modules) == {
            ('pkg.mod', '_x'),
            ('pkg', 'other'),
        }

    def test_a_deeper_module_is_recognised_by_chain_and_from_import(self) -> None:
        source = (
            'from a.b import c\n'
            '\n'
            "patch.object(a.b.c, '_x', 1)\n"
            "patch.object(c, '_y', 1)\n"
        )
        assert source_measures.patch_targets(source, ('a.b.c',)) == {
            ('a.b.c', '_x'),
            ('a.b.c', '_y'),
        }

    def test_a_single_segment_module_is_recognised_as_a_bare_name(self) -> None:
        source = "import pkg\n\npatch.object(pkg, '_x', 1)\n"
        assert source_measures.patch_targets(source, ('pkg',)) == {('pkg', '_x')}

    def test_a_receiver_is_not_prefix_matched_to_a_submodule(self) -> None:
        # Only a receiver that IS a set member counts; a submodule of one does
        # not, unlike a string target, whose leaf may run past the module.
        source = "patch.object(pkg.mod.sub, '_x', 1)\n"
        assert source_measures.patch_targets(source, MODULES) == set()


# ---------------------------------------------------------------------------
# Private-attribute reads.


class TestPrivateReads:
    def test_a_single_private_attribute_read_counts_one(self) -> None:
        assert source_measures.private_reads('obj._inflight\n', path='t.py') == 1

    def test_a_chained_private_read_counts_each_hop(self) -> None:
        # Both hops reach into internals.
        assert source_measures.private_reads('obj._a._b\n', path='t.py') == 2

    def test_a_dunder_is_not_a_private_read(self) -> None:
        # Dunders are Python protocol, not internals.
        assert source_measures.private_reads('obj.__dict__\nobj.__class__\n', path='t.py') == 0

    def test_self_and_cls_receivers_are_excluded(self) -> None:
        # A class's own helpers are not someone else's internals. This is a
        # STRUCTURAL predicate, not a name list.
        source = (
            'class T:\n'
            '    def t(self):\n'
            '        self._helper()\n'
            '        return cls._x\n'
        )
        assert source_measures.private_reads(source, path='t.py') == 0

    def test_a_public_attribute_is_not_counted(self) -> None:
        assert source_measures.private_reads('obj.snapshot()\n', path='t.py') == 0

    def test_a_docstring_quoting_a_private_read_is_not_counted(self) -> None:
        assert source_measures.private_reads('"""Reads obj._inflight."""\n', path='t.py') == 0

    def test_a_private_read_off_self_dot_something_is_counted(self) -> None:
        # `self.obj._x` reaches into another object even though the chain
        # starts at self -- only the BARE `self`/`cls` receiver is excluded.
        assert source_measures.private_reads('self.obj._x\n', path='t.py') == 1

    def test_a_private_write_counts_too(self) -> None:
        # Writes couple to an internal name exactly as reads do.
        assert source_measures.private_reads('obj._inflight = 1\n', path='t.py') == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(source_measures.MetricsError) as excinfo:
            source_measures.private_reads('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)


class TestTheTreeAndSourceFormsAgree:
    """Each measure has a source-taking form and a tree-taking one.

    A sweep parses once and calls the tree form; everything else calls the
    source form, which is a parse plus a call of the tree form. They are one
    implementation, and this is what notices a change that makes them two.
    """

    _SOURCE = (
        '"""Docstring."""\n'
        'from pkg import mod\n'
        'from unittest.mock import patch\n'
        'from pathlib import Path\n\n'
        '# a comment\n'
        'def test_x():\n'
        '    from json import dumps\n'
        "    with patch('pkg.mod.helper'):\n"
        '        assert mod._worker\n'
    )

    @pytest.mark.parametrize(
        ('from_source', 'from_tree'),
        [
            pytest.param(
                lambda src: source_measures.file_size_measures(src, path='t.py'),
                lambda src, tree: source_measures.file_size_measures_in_tree(
                    src, tree, path='t.py'
                ),
                id='file_size_measures',
            ),
            pytest.param(
                lambda src: source_measures.function_local_imports(src, path='t.py'),
                lambda src, tree: source_measures.function_local_imports_in_tree(tree),
                id='function_local_imports',
            ),
            pytest.param(
                lambda src: source_measures.reexport_names(src, path='t.py'),
                lambda src, tree: source_measures.reexport_names_in_tree(tree),
                id='reexport_names',
            ),
            pytest.param(
                lambda src: source_measures.patch_targets(src, MODULES, path='t.py'),
                lambda src, tree: source_measures.patch_targets_in_tree(tree, MODULES),
                id='patch_targets',
            ),
            pytest.param(
                lambda src: source_measures.private_reads(src, path='t.py'),
                lambda src, tree: source_measures.private_reads_in_tree(tree),
                id='private_reads',
            ),
        ],
    )
    def test_the_two_forms_return_the_same_measure(self, from_source, from_tree) -> None:
        tree = source_measures.parse_source(self._SOURCE, path='t.py')
        expected = from_source(self._SOURCE)
        assert from_tree(self._SOURCE, tree) == expected
        # ANTI-VACUITY: the snippet exercises every measure, so none is trivially
        # equal as an empty value.
        assert expected not in (0, [], set(), frozenset(), None)

    def test_parsing_remembers_nothing_between_calls(self) -> None:
        # Two callers of one source must not share a tree: a measure that
        # mutated it would corrupt the other, and the only guard would be prose.
        source = 'import json\n'
        assert source_measures.parse_source(source, path='a.py') is not (
            source_measures.parse_source(source, path='a.py')
        )


# ---------------------------------------------------------------------------
# The tracked-file listing: no listing means no measurement, never an empty one.


class TestTrackedPythonFilesFailsHard:
    """A listing that failed would read as an empty tree, so it raises instead."""

    def test_a_missing_git_is_a_named_hard_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _indexed_repo(tmp_path / 'repo', {'app/x.py': '\n'})
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))
        with pytest.raises(source_measures.MetricsError, match='could not run git'):
            source_measures.tracked_python_files(root)

    def test_a_directory_that_is_not_a_repository_is_a_named_hard_failure(
        self, tmp_path: Path
    ) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()
        with pytest.raises(source_measures.MetricsError, match='git'):
            source_measures.tracked_python_files(plain)

    def test_a_subdirectory_of_a_repository_is_refused_not_listed(
        self, tmp_path: Path
    ) -> None:
        # Listing from inside a repo yields only that subtree's files, which
        # would read as the whole tree.
        root = _indexed_repo(tmp_path / 'repo', {'app/x.py': '\n'})
        with pytest.raises(
            source_measures.MetricsError, match='not the top of a git work tree'
        ):
            source_measures.tracked_python_files(root / 'app')

    def test_the_listing_is_repo_relative_sorted_and_python_only(
        self, tmp_path: Path
    ) -> None:
        root = _indexed_repo(
            tmp_path / 'repo',
            {'b/z.py': '\n', 'a.py': '\n', 'notes.md': 'x\n', 'c/d/e.py': '\n'},
        )
        assert source_measures.tracked_python_files(root) == ('a.py', 'b/z.py', 'c/d/e.py')

    def test_the_real_checkout_is_listed_whole(self) -> None:
        # ANTI-VACUITY: a listing that collapsed to a few files would read as a
        # near-empty tree. A floor, not a count.
        tracked = source_measures.tracked_python_files(REPO_ROOT)
        assert len(tracked) >= 1500
        assert 'scripts/source_measures.py' in tracked


# ---------------------------------------------------------------------------
# The workspace domain: every tracked .py of each member, classified.

#: Declared out of alphabetical order, so declared order is what is pinned.
_DOMAIN_PYPROJECT = '[tool.uv.workspace]\nmembers = ["beta", "alpha"]\n'

_DOMAIN_TRACKED = (
    'alpha/src/alpha/__init__.py',
    'alpha/src/alpha/mod.py',
    'alpha/src/alpha/sub/deep.py',
    'alpha/tests/test_a.py',
    'beta/src/beta/b.py',
    'beta/tests/helpers/fx.py',
    'alpha/scripts/tool.py',
    'scripts/x.py',
    'scripts/legibility/y.py',
    'scripts/tests/test_x.py',
    'tests/test_y.py',
    'hooks/h.py',
    'hooks/tests/test_h.py',
)


def _domain_repo(
    root: Path,
    *,
    pyproject: str | None = _DOMAIN_PYPROJECT,
    tracked: tuple[str, ...] = _DOMAIN_TRACKED,
    untracked: tuple[str, ...] = (),
) -> Path:
    """A repo whose index holds *tracked* (each with distinct content) and whose
    root pyproject.toml is *pyproject*, absent when None."""
    files = {relpath: f'# {relpath}\n' for relpath in tracked}
    if pyproject is not None:
        files['pyproject.toml'] = pyproject
    repo = _indexed_repo(root, files)
    for relpath in untracked:
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'# {relpath}\n', encoding='utf-8')
    return repo


class TestWorkspaceDomain:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return _domain_repo(tmp_path / 'repo', untracked=('alpha/src/alpha/untracked.py',))

    def test_members_come_in_declared_order_then_the_pseudo_members(
        self, root: Path
    ) -> None:
        members = source_measures.workspace_domain(root)
        assert tuple(member.name for member in members) == ('beta', 'alpha', 'scripts', 'tests')
        assert tuple(member.pseudo for member in members) == (False, False, True, True)

    def test_each_members_files_are_classified_in_path_order(self, root: Path) -> None:
        src, tests = source_measures.FileKind.SRC, source_measures.FileKind.TESTS
        members = source_measures.workspace_domain(root)
        assert {
            member.name: [(f.path, f.kind, f.import_name) for f in member.files]
            for member in members
        } == {
            'beta': [
                ('beta/src/beta/b.py', src, 'beta.b'),
                ('beta/tests/helpers/fx.py', tests, None),
            ],
            'alpha': [
                # A package's __init__ is named by its package.
                ('alpha/src/alpha/__init__.py', src, 'alpha'),
                ('alpha/src/alpha/mod.py', src, 'alpha.mod'),
                ('alpha/src/alpha/sub/deep.py', src, 'alpha.sub.deep'),
                ('alpha/tests/test_a.py', tests, None),
            ],
            'scripts': [
                ('scripts/legibility/y.py', src, 'legibility.y'),
                # The tests root wins over the scripts src root that contains it.
                ('scripts/tests/test_x.py', tests, None),
                ('scripts/x.py', src, 'x'),
            ],
            'tests': [('tests/test_y.py', tests, None)],
        }

    def test_files_outside_every_member_root_are_not_in_the_domain(
        self, root: Path
    ) -> None:
        paths = {f.path for member in source_measures.workspace_domain(root) for f in member.files}
        outside = {
            'hooks/h.py',
            'hooks/tests/test_h.py',
            'alpha/scripts/tool.py',
            'alpha/src/alpha/untracked.py',
        }
        assert not outside & paths

    def test_every_blob_is_the_content_sha_of_its_file(self, root: Path) -> None:
        files = [f for member in source_measures.workspace_domain(root) for f in member.files]
        assert files
        for f in files:
            assert f.blob == git(root, 'hash-object', f.path).strip(), f.path

    def test_the_result_is_frozen(self, root: Path) -> None:
        member = source_measures.workspace_domain(root)[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            member.name = 'other'  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            member.files[0].blob = '0' * 40  # type: ignore[misc]

    def test_kind_is_a_structured_value_that_reads_as_its_name(self, root: Path) -> None:
        kinds = {f.kind for member in source_measures.workspace_domain(root) for f in member.files}
        assert all(isinstance(kind, source_measures.FileKind) for kind in kinds)
        assert str(source_measures.FileKind.SRC) == 'src'
        assert str(source_measures.FileKind.TESTS) == 'tests'

    def test_the_real_workspace_is_enumerated(self) -> None:
        # ANTI-VACUITY over the real checkout: floors and membership, never
        # exact counts.
        members = source_measures.workspace_domain(REPO_ROOT)
        names = tuple(member.name for member in members)
        assert len(names) >= 9
        assert names[-2:] == ('scripts', 'tests')
        assert {'orchestrator', 'shared', 'fused-memory'} <= set(names[:-2])
        by_name = {member.name: {f.path: f for f in member.files} for member in members}
        layer = by_name['scripts']['scripts/source_measures.py']
        assert (layer.kind, layer.import_name) == (source_measures.FileKind.SRC, 'source_measures')
        package = by_name['orchestrator']['orchestrator/src/orchestrator/__init__.py']
        assert package.import_name == 'orchestrator'
        script_kinds = [f.kind for f in by_name['scripts'].values()]
        assert script_kinds.count(source_measures.FileKind.SRC) >= 50
        assert script_kinds.count(source_measures.FileKind.TESTS) >= 50


def _workspace(*members: str) -> str:
    return '[tool.uv.workspace]\nmembers = [{}]\n'.format(
        ', '.join(f'"{member}"' for member in members)
    )


def _stage_conflict(root: Path, relpath: str) -> None:
    """Leave *relpath* unmerged in *root*'s index: stage 1 and stage 2, two blobs."""
    blobs = []
    for content in ('base\n', 'ours\n'):
        source = root / 'conflict-content'
        source.write_text(content, encoding='utf-8')
        blobs.append(git(root, 'hash-object', '-w', str(source)).strip())
        source.unlink()
    records = ''.join(
        f'100644 {blob} {stage}\t{relpath}\n' for stage, blob in enumerate(blobs, start=1)
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    proc = subprocess.run(
        ['git', 'update-index', '--index-info'],
        cwd=root,
        input=records,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


class TestWorkspaceDomainFailsHard:
    """A domain that cannot be enumerated whole raises, never reads as partial."""

    def test_a_missing_pyproject_is_refused(self, tmp_path: Path) -> None:
        root = _domain_repo(tmp_path / 'repo', pyproject=None)
        with pytest.raises(source_measures.MetricsError, match='pyproject.toml'):
            source_measures.workspace_domain(root)

    @pytest.mark.parametrize(
        ('pyproject', 'pattern'),
        [
            pytest.param('[tool.uv.workspace\n', 'pyproject.toml', id='invalid-toml'),
            pytest.param(
                '[project]\nname = "x"\n', r'tool\.uv\.workspace', id='no-workspace-table'
            ),
            pytest.param(_workspace(), r'tool\.uv\.workspace', id='no-members'),
            pytest.param(
                '[tool.uv.workspace]\nmembers = ["alpha", 3]\n',
                r'tool\.uv\.workspace',
                id='non-string-member',
            ),
            pytest.param(
                _workspace('alpha', 'packages/*'), r"glob.*'packages/\*'", id='glob-member'
            ),
            pytest.param(
                _workspace('alpha', 'scripts'),
                r"'scripts'.*pseudo-member",
                id='scripts-pseudo-member-collision',
            ),
            pytest.param(
                _workspace('alpha', 'tests'),
                r"'tests'.*pseudo-member",
                id='tests-pseudo-member-collision',
            ),
            pytest.param(
                _workspace('alpha', 'alpha'), r"duplicate.*'alpha'", id='duplicate-member'
            ),
        ],
    )
    def test_a_malformed_member_list_is_refused_by_name(
        self, tmp_path: Path, pyproject: str, pattern: str
    ) -> None:
        root = _domain_repo(tmp_path / 'repo', pyproject=pyproject)
        with pytest.raises(source_measures.MetricsError, match=pattern):
            source_measures.workspace_domain(root)

    def test_a_declared_member_with_no_tracked_python_is_refused(
        self, tmp_path: Path
    ) -> None:
        root = _domain_repo(
            tmp_path / 'repo',
            pyproject=_workspace('alpha', 'gamma'),
            tracked=('alpha/src/alpha/a.py', 'gamma/pyproject.toml'),
        )
        with pytest.raises(source_measures.MetricsError, match='gamma'):
            source_measures.workspace_domain(root)

    def test_an_empty_pseudo_member_is_returned_empty(self, tmp_path: Path) -> None:
        # Pseudo-members are conventions, not declarations: a tree without
        # scripts/ or tests/ legitimately has none there.
        root = _domain_repo(
            tmp_path / 'repo', pyproject=_workspace('alpha'), tracked=('alpha/src/alpha/a.py',)
        )
        members = source_measures.workspace_domain(root)
        assert [(member.name, member.files) for member in members[1:]] == [
            ('scripts', ()),
            ('tests', ()),
        ]

    def test_an_unmerged_index_entry_is_refused_naming_the_path(
        self, tmp_path: Path
    ) -> None:
        root = _domain_repo(tmp_path / 'repo')
        _stage_conflict(root, 'alpha/src/alpha/conflict.py')
        with pytest.raises(
            source_measures.MetricsError, match=r'unmerged.*alpha/src/alpha/conflict\.py'
        ):
            source_measures.workspace_domain(root)

    def test_a_missing_git_is_a_named_hard_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _domain_repo(tmp_path / 'repo')
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))
        with pytest.raises(source_measures.MetricsError, match='could not run git'):
            source_measures.workspace_domain(root)

    def test_a_subdirectory_of_the_work_tree_is_refused(self, tmp_path: Path) -> None:
        root = _domain_repo(tmp_path / 'repo')
        with pytest.raises(
            source_measures.MetricsError, match='not the top of a git work tree'
        ):
            source_measures.workspace_domain(root / 'alpha')


# ---------------------------------------------------------------------------
# Layering: the measures import nothing from a consumer, and stay lazy about
# the third-party tools.


def test_importing_the_layer_loads_no_consumer_and_no_tool() -> None:
    # A subprocess, because xdist workers share sys.modules with whatever else
    # the worker already imported.
    code = (
        'import sys\n'
        'import source_measures\n'
        "print(sorted({'merge_lane_metrics', 'complexipy', 'radon'} & set(sys.modules)))\n"
    )
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=REPO_ROOT / 'scripts',
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == '[]', proc.stdout
