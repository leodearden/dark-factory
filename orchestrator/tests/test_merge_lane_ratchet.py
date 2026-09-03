"""The merge-lane quality ratchet, and the unit tests of the instrument behind it.

PRD ``plans/merge-lane-quality-prd.md`` task alpha. The instrument itself lives
in ``scripts/merge_lane_metrics.py``; this module hosts both its detector unit
tests and the real tree-scanning gate, in the same single-file shape as the
guard it supersedes (``test_merge_queue_reachback_patch_guard.py`` carries its
``_find_merge_queue_private_patches`` fixtures and its tree sweep in one file).

Ratchet contract
----------------
``orchestrator/tests/merge_lane_ratchet_baseline.json`` freezes every measure at
its value on the commit that introduced it. The gate FAILS on any measure that
rises above its baseline (ratchet semantics): raising a measure is not allowed;
only lowering one is. Equality is permitted -- this is a ratchet, not a day-one
gate, and on the introducing commit 62 lane functions already exceed cognitive
15 and ``merge_queue.py`` is 21,550 lines. The two CEILINGS (1,500 lines per
file, cognitive 15 per function) therefore apply only to paths and qualnames
ABSENT from the baseline, so the grandfathering can only ever shrink.

A task that legitimately LOWERS a measure regenerates the baseline in the SAME
commit (``python scripts/merge_lane_metrics.py --write-baseline
orchestrator/tests/merge_lane_ratchet_baseline.json``). A task may never raise
one. Do not regenerate the baseline merely to make this test pass.

Why the instrument fails HARD where its neighbours fail soft
------------------------------------------------------------
Every sibling guard in this directory fails SOFT on an unparseable file
(``_find_merge_queue_private_patches`` returns ``[]``; ``_scans_whole_tree_py``
returns False) because they sweep the WHOLE tree, so a mid-edit or deliberately
malformed fixture file must not redden a guard about something else. This
instrument inverts that polarity for its own named cluster: an Appendix A file
the script cannot parse is not noise, it IS the finding (INV-11, no silent
fail-soft). The 559-file test sweep keeps the siblings' per-file fail-soft
polarity, but every skipped file lands in ``enumeration.unreadable`` and the
ratchet refuses to compare a partial enumeration at all -- so a degraded sweep
can never masquerade as a clean tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

# This module AST-parses the 22-path Appendix A cluster plus every *.py under
# orchestrator/tests/ (559 files at authorship time), and runs complexipy over
# the cluster. MEASURED end to end at 35-65s. The 60s ini default would be a
# coin flip, and pytest-timeout's thread method enforces it by os._exit()ing the
# xdist worker -- which with --max-worker-restart=0 truncates the whole session
# and reports against an innocent test (esc-3980-1). See
# WHOLE_TREE_SCAN_TEST_TIMEOUT in _orch_helpers.py.
#
# NOTE this mark is deliberate and NOT compelled by the family guard
# test_whole_tree_scan_timeout_guard.py: its ``_scans_whole_tree_py`` detector
# looks for a literal ``rglob('*.py')`` in the TEST file's own source, and this
# module's sweep lives in scripts/merge_lane_metrics.py instead. Adding a marked
# module cannot break that guard either -- its family invariant uses ``>=``
# floors.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# Make scripts/merge_lane_metrics.py importable. This is the sanctioned
# precedent for an orchestrator/tests/ module importing a scripts/ one -- see
# test_run_vllm_eval.py -- and needs no new pyright config: orchestrator/
# pyproject.toml's [tool.pyright] extraPaths already carries "../scripts".
_SCRIPTS = Path(__file__).parents[2] / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import merge_lane_metrics as metrics  # type: ignore[import-not-found]  # noqa: E402

_REPO_ROOT = Path(__file__).parents[2]

# The 18 orchestrator/src literal paths of PRD Appendix A, verbatim. git_ops.py
# is listed separately below because Appendix A adds it under a different rule
# (measured, but exempt from the file-size ceiling -- PRD decision 9).
_APPENDIX_A_SRC = (
    'orchestrator/src/orchestrator/merge_queue.py',
    'orchestrator/src/orchestrator/merge_gates.py',
    'orchestrator/src/orchestrator/merge_types.py',
    'orchestrator/src/orchestrator/merge_shadow.py',
    'orchestrator/src/orchestrator/merge_liveness.py',
    'orchestrator/src/orchestrator/merge_disposition.py',
    'orchestrator/src/orchestrator/merge_queue_store.py',
    'orchestrator/src/orchestrator/merge_completion.py',
    'orchestrator/src/orchestrator/merge_drift.py',
    'orchestrator/src/orchestrator/merge_speculation_controller.py',
    'orchestrator/src/orchestrator/merge_request_ledger.py',
    'orchestrator/src/orchestrator/merge_skew_tripwire.py',
    'orchestrator/src/orchestrator/lane_lifecycle.py',
    'orchestrator/src/orchestrator/offline_lane.py',
    'orchestrator/src/orchestrator/warm_lane_pool.py',
    'orchestrator/src/orchestrator/landing_evidence.py',
    'orchestrator/src/orchestrator/landed_outbox.py',
    'orchestrator/src/orchestrator/recover_main.py',
)
_GIT_OPS = 'orchestrator/src/orchestrator/git_ops.py'
_APPENDIX_A_TESTS = (
    'orchestrator/tests/_serial_merge_worker.py',
    'orchestrator/tests/_merge_queue_harness.py',
    'orchestrator/tests/conftest.py',
)
_MERGE_LANE_GLOB = 'orchestrator/src/orchestrator/merge_lane/**/*.py'


# ---------------------------------------------------------------------------
# CLUSTER_PATHS -- the SPOT source of Appendix A.


class TestClusterPathSpec:
    def test_cluster_paths_covers_every_appendix_a_src_module(self) -> None:
        assert set(_APPENDIX_A_SRC) <= set(metrics.CLUSTER_PATHS)

    def test_cluster_paths_includes_git_ops(self) -> None:
        # Measured but not gated by the file-size ceiling (PRD decision 9).
        assert _GIT_OPS in metrics.CLUSTER_PATHS

    def test_cluster_paths_includes_the_three_test_paths(self) -> None:
        assert set(_APPENDIX_A_TESTS) <= set(metrics.CLUSTER_PATHS)

    def test_cluster_paths_carries_one_merge_lane_glob(self) -> None:
        globs = [p for p in metrics.CLUSTER_PATHS if '*' in p]
        assert globs == [_MERGE_LANE_GLOB]

    def test_cluster_paths_has_exactly_the_appendix_a_entries(self) -> None:
        expected = (
            set(_APPENDIX_A_SRC)
            | {_GIT_OPS}
            | set(_APPENDIX_A_TESTS)
            | {_MERGE_LANE_GLOB}
        )
        assert set(metrics.CLUSTER_PATHS) == expected

    def test_cluster_paths_is_an_ordered_tuple(self) -> None:
        # Order is part of the spec: `params.cluster_paths` is compared against
        # it in check_against_baseline, and a stable order keeps that diff
        # readable when Appendix A is deliberately edited.
        assert isinstance(metrics.CLUSTER_PATHS, tuple)

    def test_size_ceiling_exempt_is_exactly_git_ops(self) -> None:
        # PRD decision 9: git_ops.py is the git engine for warm lanes, the
        # scheduler and recovery -- not only the merge lane -- and its split is
        # a follow-up PRD. The exemption is scoped to the CEILING; git_ops.py is
        # still ratcheted (pinned by TestSizeCeilingExemptionIsScoped).
        assert metrics.SIZE_CEILING_EXEMPT == frozenset({_GIT_OPS})

    def test_ceiling_constants(self) -> None:
        assert metrics.FILE_LINE_CEILING == 1500
        assert metrics.NEW_FUNCTION_COGNITIVE_CEILING == 15


# ---------------------------------------------------------------------------
# resolve_cluster_paths -- INV-11: a partial enumeration is never returned
# silently, and completeness is legible in the RESULT.


class TestResolveClusterPaths:
    def test_real_repo_enumeration_is_complete(self) -> None:
        enumeration = metrics.resolve_cluster_paths(_REPO_ROOT)
        assert enumeration.complete is True
        assert enumeration.unreadable == ()

    def test_real_repo_enumeration_resolves_every_literal_path(self) -> None:
        enumeration = metrics.resolve_cluster_paths(_REPO_ROOT)
        literals = {p for p in metrics.CLUSTER_PATHS if '*' not in p}
        assert literals <= set(enumeration.resolved)

    def test_resolved_paths_are_repo_relative_forward_slash(self) -> None:
        enumeration = metrics.resolve_cluster_paths(_REPO_ROOT)
        for path in enumeration.resolved:
            assert not path.startswith('/'), path
            assert '\\' not in path, path

    def test_glob_expanding_to_zero_keeps_the_enumeration_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This is orchestrator/src/orchestrator/merge_lane/** today: the package
        # arrives in PRD task zeta1. A glob that matches nothing is fine; a
        # LITERAL that matches nothing is the finding (next test).
        (tmp_path / 'a.py').write_text('x = 1\n', encoding='utf-8')
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('a.py', 'nowhere/**/*.py'))
        enumeration = metrics.resolve_cluster_paths(tmp_path)
        assert enumeration.complete is True
        assert enumeration.resolved == ('a.py',)

    def test_missing_literal_path_raises_naming_the_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('gone/missing.py',))
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.resolve_cluster_paths(tmp_path)
        assert 'gone/missing.py' in str(excinfo.value)

    def test_missing_literal_path_never_returns_a_partial_enumeration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The INV-11 seam: a present sibling must not let the missing path be
        # silently dropped from an otherwise plausible-looking result.
        (tmp_path / 'a.py').write_text('x = 1\n', encoding='utf-8')
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('a.py', 'gone/missing.py'))
        with pytest.raises(metrics.MetricsError):
            metrics.resolve_cluster_paths(tmp_path)

    def test_literal_path_that_is_a_directory_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'adir.py').mkdir()
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('adir.py',))
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.resolve_cluster_paths(tmp_path)
        assert 'adir.py' in str(excinfo.value)


class TestEnumerationRoundTrip:
    def test_to_dict_carries_the_four_completeness_keys(self) -> None:
        enumeration = metrics.Enumeration(
            requested=('a.py', 'b/**/*.py'),
            resolved=('a.py',),
            unreadable=(),
            complete=True,
        )
        assert enumeration.to_dict() == {
            'requested': ['a.py', 'b/**/*.py'],
            'resolved': ['a.py'],
            'unreadable': [],
            'complete': True,
        }

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        enumeration = metrics.Enumeration(
            requested=('a.py',), resolved=('a.py',), unreadable=('c.py',), complete=False
        )
        assert json.loads(json.dumps(enumeration.to_dict()))['complete'] is False

    def test_enumeration_is_frozen(self) -> None:
        import dataclasses

        enumeration = metrics.Enumeration(
            requested=(), resolved=(), unreadable=(), complete=True
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            enumeration.complete = False  # type: ignore[misc]

    def test_unreadable_entries_make_the_enumeration_incomplete(self) -> None:
        # `complete` is the single legible signal a caller checks; it must
        # never read True while files were skipped.
        enumeration = metrics.Enumeration(
            requested=('a.py',), resolved=(), unreadable=('a.py',), complete=False
        )
        assert enumeration.complete is False
        assert 'a.py' in enumeration.to_dict()['unreadable']


# ---------------------------------------------------------------------------
# file_size_measures -- lines and prose lines.
#
# Pinned against INLINE synthetic snippets, never against a real repo file, so
# these tests cannot drift when the cluster changes. The one real-tree
# assertion below is an explicit anti-vacuity anchor, not a definition.


class TestFileSizeMeasures:
    def test_lines_counts_physical_lines(self) -> None:
        source = 'a = 1\nb = 2\nc = 3\n'
        assert metrics.file_size_measures(source, path='t.py').lines == 3

    def test_lines_counts_a_final_line_without_a_trailing_newline(self) -> None:
        assert metrics.file_size_measures('a = 1\nb = 2', path='t.py').lines == 2

    def test_blank_lines_are_lines_but_not_prose(self) -> None:
        measures = metrics.file_size_measures('a = 1\n\n\nb = 2\n', path='t.py')
        assert measures.lines == 4
        assert measures.prose_lines == 0

    def test_module_docstring_counts_as_prose(self) -> None:
        measures = metrics.file_size_measures('"""Doc."""\na = 1\n', path='t.py')
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
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 2

    def test_async_function_docstring_counts_as_prose(self) -> None:
        source = 'async def f():\n    """Doc."""\n    return 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_multiline_docstring_counts_once_per_line_it_spans(self) -> None:
        source = '"""Line one.\n\nLine three.\n"""\na = 1\n'
        measures = metrics.file_size_measures(source, path='t.py')
        assert measures.lines == 5
        assert measures.prose_lines == 4

    def test_standalone_comment_lines_count_as_prose(self) -> None:
        source = '# one\n# two\na = 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 2

    def test_trailing_inline_comment_makes_its_code_line_prose(self) -> None:
        source = 'a = 1  # why\nb = 2\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_a_line_that_is_both_docstring_and_comment_counts_once(self) -> None:
        # The two line-number sets are unioned, not summed.
        source = '"""Doc."""  # trailing\na = 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_a_string_literal_mentioning_hash_is_not_a_comment(self) -> None:
        # THE tokenize-vs-regex discriminator: no regex over source text gets
        # this right, and getting it wrong would inflate prose_lines on any file
        # that formats a '#'-bearing string.
        source = "url = 'http://x/#frag'\nheading = '# not a comment'\n"
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 0

    def test_a_non_docstring_string_expression_is_not_prose(self) -> None:
        # Only the FIRST body element of a module/class/function is a docstring.
        source = '"""Doc."""\n"""Not a docstring."""\na = 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.file_size_measures('def (:\n', path='broken.py')
        message = str(excinfo.value)
        assert 'broken.py' in message
        assert 'SyntaxError' in message

    def test_unparseable_source_never_returns_a_zero_measure(self) -> None:
        # INV-11: an unmeasurable cluster file is the finding, not a 0 that
        # silently satisfies every ratchet comparison.
        with pytest.raises(metrics.MetricsError):
            metrics.file_size_measures('class ???:\n', path='broken.py')

    def test_measures_are_frozen(self) -> None:
        import dataclasses

        measures = metrics.file_size_measures('a = 1\n', path='t.py')
        with pytest.raises(dataclasses.FrozenInstanceError):
            measures.lines = 99  # type: ignore[misc]

    def test_real_merge_queue_line_count_anchor(self) -> None:
        # Anti-vacuity anchor: the PRD Background table's 21,550. Not a
        # definition of the measure -- if this drifts, merge_queue.py changed
        # (which the ratchet itself will report against the baseline).
        source = (_REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py').read_text(
            encoding='utf-8'
        )
        measures = metrics.file_size_measures(source, path='merge_queue.py')
        assert measures.lines == 21550
        assert measures.prose_lines > 0


# ---------------------------------------------------------------------------
# Structural import measures: function-local (reach-back) imports, and
# re-export shim names.


class TestFunctionLocalImports:
    def test_import_from_inside_a_def_is_counted(self) -> None:
        source = 'def f():\n    from x import y\n    return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_plain_import_inside_a_def_is_counted(self) -> None:
        source = 'def f():\n    import x\n    return x\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_import_inside_an_async_def_is_counted(self) -> None:
        source = 'async def f():\n    from x import y\n    return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

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
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_module_level_import_is_not_counted(self) -> None:
        source = 'from x import y\nimport z\n\n\ndef f():\n    return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 0

    def test_import_in_a_class_body_outside_any_function_is_not_counted(self) -> None:
        source = 'class C:\n    from x import y\n'
        assert metrics.function_local_imports(source, path='t.py') == 0

    def test_import_in_a_method_body_is_counted(self) -> None:
        source = 'class C:\n    def m(self):\n        from x import y\n        return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_a_docstring_quoting_an_import_is_not_counted(self) -> None:
        # AST, not regex. The satellite modules' reach-back notes quote exactly
        # this string in prose; counting them would inflate the measure.
        source = (
            'def f():\n'
            '    """Resolves via ``from orchestrator.merge_queue import X``."""\n'
            '    return 1\n'
        )
        assert metrics.function_local_imports(source, path='t.py') == 0

    def test_each_import_statement_counts_once_regardless_of_names(self) -> None:
        source = 'def f():\n    from x import a, b, c\n    return a\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.function_local_imports('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)

    def test_cluster_total_is_positive_anchor(self) -> None:
        # Anti-vacuity: the PRD's ceilings row counts the function-local imports
        # inside the package; on this tree the count is well above zero.
        total = 0
        for relpath in metrics.resolve_cluster_paths(_REPO_ROOT).resolved:
            source = (_REPO_ROOT / relpath).read_text(encoding='utf-8')
            total += metrics.function_local_imports(source, path=relpath)
        assert total > 0


class TestReexportNames:
    def test_unreferenced_module_level_import_from_names_are_reexports(self) -> None:
        source = 'from a import (B, C)\n'
        assert metrics.reexport_names(source, path='t.py') == ['B', 'C']

    def test_a_referenced_binding_is_not_a_reexport(self) -> None:
        source = 'from a import B\n\nx = B()\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_an_alias_is_reported_under_the_bound_name(self) -> None:
        source = 'from a import B as C\n'
        assert metrics.reexport_names(source, path='t.py') == ['C']

    def test_a_used_alias_is_not_a_reexport(self) -> None:
        source = 'from a import B as C\n\nx = C()\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_name_referenced_only_in_a_docstring_still_counts(self) -> None:
        # AST, not regex: prose mentioning the name does not make it used.
        source = '"""Re-exports B for consumers."""\nfrom a import B\n# B lives here\n'
        assert metrics.reexport_names(source, path='t.py') == ['B']

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
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_function_local_import_from_is_not_a_reexport(self) -> None:
        # Only MODULE-LEVEL ImportFrom bindings form the module's public
        # surface; a deferred import inside a function is the reach-back
        # measure's business, not this one's.
        source = 'def f():\n    from a import B\n    return 1\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_plain_import_is_not_counted(self) -> None:
        # The PRD's measure is the `from X import (...)` shim block; a bare
        # `import x` binds a module, not a re-exported name.
        source = 'import a\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_star_import_is_not_reported_as_a_name(self) -> None:
        source = 'from a import *\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_names_are_returned_sorted_and_deduped(self) -> None:
        source = 'from a import Z\nfrom b import A\n'
        assert metrics.reexport_names(source, path='t.py') == ['A', 'Z']

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.reexport_names('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)

    def test_merge_queue_structural_reading_overlaps_the_annotated_shims(self) -> None:
        # Anti-vacuity anchor AND the evidence for the STRUCTURAL-not-comment
        # decision: the structural predicate (a module-level ImportFrom binding
        # never referenced elsewhere -- exactly what ruff's F401 computes, which
        # is why those blocks carry the suppression) must land squarely on the
        # nine annotated `# noqa: F401  re-export shim` blocks at lines
        # 57/64/89/112/118/150/154/188/198.
        #
        # OVERLAP, not containment, and the asymmetry is the interesting part:
        # `# noqa: F401` suppresses a whole BLOCK, so a name that merge_queue.py
        # both re-exports AND uses internally sits inside an annotated block
        # while being perfectly F401-clean. MEASURED on this tree: 127 names
        # across the nine blocks, 63 of them structurally unused. So the
        # structural set is a proper subset per block (8 of the 9 blocks
        # contribute at least one; merge_speculation_controller's two names are
        # both used internally), and a proper SUPERSET cluster-wide, because it
        # also catches pure re-exports nobody annotated.
        import ast as _ast

        path = _REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py'
        source = path.read_text(encoding='utf-8')
        reported = set(metrics.reexport_names(source, path=str(path)))
        assert reported

        annotated_linenos = {57, 64, 89, 112, 118, 150, 154, 188, 198}
        tree = _ast.parse(source)
        blocks = [
            {a.asname or a.name for a in node.names}
            for node in tree.body
            if isinstance(node, _ast.ImportFrom) and node.lineno in annotated_linenos
        ]
        assert len(blocks) == len(annotated_linenos), 'shim block line numbers drifted'
        hit_blocks = [names for names in blocks if names & reported]
        assert len(hit_blocks) >= 6, f'{len(hit_blocks)}/{len(blocks)} annotated blocks hit'
        annotated_names = set().union(*blocks)
        assert len(annotated_names & reported) >= 50

    def test_deleting_the_noqa_comment_does_not_change_the_measure(self) -> None:
        # THE ungameable property, pinned directly. A comment-scanning detector
        # would zero out on this purely cosmetic edit; the structural one cannot
        # see comments at all.
        source = 'from a import (  # noqa: F401  re-export shim\n    B,\n    C,\n)\n'
        stripped = source.replace('  # noqa: F401  re-export shim', '')
        assert metrics.reexport_names(source, path='t.py') == ['B', 'C']
        assert metrics.reexport_names(stripped, path='t.py') == ['B', 'C']


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
        scores = metrics.cognitive_complexity(target)
        assert scores == {'f': 6, 'C::m': 0}

    def test_module_with_no_functions_returns_an_empty_map(self, tmp_path: Path) -> None:
        target = tmp_path / 'empty.py'
        target.write_text('X = 1\n', encoding='utf-8')
        assert metrics.cognitive_complexity(target) == {}

    def test_file_total_is_reported_separately(self, tmp_path: Path) -> None:
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        assert metrics.file_cognitive_total(target) == 6

    def test_missing_complexipy_raises_naming_the_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Never a skipped measure and never a 0: a tool the instrument cannot
        # run is an instrument failure with a named cause (INV-11).
        monkeypatch.setitem(sys.modules, 'complexipy', None)
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.cognitive_complexity(target)
        message = str(excinfo.value)
        assert 'complexipy' in message
        assert 'dev' in message

    def test_unparseable_file_raises_naming_the_path(self, tmp_path: Path) -> None:
        target = tmp_path / 'broken.py'
        target.write_text('def (:\n', encoding='utf-8')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.cognitive_complexity(target)
        assert 'broken.py' in str(excinfo.value)

    def test_merge_queue_anchor_reproduces_the_prd_background_numbers(self) -> None:
        # Anti-vacuity anchor. These are the exact figures the PRD Background
        # table quotes. complexipy majors 3/4/5 compute DIFFERENT numbers for
        # this same file (2031 / 2124 / 2092), so a silent algorithm change in a
        # future release is caught here by a named failure rather than by every
        # baseline number quietly shifting underneath the ratchet.
        target = _REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py'
        scores = metrics.cognitive_complexity(target)
        assert scores['SpeculativeMergeWorker::_verifier_loop'] == 245
        assert scores['SpeculativeMergeWorker::stop'] == 109
        assert metrics.file_cognitive_total(target) == 2133


class TestComplexipyVersionContract:
    def test_required_specifier_is_the_measured_range(self) -> None:
        assert metrics.COMPLEXIPY_REQUIRED == '>=6.2,<7'

    def test_installed_version_satisfies_the_requirement(self) -> None:
        # The executable form of the measured pin.
        metrics.require_complexipy()
        assert metrics.satisfies_complexipy_requirement(metrics.complexipy_version())

    def test_pyproject_pin_matches_the_scripts_requirement(self) -> None:
        # Goes RED the moment someone relaxes the dependency to 7.x. The pin and
        # the runtime check are two halves of one contract; letting them drift
        # would leave the ratchet silently measuring with the wrong engine.
        pyproject = (_REPO_ROOT / 'orchestrator/pyproject.toml').read_text(encoding='utf-8')
        assert f'"complexipy{metrics.COMPLEXIPY_REQUIRED}"' in pyproject

    def test_version_out_of_range_raises_naming_both_versions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metrics, 'complexipy_version', lambda: '7.0.1')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.require_complexipy()
        message = str(excinfo.value)
        assert '7.0.1' in message
        assert metrics.COMPLEXIPY_REQUIRED in message
        # The measured reason travels with the failure so the next reader does
        # not have to re-derive it.
        assert '247' in message
        assert '4.75' in message

    def test_version_below_the_floor_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(metrics, 'complexipy_version', lambda: '6.1.0')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.require_complexipy()
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
        assert metrics.satisfies_complexipy_requirement(version) is ok

    def test_missing_complexipy_distribution_raises_naming_the_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.metadata

        def _boom(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, 'version', _boom)
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.complexipy_version()
        assert 'complexipy' in str(excinfo.value)


class TestMaintainabilityIndex:
    def test_returns_a_float_for_a_tiny_module(self) -> None:
        value = metrics.maintainability_index(_TINY_SOURCE, path='tiny.py')
        assert isinstance(value, float)
        assert 0.0 <= value <= 100.0

    def test_merge_queue_anchor_is_zero(self) -> None:
        # The PRD Background table's "Maintainability index (radon) | 0" row.
        # Reported, never ratcheted -- but genuinely exercised, so the `radon`
        # dev-group entry is not dead weight.
        source = (_REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py').read_text(
            encoding='utf-8'
        )
        assert metrics.maintainability_index(source, path='merge_queue.py') == 0.0

    def test_missing_radon_raises_naming_the_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, 'radon.metrics', None)
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.maintainability_index(_TINY_SOURCE, path='tiny.py')
        assert 'radon' in str(excinfo.value)

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.maintainability_index('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Test-suite patch targets into lane internals.
#
# The two detector shapes are ported from
# test_merge_queue_reachback_patch_guard.py -- this measure subsumes that
# guard's allowlist as a COUNT, so PRD task delta can delete the guard once the
# count reaches zero. Two deliberate generalisations there: the prefix set is
# `orchestrator.merge_queue.` AND `orchestrator.merge_lane.`, and the guard's
# `forbidden` filter is dropped so all leaves count, not only the private ones.


class TestPatchTargets:
    def test_string_path_patch_is_detected(self) -> None:
        source = "patch('orchestrator.merge_queue.run_scoped_verification', x)\n"
        assert metrics.patch_targets(source) == {'run_scoped_verification'}

    def test_dotted_patch_is_detected(self) -> None:
        source = "mock.patch('orchestrator.merge_queue.advance_main', x)\n"
        assert metrics.patch_targets(source) == {'advance_main'}

    def test_string_path_setattr_is_detected(self) -> None:
        source = "monkeypatch.setattr('orchestrator.merge_queue.foo', x)\n"
        assert metrics.patch_targets(source) == {'foo'}

    def test_merge_lane_prefix_is_measured_too(self) -> None:
        # Measured before the package exists, deliberately: without it, PRD task
        # zeta2's `git mv` would move every patch target out from under the
        # measure and the count would read 0 by RELOCATION rather than by the
        # migration gamma1..gamma10 actually performs.
        source = "monkeypatch.setattr('orchestrator.merge_lane.foo', x)\n"
        assert metrics.patch_targets(source) == {'foo'}

    def test_merge_lane_submodule_path_is_measured(self) -> None:
        source = "patch('orchestrator.merge_lane.worker.spin', x)\n"
        assert metrics.patch_targets(source) == {'worker.spin'}

    def test_object_path_setattr_via_from_import_alias(self) -> None:
        source = (
            'from orchestrator import merge_queue\n'
            '\n'
            '\n'
            'def t(monkeypatch):\n'
            "    monkeypatch.setattr(merge_queue, 'x', 1)\n"
        )
        assert metrics.patch_targets(source) == {'x'}

    def test_object_path_setattr_via_import_as_alias(self) -> None:
        source = (
            'import orchestrator.merge_queue as mq\n'
            '\n'
            '\n'
            'def t(monkeypatch):\n'
            "    monkeypatch.setattr(mq, 'y', 1)\n"
        )
        assert metrics.patch_targets(source) == {'y'}

    def test_patch_object_on_the_bare_attribute_chain(self) -> None:
        source = "patch.object(orchestrator.merge_queue, 'x', 1)\n"
        assert metrics.patch_targets(source) == {'x'}

    def test_patch_object_on_a_merge_lane_alias(self) -> None:
        source = (
            'from orchestrator import merge_lane\n'
            '\n'
            '\n'
            "p = patch.object(merge_lane, 'z', 1)\n"
        )
        assert metrics.patch_targets(source) == {'z'}

    def test_patch_object_on_a_satellite_is_not_counted(self) -> None:
        # Already repointed to the defining satellite -- that is the END STATE
        # this measure is driving towards, so counting it would penalise the fix.
        source = (
            'from orchestrator import merge_gates\n'
            '\n'
            '\n'
            "p = patch.object(merge_gates, 'x', 1)\n"
        )
        assert metrics.patch_targets(source) == set()

    def test_an_unrelated_attribute_named_merge_queue_is_not_counted(self) -> None:
        source = "patch.object(workflow.merge_queue, 'x', 1)\n"
        assert metrics.patch_targets(source) == set()

    def test_a_docstring_quoting_the_dotted_path_is_not_counted(self) -> None:
        source = '"""Patches orchestrator.merge_queue.foo at the lookup site."""\n'
        assert metrics.patch_targets(source) == set()

    def test_a_comment_quoting_the_dotted_path_is_not_counted(self) -> None:
        source = "# patch('orchestrator.merge_queue.foo')\nx = 1\n"
        assert metrics.patch_targets(source) == set()

    def test_distinct_names_not_call_sites(self) -> None:
        source = (
            "patch('orchestrator.merge_queue.foo', a)\n"
            "patch('orchestrator.merge_queue.foo', b)\n"
            "patch('orchestrator.merge_queue.bar', c)\n"
        )
        assert metrics.patch_targets(source) == {'foo', 'bar'}

    def test_a_bare_module_patch_with_no_leaf_is_not_counted(self) -> None:
        source = "patch('orchestrator.merge_queue', x)\n"
        assert metrics.patch_targets(source) == set()

    def test_a_non_string_second_arg_is_not_counted(self) -> None:
        source = (
            'from orchestrator import merge_queue\n'
            '\n'
            '\n'
            'p = patch.object(merge_queue, NAME, 1)\n'
        )
        assert metrics.patch_targets(source) == set()

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.patch_targets('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)

    def test_real_tree_union_anchor(self) -> None:
        # Anti-vacuity: the PRD Background table's "79 distinct names"; the
        # string-path form alone measures 78 on this tree.
        union: set[str] = set()
        for path in sorted((_REPO_ROOT / 'orchestrator' / 'tests').rglob('*.py')):
            try:
                source = path.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                continue
            try:
                union |= metrics.patch_targets(source, path=str(path))
            except metrics.MetricsError:
                continue
        assert len(union) >= 70, len(union)
        assert 'run_scoped_verification' in union
