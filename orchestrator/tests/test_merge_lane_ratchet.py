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
