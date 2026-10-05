"""The two public seams the suite census reuses from scripts/source_measures.py: patch calls and tracked files."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest
from source_measures import (
    MetricsError,
    PatchCall,
    patch_calls_in_tree,
    patch_targets_in_tree,
    tracked_files,
    tracked_python_files,
)
from suite_census_fixtures import git_tree


def _calls(source: str) -> tuple[PatchCall, ...]:
    return patch_calls_in_tree(ast.parse(textwrap.dedent(source)))


def _shape(call: PatchCall) -> tuple[str | None, str | None, str | None]:
    receiver = None if call.receiver is None else ast.unparse(call.receiver)
    return call.target, receiver, call.attribute


class TestStringPathForm:
    @pytest.mark.parametrize(
        'source',
        [
            "patch('a.b._c')",
            "mock.patch('a.b._c')",
            "monkeypatch.setattr('a.b._c', v)",
        ],
    )
    def test_dotted_target_is_the_first_argument(self, source):
        assert [_shape(call) for call in _calls(source)] == [('a.b._c', None, None)]


class TestObjectPathForm:
    @pytest.mark.parametrize(
        ('source', 'receiver', 'attribute'),
        [
            ("patch.object(mod, '_x')", 'mod', '_x'),
            ("mock.patch.object(Cls, '_y')", 'Cls', '_y'),
            ("monkeypatch.setattr(worker, '_z', v)", 'worker', '_z'),
        ],
    )
    def test_receiver_and_attribute_are_set(self, source, receiver, attribute):
        assert [_shape(call) for call in _calls(source)] == [(None, receiver, attribute)]


def test_a_decorator_patch_is_found_from_the_function_node():
    function = ast.parse(
        textwrap.dedent(
            """
            @patch('a._b')
            def test_it():
                pass
            """
        )
    ).body[0]
    assert [_shape(call) for call in patch_calls_in_tree(function)] == [('a._b', None, None)]


@pytest.mark.parametrize(
    'source',
    [
        "setattr(obj, '_x', v)",
        "patch.dict(os.environ, {'A': '1'})",
        'patch()',
        'patch.object(mod, name_var)',
    ],
)
def test_calls_that_are_not_patch_shaped(source):
    assert _calls(source) == ()


def test_order_follows_the_walk_and_duplicates_are_kept():
    calls = _calls(
        """
        patch('a._one')
        patch('a._one')
        patch.object(mod, '_two')
        """
    )
    assert [_shape(call) for call in calls] == [
        ('a._one', None, None),
        ('a._one', None, None),
        (None, 'mod', '_two'),
    ]


def test_patch_targets_pair_each_leaf_with_its_module_over_mixed_patches():
    tree = ast.parse(
        textwrap.dedent(
            """
            import pkg.mod as m
            from pkg import lane

            patch('pkg.mod._run')
            patch.object(m, '_x')
            monkeypatch.setattr(lane, '_y', 1)
            patch('pkg.other._run_cmd')
            """
        )
    )
    assert patch_targets_in_tree(tree, ('pkg.mod', 'pkg.lane')) == {
        ('pkg.mod', '_run'),
        ('pkg.mod', '_x'),
        ('pkg.lane', '_y'),
    }


class TestTrackedFiles:
    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        root = git_tree(
            tmp_path,
            {
                'a.py': '',
                'sub/b.rs': '',
                'sub/c.py': '',
                'z.rs': '',
            },
        )
        (root / 'sub' / 'untracked.rs').write_text('')
        return root

    def test_lists_only_tracked_files_matching_the_pathspecs(self, repo):
        assert tracked_files(repo, '*.rs') == ('sub/b.rs', 'z.rs')

    def test_refuses_a_directory_below_the_top_of_the_work_tree(self, repo):
        with pytest.raises(MetricsError, match='not the top of a git work tree'):
            tracked_files(repo / 'sub')

    def test_tracked_python_files_is_the_py_listing(self, repo):
        assert tracked_python_files(repo) == tracked_files(repo, '*.py')
        assert tracked_python_files(repo) == ('a.py', 'sub/c.py')
