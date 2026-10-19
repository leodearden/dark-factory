"""Contract of shared/src/shared/defect_enumerator.py, driven through its public API only."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import MutableMapping
from pathlib import Path

import pytest

from shared.defect_enumerator import (
    Carrier,
    DisablingMark,
    EnumerationGap,
    FileEnumeration,
    GapReason,
    NonStaticMark,
    TreeEnumeration,
    enumerate_source,
    enumerate_tree,
    is_test_file,
    main,
)
from shared.defects import MalformedDefectRef, NodeId, mint_explore_ref

REF_A = mint_explore_ref('run1', 1)
REF_B = mint_explore_ref('run1', 2)

PATH = 'pkg/tests/test_x.py'


def _enumerate(source: str, *, path: str = PATH, module_root: str = 'pkg') -> FileEnumeration:
    return enumerate_source(path, textwrap.dedent(source).removeprefix('\n'), module_root=module_root)


def _node(name: str) -> NodeId:
    return NodeId(module_root='pkg', path='tests/test_x.py', name=name)


def _runnable(name: str) -> Carrier:
    return Carrier(_node(name), frozenset())


def test_a_marked_module_level_test_function_is_a_carrier():
    result = _enumerate(f'''
        import pytest

        @pytest.mark.defect({REF_A!r})
        def test_a():
            pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_a'),)}
    assert result.complete is True
    assert result.non_static == ()
    assert result.gaps == ()


def test_module_root_dot_keeps_the_whole_tree_relative_path():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        def test_a():
            pass
    ''', module_root='.')
    node = NodeId(module_root='.', path=PATH, name='test_a')
    assert result.carriers == {REF_A: (Carrier(node, frozenset()),)}


def test_an_async_test_function_is_a_carrier():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        async def test_b():
            pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_b'),)}


def test_stacked_defect_marks_give_every_ref():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        @pytest.mark.defect({REF_B!r})
        def test_a():
            pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_a'),), REF_B: (_runnable('test_a'),)}


def test_the_same_ref_twice_on_one_function_is_one_carrier():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        @pytest.mark.defect({REF_A!r})
        def test_a():
            pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_a'),)}


def test_a_mark_on_a_function_pytest_does_not_collect_is_not_a_carrier():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        def helper():
            pass
    ''')
    assert REF_A not in result.carriers
    assert result.non_static == (NonStaticMark(REF_A, 1),)


def test_bytes_source_is_read_including_a_pep_263_coding_cookie():
    source = (
        b'# -*- coding: latin-1 -*-\n'
        b'import pytest\n'
        b'\n'
        b'@pytest.mark.defect(' + repr(REF_A).encode() + b')\n'
        b'def test_a():\n'
        b'    assert "\xe9"\n'
    )
    result = enumerate_source(PATH, source, module_root='pkg')
    assert result.carriers == {REF_A: (_runnable('test_a'),)}
    assert result.complete is True


@pytest.mark.parametrize('source', [
    'import pytest\n\ndef test_a(:\n    pass\n',
    b'x = "\xff"\n',
    b'x = 1\x00\n',
], ids=['syntax-error', 'undecodable-bytes', 'nul-byte'])
def test_an_unparseable_source_is_one_syntax_error_gap(source):
    result = enumerate_source(PATH, source, module_root='pkg')
    assert result.complete is False
    assert result.carriers == {}
    assert [gap.reason for gap in result.gaps] == [GapReason.SYNTAX_ERROR]


def test_a_syntax_error_gap_carries_the_parsers_line():
    result = enumerate_source(PATH, 'import pytest\n\ndef test_a(:\n    pass\n', module_root='pkg')
    assert result.gaps == (EnumerationGap(GapReason.SYNTAX_ERROR, 3),)


@pytest.mark.parametrize(('path', 'module_root'), [
    ('other/test_x.py', 'pkg'),
    ('pkgx/test_x.py', 'pkg'),
    ('pkg', 'pkg'),
    ('/pkg/test_x.py', '.'),
    ('./pkg/test_x.py', '.'),
    ('pkg//test_x.py', '.'),
    ('../x/test_y.py', '.'),
    ('', '.'),
    (PATH, 'pkg/'),
    (PATH, '..'),
    (PATH, 'pkg/..'),
    (PATH, ''),
])
def test_a_path_outside_its_module_root_or_not_normalised_is_refused(path, module_root):
    with pytest.raises(ValueError) as excinfo:
        enumerate_source(path, 'x = 1\n', module_root=module_root)
    message = str(excinfo.value)
    assert repr(path) in message
    assert repr(module_root) in message


@pytest.mark.parametrize('mark', [
    'pytest.mark.defect',
    'pytest.mark.defect()',
    f'pytest.mark.defect({REF_A!r}, {REF_B!r})',
    f'pytest.mark.defect(ref={REF_A!r})',
    'pytest.mark.defect(SOME_NAME)',
    "pytest.mark.defect(f'explore:run1:{N}')",
    'pytest.mark.defect(*refs)',
])
def test_a_defect_mark_whose_ref_is_not_one_positional_literal_is_an_unreadable_ref_gap(mark):
    result = _enumerate(f'''
        import pytest

        @{mark}
        def test_a():
            pass
    ''')
    assert result.complete is False
    assert result.gaps == (EnumerationGap(GapReason.UNREADABLE_REF, 3),)


def test_a_literal_outside_the_ref_grammar_is_a_malformed_ref_gap():
    result = _enumerate('''
        import pytest

        @pytest.mark.defect('not-a-ref')
        def test_a():
            pass
    ''')
    assert result.complete is False
    assert result.gaps == (EnumerationGap(GapReason.MALFORMED_REF, 3),)


def test_readable_carriers_survive_beside_an_unreadable_mark():
    result = _enumerate(f'''
        @pytest.mark.defect(SOME_NAME)
        def test_a():
            pass

        @pytest.mark.defect({REF_A!r})
        def test_b():
            pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_b'),)}
    assert result.complete is False


def test_a_mark_inside_pytest_param_is_not_static():
    result = _enumerate(f'''
        import pytest

        @pytest.mark.parametrize('x', [pytest.param(1, marks=pytest.mark.defect({REF_A!r}))])
        def test_p(x):
            pass
    ''')
    assert REF_A not in result.carriers
    assert result.non_static == (NonStaticMark(REF_A, 3),)
    assert result.complete is True


def test_a_mark_added_at_runtime_is_not_static():
    result = _enumerate(f'''
        def test_a(request):
            request.node.add_marker(pytest.mark.defect({REF_B!r}))
    ''')
    assert result.carriers == {}
    assert result.non_static == (NonStaticMark(REF_B, 2),)


def test_a_mark_applied_as_a_call_is_not_static():
    result = _enumerate(f'''
        def check():
            pass

        test_applied = pytest.mark.defect({REF_A!r})(check)
    ''')
    assert result.carriers == {}
    assert result.non_static == (NonStaticMark(REF_A, 4),)


@pytest.mark.parametrize('source', [
    '''
        def test_a(request):
            request.node.add_marker(pytest.mark.defect(SOME_NAME))
    ''',
    '''
        x = 1
        MARK = pytest.mark.defect
    ''',
], ids=['add-marker', 'bare-attribute'])
def test_an_unreadable_mark_in_a_non_static_position_still_makes_the_file_incomplete(source):
    result = _enumerate(source)
    assert result.complete is False
    assert result.gaps == (EnumerationGap(GapReason.UNREADABLE_REF, 2),)
    assert result.non_static == ()


def test_non_static_marks_and_gaps_are_ordered_by_line():
    result = _enumerate(f'''
        def helper(request):
            request.node.add_marker(pytest.mark.defect({REF_B!r}))
            request.node.add_marker(pytest.mark.defect(LATER))

        MARK = pytest.mark.defect({REF_A!r})
        OTHER = pytest.mark.defect(EARLIER)
    ''')
    assert result.non_static == (NonStaticMark(REF_B, 2), NonStaticMark(REF_A, 5))
    assert result.gaps == (
        EnumerationGap(GapReason.UNREADABLE_REF, 3),
        EnumerationGap(GapReason.UNREADABLE_REF, 6),
    )


def test_a_test_class_decorator_marks_every_test_method():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        class TestA:
            def test_m(self):
                pass

            def test_n(self):
                pass

            def helper(self):
                pass
    ''')
    assert result.carriers == {REF_A: (_runnable('TestA::test_m'), _runnable('TestA::test_n'))}
    assert result.non_static == ()


def test_a_method_decorator_marks_only_that_method():
    result = _enumerate(f'''
        class TestA:
            @pytest.mark.defect({REF_B!r})
            def test_m(self):
                pass

            def test_n(self):
                pass
    ''')
    assert result.carriers == {REF_B: (_runnable('TestA::test_m'),)}


def test_a_nested_test_class_inherits_its_outer_class_marks():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        class TestOuter:
            class TestInner:
                def test_m(self):
                    pass
    ''')
    assert result.carriers == {REF_A: (_runnable('TestOuter::TestInner::test_m'),)}


def test_a_class_body_pytestmark_marks_every_test_method():
    result = _enumerate(f'''
        class TestA:
            pytestmark = pytest.mark.defect({REF_A!r})

            def test_m(self):
                pass

            def test_n(self):
                pass
    ''')
    assert result.carriers == {REF_A: (_runnable('TestA::test_m'), _runnable('TestA::test_n'))}


def test_a_module_pytestmark_marks_every_function_and_test_class_method():
    result = _enumerate(f'''
        pytestmark = [pytest.mark.defect({REF_A!r})]

        def test_a():
            pass

        class TestA:
            def test_m(self):
                pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_a'), _runnable('TestA::test_m'))}


def test_only_the_last_module_pytestmark_binding_carries():
    result = _enumerate(f'''
        pytestmark = pytest.mark.defect({REF_A!r})
        pytestmark = pytest.mark.defect({REF_B!r})

        def test_a():
            pass
    ''')
    assert result.carriers == {REF_B: (_runnable('test_a'),)}
    assert result.non_static == (NonStaticMark(REF_A, 1),)


@pytest.mark.parametrize('class_name', ['Helper', 'testLower'])
def test_a_class_pytest_does_not_collect_carries_nothing(class_name):
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        class {class_name}:
            def test_m(self):
                pass
    ''')
    assert result.carriers == {}
    assert result.non_static == (NonStaticMark(REF_A, 1),)


@pytest.mark.parametrize('constructor', ['__init__', '__new__'])
def test_a_test_class_with_its_own_constructor_carries_nothing(constructor):
    result = _enumerate(f'''
        class TestA:
            def {constructor}(self):
                pass

            @pytest.mark.defect({REF_A!r})
            def test_m(self):
                pass
    ''')
    assert result.carriers == {}
    assert result.non_static == (NonStaticMark(REF_A, 5),)


def test_a_test_function_inside_an_if_carries_nothing():
    result = _enumerate(f'''
        if True:
            @pytest.mark.defect({REF_A!r})
            def test_a():
                pass
    ''')
    assert result.carriers == {}
    assert result.non_static == (NonStaticMark(REF_A, 2),)


def test_only_the_last_definition_of_a_test_name_carries():
    result = _enumerate(f'''
        @pytest.mark.defect({REF_A!r})
        def test_a():
            pass

        @pytest.mark.defect({REF_B!r})
        def test_a():
            pass
    ''')
    assert result.carriers == {REF_B: (_runnable('test_a'),)}
    assert result.non_static == (NonStaticMark(REF_A, 1),)


_DISABLING_SPELLINGS = [
    ('pytest.mark.skip', DisablingMark.SKIP),
    ("pytest.mark.skip(reason='x')", DisablingMark.SKIP),
    ("pytest.mark.skipif(sys.platform == 'win32', reason='x')", DisablingMark.SKIPIF),
    ('pytest.mark.xfail', DisablingMark.XFAIL),
    ('pytest.mark.xfail(strict=True)', DisablingMark.XFAIL),
]

_DISABLING_SCOPES = {
    'function-decorator': ("""
        @{mark}
        @pytest.mark.defect({ref})
        def test_a():
            pass
    """, 'test_a'),
    'class-decorator': ("""
        @{mark}
        class TestA:
            @pytest.mark.defect({ref})
            def test_m(self):
                pass
    """, 'TestA::test_m'),
    'class-pytestmark': ("""
        class TestA:
            pytestmark = {mark}

            @pytest.mark.defect({ref})
            def test_m(self):
                pass
    """, 'TestA::test_m'),
    'module-pytestmark': ("""
        pytestmark = [{mark}]

        @pytest.mark.defect({ref})
        def test_a():
            pass
    """, 'test_a'),
}


@pytest.mark.parametrize(('mark', 'disabling'), _DISABLING_SPELLINGS)
@pytest.mark.parametrize('scope', sorted(_DISABLING_SCOPES))
def test_a_skip_or_xfail_mark_reaching_a_carrier_disables_it(mark, disabling, scope):
    template, name = _DISABLING_SCOPES[scope]
    result = _enumerate(template.format(mark=mark, ref=repr(REF_A)))
    (carrier,) = result.carriers[REF_A]
    assert carrier == Carrier(_node(name), frozenset({disabling}))
    assert carrier.runnable is False


def test_disabling_marks_from_every_scope_accumulate():
    result = _enumerate(f'''
        @pytest.mark.skip
        class TestA:
            @pytest.mark.xfail
            @pytest.mark.defect({REF_A!r})
            def test_m(self):
                pass
    ''')
    assert result.carriers == {
        REF_A: (Carrier(_node('TestA::test_m'), frozenset({DisablingMark.SKIP, DisablingMark.XFAIL})),),
    }


@pytest.mark.parametrize('mark', [
    'pytest.mark.slow',
    "pytest.mark.parametrize('x', [1, 2])",
    'pytest.mark.timeout(5)',
])
def test_an_unrelated_mark_leaves_the_carrier_runnable(mark):
    result = _enumerate(f'''
        @{mark}
        @pytest.mark.defect({REF_A!r})
        def test_a(x):
            pass
    ''')
    (carrier,) = result.carriers[REF_A]
    assert carrier.runnable is True


def test_an_xfail_inside_pytest_param_does_not_disable_the_function_carrier():
    result = _enumerate(f'''
        @pytest.mark.parametrize('x', [1, pytest.param(2, marks=pytest.mark.xfail)])
        @pytest.mark.defect({REF_A!r})
        def test_a(x):
            pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_a'),)}


def test_a_skip_on_a_sibling_test_does_not_leak():
    result = _enumerate(f'''
        @pytest.mark.skip
        def test_skipped():
            pass

        @pytest.mark.defect({REF_A!r})
        def test_a():
            pass
    ''')
    assert result.carriers == {REF_A: (_runnable('test_a'),)}


_MARKED = f'''
    import pytest

    @pytest.mark.defect({REF_A!r})
    def test_a():
        pass
'''


def _file(path: str, source: str) -> FileEnumeration:
    return enumerate_source(path, textwrap.dedent(source), module_root='.')


def _at(path: str, name: str) -> NodeId:
    return NodeId(module_root='.', path=path, name=name)


def test_tree_carriers_merge_per_ref_sorted_with_carriers_in_path_then_source_order():
    later = _file('b/test_two.py', f'''
        @pytest.mark.defect({REF_A!r})
        def test_z():
            pass
    ''')
    earlier = _file('a/test_one.py', f'''
        @pytest.mark.defect({REF_B!r})
        def test_y():
            pass

        @pytest.mark.defect({REF_A!r})
        def test_x():
            pass

        @pytest.mark.defect({REF_A!r})
        def test_w():
            pass
    ''')
    tree = TreeEnumeration((later, earlier))
    assert list(tree.carriers) == [REF_A, REF_B]
    assert [carrier.node_id for carrier in tree.carriers[REF_A]] == [
        _at('a/test_one.py', 'test_x'),
        _at('a/test_one.py', 'test_w'),
        _at('b/test_two.py', 'test_z'),
    ]


def test_runnable_refs_need_one_runnable_carrier_anywhere_in_the_tree():
    only_disabled = _file('a/test_one.py', f'''
        @pytest.mark.skip
        @pytest.mark.defect({REF_A!r})
        def test_a():
            pass

        @pytest.mark.xfail
        @pytest.mark.defect({REF_B!r})
        def test_b():
            pass
    ''')
    runnable_elsewhere = _file('b/test_two.py', f'''
        @pytest.mark.defect({REF_B!r})
        def test_b():
            pass
    ''')
    tree = TreeEnumeration((only_disabled, runnable_elsewhere))
    assert tree.runnable_refs == frozenset({REF_B})


def test_tree_non_static_refs_hold_the_pytest_param_ref():
    tree = TreeEnumeration((_file('a/test_one.py', f'''
        @pytest.mark.parametrize('x', [pytest.param(1, marks=pytest.mark.defect({REF_A!r}))])
        def test_p(x):
            pass
    '''),))
    assert tree.non_static_refs == frozenset({REF_A})
    assert tree.carriers == {}


def test_tree_incomplete_files_name_every_incomplete_file():
    complete = _file('a/test_ok.py', _MARKED)
    broken = _file('b/test_broken.py', 'def test_a(:\n')
    assert TreeEnumeration((complete, broken)).incomplete_files == ('b/test_broken.py',)
    assert TreeEnumeration((complete, broken)).complete is False
    assert TreeEnumeration((complete,)).incomplete_files == ()
    assert TreeEnumeration((complete,)).complete is True


def test_a_tree_refuses_two_enumerations_of_one_path():
    with pytest.raises(ValueError, match='a/test_one.py'):
        TreeEnumeration((_file('a/test_one.py', _MARKED), _file('a/test_one.py', _MARKED)))


def test_a_file_enumeration_refuses_a_key_outside_the_ref_grammar():
    with pytest.raises(MalformedDefectRef):
        FileEnumeration('a/test_one.py', {'not-a-ref': ()})


def test_a_file_enumerations_carriers_are_read_only_and_detached_from_the_callers_dict():
    given = dict(_file('a/test_one.py', _MARKED).carriers)
    enumeration = FileEnumeration('a/test_one.py', given)
    given[REF_B] = ()
    assert not isinstance(enumeration.carriers, MutableMapping)
    assert REF_B not in enumeration.carriers


@pytest.mark.parametrize(('path', 'expected'), [
    ('a/test_x.py', True),
    ('a/x_test.py', True),
    ('a/conftest.py', False),
    ('a/helper.py', False),
    ('a/test_x.txt', False),
])
def test_is_test_file_follows_pytests_default_python_files(path, expected):
    assert is_test_file(path) is expected


def _write(root: Path, relative: str, source: str = _MARKED) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(source))


def _dot_root(path: str) -> str:
    return '.'


def test_enumerate_tree_scans_only_test_files(tmp_path):
    for relative in ('pkg/tests/test_x.py', 'pkg/tests/helper.py', 'pkg/conftest.py'):
        _write(tmp_path, relative)
    tree = enumerate_tree(tmp_path, module_root_of=_dot_root)
    assert [file.path for file in tree.files] == ['pkg/tests/test_x.py']


@pytest.mark.parametrize('directory', ['.venv', '.git', 'node_modules', 'build', 'dist', 'venv', 'x.egg'])
def test_enumerate_tree_prunes_pytests_default_norecursedirs_at_any_depth(tmp_path, directory):
    _write(tmp_path, 'pkg/tests/test_x.py')
    _write(tmp_path, f'{directory}/test_in.py')
    _write(tmp_path, f'pkg/{directory}/test_in.py')
    tree = enumerate_tree(tmp_path, module_root_of=_dot_root)
    assert [file.path for file in tree.files] == ['pkg/tests/test_x.py']


def test_enumerate_tree_returns_files_sorted_by_tree_relative_path(tmp_path):
    for relative in ('b/test_b.py', 'a/test_a.py', 'a/z/test_c.py', 'a/test_d.py'):
        _write(tmp_path, relative)
    tree = enumerate_tree(tmp_path, module_root_of=_dot_root)
    assert [file.path for file in tree.files] == ['a/test_a.py', 'a/test_d.py', 'a/z/test_c.py', 'b/test_b.py']


def test_enumerate_tree_qualifies_each_file_by_its_module_root(tmp_path):
    _write(tmp_path, 'pkg/tests/test_x.py')
    _write(tmp_path, 'top/test_y.py')
    tree = enumerate_tree(tmp_path, module_root_of=lambda path: 'pkg' if path.startswith('pkg/') else '.')
    assert [carrier.node_id for carrier in tree.carriers[REF_A]] == [
        NodeId(module_root='pkg', path='tests/test_x.py', name='test_a'),
        NodeId(module_root='.', path='top/test_y.py', name='test_a'),
    ]


def test_an_unreadable_test_file_is_an_unreadable_file_gap_and_the_walk_continues(tmp_path):
    _write(tmp_path, 'a/test_ok.py')
    (tmp_path / 'a' / 'test_gone.py').symlink_to(tmp_path / 'nowhere.py')
    tree = enumerate_tree(tmp_path, module_root_of=_dot_root)
    assert tree.files == (
        FileEnumeration('a/test_gone.py', {}, gaps=(EnumerationGap(GapReason.UNREADABLE_FILE, None),)),
        _file('a/test_ok.py', _MARKED),
    )
    assert tree.incomplete_files == ('a/test_gone.py',)


def test_enumerate_tree_refuses_a_root_it_cannot_list(tmp_path):
    with pytest.raises(FileNotFoundError):
        enumerate_tree(tmp_path / 'missing', module_root_of=_dot_root)


_SHARED_SRC = Path(__file__).resolve().parents[1] / 'src'

_CLI_OK = f'''
    import pytest

    @pytest.mark.defect({REF_A!r})
    def test_a():
        pass

    @pytest.mark.skip
    @pytest.mark.defect({REF_B!r})
    def test_b():
        pass

    @pytest.mark.parametrize('x', [pytest.param(1, marks=pytest.mark.defect({REF_A!r}))])
    def test_p(x):
        pass
'''
_CLI_OK_NON_STATIC_LINE = 12
_CLI_BROKEN = 'import pytest\n\ndef test_a(:\n    pass\n'
_CLI_BROKEN_LINE = 3


def _cli_tree(root: Path, *, broken: bool) -> Path:
    _write(root, 'pkg/tests/test_ok.py', _CLI_OK.removeprefix('\n'))
    if broken:
        _write(root, 'pkg/tests/test_broken.py', _CLI_BROKEN)
    return root


def _block_under(lines: list[str], header_prefix: str) -> list[str]:
    start = next(index for index, line in enumerate(lines) if line.startswith(header_prefix))
    block = []
    for line in lines[start + 1:]:
        if not line.startswith(' '):
            break
        block.append(line)
    return block


def test_the_cli_lists_each_file_its_carriers_and_non_static_marks(tmp_path, capsys):
    rc = main([str(_cli_tree(tmp_path, broken=True))])
    lines = capsys.readouterr().out.splitlines()
    file_lines = [line for line in lines if line.startswith('pkg/')]
    assert len(file_lines) == 2
    assert 'pkg/tests/test_ok.py: complete' in file_lines
    (broken_line,) = [line for line in file_lines if line.startswith('pkg/tests/test_broken.py: incomplete')]
    assert 'syntax_error' in broken_line
    assert str(_CLI_BROKEN_LINE) in broken_line
    block = _block_under(lines, 'pkg/tests/test_ok.py:')
    assert any(REF_A in line and './pkg/tests/test_ok.py::test_a' in line for line in block)
    assert any(REF_B in line and './pkg/tests/test_ok.py::test_b' in line and 'skip' in line for line in block)
    assert any(
        REF_A in line and 'not static' in line and str(_CLI_OK_NON_STATIC_LINE) in line for line in block
    )
    assert '2 files' in lines[-1]
    assert '1 incomplete' in lines[-1]
    assert rc == 1


def test_the_cli_exits_zero_when_every_file_is_complete(tmp_path, capsys):
    assert main([str(_cli_tree(tmp_path, broken=False))]) == 0


def test_the_cli_refuses_a_tree_that_is_not_a_directory(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / 'missing')])
    assert excinfo.value.code == 2


def test_the_cli_runs_as_python_dash_m(tmp_path):
    tree = _cli_tree(tmp_path, broken=False)
    result = subprocess.run(
        [sys.executable, '-m', 'shared.defect_enumerator', str(tree)],
        env={**os.environ, 'PYTHONPATH': str(_SHARED_SRC)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert REF_A in result.stdout
