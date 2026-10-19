"""Contract of shared/src/shared/defect_enumerator.py, driven through its public API only."""

from __future__ import annotations

import textwrap

import pytest

from shared.defect_enumerator import (
    Carrier,
    EnumerationGap,
    FileEnumeration,
    GapReason,
    NonStaticMark,
    enumerate_source,
)
from shared.defects import NodeId, mint_explore_ref

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
