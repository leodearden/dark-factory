"""Contract of shared/src/shared/defect_enumerator.py, driven through its public API only."""

from __future__ import annotations

import textwrap

import pytest

from shared.defect_enumerator import (
    Carrier,
    EnumerationGap,
    FileEnumeration,
    GapReason,
    enumerate_source,
)
from shared.defects import NodeId, mint_explore_ref

REF_A = mint_explore_ref('run1', 1)
REF_B = mint_explore_ref('run1', 2)

PATH = 'pkg/tests/test_x.py'


def _enumerate(source: str, *, path: str = PATH, module_root: str = 'pkg') -> FileEnumeration:
    return enumerate_source(path, textwrap.dedent(source), module_root=module_root)


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
