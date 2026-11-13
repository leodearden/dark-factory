import ast

import pytest
from _leaf_import_guard import imported_modules, stdlib_or_allowed

ALLOWED = frozenset({'fused_memory.utils.validation'})


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param('import json, os.path', {'json', 'os.path'}, id='plain'),
        pytest.param('from fused_memory.utils import validation', {'fused_memory.utils'}, id='from'),
        pytest.param('from . import durable_queue', {'.durable_queue'}, id='relative-sibling'),
        pytest.param('from ..utils import validation', {'..utils'}, id='relative-parent'),
    ],
)
def test_imported_modules_reports_every_import(source, expected):
    assert imported_modules(ast.parse(source)) == expected


@pytest.mark.parametrize('module', ['json', 'os.path', '__future__', 'fused_memory.utils.validation'])
def test_the_stdlib_and_the_allowed_set_pass(module):
    assert stdlib_or_allowed(module, ALLOWED)


@pytest.mark.parametrize('module', ['fused_memory.services', '.durable_queue', '..utils', 'aiosqlite'])
def test_anything_else_fails(module):
    assert not stdlib_or_allowed(module, ALLOWED)
