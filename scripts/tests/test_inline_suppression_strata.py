"""The inline-suppression scanner's files import each other in one order only.

:data:`STRATA` is the one home of that order.  Each family file may import a
family module only from an EARLIER entry, and only as a statement of its module
body or directly under a module-level ``if TYPE_CHECKING:``, never inside a
function, class or any other conditional.  So there is no cycle, no reach-back
into the entry module, and no import deferred to dodge either.  A type-only
import is held to the same order as a runtime one: it creates no runtime edge,
but it is still a dependency a reader of the lower file would have to follow
upward.

The live test reads import STRUCTURE through ``ast`` from the real files in
``scripts/``.  That makes this an architecture fitness check on the code, not
a check on anyone's prose about it.  :data:`PLACEMENT_CASES` drives the same
rule over small synthetic sources, so each placement the rule accepts or
refuses is shown to be accepted or refused.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
FAMILY_PREFIX = 'inline_suppression'

#: The scanner's files, lowest stratum first.
STRATA = (
    'inline_suppression_refusal',
    'inline_suppression_kinds',
    'inline_suppression_scan',
    'inline_suppression_key',
    'inline_suppression_consumers',
    'inline_suppression_classify',
    'inline_suppressions',
)


def _family_files() -> list[Path]:
    """Every file in ``scripts/`` that belongs to the scanner's family."""
    return sorted(SCRIPTS_DIR.glob(f'{FAMILY_PREFIX}*.py'))


def _family_imports(tree: ast.Module) -> list[tuple[ast.stmt, str]]:
    """Every import in *tree* that names a family module, with the module it names."""
    found: list[tuple[ast.stmt, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found.extend(
            (node, name.partition('.')[0])
            for name in names
            if name.startswith(FAMILY_PREFIX)
        )
    return found


def _is_type_checking(test: ast.expr) -> bool:
    """Whether *test* is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``."""
    if isinstance(test, ast.Name):
        return test.id == 'TYPE_CHECKING'
    return isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'


def _permitted_import_statements(tree: ast.Module) -> set[int]:
    """The ids of the statements a family import may be.

    Those are the module body's own statements, plus the body (never the
    ``else``) of a module-level ``if TYPE_CHECKING:``.
    """
    allowed = {id(statement) for statement in tree.body}
    for statement in tree.body:
        if isinstance(statement, ast.If) and _is_type_checking(statement.test):
            allowed.update(id(nested) for nested in statement.body)
    return allowed


def _placement_violations(stem: str, source: str) -> list[str]:
    """Every family import in *source*, read as ``scripts/<stem>.py``, that breaks the order."""
    tree = ast.parse(source, filename=f'scripts/{stem}.py')
    allowed = _permitted_import_statements(tree)
    importer = STRATA.index(stem) if stem in STRATA else None
    violations: list[str] = []
    for node, imported in _family_imports(tree):
        where = f'scripts/{stem}.py:{node.lineno} imports {imported}'
        if imported not in STRATA or importer is None:
            violations.append(f'{where}, and one of the two has no place in STRATA')
            continue
        if STRATA.index(imported) >= importer:
            violations.append(
                f'{where} (STRATA index {STRATA.index(imported)}), which is not '
                f'earlier than {stem} (STRATA index {importer})'
            )
        if id(node) not in allowed:
            violations.append(
                f'{where} inside a function, class or conditional; family imports '
                'are module-body statements, or sit directly under a module-level '
                '`if TYPE_CHECKING:` for a type-only dependency'
            )
    return violations


def test_every_family_file_has_a_place_in_the_strata():
    """A new sibling must be placed in the order before it can import or be imported."""
    assert {path.stem for path in _family_files()} == set(STRATA)


def test_each_family_file_imports_only_earlier_strata_at_module_top_level():
    violations = [
        violation
        for path in _family_files()
        for violation in _placement_violations(path.stem, path.read_text(encoding='utf-8'))
    ]

    assert violations == [], '\n'.join(violations)


#: ``(importer stem, source, whether the rule accepts it)``.  A new placement
#: the rule should accept or refuse is covered by adding a ROW.
PLACEMENT_CASES = [
    pytest.param(
        'inline_suppression_scan',
        'from inline_suppression_kinds import Site\n',
        True,
        id='module-body-downward',
    ),
    pytest.param(
        'inline_suppression_scan',
        'from typing import TYPE_CHECKING\n'
        'if TYPE_CHECKING:\n'
        '    from inline_suppression_kinds import Site\n',
        True,
        id='type-checking-downward',
    ),
    pytest.param(
        'inline_suppression_scan',
        'import typing\n'
        'if typing.TYPE_CHECKING:\n'
        '    from inline_suppression_kinds import Site\n',
        True,
        id='typing-attribute-type-checking-downward',
    ),
    pytest.param(
        'inline_suppression_kinds',
        'from inline_suppression_scan import Scan\n',
        False,
        id='module-body-upward',
    ),
    pytest.param(
        'inline_suppression_kinds',
        'from typing import TYPE_CHECKING\n'
        'if TYPE_CHECKING:\n'
        '    from inline_suppression_scan import Scan\n',
        False,
        id='type-checking-upward',
    ),
    pytest.param(
        'inline_suppression_scan',
        'from typing import TYPE_CHECKING\n'
        'if TYPE_CHECKING:\n'
        '    pass\n'
        'else:\n'
        '    from inline_suppression_kinds import Site\n',
        False,
        id='type-checking-else-branch',
    ),
    pytest.param(
        'inline_suppression_scan',
        'import os\n'
        'if os.environ:\n'
        '    from inline_suppression_kinds import Site\n',
        False,
        id='runtime-conditional',
    ),
    pytest.param(
        'inline_suppression_scan',
        'def scan():\n'
        '    from inline_suppression_kinds import Site\n',
        False,
        id='function-local',
    ),
    pytest.param(
        'inline_suppression_scan',
        'from inline_suppression_unplaced import Thing\n',
        False,
        id='unplaced-module',
    ),
]


@pytest.mark.parametrize(('stem', 'source', 'accepted'), PLACEMENT_CASES)
def test_the_placement_rule_accepts_and_refuses_what_it_states(
    stem: str, source: str, accepted: bool
):
    assert (_placement_violations(stem, source) == []) is accepted
