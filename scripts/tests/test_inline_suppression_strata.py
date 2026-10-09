"""The inline-suppression scanner's files import each other in one order only.

:data:`STRATA` is the one home of that order.  Each family file may import a
family module only from an EARLIER entry, and only as a statement of its module
body, never inside a function, class or conditional.  So there is no cycle, no
reach-back into the entry module, and no import deferred to dodge either.

Both tests read import STRUCTURE through ``ast`` from the real files in
``scripts/``.  That makes this an architecture fitness check on the code, not
a check on anyone's prose about it.
"""

import ast
from pathlib import Path

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


def test_every_family_file_has_a_place_in_the_strata():
    """A new sibling must be placed in the order before it can import or be imported."""
    assert {path.stem for path in _family_files()} == set(STRATA)


def test_each_family_file_imports_only_earlier_strata_at_module_top_level():
    violations: list[str] = []
    for path in _family_files():
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        body = {id(statement) for statement in tree.body}
        importer = STRATA.index(path.stem) if path.stem in STRATA else None
        for node, imported in _family_imports(tree):
            where = f'scripts/{path.name}:{node.lineno} imports {imported}'
            if imported not in STRATA or importer is None:
                violations.append(f'{where}, and one of the two has no place in STRATA')
                continue
            if STRATA.index(imported) >= importer:
                violations.append(
                    f'{where} (STRATA index {STRATA.index(imported)}), which is not '
                    f'earlier than {path.stem} (STRATA index {importer})'
                )
            if id(node) not in body:
                violations.append(
                    f'{where} inside a function, class or conditional; family imports '
                    'are module-body statements only'
                )

    assert violations == [], '\n'.join(violations)
