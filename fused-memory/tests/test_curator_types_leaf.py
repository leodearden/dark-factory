"""``curator_types.py`` is the leaf of the curator's import graph.

The curator's data vocabulary lives below the driver, so a registry guard, the
escalator or the duplicate sweep can name a ``CandidateTask`` without importing
``task_curator`` and closing a cycle through it (plans/review-all-s11-curator-pipeline-prd.md
decision 3). Three guards hold that shape:

1. Importing the leaf in a clean interpreter loads neither the driver nor a
   heavy store client.
2. None of the modules that once formed the cycle with the driver imports it,
   at module level, under ``TYPE_CHECKING`` or inside a function.
3. The leaf itself imports only the stdlib and its declared first-party layer.
"""

from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

from _ast_guard import imported_names_from, parse_python_module

SRC_ROOT = pathlib.Path(__file__).parents[1] / 'src' / 'fused_memory'
MIDDLEWARE = SRC_ROOT / 'middleware'
CURATOR_TYPES = MIDDLEWARE / 'curator_types.py'

DRIVER = 'fused_memory.middleware.task_curator'
HEAVY_OR_DRIVER = (DRIVER, 'qdrant_client', 'graphiti_core')

FORMER_SCC_MEMBERS = (
    'curator_escalator',
    'curator_zot_duplicate_sweep',
    'cancelled_premise_blocklist',
    'operational_ask_registry',
    'recon_code_fix_premise_guard',
    'path_scope_guard',
)

ALLOWED_FIRST_PARTY = frozenset({
    'shared.task_statuses',
    'fused_memory.middleware.curation_deferral',
})

_PROBE = (
    'import json, sys\n'
    'import fused_memory.middleware.curator_types\n'
    f'print(json.dumps([name for name in {HEAVY_OR_DRIVER!r} if name in sys.modules]))\n'
)


def test_importing_curator_types_loads_neither_the_driver_nor_a_heavy_client():
    """``-I`` drops PYTHONPATH, the user site and the cwd, so this measures the
    leaf's own import closure rather than whatever the pytest process loaded."""
    result = subprocess.run(
        [sys.executable, '-I', '-c', _PROBE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [], (
        f'importing curator_types loaded {result.stdout.strip()}; the leaf must '
        'reach neither the driver nor a store client.'
    )


def _driver_imports(tree: ast.Module) -> set[str]:
    names = set(imported_names_from(tree, DRIVER))
    if 'task_curator' in imported_names_from(tree, 'fused_memory.middleware'):
        names.add('task_curator (module)')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == DRIVER for alias in node.names
        ):
            names.add(f'import {DRIVER}')
    return names


def test_no_former_scc_member_imports_the_driver():
    scanned = {
        name: _driver_imports(parse_python_module(MIDDLEWARE / f'{name}.py'))
        for name in FORMER_SCC_MEMBERS
    }
    assert scanned, 'no module scanned — this guard has lost its subject.'
    offenders = {name: sorted(names) for name, names in scanned.items() if names}
    assert not offenders, (
        'former curator-cycle member(s) still import task_curator: '
        + '; '.join(f'{name}: {", ".join(names)}' for name, names in offenders.items())
        + '. Import the vocabulary from fused_memory.middleware.curator_types instead.'
    )


def _imported_modules(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _is_allowed(module: str) -> bool:
    top_level = module.partition('.')[0]
    return (
        top_level in sys.stdlib_module_names
        or top_level == '__future__'
        or module in ALLOWED_FIRST_PARTY
    )


def test_curator_types_imports_only_the_stdlib_and_its_declared_first_party_layer():
    disallowed = sorted(
        module
        for module in _imported_modules(parse_python_module(CURATOR_TYPES))
        if not _is_allowed(module)
    )
    assert not disallowed, (
        f'curator_types.py imports {disallowed}; the leaf may import only the '
        f'stdlib and {sorted(ALLOWED_FIRST_PARTY)}.'
    )
