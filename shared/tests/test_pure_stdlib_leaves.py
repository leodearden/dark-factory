"""Per-module purity guard: designated leaves import no third-party package.

THE CONTRACT. The names in ``PURE_STDLIB_LEAVES`` are a DELIBERATE list, not
an observation. Each names a ``shared.<leaf>`` module whose *transitive
module-level* import closure is stdlib-only, and importing it must never pull
a third-party package into ``sys.modules``. Adding a third-party import to one
of these modules is allowed — but it requires REMOVING that module from this
list in the SAME change, so the loss of purity is a visible, reviewed edit
rather than a silent regression.

WHY IT REGRESSED BEFORE. Python executes a package's ``__init__.py`` before any
submodule import, so while ``shared/__init__.py`` eagerly re-exported from
``config_models``/``usage_gate``/``cli_invoke``/``async_sqlite_base``, every
one of these leaves dragged in pydantic, yaml, aiosqlite, dotenv and aiohttp —
18 third-party top-level packages — no matter how pure the leaf itself was.
``shared/__init__.py`` is now a PEP 562 lazy package; ``test_init_has_no_
runtime_submodule_imports`` below is the guard against re-coupling it.

DESIGN.
  - Pure stdlib (``ast``, ``json``, ``subprocess``) — no third-party deps.
  - The leakage check runs in a SUBPROCESS: ``sys.modules`` is process-global
    and pytest has already imported pydantic/aiosqlite/yaml/dotenv by
    collection time, so an in-process assertion is impossible. A before/after
    diff of ``sys.modules`` is immune to whatever the harness preloaded (only
    NEWLY added modules count) and needs no hand-maintained blocklist, so it
    catches a future dependency nobody remembered to list.
  - ``sys.stdlib_module_names`` supplies the exact stdlib set (it includes
    private stdlib modules like ``_abc``), so no underscore heuristic is
    needed. A pydantic-injected ``cython_runtime`` is correctly flagged.
  - Parametrized per leaf, so a regression names the offending module rather
    than failing the package as a whole.

References:
  - shared/tests/test_verify_admission.py:188 (``python -c`` child pattern)
  - shared/tests/silent_fallthrough_scan.py (stdlib AST scanner + contract)
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Same src-root expression as shared/tests/conftest.py and
# test_public_api.py:255 — read the LOCAL tree, never an installed copy.
_SRC = Path(__file__).resolve().parent.parent / 'src'

#: Modules whose transitive module-level import closure is stdlib-only.
#: See the module docstring: this list is a contract, not a snapshot.
PURE_STDLIB_LEAVES = (
    'agent_result',
    'branch_names',
    'config_dir',
    'locking',
    'mcp_envelope',
    'mcp_idempotency',
    'neutral_cwd',
    'proc_group',
    'psi',
    'pytest_jobserver',
    'safe_io',
    'task_claimant',
    'task_statuses',
    'task_transitions',
    'timestamps',
    'transcript_archive',
    'verify_admission',
)

# Child program: snapshot sys.modules, import argv[1], report the top-level
# names that appeared and are neither stdlib nor `shared` itself, as JSON.
_CHILD_SRC = """
import importlib
import json
import sys

before = set(sys.modules)
importlib.import_module(sys.argv[1])
added = {name.split('.')[0] for name in set(sys.modules) - before}
print(json.dumps(sorted(added - set(sys.stdlib_module_names) - {'shared'})))
"""


def _third_party_after_import(target: str) -> list[str]:
    """Return the third-party top-level packages ``import <target>`` pulls in."""
    env = {
        **os.environ,
        'PYTHONPATH': f'{_SRC}{os.pathsep}' + os.environ.get('PYTHONPATH', ''),
    }
    proc = subprocess.run(
        [sys.executable, '-c', _CHILD_SRC, target],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f'child failed to import {target!r} (exit {proc.returncode})\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'
    )
    return json.loads(proc.stdout)


@pytest.mark.parametrize('leaf', PURE_STDLIB_LEAVES)
def test_leaf_imports_without_third_party(leaf):
    """`import shared.<leaf>` must load zero third-party packages."""
    leaked = _third_party_after_import(f'shared.{leaf}')
    assert leaked == [], (
        f'shared.{leaf} is designated a pure-stdlib leaf but importing it '
        f'loaded {len(leaked)} third-party package(s): {", ".join(leaked)}.\n'
        f'Either drop the new dependency, or remove {leaf!r} from '
        f'PURE_STDLIB_LEAVES in this file as part of the same change.'
    )


def test_bare_package_import_loads_no_third_party():
    """`import shared` itself must stay inert — it is what coupled the leaves."""
    leaked = _third_party_after_import('shared')
    assert leaked == [], (
        f'`import shared` loaded {len(leaked)} third-party package(s): '
        f'{", ".join(leaked)}.\nshared/__init__.py must stay lazy (PEP 562); '
        f'anything it imports eagerly is paid by EVERY shared.* consumer.'
    )


def test_init_has_no_runtime_submodule_imports():
    """shared/__init__.py must import no `shared.*` submodule at runtime.

    Walks only ``tree.body`` — module level — so the ``if TYPE_CHECKING:``
    block's static-only re-exports are deliberately not descended into. This
    catches a re-coupling regression (e.g. re-adding ``from shared.locking
    import ...`` at module level) that the third-party check alone would miss,
    because a pure leaf re-imported eagerly leaks nothing yet still rebuilds
    the coupling this guard exists to prevent.
    """
    init_path = _SRC / 'shared' / '__init__.py'
    tree = ast.parse(init_path.read_text(), filename=str(init_path))

    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders += [
                f'line {node.lineno}: import {a.name}'
                for a in node.names
                if a.name.split('.')[0] == 'shared'
            ]
        # level > 0 is a relative import (`from . import locking`), which names
        # a shared.* submodule just as surely as an absolute one.
        elif isinstance(node, ast.ImportFrom) and (
            node.level > 0 or (node.module or '').split('.')[0] == 'shared'
        ):
            dots = '.' * node.level
            offenders.append(
                f'line {node.lineno}: from {dots}{node.module or ""} import '
                + ', '.join(a.name for a in node.names)
            )

    assert offenders == [], (
        'shared/__init__.py imports shared.* submodule(s) at module level:\n  '
        + '\n  '.join(offenders)
        + '\nThese run on EVERY `import shared.<anything>`. Move them under '
        '`if TYPE_CHECKING:` and resolve them lazily in __getattr__.'
    )


def test_designated_leaves_exist():
    """Every name in PURE_STDLIB_LEAVES names a real module (rename hygiene)."""
    missing = [leaf for leaf in PURE_STDLIB_LEAVES if not (_SRC / 'shared' / f'{leaf}.py').is_file()]
    assert missing == [], (
        f'PURE_STDLIB_LEAVES names module(s) that do not exist: {missing}.\n'
        'Update the list when a module is renamed or removed.'
    )
