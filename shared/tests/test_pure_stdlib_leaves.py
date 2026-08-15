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
``shared/__init__.py`` is now a PEP 562 lazy package, guarded against
re-coupling from two directions: ``test_bare_package_import_binds_no_
submodules`` measures the property in a fresh interpreter (unevadable, but
names no line) and ``test_init_has_no_runtime_submodule_imports`` scans the
AST (names the offending line, but only sees what it can parse).

DESIGN.
  - Pure stdlib (``ast``, ``json``, ``subprocess``) — no third-party deps.
  - ``run_python_child`` is the single home of the child-runner invariant;
    ``test_lazy_public_api.py`` imports it rather than keeping a second copy.
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
from typing import Any

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

# Child program: import `shared` and report which `shared.<submodule>` modules it
# loaded as a side effect. This MEASURES the property the AST guard below can
# only approximate syntactically, so it holds regardless of how an eager import
# is spelled (nested in try/if/with, or triggered by a module-level call).
_BOUND_SUBMODULES_CHILD_SRC = """
import json
import sys

import shared  # the ONLY import: this measures its side effects

print(json.dumps(sorted(n for n in sys.modules if n.startswith('shared.'))))
"""


def run_python_child(src: str, *argv: str) -> Any:
    """Run ``src`` in a fresh interpreter against the LOCAL src tree; parse its JSON.

    THE SINGLE HOME of the child-runner invariant. ``test_lazy_public_api.py``
    imports this rather than keeping a second copy (bare-module import — the
    tests dir is on ``sys.path`` via conftest, the same shape as
    ``test_usage_gate.py``'s ``from test_config_dir import ...``). It lives in a
    test module rather than a helper module of its own only because task 3896's
    lock charter covers these test files and not a new one.

    The PYTHONPATH prepend is load-bearing, not boilerplate: a bare
    ``python -c 'import shared'`` inside a task worktree resolves to the MAIN
    checkout's installed editable copy, so every assertion built on this runner
    would silently measure the wrong tree.
    """
    env = {
        **os.environ,
        'PYTHONPATH': f'{_SRC}{os.pathsep}' + os.environ.get('PYTHONPATH', ''),
    }
    proc = subprocess.run(
        [sys.executable, '-c', src, *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f'child exited {proc.returncode} (argv={list(argv)})\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'
    )
    return json.loads(proc.stdout)


def _third_party_after_import(target: str) -> list[str]:
    """Return the third-party top-level packages ``import <target>`` pulls in."""
    return run_python_child(_CHILD_SRC, target)


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


def test_bare_package_import_binds_no_submodules():
    """`import shared` must load no `shared.<submodule>` at all.

    The behavioural half of the re-coupling guard, and the stronger half: it
    measures the actual property rather than inspecting syntax, so it holds
    however an eager import is spelled — nested in ``try:``/``if:``/``with:``,
    or triggered indirectly by a module-level call the AST scan cannot follow.
    Re-importing a *pure* leaf eagerly leaks no third-party package, so
    ``test_bare_package_import_loads_no_third_party`` would stay green while
    the coupling silently returned; this is what catches that.
    """
    bound = run_python_child(_BOUND_SUBMODULES_CHILD_SRC)
    assert bound == [], (
        f'`import shared` eagerly loaded {len(bound)} submodule(s): '
        f'{", ".join(bound)}.\nshared/__init__.py must stay lazy (PEP 562) — '
        'every one of these runs on EVERY `import shared.<anything>`. Resolve '
        'them in __getattr__ instead.'
    )


def _is_type_checking_test(test: ast.expr) -> bool:
    """True for `TYPE_CHECKING` / `typing.TYPE_CHECKING`, and nothing else.

    `if not TYPE_CHECKING:` is an ``ast.UnaryOp`` and deliberately does NOT
    match: its body executes at import time, which is exactly the evasion this
    guard exists to catch.
    """
    if isinstance(test, ast.Name):
        return test.id == 'TYPE_CHECKING'
    if isinstance(test, ast.Attribute):
        return test.attr == 'TYPE_CHECKING'
    return False


def _import_time_unreachable_ids(tree: ast.Module) -> set[int]:
    """ids of the AST nodes that cannot execute when the module is imported.

    Two families, and only these two:

      - the ``body`` of an ``if TYPE_CHECKING:`` — False at runtime, so those
        imports exist purely for pyright/ruff. The ``orelse`` is NOT excluded:
        an ``else:`` branch does run.
      - the body of a ``def``/``async def`` — a deferred import inside
        ``__getattr__`` is the lazy pattern this guard exists to encourage.
        Class bodies are deliberately NOT excluded: they execute at import time.

    Everything else — including a ``try:``, ``if:``, or ``with:`` block at
    module level — stays in scope, so one level of indentation no longer hides
    a re-coupling regression.
    """
    unreachable: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or (
            isinstance(node, ast.If) and _is_type_checking_test(node.test)
        ):
            # `.body` only — never `.orelse`, and never the `if` test itself.
            for statement in node.body:
                unreachable.update(id(n) for n in ast.walk(statement))
    return unreachable


def test_init_has_no_runtime_submodule_imports():
    """shared/__init__.py must import no `shared.*` submodule at import time.

    Walks the WHOLE tree, excluding only what cannot run at import time (see
    ``_import_time_unreachable_ids``). Scanning just ``tree.body`` would let any
    nesting through — ``try: from shared.locking import ...``, ``if not
    TYPE_CHECKING: ...``, ``if sys.version_info >= (3, 12): ...`` — since each
    is an ``ast.Try``/``ast.If`` node that is itself neither ``ast.Import`` nor
    ``ast.ImportFrom``.

    This is the syntactic half of the guard; ``test_bare_package_import_binds_
    no_submodules`` above is the behavioural half. Both are kept: the runtime
    check is impossible to evade, this one names the offending line.
    """
    init_path = _SRC / 'shared' / '__init__.py'
    tree = ast.parse(init_path.read_text(), filename=str(init_path))
    unreachable = _import_time_unreachable_ids(tree)

    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in unreachable:
            continue
        if isinstance(node, ast.Import):
            offenders += [
                (node.lineno, f'line {node.lineno}: import {a.name}')
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
                (
                    node.lineno,
                    f'line {node.lineno}: from {dots}{node.module or ""} import '
                    + ', '.join(a.name for a in node.names),
                )
            )

    # ast.walk is breadth-first, so sort back into source order for the message.
    rendered = [text for _, text in sorted(offenders)]
    assert rendered == [], (
        'shared/__init__.py imports shared.* submodule(s) at import time:\n  '
        + '\n  '.join(rendered)
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
