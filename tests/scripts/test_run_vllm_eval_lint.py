"""AST-based lint tests for scripts/run_vllm_eval.py.

These tests parse the source file directly with the `ast` module so they
work without any of the runtime dependencies that the script injects via
sys.path manipulation (runpod_toolkit, orchestrator.evals.configs, etc.).
"""
import ast
import pathlib

SCRIPT = pathlib.Path(__file__).parents[2] / "scripts" / "run_vllm_eval.py"


def _is_open_call(node: ast.AST) -> bool:
    """Return True if *node* is a bare call to the builtin open()."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "open"
    )


def _has_bare_open_chain(node: ast.AST) -> bool:
    """Return True if *node* is a method-call chain whose root is open().

    Matches patterns such as ``open(path).read()``,
    ``open(path).read().strip()``, ``open(path).readline()``, etc.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    receiver = node.func.value
    return _is_open_call(receiver) or _has_bare_open_chain(receiver)


def _module_level_value(stmt: ast.stmt) -> ast.expr | None:
    """Extract the RHS expression from a module-level Assign or Expr."""
    if isinstance(stmt, ast.Assign):
        return stmt.value
    if isinstance(stmt, ast.Expr):
        return stmt.value
    return None


def test_no_bare_open_at_module_scope() -> None:
    """No bare open().read() chains should exist at module scope.

    A bare open() without a ``with`` context manager leaks the file handle.
    The fix is to use ``with open(...) as fh: value = fh.read()``.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))

    violations: list[int] = []
    for stmt in tree.body:
        value = _module_level_value(stmt)
        if value is not None and _has_bare_open_chain(value):
            violations.append(stmt.lineno)

    assert violations == [], (
        f"{SCRIPT.name}: bare open().read() chain(s) found at module scope "
        f"on line(s) {violations}. Wrap with a 'with open(...) as fh:' block."
    )
