"""AST scanner for silent-fallthrough-on-error signatures.

Detects two anti-patterns that hide errors instead of escalating them:

  Signature (a) — discarded resolver error slot:
      x, _ = [await] <resolver>(...)
    where <resolver> ∈ KNOWN_VALUE_ERROR_RESOLVERS.

  Signature (b) — silent broad-except returning an empty literal:
      except (Exception|BaseException|bare): return None/{}/()/[]/set()
    with no WARN+ logger call and no re-raise.

Design:
  - Pure stdlib (ast only) — no third-party deps.
  - find_violations(source, filename) is a pure function (no filesystem I/O)
    so unit tests can feed source strings directly.
  - SyntaxError from ast.parse() returns [] (never raises).
  - iter_first_party_files(repo_root) enumerates the first-party source tree
    (added in step-6).

References:
  - plans/silent-fallthrough-dedup-prd.md (PRD task σ)
  - orchestrator/tests/test_event_loop_antipattern_guard.py (guard pattern)
  - fused-memory/scripts/check_bare_magicmock_config.py (Violation NamedTuple)
"""
from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Violation(NamedTuple):
    """A detected silent-fallthrough violation."""

    filename: str
    lineno: int
    signature: str   # "a" or "b"
    message: str
    qualname: str = ""  # enclosing function qualname (e.g. "Harness.run")


# ---------------------------------------------------------------------------
# Signature (a) — known resolver set
# ---------------------------------------------------------------------------

#: Resolvers whose second return value is an Exception (or None) error slot.
#: Discarding this slot with ``_`` silences failures.
#:
#: Deliberately EXCLUDED:
#:   - parse_timestamp_or_warn, load_json_or_warn: (value, bool ok) primitives
#:     that log internally — the migrated tree legitimately discards their
#:     ok-flag with `_`.
#:   - get_tasks, get_status: overloaded names that also return list/scalar in
#:     other contexts — false-positive risk.
KNOWN_VALUE_ERROR_RESOLVERS: frozenset[str] = frozenset({
    "parse_tool_result",
    "get_external_statuses",
    "get_statuses",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _callee_name(node: ast.expr) -> str | None:
    """Extract the bare function/method name from a Call or Await(Call) node.

    Returns:
        ``Name.id`` for a plain call, ``Attribute.attr`` for a method call.
        None if the node is not a (possibly awaited) Call, or the callee is
        a complex expression that cannot be reduced to a single name.
    """
    if isinstance(node, ast.Await):
        node = node.value  # type: ignore[assignment]
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_broad_or_bare(handler: ast.ExceptHandler) -> bool:
    """Return True if this handler catches a broad or bare exception.

    "Broad or bare" means:
      - Bare ``except:`` (type is None)
      - ``except Exception``
      - ``except BaseException``
      - A tuple type ``except (A, B, ...)`` where at least one element is
        ``Exception`` or ``BaseException`` (by bare name).

    Narrow typed handlers (``except KeyError``, ``except (OSError, IOError)``)
    return False.
    """
    BROAD_NAMES = {"Exception", "BaseException"}

    t = handler.type
    if t is None:
        # bare except:
        return True
    if isinstance(t, ast.Name) and t.id in BROAD_NAMES:
        return True
    if isinstance(t, ast.Tuple):
        for elt in t.elts:
            if isinstance(elt, ast.Name) and elt.id in BROAD_NAMES:
                return True
    return False


def _is_empty_literal(node: ast.expr) -> bool:
    """Return True if node represents an empty data structure.

    Matched:
      - ``None`` constant
      - ``{}`` empty dict literal
      - ``[]`` empty list literal
      - ``()`` empty tuple literal
      - bare ``dict()``, ``list()``, ``set()``, ``tuple()`` calls with no args

    Deliberately excluded (per PRD design decision):
      - ``False``, ``0``, ``''`` — boolean/numeric fail-safes are excluded
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    if isinstance(node, ast.List) and not node.elts:
        return True
    if isinstance(node, ast.Tuple) and not node.elts:
        return True
    if isinstance(node, ast.Set) and not node.elts:
        return True
    # Bare calls: dict(), list(), set(), tuple() with no arguments
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id in {"dict", "list", "set", "tuple"}
            and not node.args
            and not node.keywords
        ):
            return True
    return False


def _handler_returns_empty(handler: ast.ExceptHandler) -> bool:
    """Return True if any direct-child Return of the handler returns empty.

    Does NOT descend into nested ``def`` or ``lambda`` bodies.
    """
    for node in handler.body:
        for stmt in ast.walk(node):
            # Do not descend into nested function/class definitions
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # ast.walk already entered it; skip its children by not yielding
                # We can't stop ast.walk from descending, so we use a manual DFS
                # approach instead.  Actually ast.walk is a generator over a flat
                # visit — we can't prune.  Use a targeted shallow check instead.
                continue
            if isinstance(stmt, ast.Return) and (stmt.value is None or _is_empty_literal(stmt.value)):
                return True
    return False


def _handler_returns_empty_shallow(handler: ast.ExceptHandler) -> bool:
    """Return True if any top-level Return in handler body returns empty.

    Uses a manual recursive descent that stops at nested def/lambda/class
    boundaries, unlike ast.walk which descends everywhere.
    """
    def _check_stmts(stmts: list[ast.stmt]) -> bool:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Do not descend into nested scopes
                continue
            if isinstance(stmt, ast.Return) and (stmt.value is None or _is_empty_literal(stmt.value)):
                return True
            # Recurse into sub-statement bodies (if/else, with, try, etc.)
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.stmt) and _check_stmts([child]):
                    return True
        return False

    return _check_stmts(handler.body)


def _handler_has_warn_log(handler: ast.ExceptHandler) -> bool:
    """Return True if the handler body contains a WARN+ logger call.

    Matches calls of the form ``<any>.{warning,warn,error,exception,critical,fatal}(...)``
    anywhere in the handler body (including nested blocks).
    """
    WARN_METHODS = {"warning", "warn", "error", "exception", "critical", "fatal"}
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in WARN_METHODS
        ):
            return True
    return False


def _handler_reraises(handler: ast.ExceptHandler) -> bool:
    """Return True if the handler body contains any ``raise`` statement."""
    return any(isinstance(node, ast.Raise) for node in ast.walk(handler))


# ---------------------------------------------------------------------------
# AST parent-map helpers (for qualname computation)
# ---------------------------------------------------------------------------


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Return {id(child): parent_node} for every node in *tree*."""
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node
    return parent_map


def _compute_qualname(node: ast.AST, parent_map: dict[int, ast.AST]) -> str:
    """Return the dotted qualname of the function/class scope enclosing *node*.

    Walks upward through the parent map collecting FunctionDef / AsyncFunctionDef /
    ClassDef names, then joins them in order.  Returns ``"<module>"`` if the
    node is at module scope.

    Example: a node inside ``class Harness: async def run(self): ...``
    returns ``"Harness.run"``.
    """
    parts: list[str] = []
    current = parent_map.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(current.name)
        current = parent_map.get(id(current))
    return ".".join(reversed(parts)) if parts else "<module>"


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------


def find_violations(source: str, filename: str) -> list[Violation]:
    """Scan *source* for silent-fallthrough signatures.

    Returns a list of :class:`Violation` instances, one per detected site.
    Returns ``[]`` on ``SyntaxError`` (the file is not flagged but also not
    silently skipped — callers should track and report parse failures).

    Args:
        source: UTF-8 source text of the file.
        filename: Path string used in :class:`Violation` records (for display).
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    parent_map = _build_parent_map(tree)
    violations: list[Violation] = []

    for node in ast.walk(tree):
        # ------------------------------------------------------------------ #
        # Signature (a): x, _ = [await] <resolver>(...)                      #
        # ------------------------------------------------------------------ #
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Tuple)
                and len(target.elts) == 2
                and isinstance(target.elts[1], ast.Name)
                and target.elts[1].id == "_"
            ):
                callee = _callee_name(node.value)
                if callee in KNOWN_VALUE_ERROR_RESOLVERS:
                    violations.append(Violation(
                        filename=filename,
                        lineno=node.lineno,
                        signature="a",
                        message=(
                            f"discarded error slot from {callee}(...) — "
                            f"bind the error to a named variable and "
                            f"escalate via resolver_failed() or raise"
                        ),
                        qualname=_compute_qualname(node, parent_map),
                    ))

        # ------------------------------------------------------------------ #
        # Signature (b): except (broad): return <empty>  (no warn, no raise) #
        # ------------------------------------------------------------------ #
        if isinstance(node, ast.ExceptHandler) and (
            _is_broad_or_bare(node)
            and _handler_returns_empty_shallow(node)
            and not _handler_has_warn_log(node)
            and not _handler_reraises(node)
        ):
            exc_desc = (
                "bare except"
                if node.type is None
                else ast.unparse(node.type)
            )
            violations.append(Violation(
                filename=filename,
                lineno=node.lineno,
                signature="b",
                message=(
                    f"silent broad-except ({exc_desc}) returns empty literal "
                    f"without logging WARN+ or re-raising — add "
                    f"logger.warning/error/exception(...) before the return"
                ),
                qualname=_compute_qualname(node, parent_map),
            ))

    return violations


# ---------------------------------------------------------------------------
# First-party file enumeration
# ---------------------------------------------------------------------------

#: The 7 scope roots searched for first-party source (relative to repo root).
_SCOPE_ROOTS = [
    "orchestrator/src",
    "fused-memory/src",
    "dashboard/src",
    "escalation/src",
    "shared/src",
    "sampler/src",
    "scripts",
]

#: Path components whose presence causes a file to be excluded from the scan.
_EXCLUDED_PATH_PARTS: frozenset[str] = frozenset({"mem0", "graphiti", "tests"})

#: File names that are always excluded regardless of location.
_EXCLUDED_NAMES: frozenset[str] = frozenset({"conftest.py"})


def iter_first_party_files(repo_root: Path) -> Iterator[Path]:
    """Yield absolute paths to first-party Python source files.

    Searches the 7 scope roots under *repo_root* and applies exclusions:
      - Path components named ``mem0``, ``graphiti``, or ``tests``
      - Files named ``test_*.py`` or ``conftest.py``

    Args:
        repo_root: Absolute path to the repository root.  Validated against
            sentinel directories (``shared/src``, ``orchestrator/src``); raises
            ``RuntimeError`` if they are absent (prevents a mis-resolved root
            from producing a silently-empty scan).
    """
    # Sentinel validation: fail loudly if repo_root is wrong
    for sentinel in ("shared/src", "orchestrator/src"):
        if not (repo_root / sentinel).is_dir():
            raise RuntimeError(
                f"Repo root validation failed: {sentinel!r} not found under "
                f"{repo_root!r}.  Ensure iter_first_party_files() receives the "
                f"correct repository root (not a sub-directory)."
            )

    for root_rel in _SCOPE_ROOTS:
        root_abs = repo_root / root_rel
        if not root_abs.is_dir():
            continue
        for py_file in sorted(root_abs.rglob("*.py")):
            # Exclude by path component
            if any(part in _EXCLUDED_PATH_PARTS for part in py_file.parts):
                continue
            # Exclude test files and conftest
            name = py_file.name
            if name.startswith("test_") or name in _EXCLUDED_NAMES:
                continue
            yield py_file
