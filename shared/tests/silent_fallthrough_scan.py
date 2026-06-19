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
            if isinstance(stmt, ast.Return):
                if stmt.value is None or _is_empty_literal(stmt.value):
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
            if isinstance(stmt, ast.Return):
                if stmt.value is None or _is_empty_literal(stmt.value):
                    return True
            # Recurse into sub-statement bodies (if/else, with, try, etc.)
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.stmt):
                    if _check_stmts([child]):
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
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
    return False


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

    violations: list[Violation] = []

    for node in ast.walk(tree):
        # ------------------------------------------------------------------ #
        # Signature (a): x, _ = [await] <resolver>(...)                      #
        # ------------------------------------------------------------------ #
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
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
                        ))

        # ------------------------------------------------------------------ #
        # Signature (b): except (broad): return <empty>  (no warn, no raise) #
        # ------------------------------------------------------------------ #
        if isinstance(node, ast.ExceptHandler):
            if (
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
                ))

    return violations
