"""Static AST scan of test modules for two wait anti-patterns: a suppressed
result wait, which turns a hang into a confusing ``outcome is None``, and a
worker stop that a ``wait_responsive`` give-up can skip, which leaks the
worker into the next test.

Pure: source text in, pytest item ids out. It bills no wall clock. It shares
only the item traversal and the ``asyncio.wait_for`` call shape with
_wait_budget_audit.py, so both scans see the same items.

Each scan must never raise: a crash would fail the whole suite over an
unrelated edit to a scanned file. Unparseable source has no items, so it
yields [].
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from _wait_budget_audit import collect_test_items, is_asyncio_wait_for_call, item_id


def _is_timeout_error(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id == 'TimeoutError':
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'TimeoutError'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'asyncio'
    )


def _is_suppress_timeout_error_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_suppress = (isinstance(func, ast.Attribute) and func.attr == 'suppress') or (
        isinstance(func, ast.Name) and func.id == 'suppress'
    )
    return is_suppress and any(_is_timeout_error(arg) for arg in node.args)


def _catches_timeout_error(handler: ast.ExceptHandler) -> bool:
    caught = handler.type
    if caught is None:
        return False
    if isinstance(caught, ast.Tuple):
        return any(_is_timeout_error(elt) for elt in caught.elts)
    return _is_timeout_error(caught)


def _swallows(handler: ast.ExceptHandler) -> bool:
    """Whether every statement of *handler* is ``pass`` or an assignment of
    the literal None."""
    return all(
        isinstance(stmt, ast.Pass)
        or (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is None
        )
        for stmt in handler.body
    )


def _encloses_wait_for(nodes: Iterable[ast.AST]) -> bool:
    return any(is_asyncio_wait_for_call(node) for top in nodes for node in ast.walk(top))


def _is_suppressed_result_wait(node: ast.AST) -> bool:
    if isinstance(node, ast.With):
        return any(
            _is_suppress_timeout_error_call(item.context_expr) for item in node.items
        ) and _encloses_wait_for([node])
    if isinstance(node, ast.Try):
        return any(
            _catches_timeout_error(handler) and _swallows(handler) for handler in node.handlers
        ) and _encloses_wait_for(node.body)
    return False


def suppressed_result_wait_items(source: str) -> list[str]:
    """Every pytest item of *source* holding the executable form of the
    ``outcome is None`` anti-pattern that
    test_merge_queue_concurrent_verify.py::_await_outcome's docstring names
    ("The shape this replaces"), as sorted item ids. Either spelling counts:

    (a) a ``with contextlib.suppress(TimeoutError):`` block (or bare
        ``suppress``, or ``asyncio.TimeoutError``) enclosing an
        ``asyncio.wait_for(...)`` call;
    (b) a ``try:`` body enclosing an ``asyncio.wait_for(...)`` call whose
        ``except`` handler catches ``TimeoutError`` (bare, ``asyncio.``, or
        inside a tuple) and SWALLOWS it: every statement is ``pass`` or
        assigns the literal None.

    Both convert a genuine pipeline hang or a missing cascade into a
    confusing ``outcome is None`` assertion failure instead of failing
    loudly by label (task 4219).

    The swallowing requirement on (b) matters. A handler that catches the
    timeout and then fails LOUDLY -- cleanup, then ``pytest.fail(...)`` -- is
    the desired behaviour; matching on the caught type alone false-positived
    on two such real sites (task 4219 amendment). So a handler that does
    anything beyond swallowing is treated as already loud: a false negative
    is the safe direction here.

    Not reported: ``suppress(Exception)`` (the blessed best-effort
    teardown-join shape in test_merge_queue_concurrent_verify.py::_stop_worker),
    a ``suppress(TimeoutError)`` or ``try`` enclosing no ``asyncio.wait_for``,
    and a handler that does not catch ``TimeoutError``.
    """
    return sorted(
        {
            item_id(owner, function.name)
            for owner, function in collect_test_items(source)
            if any(map(_is_suppressed_result_wait, ast.walk(function)))
        }
    )


def _is_wait_responsive_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'wait_responsive'
    )


def _is_worker_stop_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == '_stop_worker'
    return (
        isinstance(func, ast.Attribute)
        and func.attr == 'stop'
        and isinstance(func.value, ast.Name)
        and func.value.id == 'worker'
    )


def _has_unguarded_worker_stop(function: ast.AST) -> bool:
    wait_calls = [node for node in ast.walk(function) if _is_wait_responsive_call(node)]
    if not wait_calls:
        return False
    guarded: set[ast.AST] = set()
    for try_node in ast.walk(function):
        if not isinstance(try_node, ast.Try):
            continue
        enclosed = {id(node) for stmt in try_node.body for node in ast.walk(stmt)}
        # A give-up raised by a wait OUTSIDE this try's body skips its
        # finally, so the finally guards nothing unless it encloses them all.
        if all(id(wait) in enclosed for wait in wait_calls):
            guarded.update(
                node
                for stmt in try_node.finalbody
                for node in ast.walk(stmt)
                if _is_worker_stop_call(node)
            )
    return any(
        _is_worker_stop_call(node) and node not in guarded for node in ast.walk(function)
    )


def unguarded_worker_teardown_items(source: str) -> list[str]:
    """Every pytest item of *source* that breaks the esc-3980-4 leak
    invariant recorded in test_merge_queue_concurrent_verify.py::_stop_worker's
    docstring, as sorted item ids.

    ``wait_responsive`` gives up by raising ``_pytest.outcomes.Failed`` (a
    ``BaseException``), so on a straight-line body a give-up -- like a
    mid-body ``assert`` -- skips the worker stop and leaks a live
    ``SpeculativeMergeWorker`` plus its run task into pytest-asyncio teardown.
    That leak is why one red test used to cascade into unrelated failures
    (task 4219).

    An item is reported when it calls ``wait_responsive(...)`` and calls a
    worker stop -- ``worker.stop()`` or the shared ``_stop_worker(...)``
    helper -- anywhere other than inside a ``finally:`` whose ``try:`` body
    encloses EVERY ``wait_responsive`` call in the item. Crediting any
    finally at all would miss a wait that sits before the ``try:``, whose
    give-up skips that finally too (task 4219 amendment). Recognising
    ``_stop_worker`` keeps the scan from going vacuous for an item that
    adopts the other blessed teardown shape.

    Deliberately scoped to items that call ``wait_responsive``: unmigrated
    straight-line teardown elsewhere is not this scan's business, which is
    to stop the migrated set regressing. Deliberately narrow to a variable
    literally named ``worker``: a false negative on an oddly named variable
    is the safe direction for a regression guard.
    """
    return sorted(
        {
            item_id(owner, function.name)
            for owner, function in collect_test_items(source)
            if _has_unguarded_worker_stop(function)
        }
    )
