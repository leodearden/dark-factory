"""Static AST scan of test modules for two wait anti-patterns: a suppressed
result wait, which turns a hang into a confusing ``outcome is None``, and a
worker stop that a ``wait_responsive`` give-up can skip, which leaks the
worker into the next test.

Pure: source text in, pytest item ids out (for the teardown scan, a census
of them). It bills no wall clock. It shares
only the item traversal and the ``asyncio.wait_for`` call shape with
_wait_budget_audit.py, so both scans see the same items.

Each scan must never raise: a crash would fail the whole suite over an
unrelated edit to a scanned file. Unparseable source has no items, so it
yields none.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from _wait_budget_audit import (
    collect_test_items,
    is_asyncio_wait_for_call,
    is_responsive_wait_call,
    item_id,
)


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
    teardown-join shape in _merge_worker_drive.py::stop_merge_worker),
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


@dataclass(frozen=True)
class WorkerTeardownScan:
    """Sorted item ids: those calling a responsive wait (the scan's scope), those of them that also stop a worker, and those with a stop outside a guarding finally."""

    waiting_items: tuple[str, ...]
    stopping_items: tuple[str, ...]
    unguarded_items: tuple[str, ...]


def _calls_create_task(call: ast.Call) -> bool:
    func = call.func
    return (isinstance(func, ast.Attribute) and func.attr == 'create_task') or (
        isinstance(func, ast.Name) and func.id == 'create_task'
    )


def _run_task_receivers(function: ast.AST) -> frozenset[str]:
    """Every Name ``R`` that *function* runs as a task: ``create_task(R.run(...))``."""
    receivers: set[str] = set()
    for node in ast.walk(function):
        if not (isinstance(node, ast.Call) and _calls_create_task(node) and node.args):
            continue
        task = node.args[0]
        if (
            isinstance(task, ast.Call)
            and isinstance(task.func, ast.Attribute)
            and task.func.attr == 'run'
            and isinstance(task.func.value, ast.Name)
        ):
            receivers.add(task.func.value.id)
    return frozenset(receivers)


def _is_worker_stop_call(node: ast.AST, receivers: frozenset[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == 'stop_merge_worker'
    return (
        isinstance(func, ast.Attribute)
        and func.attr == 'stop'
        and isinstance(func.value, ast.Name)
        and func.value.id in receivers
    )


def _guarded_nodes(function: ast.AST, wait_calls: list[ast.AST]) -> set[int]:
    """Ids of every node inside a ``finally:`` whose ``try:`` body encloses all of *wait_calls*."""
    guarded: set[int] = set()
    for try_node in ast.walk(function):
        if not isinstance(try_node, ast.Try):
            continue
        enclosed = {id(node) for stmt in try_node.body for node in ast.walk(stmt)}
        # A give-up raised by a wait OUTSIDE this try's body skips its
        # finally, so the finally guards nothing unless it encloses them all.
        if all(id(wait) in enclosed for wait in wait_calls):
            guarded.update(id(node) for stmt in try_node.finalbody for node in ast.walk(stmt))
    return guarded


def _classify_teardown(function: ast.AST) -> tuple[bool, bool, bool]:
    """Whether *function* (waits, stops a worker, stops one unguarded)."""
    wait_calls = [node for node in ast.walk(function) if is_responsive_wait_call(node)]
    if not wait_calls:
        return False, False, False
    receivers = frozenset({'worker'}) | _run_task_receivers(function)
    stop_calls = [node for node in ast.walk(function) if _is_worker_stop_call(node, receivers)]
    guarded = _guarded_nodes(function, wait_calls)
    return True, bool(stop_calls), any(id(stop) not in guarded for stop in stop_calls)


def scan_worker_teardown(source: str) -> WorkerTeardownScan:
    """The esc-3980-4 teardown census of *source*'s pytest items
    (see :func:`unguarded_worker_teardown_items` for the rule). Never raises:
    unparseable source yields three empty tuples."""
    waiting: set[str] = set()
    stopping: set[str] = set()
    unguarded: set[str] = set()
    for owner, function in collect_test_items(source):
        waits, stops, unguarded_stop = _classify_teardown(function)
        item = item_id(owner, function.name)
        if waits:
            waiting.add(item)
        if stops:
            stopping.add(item)
        if unguarded_stop:
            unguarded.add(item)
    return WorkerTeardownScan(
        waiting_items=tuple(sorted(waiting)),
        stopping_items=tuple(sorted(stopping)),
        unguarded_items=tuple(sorted(unguarded)),
    )


def unguarded_worker_teardown_items(source: str) -> list[str]:
    """Every pytest item of *source* that breaks the esc-3980-4 leak
    invariant recorded in _merge_worker_drive.py::stop_merge_worker's
    docstring, as sorted item ids.

    ``wait_responsive`` gives up by raising ``_pytest.outcomes.Failed`` (a
    ``BaseException``), so on a straight-line body a give-up -- like a
    mid-body ``assert`` -- skips the worker stop and leaks a live
    ``SpeculativeMergeWorker`` plus its run task into pytest-asyncio teardown.
    That leak is why one red test used to cascade into unrelated failures
    (task 4219).

    An item is reported when it calls a responsive wait (wait_responsive or
    poll_responsive, per _wait_budget_audit.py::is_responsive_wait_call) and
    calls a worker stop -- ``<receiver>.stop()`` or the shared
    ``stop_merge_worker(...)`` helper -- anywhere other than inside a
    ``finally:`` whose ``try:`` body encloses EVERY responsive wait in the
    item. Crediting any
    finally at all would miss a wait that sits before the ``try:``, whose
    give-up skips that finally too (task 4219 amendment). Recognising
    ``stop_merge_worker`` keeps the scan from going vacuous for an item that
    adopts the other blessed teardown shape.

    Deliberately scoped to items that call a responsive wait: unmigrated
    straight-line teardown elsewhere is not this scan's business, which is
    to stop the migrated set regressing. A receiver is a worker by
    structure, not by a list of names: a variable named ``worker``, or any
    Name whose ``.run()`` the item hands to ``create_task(...)``. A stop on
    any other receiver is not a worker stop, the safe direction for a
    regression guard.
    """
    return list(scan_worker_teardown(source).unguarded_items)
