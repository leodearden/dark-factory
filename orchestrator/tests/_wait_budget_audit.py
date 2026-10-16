"""Static AST audit of the real wall clock a test's waits can consume.

Pure: source text and a module namespace in, numbers and names out -- no
file I/O, no imports of the code it scans -- so every function here can be
driven with foreign input.  Callers own reading files and importing modules.

The bill is a conservative estimate of the wall clock each pytest item's
recognised waits can take (``_call_wait_budget``), for a coverage guard to
compare with the timeout the item actually runs under.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterator, Mapping

from _orch_helpers import (
    MERGE_GATE_BARRIER_TIMEOUT,
    MERGE_RESULT_TIMEOUT,
    PYPROJECT_DEFAULT_TIMEOUT,
    RESPONSIVE_WAIT_STRETCH,
    RESPONSIVE_WAIT_WALL_CAP,
    STOP_WORKER_JOIN_TIMEOUT,
    VERIFY_CLI_PER_TEST_TIMEOUT,
)

# task 3492: the shared helper constants a scanned call site may spell without
# importing them into its own namespace, plus the hidden defaults of the
# helpers it bills.  Referenced directly (not re-derived) so the audit can
# never drift from them.
_KNOWN_WAIT_CONSTANTS: dict[str, float] = {
    'MERGE_RESULT_TIMEOUT': float(MERGE_RESULT_TIMEOUT),
    # task 3980: `wait_responsive`'s hard wall backstop and the merge-pipeline
    # gate-barrier nominal, so a scanned call site spelling either one
    # resolves to its value rather than billing as unbounded.
    'RESPONSIVE_WAIT_WALL_CAP': float(RESPONSIVE_WAIT_WALL_CAP),
    'MERGE_GATE_BARRIER_TIMEOUT': float(MERGE_GATE_BARRIER_TIMEOUT),
}


def _wait_seconds(value: object) -> float | None:
    """*value* as a wait in seconds, or None when it is not a plain number
    (a bool is not a wait). A negative number waits no time, so 0.0."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(float(value), 0.0)
    return None


def _bound_names(fn: ast.AST) -> set[str]:
    """Every name *fn* binds anywhere in its body: a parameter, an
    assignment/for/with/walrus/del target, an except-as name, an import
    alias, a match capture, or a nested def/class. Such a name is not the
    module-level value of the same name."""
    bound: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split('.')[0])
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node is not fn
        ):
            bound.add(node.name)
    return bound


def _visible_wait_constants(
    fn: ast.AST, module_constants: Mapping[str, object] | None
) -> dict[str, float]:
    """The resolvable value of every name *fn* reads and never binds: the
    scanned module's namespace value when it is a number, else the
    ``_KNOWN_WAIT_CONSTANTS`` value. The namespace wins because it is what
    the module's code actually sees."""
    namespace = module_constants or {}
    bound = _bound_names(fn)
    visible: dict[str, float] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Name) or node.id in bound or node.id in visible:
            continue
        seconds = _wait_seconds(namespace.get(node.id))
        if seconds is None:
            seconds = _KNOWN_WAIT_CONSTANTS.get(node.id)
        if seconds is not None:
            visible[node.id] = seconds
    return visible


def _resolve_wait_value(node: ast.expr | None, visible: Mapping[str, float]) -> float:
    """Resolve a wait-value expression to the most wall clock it can allow.

    Never guesses. In order: a numeric literal resolves to itself (a
    negative one waits no time, so 0.0); a name resolves through *visible*
    (``_visible_wait_constants``): the scanned module's numeric value for
    it, else ``_KNOWN_WAIT_CONSTANTS``, and never when the method binds the
    name itself. Anything else -- a shadowed or unknown name, an attribute
    access, arithmetic, the literal None, a missing value -- resolves to
    ``math.inf``: the scan cannot read it, so it may allow any wall clock,
    and no timeout mark clears an unbounded bill. Failing closed is what
    keeps every recognised shape's bill an upper bound.
    """
    if isinstance(node, ast.Constant):
        seconds = _wait_seconds(node.value)
        return math.inf if seconds is None else seconds
    if isinstance(node, ast.Name) and node.id in visible:
        return visible[node.id]
    return math.inf


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    """Return the value of *call*'s ``name=`` keyword argument, or None."""
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _wait_responsive_budget(call: ast.Call, visible: Mapping[str, float]) -> float:
    """Return the worst-case WALL clock a ``wait_responsive(...)`` call can
    consume -- an unconditional upper bound, not an estimate (task 3980).

    The helper gives up once its wall clock reaches its cap: an explicit
    ``max_wall_s`` when one is passed, else the nominal ``timeout`` (hidden
    default ``MERGE_RESULT_TIMEOUT``) stretched by ``RESPONSIVE_WAIT_STRETCH``
    and clamped to ``RESPONSIVE_WAIT_WALL_CAP``. So an explicit
    ``max_wall_s`` other than the literal None is billed as resolved -- one
    the scan cannot read is unbounded -- and otherwise the stretched,
    clamped nominal is billed, which an unreadable nominal cannot push past
    the clamp.
    """
    max_wall = _keyword(call, 'max_wall_s')
    explicit_none = isinstance(max_wall, ast.Constant) and max_wall.value is None
    if max_wall is not None and not explicit_none:
        return _resolve_wait_value(max_wall, visible)

    timeout = _keyword(call, 'timeout')
    nominal = (
        _KNOWN_WAIT_CONSTANTS['MERGE_RESULT_TIMEOUT']
        if timeout is None
        else _resolve_wait_value(timeout, visible)
    )
    return min(
        nominal * RESPONSIVE_WAIT_STRETCH,
        _KNOWN_WAIT_CONSTANTS['RESPONSIVE_WAIT_WALL_CAP'],
    )


def _call_wait_budget(call: ast.Call, visible: Mapping[str, float]) -> float:
    """Return the wait-budget contribution of a single ``ast.Call`` node.

    Recognises exactly four call shapes -- ``asyncio.wait_for(..., timeout=N)``
    (the attribute's value must itself be the ``asyncio`` name, so
    ``gate.wait_for(...)`` or ``self.wait_for(...)`` do NOT match),
    ``_await_outcome(...)`` (whose hidden default is ``MERGE_RESULT_TIMEOUT``
    when no ``timeout=`` kwarg is given), ``wait_responsive(...)``
    (task 3980), billed by ``_wait_responsive_budget`` as an unconditional
    upper bound on its wall clock, and the ``_stop_worker(...)`` teardown,
    whose join is billed its ``join_timeout=`` kwarg or hidden default
    ``STOP_WORKER_JOIN_TIMEOUT``. Within a recognised shape, a named value
    resolves through *visible* (the scanned module's numeric constants, else
    ``_KNOWN_WAIT_CONSTANTS``, never a name the method rebinds) and a value
    the scan cannot read bills ``math.inf`` (``_resolve_wait_value``), so a
    recognised wait is never under-billed. Every
    other call shape -- ``asyncio.sleep``, ``.wait()``, ``.result()``,
    ``.join()``, a non-``asyncio`` ``.wait_for(...)``, attribute access,
    arithmetic, unknown names -- contributes 0.0 and is skipped silently
    (this is a conservative FLOOR over call *shapes*: an unrecognised shape
    can only under-count, never fabricate a wait; see ``_method_wait_budget``
    for the orthogonal control-flow and repetition imprecision).
    """
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == 'wait_for'
        and isinstance(func.value, ast.Name)
        and func.value.id == 'asyncio'
    ):
        timeout = _keyword(call, 'timeout')
        if timeout is None and len(call.args) >= 2:
            timeout = call.args[1]
        return _resolve_wait_value(timeout, visible)
    if isinstance(func, ast.Name) and func.id == '_await_outcome':
        timeout = _keyword(call, 'timeout')
        if timeout is None:
            return _KNOWN_WAIT_CONSTANTS['MERGE_RESULT_TIMEOUT']
        return _resolve_wait_value(timeout, visible)
    if isinstance(func, ast.Name) and func.id == 'wait_responsive':
        return _wait_responsive_budget(call, visible)
    if isinstance(func, ast.Name) and func.id == '_stop_worker':
        join_timeout = _keyword(call, 'join_timeout')
        if join_timeout is None:
            return STOP_WORKER_JOIN_TIMEOUT
        return _resolve_wait_value(join_timeout, visible)
    return 0.0


def _method_wait_budget(
    fn: ast.AST, module_constants: Mapping[str, object] | None = None
) -> float:
    """Sum every recognised wait-call contribution anywhere in *fn*'s body,
    resolving named values against *module_constants*, the scanned module's
    namespace (see ``_visible_wait_constants``).

    Control-flow-BLIND by construction: ``ast.walk`` visits every call in
    the subtree regardless of branch, so waits in mutually exclusive
    ``if``/``else`` arms or ``try``/``except`` handlers are summed
    together rather than maxed -- the computed total can OVER-count
    relative to any single real execution path (pinned by
    test_wait_budget_audit.py::TestWaitBudgetAuditHelper::test_mutually_exclusive_branches_are_summed_not_maxed).
    This is a deliberate, tolerated imprecision, not a design goal: it runs
    in the opposite direction from ``_call_wait_budget``'s shape-based
    under-count (an unrecognised call shape contributes 0.0), so the
    "conservative FLOOR" framing applies to shape recognition only, not to
    control flow.

    It is also REPETITION-blind, which under-bills: each call site is billed
    once, so a wait inside a loop, or inside a helper the test calls
    repeatedly, counts once however often it runs. A test that repeats waits
    that way needs headroom over its bill that this scan cannot see.

    The combined result is a same-ballpark estimate meant to catch gross
    under-marking, not a number that is exactly load-bearing.
    """
    visible = _visible_wait_constants(fn, module_constants)
    return sum(
        _call_wait_budget(node, visible) for node in ast.walk(fn) if isinstance(node, ast.Call)
    )


def _iter_test_methods(
    source: str,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Statically parse *source* and return every ``(class_name, method)``
    pair for a ``test_*`` method defined directly in the body of a
    top-level ``Test*`` class.

    The shared traversal behind `suppressed_result_wait_methods` and
    `unguarded_worker_teardown_methods` (task 4219 amendment), so a blind
    spot (nested classes, module-level functions, non-``Test*`` classes) is
    fixed in one place.

    Must never raise: a crash here would fail the whole suite over an
    unrelated edit to a scanned file. Unparseable source returns [].
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    pairs: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for class_node in ast.iter_child_nodes(tree):
        if not (isinstance(class_node, ast.ClassDef) and class_node.name.startswith('Test')):
            continue
        for method in class_node.body:
            if (
                isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name.startswith('test_')
            ):
                pairs.append((class_node.name, method))
    return pairs


def _iter_test_items(
    tree: ast.Module,
) -> Iterator[tuple[str | None, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield ``(owner_class_name, function)`` for every pytest item *tree*
    defines: each ``test*`` method directly in a top-level ``Test*`` class,
    and each top-level ``test*`` function (owner None). Those are pytest's
    default ``python_classes`` / ``python_functions`` prefixes, which
    orchestrator/pyproject.toml does not override. Nested classes, inherited
    methods and conditionally defined classes are out of scope."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('test'):
                yield None, node
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            for method in node.body:
                if (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name.startswith('test')
                ):
                    yield node.name, method


def item_wait_budgets(
    source: str, module_constants: Mapping[str, object] | None = None
) -> dict[str, float]:
    """Statically scan *source* and bill each pytest item the summed
    sequential real-time wait budget of its body (task 3492's audit
    engine), keyed ``'Class::method'`` or, for a module-level test function,
    its bare name. A duplicate key keeps the larger bill.

    Per item, never a class-wide sum: pytest-timeout applies a class-level
    ``@pytest.mark.timeout`` mark to EACH test method independently, so a
    sum would false-positive on many-light-methods classes and could dilute
    one heavy method below a threshold.

    *module_constants* is the scanned module's namespace, which resolves the
    names its waits spell (``_visible_wait_constants``).

    Must never raise: a crash here would fail the whole suite over an
    unrelated edit to a scanned file. Unparseable source returns {}.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    budgets: dict[str, float] = {}
    for owner, function in _iter_test_items(tree):
        key = function.name if owner is None else f'{owner}::{function.name}'
        budget = _method_wait_budget(function, module_constants)
        budgets[key] = max(budget, budgets.get(key, budget))
    return budgets


def worst_per_method_wait_budget(
    source: str, module_constants: Mapping[str, object] | None = None
) -> dict[str, float]:
    """Per top-level ``Test*`` class, the MAX of its methods' bills in
    ``item_wait_budgets``; module-level test functions are not included.
    Unparseable source returns {}."""
    budgets: dict[str, float] = {}
    for key, budget in item_wait_budgets(source, module_constants).items():
        class_name, separator, _ = key.partition('::')
        if separator:
            budgets[class_name] = max(budget, budgets.get(class_name, budget))
    return budgets


def suppressed_result_wait_methods(source: str) -> dict[str, set[str]]:
    """Statically scan *source* for the executable form of the `outcome is
    None` anti-pattern that
    test_merge_queue_concurrent_verify.py::_await_outcome's docstring names
    ("The shape this replaces"): either (a) a `with contextlib.suppress(TimeoutError):`
    block enclosing an `asyncio.wait_for(...)` call, or (b) a
    `try: outcome = await asyncio.wait_for(...) except TimeoutError: outcome
    = None` block -- an `except` handler catching `TimeoutError` (bare or
    `asyncio.TimeoutError`, alone or inside a tuple) AND whose body
    SWALLOWS the timeout (every statement is either `pass` or an assignment
    of the literal `None`) whose `try:` body contains an
    `asyncio.wait_for(...)` call. Both spellings convert a genuine pipeline
    hang or a missing cascade into a confusing `outcome is None` assertion
    failure instead of failing loudly by label (task 4219; the try/except
    spelling was added in the task 4219 amendment pass after a reviewer
    found it undetected at
    test_merge_queue_concurrent_verify.py::TestHaltAndUnavailable::test_runner_unavailable_quarantine_fallback).

    The swallowing-body requirement on spelling (b) matters: a handler that
    catches `TimeoutError` and then fails LOUDLY -- e.g. `except
    TimeoutError: ...cleanup...; pytest.fail('descriptive RED message')`,
    the shape test_merge_queue_concurrent_verify.py uses at
    `TestOverlapSignal::test_both_verifies_enter_before_either_released`
    and `TestLastItemOfBurstFinalizes::test_single_item_resolves_with_free_remote_slot`
    -- is already the DESIRED behaviour, not the anti-pattern, and matching
    on the caught exception type alone (an earlier version of this scanner,
    task 4219 amendment) false-positived on both (found by running the
    scanner against that module's source, not by review). Conservative
    by construction, matching `_call_wait_budget`'s floor-over-shapes
    stance: an `except TimeoutError:` handler that does anything
    OTHER than `pass` or assign `None` is treated as already-loud and is
    NOT reported, even if it happens to also be confusing in some other way
    -- a false negative here is the safe direction.

    For each top-level ``Test*`` class, reports every ``test_*`` method
    whose body contains at least one such suppressed wait, as a
    ``{class_name: {method_names}}`` mapping. ``suppress(Exception)`` (the
    blessed best-effort teardown-join shape of the worker-task join in
    test_merge_queue_concurrent_verify.py::_stop_worker), a ``suppress(TimeoutError)`` block enclosing no
    ``asyncio.wait_for`` call, a ``try/except`` whose handler does not
    catch ``TimeoutError``, and a ``try/except TimeoutError`` whose handler
    does anything other than swallow (``pass`` / assign ``None``) are all
    deliberately NOT reported.

    Must never raise: a crash here would fail the whole suite over an
    unrelated edit to a scanned file (delegates to `_iter_test_methods`,
    which returns [] on unparseable source). Unparseable source returns {}.
    """

    def _is_timeout_error_arg(node: ast.expr) -> bool:
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
        return is_suppress and any(_is_timeout_error_arg(arg) for arg in node.args)

    def _is_asyncio_wait_for_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'wait_for'
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'asyncio'
        )

    def _is_timeout_error_handler(handler: ast.ExceptHandler) -> bool:
        node = handler.type
        if node is None:
            return False
        if _is_timeout_error_arg(node):
            return True
        return isinstance(node, ast.Tuple) and any(
            _is_timeout_error_arg(elt) for elt in node.elts
        )

    def _is_none_assignment(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is None
        )

    def _is_swallowing_handler(handler: ast.ExceptHandler) -> bool:
        return all(
            isinstance(stmt, ast.Pass) or _is_none_assignment(stmt)
            for stmt in handler.body
        )

    result: dict[str, set[str]] = {}
    for class_name, method in _iter_test_methods(source):
        for node in ast.walk(method):
            if isinstance(node, ast.With):
                if not any(
                    _is_suppress_timeout_error_call(item.context_expr) for item in node.items
                ):
                    continue
                if any(_is_asyncio_wait_for_call(n) for n in ast.walk(node)):
                    result.setdefault(class_name, set()).add(method.name)
                    break
            elif isinstance(node, ast.Try):
                if not any(
                    _is_timeout_error_handler(h) and _is_swallowing_handler(h)
                    for h in node.handlers
                ):
                    continue
                if any(
                    _is_asyncio_wait_for_call(n)
                    for stmt in node.body
                    for n in ast.walk(stmt)
                ):
                    result.setdefault(class_name, set()).add(method.name)
                    break
    return result


def unguarded_worker_teardown_methods(source: str) -> list[str]:
    """Statically scan *source* for the esc-3980-4 leak invariant recorded
    in test_merge_queue_concurrent_verify.py::_stop_worker's docstring: `wait_responsive` gives up by raising
    `_pytest.outcomes.Failed` (a
    ``BaseException`` subclass), so on a straight-line body a give-up --
    like a mid-body ``assert`` -- skips the worker-stop call entirely and
    leaks a live ``SpeculativeMergeWorker`` plus its run task into
    pytest-asyncio teardown. That leak is why one red test used to cascade
    into unrelated failures elsewhere in the session (task 4219).

    For each top-level ``Test*`` class, reports every ``test_*`` method
    that calls ``wait_responsive(...)`` AND calls a worker-stop -- either
    ``worker.stop()`` or the shared ``_stop_worker(worker, ...)`` teardown
    helper -- from somewhere that is NOT inside a ``finally:`` block whose guarded ``try:`` body encloses
    EVERY ``wait_responsive`` call in the method, as a sorted list of
    ``'ClassName::method_name'`` strings. Requiring the ``try:`` to enclose
    every ``wait_responsive`` call (not just crediting any ``finally`` in
    the method) matters: a give-up from a ``wait_responsive`` call that
    sits OUTSIDE that ``try:`` -- e.g. before it -- is not caught by its
    ``finally`` either, so the leak window stays open even though a
    finally-wrapped stop call is present somewhere in the method
    (task 4219 amendment; found by review). Recognising the ``_stop_worker``
    helper alongside ``worker.stop()`` matters for the same reason: a
    straight-line ``await _stop_worker(worker, worker_task)`` leaks exactly
    like a straight-line ``await worker.stop()`` would, and the guard must
    not go quietly vacuous for a method that adopts the OTHER blessed
    teardown shape.

    Deliberately scoped to methods that call ``wait_responsive``: scanned
    modules carry pre-existing straight-line teardown sites that have not
    been migrated yet, and this guard's job is to stop the migrated set
    from regressing, not to flag every unmigrated one. A method calling
    ``wait_responsive`` but never a worker-stop at all is not reported
    either -- there is no stop call to be unguarded.

    Deliberately narrow to a variable literally named ``worker``: the
    merge-lane tests name the worker ``worker`` at every relevant site, and
    a false NEGATIVE (missing an oddly-named variable) is the correct
    failure direction for a guard whose job is to stop a known regression
    -- matching `_call_wait_budget`'s documented conservative-floor stance.

    Must never raise: a crash here would fail the whole suite over an
    unrelated edit to a scanned file (delegates to `_iter_test_methods`, which
    returns [] on unparseable source). Unparseable source returns [].
    """

    def _is_wait_responsive_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'wait_responsive'
        )

    def _is_worker_stop_call(node: ast.AST) -> bool:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'stop'
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'worker'
        ):
            return True
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == '_stop_worker'
        )

    offenders: list[str] = []
    for class_name, method in _iter_test_methods(source):
        wait_calls = [n for n in ast.walk(method) if _is_wait_responsive_call(n)]
        if not wait_calls:
            continue

        stop_calls = [n for n in ast.walk(method) if _is_worker_stop_call(n)]
        if not stop_calls:
            continue

        guarded: set[ast.AST] = set()
        for try_node in ast.walk(method):
            if not isinstance(try_node, ast.Try):
                continue
            try_body_ids = {id(n) for stmt in try_node.body for n in ast.walk(stmt)}
            if not all(id(w) in try_body_ids for w in wait_calls):
                # This try's body does not enclose every wait_responsive
                # call in the method, so a give-up raised by one it does
                # NOT enclose is not caught by its finally -- the esc-3980-4
                # leak window stays open. Do not credit its finalbody stop
                # call(s) as guarding anything.
                continue
            for final_stmt in try_node.finalbody:
                for n in ast.walk(final_stmt):
                    if _is_worker_stop_call(n):
                        guarded.add(n)

        if any(call not in guarded for call in stop_calls):
            offenders.append(f'{class_name}::{method.name}')

    return sorted(offenders)


# The per-test budget an UNMARKED class actually runs under: the tighter of
# the ini default (a bare local run) and the `--timeout` verify passes on its
# CLI (the merge-gating run).  Why the two differ, and why verify's is the
# tighter one, is recorded once at `_orch_helpers.VERIFY_CLI_PER_TEST_TIMEOUT`.
AMBIENT_TEST_TIMEOUT = min(PYPROJECT_DEFAULT_TIMEOUT, VERIFY_CLI_PER_TEST_TIMEOUT)


def timeout_mark_offenders(
    budgets: dict[str, float],
    resolve: Callable[[str], object | None],
) -> list[str]:
    """Pure offender-accumulation for the timeout-mark coverage guard: every
    class's computed worst-case per-method *budget* must be cleared by the
    timeout it actually runs under.

    For each ``class_name -> budget`` pair, in sorted order:

    - a budget <= 0 has nothing to cover and is skipped;
    - a *class_name* that *resolve* cannot look up is an offender;
    - a class carrying a ``timeout`` mark runs under THAT mark, whatever
      the ambient budget -- a mark replaces it rather than raising a floor
      under it -- so it is an offender iff the mark value is None or below
      *budget*, at any budget;
    - an unmarked class runs under ``AMBIENT_TEST_TIMEOUT``, so it is an
      offender iff *budget* reaches it.

    Returns one formatted offender string per failing class; an empty list
    means every class is adequately covered.

    Pure (independent of any module's ``globals()``) so its failure
    branches -- unresolvable class name, missing mark, mark value too
    tight -- are each directly testable with synthetic stubs.
    """
    offenders: list[str] = []
    for class_name, budget in sorted(budgets.items()):
        if budget <= 0:
            continue
        budget_text = (
            'unbounded (some wait in the method has no resolvable '
            'wall-clock cap)'
            if math.isinf(budget)
            else f'{budget}s'
        )

        cls = resolve(class_name)
        if cls is None:
            offenders.append(
                f'{class_name}: computed budget {budget_text}, but the '
                f'class could not be resolved from module globals to '
                f'inspect its marks.'
            )
            continue

        mark = next(
            (m for m in getattr(cls, 'pytestmark', []) if m.name == 'timeout'),
            None,
        )
        if mark is None:
            if budget < AMBIENT_TEST_TIMEOUT:
                continue
            offenders.append(
                f'{class_name}: computed worst-case per-method wait '
                f'budget is {budget_text}, reaching the '
                f'{AMBIENT_TEST_TIMEOUT}s an unmarked class runs under '
                f'(the tighter of the {PYPROJECT_DEFAULT_TIMEOUT}s ini '
                f'default and verify\'s --timeout='
                f'{VERIFY_CLI_PER_TEST_TIMEOUT}), but the class carries '
                f'no @pytest.mark.timeout mark at all.'
            )
            continue

        mark_value = mark.args[0] if mark.args else None
        if mark_value is None or mark_value < budget:
            offenders.append(
                f'{class_name}: computed worst-case per-method wait '
                f'budget is {budget_text}, but its @pytest.mark.timeout '
                f'mark is only {mark_value!r} -- too tight to clear '
                f'it.'
            )
    return offenders
