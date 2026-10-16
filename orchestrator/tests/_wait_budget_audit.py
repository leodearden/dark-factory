"""Static AST audit of the real wall clock a test's waits can consume,
beside the timeout each pytest item actually runs under.

Pure: source text and a module namespace in, numbers and names out -- no
file I/O, no imports of the code it scans -- so every function here can be
driven with foreign input. Callers own reading files and importing modules;
``audit_module`` takes the import as a callback so it runs only when there
is something to bill.

The bill is a conservative estimate of the wall clock each pytest item's
recognised waits can take (``_call_wait_budget``); the limit is the timeout
pytest-timeout applies to the item (``_item_limit``).
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import math
from collections.abc import Callable, Mapping

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


def _as_seconds(value: object) -> float | None:
    """*value* as a duration in seconds, or None when it is not a plain
    number (a bool is not a duration). A negative number lasts no time, so
    0.0."""
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
        seconds = _as_seconds(namespace.get(node.id))
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
        seconds = _as_seconds(node.value)
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


def is_asyncio_wait_for_call(node: ast.AST) -> bool:
    """Whether *node* is an ``asyncio.wait_for(...)`` call spelled through
    the ``asyncio`` name: ``gate.wait_for(...)`` and ``self.wait_for(...)``
    are not."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'wait_for'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'asyncio'
    )


def _call_wait_budget(call: ast.Call, visible: Mapping[str, float]) -> float:
    """Return the wait-budget contribution of a single ``ast.Call`` node.

    Recognises exactly four call shapes -- ``asyncio.wait_for(..., timeout=N)``
    (``is_asyncio_wait_for_call``), ``_await_outcome(...)`` (whose hidden default is ``MERGE_RESULT_TIMEOUT``
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
    if is_asyncio_wait_for_call(call):
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


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def collect_test_items(source: str) -> list[tuple[str | None, _FunctionNode]]:
    """``(owner_class_name, function)`` for every pytest item *source*
    defines: each ``test*`` method directly in a top-level ``Test*`` class,
    and each top-level ``test*`` function (owner None). Those are pytest's
    default ``python_classes`` / ``python_functions`` prefixes, which
    orchestrator/pyproject.toml does not override. Nested classes, inherited
    methods and conditionally defined classes are out of scope.

    The one traversal behind every scan of test items, here and in
    _wait_hygiene_scan.py. Must never raise: unparseable source has no
    items.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    items: list[tuple[str | None, _FunctionNode]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('test'):
                items.append((None, node))
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            items.extend(
                (node.name, method)
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name.startswith('test')
            )
    return items


def _item_bills(
    items: list[tuple[str | None, _FunctionNode]],
    module_constants: Mapping[str, object] | None,
) -> dict[tuple[str | None, str], float]:
    """Each pytest item's bill, keyed ``(owner_class_name, function_name)``;
    a name defined twice keeps the larger bill."""
    bills: dict[tuple[str | None, str], float] = {}
    for owner, function in items:
        key = (owner, function.name)
        budget = _method_wait_budget(function, module_constants)
        bills[key] = max(budget, bills.get(key, budget))
    return bills


_ITEM_SEPARATOR = '::'


def item_id(owner: str | None, function_name: str) -> str:
    """The item's id within its module: ``'Class::method'``, or a
    module-level function's bare name."""
    return function_name if owner is None else f'{owner}{_ITEM_SEPARATOR}{function_name}'


def is_module_level_item(item: str) -> bool:
    """Whether *item*, an id this module built, names a module-level test
    function rather than a class's test method."""
    return _ITEM_SEPARATOR not in item


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
    bills = _item_bills(collect_test_items(source), module_constants)
    return {item_id(owner, name): budget for (owner, name), budget in bills.items()}


# The per-test budget an UNMARKED class actually runs under: the tighter of
# the ini default (a bare local run) and the `--timeout` verify passes on its
# CLI (the merge-gating run).  Why the two differ, and why verify's is the
# tighter one, is recorded once at `_orch_helpers.VERIFY_CLI_PER_TEST_TIMEOUT`.
AMBIENT_TEST_TIMEOUT = min(PYPROJECT_DEFAULT_TIMEOUT, VERIFY_CLI_PER_TEST_TIMEOUT)


class LimitSource(enum.Enum):
    """Where the timeout a pytest item runs under comes from."""

    FUNCTION_MARK = "the test's own @pytest.mark.timeout"
    CLASS_MARK = "its class's @pytest.mark.timeout"
    MODULE_MARK = "its module's pytestmark timeout"
    AMBIENT = (
        f'the ambient budget, since no timeout mark sets seconds: the tighter '
        f"of the {PYPROJECT_DEFAULT_TIMEOUT}s ini default and verify's "
        f'--timeout={VERIFY_CLI_PER_TEST_TIMEOUT}'
    )
    UNRESOLVED = 'unreadable: the test is not in its module namespace'

    @property
    def is_mark(self) -> bool:
        """Whether the limit came from a timeout mark, at any level."""
        return self not in (LimitSource.AMBIENT, LimitSource.UNRESOLVED)


@dataclasses.dataclass(frozen=True)
class ItemWaitAudit:
    """One pytest item's wait bill beside the timeout it runs under.
    *limit* is None when no number could be read for it, and ``math.inf``
    when its mark disables the timeout."""

    item: str
    budget: float
    limit: float | None
    limit_source: LimitSource

    @property
    def offends(self) -> bool:
        """Whether the bill breaches the limit: under the ambient budget an
        item offends once its bill REACHES it; under a mark, when the mark
        is below the bill (a mark replaces the ambient budget, at any bill),
        so never when the mark disables the timeout; with no readable limit
        -- unresolved, or an unreadable mark -- always."""
        if self.limit is None:
            return True
        if self.limit_source is LimitSource.AMBIENT:
            return self.budget >= self.limit
        return self.limit < self.budget

    def describe(self) -> str:
        bill = (
            'unbounded (some wait has no resolvable wall-clock cap)'
            if math.isinf(self.budget)
            else f'{self.budget}s'
        )
        if self.limit is None:
            limit = 'none'
        elif math.isinf(self.limit):
            limit = 'disabled'
        else:
            limit = f'{self.limit}s'
        return (
            f'{self.item}: waits bill {bill}; limit {limit}, from '
            f'{self.limit_source.value}'
        )


def _mark_list(carrier: object) -> list[object]:
    """A ``pytestmark`` value as a list: one MarkDecorator or Mark, or a
    list of them; absent is empty."""
    if carrier is None:
        return []
    return list(carrier) if isinstance(carrier, (list, tuple)) else [carrier]


def _class_marks(owner_cls: object) -> list[object]:
    """The marks pytest reads for a class: every class in its MRO
    contributes the marks it was itself decorated with, base classes FIRST,
    as ``_pytest.mark.structures.get_unpacked_marks`` collects them. A plain
    ``getattr`` would see only the nearest decorated class's list."""
    if not isinstance(owner_cls, type):
        return _mark_list(getattr(owner_cls, 'pytestmark', None))
    return [
        mark
        for klass in reversed(owner_cls.__mro__)
        for mark in _mark_list(klass.__dict__.get('pytestmark'))
    ]


def _closest_timeout_mark(
    function_obj: object,
    owner_cls: object | None,
    module_namespace: Mapping[str, object],
) -> tuple[object, LimitSource] | None:
    """The timeout mark pytest-timeout applies to an item, and the level it
    sits at: the first one in the test's own marks, then its class's
    (``_class_marks``), then the module's ``pytestmark``. None when no level
    carries one.

    The closest level wins, as in ``_pytest.nodes.Node.get_closest_marker``;
    why a mark then wins over the CLI/ini budget outright is recorded in
    shared/src/shared/testing_timeout_markers.py.
    """
    levels = (
        (_mark_list(getattr(function_obj, 'pytestmark', None)), LimitSource.FUNCTION_MARK),
        (_class_marks(owner_cls), LimitSource.CLASS_MARK),
        (_mark_list(module_namespace.get('pytestmark')), LimitSource.MODULE_MARK),
    )
    for marks, source in levels:
        for mark in marks:
            if getattr(mark, 'name', None) == 'timeout':
                return getattr(mark, 'mark', mark), source
    return None


def _mark_limit(mark: object, source: LimitSource) -> tuple[float | None, LimitSource]:
    """The limit a timeout mark sets, read as pytest-timeout reads it: its
    first argument, else its ``timeout=`` keyword.

    A mark that sets no seconds (``method=`` alone, or ``timeout=None``)
    leaves the ini/CLI timeout in force, so the ambient budget -- a farther
    mark does not step in. Seconds <= 0 disable the timeout, so
    ``math.inf``. Seconds that are not a plain number are unreadable: None.
    """
    args = getattr(mark, 'args', ())
    value = args[0] if args else getattr(mark, 'kwargs', {}).get('timeout')
    if value is None:
        return float(AMBIENT_TEST_TIMEOUT), LimitSource.AMBIENT
    seconds = _as_seconds(value)
    if seconds is None:
        return None, source
    return (math.inf if seconds == 0 else seconds), source


def _item_limit(
    owner: str | None, function_name: str, namespace: Mapping[str, object]
) -> tuple[float | None, LimitSource]:
    """The timeout an item runs under and where it comes from, read off the
    item's real objects in *namespace*."""
    owner_cls = None if owner is None else namespace.get(owner)
    function_obj = (
        namespace.get(function_name)
        if owner is None
        else getattr(owner_cls, function_name, None)
    )
    if function_obj is None:
        return None, LimitSource.UNRESOLVED
    closest = _closest_timeout_mark(function_obj, owner_cls, namespace)
    if closest is None:
        return float(AMBIENT_TEST_TIMEOUT), LimitSource.AMBIENT
    return _mark_limit(*closest)


def audit_module(
    source: str, load_namespace: Callable[[], Mapping[str, object] | None]
) -> list[ItemWaitAudit] | None:
    """Every item of *source* with a wait bill above zero, beside the
    timeout it runs under, sorted by item. *source* is parsed once.

    *load_namespace* returns the module's runtime globals, which resolve the
    names its waits spell and hold the objects whose marks are read. It is
    called only when some item has a recognised wait: an item that bills
    nothing without the namespace bills nothing with it, since a name
    resolves to no more than unbounded.

    None when there is nothing to audit: *source* does not parse, no item
    bills, or *load_namespace* returns None. Never raises on its own
    account.
    """
    items = collect_test_items(source)
    if not any(_method_wait_budget(function) > 0 for _, function in items):
        return None
    namespace = load_namespace()
    if namespace is None:
        return None
    audits = [
        ItemWaitAudit(item_id(owner, name), budget, *_item_limit(owner, name, namespace))
        for (owner, name), budget in _item_bills(items, namespace).items()
        if budget > 0
    ]
    return sorted(audits, key=lambda audit: audit.item)
