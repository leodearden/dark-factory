"""Structural guards on how the curator and the task interceptor fail open, or defer.

All four guard one hazard: a curation that no judgement stands behind, yet
which no signal distinguishes from a healthy one.

1. ``TaskCurator`` issues its LLM call from one method, ``_dispatch_llm``, in
   one ``try``, and every exception arm of that ``try`` hands the exception to
   the one failure mapper, ``_failure_decision``. Every return of the mapper
   is a ``CuratorDecision.deferred`` (a defer creates nothing, so it cannot
   pass for a healthy create) or a ``_degraded_create`` (which counts it), and
   the mapper reports to the escalator through ``_escalate_failure``, which
   calls ``report_failure``. Neither method constructs a ``CuratorDecision``
   directly.
2. No ``TaskCurator`` method builds an ``action='create'`` decision except the
   ``_degraded_create`` funnel, which is what advances the degraded streak.
3. Every ``CuratorDecision(action='create', ...)`` in ``task_curator.py``
   and in ``task_interceptor.py`` states its ``degraded=`` classification
   explicitly, as a literal. In the interceptor that literal decides whether
   the created task carries the ``dedupe_skipped`` marker.
4. ``CuratorDecision.deferred`` (in ``curator_types.py``) is the only builder
   of ``action='defer'`` in the vocabulary, the curator and the interceptor, so
   every defer carries a ``DeferReason``.

The behavioural halves live in test_task_curator.py and
test_curation_pipeline.py. Those pin the paths that exist today; these assert
over the shapes themselves, so the requirements land on an arm or branch added
tomorrow without anyone remembering to extend a list. Asserted over the PARSED
module, so prose that merely mentions a name cannot satisfy a check.

Deliberately NOT asserted: which exception types the mapper distinguishes,
their order, what it passes to ``report_failure``, or which way a branch is
classified. Those are design choices; the invariants are only that no failure
escapes the mapper undecided, no create is uncounted, no fail-open leaves its
classification to a default, and no defer is built without its reason.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, parse_python_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SRC_ROOT = pathlib.Path(__file__).parents[1] / 'src' / 'fused_memory'
TASK_CURATOR = SRC_ROOT / 'middleware' / 'task_curator.py'
CURATOR_TYPES = SRC_ROOT / 'middleware' / 'curator_types.py'
TASK_INTERCEPTOR = SRC_ROOT / 'middleware' / 'task_interceptor.py'

CURATOR_CLASS = 'TaskCurator'
LLM_CALL = '_invoke_curator'
DISPATCH = '_dispatch_llm'
MAPPER = '_failure_decision'
ESCALATE = '_escalate_failure'
REPORT_CALL = 'report_failure'
DEFER_FUNNEL = 'deferred'

# The single funnel every degraded decision must leave through, so that
# counting degradations does not mean enumerating their causes.
DECISION_TYPE = 'CuratorDecision'
DEGRADED_FUNNEL = '_degraded_create'


def _curator_method(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for method in _curator_methods():
        if method.name == name:
            return method
    raise AssertionError(
        f'{TASK_CURATOR.name}: no def {name}() inside class {CURATOR_CLASS} — '
        f'this guard has lost its subject and is asserting nothing; re-point it '
        f'rather than deleting it.'
    )


def _llm_try() -> ast.Try:
    """The one ``try`` in ``_dispatch_llm`` whose body issues the LLM call."""
    matches = [
        node
        for node in ast.walk(_curator_method(DISPATCH))
        if isinstance(node, ast.Try)
        and any(calls_named(stmt, LLM_CALL) for stmt in node.body)
    ]
    assert len(matches) == 1, (
        f'expected exactly one try: block in {CURATOR_CLASS}.{DISPATCH} '
        f'whose body calls {LLM_CALL}(), found {len(matches)} '
        f'(lines {[node.lineno for node in matches]})'
    )
    return matches[0]


def _handler_label(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return 'bare except'
    return f'except {ast.unparse(handler.type)}'


def test_only_the_dispatch_calls_the_llm():
    callers = sorted(
        method.name for method in _curator_methods() if calls_named(method, LLM_CALL)
    )
    assert callers == [DISPATCH], (
        f'{CURATOR_CLASS} calls {LLM_CALL}() from {callers}, expected only '
        f'{DISPATCH}(). A second call site is a second set of exception arms '
        f'for this guard to miss; route the call through {DISPATCH}().'
    )
    _llm_try()


def test_every_llm_exception_arm_hands_the_failure_to_the_mapper():
    handlers = _llm_try().handlers
    assert handlers, (
        f'{CURATOR_CLASS}.{DISPATCH} calls {LLM_CALL}() inside a try with no '
        f'exception arm — re-point this guard.'
    )
    unmapped = [handler for handler in handlers if not calls_named(handler, MAPPER)]
    assert not unmapped, (
        f'{CURATOR_CLASS}.{DISPATCH} has exception arm(s) around {LLM_CALL}() '
        f'that never call {MAPPER}(): '
        + ', '.join(
            f'{_handler_label(handler)} at line {handler.lineno}'
            for handler in unmapped
        )
        + '. A curator failure that reaches a silent arm still returns '
        "action='create', which is indistinguishable downstream from a healthy "
        'create — the shape of the 2026-08-13 outage. Decide it in the one '
        f'mapper, {MAPPER}().'
    )


def _returned_call_name(node: ast.Return) -> str | None:
    value = node.value
    if isinstance(value, ast.Await):
        value = value.value
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_every_mapper_return_is_a_defer_or_a_counted_create():
    mapper = _curator_method(MAPPER)
    returns = [node for node in ast.walk(mapper) if isinstance(node, ast.Return)]
    assert returns, f'{CURATOR_CLASS}.{MAPPER} returns nothing — re-point this guard.'
    stray = [
        node for node in returns
        if _returned_call_name(node) not in (DEFER_FUNNEL, DEGRADED_FUNNEL)
    ]
    assert not stray, (
        f'{CURATOR_CLASS}.{MAPPER} returns something other than '
        f'{DECISION_TYPE}.{DEFER_FUNNEL}(...) or {DEGRADED_FUNNEL}(...) at '
        f'line(s) {sorted(node.lineno for node in stray)}. A failure must '
        'either defer, which creates nothing, or create through the funnel '
        'that counts it on the degraded streak.'
    )
    assert calls_named(mapper, ESCALATE) and calls_named(_curator_method(ESCALATE), REPORT_CALL), (
        f'{CURATOR_CLASS}.{MAPPER} never reaches {REPORT_CALL}() through '
        f'{ESCALATE}(): an LLM failure would be decided without the escalator '
        'hearing of it.'
    )


def test_dispatch_and_mapper_construct_no_decision_directly():
    """Degraded decisions leave through one funnel, so the streak alarm needs
    no enumerated list of degraded reasons to stay hole-free."""
    direct = [
        (name, call.lineno)
        for name in (DISPATCH, MAPPER)
        for call in calls_named(_curator_method(name), DECISION_TYPE)
    ]
    assert not direct, (
        f'{CURATOR_CLASS} constructs {DECISION_TYPE} directly in '
        + ', '.join(f'{name}() at line {line}' for name, line in direct)
        + f'. Every degraded decision must be built by {DEGRADED_FUNNEL}() '
        'instead, which is what increments the degraded streak — a decision '
        'constructed here bypasses the counter and is invisible to the alarm.'
    )


def _curator_methods() -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = parse_python_module(TASK_CURATOR)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == CURATOR_CLASS:
            return [
                child for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    raise AssertionError(
        f'{TASK_CURATOR.name}: no class {CURATOR_CLASS} — this guard has lost '
        f'its subject; re-point it rather than deleting it.'
    )


def _is_create_construction(call: ast.Call) -> bool:
    action = next((kw.value for kw in call.keywords if kw.arg == 'action'), None)
    return isinstance(action, ast.Constant) and action.value == 'create'


def test_no_curator_method_but_the_funnel_builds_a_create():
    """The whole class, not just the dispatch: a create the curator builds for
    itself — rather than parsing from a model response — is by definition a
    fail-open, so it must be counted, at every batch size."""
    methods = _curator_methods()
    assert any(m.name == DEGRADED_FUNNEL for m in methods), (
        f'{CURATOR_CLASS} has no {DEGRADED_FUNNEL}() — re-point this guard.'
    )
    rogue = [
        (method.name, call.lineno)
        for method in methods
        if method.name != DEGRADED_FUNNEL
        for call in calls_named(method, DECISION_TYPE)
        if _is_create_construction(call)
    ]
    assert not rogue, (
        f'{CURATOR_CLASS} method(s) construct an action=\'create\' '
        f'{DECISION_TYPE} directly: '
        + ', '.join(f'{name}() at line {line}' for name, line in rogue)
        + f'. Build it through {DEGRADED_FUNNEL}() so the degraded streak '
        'counts it; a create built anywhere else is invisible to the alarm.'
    )


# --- Invariant 3: every fail-open create states its classification ---------
#
# Module-level parsers have no curator to count against, so they mark
# degradation on the decision itself and the caller counts it. The
# interceptor's batch fallbacks build creates no verdict judged, and there
# task_interceptor.py::_dedupe_skipped_reason stamps the dedupe_skipped marker
# only when ``decision.degraded``: a fallback left at the False default would
# ship a fail-open create unmarked. Whether a branch is degraded is therefore
# part of the data each branch writes, never a default nobody chose.


def _module_create_constructions(module: pathlib.Path) -> list[ast.Call]:
    return [
        call
        for call in calls_named(parse_python_module(module), DECISION_TYPE)
        if _is_create_construction(call)
    ]


def _declares_degradedness(call: ast.Call) -> bool:
    degraded = next((kw.value for kw in call.keywords if kw.arg == 'degraded'), None)
    return isinstance(degraded, ast.Constant) and isinstance(degraded.value, bool)


def _justification_text(call: ast.Call) -> str:
    justification = next(
        (kw.value for kw in call.keywords if kw.arg == 'justification'), None
    )
    return ast.unparse(justification) if justification is not None else '<none>'


@pytest.mark.parametrize(
    'module',
    [
        pytest.param(TASK_CURATOR, id='curator'),
        pytest.param(TASK_INTERCEPTOR, id='interceptor'),
    ],
)
def test_every_fail_open_create_declares_degradedness(module: pathlib.Path):
    creates = _module_create_constructions(module)
    assert creates, (
        f'no {DECISION_TYPE}(action=\'create\') in {module.name} — this '
        f'guard has lost its subject; re-point it rather than deleting it.'
    )
    unclassified = [call for call in creates if not _declares_degradedness(call)]
    assert not unclassified, (
        f'{module.name} builds fail-open create(s) without a literal '
        'degraded=True/False: '
        + ', '.join(
            f'line {call.lineno} ({_justification_text(call)})'
            for call in unclassified
        )
        + '. Classify each by what its verdict DEPENDS ON. degraded=True when '
        'no usable decision was obtained, or only the RESPONSE was consulted '
        '(the model broke the output contract): sustained, that is an outage '
        'the streak alarm must see, and the created task must carry the '
        'dedupe_skipped marker. degraded=False when POOL STATE declined an '
        'otherwise usable decision: routine on a healthy curator, so counting '
        'it would fire the alarm on a working service. Give each cause its own '
        'branch — a condition ORing a payload failure into a state veto hands '
        'the payload failure the veto\'s classification.'
    )


# --- Invariant 4: one builder of a defer ------------------------------------


def _decision_class() -> ast.ClassDef:
    for node in ast.walk(parse_python_module(CURATOR_TYPES)):
        if isinstance(node, ast.ClassDef) and node.name == DECISION_TYPE:
            return node
    raise AssertionError(
        f'{CURATOR_TYPES.name}: no class {DECISION_TYPE} — this guard has lost '
        f'its subject; re-point it rather than deleting it.'
    )


def _defer_funnel() -> ast.FunctionDef:
    for child in _decision_class().body:
        if isinstance(child, ast.FunctionDef) and child.name == DEFER_FUNNEL:
            return child
    raise AssertionError(
        f'{DECISION_TYPE} has no {DEFER_FUNNEL}() — this guard has lost its '
        f'subject; re-point it rather than deleting it.'
    )


def _builds_a_defer(call: ast.Call) -> bool:
    action = next((kw.value for kw in call.keywords if kw.arg == 'action'), None)
    return isinstance(action, ast.Constant) and action.value == 'defer'


def test_no_module_builds_a_defer_but_its_constructor():
    """Any call passing ``action='defer'`` — ``CuratorDecision(...)``,
    ``cls(...)`` or ``dataclasses.replace(...)`` — outside the funnel would
    yield a defer with no ``DeferReason`` for the store to park it under."""
    funnel_calls = {
        id(node) for node in ast.walk(_defer_funnel()) if isinstance(node, ast.Call)
    }
    assert any(
        _builds_a_defer(node)
        for node in ast.walk(_defer_funnel())
        if isinstance(node, ast.Call)
    ), f'{DECISION_TYPE}.{DEFER_FUNNEL}() builds no action=\'defer\' — re-point this guard.'
    rogue = [
        (module.name, node.lineno)
        for module in (CURATOR_TYPES, TASK_CURATOR, TASK_INTERCEPTOR)
        for node in ast.walk(parse_python_module(module))
        if isinstance(node, ast.Call)
        and _builds_a_defer(node)
        and id(node) not in funnel_calls
    ]
    assert not rogue, (
        "action='defer' built outside "
        f'{DECISION_TYPE}.{DEFER_FUNNEL}(): '
        + ', '.join(f'{name} line {line}' for name, line in rogue)
        + f'. Build every defer through {DECISION_TYPE}.{DEFER_FUNNEL}(reason), '
        'which is what guarantees it carries the reason it is parked under.'
    )
