"""Structural guards on how ``task_curator.py`` fails open, or defers.

All four guard one hazard: a curation that no judgement stands behind, yet
which no signal distinguishes from a healthy one.

1. Every exception arm around ``TaskCurator.curate()``'s LLM call either
   routes through ``report_failure`` or returns ``CuratorDecision.deferred``.
   A defer creates nothing, so it cannot pass for a healthy create.
2. No ``TaskCurator`` method builds an ``action='create'`` decision except the
   ``_degraded_create`` funnel, which is what advances the degraded streak.
3. Every ``CuratorDecision(action='create', ...)`` in the module states its
   ``degraded=`` classification explicitly, as a literal.
4. ``CuratorDecision.deferred`` (in ``curator_types.py``) is the only builder
   of ``action='defer'`` in the vocabulary, the curator and the interceptor, so
   every defer carries a ``DeferReason``.

The behavioural halves live in test_task_curator.py. Those pin the paths that
exist today; these assert over the shapes themselves, so the requirements land
on an arm or branch added tomorrow without anyone remembering to extend a list.
Asserted over the PARSED module, so prose that merely mentions a name cannot
satisfy a check.

Deliberately NOT asserted: which exception types the arms catch, their order,
what each passes to ``report_failure``, or which way a branch is classified.
Those are design choices; the invariants are only that no arm is silent, no
create is uncounted, no fail-open leaves its classification to a default, and
no defer is built without its reason.
"""

from __future__ import annotations

import ast
import pathlib

from _ast_guard import calls_named, parse_python_module

SRC_ROOT = pathlib.Path(__file__).parents[1] / 'src' / 'fused_memory'
TASK_CURATOR = SRC_ROOT / 'middleware' / 'task_curator.py'
CURATOR_TYPES = SRC_ROOT / 'middleware' / 'curator_types.py'
TASK_INTERCEPTOR = SRC_ROOT / 'middleware' / 'task_interceptor.py'

CURATOR_CLASS = 'TaskCurator'
CURATE_METHOD = 'curate'

# The batch path calls the same invoker, but from
# ``_call_llm_batch_with_fallback`` rather than ``curate()``, so the one
# ``try`` in ``curate()`` that calls it is still the single-candidate call.
LLM_CALL = '_invoke_curator'
REPORT_CALL = 'report_failure'
DEFER_FUNNEL = 'deferred'

# The single funnel every degraded decision must leave through, so that
# counting degradations does not mean enumerating their causes.
DECISION_TYPE = 'CuratorDecision'
DEGRADED_FUNNEL = '_degraded_create'

# The arms as of task 4503: the pool-unavailable errors (which defer),
# CuratorFailureError, and the catch-all. A handler list shorter than this
# means an arm was deleted rather than instrumented, which would satisfy the
# silence check vacuously.
EXPECTED_ARMS = 3


def _curate_def() -> ast.AsyncFunctionDef:
    tree = parse_python_module(TASK_CURATOR)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == CURATOR_CLASS):
            continue
        for child in node.body:
            if isinstance(child, ast.AsyncFunctionDef) and child.name == CURATE_METHOD:
                return child
    raise AssertionError(
        f'{TASK_CURATOR.name}: no async def {CURATE_METHOD}() inside '
        f'class {CURATOR_CLASS} — this guard has lost its subject and is '
        f'asserting nothing; re-point it rather than deleting it.'
    )


def _llm_try() -> ast.Try:
    """The one ``try`` in ``curate()`` whose body issues the LLM call."""
    curate = _curate_def()
    matches = [
        node
        for node in ast.walk(curate)
        if isinstance(node, ast.Try)
        and any(calls_named(stmt, LLM_CALL) for stmt in node.body)
    ]
    assert len(matches) == 1, (
        f'expected exactly one try: block in {CURATOR_CLASS}.{CURATE_METHOD} '
        f'whose body calls {LLM_CALL}(), found {len(matches)} '
        f'(lines {[node.lineno for node in matches]})'
    )
    return matches[0]


def _handler_label(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return 'bare except'
    return f'except {ast.unparse(handler.type)}'


def test_llm_call_is_wrapped_in_the_expected_arms():
    handlers = _llm_try().handlers
    assert len(handlers) >= EXPECTED_ARMS, (
        f'{CURATOR_CLASS}.{CURATE_METHOD} guards {LLM_CALL}() with '
        f'{len(handlers)} exception arm(s), expected at least {EXPECTED_ARMS}: '
        + ', '.join(
            f'{_handler_label(handler)} at line {handler.lineno}'
            for handler in handlers
        )
    )


def test_every_llm_exception_arm_reports_failure_or_defers():
    silent = [
        handler
        for handler in _llm_try().handlers
        if not calls_named(handler, REPORT_CALL)
        and not calls_named(handler, DEFER_FUNNEL)
    ]
    assert not silent, (
        f'{CURATOR_CLASS}.{CURATE_METHOD} has exception arm(s) around '
        f'{LLM_CALL}() that neither call {REPORT_CALL}() nor return '
        f'{DECISION_TYPE}.{DEFER_FUNNEL}(): '
        + ', '.join(
            f'{_handler_label(handler)} at line {handler.lineno}'
            for handler in silent
        )
        + '. A curator failure that reaches a silent arm still returns '
        "action='create', which is indistinguishable downstream from a healthy "
        'create — the shape of the 2026-08-13 outage. Route it through the '
        f'escalator ({REPORT_CALL}) as the sibling arms do, or defer it.'
    )


def test_curate_constructs_no_decision_directly():
    """Degraded decisions leave through one funnel, so the streak alarm needs
    no enumerated list of degraded reasons to stay hole-free."""
    direct = calls_named(_curate_def(), DECISION_TYPE)
    assert not direct, (
        f'{CURATOR_CLASS}.{CURATE_METHOD} constructs {DECISION_TYPE} directly at '
        f'line(s) {sorted(call.lineno for call in direct)}. Every degraded '
        f'decision must be built by {DEGRADED_FUNNEL}() instead, which is what '
        f'increments the degraded streak — a decision constructed here bypasses '
        f'the counter and is invisible to the alarm.'
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
    """The whole class, not just curate(): a create the curator builds for
    itself — rather than parsing from a model response — is by definition a
    fail-open, so it must be counted, on the batch path as on the single one."""
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
# degradation on the decision itself and the caller counts it. Whether a
# branch is degraded is therefore part of the data each branch writes, never
# a default nobody chose.


def _module_create_constructions() -> list[ast.Call]:
    return [
        call
        for call in calls_named(parse_python_module(TASK_CURATOR), DECISION_TYPE)
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


def test_every_fail_open_create_declares_degradedness():
    creates = _module_create_constructions()
    assert creates, (
        f'no {DECISION_TYPE}(action=\'create\') in {TASK_CURATOR.name} — this '
        f'guard has lost its subject; re-point it rather than deleting it.'
    )
    unclassified = [call for call in creates if not _declares_degradedness(call)]
    assert not unclassified, (
        f'{TASK_CURATOR.name} builds fail-open create(s) without a literal '
        'degraded=True/False: '
        + ', '.join(
            f'line {call.lineno} ({_justification_text(call)})'
            for call in unclassified
        )
        + '. Classify each by what its verdict DEPENDS ON. degraded=True when '
        'no usable decision was obtained, or only the RESPONSE was consulted '
        '(the model broke the output contract): sustained, that is an outage '
        'the streak alarm must see. degraded=False when POOL STATE declined an '
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
