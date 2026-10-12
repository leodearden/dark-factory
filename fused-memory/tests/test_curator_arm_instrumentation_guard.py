"""Structural guards on how ``task_curator.py`` fails open.

All three guard one hazard: a degraded curation that no signal distinguishes
from a healthy one, which returns ``action='create'`` exactly as a healthy
create does.

1. Every exception arm around ``TaskCurator.curate()``'s LLM call routes
   through ``report_failure``.
2. No ``TaskCurator`` method builds an ``action='create'`` decision except the
   ``_degraded_create`` funnel, which is what advances the degraded streak.
3. Every ``CuratorDecision(action='create', ...)`` in the module states its
   ``degraded=`` classification explicitly, as a literal.

The behavioural halves live in test_task_curator.py. Those pin the paths that
exist today; these assert over the shapes themselves, so the requirements land
on an arm or branch added tomorrow without anyone remembering to extend a list.
Asserted over the PARSED module, so prose that merely mentions a name cannot
satisfy a check.

Deliberately NOT asserted: which exception types the arms catch, their order,
what each passes to ``report_failure``, or which way a branch is classified.
Those are design choices; the invariants are only that no arm is silent, no
create is uncounted, and no fail-open leaves its classification to a default.
"""

from __future__ import annotations

import ast
import pathlib

from _ast_guard import calls_named, parse_python_module

SRC_ROOT = pathlib.Path(__file__).parents[1] / 'src' / 'fused_memory'
TASK_CURATOR = SRC_ROOT / 'middleware' / 'task_curator.py'

CURATOR_CLASS = 'TaskCurator'
CURATE_METHOD = 'curate'

# ``calls_named`` matches an attribute's name exactly, so the sibling batch
# path ``self._call_llm_batch(...)`` cannot select the wrong ``try``.
LLM_CALL = '_call_llm'
REPORT_CALL = 'report_failure'

# The single funnel every degraded decision must leave through, so that
# counting degradations does not mean enumerating their causes.
DECISION_TYPE = 'CuratorDecision'
DEGRADED_FUNNEL = '_degraded_create'

# The arms as of the fix: AllAccountsCappedException, CuratorFailureError, and
# the catch-all. A handler list shorter than this means an arm was deleted
# rather than instrumented, which would satisfy the silence check vacuously.
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


def test_every_llm_exception_arm_reports_failure():
    silent = [
        handler
        for handler in _llm_try().handlers
        if not calls_named(handler, REPORT_CALL)
    ]
    assert not silent, (
        f'{CURATOR_CLASS}.{CURATE_METHOD} has exception arm(s) around '
        f'{LLM_CALL}() that never call {REPORT_CALL}(): '
        + ', '.join(
            f'{_handler_label(handler)} at line {handler.lineno}'
            for handler in silent
        )
        + '. A curator failure that reaches a silent arm still returns '
        "action='create', which is indistinguishable downstream from a healthy "
        'create — the shape of the 2026-08-13 outage. Route it through the '
        f'escalator ({REPORT_CALL}) as the sibling arms do.'
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
