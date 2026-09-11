"""Structural guards on how ``TaskCurator.curate()`` handles a failed curation.

Two invariants, both structural, both about the same hazard: a degraded
curation that no signal distinguishes from a healthy one.

1. Every exception arm around the LLM call routes through ``report_failure``.
2. ``curate()`` constructs no ``CuratorDecision`` directly — every degraded
   decision leaves through the ``_degraded_create`` funnel that counts it.

``TaskCurator.curate()`` wraps its LLM call in one ``try`` with three handlers.
Two of them escalate; the third — the catch-all — did not, and that asymmetry
is what let the 2026-08-13 to 08-18 outage run for five days: a
``FileNotFoundError`` for the ``claude`` binary landed in the silent arm, which
logged at WARNING and returned ``action='create'``, a decision shape
indistinguishable downstream from a healthy create.

The behavioural halves of the fix are asserted in test_task_curator.py by
TestUnexpectedExceptionArmReports and TestConsecutiveDegradedAlarm. Those tests
pin the paths that exist TODAY, and neither can fail for a path added TOMORROW
— a fourth ``except`` clause, or a sixth degraded branch building its own
decision, reintroduces the original defect while every behavioural test stays
green, because no test drives a failure nobody has thought of yet. These guards
assert over the shapes themselves, so the requirements land on any new arm or
branch without anyone remembering to extend a list.

Asserted over the PARSED module, not its text, so prose that merely mentions
``report_failure`` — a comment in a handler explaining why it does not need to
report — cannot satisfy the check.

Deliberately NOT asserted: which exception types the arms catch, their order,
what each passes to ``report_failure``, or what ``_degraded_create`` does with
the decision it builds. Those are design choices the code is free to revise;
the invariants are only that no arm is silent and no degraded decision is
uncounted.
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
    """Degraded decisions leave through one funnel, not five constructions.

    This is what makes the streak alarm hole-free BY CONSTRUCTION rather than
    by an enumerated list of degraded reasons. A future sixth degraded path
    that builds its own CuratorDecision would not be counted, and — exactly
    like the silent arm this task started from — nothing would look wrong.
    """
    direct = calls_named(_curate_def(), DECISION_TYPE)
    assert not direct, (
        f'{CURATOR_CLASS}.{CURATE_METHOD} constructs {DECISION_TYPE} directly at '
        f'line(s) {sorted(call.lineno for call in direct)}. Every degraded '
        f'decision must be built by {DEGRADED_FUNNEL}() instead, which is what '
        f'increments the degraded streak — a decision constructed here bypasses '
        f'the counter and is invisible to the alarm.'
    )
