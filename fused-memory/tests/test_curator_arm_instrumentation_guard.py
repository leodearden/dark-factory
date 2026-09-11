"""Structural guard: every exception arm around the curator's LLM call reports.

``TaskCurator.curate()`` wraps its LLM call in one ``try`` with three handlers.
Two of them escalate; the third — the catch-all — did not, and that asymmetry
is what let the 2026-08-13 to 08-18 outage run for five days: a
``FileNotFoundError`` for the ``claude`` binary landed in the silent arm, which
logged at WARNING and returned ``action='create'``, a decision shape
indistinguishable downstream from a healthy create.

The behavioural half of that fix is asserted by
test_task_curator.py::TestUnexpectedExceptionArmReports, which exercises the
catch-all arm through ``curate()``. That test pins the arms that exist TODAY.
It cannot fail for an arm added TOMORROW — a fourth ``except`` clause catching
some newly-interesting exception type reintroduces exactly the original defect
while every behavioural test stays green, because no test drives an exception
nobody has thought of yet. This guard closes that: it asserts over the shape of
the handler list itself, so the requirement lands on any arm, named or not.

Asserted over the PARSED module, not its text, so prose that merely mentions
``report_failure`` — a comment in a handler explaining why it does not need to
report — cannot satisfy the check.

Deliberately NOT asserted: which exception types the arms catch, their order,
or what each passes to ``report_failure``. Those are design choices the arms
are free to revise; the invariant is only that no arm is silent.
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
