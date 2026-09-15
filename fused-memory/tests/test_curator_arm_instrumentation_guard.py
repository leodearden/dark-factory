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


# --- Invariant 3: _parse_decision_dict's fail-open returns are classified ----
#
# The same hazard as invariant 2, one function away. _parse_decision_dict
# cannot use the _degraded_create funnel (it is a module-level parser with no
# curator to count against), so it marks degradation with the `degraded=`
# field instead — and whether each fail-open branch sets it was, until
# esc-4448-9, decided by whoever wrote the branch. Five of the eight branches
# were payload failures and two of those shipped unmarked, which held the
# streak at 0 through a total dedupe bypass.
#
# So this guard asserts the classification is DELIBERATE rather than
# defaulted: every fail-open return must either set `degraded=` explicitly,
# or be named in the state-veto allowlist below. Adding a branch without
# doing one or the other fails here, which is the only place a reader is
# forced to answer the question the field's contract asks.
PARSE_FN = '_parse_decision_dict'

# Branches whose verdict turns on POOL STATE, not on the response — the model
# rendered a usable decision and a downstream guard declined to act on it.
# Routine on a healthy curator, so deliberately NOT degraded. Keyed by the
# justification marker each branch emits. See CuratorDecision.degraded.
STATE_VETO_MARKERS = (
    'invalid-target',
    'unknown-status-target',
    'invalid-combine-target',
)


def _parse_decision_dict_def() -> ast.FunctionDef:
    tree = parse_python_module(TASK_CURATOR)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == PARSE_FN:
            return node
    raise AssertionError(
        f'{TASK_CURATOR.name}: no def {PARSE_FN}() — this guard has lost its '
        f'subject and is asserting nothing; re-point it rather than deleting it.'
    )


def _fail_open_returns() -> list[ast.Call]:
    """Every ``return CuratorDecision(action='create', ...)`` in the parser.

    The terminal ``return`` carries the parsed action through a variable, so
    keying on the literal ``'create'`` selects exactly the fail-open branches
    and never the success path.
    """
    found = []
    for node in ast.walk(_parse_decision_dict_def()):
        if not isinstance(node, ast.Return):
            continue
        for call in calls_named(node, DECISION_TYPE):
            action = next(
                (kw.value for kw in call.keywords if kw.arg == 'action'), None
            )
            if isinstance(action, ast.Constant) and action.value == 'create':
                found.append(call)
    return found


def _marker_text(call: ast.Call) -> str:
    justification = next(
        (kw.value for kw in call.keywords if kw.arg == 'justification'), None
    )
    return ast.unparse(justification) if justification is not None else ''


def test_every_fail_open_parse_branch_declares_degradedness():
    unclassified = [
        call
        for call in _fail_open_returns()
        if not any(kw.arg == 'degraded' for kw in call.keywords)
        and not any(marker in _marker_text(call) for marker in STATE_VETO_MARKERS)
    ]
    assert not unclassified, (
        f'{PARSE_FN}() has fail-open branch(es) at line(s) '
        f'{sorted(call.lineno for call in unclassified)} that neither set '
        f'degraded= nor match a known state veto '
        f'({", ".join(STATE_VETO_MARKERS)}): '
        + ', '.join(_marker_text(call) or '<no justification>' for call in unclassified)
        + '. Classify it by what the verdict DEPENDS ON: if only the response '
        'was consulted, the model broke the output contract and the branch is '
        'degraded=True; if pool state declined an otherwise usable decision, '
        'add its marker to STATE_VETO_MARKERS. Leaving it defaulted makes a '
        'sustained dedupe bypass read as health (esc-4448-9).'
    )


def test_state_veto_allowlist_still_matches_real_branches():
    """A marker that no longer matches any branch would silently widen the
    allowlist past its subject, letting a future payload failure inherit an
    exemption written for a veto that no longer exists."""
    markers = [_marker_text(call) for call in _fail_open_returns()]
    orphaned = [
        veto for veto in STATE_VETO_MARKERS
        if not any(veto in marker for marker in markers)
    ]
    assert not orphaned, (
        f'STATE_VETO_MARKERS entries match no branch in {PARSE_FN}(): '
        f'{orphaned}. Remove them rather than leaving a dead exemption.'
    )
