"""Rule D ``timing-dependence``: a test whose verdict depends on how fast the host runs.

Stance (task 8254): timing belongs in a test only as the BEHAVIOUR under test,
and then on an injected or virtual clock in-process, or against a real child by
that child's own time.  Sleeping until something is ready never belongs.

Legs, each one site:
  SLEEP: a ``time.sleep`` / ``asyncio.sleep`` call through the module.
  ELAPSED_ASSERT: an assert comparing a REAL elapsed time with a numeric literal.
    A real clock read is the ``time`` module's or an event loop's, never an
    injected clock's, so the sanctioned ``fake_clock`` path is never a site.
    Names resolve through bindings in the assert's own innermost function.

Hang floor: a literal sleep of ``HANG_SIMULATOR_FLOOR_SECS`` or more, or inf,
simulates a hang.  A passing test never waits one out, it cancels it, so it
cannot make a verdict load-sensitive; the census found no literal sleep between
30 and 60 s, so the floor separates two populations rather than cutting one.

Residuals: a sleep whose duration is a named constant is not resolved, so it is
a site whatever its value.  An elapsed value read off code under test
(``result.duration_s``) is not seen.

Per-node like Rules B and C, so the main checker folds it into its one walk.
Stdlib only, for the reason _lint_cli.py gives.
"""
from __future__ import annotations

import ast
import enum
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

# Sibling import that survives `python3 -I`: see _lint_cli.py's module docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _lint_cli import DebtBaseline, Violation
finally:
    del sys.path[0]

RULE_CODE = 'timing-dependence'
HANG_SIMULATOR_FLOOR_SECS = 60

_FAKE_CLOCK_REF = 'shared/src/shared/pytest_fake_clock.py::fake_clock'
_VIRTUAL_CLOCK_REF = 'shared/src/shared/testing_virtual_clock.py::virtual_clock_test'
_MARKER_CLAUSE = (
    " Mark the test @pytest.mark.timing(reason='<why>') only when a deadline is the"
    ' behaviour under test; the marker covers only with a non-empty literal reason=.'
)


class TimingKind(enum.Enum):
    SLEEP = 'sleep'
    ELAPSED_ASSERT = 'elapsed assert'


class TimingSite(NamedTuple):
    lineno: int
    col_offset: int
    kind: TimingKind
    snippet: str
    is_real_thread_sleep: bool


class ScopeSpan(NamedTuple):
    """The lines of one function, the scope a binding or an assert belongs to."""

    first: int
    last: int


class ClockBinding(NamedTuple):
    """``name = <real clock read>``."""

    name: str
    lineno: int


class ElapsedBinding(NamedTuple):
    """``name = <left> - x``, optionally under a call; *left_name* None means a real read."""

    name: str
    lineno: int
    left_name: str | None


class ElapsedAssertCandidate(NamedTuple):
    """An assert comparing something with a numeric literal, before its names resolve."""

    lineno: int
    col_offset: int
    snippet: str
    direct: bool
    names: tuple[str, ...]
    left_names: tuple[str, ...]


TimingFact = TimingSite | ScopeSpan | ClockBinding | ElapsedBinding | ElapsedAssertCandidate


def _numeric_literal(node: ast.expr) -> float | None:
    """The value of an int or float literal, optionally negated; a bool is not numeric."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign, node = -1, node.operand
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        return None
    value = node.value
    return sign * value if isinstance(value, int | float) else None


def _is_infinity(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == 'inf' and isinstance(node.value, ast.Name) and node.value.id == 'math'
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'float'
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and str(node.args[0].value).strip().lower() in ('inf', 'infinity')
    )


_TIME_MODULE_READS = frozenset(
    {'time', 'monotonic', 'perf_counter', 'time_ns', 'monotonic_ns', 'perf_counter_ns'}
)


def _is_event_loop(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id.endswith('loop')
    if isinstance(node, ast.Attribute):
        return node.attr.endswith('loop')
    return (
        isinstance(node, ast.Call)
        and not node.args
        and _module_function(node, ('asyncio',), 'get_running_loop', 'get_event_loop') is not None
    )


def _is_real_clock_read(node: ast.expr) -> bool:
    """A zero-argument read of the ``time`` module's clock or an event loop's."""
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if isinstance(func.value, ast.Name) and func.value.id == 'time':
        return func.attr in _TIME_MODULE_READS
    return func.attr == 'time' and _is_event_loop(func.value)


def _module_function(call: ast.Call, modules: Iterable[str], *names: str) -> str | None:
    """The module *call* goes through, when it is ``<module>.<one of names>(...)``."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in names and isinstance(func.value, ast.Name):
        return func.value.id if func.value.id in modules else None
    return None


def _sleep_duration(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    return next((kw.value for kw in call.keywords if kw.arg in ('delay', 'secs')), None)


def _never_waits_real_time(duration: ast.expr) -> bool:
    """A zero sleep only yields; one at or past the floor is a hang simulator."""
    if _is_infinity(duration):
        return True
    value = _numeric_literal(duration)
    return value is not None and (value == 0 or value >= HANG_SIMULATOR_FLOOR_SECS)


def _sleep_site(call: ast.Call) -> TimingSite | None:
    module = _module_function(call, ('time', 'asyncio'), 'sleep')
    if module is None:
        return None
    duration = _sleep_duration(call)
    if duration is not None and _never_waits_real_time(duration):
        return None
    return TimingSite(
        call.lineno, call.col_offset, TimingKind.SLEEP, ast.unparse(call), module == 'time'
    )


def _subtraction_left(node: ast.expr) -> ast.expr | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return node.left
    return None


def _is_direct_elapsed(node: ast.expr) -> bool:
    left = _subtraction_left(node)
    return left is not None and _is_real_clock_read(left)


def _binding(node: ast.Assign | ast.AnnAssign) -> tuple[str, ast.expr] | None:
    """``(name, value)`` for a single-Name-target binding with a value, else None."""
    if isinstance(node, ast.AnnAssign):
        target, value = node.target, node.value
    elif len(node.targets) == 1:
        target, value = node.targets[0], node.value
    else:
        return None
    if not isinstance(target, ast.Name) or value is None:
        return None
    return target.id, value


def _elapsed_left(value: ast.expr) -> ast.expr | None:
    """The left operand of ``<left> - x``, or of one passed first to a call (``round``)."""
    if isinstance(value, ast.Call) and value.args:
        value = value.args[0]
    return _subtraction_left(value)


def _binding_facts(node: ast.Assign | ast.AnnAssign) -> tuple[TimingFact, ...]:
    bound = _binding(node)
    if bound is None:
        return ()
    name, value = bound
    if _is_real_clock_read(value):
        return (ClockBinding(name, node.lineno),)
    left = _elapsed_left(value)
    if left is not None and _is_real_clock_read(left):
        return (ElapsedBinding(name, node.lineno, None),)
    if isinstance(left, ast.Name):
        return (ElapsedBinding(name, node.lineno, left.id),)
    return ()


def _literal_comparisons(test: ast.expr) -> Iterable[list[ast.expr]]:
    for node in ast.walk(test):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(_numeric_literal(operand) is not None for operand in operands):
                yield operands


def _elapsed_assert_candidate(node: ast.Assert) -> ElapsedAssertCandidate | None:
    operands = [operand for found in _literal_comparisons(node.test) for operand in found]
    direct = any(_is_direct_elapsed(operand) for operand in operands)
    names = tuple(operand.id for operand in operands if isinstance(operand, ast.Name))
    lefts = (_subtraction_left(operand) for operand in operands)
    left_names = tuple(left.id for left in lefts if isinstance(left, ast.Name))
    if not (direct or names or left_names):
        return None
    return ElapsedAssertCandidate(
        node.lineno, node.col_offset, ast.unparse(node.test), direct, names, left_names
    )


def _optional(fact: TimingFact | None) -> tuple[TimingFact, ...]:
    return (fact,) if fact is not None else ()


def timing_facts(node: ast.AST, *, deadline_already_reported: bool) -> tuple[TimingFact, ...]:
    """Rule D's facts about ONE node of a walk; ``timing_violations`` resolves them per file."""
    if isinstance(node, ast.Call):
        return _optional(_sleep_site(node))
    if isinstance(node, ast.Assign | ast.AnnAssign):
        return _binding_facts(node)
    if isinstance(node, ast.Assert):
        return _optional(_elapsed_assert_candidate(node))
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return (ScopeSpan(node.lineno, node.end_lineno or node.lineno),)
    return ()


def _innermost_scope(lineno: int, spans: Iterable[ScopeSpan]) -> ScopeSpan | None:
    """The smallest function span holding *lineno*; None is module scope."""
    holding = [span for span in spans if span.first <= lineno <= span.last]
    return min(holding, key=lambda span: (span.last - span.first, -span.first), default=None)


class _ScopedClockNames(NamedTuple):
    """A file's clock and elapsed bindings, each keyed by (innermost scope, name)."""

    clocks: frozenset[tuple[ScopeSpan | None, str]]
    elapsed_lefts: dict[tuple[ScopeSpan | None, str], tuple[str | None, ...]]

    def is_clock(self, scope: ScopeSpan | None, name: str) -> bool:
        return (scope, name) in self.clocks

    def is_real_elapsed(self, scope: ScopeSpan | None, name: str) -> bool:
        lefts = self.elapsed_lefts.get((scope, name), ())
        return any(left is None or self.is_clock(scope, left) for left in lefts)


def _scoped_clock_names(facts: Iterable[TimingFact], spans: list[ScopeSpan]) -> _ScopedClockNames:
    clocks: set[tuple[ScopeSpan | None, str]] = set()
    elapsed: dict[tuple[ScopeSpan | None, str], tuple[str | None, ...]] = {}
    for fact in facts:
        if isinstance(fact, ClockBinding):
            clocks.add((_innermost_scope(fact.lineno, spans), fact.name))
        elif isinstance(fact, ElapsedBinding):
            key = (_innermost_scope(fact.lineno, spans), fact.name)
            elapsed[key] = (*elapsed.get(key, ()), fact.left_name)
    return _ScopedClockNames(frozenset(clocks), elapsed)


def _is_elapsed_assert(
    candidate: ElapsedAssertCandidate, names: _ScopedClockNames, spans: list[ScopeSpan]
) -> bool:
    if candidate.direct:
        return True
    scope = _innermost_scope(candidate.lineno, spans)
    return any(
        names.is_real_elapsed(scope, name) for name in candidate.names + candidate.left_names
    ) or any(names.is_clock(scope, name) for name in candidate.left_names)


def _elapsed_assert_sites(facts: tuple[TimingFact, ...]) -> list[TimingSite]:
    candidates = [fact for fact in facts if isinstance(fact, ElapsedAssertCandidate)]
    if not candidates:
        return []
    spans = [fact for fact in facts if isinstance(fact, ScopeSpan)]
    names = _scoped_clock_names(facts, spans)
    return [
        TimingSite(c.lineno, c.col_offset, TimingKind.ELAPSED_ASSERT, c.snippet, False)
        for c in candidates
        if _is_elapsed_assert(c, names, spans)
    ]


def uncovered_timing_sites(facts: Iterable[TimingFact]) -> list[TimingSite]:
    """The sites among *facts*, a whole file's, that no coverage excuses."""
    facts = tuple(facts)
    sites = [fact for fact in facts if isinstance(fact, TimingSite)]
    return sites + _elapsed_assert_sites(facts)


def _sleep_msg(site: TimingSite) -> str:
    return (
        f'{RULE_CODE}: `{site.snippet}` sleeps in real time, so the verdict moves with'
        ' host load. Wait on the event, or a polled predicate, the code under test'
        ' signals instead of sleeping until it is ready; to advance time, take the'
        f' fake_clock fixture ({_FAKE_CLOCK_REF}) or run the scenario on'
        f' virtual_clock_test ({_VIRTUAL_CLOCK_REF}).'
        + _MARKER_CLAUSE
        + f' A literal sleep of {HANG_SIMULATOR_FLOOR_SECS} s or more, or inf, simulates'
        ' a hang and is not a site.'
    )


def _elapsed_assert_msg(site: TimingSite) -> str:
    return (
        f'{RULE_CODE}: `assert {site.snippet}` compares a real elapsed time with a'
        ' written number. A wall-clock performance assertion does not belong in a merge'
        ' gate: under load it fails a correct test. Measure time on the fake_clock'
        f' fixture ({_FAKE_CLOCK_REF}) or run the scenario on virtual_clock_test'
        f' ({_VIRTUAL_CLOCK_REF}), and assert the exact virtual elapsed time.'
        + _MARKER_CLAUSE
    )


_MESSAGES = {
    TimingKind.SLEEP: _sleep_msg,
    TimingKind.ELAPSED_ASSERT: _elapsed_assert_msg,
}


def timing_violations(
    facts: Iterable[TimingFact], *, filename: str, budgets: DebtBaseline
) -> list[Violation]:
    """Rule D's violations for one file, from every fact its walk produced."""
    return [
        Violation(filename, site.lineno, site.col_offset, _MESSAGES[site.kind](site))
        for site in uncovered_timing_sites(facts)
    ]
