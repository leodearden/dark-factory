"""Rule D ``timing-dependence``: a test whose verdict depends on how fast the host runs.

Stance (task 8254): timing belongs in a test only as the BEHAVIOUR under test,
and then on an injected or virtual clock in-process, or against a real child by
that child's own time.  Sleeping until something is ready never belongs.

Legs, each one site:
  SLEEP: a ``time.sleep`` / ``asyncio.sleep`` call through the module.

Hang floor: a literal sleep of ``HANG_SIMULATOR_FLOOR_SECS`` or more, or inf,
simulates a hang.  A passing test never waits one out, it cancels it, so it
cannot make a verdict load-sensitive; the census found no literal sleep between
30 and 60 s, so the floor separates two populations rather than cutting one.

Residuals: a sleep whose duration is a named constant is not resolved, so it is
a site whatever its value.

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


class TimingSite(NamedTuple):
    lineno: int
    col_offset: int
    kind: TimingKind
    snippet: str
    is_real_thread_sleep: bool


TimingFact = TimingSite


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


def _module_function(call: ast.Call, modules: Iterable[str], name: str) -> str | None:
    """The module *call* reaches *name* through, when it is ``<module>.<name>(...)``."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == name and isinstance(func.value, ast.Name):
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


def timing_facts(node: ast.AST, *, deadline_already_reported: bool) -> tuple[TimingFact, ...]:
    """Rule D's facts about ONE node of a walk; ``timing_violations`` resolves them per file."""
    if isinstance(node, ast.Call):
        site = _sleep_site(node)
        return (site,) if site is not None else ()
    return ()


def uncovered_timing_sites(facts: Iterable[TimingFact]) -> list[TimingSite]:
    """The sites among *facts*, a whole file's, that no coverage excuses."""
    return [fact for fact in facts if isinstance(fact, TimingSite)]


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


def timing_violations(
    facts: Iterable[TimingFact], *, filename: str, budgets: DebtBaseline
) -> list[Violation]:
    """Rule D's violations for one file, from every fact its walk produced."""
    return [
        Violation(filename, site.lineno, site.col_offset, _sleep_msg(site))
        for site in uncovered_timing_sites(facts)
    ]
