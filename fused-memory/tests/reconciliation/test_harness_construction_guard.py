"""Only ``reconciliation/harness_fixtures.py`` may construct a ``ReconciliationHarness`` in these tests.

One construction site makes a constructor change a one-file edit, and stops
hand-copied wiring from drifting apart (task 6075). The module set is
discovered, not hand-listed, and the check is AST-based, so prose naming the
class cannot trip it.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from _ast_guard import calls_named, parse_python_module

TESTS_ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
FACTORY = 'reconciliation/harness_fixtures.py'
CONSTRUCTOR = 'ReconciliationHarness'


@cache
def _discover_constructing_modules() -> frozenset[str]:
    found: set[str] = set()
    for path in sorted(TESTS_ROOT.glob('**/*.py')):
        if path.resolve() == SELF:
            continue
        if CONSTRUCTOR not in path.read_text():
            continue
        if calls_named(parse_python_module(path), CONSTRUCTOR):
            found.add(path.relative_to(TESTS_ROOT).as_posix())
    return frozenset(found)


def test_discovery_finds_the_shared_factory():
    constructing_modules = _discover_constructing_modules()
    assert FACTORY in constructing_modules, (
        f'discovery no longer finds {FACTORY} constructing {CONSTRUCTOR}(...) '
        f'(found {sorted(constructing_modules) or "nothing"}). The discriminator has '
        f'gone stale; fix it, and do NOT weaken this floor, or the guard below '
        f'silently checks nothing.'
    )


def test_only_the_shared_factory_constructs_the_harness():
    offenders = sorted(_discover_constructing_modules() - {FACTORY})
    assert not offenders, (
        f'{offenders} call {CONSTRUCTOR}(...) directly. Build it through '
        f'reconciliation.harness_fixtures.make_harness (or open_harness for a harness '
        f'on real stores under tmp_path); both forward every constructor kwarg.'
    )
