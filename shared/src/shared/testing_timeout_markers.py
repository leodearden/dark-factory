"""Census of ``pytest.mark.timeout(...)`` markers, and the band in which one inverts a verify budget.

A MARKER IS A TWO-WAY OVERRIDE, NEVER A FLOOR.  Read verbatim from
``pytest_timeout.py::_get_item_settings`` in the installed package::

    if marker is not None:
        timeout = _validate_timeout(settings.timeout, "marker")
    if timeout is None:
        timeout = item.config._env_timeout

The marker wins unconditionally.  ``config._env_timeout`` -- fed by CLI
``--timeout``, then ``PYTEST_TIMEOUT``, then the ini ``timeout`` -- is consulted
only when the marker yielded None.  So ``@pytest.mark.timeout(N)`` REPLACES
whatever budget is in force: it raises the budget wherever the ambient one is
smaller, and LOWERS it under verify's ``--timeout``.

THE INVERSION BAND.  Against a verify budget B, a marker at N falls in one of
three regimes:

* N <= DELIBERATE_TIGHT_BOUND_CEILING -- small enough to read as a deliberate
  tight bound rather than a slow test's opt-out;
* DELIBERATE_TIGHT_BOUND_CEILING < N < B -- INVERTS.  Too large to read as a
  deliberate fast bound, and below the budget verify passes, so the author's
  intended LOOSENING for a slow test silently becomes a TIGHTENING of the run
  that gates their merge: B becomes N;
* N >= B -- loosens under both.  Safe.

Only the middle band contradicts its author's evident intent.  What a breach
there costs depends on the suite's ``timeout_method``; that trade lives in
``tests/scripts/test_timeout_method_policy.py``.

Each package instantiates the guard in its own
``tests/test_timeout_marker_inversion_guard.py`` with its OWN B, read through
:func:`verify_cli_timeout` from the ``test_command`` in its orchestrator.yaml.

Pure stdlib and pytest-free, like every ``shared.testing_*`` module, and
registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

from shared.pytest_mark_grammar import mark_elements, marker_name, pytestmark_value

__all__ = [
    'DELIBERATE_TIGHT_BOUND_CEILING',
    'GrandfatherRatchet',
    'SiteKind',
    'TimeoutSite',
    'TreeScan',
    'grandfather_ratchet',
    'inversion_failure_message',
    'inverts',
    'scan_python_tree',
    'stale_grandfather_message',
    'timeout_marker_sites',
    'verify_cli_timeout',
]

T = TypeVar('T')

#: The band's LOWER edge: the largest N that still reads as a DELIBERATE tight
#: bound rather than a slow test's opt-out.
#:
#: A LITERAL, and deliberately NOT a mirror of any package's ini ``timeout``.
#: It was first written as one, while orchestrator's ini default was 60.
#: Commit 64e24b547f raised that default 60 -> 300 fleet-wide and the mirror
#: followed it: (300, 300) is EMPTY, so every sweep built on the band passed
#: VACUOUSLY.  Task 5442 then raised the default to 540, where a mirror would
#: sit ABOVE the verify budget and invert the band outright.  So the edge stays
#: at the value the design always used, and :func:`inverts` refuses a budget
#: that would empty the band.
DELIBERATE_TIGHT_BOUND_CEILING = 60

_MODULE_QUALNAME = '<module>'
_PYTESTMARK_QUALNAME = '<pytestmark>'

_TIMEOUT_FLAG = re.compile(r'--timeout[=\s](\d+)')


class SiteKind(StrEnum):
    """Which of the four binding forms a timeout marker was written in."""

    DECORATOR = 'decorator'
    CLASS_DECORATOR = 'class-decorator'
    MODULE_PYTESTMARK = 'module-pytestmark'
    CLASS_PYTESTMARK = 'class-pytestmark'


class TimeoutSite(NamedTuple):
    """One ``pytest.mark.timeout(...)`` occurrence in a module.

    ``qualname`` keys the site stably across ordinary edits: ``test_a``,
    ``TestThing::test_a``, ``TestThing`` (class decorator), ``<module>``
    (module ``pytestmark``) or ``TestThing::<pytestmark>``.  Angle brackets
    cannot occur in an identifier, so these never collide with a real name.

    ``seconds`` is None when the argument is absent or not statically
    resolvable: "no opinion", never "too small".

    ``spelling`` is the argument's unparsed source (``'300'``,
    ``'VERIFY_CLI_PER_TEST_TIMEOUT'``), empty when there is none.  A literal and
    a constant naming the same number resolve identically; only the spelling
    tells a marker that moves with a re-derivation from one that must be found
    and hand-edited.
    """

    qualname: str
    kind: SiteKind
    seconds: float | None
    lineno: int
    spelling: str


def _timeout_call_arg(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == 'timeout':
            return keyword.value
    return None


def _resolve_seconds(arg: ast.expr | None, sanctioned: Mapping[str, float]) -> float | None:
    if arg is None:
        return None
    if (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, int | float)
        and not isinstance(arg.value, bool)
    ):
        return float(arg.value)
    if isinstance(arg, ast.Name):
        return sanctioned.get(arg.id)
    if isinstance(arg, ast.Attribute):
        return sanctioned.get(arg.attr)
    return None


def _sites_among(
    elements: list[ast.expr],
    qualname: str,
    kind: SiteKind,
    sanctioned: Mapping[str, float],
) -> list[TimeoutSite]:
    sites: list[TimeoutSite] = []
    for element in elements:
        if not isinstance(element, ast.Call) or marker_name(element) != 'timeout':
            continue
        arg = _timeout_call_arg(element)
        sites.append(
            TimeoutSite(
                qualname=qualname,
                kind=kind,
                seconds=_resolve_seconds(arg, sanctioned),
                lineno=element.lineno,
                spelling='' if arg is None else ast.unparse(arg),
            )
        )
    return sites


def timeout_marker_sites(
    tree: ast.Module, sanctioned: Mapping[str, float]
) -> tuple[TimeoutSite, ...]:
    """Every ``pytest.mark.timeout(...)`` site in *tree*, in source order.

    Both argument spellings pytest-timeout accepts are read, ``timeout(300)``
    and ``timeout(timeout=300)``.  Seconds resolve, deliberately tinily, from a
    numeric literal (``bool`` excluded: ``timeout(True)`` is not 1s) or from a
    bare or dotted NAME looked up by its trailing identifier in *sanctioned*.
    Anything else -- arithmetic, a call, a name *sanctioned* lacks -- is None.
    The census is therefore a FLOOR: an in-band value reached through an
    indirection it cannot follow is not seen.

    Classes are walked at any depth; function bodies are not, because a
    function defined inside another is not a collected pytest item.
    """
    sites: list[TimeoutSite] = []

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for statement in body:
            bound = pytestmark_value(statement)
            if bound is not None:
                if prefix:
                    qualname, kind = f'{prefix}{_PYTESTMARK_QUALNAME}', SiteKind.CLASS_PYTESTMARK
                else:
                    qualname, kind = _MODULE_QUALNAME, SiteKind.MODULE_PYTESTMARK
                sites.extend(_sites_among(mark_elements(bound), qualname, kind, sanctioned))
            if isinstance(statement, ast.ClassDef):
                qualname = f'{prefix}{statement.name}'
                sites.extend(
                    _sites_among(
                        statement.decorator_list, qualname, SiteKind.CLASS_DECORATOR, sanctioned
                    )
                )
                walk(statement.body, f'{qualname}::')
            elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                sites.extend(
                    _sites_among(
                        statement.decorator_list,
                        f'{prefix}{statement.name}',
                        SiteKind.DECORATOR,
                        sanctioned,
                    )
                )

    walk(tree.body, '')
    return tuple(sites)


def inverts(seconds: float | None, *, verify_cli_budget: int) -> bool:
    """True iff a marker at *seconds* TIGHTENS verify while reading as a loosening.

    The band is ``(DELIBERATE_TIGHT_BOUND_CEILING, verify_cli_budget)``, open at
    both ends: a mark AT the ceiling is still a deliberate tight bound, and one
    AT the budget is the recommended remediation.  None cannot invert.

    Raises ValueError when *verify_cli_budget* leaves the band empty, since a
    sweep over an empty band would pass vacuously.
    """
    if verify_cli_budget <= DELIBERATE_TIGHT_BOUND_CEILING:
        raise ValueError(
            f'the inversion band ({DELIBERATE_TIGHT_BOUND_CEILING}, {verify_cli_budget}) '
            f'is empty: a verify --timeout of {verify_cli_budget} does not exceed '
            f'DELIBERATE_TIGHT_BOUND_CEILING={DELIBERATE_TIGHT_BOUND_CEILING}, so no '
            'marker could ever invert and every sweep would pass vacuously.'
        )
    return seconds is not None and DELIBERATE_TIGHT_BOUND_CEILING < seconds < verify_cli_budget


def verify_cli_timeout(test_command: str) -> int | None:
    """The per-test ``--timeout`` *test_command* passes pytest, else None.

    Reads both ``--timeout=300`` and ``--timeout 300``, the spelling
    ``tests/scripts/test_fallback_verify_config.py`` pins on the fleet chain.
    In a chained command the FIRST flag wins.
    """
    match = _TIMEOUT_FLAG.search(test_command)
    return None if match is None else int(match.group(1))


@dataclass(frozen=True)
class TreeScan(Generic[T]):
    """One pass over a tree's ``*.py`` files, with the counters that prove it read them.

    ``items`` is everything the extract callback yielded, in sorted-path order.
    ``examined`` counts the files that decoded, parseable or not.
    ``unreadable`` names, relative to the root, the files that did not, so a
    sweep that silently stops reading cannot pass as a clean tree.  Frozen,
    because callers memoise one scan and share it.
    """

    items: tuple[T, ...]
    examined: int
    unreadable: tuple[str, ...]


def scan_python_tree(
    root: Path, extract: Callable[[str, ast.Module], Iterable[T]]
) -> TreeScan[T]:
    """Read and parse every ``*.py`` under *root* once, handing each tree to *extract*.

    *extract* receives the module's POSIX path relative to *root* (never its
    basename, which nested modules share) and its parsed tree; one callback can
    therefore feed several extractions from a single parse.  No tree outlives
    its callback.

    FAIL-SOFT: a file that does not decode as UTF-8 or cannot be read is
    ``unreadable``; one that does not parse is examined but never extracted.  A
    test tree holds deliberately malformed fixtures, and they must not turn a
    census red for a reason unrelated to what it counts.
    """
    items: list[T] = []
    unreadable: list[str] = []
    examined = 0
    for py_file in sorted(root.rglob('*.py')):
        module = py_file.relative_to(root).as_posix()
        try:
            source = py_file.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            unreadable.append(module)
            continue
        examined += 1
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        items.extend(extract(module, tree))
    return TreeScan(tuple(items), examined, tuple(unreadable))


def inversion_failure_message(
    offenders: Iterable[tuple[str, TimeoutSite]],
    *,
    verify_cli_budget: int,
    slow_test_marker: str,
) -> str:
    """The failure text every package's guard prints for its in-band *offenders*.

    *slow_test_marker* is the package's own spelling of "this test is slow",
    offered first among the two remedies.
    """
    ordered = sorted(offenders, key=lambda pair: (pair[0], pair[1].qualname))
    sites = '\n  '.join(
        f'{module}::{site.qualname} ({site.kind}, line {site.lineno}) pins {site.seconds:g}s'
        for module, site in ordered
    )
    return (
        f'{len(ordered)} timeout marker(s) in the inversion band '
        f'({DELIBERATE_TIGHT_BOUND_CEILING} < N < {verify_cli_budget}).\n\n'
        'A marker there is a TWO-WAY override, not a floor: it REPLACES the '
        'ambient budget in both directions, so a number big enough to give a '
        'slow test room, yet below the '
        f'--timeout={verify_cli_budget} verify passes, silently clamps the run '
        'that actually gates your merge to N. Under timeout_method="thread" a '
        'breach does not even fail that one test: it kills the whole xdist '
        'worker. Which suites run thread, and why: '
        'tests/scripts/test_timeout_method_policy.py.\n\n'
        'Write one of:\n\n'
        f'{textwrap.indent(slow_test_marker, "    ")}\n\n'
        f'    @pytest.mark.timeout(N)  # N <= {DELIBERATE_TIGHT_BOUND_CEILING}, a '
        'DELIBERATE tight bound\n\n'
        'The second is for a test that asserts something happens FAST; it is '
        'small enough to read as that deliberate bound, which is why it is '
        'allowed. Anything in between inverts. Full rationale: '
        'shared/src/shared/testing_timeout_markers.py.'
        f'\n\nOffending sites:\n  {sites}'
    )


@dataclass(frozen=True)
class GrandfatherRatchet:
    """In-band sites checked against a per-site allowlist that may only shrink.

    ``new_offenders`` are the in-band sites the allowlist does not name.
    ``stale`` are the allowlist's ``(module, qualname)`` keys, sorted, that name
    no live in-band site -- a raised or deleted marker, or a renamed test --
    and so must be removed before they admit a newcomer reusing the name.
    """

    new_offenders: tuple[tuple[str, TimeoutSite], ...]
    stale: tuple[tuple[str, str], ...]


def grandfather_ratchet(
    in_band: Iterable[tuple[str, TimeoutSite]],
    grandfathered: frozenset[tuple[str, str]],
) -> GrandfatherRatchet:
    """Split *in_band* against *grandfathered*, keyed per SITE as ``(module, qualname)``.

    Per site and never a per-module count, because a count nets to zero when
    one marker is added and another removed in the same module.
    """
    in_band = tuple(in_band)
    live = {(module, site.qualname) for module, site in in_band}
    return GrandfatherRatchet(
        new_offenders=tuple(
            (module, site)
            for module, site in in_band
            if (module, site.qualname) not in grandfathered
        ),
        stale=tuple(sorted(grandfathered - live)),
    )


def stale_grandfather_message(stale: Iterable[tuple[str, str]]) -> str:
    """The failure text for allowlist entries :func:`grandfather_ratchet` found stale."""
    entries = tuple(stale)
    listing = '\n  '.join(f'{module}::{qualname}' for module, qualname in entries)
    return (
        f'{len(entries)} grandfathered timeout marker site(s) no longer sit in '
        'the inversion band: the marker was raised or removed, or the test was '
        'renamed. Delete the entries from the allowlist, which may only ever '
        'shrink; a stale entry would silently admit a new in-band marker that '
        f'reuses its name.\n  {listing}'
    )
