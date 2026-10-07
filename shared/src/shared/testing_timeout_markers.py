"""Census of ``pytest.mark.timeout(...)`` markers, and the band in which one inverts a verify budget.

Pure stdlib and pytest-free, like every ``shared.testing_*`` module, and
registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from enum import StrEnum
from typing import NamedTuple

from shared.pytest_mark_grammar import mark_elements, marker_name, pytestmark_value

__all__ = [
    'DELIBERATE_TIGHT_BOUND_CEILING',
    'SiteKind',
    'TimeoutSite',
    'inverts',
    'timeout_marker_sites',
]

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
