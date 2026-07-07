"""Clock-discipline guard: no bare `datetime.now()` reads in the data layer.

Request-scoped code must resolve `now` once (via
:func:`dashboard.data.utils.resolve_now`) and thread it through, rather than
letting each function read the live clock independently. This module parses
each `dashboard/src/dashboard/data/*.py` file with `ast` and flags every
`<expr>.now(...)` Call node that is neither tagged `# clock-exempt:` on its
physical source line nor part of `resolve_now`'s own definition (the
sanctioned single clock-read site).

The matcher intentionally does not require the receiver to be a bare
`datetime` name: it flags any `.now(...)` attribute call, so an aliased
import (``from datetime import datetime as dt`` then ``dt.now(UTC)``, or
``import datetime as _dt`` then ``_dt.datetime.now()``) can't silently
evade the guard. This trades a slightly higher false-positive rate (any
unrelated `.now()`-named method would also be flagged) for closing that
coverage gap; false positives are handled the same way as everything else
— an explicit `# clock-exempt:` tag.
"""

from __future__ import annotations

import ast
from pathlib import Path

_EXEMPT_MARKER = '# clock-exempt:'


def find_clock_violations(source: str) -> list[tuple[int, str]]:
    """Return ``(line, text)`` for every bare ``<expr>.now(...)`` call in *source*.

    Matches any attribute-call named ``now`` — not just calls on a bare
    ``datetime`` name — so aliased imports can't evade the guard (see the
    module docstring). A call is exempt if its physical source line
    contains the substring ``# clock-exempt:``, or if the call lives inside
    a ``def resolve_now(...):`` (the sanctioned single clock-read site).
    """
    tree = ast.parse(source)
    lines = source.splitlines()

    resolve_now_spans: list[tuple[int, int]] = [
        (node.lineno, getattr(node, 'end_lineno', node.lineno))
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'resolve_now'
    ]

    def _in_resolve_now(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in resolve_now_spans)

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'now'):
            continue
        lineno = func.value.lineno
        line_text = lines[lineno - 1] if 0 < lineno <= len(lines) else ''
        if _EXEMPT_MARKER in line_text:
            continue
        if _in_resolve_now(lineno):
            continue
        violations.append((lineno, line_text))
    return violations


# ---------------------------------------------------------------------------
# Checker unit tests (fixtures, not the real tree)
# ---------------------------------------------------------------------------


def test_negative_fixture_fires():
    """An untagged bare call outside resolve_now is flagged."""
    violations = find_clock_violations('now = datetime.now(UTC)\n')
    assert len(violations) == 1, f'expected exactly one violation, got {violations!r}'
    assert violations[0][0] == 1


def test_single_capture_writer_tag_passes():
    """The `single-capture writer` exemption flavor silences the guard."""
    source = 'now = datetime.now(UTC)  # clock-exempt: single-capture writer\n'
    assert find_clock_violations(source) == []


def test_deferred_consolidation_tag_passes():
    """The `deferred-consolidation` (grandfather) exemption flavor silences the guard."""
    source = 'now = datetime.now(UTC)  # clock-exempt: deferred-consolidation (task 2281)\n'
    assert find_clock_violations(source) == []


def test_docstring_mention_ignored():
    """A docstring mentioning `datetime.now(UTC)` is a Constant, not a Call — never flagged."""
    source = (
        'def foo():\n'
        '    """Uses datetime.now(UTC)."""\n'
        '    return 1\n'
    )
    assert find_clock_violations(source) == []


def test_resolve_now_definition_allowed():
    """resolve_now's own bare clock read is the sanctioned site and is allowed."""
    source = 'def resolve_now(now):\n    return now if now is not None else datetime.now(UTC)\n'
    assert find_clock_violations(source) == []


def test_multiline_call_tag_on_base_line():
    """A call split across lines is allowed when the tag sits on the `datetime.now(` line."""
    source = (
        'value = (\n'
        '    datetime.now(UTC)  # clock-exempt: single-capture writer\n'
        '    - timedelta(days=1)\n'
        ')\n'
    )
    assert find_clock_violations(source) == []


def test_aliased_datetime_import_fires():
    """`from datetime import datetime as dt; dt.now(UTC)` can't evade the guard."""
    source = 'now = dt.now(UTC)  # dt is `datetime` imported under an alias\n'
    violations = find_clock_violations(source)
    assert len(violations) == 1, f'expected exactly one violation, got {violations!r}'
    assert violations[0][0] == 1


def test_aliased_module_import_fires():
    """`import datetime as _dt; _dt.datetime.now()` can't evade the guard either."""
    source = 'now = _dt.datetime.now()\n'
    violations = find_clock_violations(source)
    assert len(violations) == 1, f'expected exactly one violation, got {violations!r}'
    assert violations[0][0] == 1


# ---------------------------------------------------------------------------
# Acceptance test: the real data-layer tree
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'data'


def test_no_bare_clock_reads_in_data_modules():
    """No `dashboard/src/dashboard/data/*.py` module has an untagged bare clock read."""
    violations: list[str] = []
    for path in sorted(_DATA_DIR.glob('*.py')):
        source = path.read_text()
        rel = path.relative_to(_DATA_DIR.parent.parent)
        for lineno, text in find_clock_violations(source):
            violations.append(f'{rel}:{lineno}: {text.strip()}')

    assert not violations, (
        'Bare datetime.now() reads found (missing resolve_now() or a '
        '`# clock-exempt:` tag):\n' + '\n'.join(violations)
    )
