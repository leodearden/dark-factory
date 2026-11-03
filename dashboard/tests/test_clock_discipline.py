"""Clock-discipline guards: no bare `.now()` reads, no SQL-side `datetime()` calls.

Request-scoped code must resolve `now` once (via
:func:`dashboard.data.utils.resolve_now`) and thread it through, rather than
letting each function read the live clock independently. This module parses
each `dashboard/src/dashboard/data/*.py` file with `ast` and flags every
`<expr>.now(...)` Call node that is neither tagged `# clock-exempt:` on its
physical source line nor part of `resolve_now`'s own definition (the
sanctioned single clock-read site).

`dashboard/src/dashboard/app.py` — the route/composition layer sitting on
top of the data layer — is scanned too. A future regression there (e.g. a
route reverting to a per-leg `datetime.now(UTC)` instead of one shared
capture) would reintroduce exactly the cross-DB
clock-skew race this guard exists to prevent, so the route layer needs the
same protection as the data layer. Route handlers legitimately read the
clock once per request — there is nothing upstream to inject a `now` from,
so they call `datetime.now(UTC)` directly rather than `resolve_now(None)` —
and either thread that single value through a fan-out (`now=now` kwargs,
mirroring how `resolve_now` callers behave in the data layer) or use it
once locally (e.g. a ticket-age computation). Each such site is tagged
`# clock-exempt: single-capture route`. The scan covers
`dashboard/data/*.py`, `app.py`, `dashboard/api/*.py` and `loops.py` — the
data layer plus every module the composition layer was split into (task
5586 moved two tagged single-capture routes, `api_burndown` and
`api_merge_queue`, out of `app.py` into `dashboard/api/`, and the guard
follows them rather than quietly shedding the coverage). `_scanned_modules`
is the one definition of that set: every real-tree test below reads it, and
it fails loudly when any part of the set goes missing. No module beyond
those is scanned by either guard; that boundary is intentional, not an
oversight.

The matcher intentionally does not require the receiver to be a bare
`datetime` name: it flags any `.now(...)` attribute call, so an aliased
import (``from datetime import datetime as dt`` then ``dt.now(UTC)``, or
``import datetime as _dt`` then ``_dt.datetime.now()``) can't silently
evade the guard. This trades a slightly higher false-positive rate (any
unrelated `.now()`-named method would also be flagged) for closing that
coverage gap; false positives are handled the same way as everything else
— an explicit `# clock-exempt:` tag.

The second guard flags every non-docstring string literal that calls
SQLite's `datetime()`, over the same module boundary. That function renders
`'YYYY-MM-DD HH:MM:SS'` — a space separator and no UTC offset — while the
columns it gets compared against hold ISO text with a `T` and an offset, so
a lexical TEXT comparison misplaces rows on the cutoff's own calendar date.
Two shapes have shipped: `datetime('now', ...)` (task 4624) and the
column-anchored `datetime(MAX(completed_at), '-N days')` (removed by task
5594; this guard broadened to catch it by task 5155). The matcher therefore
flags the function whatever its arguments or letter case. SQL `date()` and
`strftime()` are not flagged: the data layer uses them for bucket labels and
integer-epoch arithmetic, never for a lexically compared cutoff. A Python
`datetime(...)` constructor is a Call node, not a string literal, so it is
never inspected. A real query's fix is always a cutoff computed in Python
and bound as a parameter. A literal that only names the function in prose
(a log or exception message, say) takes the same `# clock-exempt:` tag as a
`.now()` read, on the line where the literal starts.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

_EXEMPT_MARKER = '# clock-exempt:'
_DEFERRED_CONSOLIDATION_TAG = f'{_EXEMPT_MARKER} deferred-consolidation'
_SQL_DATETIME_CALL = re.compile(r'\bdatetime\s*\(', re.IGNORECASE)


def _line_is_exempt(lines: list[str], lineno: int) -> bool:
    """True when 1-based physical line *lineno* carries the ``# clock-exempt:`` tag."""
    return _EXEMPT_MARKER in lines[lineno - 1]


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
        if _line_is_exempt(lines, lineno) or _in_resolve_now(lineno):
            continue
        violations.append((lineno, lines[lineno - 1]))
    return violations


def _iter_non_docstring_string_literals(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Yield ``(lineno, value)`` for every string literal in *tree* except docstrings.

    Comments never reach the AST; docstrings do, and are excluded because
    prose naming a forbidden SQL spelling is not a query that reaches SQLite.
    """
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_ids.add(id(body[0].value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_ids
        ):
            yield node.lineno, node.value


def find_sql_datetime_violations(source: str) -> list[tuple[int, str]]:
    """Return ``(line, excerpt)`` for every non-docstring literal calling SQL ``datetime()``.

    A match is reported off the AST literal, never off a physical line,
    because implicitly concatenated SQL folds into one literal that no
    single line contains. A literal is exempt when the line it starts on
    carries ``# clock-exempt:`` — the same escape hatch the ``.now()`` guard
    honors. The excerpt's whitespace is collapsed so a triple-quoted literal
    cannot put newlines into a joined failure message.
    """
    lines = source.splitlines()
    return [
        (lineno, ' '.join(value.split())[:120])
        for lineno, value in _iter_non_docstring_string_literals(ast.parse(source))
        if _SQL_DATETIME_CALL.search(value) and not _line_is_exempt(lines, lineno)
    ]


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
# SQL-side datetime() guard: checker unit tests (fixtures, not the real tree)
# ---------------------------------------------------------------------------

# Implicit concatenation folds these adjacent fragments into ONE literal
# (starting on line 4) that calls datetime(), while no physical line does:
# the split falls between `date` and `time(`.
_FOLDED_SQL_SOURCE = '''
def q():
    return (
        "SELECT project_id FROM task_results "
        "WHERE completed_at >= date"
        "time('now', ? || ' days')"
    )
'''

_SINGLE_QUOTED_NOW_SOURCE = '''
def q():
    return "SELECT 1 FROM t WHERE completed_at >= datetime('now', ? || ' days')"
'''

_DOUBLE_QUOTED_NOW_SOURCE = """
def q():
    return 'SELECT 1 FROM t WHERE completed_at >= datetime("now", ? || " days")'
"""

_COLUMN_ANCHORED_SOURCE = '''
def q():
    return "SELECT project_id, datetime(MAX(completed_at), '-7 days') AS cutoff FROM task_results GROUP BY project_id"
'''

_UPPERCASE_SOURCE = '''
def q():
    return "SELECT 1 FROM t WHERE completed_at >= DATETIME('now', '-7 days')"
'''

_PROSE_DOCSTRING_SOURCE = '''
"""Module prose explaining why datetime('now', ...) must not reach SQLite."""


def q():
    """Function prose naming datetime("now", ...) the same way."""
    return 'SELECT 1'
'''

_PROSE_COMMENT_SOURCE = '''
# Comment naming datetime('now', ? || ' days') in prose.
def q():
    return 'SELECT 1'  # and datetime("now", ...) named again here
'''

_BUCKETING_SOURCE = '''
def q():
    return (
        "SELECT strftime('%Y-%m-%dT%H:00', completed_at) AS hour, "
        "date(completed_at) AS day FROM task_results"
    )
'''

_PYTHON_DATETIME_CONSTRUCTOR_SOURCE = 'epoch = datetime(1970, 1, 1, tzinfo=UTC)\n'

_PROSE_MESSAGE_TAG = '  # clock-exempt: prose message, not SQL'
_TAGGED_PROSE_MESSAGE_SOURCE = f'''
def check():
    raise TypeError('expected a datetime(...) value'){_PROSE_MESSAGE_TAG}
'''


def test_sql_datetime_folded_literal_fires():
    """A folded multi-line SQL literal is flagged at the line its concatenation starts."""
    violations = find_sql_datetime_violations(_FOLDED_SQL_SOURCE)

    assert len(violations) == 1, f'expected exactly one violation, got {violations!r}'
    assert violations[0][0] == 4
    assert "datetime('now'" in violations[0][1], (
        f'expected the excerpt to carry the folded literal value, got {violations[0][1]!r}'
    )
    assert not any(_SQL_DATETIME_CALL.search(line) for line in _FOLDED_SQL_SOURCE.splitlines()), (
        'fixture is no longer a folded literal: some physical line now matches '
        'on its own, so it would no longer discriminate an AST-literal report '
        'from a physical-line re-scan'
    )


@pytest.mark.parametrize(
    'source',
    [
        pytest.param(_SINGLE_QUOTED_NOW_SOURCE, id='single-quoted-now'),
        pytest.param(_DOUBLE_QUOTED_NOW_SOURCE, id='double-quoted-now'),
        pytest.param(_COLUMN_ANCHORED_SOURCE, id='column-anchored'),
        pytest.param(_UPPERCASE_SOURCE, id='uppercase'),
    ],
)
def test_sql_datetime_single_line_literal_fires(source: str):
    """Every spelling of a one-line SQL datetime() call is flagged at its line."""
    violations = find_sql_datetime_violations(source)

    assert len(violations) == 1, f'expected exactly one violation, got {violations!r}'
    assert violations[0][0] == 3


def test_sql_datetime_docstring_mention_ignored():
    """Module and function docstrings naming datetime() in prose are not queries."""
    assert find_sql_datetime_violations(_PROSE_DOCSTRING_SOURCE) == []


def test_sql_datetime_comment_mention_ignored():
    """A `#` comment naming datetime() is not a query."""
    assert find_sql_datetime_violations(_PROSE_COMMENT_SOURCE) == []


def test_sql_date_and_strftime_bucketing_ignored():
    """The data layer's `date()` / `strftime()` bucket labels are not datetime() calls."""
    assert find_sql_datetime_violations(_BUCKETING_SOURCE) == []


def test_python_datetime_constructor_ignored():
    """A Python `datetime(...)` constructor is a Call node, not a SQL string literal."""
    assert find_sql_datetime_violations(_PYTHON_DATETIME_CONSTRUCTOR_SOURCE) == []


def test_sql_datetime_exempt_tag_on_start_line_passes():
    """The `.now()` guard's `# clock-exempt:` escape hatch silences the SQL guard too."""
    untagged = _TAGGED_PROSE_MESSAGE_SOURCE.replace(_PROSE_MESSAGE_TAG, '')

    assert len(find_sql_datetime_violations(untagged)) == 1, (
        'fixture no longer fires untagged, so the tagged case proves nothing'
    )
    assert find_sql_datetime_violations(_TAGGED_PROSE_MESSAGE_SOURCE) == []


# ---------------------------------------------------------------------------
# Acceptance tests: the real tree (data layer + composition layer)
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent.parent / 'src'
_DATA_DIR = _SRC_ROOT / 'dashboard' / 'data'
_APP_PY = _SRC_ROOT / 'dashboard' / 'app.py'
_API_DIR = _SRC_ROOT / 'dashboard' / 'api'
_LOOPS_PY = _SRC_ROOT / 'dashboard' / 'loops.py'

# Seven route modules plus the package marker. A rename or a further split
# must fail loudly here rather than silently shrinking the scan.
_MIN_API_MODULES = 8


def _scanned_modules() -> list[Path]:
    """Every module the guards scan: the one definition of their shared boundary.

    Raises rather than returning a shorter list when a glob comes up empty
    or short, or a fixed target is missing, because a check that quietly
    stops checking is indistinguishable from a passing one.
    """
    data_files = sorted(_DATA_DIR.glob('*.py'))
    api_files = sorted(_API_DIR.glob('*.py'))

    assert data_files, f'no modules found under {_DATA_DIR}'
    assert len(api_files) >= _MIN_API_MODULES, (
        f'only {len(api_files)} modules found under {_API_DIR} '
        f'({[p.name for p in api_files]}) — fewer than the '
        f'{_MIN_API_MODULES} that existed when this guard was written'
    )
    for fixed_target in (_APP_PY, _LOOPS_PY):
        assert fixed_target.is_file(), f'scan target is missing: {fixed_target}'

    return data_files + [_APP_PY] + api_files + [_LOOPS_PY]


def test_no_bare_clock_reads_in_scanned_modules():
    """No scanned module has an untagged bare clock read outside `resolve_now`."""
    violations = [
        f'{path.relative_to(_SRC_ROOT)}:{lineno}: {text.strip()}'
        for path in _scanned_modules()
        for lineno, text in find_clock_violations(path.read_text())
    ]

    assert not violations, (
        'Bare datetime.now() reads found (missing resolve_now() or a '
        '`# clock-exempt:` tag):\n' + '\n'.join(violations)
    )


def test_no_deferred_consolidation_markers_remain():
    """No scanned module carries the `deferred-consolidation` grandfather tag (task 2281).

    Task 2192 grandfather-tagged 24 pre-existing bare clock reads across the 7
    data modules with `# clock-exempt: deferred-consolidation (task 2281)` to
    unblock the guard without doing the real work. Task 2281 retires every one
    of those markers by converting each site to either real `resolve_now(now)`
    threading or a genuine `# clock-exempt: single-capture ...` justification
    tag. This is the outer double-loop acceptance test: RED until the final
    module's marker is removed.
    """
    violations = [
        f'{path.relative_to(_SRC_ROOT)}:{lineno}: {text.strip()}'
        for path in _scanned_modules()
        for lineno, text in enumerate(path.read_text().splitlines(), start=1)
        if _DEFERRED_CONSOLIDATION_TAG in text
    ]

    assert not violations, (
        'deferred-consolidation clock-exempt markers still present (task 2281 '
        'must convert each to resolve_now() threading or a genuine '
        'single-capture justification tag):\n' + '\n'.join(violations)
    )


def test_no_sql_datetime_calls_in_scanned_modules():
    """No scanned module hands SQLite a `datetime()` call to compare against."""
    violations = [
        f'{path.relative_to(_SRC_ROOT)}:{lineno}: {excerpt}'
        for path in _scanned_modules()
        for lineno, excerpt in find_sql_datetime_violations(path.read_text())
    ]

    assert not violations, (
        "SQL-side datetime() call(s) found. SQLite's datetime() renders "
        "'YYYY-MM-DD HH:MM:SS' (space-separated, no UTC offset), so a lexical "
        'TEXT comparison against an ISO-with-offset column is wrong on the '
        'boundary day. Bind a Python-computed cutoff instead (see '
        'dashboard.data.performance._cutoff), or tag a literal that is prose '
        'rather than SQL `# clock-exempt:` on its start line:\n'
        + '\n'.join(violations)
    )
