"""Clock-discipline guard: no bare `datetime.now()` reads in the data layer.

Request-scoped code must resolve `now` once (via
:func:`dashboard.data.utils.resolve_now`) and thread it through, rather than
letting each function read the live clock independently. This module parses
each `dashboard/src/dashboard/data/*.py` file with `ast` and flags every
`<expr>.now(...)` Call node that is neither tagged `# clock-exempt:` on its
physical source line nor part of `resolve_now`'s own definition (the
sanctioned single clock-read site).

`dashboard/src/dashboard/app.py` — the route/composition layer sitting on
top of the data layer — is scanned too (see
`test_no_bare_clock_reads_in_app_composition_layer` below). A future
regression there (e.g. a route reverting to a per-leg `datetime.now(UTC)`
instead of one shared capture) would reintroduce exactly the cross-DB
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
follows them rather than quietly shedding the coverage). No module beyond
those is scanned; that boundary is intentional, not an oversight.

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

import pytest

_EXEMPT_MARKER = '# clock-exempt:'
_DEFERRED_CONSOLIDATION_TAG = f'{_EXEMPT_MARKER} deferred-consolidation'


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


# ---------------------------------------------------------------------------
# Acceptance tests: the real tree (data layer + composition layer)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'data'
_APP_PY = Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'app.py'
_API_DIR = Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'api'
_LOOPS_PY = Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'loops.py'

# Seven route modules plus the package marker. A rename or a further split
# must fail loudly here rather than silently shrinking the scan.
_MIN_API_MODULES = 8


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


def test_no_deferred_consolidation_markers_remain():
    """No data module carries the `deferred-consolidation` grandfather tag (task 2281).

    Task 2192 grandfather-tagged 24 pre-existing bare clock reads across the 7
    data modules with `# clock-exempt: deferred-consolidation (task 2281)` to
    unblock the guard without doing the real work. Task 2281 retires every one
    of those markers by converting each site to either real `resolve_now(now)`
    threading or a genuine `# clock-exempt: single-capture ...` justification
    tag. This is the outer double-loop acceptance test: RED until the final
    module's marker is removed.
    """
    violations: list[str] = []
    for path in sorted(_DATA_DIR.glob('*.py')):
        rel = path.relative_to(_DATA_DIR.parent.parent)
        for lineno, text in enumerate(path.read_text().splitlines(), start=1):
            if _DEFERRED_CONSOLIDATION_TAG in text:
                violations.append(f'{rel}:{lineno}: {text.strip()}')

    assert not violations, (
        'deferred-consolidation clock-exempt markers still present (task 2281 '
        'must convert each to resolve_now() threading or a genuine '
        'single-capture justification tag):\n' + '\n'.join(violations)
    )


def test_no_bare_clock_reads_in_app_composition_layer():
    """`app.py` route handlers must tag or thread every clock read too.

    This is the layer that calls into the data layer's aggregate functions
    (e.g. `aggregate_burndown_series`, `aggregate_cost_summary`) — a route
    that silently reverted to reading the clock per fan-out leg instead of
    capturing `now` once and threading it through would reintroduce the
    same clock-skew race this guard blocks in the data layer, just one
    level up the call stack. See the module docstring for the
    `single-capture route` tag convention this test enforces.
    """
    source = _APP_PY.read_text()
    violations = [f'{_APP_PY.name}:{lineno}: {text.strip()}' for lineno, text in find_clock_violations(source)]

    assert not violations, (
        'Bare datetime.now() reads found in app.py (missing a '
        '`# clock-exempt:` tag):\n' + '\n'.join(violations)
    )


def test_no_bare_clock_reads_in_extracted_route_and_loop_modules():
    """The modules split out of `app.py` carry the same clock discipline.

    Task 5586 moved six route handlers and the two background samplers out
    of `app.py`. Two of those handlers (`api_burndown`, `api_merge_queue`)
    read the clock once per request under a `# clock-exempt: single-capture
    route` tag, so scanning only `app.py` after the move would drop their
    coverage without a single test turning red — the exact way a guard rots.
    The member count is asserted first so a later rename or split fails
    loudly instead of quietly emptying the scan.
    """
    api_files = sorted(_API_DIR.glob('*.py'))

    assert len(api_files) >= _MIN_API_MODULES, (
        f'only {len(api_files)} modules found under {_API_DIR} '
        f'({[p.name for p in api_files]}) — fewer than the '
        f'{_MIN_API_MODULES} that existed when this guard was written; '
        'a check that quietly stops checking is indistinguishable from a '
        'passing one'
    )
    assert _LOOPS_PY.is_file(), f'scan target is missing: {_LOOPS_PY}'

    scanned = api_files + [_LOOPS_PY]

    violations: list[str] = []
    for path in scanned:
        rel = path.relative_to(_API_DIR.parent.parent)
        for lineno, text in find_clock_violations(path.read_text()):
            violations.append(f'{rel}:{lineno}: {text.strip()}')

    assert not violations, (
        'Bare datetime.now() reads found in the modules extracted from '
        'app.py (missing a `# clock-exempt:` tag):\n' + '\n'.join(violations)
    )


def test_no_sql_datetime_calls_in_dashboard_modules():
    """No scanned module hands SQLite a `datetime()` call to compare against."""
    scanned = (
        sorted(_DATA_DIR.glob('*.py'))
        + [_APP_PY]
        + sorted(_API_DIR.glob('*.py'))
        + [_LOOPS_PY]
    )

    assert _DATA_DIR / 'performance.py' in scanned, (
        f'{_DATA_DIR / "performance.py"} is missing from the scan list; a check '
        'that quietly stops checking is indistinguishable from a passing one'
    )

    violations: list[str] = []
    for path in scanned:
        rel = path.relative_to(_DATA_DIR.parent.parent)
        for lineno, excerpt in find_sql_datetime_violations(path.read_text()):
            violations.append(f'{rel}:{lineno}: {excerpt}')

    assert not violations, (
        "SQL-side datetime() call(s) found. SQLite's datetime() renders "
        "'YYYY-MM-DD HH:MM:SS' (space-separated, no UTC offset), so a lexical "
        'TEXT comparison against an ISO-with-offset column is wrong on the '
        'boundary day. Bind a Python-computed cutoff instead (see '
        'dashboard.data.performance._cutoff):\n' + '\n'.join(violations)
    )
