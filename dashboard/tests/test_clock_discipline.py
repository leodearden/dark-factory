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
same protection as the data layer. Routes read the present only through
`dashboard/src/dashboard/api/clock.py::app_clock`, the clock their app has
installed, so a test can pin every instant a route serves. A stricter route
guard enforces that over app.py and `dashboard/api/*.py` (minus the seam's own
module): no `.now(` call there even when tagged, and no `resolve_now` call.
A third guard imports each route module and requires every call to a
dashboard function that takes `now`, `served_at` or `measured_at` to pass it,
so an installed clock controls a route's whole body, not just its stamp.
The scan covers
`dashboard/data/*.py`, `app.py`, `dashboard/api/*.py` and `loops.py` — the
data layer plus every module the composition layer was split into (task
5586 moved two tagged single-capture routes, `api_burndown` and
`api_merge_queue`, out of `app.py` into `dashboard/api/`, and the guard
follows them rather than quietly shedding the coverage). `_scanned_modules`
is the one definition of that set: every real-tree test below reads it, and
it fails loudly when any part of the set goes missing. No module beyond
those is scanned by any guard here; that boundary is intentional, not an
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
import importlib
import inspect
import re
import types
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


def _is_resolve_now_call(node: ast.Call) -> bool:
    func = node.func
    return (isinstance(func, ast.Name) and func.id == 'resolve_now') or (
        isinstance(func, ast.Attribute) and func.attr == 'resolve_now'
    )


def find_route_clock_violations(source: str) -> list[tuple[int, str]]:
    """Return ``(line, text)`` for every clock a route module reads past its app clock.

    Flags every ``<expr>.now(...)`` call whatever its line is tagged, and every
    call to ``resolve_now``: a route reads the present only by calling the
    clock ``app_clock`` hands it.
    """
    lines = source.splitlines()
    return [
        (node.lineno, lines[node.lineno - 1])
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == 'now')
            or _is_resolve_now_call(node)
        )
    ]


_CLOCK_PARAMETERS = frozenset({'now', 'served_at', 'measured_at'})


def _resolve_callee(module: types.ModuleType, func: ast.expr) -> object:
    """The object a ``name(...)`` or ``name.attr(...)`` call reaches through *module*'s globals."""
    if isinstance(func, ast.Name):
        return vars(module).get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return getattr(vars(module).get(func.value.id), func.attr, None)
    return None


_POSITIONAL_KINDS = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _passes(call: ast.Call, parameters: list[inspect.Parameter], parameter: inspect.Parameter) -> bool:
    if any(keyword.arg in (parameter.name, None) for keyword in call.keywords):
        return True
    if parameter.kind not in _POSITIONAL_KINDS:
        return False
    return (
        any(isinstance(arg, ast.Starred) for arg in call.args)
        or parameters.index(parameter) < len(call.args)
    )


def find_unthreaded_clock_calls(module: types.ModuleType, source: str) -> list[tuple[int, str]]:
    """Return ``(line, message)`` for every call in *source* that leaves a clocked callee's instant unset.

    A callee is clocked when it resolves through *module*'s globals to a
    ``dashboard.*`` function with a parameter in :data:`_CLOCK_PARAMETERS`.
    The call passes that parameter by keyword, by position, or through
    ``**kwargs``; ``*args`` counts only for a positional parameter, since it
    can never fill a keyword-only one. Otherwise the callee reads the live clock.
    """
    violations: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        callee = _resolve_callee(module, node.func)
        if not (inspect.isfunction(callee) and callee.__module__.startswith('dashboard.')):
            continue
        parameters = list(inspect.signature(callee).parameters.values())
        violations.extend(
            (node.lineno, f'{callee.__module__}.{callee.__qualname__} missing {parameter.name}=')
            for parameter in parameters
            if parameter.name in _CLOCK_PARAMETERS and not _passes(node, parameters, parameter)
        )
    return sorted(violations)


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


def test_route_guard_ignores_the_exemption_tag():
    """A tagged `.now(` that the module-wide guard accepts is still a route violation."""
    source = 'now = datetime.now(UTC)  # clock-exempt: single-capture route\n'
    assert find_clock_violations(source) == []
    assert [lineno for lineno, _ in find_route_clock_violations(source)] == [1]


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('render_at = resolve_now(None)\n', id='bare-name'),
        pytest.param('render_at = utils.resolve_now(None)\n', id='attribute'),
    ],
)
def test_route_guard_flags_resolve_now(source: str):
    """A route that resolves its own `now` bypasses the installed clock."""
    assert [lineno for lineno, _ in find_route_clock_violations(source)] == [1]


def test_route_guard_accepts_the_app_clock():
    """Calling the clock `app_clock` returns is the one sanctioned route read."""
    source = 'clock = app_clock(request.app)\nrender_at = clock()\nserved_at = clock()\n'
    assert find_route_clock_violations(source) == []


_CLOCKED_ROUTE_SOURCE = '''
async def reader(db, *, now=None):
    return now


def positional_reader(db, now=None):
    return now


async def route(db, now, options):
    await reader(db)
    await reader(db, now=now)
    await reader(db, **options)
    await reader(db, *options)
    positional_reader(db)
    positional_reader(db, now)
    positional_reader(db, *options)
'''


def _fixture_route_module() -> types.ModuleType:
    module = types.ModuleType('dashboard._clocked_route_fixture')
    exec(compile(_CLOCKED_ROUTE_SOURCE, '<fixture>', 'exec'), vars(module))
    return module


def test_unthreaded_clock_guard_flags_only_the_calls_that_drop_the_instant():
    """An un-threaded call fires, and so does ``*args`` against a keyword-only instant.

    Keyword, positional and ``**kwargs`` threading do not, nor does ``*args``
    into a positional instant.
    """
    violations = find_unthreaded_clock_calls(_fixture_route_module(), _CLOCKED_ROUTE_SOURCE)

    assert violations == [
        (11, 'dashboard._clocked_route_fixture.reader missing now='),
        (14, 'dashboard._clocked_route_fixture.reader missing now='),
        (15, 'dashboard._clocked_route_fixture.positional_reader missing now='),
    ]


def test_unthreaded_clock_guard_ignores_callees_outside_dashboard():
    """A clocked function from outside the dashboard package is not the route's to thread."""
    module = types.ModuleType('elsewhere')
    exec(compile(_CLOCKED_ROUTE_SOURCE, '<fixture>', 'exec'), vars(module))

    assert find_unthreaded_clock_calls(module, _CLOCKED_ROUTE_SOURCE) == []


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
_CLOCK_PY = _API_DIR / 'clock.py'

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


def _route_modules() -> list[Path]:
    """The route layer: app.py plus every dashboard/api module except the clock seam itself.

    ``api/clock.py`` is exempt the way ``resolve_now``'s own definition is: it
    is the sanctioned reader. It must exist, or the exemption names nothing.
    """
    assert _CLOCK_PY.is_file(), f'the route clock seam is missing: {_CLOCK_PY}'
    return [
        path for path in _scanned_modules()
        if path == _APP_PY or (path.parent == _API_DIR and path != _CLOCK_PY)
    ]


def test_routes_read_the_present_only_through_the_app_clock():
    """No route module reads a clock other than the one its app installed."""
    violations = [
        f'{path.relative_to(_SRC_ROOT)}:{lineno}: {text.strip()}'
        for path in _route_modules()
        for lineno, text in find_route_clock_violations(path.read_text())
    ]

    assert not violations, (
        'Route clock reads that bypass dashboard.api.clock.app_clock (take '
        '`clock = app_clock(request.app)` and call it instead):\n' + '\n'.join(violations)
    )


def _module_name(path: Path) -> str:
    parts = path.relative_to(_SRC_ROOT).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def test_routes_pass_their_instant_to_every_clocked_callee():
    """Every route call to a dashboard function taking now/served_at/measured_at passes it.

    A reader handed to ``asyncio.to_thread`` is an argument, not a Call node, and is out of reach.
    So is a bound method or a wrapped callable (``functools.partial``, an
    ``lru_cache`` wrapper): ``inspect.isfunction`` does not admit them.
    """
    violations = [
        f'{path.relative_to(_SRC_ROOT)}:{lineno}: {message}'
        for path in _route_modules()
        for lineno, message in find_unthreaded_clock_calls(
            importlib.import_module(_module_name(path)), path.read_text(),
        )
    ]

    assert not violations, (
        'Route calls that leave a reader on the live clock (pass the route '
        'instant from app_clock):\n' + '\n'.join(violations)
    )


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
