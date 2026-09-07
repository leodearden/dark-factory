"""Guard: no ``@pytest.mark.timeout(N)`` may INVERT into a tighter clamp under verify.

(Module docstring completed in a later step; this file is being built up
test-first.  See task 5147.)
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import NamedTuple

import yaml
from _orch_helpers import (
    ORCH_DIR,
    PYPROJECT_DEFAULT_TIMEOUT,
    VERIFY_CLI_PER_TEST_TIMEOUT,
)

from orchestrator.pytest_markers import _marker_name, _pytestmark_value

#: The per-module merge-verify config whose ``test_command`` carries the
#: ``--timeout=N`` that verify actually passes to pytest.  Resolved from
#: ``ORCH_DIR`` (itself resolved from ``_orch_helpers.__file__``) and never
#: from the process CWD: merge-verify runs pytest from the ``orchestrator/``
#: cwd while a plain ``pytest orchestrator/tests`` runs from the repo root,
#: and this pin must read identically under both.
_ORCH_YAML = ORCH_DIR / 'orchestrator.yaml'

#: This directory, resolved from THIS FILE and never from the process CWD, for
#: the same reason ``_ORCH_YAML`` is: the census must come out identical under
#: merge-verify (cwd ``orchestrator/``) and a bare ``pytest orchestrator/tests``
#: (cwd the repo root).  Same idiom as
#: test_whole_tree_scan_timeout_guard.py::_TESTS_DIR and
#: test_marker_registration_drift.py::TESTS_DIR.
_TESTS_DIR = Path(__file__).resolve().parent

#: Same spelling as tests/scripts/test_fallback_verify_config.py, which pins
#: the FLEET-chain side of this same budget (``--timeout > 60`` on every
#: pytest segment of dark-factory-orchestrator.yaml, and ``--timeout >= 300``
#: on every per-module orchestrator.yaml).  Both the ``--timeout=300`` and
#: ``--timeout 300`` spellings are accepted, exactly as it does.
_TIMEOUT_FLAG_RE = re.compile(r'--timeout[=\s](\d+)')

#: Names a ``pytest.mark.timeout(...)`` argument may resolve to, and the
#: seconds each one carries.  A literal MAP rather than an import, because none
#: of the three is importable from ``_orch_helpers``: two are defined
#: FILE-LOCALLY in the modules that use them --
#: ``HEAVY_BARRIER_TEST_TIMEOUT = 5 * MERGE_RESULT_TIMEOUT + 75  # 300s``
#: (test_merge_queue_concurrent_verify.py) and ``PYTEST_TIMEOUT = 960`` (in
#: BOTH test_pytest_marker_deselection.py and test_warm_lane_bash_suite.py).
#: This is where the shape departs from its template:
#: ``_SANCTIONED_CEILING_NAMES`` in test_whole_tree_scan_timeout_guard.py is a
#: one-element frozenset paired with a single hard-coded
#: ``float(WHOLE_TREE_SCAN_TEST_TIMEOUT)``, which does not generalise to three
#: names at two distinct values.
#:
#: These are cross-module MIRRORS, so state the failure mode plainly: a mirror
#: that goes stale can only ever produce a false OFFENDER, never a false pass.
#: Every value here sits OUTSIDE the inversion band (300 at its open upper
#: edge, 960 well clear), so the only way a wrong number changes an answer is
#: by dragging a name INTO the band -- which fails loudly at commit time and
#: names the site.  Silence is not among the outcomes.
_SANCTIONED_TIMEOUT_NAMES: dict[str, float] = {
    'WHOLE_TREE_SCAN_TEST_TIMEOUT': 300.0,
    'HEAVY_BARRIER_TEST_TIMEOUT': 300.0,
    'PYTEST_TIMEOUT': 960.0,
    'VERIFY_CLI_PER_TEST_TIMEOUT': float(VERIFY_CLI_PER_TEST_TIMEOUT),
}

#: Qualname suffix for a ``pytestmark`` binding inside a class body, and the
#: whole qualname for a module-level one.  Angle brackets because no Python
#: identifier can contain them, so these can never collide with a real
#: function or class name in the allowlist's ``(module, qualname)`` key.
_MODULE_QUALNAME = '<module>'
_PYTESTMARK_QUALNAME = '<pytestmark>'


class _Site(NamedTuple):
    """One ``pytest.mark.timeout(...)`` occurrence found in a source file.

    ``seconds`` is None when the argument is present but UNRESOLVABLE, or
    absent entirely -- "no opinion", never "too small".  See
    :func:`_timeout_marker_sites`.
    """

    qualname: str
    kind: str
    seconds: float | None
    lineno: int


def _timeout_call_arg(call: ast.Call) -> ast.expr | None:
    """The seconds expression a ``pytest.mark.timeout(...)`` *call* pins.

    Both spellings pytest-timeout accepts are read: positional ``timeout(300)``
    and keyword ``timeout(timeout=300)``.  A call with neither (``timeout()``,
    or one passing only ``method=``) yields None.  Same contract as
    test_whole_tree_scan_timeout_guard.py's function of this name; kept
    separate rather than imported because that module is a guard, not a
    helper library, and importing across two guards would couple their
    collection order.
    """
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg == 'timeout':
            return keyword.value
    return None


def _resolve_seconds(arg: ast.expr | None) -> float | None:
    """Seconds *arg* pins, if statically knowable.

    RESOLUTION, deliberately tiny -- generalised from
    ``_module_level_timeout_ceiling``'s rules
    (test_whole_tree_scan_timeout_guard.py):

    * a numeric literal resolves to itself.  ``bool`` is excluded explicitly:
      it is an ``int`` subclass, so ``timeout(True)`` would otherwise resolve
      to 1.0 and read as an absurdly tight bound;
    * a name in :data:`_SANCTIONED_TIMEOUT_NAMES`, bare (``ast.Name``, the
      house ``from _orch_helpers import`` idiom) or dotted (``ast.Attribute``,
      compared on the trailing name only), resolves to that constant's value;
    * ANYTHING else -- arithmetic, an ``int(...)`` call, an f-string, an
      unfamiliar constant -- is UNKNOWABLE and yields None.

    None means "no opinion", never "too small": :func:`_inverts` must not treat
    it as an offence.  The consequence, stated rather than hidden: a marker
    that pins an in-band value through an indirection this grammar cannot
    follow is NOT caught.  Like the whole sweep, this is a FLOOR.
    """
    if arg is None:
        return None
    if (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, int | float)
        and not isinstance(arg.value, bool)
    ):
        return float(arg.value)
    name: str | None = None
    if isinstance(arg, ast.Name):
        name = arg.id
    elif isinstance(arg, ast.Attribute):
        name = arg.attr
    if name is None:
        return None
    return _SANCTIONED_TIMEOUT_NAMES.get(name)


def _timeout_sites_in(elements: list[ast.expr], qualname: str, kind: str) -> list[_Site]:
    """Every ``timeout`` mark among *elements*, as sites keyed *qualname*/*kind*.

    *elements* is a decorator list or the unpacked value of a ``pytestmark``
    binding.  Non-``timeout`` marks yield no site at all (rather than a site
    with None seconds), so an ``@pytest.mark.asyncio`` never shows up in a
    census of timeout coverage.
    """
    sites: list[_Site] = []
    for element in elements:
        if not isinstance(element, ast.Call) or _marker_name(element) != 'timeout':
            continue
        sites.append(
            _Site(
                qualname=qualname,
                kind=kind,
                seconds=_resolve_seconds(_timeout_call_arg(element)),
                lineno=element.lineno,
            )
        )
    return sites


def _mark_elements(value: ast.expr) -> list[ast.expr]:
    """A ``pytestmark`` binding's marks, unwrapping the list/tuple form."""
    return list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]


def _timeout_marker_sites(source: str) -> tuple[_Site, ...]:
    """Every ``pytest.mark.timeout(...)`` site in *source*, with its resolved seconds.

    WHY THIS EXISTS SEPARATELY from
    ``test_whole_tree_scan_timeout_guard.py::_module_level_timeout_ceiling``:
    that helper answers "what MODULE-LEVEL ceiling does this file pin", which
    is the right question for a per-FILE family invariant and the wrong one
    here.  A module-level ``pytestmark`` is the only form that is a sound LOWER
    bound on every collected item, which is exactly why that guard reads it and
    nothing else -- and exactly why it is blind to the population this module
    polices.  The measured census found 62 in-band markers and only a handful
    were module-level; the dominant spellings are the per-test DECORATOR and
    the per-CLASS decorator, neither of which that helper can see.  So this
    generalises its value resolution rather than replacing it: the existing
    guard keeps its narrower, stricter family invariant untouched.

    FOUR BINDING FORMS are collected, each keyed by a qualname that identifies
    the site stably across ordinary edits (the allowlist is keyed on
    ``(module, qualname)``, never a line number):

    * a function/method decorator -> ``test_a`` or ``TestThing::test_a``;
    * a class decorator -> ``TestThing``;
    * a module-level ``pytestmark`` -> ``<module>``;
    * a class-level ``pytestmark`` -> ``TestThing::<pytestmark>``.

    Nesting deeper than one class is walked for classes but not for closures:
    a decorator on a function defined INSIDE another function is not a
    collected pytest item, so it is not a site.

    ``_marker_name`` and ``_pytestmark_value`` are imported from
    :mod:`orchestrator.pytest_markers` rather than re-derived, for the same
    reason test_whole_tree_scan_timeout_guard.py imports them: the grammar of a
    ``pytest.mark.NAME`` element and of a ``pytestmark`` binding (``Assign`` vs
    ``AnnAssign``, list/tuple element forms) belongs in exactly one place, and
    a rename there should break this import loudly at collection rather than
    let two readings of the same syntax drift apart.

    FAIL-SOFT: unparseable source yields an EMPTY tuple and never raises.  The
    sweep reads every ``*.py`` under this directory, deliberately-malformed
    fixtures included, and a parse failure must not turn a timeout-coverage
    guard red for a reason unrelated to timeout coverage.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()

    sites: list[_Site] = []

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for statement in body:
            bound = _pytestmark_value(statement)
            if bound is not None:
                qualname = f'{prefix}{_PYTESTMARK_QUALNAME}' if prefix else _MODULE_QUALNAME
                kind = 'class-pytestmark' if prefix else 'module-pytestmark'
                sites.extend(_timeout_sites_in(_mark_elements(bound), qualname, kind))
            if isinstance(statement, ast.ClassDef):
                qualname = f'{prefix}{statement.name}'
                sites.extend(
                    _timeout_sites_in(statement.decorator_list, qualname, 'class-decorator')
                )
                walk(statement.body, f'{qualname}::')
            elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                sites.extend(
                    _timeout_sites_in(
                        statement.decorator_list, f'{prefix}{statement.name}', 'decorator'
                    )
                )

    walk(tree.body, '')
    return tuple(sites)


def _inverts(seconds: float | None) -> bool:
    """True iff a marker at *seconds* TIGHTENS verify while reading as a loosening.

    The band is ``(PYPROJECT_DEFAULT_TIMEOUT, VERIFY_CLI_PER_TEST_TIMEOUT)``,
    open at both ends.  Named against the constants rather than their numbers,
    which live -- with the full rationale -- in _orch_helpers.py; the three
    regimes those two edges carve out are:

    * at or below the ini default -- tightens under BOTH budgets, so a
      deliberate tight bound.  NOT an offence;
    * strictly between them -- INVERTS.  Written to loosen against the default
      the author was reading, it silently becomes a tightening under verify's
      CLI budget.  The sign of the marker's effect flips with context, which is
      the defect;
    * at or above the CLI budget -- loosens under both.  Safe.

    None is not a number and cannot invert: unresolvable means "no opinion",
    never "too small".
    """
    return seconds is not None and PYPROJECT_DEFAULT_TIMEOUT < seconds < VERIFY_CLI_PER_TEST_TIMEOUT


class TestVerifyCliBudgetConstant:
    """``VERIFY_CLI_PER_TEST_TIMEOUT`` -- the budget the whole guard is built on.

    Pinned here for the same reason ``PYPROJECT_DEFAULT_TIMEOUT`` is pinned by
    test_whole_tree_scan_timeout_guard.py::TestTimeoutConstants: a shared
    timeout constant that merely *claims* in a comment to mirror a config file
    is drift-prone, and the executable link is what makes the claim true.  This
    class is that shape one anchor over -- it reads the REAL
    orchestrator/orchestrator.yaml at runtime instead of citing a line number.
    """

    def test_verify_cli_budget_is_three_hundred(self) -> None:
        """The constant names the verify CLI per-test budget, currently 300s."""
        assert VERIFY_CLI_PER_TEST_TIMEOUT == 300

    def test_the_inversion_band_is_non_empty(self) -> None:
        """The band ``(PYPROJECT_DEFAULT_TIMEOUT, VERIFY_CLI_PER_TEST_TIMEOUT)`` must be real.

        The entire guard is the predicate "N sits strictly between the ini
        default and the verify CLI budget".  If the two ever converge or
        invert, that predicate becomes unsatisfiable and every sweep below
        would pass VACUOUSLY -- green because nothing can offend, not because
        nothing does.  Asserted rather than assumed, since both edges are
        mirrors of config files that can move independently.
        """
        assert PYPROJECT_DEFAULT_TIMEOUT < VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'the inversion band ({PYPROJECT_DEFAULT_TIMEOUT}, '
            f'{VERIFY_CLI_PER_TEST_TIMEOUT}) is empty -- the ini default has '
            'caught up with the verify CLI budget, so nothing can invert and '
            'this whole module would pass vacuously. Revisit it before '
            'changing either constant.'
        )

    def test_constant_mirrors_the_real_verify_test_command(self) -> None:
        """``VERIFY_CLI_PER_TEST_TIMEOUT`` must equal the ``--timeout`` verify really passes.

        THE DRIFT PIN.  The constant is deliberately a literal ``300`` rather
        than an expression over ``PYPROJECT_DEFAULT_TIMEOUT`` (see its comment
        in _orch_helpers.py), which means nothing about the Python source keeps
        it honest.  THIS test is what does: it re-reads
        ``orchestrator/orchestrator.yaml``'s ``test_command`` and extracts the
        ``--timeout`` token that pytest-timeout will actually see as
        ``config._env_timeout``.  If an operator retunes that flag, the band's
        upper edge moves with it and this fails loudly instead of leaving the
        guard silently policing a budget nobody passes any more.
        """
        test_command = yaml.safe_load(_ORCH_YAML.read_text(encoding='utf-8'))['test_command']

        match = _TIMEOUT_FLAG_RE.search(test_command)
        assert match, (
            f'{_ORCH_YAML} test_command carries no --timeout override '
            f'(got: {test_command!r}). VERIFY_CLI_PER_TEST_TIMEOUT models that '
            'flag, so without it the constant models nothing -- and every test '
            f'here silently falls back to the {PYPROJECT_DEFAULT_TIMEOUT}s '
            'pyproject default. This is also pinned from the other side by '
            'tests/scripts/test_fallback_verify_config.py::'
            'test_per_module_merge_verify_raises_per_test_timeout.'
        )
        configured = int(match.group(1))
        assert configured == VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'VERIFY_CLI_PER_TEST_TIMEOUT ({VERIFY_CLI_PER_TEST_TIMEOUT}) no '
            f'longer mirrors --timeout={configured} in {_ORCH_YAML}. Update the '
            'constant in orchestrator/tests/_orch_helpers.py -- the inversion '
            'band this module polices is (PYPROJECT_DEFAULT_TIMEOUT, '
            'VERIFY_CLI_PER_TEST_TIMEOUT), so a stale upper edge either lets a '
            'genuinely-inverting marker through or manufactures false '
            'offenders.'
        )


# ---------------------------------------------------------------------------
# _timeout_marker_sites(source) -- inline-fixture unit tests.
#
# Same shape as the sibling guards' pure-detector tests
# (test_whole_tree_scan_timeout_guard.py::test_detector_flags_rglob_py,
# test_raw_semaphore_access_guard.py, test_prune_chokepoint_guard.py): the
# extractor is exercised against synthetic snippets so every resolution rule is
# pinned DIRECTLY, rather than only ever being reached through the real tree --
# which goes green by construction once the allowlist lands and would otherwise
# leave the negative cases untested forever.
# ---------------------------------------------------------------------------


def _sites(source: str) -> dict[str, float | None]:
    """``{qualname: seconds}`` for *source*, dedented so fixtures can be indented."""
    return {site.qualname: site.seconds for site in _timeout_marker_sites(textwrap.dedent(source))}


def test_extractor_reads_a_bare_literal_function_decorator() -> None:
    """The canonical spelling, and the one the named regression instance used."""
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(120)
        def test_slow() -> None:
            pass
        """
    ) == {'test_slow': 120.0}


def test_extractor_reads_the_keyword_spelling() -> None:
    """``timeout(timeout=120)`` -- the second spelling pytest-timeout accepts.

    Read for the same reason ``_timeout_call_arg`` reads it: a marker written
    this way clamps exactly as hard as the positional form, so skipping it
    would leave a silent hole in the sweep.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(timeout=120)
        def test_slow() -> None:
            pass
        """
    ) == {'test_slow': 120.0}


def test_extractor_reads_module_level_pytestmark() -> None:
    """A module-level ``pytestmark`` binds every item in the file."""
    assert _sites(
        """
        import pytest

        pytestmark = pytest.mark.timeout(120)
        """
    ) == {'<module>': 120.0}


def test_extractor_reads_list_form_pytestmark() -> None:
    """The list form, where the timeout mark sits among unrelated siblings."""
    assert _sites(
        """
        import pytest

        pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(120)]
        """
    ) == {'<module>': 120.0}


def test_extractor_reads_a_class_decorator() -> None:
    """A class decorator binds every method in the class at once.

    The dominant in-band spelling in this repo by count: the merge-queue and
    crash-recovery modules carry ~34 of them at 180s, so an extractor blind to
    class decorators would miss most of the population.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(180)
        class TestThing:
            def test_a(self) -> None:
                pass
        """
    ) == {'TestThing': 180.0}


def test_extractor_reads_class_level_pytestmark() -> None:
    """``pytestmark`` inside a class body -- same binding, different syntax."""
    assert _sites(
        """
        import pytest

        class TestThing:
            pytestmark = pytest.mark.timeout(180)

            def test_a(self) -> None:
                pass
        """
    ) == {'TestThing::<pytestmark>': 180.0}


def test_extractor_gives_a_method_a_dotted_qualname() -> None:
    """A decorated METHOD is keyed ``TestClass::test_method``.

    The allowlist is keyed on ``(module, qualname)`` rather than a line number
    so it survives ordinary edits, which means the qualname must actually
    disambiguate: two classes in one module routinely carry same-named
    methods, and a bare ``test_a`` would silently collapse them into one entry.
    """
    assert _sites(
        """
        import pytest

        class TestOne:
            @pytest.mark.timeout(120)
            def test_a(self) -> None:
                pass

        class TestTwo:
            @pytest.mark.timeout(150)
            def test_a(self) -> None:
                pass
        """
    ) == {'TestOne::test_a': 120.0, 'TestTwo::test_a': 150.0}


def test_extractor_resolves_a_sanctioned_constant_name() -> None:
    """Bare and dotted sanctioned names both resolve to their seconds.

    The house idiom is a bare ``from _orch_helpers import ...``; the dotted
    form is accepted too, comparing on the trailing name only, exactly as
    ``_module_level_timeout_ceiling`` does.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)
        def test_bare() -> None:
            pass

        @pytest.mark.timeout(_orch_helpers.WHOLE_TREE_SCAN_TEST_TIMEOUT)
        def test_dotted() -> None:
            pass

        @pytest.mark.timeout(PYTEST_TIMEOUT)
        def test_warm_lane() -> None:
            pass
        """
    ) == {'test_bare': 300.0, 'test_dotted': 300.0, 'test_warm_lane': 960.0}


def test_extractor_yields_none_for_an_unresolvable_expression() -> None:
    """A computed argument is UNKNOWABLE -- None, never a number.

    "None means no opinion, never too small" is the polarity
    ``_module_level_timeout_ceiling`` already establishes, and it matters here
    for a concrete population: test_laptop_warm_verify_boundary.py's five
    ``int(...)``-derived marks. Those modules carry their own file-local
    derived-budget adequacy guards, so resolving them to a guess and failing
    them would manufacture offenders whose only "fix" is deleting a sounder,
    more specific guard.
    """
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout(int(ROW_BUDGET * 2))
        def test_derived() -> None:
            pass
        """
    ) == {'test_derived': None}


def test_extractor_yields_none_for_a_zero_arg_timeout_mark() -> None:
    """``timeout()`` (or one passing only ``method=``) pins no seconds."""
    assert _sites(
        """
        import pytest

        @pytest.mark.timeout()
        def test_a() -> None:
            pass

        @pytest.mark.timeout(method='signal')
        def test_b() -> None:
            pass
        """
    ) == {'test_a': None, 'test_b': None}


def test_extractor_ignores_marks_that_are_not_timeout() -> None:
    """A non-``timeout`` mark is not a site at all -- not a site with None."""
    assert (
        _sites(
            """
            import pytest

            @pytest.mark.asyncio
            @pytest.mark.slow
            def test_a() -> None:
                pass
            """
        )
        == {}
    )


def test_extractor_fails_soft_on_a_syntax_error() -> None:
    """Unparseable source yields an EMPTY tuple and never raises.

    The sweep below reads every ``*.py`` under this directory, which includes
    deliberately-malformed fixtures. A parse failure must not turn a
    TIMEOUT-COVERAGE guard red for a reason unrelated to timeout coverage --
    the very class of misattributed failure this module exists to prevent.
    """
    assert _timeout_marker_sites('def test_a(:\n') == ()


# ---------------------------------------------------------------------------
# _inverts(seconds) -- boundary table.
# ---------------------------------------------------------------------------


def test_the_band_edges_are_exactly_where_the_design_puts_them() -> None:
    """``_inverts`` is True only strictly inside ``(60, 300)``.

    THE WHOLE DESIGN LIVES IN THESE EDGES, so every one is pinned rather than
    left to a spot check.

    CLOSED AT THE BOTTOM -- ``N <= PYPROJECT_DEFAULT_TIMEOUT`` is NOT an
    offence, even though the task text says "N < 300".  A marker at or under
    the ini default tightens under BOTH budgets, which makes it unambiguously a
    deliberate tight bound rather than an accident: its author chose tighter
    than even a bare local ``pytest`` would give them.  The worked example is
    test_verify_clock_stop.py, which carries 13 marks at 15s because those
    tests assert a watchdog fires FAST -- raising them to 300 would blunt the
    assertion AND turn each hang test into a 300s stall.  Reading "N < 300"
    literally would sweep those in and be actively wrong.

    OPEN AT THE TOP -- ``N >= VERIFY_CLI_PER_TEST_TIMEOUT`` loosens under both
    budgets and is safe, which is what keeps the sanctioned ceilings
    (WHOLE_TREE_SCAN_TEST_TIMEOUT and HEAVY_BARRIER_TEST_TIMEOUT at 300,
    PYTEST_TIMEOUT at 960) out of the guard's way -- including this module's
    own ``pytestmark``.

    None -> False is the fail-soft polarity from :func:`_resolve_seconds`:
    "no opinion", never "too small".
    """
    assert [
        _inverts(seconds)
        for seconds in (None, 15, 60, 61, 90, 120, 150, 180, 299, 300, 360, 960)
    ] == [
        False,  # None       -- unresolvable: no opinion, never an offence
        False,  # 15         -- test_verify_clock_stop.py's watchdog marks
        False,  # 60         -- exactly the ini default: expresses no opinion
        True,  # 61          -- first inverting value
        True,  # 90          -- measured, test_merge_queue.py
        True,  # 120         -- measured, the named regression instance
        True,  # 150         -- measured, test_offline_lane_integration.py
        True,  # 180         -- measured, the most common in-band value (34 sites)
        True,  # 299         -- last inverting value
        False,  # 300        -- WHOLE_TREE_SCAN / HEAVY_BARRIER / the CLI budget
        False,  # 360        -- loosens under both
        False,  # 960        -- PYTEST_TIMEOUT (warm-lane bash bucket)
    ]


# ---------------------------------------------------------------------------
# The named regression instance.
# ---------------------------------------------------------------------------

#: The module whose in-band marker is the CONFIRMED, MEASURED instance of this
#: defect -- the one that motivated the task.  Named as a constant so the pin
#: below reads as a regression test for a specific historical failure rather
#: than as an arbitrary sample of the tree.
_NAMED_REGRESSION_MODULE = 'test_aiosqlite_leak_isolation.py'


def test_the_aiosqlite_leak_isolation_regression_is_fixed() -> None:
    """No marker in test_aiosqlite_leak_isolation.py may sit in the inversion band.

    THE MEASURED INSTANCE, recorded here so the number is never re-guessed.

    ``test_a_thread_exception_actually_fails_a_test_under_this_projects_inifile``
    carried ``@pytest.mark.timeout(120)``.  It spawns TWO full pytest
    subprocesses -- a treatment arm and a control arm -- and measures 15.27s
    unloaded (``8 passed in 17.72s`` for the module).  This suite's own
    measured load inflation for that class of work is ~4.8x: 30.75s at loadavg
    120-176 for a 6.46s unloaded scan, recorded against
    WHOLE_TREE_SCAN_TEST_TIMEOUT in _orch_helpers.py.  15.27 x 4.8 is ~73s,
    already 61% of the old 120s budget -- and the loadavg 250-423 step at which
    xdist worker DEATHS were actually observed sits one further inflation step
    beyond that.  So 120 was genuinely too tight; VERIFY_CLI_PER_TEST_TIMEOUT
    leaves ~19.6x headroom over the measurement.

    WHY IT COST WHOLE RUNS rather than one test: ``timeout_method = "thread"``
    answers a breach by ``os._exit()``ing the xdist worker, and
    ``--max-worker-restart=0`` declines to replace it, truncating the session
    and blaming whatever innocent test shared the dead worker.  That is how
    this one marker was blamed for failures in tasks 4176, 4384 and 4405.

    Asserted over the WHOLE module rather than the one qualname: the point is
    that this file stays clean, not that one line stays fixed.
    """
    source = (_TESTS_DIR / _NAMED_REGRESSION_MODULE).read_text(encoding='utf-8')

    offenders = [site for site in _timeout_marker_sites(source) if _inverts(site.seconds)]

    assert not offenders, (
        f'{_NAMED_REGRESSION_MODULE} has regressed to an inverting timeout '
        f'marker: {[(s.qualname, s.seconds) for s in offenders]}. This module '
        'spawns two full pytest subprocesses per test; at 15.27s unloaded and '
        "this suite's measured ~4.8x load inflation it reaches ~73s, which a "
        'marker inside the band clamps below. Use VERIFY_CLI_PER_TEST_TIMEOUT.'
    )
