"""Static guard: every pytest marker APPLIED under orchestrator/tests/ must be
REGISTERED, so a typo (e.g. ``@pytest.mark.slwo``) fails loudly at collection
instead of silently degrading to an unknown-marker warning forever (task 3532,
follow-up to task 3506 — escalation agent-followup-3506).

WHY NOT ``--strict-markers``. The task offered two options; this module is
option (b), and the choice was a MEASUREMENT, not a preference. On pytest
9.0.3 (the version this project resolves), ``--strict-markers`` placed in
``[tool.pytest.ini_options] addopts`` is INERT: a planted
``@pytest.mark.slwo`` test collects clean, emits only
``PytestUnknownMarkWarning``, and PASSES — exit 0 — even though
``pytestconfig.option.strict_markers`` reads True and ``getini('addopts')``
contains the flag, i.e. addopts IS parsed and the option IS set; it simply
does not gate collection. The identical flag on the COMMAND LINE does gate:
``Failed: 'slwo' not found in `markers` configuration option``, exit 2.
Reproduced in a bare, conftest-free project on pytest 9.0.3, in both the
addopts string form and the TOML-list form. Shipping ``--strict-markers`` in
addopts would therefore look like enforcement while enforcing nothing — worse
than doing nothing, since it would stop the next reader from looking. See
``orchestrator/pyproject.toml``'s comment above ``markers = [`` for the short
version of this note.

WHAT "REGISTERED" MEANS. The effective set from ``Config.getini('markers')``
— NOT a ``tomllib`` read of pyproject.toml's ``markers`` list.
``real_psi_reader`` is registered dynamically at ``tests/conftest.py:403``
(task 2418) via ``config.addinivalue_line('markers', ...)``, deliberately
outside pyproject.toml's locked scope; a TOML-only comparison would
false-fail on it. ``test_conftest_registered_markers_count_as_registered``
below pins this so the guard can't later be "simplified" back into a
TOML-only read.

WHY A STATIC DISK SWEEP, not a ``pytest_collection_modifyitems`` / live
``session.items`` walk. ``verify_plan.classify_file`` is purely path-based,
so a diff touching only THIS file produces a FILE_SCOPED run of exactly this
file; ``session.items`` would then hold only this module's own items, and the
guard would pass while checking nothing else — the exact silent-vacuity trap
this repo refuses (cf. ``test_warm_lane_bash_bucket_placement.py``). A disk
sweep of ``tests_dir.rglob('*.py')`` reads every file regardless of what a
given invocation happened to collect.

RESIDUAL GAPS, deliberately out of scope and recorded rather than silently
omitted:
* a typo inside an ``-m`` EXPRESSION (``-m 'not slwo'``) stays silent —
  pytest never validates marker EXPRESSIONS, under ``--strict-markers`` or
  here;
* ``pytest.param(marks=...)`` and ``item.add_marker(...)`` are not swept —
  zero uses under ``orchestrator/tests/`` today (grep-verified);
* an aliased pytest import (``import pytest as _pytest``) would not be
  recognised as a marker application — two such aliases exist today
  (test_overlap_footprint.py, test_multihost_verify_integration.py) but both
  use the alias only for ``.raises``, never ``.mark``, so this is a real but
  currently-inert gap;
* sibling packages (shared, fused-memory, escalation) have their own markers
  (e.g. ``integration``) and no equivalent guard.

This module is built in two halves, in that order: unit tests of the pure
static helpers against synthetic strings/trees (``TestAppliedMarkerNames``,
``TestUnregisteredMarkers``), then the wired guard against the REAL tree and
the REAL pytest config (``TestMarkerRegistrationDrift``).
"""
from __future__ import annotations

import ast
import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest


def _marker_name(element: ast.expr) -> str | None:
    """The marker name in a ``pytest.mark.NAME`` / ``pytest.mark.NAME(...)`` element.

    Mirrors ``orchestrator.pytest_markers._marker_name`` BY CONSTRUCTION, not
    by import — see the module docstring's note on the two modules'
    deliberately opposite fail-safe polarities. Anything else — a bare
    constant, a local name, an unrelated attribute chain — yields None and is
    skipped silently, without suppressing its siblings.
    """
    if isinstance(element, ast.Call):
        element = element.func
    if not isinstance(element, ast.Attribute):
        return None
    owner = element.value
    if (
        isinstance(owner, ast.Attribute)
        and owner.attr == 'mark'
        and isinstance(owner.value, ast.Name)
        and owner.value.id == 'pytest'
    ):
        return element.attr
    return None


def _is_pytestmark_target(node: ast.expr) -> bool:
    """True iff *node* is the bare name ``pytestmark``."""
    return isinstance(node, ast.Name) and node.id == 'pytestmark'


def _pytestmark_value(statement: ast.stmt) -> ast.expr | None:
    """The value *statement* binds to ``pytestmark``, else None.

    Covers both the plain ``pytestmark = ...`` and the annotated
    ``pytestmark: list = ...`` spellings; an annotation with no value binds
    nothing. Anything other than an ``Assign``/``AnnAssign`` — including
    every ``ast.expr`` node ``ast.walk`` also yields — falls through to None.
    """
    if isinstance(statement, ast.Assign):
        if any(_is_pytestmark_target(target) for target in statement.targets):
            return statement.value
        return None
    if isinstance(statement, ast.AnnAssign) and _is_pytestmark_target(statement.target):
        return statement.value
    return None


def _names_from_marker_value(value: ast.expr) -> frozenset[str]:
    """Accepted value shapes: a bare element, or a list/tuple of elements."""
    elements = list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]
    return frozenset(
        name for name in (_marker_name(element) for element in elements) if name is not None
    )


def _applied_marker_names(source: str) -> frozenset[str]:
    """Every pytest marker name *source* APPLIES via a decorator or ``pytestmark``.

    An UPPER-BOUND sweep: walked with ``ast.walk`` (not just ``tree.body``),
    so class-level ``pytestmark`` and decorators on nested classes are
    reached — the deliberate difference from
    ``orchestrator.pytest_markers.module_level_marker_names``, which walks
    only ``tree.body`` because its contract is a module-wide LOWER bound.
    Collects from two node kinds: the ``decorator_list`` of every
    ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``, and the value of any
    ``Assign``/``AnnAssign`` binding the name ``pytestmark`` (unpacking a
    ``List``/``Tuple`` value into its elements).

    KNOWN LIMITATIONS, acknowledged rather than silently omitted:
    ``pytest.param(marks=...)`` and ``item.add_marker(...)`` are not swept —
    zero uses under ``orchestrator/tests/`` today (grep-verified); nor is a
    marker applied through an aliased pytest import
    (``import pytest as _pytest``) — two such aliases exist today
    (test_overlap_footprint.py, test_multihost_verify_integration.py) but
    both use the alias only for ``.raises``, never ``.mark``, so this is a
    real but currently-inert gap.

    LOUD ON FAILURE — the OPPOSITE of ``pytest_markers``' fail-safe polarity:
    a ``SyntaxError`` from ``ast.parse`` is left to propagate rather than
    being swallowed into an empty set, because a silent empty set here would
    make a drift guard built on top of this silently vacuous.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            for decorator in node.decorator_list:
                name = _marker_name(decorator)
                if name is not None:
                    names.add(name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            value = _pytestmark_value(node)
            if value is not None:
                names |= _names_from_marker_value(value)
    return frozenset(names)


def _registered_marker_names(ini_lines: Sequence[str]) -> frozenset[str]:
    """Normalises ``Config.getini('markers')`` entries exactly as pytest's own
    ``MarkGenerator`` does: ``line.split(':', 1)[0].split('(', 1)[0].strip()``.

    Needed because a custom entry arrives as ``'slow: marks heavyweight ...'``
    while a builtin arrives WITH a call signature, e.g. ``'parametrize(argnames,
    argvalues): call a test ...'`` — the second split strips that signature
    too. A bare name with neither shape is returned unchanged (stripped).
    """
    return frozenset(line.split(':', 1)[0].split('(', 1)[0].strip() for line in ini_lines)


def _unregistered_markers(tests_dir: Path, registered: frozenset[str]) -> dict[str, set[str]]:
    """``{marker_name: {relative file paths}}`` for every marker applied under
    *tests_dir* that is absent from *registered*.

    Sweeps every ``*.py`` file from DISK (``sorted(tests_dir.rglob('*.py'))``
    — sorted for deterministic failure messages under xdist), never pytest's
    own collection, so the result does not depend on what any one invocation
    happened to collect (see the module docstring's rejected-alternative
    note). Reported paths are relative to *tests_dir*.

    A file that fails to parse or read (``SyntaxError``/``OSError``) is
    re-raised as an ``AssertionError`` naming the offending file — never
    swallowed, never silently skipped, the same loud-on-failure polarity as
    ``_applied_marker_names`` itself. Returns ``{}`` when every applied
    marker is registered.
    """
    unregistered: dict[str, set[str]] = {}
    for path in sorted(tests_dir.rglob('*.py')):
        relative = str(path.relative_to(tests_dir))
        try:
            applied = _applied_marker_names(path.read_text())
        except (SyntaxError, OSError) as exc:
            raise AssertionError(
                f'{relative} could not be read/parsed while sweeping '
                f'{tests_dir} for applied pytest markers: {exc!r}. Fix the '
                f'file — a silently-skipped file would make this guard '
                f'vacuous for it.'
            ) from exc
        for name in applied - registered:
            unregistered.setdefault(name, set()).add(relative)
    return unregistered


class TestAppliedMarkerNames:
    """``_applied_marker_names(source: str) -> frozenset[str]``.

    Every marker name a module APPLIES via a decorator (function or class) or
    a ``pytestmark`` assignment (module- or class-level). Pure: parses a
    source string, touches no filesystem. Unlike
    ``orchestrator.pytest_markers.module_level_marker_names`` — a LOWER-BOUND
    walk of ``tree.body`` only, by design, because it feeds a widen-only
    decision — this is an UPPER-BOUND sweep that also reaches class-level
    ``pytestmark`` and decorators on nested classes, because for a drift
    guard under-sweeping is the unsafe direction: a missed marker is missed
    drift.
    """

    # -- application forms that occur for real under orchestrator/tests/ ------

    def test_bare_function_decorator(self):
        """``@pytest.mark.slow`` — mirrors
        test_verify_plan_integration.py::test_scoped_pytest_producer_and_runner_agree_on_scope.
        """
        source = (
            'import pytest\n\n\n'
            '@pytest.mark.slow\n'
            'def test_scoped_pytest_producer_and_runner_agree_on_scope():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'slow'})

    def test_call_form_function_decorator(self):
        """``@pytest.mark.timeout(120)`` — mirrors
        test_warm_lane_bash_bucket_placement.py:228."""
        source = (
            'import pytest\n\n\n'
            '@pytest.mark.timeout(120)\n'
            'def test_the_configured_lane_command_actually_collects_the_bucket():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'timeout'})

    def test_class_decorator(self):
        """A marker decorator applied directly to a test class."""
        source = (
            'import pytest\n\n\n'
            "@pytest.mark.usefixtures('x')\n"
            'class TestThing:\n'
            '    def test_it(self):\n'
            '        pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'usefixtures'})

    def test_module_level_pytestmark_bare(self):
        """``pytestmark = pytest.mark.asyncio`` — mirrors
        test_provenance_gate_integration.py:46."""
        source = 'import pytest\n\npytestmark = pytest.mark.asyncio\n'
        assert _applied_marker_names(source) == frozenset({'asyncio'})

    def test_module_level_pytestmark_list(self):
        """Verbatim shape of test_warm_lane_bash_suite.py:240-244."""
        source = (
            'import pytest\n\n'
            'pytestmark = [\n'
            '    pytest.mark.warm_lane_bash,\n'
            '    pytest.mark.timeout(120),\n'
            "    pytest.mark.xdist_group('warm_lane_bash'),\n"
            ']\n'
        )
        assert _applied_marker_names(source) == frozenset({
            'warm_lane_bash', 'timeout', 'xdist_group',
        })

    def test_module_level_pytestmark_tuple(self):
        source = 'import pytest\n\npytestmark = (pytest.mark.asyncio, pytest.mark.slow)\n'
        assert _applied_marker_names(source) == frozenset({'asyncio', 'slow'})

    def test_class_level_pytestmark(self):
        """``pytestmark = pytest.mark.asyncio`` bound INSIDE a class body —
        mirrors test_stranded_verified_green.py:267
        (``class TestDetectVerifiedGreen: pytestmark = pytest.mark.asyncio``).
        This is precisely the form ``module_level_marker_names`` deliberately
        excludes (``tree.body``-only); this extractor must NOT exclude it.
        """
        source = (
            'import pytest\n\n\n'
            'class TestDetectVerifiedGreen:\n'
            '    pytestmark = pytest.mark.asyncio\n\n'
            '    async def test_positive_match(self):\n'
            '        pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'asyncio'})

    def test_multiple_stacked_decorators(self):
        source = (
            'import pytest\n\n\n'
            '@pytest.mark.slow\n'
            '@pytest.mark.timeout(30)\n'
            'def test_thing():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset({'slow', 'timeout'})

    # -- negative cases: must yield NO names -----------------------------------

    def test_marker_name_inside_a_string_literal_is_ignored(self):
        """The exact trap AST protects against. test_pytest_marker_deselection.py
        embeds real marker shapes as SYNTHETIC SOURCE STRING CONSTANTS at its
        lines 376/481/661 (e.g. ``_MARKED_SOURCE = 'import pytest\\npytestmark
        = pytest.mark.warm_lane_bash\\n...'``). A grep sweep would misread
        that line as an applied marker; ast sees an ``ast.Constant`` string
        and correctly ignores it — this is why AST is used instead of grep.
        """
        source = (
            "_MARKED_SOURCE = 'import pytest\\npytestmark = "
            "pytest.mark.warm_lane_bash\\n\\n\\ndef test_x():\\n    pass\\n'\n"
        )
        assert _applied_marker_names(source) == frozenset()

    def test_unrelated_attribute_chain_is_ignored(self):
        source = (
            'import other\n\n\n'
            '@other.mark.thing\n'
            'def test_thing():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset()

    def test_functools_wraps_decorator_is_ignored(self):
        source = (
            'import functools\n\n\n'
            'def _identity(f):\n'
            '    return f\n\n\n'
            '@functools.wraps(_identity)\n'
            'def test_thing():\n'
            '    pass\n'
        )
        assert _applied_marker_names(source) == frozenset()

    def test_pytestmark_named_local_with_non_marker_value_is_ignored(self):
        """A ``pytestmark``-named local whose value is not a marker access."""
        source = (
            'def f():\n'
            "    pytestmark = 'not a real marker'\n"
            '    return pytestmark\n'
        )
        assert _applied_marker_names(source) == frozenset()

    # -- loud failure: the OPPOSITE of pytest_markers.py's swallow policy -----

    def test_syntax_error_raises_not_swallowed(self):
        """Opposite of ``orchestrator.pytest_markers``' fail-safe polarity: a
        silent empty set here would be a silently-vacuous drift guard, so an
        unparseable file must RAISE rather than resolve to "no markers".
        """
        with pytest.raises(SyntaxError):
            _applied_marker_names('def f(:\n    pass\n')


class TestUnregisteredMarkers:
    """``_registered_marker_names`` and ``_unregistered_markers`` — the
    drift-diff half, unit-tested against synthetic ini lines and synthetic
    ``tmp_path`` trees. This is the efficacy proof that the guard can
    actually FAIL: the real tree (wired in ``TestMarkerRegistrationDrift``
    below) is green on arrival by construction, since every marker in use
    today is already registered.
    """

    # -- _registered_marker_names: pytest's own MarkGenerator normalisation ---
    # Real shapes taken verbatim (as a prefix) from a live
    # ``pytestconfig.getini('markers')`` probe of this project.

    def test_colon_form_strips_the_description(self):
        line = (
            'slow: marks heavyweight tests that shell out to real subprocesses '
            '(uv run pytest, plan-tools MCP servers) rather than mocking them, '
            'so they run materially slower than the rest of the suite.'
        )
        assert _registered_marker_names([line]) == frozenset({'slow'})

    def test_paren_form_strips_the_call_signature_and_the_description(self):
        """Builtins arrive with a call signature, e.g. ``parametrize(argnames,
        argvalues): call a test ...`` — the second split exists for this."""
        line = (
            'parametrize(argnames, argvalues): call a test function multiple '
            'times passing in different arguments in turn.'
        )
        assert _registered_marker_names([line]) == frozenset({'parametrize'})

    def test_bare_name_with_no_description(self):
        assert _registered_marker_names(['anyio']) == frozenset({'anyio'})

    def test_whitespace_padded_input_is_stripped(self):
        assert _registered_marker_names(['  slow  ']) == frozenset({'slow'})

    def test_multiple_lines_all_normalised(self):
        assert _registered_marker_names(['slow: x', 'parametrize(a): y', 'anyio']) == frozenset({
            'slow', 'parametrize', 'anyio',
        })

    # -- _unregistered_markers: the sweep + diff, over a synthetic tree -------

    def test_a_clean_tree_yields_nothing(self, tmp_path: Path):
        (tmp_path / 'test_ok.py').write_text(
            'import pytest\n\n\n@pytest.mark.slow\ndef test_x():\n    pass\n'
        )
        assert _unregistered_markers(tmp_path, frozenset({'slow'})) == {}

    def test_a_planted_typo_is_reported_with_a_relative_path(self, tmp_path: Path):
        """The task description's own typo example."""
        (tmp_path / 'test_typo.py').write_text(
            'import pytest\n\n\n@pytest.mark.slwo\ndef test_x():\n    pass\n'
        )
        assert _unregistered_markers(tmp_path, frozenset({'slow'})) == {
            'slwo': {'test_typo.py'},
        }

    def test_the_typo_via_class_level_pytestmark_is_also_reported(self, tmp_path: Path):
        (tmp_path / 'test_typo.py').write_text(
            'import pytest\n\n\nclass TestX:\n    pytestmark = pytest.mark.slwo\n'
        )
        assert _unregistered_markers(tmp_path, frozenset({'slow'})) == {
            'slwo': {'test_typo.py'},
        }

    def test_the_same_typo_in_two_files_is_one_key_with_both_paths(self, tmp_path: Path):
        for name in ('test_a.py', 'test_b.py'):
            (tmp_path / name).write_text(
                'import pytest\n\n\n@pytest.mark.slwo\ndef test_x():\n    pass\n'
            )
        assert _unregistered_markers(tmp_path, frozenset({'slow'})) == {
            'slwo': {'test_a.py', 'test_b.py'},
        }

    def test_a_syntax_error_fails_loudly_naming_the_file(self, tmp_path: Path):
        """Never swallowed, never silently skipped — the opposite of
        ``orchestrator.pytest_markers``' fail-safe polarity, same as
        ``_applied_marker_names`` itself."""
        (tmp_path / 'test_broken.py').write_text('def f(:\n    pass\n')
        with pytest.raises(AssertionError, match='test_broken.py'):
            _unregistered_markers(tmp_path, frozenset({'slow'}))

    def test_recurses_into_nested_subdirs_and_ignores_non_py_siblings(self, tmp_path: Path):
        nested = tmp_path / 'nested'
        nested.mkdir()
        (nested / 'test_nested.py').write_text(
            'import pytest\n\n\n@pytest.mark.slwo\ndef test_x():\n    pass\n'
        )
        (tmp_path / 'not_a_test.txt').write_text('pytest.mark.slwo')
        assert _unregistered_markers(tmp_path, frozenset({'slow'})) == {
            'slwo': {'nested/test_nested.py'},
        }


# ---------------------------------------------------------------------------
# The wired guard — REAL tests/ tree vs REAL pytest config
# ---------------------------------------------------------------------------

# Resolved from THIS FILE, never the process CWD: merge-verify runs pytest
# from the orchestrator/ cwd (orchestrator/orchestrator.yaml) while a plain
# `pytest orchestrator/tests` runs from the repo root, and this must resolve
# identically under both. Same idiom as test_warm_lane_bash_bucket_placement.py.
TESTS_DIR = Path(__file__).resolve().parent
ORCH_PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'

#: A floor, not an equality — 494 measured at authorship time — so adding
#: test files over time never breaks this guard.
_MIN_EXPECTED_TEST_FILES = 400

#: One witness marker per extraction path this module supports, so a broken
#: extractor that returns an empty/partial set cannot make the drift guard
#: below pass vacuously.
_WITNESS_MARKER_NAMES = frozenset({
    'slow',               # function decorator
    'warm_lane_bash',     # module-level pytestmark, list form
    'real_psi_reader',    # function decorator; registered via conftest.py:403
    'asyncio',            # module-level AND class-level pytestmark
    'timeout',
    'xdist_group',
    'parametrize',
    'exercise_merge_verify',
})


class TestMarkerRegistrationDrift:
    """The deliverable guard, wired to the REAL tests/ tree and REAL config.

    No test here currently fails — every marker applied under
    orchestrator/tests/ today is already registered (task 3532's premise) —
    so this class is green on arrival. Its value is entirely in failing on
    the NEXT typo; ``TestUnregisteredMarkers`` above already carries the
    burden of proving the underlying mechanism can actually fail.
    """

    def test_every_marker_applied_under_tests_is_registered(self, pytestconfig):
        """THE guard."""
        registered = _registered_marker_names(pytestconfig.getini('markers'))
        unregistered = _unregistered_markers(TESTS_DIR, registered)
        assert unregistered == {}, (
            f'The following pytest markers are applied under {TESTS_DIR} but '
            f'are not registered: {unregistered!r}. A name that looks like a '
            f'typo of an existing marker probably is one. Register it either '
            f"in orchestrator/pyproject.toml's [tool.pytest.ini_options] "
            f"markers list, or via config.addinivalue_line('markers', ...) "
            f'in tests/conftest.py.'
        )

    def test_the_sweep_is_not_vacuous(self):
        """A broken extractor returning ``frozenset()`` would make the guard
        above pass forever. In the spirit of
        test_warm_lane_bash_bucket_placement.py's live ``--collect-only``
        probe, pin that the sweep actually visits the real tree and actually
        finds markers spanning every extraction path this module supports.
        """
        files = sorted(TESTS_DIR.rglob('*.py'))
        assert len(files) >= _MIN_EXPECTED_TEST_FILES, (
            f'only found {len(files)} .py files under {TESTS_DIR}, expected '
            f'at least {_MIN_EXPECTED_TEST_FILES}. The sweep may be looking '
            f'in the wrong place.'
        )
        applied: set[str] = set()
        for path in files:
            applied |= _applied_marker_names(path.read_text())
        missing_witnesses = _WITNESS_MARKER_NAMES - applied
        assert not missing_witnesses, (
            f'expected witness markers {sorted(missing_witnesses)} were not '
            f'found while sweeping {TESTS_DIR} — the extractor may be broken.'
        )

    def test_conftest_registered_markers_count_as_registered(self, pytestconfig):
        """Pins the second critical finding: "registered" means the
        EFFECTIVE ``getini('markers')`` set, not a bare ``tomllib`` read of
        pyproject.toml. ``real_psi_reader`` is registered dynamically at
        tests/conftest.py:403 (task 2418), deliberately outside
        pyproject.toml's locked scope. A future reader "simplifying" this
        guard into a bare TOML comparison would produce a false failure on
        day one.
        """
        with ORCH_PYPROJECT.open('rb') as handle:
            toml_data = tomllib.load(handle)
        toml_markers = _registered_marker_names(
            toml_data['tool']['pytest']['ini_options'].get('markers', [])
        )
        assert 'real_psi_reader' not in toml_markers, (
            "real_psi_reader unexpectedly appears in pyproject.toml's markers "
            "list — if it was added there, this test (and its rationale) is "
            "stale and should be revisited, not silenced."
        )
        effective = _registered_marker_names(pytestconfig.getini('markers'))
        assert 'real_psi_reader' in effective, (
            'real_psi_reader is absent from the effective getini(markers) '
            'set — tests/conftest.py may have stopped registering it '
            '(task 2418).'
        )
