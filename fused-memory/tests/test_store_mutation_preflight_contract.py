"""Contract tests for `_store_mutation_preflight_contract`, the shared
store-mutation-preflight test scaffolding (task 4480).

Pairs with `_store_mutation_preflight_contract.py` exactly as
`test_ast_guard.py` pairs with `_ast_guard.py`: the helper is
underscore-prefixed so pytest does not collect it, and is imported bare
(`from _store_mutation_preflight_contract import ...`) because
`tests/conftest.py` inserts this directory at the front of `sys.path`.

Two things live here:

1. Unit coverage of the four exports as REAL behaviour — the fixture factory
   actually sets and restores the attribute, `deny` actually installs a
   raiser, `fail_closed_records` actually filters. None of these assert on
   docstring prose: `neutralise_fixture` still composes its `__doc__` from the
   shared rationale plus the per-suite note (which is what makes
   `pytest --fixtures -v` useful -- the `-v` is required, since pytest hides
   underscore-prefixed fixtures), but that composition has no runtime effect and
   so is read inline rather than pinned by a test.

2. The whole-tree AST drift guards that keep the extraction from silently
   re-diverging once it has landed. Extraction alone is a one-time dedupe; the
   guards are what stop a suite copied from an older template reintroducing a
   second source of truth for the same contract.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import types

import pytest
from _ast_guard import parse_python_module
from _store_mutation_preflight_contract import (
    FAIL_CLOSED_MARKERS,
    SENTINEL,
    deny,
    fail_closed_records,
    neutralise_fixture,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _underlying(fixture):
    """The plain function wrapped by a `pytest.fixture(...)`-decorated object.

    pytest >= 8.4 returns a `FixtureFunctionDefinition` holding the original
    callable on `_fixture_function`; older pytest returned the function itself
    with a `_pytestfixturefunction` marker attached. Tolerating both keeps this
    test from being a pytest-version pin.
    """
    return getattr(fixture, '_fixture_function', fixture)


def _is_autouse(fixture) -> bool:
    """Whether pytest will register *fixture* as autouse."""
    marker = getattr(fixture, '_fixture_function_marker', None)
    if marker is None:  # pytest < 8.4
        marker = getattr(fixture, '_pytestfixturefunction', None)
    assert marker is not None, f'{fixture!r} is not a pytest fixture at all'
    return marker.autouse


def _stub_script_module(*, with_guard: bool = True) -> types.ModuleType:
    """A stand-in for one of the `scripts/` modules under test.

    Carries its own `StoreMutationUnavailable` **subclass of `RuntimeError`**,
    matching the real
    `fused-memory/src/fused_memory/utils/store_mutation_preflight.py::StoreMutationUnavailable`.
    The real subclassing matters: the guarded scripts wrap per-record work in
    `except Exception` handlers, which is precisely why the guard must run
    once for the whole run rather than per record — a stub raising a bare
    `Exception` would not exercise that.
    """
    mod = types.ModuleType('_stub_script_module')

    class StoreMutationUnavailable(RuntimeError):
        pass

    # Populate via __dict__.update: pyright rejects attribute assignment on a
    # bare ModuleType instance (reportAttributeAccessIssue), but a module
    # namespace is a plain dict[str, Any], so this is runtime-equivalent and
    # type-clean -- the same spelling escalation/tests/conftest.py uses for its
    # module stubs. `monkeypatch.setattr(mod, ...)` still sees these, since
    # attribute access on a module IS its `__dict__`.
    mod.__dict__.update(StoreMutationUnavailable=StoreMutationUnavailable)
    if with_guard:
        def _real_guard(*, operation: str) -> None:
            raise AssertionError(f'the real guard ran for {operation!r}')

        mod.__dict__.update(assert_store_mutation_allowed=_real_guard)
    return mod


# ---------------------------------------------------------------------------
# neutralise_fixture
# ---------------------------------------------------------------------------


class TestNeutraliseFixture:
    """`neutralise_fixture(mod, *, note)` builds the autouse fixture that keeps
    a MOCK-unit suite independent of the real `~/.mem0`."""

    def test_returns_an_autouse_fixture(self) -> None:
        fx = neutralise_fixture(_stub_script_module(), note='a note')
        assert _is_autouse(fx) is True

    def test_sets_a_noop_guard_and_restores_it(self) -> None:
        """The whole point: while the fixture is active the guard is a no-op,
        and afterwards the module is exactly as it was."""
        mod = _stub_script_module()
        original = mod.assert_store_mutation_allowed
        fx = neutralise_fixture(mod, note='a note')

        mp = pytest.MonkeyPatch()
        try:
            _underlying(fx)(mp)
            # Neutralised: calling it neither raises nor does anything.
            assert mod.assert_store_mutation_allowed(operation='anything') is None
            assert mod.assert_store_mutation_allowed is not original
        finally:
            mp.undo()

        assert mod.assert_store_mutation_allowed is original

    def test_note_is_required(self) -> None:
        """The forcing function: a conversion that forgets to carry its suite's
        per-script rationale forward is a TypeError, not a silent prose loss."""
        with pytest.raises(TypeError):
            neutralise_fixture(_stub_script_module())  # type: ignore[call-arg]

    def test_breaks_loudly_when_the_script_has_no_guard(self) -> None:
        """Deliberately NOT `raising=False`.

        Every one of the docstrings this factory replaces calls this out: if the
        guard is ever removed from the script, the fixture must break loudly
        rather than silently no-op and leave the suite asserting nothing.
        """
        mod = _stub_script_module(with_guard=False)
        fx = neutralise_fixture(mod, note='a note')

        mp = pytest.MonkeyPatch()
        try:
            with pytest.raises(AttributeError):
                _underlying(fx)(mp)
        finally:
            mp.undo()


# ---------------------------------------------------------------------------
# deny
# ---------------------------------------------------------------------------


class TestDeny:
    """`deny(mod, monkeypatch)` rigs the preflight to refuse, as it would
    inside an agent sandbox."""

    def test_installs_a_sentinel_raiser(self) -> None:
        mod = _stub_script_module()
        mp = pytest.MonkeyPatch()
        try:
            deny(mod, mp)
            with pytest.raises(mod.StoreMutationUnavailable) as excinfo:
                mod.assert_store_mutation_allowed(operation='sweep')
        finally:
            mp.undo()

        assert str(excinfo.value) == SENTINEL

    def test_raises_the_modules_own_exception_type(self) -> None:
        """Parameterised on *mod*, so the raiser is always the type the script
        under test will actually catch — and it is a RuntimeError subclass, the
        shape the scripts' per-record `except Exception` handlers would swallow."""
        mod = _stub_script_module()
        mp = pytest.MonkeyPatch()
        try:
            deny(mod, mp)
            with pytest.raises(mod.StoreMutationUnavailable):
                mod.assert_store_mutation_allowed(operation='sweep')
            assert issubclass(mod.StoreMutationUnavailable, RuntimeError)
        finally:
            mp.undo()

    def test_accepts_arbitrary_positional_and_keyword_args(self) -> None:
        """Call sites pass keywords (`operation=`), but the raiser must not
        constrain the signature it stands in for."""
        mod = _stub_script_module()
        mp = pytest.MonkeyPatch()
        try:
            deny(mod, mp)
            with pytest.raises(mod.StoreMutationUnavailable):
                mod.assert_store_mutation_allowed()
            with pytest.raises(mod.StoreMutationUnavailable):
                mod.assert_store_mutation_allowed('positional', 2, kw='x', other=None)
        finally:
            mp.undo()

    def test_is_undone_with_the_monkeypatch(self) -> None:
        mod = _stub_script_module()
        original = mod.assert_store_mutation_allowed
        mp = pytest.MonkeyPatch()
        deny(mod, mp)
        mp.undo()
        assert mod.assert_store_mutation_allowed is original


# ---------------------------------------------------------------------------
# fail_closed_records
# ---------------------------------------------------------------------------


_LOGGER = 'contract_probe_logger'
_OTHER_LOGGER = 'contract_probe_other_logger'

# Probe messages are COMPOSED from the exported markers rather than spelled
# out, for two reasons. First, `TestNoInlinedFailClosedMarker` below forbids
# spelling the fail-closed marker as a literal anywhere under `tests/` --
# INCLUDING this module, since that guard's sweep is exemption-free -- so
# composing is the only way to build a probe that carries it. Second, a marker
# rename then cannot leave these probes silently testing the wrong string.
#
# Composed by ITERATION, never by unpacking: `a, b = FAIL_CLOSED_MARKERS` would
# make a third marker an import-time `ValueError`, i.e. a collection error for
# this whole module, which is the worst available failure mode for what should
# be a one-line change in the helper. Iterating instead means a third marker
# widens the positive probe and mints its own exclusion case automatically.


def _message_carrying(markers) -> str:
    """A plausible guard-site ERROR message carrying exactly *markers*."""
    return f"sweep refused -- {'; '.join(markers)} -- see above"


_ALL_MARKERS = _message_carrying(FAIL_CLOSED_MARKERS)


def _message_omitting(marker: str) -> str:
    """The same message with exactly one clause dropped.

    The point of dropping only ONE is that the record still looks like a real
    diagnosis: the helper's filter must reject it on the strength of the single
    missing clause, not because the message is obviously junk. This is the
    generated form of the old hand-written pair (`'... : boom'`, `'fatal error
    during sweep: ...'`), which covered the same two cases at a fixed arity.
    """
    return _message_carrying([m for m in FAIL_CLOSED_MARKERS if m != marker])


class TestFailClosedRecords:
    """`fail_closed_records(caplog, logger_name)` isolates the guard site's OWN
    diagnosis from every other record in the log.

    Each exclusion is asserted independently, so a filter that silently drops
    one of its clauses fails a named test rather than passing on the strength
    of the others. The per-marker cases are generated from
    `FAIL_CLOSED_MARKERS`, so that stays true at any arity.
    """

    def test_returns_the_matching_record(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).error(_ALL_MARKERS)
        assert [r.getMessage() for r in fail_closed_records(caplog, _LOGGER)] == [_ALL_MARKERS]

    def test_excludes_another_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_OTHER_LOGGER).error(_ALL_MARKERS)
        assert fail_closed_records(caplog, _LOGGER) == []

    def test_excludes_below_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).warning(_ALL_MARKERS)
        assert fail_closed_records(caplog, _LOGGER) == []

    @pytest.mark.parametrize('omitted', FAIL_CLOSED_MARKERS)
    def test_excludes_a_record_missing_any_one_marker(
        self, omitted: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each marker is required INDEPENDENTLY: a record carrying every other
        clause is still not this guard site's diagnosis.

        Parametrised over `FAIL_CLOSED_MARKERS` rather than written out once
        per marker, so a filter that silently drops a clause fails a case pytest
        has NAMED for the dropped clause (it ids each case by the marker text),
        and so a third marker arrives with its own exclusion case generated.

        The marker texts are deliberately not quoted in this docstring: drift
        guard #3 below sweeps docstrings too, because a docstring IS an
        `ast.Constant`. (Measured: quoting one here turned that guard red.)
        """
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).error(_message_omitting(omitted))
        assert fail_closed_records(caplog, _LOGGER) == []

    def test_includes_critical(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pins the deliberate `levelno >= logging.ERROR` superset.

        The 12 spellings this replaces were 11x `levelname == 'ERROR'` and 1x
        `levelno >= logging.ERROR`; the superset was chosen because every call
        site is a positive assertion, so a broader predicate can only add
        matches and can never turn a green assertion red.
        """
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).critical(_ALL_MARKERS)
        assert [r.getMessage() for r in fail_closed_records(caplog, _LOGGER)] == [_ALL_MARKERS]

    def test_filters_a_mixed_log_down_to_the_guard_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The real shape: `main`'s own generic ERROR shares this logger and
        level, so only the markers can tell the two records apart."""
        with caplog.at_level(logging.DEBUG):
            log = logging.getLogger(_LOGGER)
            log.info(_ALL_MARKERS)
            log.error('fatal error during sweep')
            logging.getLogger(_OTHER_LOGGER).error(_ALL_MARKERS)
            log.error(_ALL_MARKERS)
        assert [r.getMessage() for r in fail_closed_records(caplog, _LOGGER)] == [_ALL_MARKERS]


# ---------------------------------------------------------------------------
# Drift guard #1 — no test module may hand-roll the neutralising fixture
# ---------------------------------------------------------------------------

TESTS_DIR = pathlib.Path(__file__).parent

# Real-time budget for the three whole-tree sweeps below, opting up from the
# project default of 60s (`fused-memory/pyproject.toml`).
#
# REQUIRED, not defensive. That default runs under `timeout_method = "thread"`,
# whose handler ends in `os._exit(1)` -- so a test that overruns it does not
# fail, it KILLS ITS ENTIRE xdist WORKER, and a worker lost near end-of-suite
# leaves the controller waiting forever for a completion signal that never
# arrives. pyproject spells this out and states the rule: "Tests with a
# real-time budget approaching 60s MUST set @pytest.mark.timeout(N) to opt up
# -- otherwise a single slow run takes down the whole suite."
#
# These three qualify since the sweep went recursive: each parses/walks 328
# modules (~1.46M AST nodes), measured at 20-49s per sweep on a machine at load
# average 290. That is 82% of the default cap, i.e. inside its noise band --
# and the failure it would cause is a whole-suite deadlock, not a red test.
# Sized at 5x the observed worst case: this is a backstop against a genuine
# hang, NOT a performance assertion, so it must not trip on load alone.
_SWEEP_TIMEOUT = 300

# The name every one of the 14 suites gave its hand-rolled autouse fixture
# before the extraction. After it, the name exists nowhere in `tests/`: suites
# call `neutralise_fixture(...)` and bind the result to `_neutralise`.
#
# Kept as a NAMED tripwire, but it is no longer the whole guard: a name-only
# check is permanently green against the only thing it can see, since the one
# spelling it knows now appears nowhere. A contributor hand-rolling the fixture
# would almost certainly call it something else (`_neutralise_preflight`,
# `_no_preflight`), and the name check would wave it through. So the guard also
# keys on the SHAPE — see `_neutralising_autouse_fixture` below.
_HAND_ROLLED_FIXTURE_NAME = '_neutralise_store_mutation_preflight'


def _is_autouse_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether *node* is decorated `@pytest.fixture(autouse=True)`."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        # Match the callee node directly rather than `ast.unparse`-ing it:
        # this runs on every decorator of every function in the tree, and
        # unparse is the expensive spelling. Covers `@pytest.fixture(...)`
        # (Attribute) and a bare `@fixture(...)` (Name).
        callee = decorator.func
        if isinstance(callee, ast.Attribute):
            callee_name = callee.attr
        elif isinstance(callee, ast.Name):
            callee_name = callee.id
        else:
            continue
        if callee_name != 'fixture':
            continue
        for keyword in decorator.keywords:
            if (
                keyword.arg == 'autouse'
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                return True
    return False


def _neutralises_the_guard(stmt: ast.stmt) -> bool:
    """Whether *stmt* is a `monkeypatch.setattr(<mod>,
    'assert_store_mutation_allowed', ...)` call."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    call = stmt.value
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'setattr'):
        return False
    return any(
        isinstance(arg, ast.Constant) and arg.value == 'assert_store_mutation_allowed'
        for arg in call.args
    )


def _neutralising_autouse_fixture(node: ast.AST) -> bool:
    """Whether *node* is a hand-rolled neutralising fixture under ANY name.

    The shape, not the name: an autouse fixture that rebinds
    `assert_store_mutation_allowed`. Both halves are load-bearing.

    AUTOUSE is what keeps this off the ~70 legitimate per-scenario
    `monkeypatch.setattr(..., 'assert_store_mutation_allowed', ...)` sites that
    guard #2's docstring enumerates — the `calls.append(kw)` recorders, the
    `order.append('preflight')` sequencers, and in particular the pass-through
    `lambda **_kw: None` re-rigs inside "unchanged when the preflight passes"
    tests, which are byte-identical to the fixture's own body. Those live in
    test bodies, never in an autouse fixture, and they must NOT be shared.
    Rebinding THE GUARD is what distinguishes it from the other 29 autouse
    fixtures in this tree.

    Verified false-positive-free when written (2026-09-04): 30 autouse fixtures
    across `tests/`, 0 of which rebind the guard, because every suite that
    needs one now calls `neutralise_fixture`.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if not _is_autouse_fixture(node):
        return False
    return any(_neutralises_the_guard(stmt) for stmt in node.body)


def _test_modules() -> list[pathlib.Path]:
    """Every collected test module in this TREE, in a stable order.

    RECURSIVE (`rglob`), not top-level. The guarded suites all happen to live
    at the top level today, but `tests/reconciliation/`, `tests/server/` and
    `tests/middleware/` hold collected test modules too, and a guarded suite
    added under one of them would otherwise be free to reintroduce exactly the
    second source of truth these guards exist to forbid — silently, since
    nothing else would notice. The guards are cheap insurance only if their
    scope matches pytest's.

    Deliberately globs `test_*.py` and NOT `*.py`: the helper home
    `_store_mutation_preflight_contract.py` is underscore-prefixed (so pytest
    does not collect it) and is therefore already excluded. Do not "fix" this
    glob to `*.py` — that would sweep the helper itself and make every guard
    below unsatisfiable.

    Cost: 328 modules and ~1.46M AST nodes, against 257 modules / ~1.25M nodes
    top-level. Wall clock is dominated by the parse and swings widely with
    machine load (6-23s measured for the identical work on 2026-09-04), so no
    figure is pinned here. What is structural: the parse is paid ONCE per
    session, not once per guard — `parse_python_module` is `functools.cache`d on
    the path, so the three guards below share it, and each then pays only its
    own `ast.walk` (~2-5s).
    """
    return sorted(
        path
        for path in TESTS_DIR.rglob('test_*.py')
        if '__pycache__' not in path.parts
    )


class TestNoHandRolledNeutraliseFixture:
    """Whole-tree AST drift guard: no test module may hand-roll the
    neutralising fixture — under the historical name, or under any other.

    Extraction alone is a one-time dedupe — the moment someone copies an older
    suite as a template, the two-sources-of-truth problem this helper exists to
    close comes straight back. This is what makes the collapse durable.

    Catches TWO shapes, because the name alone catches nothing: a definition of
    `_neutralise_store_mutation_preflight` (the pre-extraction spelling, now
    absent everywhere — a tripwire for a literal revert), and an autouse fixture
    under ANY name whose body rebinds `assert_store_mutation_allowed`, which is
    what a rename would actually look like. The second is the one with teeth.

    Asserts over PARSED source (`_ast_guard.parse_python_module`), so prose in a
    docstring that merely names the old fixture cannot trip it.
    """

    @pytest.mark.timeout(_SWEEP_TIMEOUT)
    def test_no_test_module_defines_its_own_neutralise_fixture(self) -> None:
        offenders: list[str] = []
        for path in _test_modules():
            tree = parse_python_module(path)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name == _HAND_ROLLED_FIXTURE_NAME:
                    offenders.append(f'{path.relative_to(TESTS_DIR)}:{node.lineno}:{node.name} (name)')
                elif _neutralising_autouse_fixture(node):
                    offenders.append(f'{path.relative_to(TESTS_DIR)}:{node.lineno}:{node.name} (shape)')

        assert not offenders, (
            f'{len(offenders)} hand-rolled neutralising fixture(s) found — an '
            'autouse fixture rebinding `assert_store_mutation_allowed`, or a '
            f'definition of `{_HAND_ROLLED_FIXTURE_NAME}`. Route each through '
            '`_store_mutation_preflight_contract.neutralise_fixture(_mod, '
            "note='<this suite's seam and mock substrate>')` instead:\n  "
            + '\n  '.join(offenders)
        )


# ---------------------------------------------------------------------------
# Drift guard #2 — no test module may hand-roll a StoreMutationUnavailable raiser
# ---------------------------------------------------------------------------


class TestNoHandRolledDenyRaiser:
    """Whole-tree AST drift guard: no test module may `raise
    StoreMutationUnavailable(...)` itself. Rigging the preflight to refuse goes
    through `_store_mutation_preflight_contract.deny`, so the sentinel is
    spelled in exactly one place.

    KEYED ON THE `raise` SHAPE, NOT ON `monkeypatch.setattr`. There are ~70
    `monkeypatch.setattr(..., 'assert_store_mutation_allowed', ...)` sites
    across these suites, and the large majority are per-scenario rigs that must
    NOT be shared: `lambda **kw: calls.append(kw)` operation-name recorders,
    `order.append('preflight')` sequencing probes, and pass-through
    `lambda **_kw: None` re-rigs inside "unchanged when the preflight passes"
    tests. A setattr-keyed guard would condemn every one of those and need a
    large allowlist, which is self-defeating for a drift guard. The `raise`
    shape isolates exactly the deny helper.

    False-positive-free by construction: `pytest.raises(_mod.
    StoreMutationUnavailable)` parses as an `ast.Call`, never an `ast.Raise`,
    so the assertion sites that name the exception are untouched.
    """

    @pytest.mark.timeout(_SWEEP_TIMEOUT)
    def test_no_test_module_raises_store_mutation_unavailable(self) -> None:
        offenders: list[str] = []
        for path in _test_modules():
            tree = parse_python_module(path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Raise)
                    and node.exc is not None
                    and 'StoreMutationUnavailable' in ast.unparse(node.exc)
                ):
                    offenders.append(f'{path.relative_to(TESTS_DIR)}:{node.lineno}')

        assert not offenders, (
            f'{len(offenders)} hand-rolled `StoreMutationUnavailable` raiser(s) '
            'found. Rig the refusal through '
            '`_store_mutation_preflight_contract.deny(<script module>, '
            'monkeypatch)` instead, so the sentinel is spelled once:\n  '
            + '\n  '.join(offenders)
        )


# ---------------------------------------------------------------------------
# Drift guard #3 — the fail-closed marker literal lives only in the helper
# ---------------------------------------------------------------------------

# Guarding the marker LITERAL rather than a `_fail_closed_records` function
# name is the stronger check: it also catches a re-divergence that inlines the
# filter into a test body, or renames the helper.
#
# Guards the FAIL-CLOSED marker only -- `FAIL_CLOSED_MARKERS[0]`, not the whole
# tuple -- and that is deliberate, not an arity oversight. Measured 2026-09-04
# across all 328 test modules: the fail-closed marker appears as an
# `ast.Constant` in 0 of them, while the remedy noun (`FAIL_CLOSED_MARKERS[1]`)
# appears in 48, because it is ordinary English that unrelated suites have
# every right to mention. Sweeping the whole tuple would buy 48 false positives
# and an allowlist, which is self-defeating for a drift guard -- the same
# reasoning guard #2 gives for keying on the `raise` shape rather than setattr.
#
# The narrower sweep is still sufficient for the property that matters: the two
# markers travel together in one tuple, and a suite inlining the diagnosis has
# to spell the distinctive half in order to match a real record.
_GUARDED_MARKER = FAIL_CLOSED_MARKERS[0]


class TestNoInlinedFailClosedMarker:
    """Whole-tree AST drift guard: the fail-closed marker may not be spelled in
    any test module. Filtering a guard site's own diagnosis out of the log goes
    through `_store_mutation_preflight_contract.fail_closed_records`.

    Sweeps `ast.Constant` string nodes, so docstring prose that refers to "the
    fail-closed marker" without quoting it does not trip the guard -- which is
    what lets the surviving per-suite rationale keep discussing the contract in
    words.

    The sweep is exemption-free: it covers every `test_*.py` in the whole
    `tests/` tree, subdirectories and this module included, so there is no blind
    spot in which the guard could tolerate the very literal it forbids
    everywhere else.

    The scripts under `scripts/` do contain the literal, since they EMIT it,
    and are correctly outside this sweep.
    """

    @pytest.mark.timeout(_SWEEP_TIMEOUT)
    def test_no_test_module_spells_the_fail_closed_marker(self) -> None:
        offenders: list[str] = []
        for path in _test_modules():
            tree = parse_python_module(path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and _GUARDED_MARKER in node.value
                ):
                    offenders.append(f'{path.relative_to(TESTS_DIR)}:{node.lineno}')

        assert not offenders, (
            f'{len(offenders)} inlined `{_GUARDED_MARKER}` literal(s) found. '
            'Filter the guard record through '
            '`_store_mutation_preflight_contract.fail_closed_records(caplog, '
            "'<the script's logger name>')` instead, so the fail-closed "
            'marker is spelled once:\n  ' + '\n  '.join(offenders)
        )
