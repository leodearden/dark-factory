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
   docstring prose; the one docstring assertion pins *composition*, not
   wording.

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

    mod.StoreMutationUnavailable = StoreMutationUnavailable
    if with_guard:
        def _real_guard(*, operation: str) -> None:
            raise AssertionError(f'the real guard ran for {operation!r}')

        mod.assert_store_mutation_allowed = _real_guard
    return mod


# ---------------------------------------------------------------------------
# The pinned literals
# ---------------------------------------------------------------------------


class TestPinnedLiterals:
    """The two literals the whole contract is keyed on live in exactly one place.

    Exported rather than inlined so that adding a third fail-closed marker is a
    one-line change here instead of a 12-file sweep — the drift scenario this
    extraction exists to close.
    """

    def test_sentinel_value(self) -> None:
        assert SENTINEL == 'SENTINEL-store-unwritable'

    def test_fail_closed_markers(self) -> None:
        assert FAIL_CLOSED_MARKERS == ('NOT started (fail-closed)', 'MCP server')


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

    def test_doc_composes_shared_rationale_with_the_note(self) -> None:
        """The generated `__doc__` is <shared rationale> + <this suite's note>.

        Asserts COMPOSITION, not wording: each note appears, and stripping each
        note leaves the same non-empty shared remainder. That pins the property
        the extraction depends on without pinning a word of the prose.
        """
        note_a = 'ALPHA-per-suite-note'
        note_b = 'BETA-per-suite-note'
        doc_a = _underlying(neutralise_fixture(_stub_script_module(), note=note_a)).__doc__
        doc_b = _underlying(neutralise_fixture(_stub_script_module(), note=note_b)).__doc__

        assert note_a in doc_a and note_b in doc_b
        shared = doc_a.replace(note_a, '')
        assert shared == doc_b.replace(note_b, '')
        assert shared.strip(), 'the shared rationale must actually be carried'

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
_BOTH_MARKERS = 'sweep NOT started (fail-closed): route it through the MCP server'


class TestFailClosedRecords:
    """`fail_closed_records(caplog, logger_name)` isolates the guard site's OWN
    diagnosis from every other record in the log.

    Each exclusion is asserted independently, so a filter that silently drops
    one of its four clauses fails a named test rather than passing on the
    strength of the other three.
    """

    def test_returns_the_matching_record(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).error(_BOTH_MARKERS)
        assert [r.getMessage() for r in fail_closed_records(caplog, _LOGGER)] == [_BOTH_MARKERS]

    def test_excludes_another_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_OTHER_LOGGER).error(_BOTH_MARKERS)
        assert fail_closed_records(caplog, _LOGGER) == []

    def test_excludes_below_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).warning(_BOTH_MARKERS)
        assert fail_closed_records(caplog, _LOGGER) == []

    def test_excludes_a_record_missing_the_fail_closed_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).error('fatal error during sweep: route it '
                                             'through the MCP server')
        assert fail_closed_records(caplog, _LOGGER) == []

    def test_excludes_a_record_missing_the_remedy_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).error('sweep NOT started (fail-closed): boom')
        assert fail_closed_records(caplog, _LOGGER) == []

    def test_includes_critical(self, caplog: pytest.LogCaptureFixture) -> None:
        """Pins the deliberate `levelno >= logging.ERROR` superset.

        The 12 spellings this replaces were 11x `levelname == 'ERROR'` and 1x
        `levelno >= logging.ERROR`; the superset was chosen because every call
        site is a positive assertion, so a broader predicate can only add
        matches and can never turn a green assertion red.
        """
        with caplog.at_level(logging.DEBUG):
            logging.getLogger(_LOGGER).critical(_BOTH_MARKERS)
        assert [r.getMessage() for r in fail_closed_records(caplog, _LOGGER)] == [_BOTH_MARKERS]

    def test_filters_a_mixed_log_down_to_the_guard_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The real shape: `main`'s own generic ERROR shares this logger and
        level, so only the markers can tell the two records apart."""
        with caplog.at_level(logging.DEBUG):
            log = logging.getLogger(_LOGGER)
            log.info(_BOTH_MARKERS)
            log.error('fatal error during sweep')
            logging.getLogger(_OTHER_LOGGER).error(_BOTH_MARKERS)
            log.error(_BOTH_MARKERS)
        assert [r.getMessage() for r in fail_closed_records(caplog, _LOGGER)] == [_BOTH_MARKERS]


# ---------------------------------------------------------------------------
# Drift guard #1 — no test module may hand-roll the neutralising fixture
# ---------------------------------------------------------------------------

TESTS_DIR = pathlib.Path(__file__).parent

# The name every one of the 14 suites gave its hand-rolled autouse fixture
# before the extraction. After it, the name exists nowhere in `tests/`: suites
# call `neutralise_fixture(...)` and bind the result to `_neutralise`.
_HAND_ROLLED_FIXTURE_NAME = '_neutralise_store_mutation_preflight'


def _test_modules() -> list[pathlib.Path]:
    """Every collected test module in this directory, in a stable order.

    Deliberately globs `test_*.py` and NOT `*.py`: the helper home
    `_store_mutation_preflight_contract.py` is underscore-prefixed (so pytest
    does not collect it) and is therefore already excluded. Do not "fix" this
    glob to `*.py` — that would sweep the helper itself and make every guard
    below unsatisfiable.
    """
    return sorted(TESTS_DIR.glob('test_*.py'))


class TestNoHandRolledNeutraliseFixture:
    """Whole-tree AST drift guard: no test module may define its own
    `_neutralise_store_mutation_preflight`.

    Extraction alone is a one-time dedupe — the moment someone copies an older
    suite as a template, the two-sources-of-truth problem this helper exists to
    close comes straight back. This is what makes the collapse durable.

    Asserts over PARSED source (`_ast_guard.parse_python_module`), so prose in a
    docstring that merely names the old fixture cannot trip it.
    """

    def test_no_test_module_defines_its_own_neutralise_fixture(self) -> None:
        offenders: list[str] = []
        for path in _test_modules():
            tree = parse_python_module(path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == _HAND_ROLLED_FIXTURE_NAME
                ):
                    offenders.append(f'{path.name}:{node.lineno}:{node.name}')

        assert not offenders, (
            f'{len(offenders)} hand-rolled `{_HAND_ROLLED_FIXTURE_NAME}` '
            'definition(s) found. Route each through '
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
                    offenders.append(f'{path.name}:{node.lineno}')

        assert not offenders, (
            f'{len(offenders)} hand-rolled `StoreMutationUnavailable` raiser(s) '
            'found. Rig the refusal through '
            '`_store_mutation_preflight_contract.deny(<script module>, '
            'monkeypatch)` instead, so the sentinel is spelled once:\n  '
            + '\n  '.join(offenders)
        )
