"""Guard: no ``@pytest.mark.timeout(N)`` may INVERT into a tighter clamp under verify.

(Module docstring completed in a later step; this file is being built up
test-first.  See task 5147.)
"""

from __future__ import annotations

import re
import textwrap

import yaml
from _orch_helpers import (
    ORCH_DIR,
    PYPROJECT_DEFAULT_TIMEOUT,
    VERIFY_CLI_PER_TEST_TIMEOUT,
)

#: The per-module merge-verify config whose ``test_command`` carries the
#: ``--timeout=N`` that verify actually passes to pytest.  Resolved from
#: ``ORCH_DIR`` (itself resolved from ``_orch_helpers.__file__``) and never
#: from the process CWD: merge-verify runs pytest from the ``orchestrator/``
#: cwd while a plain ``pytest orchestrator/tests`` runs from the repo root,
#: and this pin must read identically under both.
_ORCH_YAML = ORCH_DIR / 'orchestrator.yaml'

#: Same spelling as tests/scripts/test_fallback_verify_config.py, which pins
#: the FLEET-chain side of this same budget (``--timeout > 60`` on every
#: pytest segment of dark-factory-orchestrator.yaml, and ``--timeout >= 300``
#: on every per-module orchestrator.yaml).  Both the ``--timeout=300`` and
#: ``--timeout 300`` spellings are accepted, exactly as it does.
_TIMEOUT_FLAG_RE = re.compile(r'--timeout[=\s](\d+)')


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
