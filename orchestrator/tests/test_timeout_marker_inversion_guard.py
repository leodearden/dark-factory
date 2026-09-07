"""Guard: no ``@pytest.mark.timeout(N)`` may INVERT into a tighter clamp under verify.

(Module docstring completed in a later step; this file is being built up
test-first.  See task 5147.)
"""

from __future__ import annotations

import re

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
