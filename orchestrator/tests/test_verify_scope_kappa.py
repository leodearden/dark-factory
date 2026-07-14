"""Tests for orchestrator.verify's plan-authoritative execution (task κ).

verify-scope-inversion PRD task κ (plans/verify-scope-inversion-prd.md):
``run_scoped_verification`` becomes derive→execute→aggregate, EXECUTING the
``VerifyPlan`` ``derive_verify_plan`` produces instead of re-deriving scope via
the hand-mirrored ``scope_module_config`` decision tree (deleted in step-6).
This changes WHO decides verify scope, not WHAT is decided — every golden
below pins that the plan-driven execution reproduces the pre-refactor scope
decisions byte-identically.

GOLDEN fixtures:

- ``ROOT_CONFTEST_DIFF`` / ``DATA_MODULE_DIFF`` are the W7 corpus (task-1077
  commit cb7277926d, task-1852 commit 7c9b316260), reused directly from
  ``test_verify_plan.py`` so both suites share one provenance-pinned source of
  truth rather than a second hand-copied literal.
- ``STRUCTURAL_FILE_DIFF`` / ``SOURCE_ONLY_ZERO_PYTEST_DIFF`` /
  ``FALLBACK_SUBPROJECT_DIFF`` / ``UNREGISTERED_PATH_DIFF`` are new shapes
  this task's goldens need that test_verify_plan.py's corpus doesn't already
  cover (D2 structural widening, the zero-pytest SKIPPED shape, task
  2344/2355 subproject rescoping, and the scoped-fallback-never-global-fanout
  boundary row respectively).

Spy helpers:

- :func:`_run_verification_spy` / :func:`_executed_module_configs` — the
  module-config-level spy (patches ``orchestrator.verify.run_verification``),
  modeled on ``TestRunScopedVerificationPlan``'s
  ``mock_run_verification.await_args.args[2]`` pattern (test_verify.py),
  generalised to capture every call in order instead of just the last.
- :func:`_run_cmd_spy` — the raw-shell-string-level spy (patches
  ``orchestrator.verify._run_cmd``), modeled on
  ``TestRunScopedVerificationSkipsUntouched``'s ``fake_run_cmd`` (test_verify.py).

Autouse fixtures (``_neutralize_verify_admission``, ``_clear_probe_cache``,
``_mock_merge_queue_verification``, ...) live in ``orchestrator/tests/conftest.py``
and apply to this module automatically — nothing to import or opt into here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from test_verify import _canned_passing_result
from test_verify_plan import (  # noqa: F401 — reused by this module's byte-identical goldens (steps 3/7/9)
    DATA_MODULE_DIFF,
    ROOT_CONFTEST_DIFF,
)

from orchestrator.config import ModuleConfig

# ---------------------------------------------------------------------------
# GOLDEN diff shapes (task κ corpus) — see module docstring for provenance.
# ---------------------------------------------------------------------------

# A Protocol-defining source file (D2): file-scoped pyright cannot verify
# cross-file Protocol conformance, so a STRUCTURAL file must widen pyright to
# the unscoped package-wide command in BOTH the module-config and fallback
# paths. Unlike ROOT_CONFTEST_DIFF/DATA_MODULE_DIFF (path-only — classify_file
# never needs their content), STRUCTURAL detection is content-based, so a
# real worktree file with STRUCTURAL_FILE_CONTENT must be written for this
# path in each test that uses it.
STRUCTURAL_FILE_DIFF: list[str] = ['mymod/interfaces.py']
STRUCTURAL_FILE_CONTENT: str = (
    'from typing import Protocol\n\n\nclass Foo(Protocol):\n    def method(self) -> None: ...\n'
)

# A plain source file with no collectable test alongside it — the
# verify_plan.py:318-322 "no collectable test files touched" SKIPPED pytest
# shape (zero pytest invocations for this module; never a fabricated rc=5
# "no tests ran" run).
SOURCE_ONLY_ZERO_PYTEST_DIFF: list[str] = ['mymod/helpers.py']

# Fallback-subproject (cockpit-shaped, tasks 2344/2355): every touched file
# lives under a single top-level directory that carries its own
# pyproject.toml, so the fallback TEST command scopes to run *inside* that
# subproject (`cd cockpit && uv run pytest tests/test_c3.py`) and TYPE/LINT
# rescope into its own uv context. A test using this shape must also create
# `<worktree>/cockpit/pyproject.toml` — _single_subproject_prefix's
# discriminator — for the subproject-scoping branch to fire.
FALLBACK_SUBPROJECT_DIFF: list[str] = ['cockpit/tests/test_c3.py']

# Unregistered-path diff (only tests/scripts/ — boundary row 9): no
# module_configs prefix matches, so this drives the plan-driven FALLBACK
# branch (scoped commands only) — never the whole-repo global fan-out chain,
# the wall-clock-costly path this task must not regress into.
UNREGISTERED_PATH_DIFF: list[str] = ['tests/scripts/test_deploy.py']


# ---------------------------------------------------------------------------
# Spy helpers
# ---------------------------------------------------------------------------


def _run_verification_spy() -> AsyncMock:
    """AsyncMock stand-in for ``orchestrator.verify.run_verification``.

    Returns a canned passing ``VerifyResult`` for every call — never spawns a
    real subprocess. Patch via
    ``patch.object(verify, 'run_verification', new=_run_verification_spy())``
    and recover the ordered list of executed ``ModuleConfig``(s) afterward via
    :func:`_executed_module_configs`.
    """
    return AsyncMock(return_value=_canned_passing_result())


def _executed_module_configs(mock: AsyncMock) -> list[ModuleConfig]:
    """The ordered list of ``ModuleConfig``(s) *mock* was awaited with.

    *mock* is a spy built by :func:`_run_verification_spy`.
    ``run_verification``'s signature is
    ``(worktree, config, module_config=None, *, ...)`` — ``module_config`` is
    its 3rd positional argument at every ``run_scoped_verification`` call site
    that passes one (the module-config and fallback execution branches); the
    force_workspace/global/no-scope branches call it with only
    ``(worktree, config)`` and are excluded here, since there is no
    ``ModuleConfig`` to compare against a plan run in that case.
    """
    return [
        call.args[2]
        for call in mock.await_args_list
        if len(call.args) > 2 and call.args[2] is not None
    ]


def _run_cmd_spy() -> tuple[list[str], object]:
    """A ``_run_cmd`` fake recording every raw shell command string invoked.

    Mirrors ``TestRunScopedVerificationSkipsUntouched``'s ``fake_run_cmd``
    (test_verify.py) — patch via
    ``patch('orchestrator.verify._run_cmd', side_effect=<the returned fake>)``.
    Every call is a canned pass: ``(rc=0, '', timed_out=False)``.

    Returns ``(calls, fake)``: *calls* accumulates the raw command strings in
    call order; *fake* is the coroutine function to hand to ``patch(...)``.
    """
    calls: list[str] = []

    async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        calls.append(cmd)
        return 0, '', False

    return calls, fake_run_cmd
