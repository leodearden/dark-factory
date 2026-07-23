"""Cold-verify shared-venv pre-provision — task 2997 (esc-2913-3).

On a COLD verify worktree the shared ``.venv`` is populated only as a SIDE
EFFECT of the TEST leg's ``cd <module> && uv run pytest``.  The full-repo-scope
root LINT (``uv run ruff check …``) and TYPE (``… npx pyright``) commands need
ruff + dev-deps that live in ``shared/pyproject.toml`` and each module's dev
group.  Launched ~11 ms alongside TEST in one ``asyncio.gather``, they observe a
freshly-created-but-empty ``.venv`` and fail spuriously (``Failed to spawn:
ruff``; ``Import "pytest" could not be resolved``).

The fix: when ``config.verify_cold_preprovision_command`` is non-empty,
``run_verification`` runs that command ONCE (coalesced per worktree) through
``_run_cmd`` BEFORE the concurrent test/lint/type gather, gated on ``is_cold``,
so the venv is populated before the racing commands spawn.

These tests reuse test_verify_sequential_lint_first.py's
``patch('orchestrator.verify._run_cmd', side_effect=fake_cmd)`` seam: fake_cmd
records every ``(cmd, cwd)`` in invocation order and returns
``(rc, out, timed_out)`` by token substring.  All check legs use ``echo``
commands (ToolKind.OPAQUE, never PYTEST) so the pytest-only env-recovery retry
never fires — only the behaviour under test is exercised.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import OrchestratorConfig

# Distinct tokens so the spy can attribute each _run_cmd invocation to a leg.
PROVISION_CMD = 'uv sync --project shared --extra dev'
TEST_CMD = 'echo TESTLEG'
LINT_CMD = 'echo LINTLEG'
TYPE_CMD = 'echo TYPELEG'
_CHECK_TOKENS = ('TESTLEG', 'LINTLEG', 'TYPELEG')


def _is_provision(cmd: str) -> bool:
    return 'uv sync' in cmd


def _is_check(cmd: str) -> bool:
    return any(tok in cmd for tok in _CHECK_TOKENS)


def _make_config(project_root: Path, *, preprovision: str = '') -> OrchestratorConfig:
    """A concurrent-verify config whose three legs are OPAQUE echo commands."""
    return OrchestratorConfig(
        project_root=project_root,
        test_command=TEST_CMD,
        lint_command=LINT_CMD,
        type_check_command=TYPE_CMD,
        concurrent_verify=True,
        verify_cold_preprovision_command=preprovision,
    )


def _spy(invoked: list[tuple[str, Path]], *, provision_rc: int = 0, provision_out: str = ''):
    """A fake ``_run_cmd`` that records (cmd, cwd) and returns rc/out by leg.

    The provision leg (``uv sync``) returns ``(provision_rc, provision_out)`` so
    a caller can force the fail-open path; every check leg returns rc=0.
    """

    async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        invoked.append((cmd, cwd))
        if _is_provision(cmd):
            return provision_rc, provision_out, False
        return 0, '', False

    return fake_cmd


def _make_warm_worktree(tmp_path: Path) -> Path:
    """A worktree that ``_is_verify_cold`` classifies WARM (``.task/verify_warmed``)."""
    task_dir = tmp_path / '.task'
    task_dir.mkdir()
    (task_dir / 'verify_warmed').touch()
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_preprovision_guard():
    """Clear the module-level coalescing guard so each test is hermetic.

    The guard is keyed by resolved worktree path and pytest ``tmp_path`` is
    unique per test, so keys never collide — but clearing keeps tests
    independent regardless of ordering.  Tolerant of the guard not existing yet
    (the RED steps before the impl lands) via ``getattr`` so those tests fail on
    their own assertion, not a fixture AttributeError.
    """
    def _clear() -> None:
        for name in ('_PREPROVISION_LOCKS', '_PREPROVISION_DONE'):
            obj = getattr(verify, name, None)
            if obj is not None:
                obj.clear()

    _clear()
    yield
    _clear()


class TestColdPreprovisionCore:
    """Core behaviour + the two inherent guards (empty-knob no-op, warm skip)."""

    @pytest.mark.asyncio
    async def test_cold_concurrent_provision_runs_first(self, tmp_path: Path):
        """(a) COLD (is_merge_verify) + concurrent + knob set ⇒ the provision
        command runs through ``_run_cmd`` in the worktree cwd and is the FIRST
        invocation, before any of the test/lint/type check legs — regardless of
        gather scheduling.  RED today: no pre-provision code exists yet."""
        config = _make_config(tmp_path, preprovision=PROVISION_CMD)
        invoked: list[tuple[str, Path]] = []
        with patch('orchestrator.verify._run_cmd', side_effect=_spy(invoked)):
            result = await verify.run_verification(tmp_path, config, is_merge_verify=True)

        provisions = [(c, cwd) for (c, cwd) in invoked if _is_provision(c)]
        assert len(provisions) == 1, f'provision must run exactly once: {invoked!r}'
        # First _run_cmd invocation overall is the provision, in the worktree cwd.
        assert _is_provision(invoked[0][0]), f'provision must be first: {invoked!r}'
        assert invoked[0][1] == tmp_path, f'provision cwd must be the worktree: {invoked!r}'
        # All three checks still ran, and every one of them ran AFTER the provision.
        checks_idx = [i for i, (c, _) in enumerate(invoked) if _is_check(c)]
        assert len(checks_idx) == 3, f'all three checks must run: {invoked!r}'
        assert min(checks_idx) > 0, f'provision must precede every check: {invoked!r}'
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_empty_knob_never_provisions(self, tmp_path: Path):
        """(b) Knob at its '' default ⇒ provision is NEVER executed (only the
        three check legs run) and the verify still passes — byte-identical to
        pre-2997 behaviour for every unconfigured target/test double."""
        config = _make_config(tmp_path, preprovision='')
        invoked: list[tuple[str, Path]] = []
        with patch('orchestrator.verify._run_cmd', side_effect=_spy(invoked)):
            result = await verify.run_verification(tmp_path, config, is_merge_verify=True)

        assert [c for (c, _) in invoked if _is_provision(c)] == [], (
            f'no provision when knob empty: {invoked!r}'
        )
        assert len([c for (c, _) in invoked if _is_check(c)]) == 3, invoked
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_warm_verify_never_provisions(self, tmp_path: Path):
        """(c) WARM verify (is_cold False via ``.task/verify_warmed``) + knob set
        ⇒ provision is NEVER executed.  The race only exists on a cold/possibly-
        empty venv; a warm lane retains its populated venv."""
        wt = _make_warm_worktree(tmp_path)
        config = _make_config(wt, preprovision=PROVISION_CMD)
        invoked: list[tuple[str, Path]] = []
        with patch('orchestrator.verify._run_cmd', side_effect=_spy(invoked)):
            # is_merge_verify defaults False → is_cold = _is_verify_cold(wt) = False.
            result = await verify.run_verification(wt, config)

        assert [c for (c, _) in invoked if _is_provision(c)] == [], (
            f'no provision on a warm worktree: {invoked!r}'
        )
        assert result.passed is True
