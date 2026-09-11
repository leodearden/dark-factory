"""Mis-resolved pyright interpreter, end-to-end through ``run_verification``.

Task 3367 / esc-3359-1.  A cold merge worktree type-checked a DOCS-ONLY diff
against an interpreter holding none of the workspace's third-party packages —
``[tool.pyright]`` pinned no ``venvPath``/``venv``, so ``cd <sub> && npx
pyright`` resolved from the ambient ``VIRTUAL_ENV``/``PATH``, both of which
``verify._target_subprocess_env`` deliberately strips.  509+ phantom
``reportMissingImports`` errors, classified ``unknown_test_failure``: read as a
branch defect, burning a merge cycle and a steward escalation on a branch that
changed no Python at all.

The config fix (per-member ``venvPath``/``venv`` pins) prevents recurrence; this
file pins the LEGIBILITY half — that if it ever happens again the failure is
loud, distinct and correctly attributed to the environment.

Reuses ``test_verify_cold_preprovision.py``'s seam: ``patch(
'orchestrator.verify._run_cmd', side_effect=...)`` with a fake that dispatches
on a per-leg substring token.  All three legs are ``echo`` commands
(ToolKind.OPAQUE, never PYTEST) — the same ToolKind the real fleet chain
``cd fused-memory && npx pyright && ...`` resolves to, and the one that made the
incident fall through ``_classify_opaque`` to ``unknown_test_failure``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import OrchestratorConfig

TEST_CMD = 'echo TESTLEG'
LINT_CMD = 'echo LINTLEG'
TYPE_CMD = 'echo TYPELEG'


def _unresolved_import_lines(*modules: str) -> str:
    """Render *modules* in pyright's REAL unresolved-import line shape."""
    return '\n'.join(
        f'  /repo/src/fused_memory/server.py:{i + 8}:8 - error: Import "{mod}" '
        f'could not be resolved (reportMissingImports)'
        for i, mod in enumerate(modules)
    )


# The incident's signature: the baseline sentinel `pytest` unresolved alongside
# many other workspace third-party packages. Deliberately >5 distinct modules
# and many LINES per module, so the "one log line, not one per match" assertion
# below has something to collapse.
PHANTOM_TYPE_OUTPUT = (
    _unresolved_import_lines(
        'pytest', 'pydantic', 'aiosqlite', 'openai', 'qdrant_client', 'graphiti_core'
    )
    + '\n'
    + _unresolved_import_lines(
        'pytest', 'pydantic', 'aiosqlite', 'openai', 'qdrant_client', 'graphiti_core'
    )
    + '\n509 errors, 0 warnings, 0 informations\n'
)
PHANTOM_DISTINCT_MODULE_COUNT = 6

# Negative control: one genuine undeclared import — a real branch defect.
GENUINE_MISSING_IMPORT_OUTPUT = (
    _unresolved_import_lines('brand_new_lib') + '\n1 error, 0 warnings, 0 informations\n'
)


def _make_config(project_root: Path) -> OrchestratorConfig:
    """A config whose three legs are independently-controllable OPAQUE echoes."""
    return OrchestratorConfig(
        project_root=project_root,
        test_command=TEST_CMD,
        lint_command=LINT_CMD,
        type_check_command=TYPE_CMD,
        concurrent_verify=True,
    )


def _spy(invoked: list[str], *, type_out: str):
    """Fake ``_run_cmd``: the incident's rc pattern — test 0, lint 0, TYPE 1.

    Note the TEST leg PASSES.  That is the shape that makes the env-recovery
    retry gate load-bearing (see ``TestEnvRecoveryRetryGate`` below).
    """

    async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        invoked.append(cmd)
        if 'TYPELEG' in cmd:
            return 1, type_out, False
        return 0, '', False

    return fake_cmd


def _misresolution_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """ERROR records naming the interpreter mis-resolution condition."""
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and 'workspace third-party packages' in r.getMessage()
    ]


class TestInterpreterMisresolutionIsLoudAndEnvTransient:
    """The failure must be legible AS an environment fault, not a branch defect."""

    @pytest.mark.asyncio
    async def test_category_is_env_transient_end_to_end(self, tmp_path: Path) -> None:
        """Not just the classifier unit — the whole ``run_verification`` path.

        At the merge lane ``env_transient`` is in ``INFRA_TRANSIENT_CATEGORIES``,
        which routes this to a loud infra_issue hold that never blames the
        branch — the outcome that would have saved esc-3359-1's merge cycle.
        """
        config = _make_config(tmp_path)
        invoked: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=_spy(invoked, type_out=PHANTOM_TYPE_OUTPUT),
        ):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is False
        assert result.category == 'env_transient', (
            f'a mis-resolved pyright interpreter is an ENVIRONMENT fault, not a '
            f'branch defect; got category={result.category!r} '
            f'(the incident classified unknown_test_failure)'
        )

    @pytest.mark.asyncio
    async def test_emits_exactly_one_distinct_error_log_naming_the_condition(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One legible statement — not one per matching output line.

        Hundreds of phantom import lines must collapse to a single ERROR that
        names the failing check, the distinct-module count, the fix surface and
        the task/escalation reference.
        """
        config = _make_config(tmp_path)
        invoked: list[str] = []
        with caplog.at_level(logging.ERROR, logger='orchestrator.verify'), patch(
            'orchestrator.verify._run_cmd',
            side_effect=_spy(invoked, type_out=PHANTOM_TYPE_OUTPUT),
        ):
            await verify.run_verification(tmp_path, config)

        records = _misresolution_records(caplog)
        assert len(records) == 1, (
            f'expected exactly ONE mis-resolution ERROR record per failing '
            f'check (the phantom output has '
            f'{PHANTOM_TYPE_OUTPUT.count("reportMissingImports")} matching '
            f'lines, which must collapse to one statement); got {len(records)}: '
            f'{[r.getMessage() for r in records]!r}'
        )
        message = records[0].getMessage()
        assert 'type' in message, (
            f'the log must name WHICH check failed, so an operator does not '
            f'have to guess the leg; got {message!r}'
        )
        assert str(PHANTOM_DISTINCT_MODULE_COUNT) in message, (
            f'the log must report the DISTINCT unresolved top-level module '
            f'count ({PHANTOM_DISTINCT_MODULE_COUNT}) — the number that makes '
            f'"the interpreter is wrong" legible as distinct from "one import '
            f'is missing"; got {message!r}'
        )
        assert 'venvPath' in message, (
            f'the log must point at the FIX SURFACE ([tool.pyright] venvPath/'
            f'venv in the checked subproject pyproject.toml); got {message!r}'
        )
        assert '3367' in message, (
            f'the log must cite task 3367 / esc-3359-1 so the next operator '
            f'reaches the prior analysis instead of re-deriving it; '
            f'got {message!r}'
        )

    @pytest.mark.asyncio
    async def test_genuine_missing_import_stays_a_branch_defect(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Negative control: a real defect is neither excused nor mislabelled."""
        config = _make_config(tmp_path)
        invoked: list[str] = []
        with caplog.at_level(logging.ERROR, logger='orchestrator.verify'), patch(
            'orchestrator.verify._run_cmd',
            side_effect=_spy(invoked, type_out=GENUINE_MISSING_IMPORT_OUTPUT),
        ):
            result = await verify.run_verification(tmp_path, config)

        assert result.passed is False
        assert result.category != 'env_transient', (
            'a single genuine undeclared import must stay attributable to the '
            'branch — excusing it as environmental would let a real regression '
            'through the merge gate as an infra hold'
        )
        records = _misresolution_records(caplog)
        assert not records, (
            f'no mis-resolution ERROR may fire for a genuine branch defect; '
            f'got {[r.getMessage() for r in records]!r}'
        )


class TestEnvRecoveryRetryGate:
    """The env-recovery retry must fire only when the TEST leg is the failure.

    Task 3367.  ``run_verification``'s bounded env-recovery retry re-runs ONLY
    the test command, serialised.  Its gate was ``category == ENV_TRANSIENT and
    attempt.test.cmd is not None`` — with no check that the test leg actually
    failed, because until task 3367 a lint/type failure could not classify
    ENV_TRANSIENT at all.  Now that a TYPE leg can, the missing rc check became
    load-bearing: in the incident shape (test rc=0 after 1320.9s, lint rc=0,
    type rc=1) it would re-run an already-green ~22-minute test suite to
    "recover" a failure that suite cannot possibly recover.
    """

    # Same rendering as test_verify_env_transient.py's constant of the same
    # name: render(serial_pytest(parse_config_command('uv run pytest tests/'))).
    RECOVERED_TEST_COMMAND = 'uv run pytest -p no:xdist -o addopts= tests/'

    @pytest.mark.asyncio
    async def test_env_transient_from_type_leg_does_not_rerun_a_passing_test_command(
        self, tmp_path: Path
    ) -> None:
        """A passing TEST leg is invoked exactly once, never re-run."""
        config = _make_config(tmp_path)
        invoked: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=_spy(invoked, type_out=PHANTOM_TYPE_OUTPUT),
        ):
            result = await verify.run_verification(tmp_path, config)

        assert result.category == 'env_transient'
        test_invocations = [c for c in invoked if 'TESTLEG' in c]
        assert len(test_invocations) == 1, (
            f'the TEST leg passed (rc=0), so re-running it cannot change the '
            f'verdict — it only spends a full test-suite wall-clock (1320.9s in '
            f'esc-3359-1) before returning the same red. Expected exactly 1 '
            f'invocation, got {len(test_invocations)}: {test_invocations!r}'
        )

    @pytest.mark.asyncio
    async def test_env_transient_from_a_failing_test_leg_still_recovers(
        self, tmp_path: Path
    ) -> None:
        """Companion regression guard: task 2048's recovery must stay intact.

        When the TEST leg ITSELF fails with the shared-venv-mutation signature,
        the bounded single serial retry must still fire — the gate tightening
        narrows the trigger to a failing test leg, it does not remove it.
        """
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='uv run pytest tests/',
            lint_command='echo lint',
            type_check_command='echo type',
        )
        xdist_vanished = (
            'pytest: error: unrecognized arguments: -n --dist '
            '--max-worker-restart=0\n'
        )
        invoked: list[str] = []

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            invoked.append(cmd)
            if 'pytest' in cmd:
                return 4, xdist_vanished, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(tmp_path, config)

        assert result.category == 'env_transient'
        pytest_invocations = [c for c in invoked if 'pytest' in c]
        assert len(pytest_invocations) == 2, (
            f'expected the original run plus exactly one bounded serial '
            f'recovery retry, got {len(pytest_invocations)}: '
            f'{pytest_invocations!r}'
        )
        assert self.RECOVERED_TEST_COMMAND in pytest_invocations, (
            f'the recovery invocation must carry the serial_pytest markers '
            f'(-p no:xdist, -o addopts=); got {pytest_invocations!r}'
        )
