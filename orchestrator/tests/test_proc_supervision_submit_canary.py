"""The RP-4 registration-time canary of orchestrator.proc_supervision (task 5508).

Behaviour pinned: when a detached restart plan carries an on-failure
escalation, registering it WARNs iff ``escalation.submit`` fails to import in
this interpreter, and the restart is scheduled regardless.

Failures are simulated only through ``sys.modules`` (restored by monkeypatch),
never by patching the probe, so these cells hold for any probe that really
imports the module. The real chain they break is
``escalation.submit`` -> ``escalation.queue`` -> ``shared.timestamps``.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.proc_supervision import (
    EscalationSpec,
    RestartDisposition,
    RestartPlan,
)

_LOGGER = 'orchestrator.proc_supervision'


async def _ok_runner(*args: object, **kwargs: object) -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(b'', None))
    proc.returncode = 0
    return proc


def _detached_plan(queue_dir: Path, *, with_spec: bool) -> RestartPlan:
    spec = EscalationSpec(
        queue_dir=str(queue_dir),
        task_id='task-99',
        summary='Self-restart fire-time failure',
    )
    return RestartPlan(
        script=Path('/proj/scripts/restart-orchestrator.sh'),
        args=['--foo'],
        cwd=Path('/proj'),
        target_unit='orch.service',
        own_unit='orch.service',
        on_failure_escalation=spec if with_spec else None,
        verify=None,
        transient_unit='orch-redeploy-restart-99.service',
        on_active_secs=10,
    )


def _exploding_module(module_name: str) -> types.ModuleType:
    module = types.ModuleType(module_name)

    def __getattr__(name: str) -> object:
        raise RuntimeError(f'simulated import-time failure reading {module_name}.{name}')

    module.__getattr__ = __getattr__
    return module


def _break_submit_import(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    if how == 'submit_absent':
        monkeypatch.setitem(sys.modules, 'escalation.submit', None)
        return
    monkeypatch.delitem(sys.modules, 'escalation.submit', raising=False)
    monkeypatch.delitem(sys.modules, 'escalation.queue', raising=False)
    if how == 'transitive_import_error':
        monkeypatch.setitem(sys.modules, 'shared.timestamps', None)
    elif how == 'transitive_raises_at_import':
        monkeypatch.setitem(
            sys.modules, 'shared.timestamps', _exploding_module('shared.timestamps'),
        )
    else:
        raise ValueError(f'unknown simulation {how!r}')


def _submit_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and 'escalation.submit' in r.getMessage()
    ]


@pytest.mark.asyncio
class TestRegistrationTimeSubmitImportCanary:

    @pytest.mark.parametrize(
        ('how', 'expected_exc_type', 'cause_tokens'),
        [
            ('submit_absent', ModuleNotFoundError,
             ['escalation.submit', 'ModuleNotFoundError']),
            ('transitive_import_error', ModuleNotFoundError,
             ['shared.timestamps', 'ModuleNotFoundError']),
            ('transitive_raises_at_import', RuntimeError, ['RuntimeError']),
        ],
    )
    async def test_failed_submit_import_warns_and_still_schedules(
        self,
        how: str,
        expected_exc_type: type[BaseException],
        cause_tokens: list[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _break_submit_import(monkeypatch, how)

        with pytest.raises(expected_exc_type):
            importlib.import_module('escalation.submit')
        if how != 'submit_absent':
            assert importlib.util.find_spec('escalation.submit') is not None, (
                'the transitive simulations must stay FINDABLE but unimportable '
                '— the blind spot of a find_spec-only probe (task 5508)'
            )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            outcome = await _detached_plan(tmp_path, with_spec=True).execute(
                runner=_ok_runner,
            )

        warnings = _submit_warnings(caplog)
        assert any(all(token in w for token in cause_tokens) for w in warnings), (
            f'a failed escalation.submit import must be reported with its cause '
            f'{cause_tokens!r} at registration time: {warnings!r}'
        )
        assert outcome.disposition == RestartDisposition.SCHEDULED, (
            'the canary reports, it must never block the restart'
        )

    async def test_importable_submit_module_logs_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await _detached_plan(tmp_path, with_spec=True).execute(runner=_ok_runner)

        noisy = _submit_warnings(caplog)
        assert not noisy, f'the canary must stay silent when the import is healthy: {noisy!r}'

    async def test_plan_without_escalation_spec_never_probes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _break_submit_import(monkeypatch, 'transitive_import_error')

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            outcome = await _detached_plan(tmp_path, with_spec=False).execute(
                runner=_ok_runner,
            )

        noisy = _submit_warnings(caplog)
        assert not noisy, f'a plan with no on-failure escalation has no submit child to probe: {noisy!r}'
        assert outcome.disposition == RestartDisposition.SCHEDULED
