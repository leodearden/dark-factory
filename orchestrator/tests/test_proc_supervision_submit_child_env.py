"""The RP-4 submit child's PYTHONPATH reaches that one command only (task 5509).

A detached ``RestartPlan`` registers one ``systemd-run --user`` transient unit
whose ``/bin/sh -c`` wrapper runs the deploy payload and, when it fails, the
``python -m escalation submit`` child. Each cell fires the REAL registered argv
the way systemd-run would and observes the ``$PYTHONPATH`` each child actually
received: the submit child gets the import roots it needs, the deploy payload
gets nothing injected.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from _proc_supervision_doubles import FakeRunner, stub_find_spec
from escalation.queue import EscalationQueue

from orchestrator.proc_supervision import (
    EscalationSpec,
    RestartDisposition,
    RestartOutcome,
    RestartPlan,
)

_LOGGER = 'orchestrator.proc_supervision'
_UNSET = '<unset>'
_TASK_ID = 'task-99'
_PAYLOAD_RC = 7
# Inside pytest-timeout's 60s per-test budget, so a wedged child is named by
# the AssertionError below rather than by an os._exit of the xdist worker
# (same reasoning as test_deterministic_runner.py::_CHILD_TIMEOUT_SECS).
_CHILD_TIMEOUT_SECS = 30
# A strict subset of any real user manager's environment: a unit that works
# here cannot be leaning on something only the caller's environment supplies.
_USER_MANAGER_ENV = {'PATH': '/usr/bin:/bin', 'HOME': str(Path.home())}


def _env_recording_stub(path: Path, record: Path, exit_code: int) -> Path:
    """An executable that writes its own ``$PYTHONPATH`` (or :data:`_UNSET`) to *record*."""
    path.write_text(
        '#!/bin/sh\n'
        f'printf %s "${{PYTHONPATH-{_UNSET}}}" > {shlex.quote(str(record))}\n'
        f'exit {exit_code}\n'
    )
    path.chmod(0o755)
    return path


def _systemd_run_env_and_cwd(options: tuple[str, ...]) -> tuple[dict[str, str], str]:
    """The environment and cwd systemd-run would give the unit for *options*.

    MEASURED in task 3453: ``systemd-run --user`` propagates none of the
    caller's environment; the unit starts from the user manager's own. That is
    modelled as :data:`_USER_MANAGER_ENV`, so nothing of this process (its
    PYTHONPATH, VIRTUAL_ENV or PATH) leaks in. ``--setenv=K=V`` options then
    set a variable for the whole unit.
    """
    env = dict(_USER_MANAGER_ENV)
    cwd = None
    for option in options:
        if option.startswith('--setenv='):
            key, _, value = option.removeprefix('--setenv=').partition('=')
            env[key] = value
        elif option.startswith('--working-directory='):
            cwd = option.removeprefix('--working-directory=')
    assert cwd is not None, f'no --working-directory option in {options!r}'
    return env, cwd


async def _fire_like_systemd_run(argv: tuple[str, ...]) -> tuple[int, str]:
    """Run the registered unit's command now; return ``(rc, combined_output)``."""
    sh_at = argv.index('/bin/sh')
    env, cwd = _systemd_run_env_and_cwd(argv[:sh_at])
    proc = await asyncio.create_subprocess_exec(
        *argv[sh_at:],
        env=env,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=_CHILD_TIMEOUT_SECS)
    except TimeoutError:
        proc.kill()
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        except TimeoutError:
            out = b''
        raise AssertionError(
            f'the fired unit did not exit within {_CHILD_TIMEOUT_SECS}s and was '
            f'killed. argv: {argv!r}. Output before the kill: '
            f'{out.decode(errors="replace")!r}'
        ) from None
    output = out.decode(errors='replace')
    rc = proc.returncode
    assert rc is not None, f'communicate() returned but the child has no exit status: {output!r}'
    return rc, output


def _rp4_plan(workdir: Path, payload: Path) -> RestartPlan:
    return RestartPlan(
        script=payload,
        args=['--foo'],
        cwd=workdir,
        target_unit='orch.service',
        own_unit='orch.service',
        on_failure_escalation=EscalationSpec(
            queue_dir=str(workdir / 'escalations'),
            task_id=_TASK_ID,
            summary='Self-restart fire-time failure',
        ),
        verify=None,
        transient_unit='orch-redeploy-restart-99.service',
        on_active_secs=10,
    )


async def _register(plan: RestartPlan) -> tuple[tuple[str, ...], RestartOutcome]:
    runner = FakeRunner(returncode=0)
    outcome = await plan.execute(runner=runner)
    argv, _kwargs = runner.calls[0]
    return argv, outcome


@dataclass(frozen=True)
class _FiredUnit:
    argv: tuple[str, ...]
    outcome: RestartOutcome
    payload_pythonpath: str
    submit_pythonpath: str


async def _fire_failing_deploy(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> _FiredUnit:
    """Register and fire a deploy whose payload fails, recording each child's PYTHONPATH.

    ``sys.executable`` is patched to a recording stub; this reaches the real
    wrapper because ``RestartPlan`` reads it at execute() time.
    """
    payload_record = workdir / 'payload.pythonpath'
    submit_record = workdir / 'submit.pythonpath'
    payload = _env_recording_stub(workdir / 'payload.sh', payload_record, _PAYLOAD_RC)
    monkeypatch.setattr(
        sys, 'executable',
        str(_env_recording_stub(workdir / 'stub-python', submit_record, 0)),
    )
    argv, outcome = await _register(_rp4_plan(workdir, payload))

    rc, output = await _fire_like_systemd_run(argv)

    assert rc == _PAYLOAD_RC, f'expected the payload rc {_PAYLOAD_RC}, got {rc}: {output!r}'
    assert submit_record.exists(), f'the on-failure submit branch never ran: {output!r}'
    return _FiredUnit(
        argv=argv,
        outcome=outcome,
        payload_pythonpath=payload_record.read_text(),
        submit_pythonpath=submit_record.read_text(),
    )


def _roots(pythonpath: str) -> list[str]:
    assert pythonpath != _UNSET, 'the submit child received no PYTHONPATH at all'
    return pythonpath.split(os.pathsep)


def _resolves(roots: list[str], package: str) -> bool:
    return any((Path(r) / package / '__init__.py').exists() for r in roots)


def _import_roots(package: str) -> list[str]:
    """The directories *package* is imported from, whatever the install layout."""
    spec = importlib.util.find_spec(package)
    assert spec is not None and spec.submodule_search_locations, (
        f'{package!r} must resolve in the test interpreter'
    )
    return list(dict.fromkeys(
        str(Path(loc).resolve().parent) for loc in spec.submodule_search_locations
    ))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
class TestSubmitChildPythonPathScope:
    async def test_deploy_payload_gets_no_pythonpath_while_submit_child_does(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv('PYTHONPATH', raising=False)

        fired = await _fire_failing_deploy(tmp_path, monkeypatch)

        assert fired.payload_pythonpath == _UNSET, (
            'the deploy payload shares the transient unit with the submit child '
            f'but must not inherit its PYTHONPATH: {fired.payload_pythonpath!r}'
        )
        roots = _roots(fired.submit_pythonpath)
        for root in roots:
            assert root, f'no empty PYTHONPATH entry may be supplied: {roots!r}'
            assert Path(root).is_absolute(), (
                f'{root!r} must be absolute: the unit runs under the systemd '
                'user manager, not this process'
            )
        assert _resolves(roots, 'escalation'), f'no root resolves `escalation`: {roots!r}'
        assert _resolves(roots, 'shared'), (
            'no root resolves `shared`, which the submit child imports via '
            f'escalation.queue -> shared.timestamps: {roots!r}'
        )

    async def test_systemd_run_argv_carries_no_setenv_option(self, tmp_path: Path) -> None:
        argv, _outcome = await _register(_rp4_plan(tmp_path, tmp_path / 'payload.sh'))

        options = argv[:argv.index('/bin/sh')]
        assert not [o for o in options if o.startswith('--setenv=')], (
            f'a --setenv option sets the variable for the whole unit: {options!r}'
        )
        assert argv[:-1] == (
            'systemd-run', '--user',
            '--on-active=10',
            '--unit=orch-redeploy-restart-99.service',
            '--collect',
            f'--working-directory={tmp_path}',
            '/bin/sh', '-c',
        )

    async def test_ambient_pythonpath_resupplied_first_to_submit_child_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = '/tmp/df 5509 ambient-sentinel'
        monkeypatch.setenv('PYTHONPATH', sentinel)

        fired = await _fire_failing_deploy(tmp_path, monkeypatch)

        roots = _roots(fired.submit_pythonpath)
        assert roots[0] == sentinel, (
            "this process's own PYTHONPATH must reach the submit child first and "
            f'byte-exact: {roots!r}'
        )
        assert fired.payload_pythonpath == _UNSET, fired.payload_pythonpath

        monkeypatch.delenv('PYTHONPATH')
        second = tmp_path / 'second'
        second.mkdir()

        refired = await _fire_failing_deploy(second, monkeypatch)

        reroots = _roots(refired.submit_pythonpath)
        assert reroots, 'with PYTHONPATH unset the resolved-root floor must still be supplied'
        assert sentinel not in reroots, (
            f'the value is recomputed per call, never cached: {reroots!r}'
        )

    async def test_unresolvable_packages_degrade_loudly_and_never_block(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv('PYTHONPATH', raising=False)
        stub_find_spec(monkeypatch, {'escalation', 'shared'})

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            fired = await _fire_failing_deploy(tmp_path, monkeypatch)

        assert fired.submit_pythonpath == _UNSET, (
            f'nothing resolved, so nothing may be injected: {fired.submit_pythonpath!r}'
        )
        assert any('escalation' in m and 'PYTHONPATH' in m for m in _warnings(caplog)), (
            'an unresolvable submit-child package must be reported by name '
            f'alongside the PYTHONPATH consequence: {_warnings(caplog)!r}'
        )
        assert fired.outcome.disposition == RestartDisposition.SCHEDULED, (
            'a degraded escalation guarantee must NEVER block the restart'
        )

    async def test_partial_resolution_supplies_what_resolved(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv('PYTHONPATH', raising=False)
        escalation_roots = _import_roots('escalation')
        stub_find_spec(monkeypatch, {'shared'})

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            fired = await _fire_failing_deploy(tmp_path, monkeypatch)

        roots = _roots(fired.submit_pythonpath)
        assert _resolves(roots, 'escalation'), (
            f'the root that DID resolve must still be supplied: {roots!r}'
        )
        assert roots == escalation_roots, (
            'only the escalation import root may be supplied; the stubbed-'
            f'unresolvable `shared` contributes none of its own: {roots!r} '
            f'!= {escalation_roots!r}'
        )
        assert any('shared' in m for m in _warnings(caplog)), _warnings(caplog)
        assert fired.outcome.disposition == RestartDisposition.SCHEDULED

    async def test_real_submit_child_files_l2_with_only_the_prefixed_pythonpath(
        self, tmp_path: Path,
    ) -> None:
        payload = _env_recording_stub(
            tmp_path / 'payload.sh', tmp_path / 'payload.pythonpath', _PAYLOAD_RC,
        )
        argv, _outcome = await _register(_rp4_plan(tmp_path, payload))

        rc, output = await _fire_like_systemd_run(argv)

        assert rc == _PAYLOAD_RC, (
            f'expected the payload rc {_PAYLOAD_RC} (97 means the submit failed), '
            f'got {rc}: {output!r}'
        )
        filed = EscalationQueue(tmp_path / 'escalations').get_by_task(
            _TASK_ID, agent_role='orchestrator-deterministic',
        )
        assert len(filed) == 1, f'exactly one L2 must be filed, got {filed!r}: {output!r}'


@pytest.mark.asyncio
class TestNoSubmitChildNoEnvironment:
    async def test_no_setenv_when_plan_has_no_escalation_spec(self) -> None:
        """A plan with no EscalationSpec has no submit child, so its argv carries
        no --setenv option and stays byte-identical."""
        runner = FakeRunner(returncode=0)
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-orchestrator.sh'),
            args=['--foo'],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=None,
            verify=None,
            transient_unit='orch-redeploy-restart-99.service',
            on_active_secs=10,
        )

        await plan.execute(runner=runner)

        argv, _kwargs = runner.calls[0]
        assert not [t for t in argv if isinstance(t, str) and t.startswith('--setenv=')], (
            'no --setenv token may be emitted when there is no submit child to '
            f'protect: {argv!r}'
        )
        expected_payload = ' '.join(
            shlex.quote(p) for p in ['/proj/scripts/restart-orchestrator.sh', '--foo']
        )
        assert argv == (
            'systemd-run', '--user',
            '--on-active=10',
            '--unit=orch-redeploy-restart-99.service',
            '--collect',
            '--working-directory=/proj',
            '/bin/sh', '-c', expected_payload,
        )
