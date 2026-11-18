"""The RP-4 submit child's PYTHONPATH reaches that one command only (task 5509).

A detached ``RestartPlan`` registers one ``systemd-run --user`` transient unit
whose ``/bin/sh -c`` wrapper runs the deploy payload and, when it fails, the
``python -m escalation submit`` child. Each cell fires the REAL registered argv
the way systemd-run would and observes the ``$PYTHONPATH`` each child actually
received: the submit child gets the import roots it needs, the deploy payload
gets nothing injected.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import os
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from _proc_supervision_doubles import (
    FakeRunner,
    detached_rp4_plan,
    fire_like_systemd_run,
    read_env_probe,
    stub_find_spec,
    write_env_probe,
)
from _proc_supervision_doubles import tmp_queue_dir as tmp_queue_dir
from escalation.queue import EscalationQueue

from orchestrator.proc_supervision import (
    RestartDisposition,
    RestartOutcome,
    RestartPlan,
)

_LOGGER = 'orchestrator.proc_supervision'
_PAYLOAD_RC = 7


def _rp4_plan(workdir: Path, payload: Path) -> RestartPlan:
    return dataclasses.replace(
        detached_rp4_plan(workdir / 'escalations', with_spec=True),
        script=payload,
        cwd=workdir,
    )


async def _register(plan: RestartPlan) -> tuple[tuple[str, ...], RestartOutcome]:
    runner = FakeRunner(returncode=0)
    outcome = await plan.execute(runner=runner)
    argv, _kwargs = runner.calls[0]
    return argv, outcome


@dataclasses.dataclass(frozen=True)
class _FiredUnit:
    outcome: RestartOutcome
    payload_env: Mapping[str, str]
    submit_env: Mapping[str, str]


async def _fire_failing_deploy(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> _FiredUnit:
    """Register and fire a deploy whose payload fails, recording each child's environment.

    ``sys.executable`` is patched to a recording stub; this reaches the real
    wrapper because ``RestartPlan`` reads it at execute() time.
    """
    payload_record = workdir / 'payload.env'
    submit_record = workdir / 'submit.env'
    payload = write_env_probe(
        workdir / 'payload.sh', record_to=payload_record, exit_code=_PAYLOAD_RC,
    )
    monkeypatch.setattr(
        sys, 'executable',
        str(write_env_probe(workdir / 'stub-python', record_to=submit_record, exit_code=0)),
    )
    argv, outcome = await _register(_rp4_plan(workdir, payload))

    rc, output = await fire_like_systemd_run(argv)

    assert rc == _PAYLOAD_RC, f'expected the payload rc {_PAYLOAD_RC}, got {rc}: {output!r}'
    assert submit_record.exists(), f'the on-failure submit branch never ran: {output!r}'
    return _FiredUnit(
        outcome=outcome,
        payload_env=read_env_probe(payload_record),
        submit_env=read_env_probe(submit_record),
    )


def _roots(env: Mapping[str, str]) -> list[str]:
    assert 'PYTHONPATH' in env, 'the submit child received no PYTHONPATH at all'
    return env['PYTHONPATH'].split(os.pathsep)


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

        assert 'PYTHONPATH' not in fired.payload_env, (
            'the deploy payload shares the transient unit with the submit child '
            f"but must not inherit its PYTHONPATH: {fired.payload_env['PYTHONPATH']!r}"
        )
        roots = _roots(fired.submit_env)
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
        sentinel = '/tmp/df 5509 $x ${HOME} $$'
        monkeypatch.setenv('PYTHONPATH', sentinel)

        fired = await _fire_failing_deploy(tmp_path, monkeypatch)

        roots = _roots(fired.submit_env)
        assert roots[0] == sentinel, (
            "this process's own PYTHONPATH must reach the submit child first and "
            "byte-exact, through systemd's ExecStart `$` expansion and the shell: "
            f'{roots!r}'
        )
        assert 'PYTHONPATH' not in fired.payload_env, fired.payload_env

        monkeypatch.delenv('PYTHONPATH')
        second = tmp_path / 'second'
        second.mkdir()

        refired = await _fire_failing_deploy(second, monkeypatch)

        reroots = _roots(refired.submit_env)
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

        assert 'PYTHONPATH' not in fired.submit_env, (
            f"nothing resolved, so nothing may be injected: {fired.submit_env['PYTHONPATH']!r}"
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

        roots = _roots(fired.submit_env)
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
        """Guards that the prefixed command still runs and files the L2 byte-exact.

        Not evidence that the prefix is necessary: an editable-install venv
        imports the workspace packages with no PYTHONPATH at all. Necessity is
        pinned by the stub-interpreter cells above.
        """
        payload = write_env_probe(
            tmp_path / 'payload.sh', record_to=tmp_path / 'payload.env', exit_code=_PAYLOAD_RC,
        )
        plan = _rp4_plan(tmp_path, payload)
        assert plan.on_failure_escalation is not None
        spec = dataclasses.replace(
            plan.on_failure_escalation, summary='fire-time failure $x ${HOME} $$ end',
        )
        argv, _outcome = await _register(dataclasses.replace(plan, on_failure_escalation=spec))

        rc, output = await fire_like_systemd_run(argv)

        assert rc == _PAYLOAD_RC, (
            f'expected the payload rc {_PAYLOAD_RC} (97 means the submit failed), '
            f'got {rc}: {output!r}'
        )
        filed = EscalationQueue(Path(spec.queue_dir)).get_by_task(
            spec.task_id, agent_role=spec.agent_role,
        )
        assert [e.summary for e in filed] == [spec.summary], (
            'exactly one L2 must be filed, its summary intact through '
            f"systemd's ExecStart `$` expansion: {filed!r}: {output!r}"
        )


@pytest.mark.asyncio
class TestNoSubmitChildNoEnvironment:
    async def test_no_setenv_when_plan_has_no_escalation_spec(
        self, tmp_queue_dir: Path,
    ) -> None:
        """A plan with no EscalationSpec has no submit child, so its argv carries
        no --setenv option and stays byte-identical."""
        runner = FakeRunner(returncode=0)
        plan = detached_rp4_plan(tmp_queue_dir, with_spec=False)

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
