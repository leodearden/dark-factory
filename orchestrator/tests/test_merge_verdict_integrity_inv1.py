"""PRD §7 two-way boundary capstone for INV-1 (task 2883, leaf α).

INV-1: a merge-role verify PASS is adoptable ONLY if it executed the project's
merge gate on the vouched-for tree. Any "nothing to run" resolution (no source
files, empty existing_files, empty command set) must NOT trivially pass — it
escalates to the full gate, and if no full-gate command exists it FAILs loud.

This capstone composes the REAL substrate end-to-end across BOTH runner
localities (PRD §7):

  (1) Pipeline -> LocalRunner: a real LocalRunner.run_merge_verify over a
      zero-module-config, no-source (clobbered/docs-only) diff comes back
      full-gate-run (command executed, trivial False) OR loud FAIL
      (merge_no_evidence) — never a bare trivial PASS — and records a
      trivial_pass_escalated event.

  (2) Pipeline -> fake RemoteRunner: build_merge_verify_spec never hands a
      runner a no-commands spec for a zero-module-config project with a live
      gate — spec.global_verify_command carries the full gate over the wire,
      so the remote runs the SAME gate as local (fidelity, incident 966f23a6).

Only the non-deterministic command runner (verify._run_cmd) is faked, via a
recording spy — mirroring the guard_spy harness in test_verify.py and the
_FakeRunner / _RecordingEventStore doubles in test_multihost_verify_integration.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.verify import VerifyResult, run_scoped_verification
from orchestrator.verify_runner import (
    LocalRunner,
    MergeVerifySpec,
    RunnerUnavailable,
    UnscopedTypecheckSpec,
    build_merge_verify_spec,
    spec_from_json,
    spec_to_json,
)

_SENTINEL_CMD = '__scope_all_cmd__'


# ---------------------------------------------------------------------------
# Doubles — recording event store + recording command spy + fake remote runner
# ---------------------------------------------------------------------------


class _RecordingEventStore:
    """Minimal EventStore stand-in capturing emit() calls in-memory."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, str | None, dict[str, Any]]] = []

    def emit(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        duration_ms: float | None = None,
        **kw: Any,
    ) -> None:
        self.events.append((event_type, task_id, dict(data or {})))

    def events_of(self, event_type: Any) -> list[tuple[Any, str | None, dict[str, Any]]]:
        return [e for e in self.events if e[0] == event_type]


def _run_cmd_spy() -> tuple[Any, list[str]]:
    """Return (spy, calls): a fake verify._run_cmd that records command strings
    and always succeeds (rc=0), so the full gate is observable without a real
    toolchain (mirrors test_verify.py::guard_spy)."""
    calls: list[str] = []

    async def _spy(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        calls.append(cmd)
        return 0, '', False

    return _spy, calls


class _RemoteFakeRunner:
    """Fake remote VerifyRunner (is_local=False) — records the spec it is handed
    and returns a canned PASS. Mirrors test_multihost_verify_integration._FakeRunner."""

    is_local: ClassVar[bool] = False

    def __init__(self, name: str = 'laptop') -> None:
        self.name = name
        self.received_specs: list[MergeVerifySpec] = []

    async def health(self) -> bool:
        return True

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        self.received_specs.append(spec)
        return VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='', summary='ok',
        )


def _passing_unscoped_double() -> AsyncMock:
    return AsyncMock(
        return_value=MagicMock(
            broken=False, timed_out=False,
            failing_subprojects=[], timed_out_subprojects=[],
        )
    )


def _write_docs(tmp_path: Path) -> None:
    docs = tmp_path / 'docs'
    docs.mkdir(parents=True, exist_ok=True)
    (docs / 'x.md').write_text('# doc\n')


# ---------------------------------------------------------------------------
# (1) Pipeline -> LocalRunner boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInv1LocalRunnerBoundary:
    """A real LocalRunner.run_merge_verify over a zero-module-config no-source
    diff never returns a bare trivial PASS: it runs the full gate or loud-FAILs,
    and records a trivial_pass_escalated event."""

    async def test_zero_module_global_command_runs_full_gate_and_emits_event(
        self, tmp_path: Path,
    ):
        _write_docs(tmp_path)
        spy, calls = _run_cmd_spy()
        event_store = _RecordingEventStore()
        # Zero-module-config (reify-like) project with a LIVE global gate command.
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command=_SENTINEL_CMD, lint_command='', type_check_command='',
        )
        spec = build_merge_verify_spec(config, [], ('docs/x.md',))
        runner = LocalRunner(
            merge_wt=tmp_path,
            config=config,
            module_configs=[],
            task_files=('docs/x.md',),
            run_scoped=run_scoped_verification,
            run_unscoped=_passing_unscoped_double(),
            task_id='t-2883',
            event_store=event_store,
        )

        with patch('orchestrator.verify._run_cmd', side_effect=spy):
            result = await runner.run_merge_verify('sha-full', spec)

        # The full gate RAN (never a bare 0ms trivial pass) ...
        assert _SENTINEL_CMD in ' | '.join(calls), (
            f'Expected the escalated global full gate to run; got calls={calls}'
        )
        assert result.passed is True
        assert result.trivial is False
        # ... and the escalation fact was recorded.
        escalated = event_store.events_of(EventType.trivial_pass_escalated)
        assert len(escalated) == 1, f'Expected exactly one escalation event; got {escalated}'
        _, _, data = escalated[0]
        assert data['reason'] == 'no_source_files'
        assert data['resolution'] == 'full_gate'

    async def test_zero_module_no_command_loud_fails_and_emits_event(
        self, tmp_path: Path,
    ):
        _write_docs(tmp_path)
        spy, calls = _run_cmd_spy()
        event_store = _RecordingEventStore()
        # A command-less config: the merge gate has NOTHING to run -> loud FAIL,
        # never a vacuous PASS.
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='', lint_command='', type_check_command='',
        )
        spec = build_merge_verify_spec(config, [], ('docs/x.md',))
        runner = LocalRunner(
            merge_wt=tmp_path,
            config=config,
            module_configs=[],
            task_files=('docs/x.md',),
            run_scoped=run_scoped_verification,
            run_unscoped=_passing_unscoped_double(),
            task_id='t-2883',
            event_store=event_store,
        )

        with patch('orchestrator.verify._run_cmd', side_effect=spy):
            result = await runner.run_merge_verify('sha-loud', spec)

        assert result.passed is False
        assert result.category == 'merge_no_evidence'
        assert calls == [], f'No command should run for a command-less loud FAIL; got {calls}'
        escalated = event_store.events_of(EventType.trivial_pass_escalated)
        assert len(escalated) == 1, f'Expected exactly one escalation event; got {escalated}'
        _, _, data = escalated[0]
        assert data['resolution'] == 'loud_fail'


# ---------------------------------------------------------------------------
# (2) Pipeline -> fake RemoteRunner boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInv1RemoteRunnerBoundary:
    """The pipeline never hands a runner a no-commands spec for a
    zero-module-config project with a live gate — spec.global_verify_command
    carries the full gate over the wire."""

    async def test_pipeline_ships_full_gate_to_remote_not_empty(self):
        config = OrchestratorConfig(
            test_command=_SENTINEL_CMD, lint_command='', type_check_command='',
        )
        # A zero-module-config project: verify_commands is empty, so a naive
        # remote would fall back to its own (stale) globals -> the 966f23a6 hole.
        spec = build_merge_verify_spec(config, [], ('docs/x.md',))
        assert spec.verify_commands == ()
        # ... but the pipeline ships the LIVE full gate.
        assert spec.global_verify_command is not None, (
            'INV-1: a zero-module-config spec must carry the global full gate, '
            'never an empty command set that would resolve to a bare PASS'
        )
        assert spec.global_verify_command.test_command == _SENTINEL_CMD

        # Wire fidelity: the gate survives the JSON transport to the remote host.
        restored = spec_from_json(spec_to_json(spec))
        assert restored.global_verify_command == spec.global_verify_command

        # The fake remote runner accepts the populated spec (two-way boundary).
        remote = _RemoteFakeRunner()
        assert remote.is_local is False
        result = await remote.run_merge_verify('sha-remote', restored)
        assert result.passed is True
        assert remote.received_specs[0].global_verify_command is not None

    async def test_commandless_zero_module_config_yields_no_gate_spec(self):
        """A truly command-less project yields global_verify_command=None; that
        no-evidence resolution is caught as a loud FAIL on the run side (covered
        by the LocalRunner boundary above), NOT shipped as a passable gate."""
        config = OrchestratorConfig(
            test_command='', lint_command='', type_check_command='',
        )
        spec = build_merge_verify_spec(config, [], ('docs/x.md',))
        assert spec.verify_commands == ()
        assert spec.global_verify_command is None

    async def test_remote_runner_unavailable_raises_not_passes(self):
        """Sanity guard on the fake: an unavailable remote never silently PASSes."""
        class _Unavailable(_RemoteFakeRunner):
            async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
                raise RunnerUnavailable('down')

        with pytest.raises(RunnerUnavailable):
            await _Unavailable().run_merge_verify('sha', _make_min_spec())


def _make_min_spec() -> MergeVerifySpec:
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )
