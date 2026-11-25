"""Run a project's full merge gate on one named verify host through the production remote path, and record the verdict.

The operator's guide is ``docs/verify-host-preflight.md``; the design is
``plans/verify-host-roster-and-rented-trial-prd.md`` §6.2.  No event store
reaches this module, so a preflight is never trust evidence or trial data:
its record file is its only output.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from shared.safe_io import atomic_write_text

from orchestrator.verify_hosts import remote_runner_for
from orchestrator.verify_runner import (
    MergeVerifySpec,
    RemoteRunner,
    RunnerUnavailable,
    build_merge_verify_spec,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig, VerifyRunnerConfig
    from orchestrator.deployed_sha import DeployedSha
    from orchestrator.remote_gate import EnsureReport
    from orchestrator.verify import VerifyResult

#: The archive key a preflight verifies under: remote archives and records share data/verify-logs/preflight/.
PREFLIGHT_ARCHIVE_KEY = 'preflight'


class PreflightRefused(Exception):
    """The preflight cannot run as asked; raised before any runner is built or dispatched."""


class PreflightRunner(Protocol):
    """What a preflight calls on a verify host: RemoteRunner's own dispatch surface."""

    name: str

    async def ensure_gate(self) -> EnsureReport: ...

    async def run_merge_verify(
        self, merge_sha: str, spec: MergeVerifySpec, *, task_id: str, archive_root: Path,
    ) -> VerifyResult: ...


BuildRunner = Callable[['VerifyRunnerConfig'], PreflightRunner]


def full_gate_spec(config: OrchestratorConfig) -> MergeVerifySpec:
    """The lane's merge-verify spec over the whole module registry, at full breadth, unscoped."""
    spec = build_merge_verify_spec(config, list(config.module_configs_or_empty.values()), None)
    return dataclasses.replace(spec, merge_verify_breadth='full')


def resolve_commit(project_root: Path, rev: str) -> str:
    """The full sha of the commit *rev* names in *project_root*; PreflightRefused when it names none."""
    completed = subprocess.run(
        ['git', '-C', str(project_root), 'rev-parse', '--verify', '--quiet',
         '--end-of-options', f'{rev}^{{commit}}'],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise PreflightRefused(
            f'{rev!r} names no commit in {project_root}: {completed.stderr.strip() or "unknown revision"}'
        )
    return completed.stdout.strip()


def refuse_persistent_merge_worktree(config: OrchestratorConfig) -> None:
    """PreflightRefused when the project reuses one merge worktree, which a preflight beside the lane would share."""
    if config.git.persistent_merge_worktree:
        raise PreflightRefused(
            'git.persistent_merge_worktree is on in this project config: a preflight '
            'verify would share the lane\'s one merge worktree, so it is refused'
        )


def preflight_targets(
    config: OrchestratorConfig, runner: str, reference: str | None,
) -> tuple[VerifyRunnerConfig, VerifyRunnerConfig | None]:
    """The declared runner configs for *runner* and *reference*, enabled or not, each one a usable target."""
    primary = _declared(config, runner)
    if reference == runner:
        raise PreflightRefused(f'the reference runner must differ from the runner, both are {runner!r}')
    secondary = None if reference is None else _declared(config, reference)
    for target in (primary, secondary):
        if target is not None:
            _require_usable(target)
    return primary, secondary


def _declared(config: OrchestratorConfig, name: str) -> VerifyRunnerConfig:
    found = next((r for r in config.verify_runners if r.name == name), None)
    if found is None:
        raise PreflightRefused(
            f'{name!r} is not a declared verify runner; declared: {[r.name for r in config.verify_runners]!r}'
        )
    return found


def _require_usable(target: VerifyRunnerConfig) -> None:
    try:
        target.gate_paths()
    except ValueError as exc:
        raise PreflightRefused(f'verify runner {target.name!r} cannot be gated: {exc}') from exc
    if target.name in ('', '.', '..') or '/' in target.name:
        raise PreflightRefused(
            f'verify runner name {target.name!r} is not one path segment, so it cannot name a record file'
        )


@dataclass(frozen=True)
class ModuleOutcome:
    """One dispatched module's failing junit ids; None when the verdict carries no attribution."""

    failing_test_ids: tuple[str, ...] | None

    @property
    def failing_count(self) -> int | None:
        return None if self.failing_test_ids is None else len(self.failing_test_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            'failing_test_ids': _listed(self.failing_test_ids),
            'failing_count': self.failing_count,
        }


@dataclass(frozen=True)
class NoVerdict:
    """Why a runner produced no verdict: its RunnerUnavailable kind and message."""

    kind: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {'kind': self.kind, 'message': self.message}


@dataclass(frozen=True)
class RunnerVerdict:
    """What one runner's preflight verify answered, as its record states it."""

    runner: str
    sha: str
    gate: DeployedSha
    passed: bool
    wall_secs: float
    ensure_gate_secs: float | None
    gate_built: bool | None
    summary: str
    category: str
    trivial: bool
    no_verdict: NoVerdict | None
    modules: Mapping[str, ModuleOutcome]
    failing_test_ids: tuple[str, ...] | None
    skipped_test_ids: tuple[str, ...] | None

    def to_dict(self) -> dict[str, object]:
        return {
            'runner': self.runner,
            'sha': self.sha,
            'gate': self.gate.to_wire(),
            'passed': self.passed,
            'wall_secs': self.wall_secs,
            'ensure_gate_secs': self.ensure_gate_secs,
            'gate_built': self.gate_built,
            'summary': self.summary,
            'category': self.category,
            'trivial': self.trivial,
            'no_verdict': None if self.no_verdict is None else self.no_verdict.to_dict(),
            'modules': {prefix: outcome.to_dict() for prefix, outcome in self.modules.items()},
            'failing_test_ids': _listed(self.failing_test_ids),
            'skipped_test_ids': _listed(self.skipped_test_ids),
        }


@dataclass(frozen=True)
class PreflightOutcome:
    """Every verdict one preflight produced, and the record files it wrote."""

    primary: RunnerVerdict
    record_paths: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        return self.primary.passed


def _listed(ids: tuple[str, ...] | None) -> list[str] | None:
    return None if ids is None else list(ids)


def _tupled(ids: list[str] | None) -> tuple[str, ...] | None:
    return None if ids is None else tuple(ids)


@dataclass(frozen=True)
class _Ensured:
    secs: float
    built: bool


@dataclass(frozen=True)
class _Dispatch:
    """One commit's full-gate dispatch, the same for every runner it is sent to."""

    sha: str
    spec: MergeVerifySpec
    archive_root: Path
    gate: DeployedSha
    clock: Callable[[], float]

    async def on(self, name: str, runner: PreflightRunner) -> RunnerVerdict:
        """*runner*'s verdict; a RunnerUnavailable becomes a no-verdict record, anything else propagates."""
        start = self.clock()
        ensured: _Ensured | None = None
        try:
            report = await runner.ensure_gate()
            ensured = _Ensured(secs=self.clock() - start, built=report.built)
            result = await runner.run_merge_verify(
                self.sha, self.spec, task_id=PREFLIGHT_ARCHIVE_KEY, archive_root=self.archive_root,
            )
        except RunnerUnavailable as exc:
            return self._unavailable(
                name, NoVerdict(kind=exc.kind, message=str(exc)),
                wall_secs=self.clock() - start, ensured=ensured,
            )
        return self._verdict(name, result, wall_secs=self.clock() - start, ensured=ensured)

    def _verdict(
        self, name: str, result: VerifyResult, *, wall_secs: float, ensured: _Ensured,
    ) -> RunnerVerdict:
        return RunnerVerdict(
            runner=name, sha=self.sha, gate=self.gate,
            passed=result.passed and not result.trivial,
            wall_secs=wall_secs, ensure_gate_secs=ensured.secs, gate_built=ensured.built,
            summary=result.summary, category=result.category, trivial=result.trivial,
            no_verdict=None,
            modules=self._modules(result.failing_test_ids_by_module),
            failing_test_ids=_tupled(result.failing_test_ids),
            skipped_test_ids=_tupled(result.skipped_test_ids),
        )

    def _unavailable(
        self, name: str, no_verdict: NoVerdict, *, wall_secs: float, ensured: _Ensured | None,
    ) -> RunnerVerdict:
        return RunnerVerdict(
            runner=name, sha=self.sha, gate=self.gate,
            passed=False,
            wall_secs=wall_secs,
            ensure_gate_secs=None if ensured is None else ensured.secs,
            gate_built=None if ensured is None else ensured.built,
            summary='', category='', trivial=False,
            no_verdict=no_verdict,
            modules=self._modules(None),
            failing_test_ids=None,
            skipped_test_ids=None,
        )

    def _modules(self, by_module: dict[str, list[str]] | None) -> dict[str, ModuleOutcome]:
        return {
            vc.prefix: ModuleOutcome(
                None if by_module is None else tuple(by_module.get(vc.prefix, ())),
            )
            for vc in self.spec.verify_commands
        }


async def run_preflight(
    config: OrchestratorConfig,
    *,
    runner: str,
    sha: str | None,
    reference: str | None,
    deployed_sha: DeployedSha,
    build_runner: BuildRunner,
    clock: Callable[[], float] = time.monotonic,
) -> PreflightOutcome:
    """Verify one commit (default: the main tip) at full breadth on *runner*, and record the verdict.

    Every PreflightRefused is raised before any runner is built or awaited.
    """
    refuse_persistent_merge_worktree(config)
    primary_config, _reference_config = preflight_targets(config, runner, reference)
    dispatch = _Dispatch(
        sha=resolve_commit(config.project_root, sha or config.git.main_branch),
        spec=full_gate_spec(config),
        archive_root=_archive_root(config),
        gate=deployed_sha,
        clock=clock,
    )
    primary = await dispatch.on(primary_config.name, build_runner(primary_config))
    return PreflightOutcome(primary=primary, record_paths=(_write_record(config, primary),))


def production_runner_builder(config: OrchestratorConfig, deployed_sha: DeployedSha) -> BuildRunner:
    """Build each preflight runner exactly as the harness builds a remote verify runner."""
    def build(runner_config: VerifyRunnerConfig) -> RemoteRunner:
        return remote_runner_for(config, runner_config, config.project_root, deployed_sha=deployed_sha)
    return build


def _archive_root(config: OrchestratorConfig) -> Path:
    return config.project_root / 'data' / 'verify-logs'


def record_path(config: OrchestratorConfig, runner: str, sha: str) -> Path:
    """Where the preflight record of *runner* verifying *sha* is written."""
    return _archive_root(config) / PREFLIGHT_ARCHIVE_KEY / f'{runner}-{sha}.json'


def _write_record(config: OrchestratorConfig, verdict: RunnerVerdict) -> Path:
    path = record_path(config, verdict.runner, verdict.sha)
    atomic_write_text(path, json.dumps(verdict.to_dict(), indent=2, sort_keys=True) + '\n', mkdir=True)
    return path
