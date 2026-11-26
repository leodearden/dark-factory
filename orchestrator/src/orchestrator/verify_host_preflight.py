"""Run a project's full merge gate on one named verify host through the production remote path, and record the verdict.

The operator's guide is ``docs/verify-host-preflight.md``; the design is
``plans/verify-host-roster-and-rented-trial-prd.md`` §6.2.  No event store
reaches this module, so a preflight is never trust evidence or trial data:
its record file is its only output.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import yaml
from shared.governed_exceptions import (
    MalformedDisposition,
    TaskRef,
    TicketRef,
    owner_from_record,
    owner_record,
)
from shared.project_layout import verify_logs_dir
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
    from orchestrator.flake_ledger import FlakeSuppression
    from orchestrator.remote_gate import EnsureReport, RunnerUnavailableKind
    from orchestrator.verify import VerifyResult

#: The archive key a preflight verifies under: remote archives and records share data/verify-logs/preflight/.
PREFLIGHT_ARCHIVE_KEY = 'preflight'

#: A project's owned skip-parity exceptions, relative to its project_root; absent means none.
SKIP_PARITY_EXCEPTIONS_RELPATH = Path('config/verify-host-skip-parity-exceptions.yaml')


class PreflightRefused(Exception):
    """The preflight cannot run as asked; raised before any runner is built or dispatched."""


class PreflightRunner(Protocol):
    """What a preflight calls on a verify host: RemoteRunner's own dispatch surface."""

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
            'git.persistent_merge_worktree is on in this project config; a preflight '
            'runs beside the lane only in an ephemeral worktree, so it is refused'
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
class SkipParityException:
    """A junit test id that may be skipped on one runner and run on the other, and who removes it."""

    test_id: str
    owner: TaskRef | TicketRef
    reason: str


class SkipParityExceptionsError(ValueError):
    """The skip-parity exceptions file is malformed; the message names the file and the entry."""


def load_skip_parity_exceptions(path: Path) -> tuple[SkipParityException, ...]:
    """Every exception in *path*, each owned and explained; an absent file holds none."""
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return ()
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SkipParityExceptionsError(f'{path}: not YAML: {exc}') from exc
    entries = document.get('exceptions') if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise SkipParityExceptionsError(f'{path}: the top level must be a mapping holding an `exceptions` list')
    loaded = tuple(_skip_parity_exception(path, index, entry) for index, entry in enumerate(entries))
    seen: set[str] = set()
    for exception in loaded:
        if exception.test_id in seen:
            raise SkipParityExceptionsError(f'{path}: test_id {exception.test_id!r} is listed twice')
        seen.add(exception.test_id)
    return loaded


def _skip_parity_exception(path: Path, index: int, entry: object) -> SkipParityException:
    if not isinstance(entry, dict):
        raise SkipParityExceptionsError(f'{path}: entry {index} is not a mapping')
    test_id = entry.get('test_id')
    if not isinstance(test_id, str) or not test_id.strip():
        raise SkipParityExceptionsError(f'{path}: entry {index} has no test_id')
    if 'owner' not in entry:
        raise SkipParityExceptionsError(
            f'{path}: entry {test_id!r} has no owner; every exception names the task or ticket '
            'that will remove it (INV-12)'
        )
    try:
        owner = owner_from_record(entry['owner'])
    except MalformedDisposition as exc:
        raise SkipParityExceptionsError(f'{path}: entry {test_id!r} has a malformed owner: {exc}') from exc
    reason = entry.get('reason')
    if not isinstance(reason, str) or not reason.strip():
        raise SkipParityExceptionsError(f'{path}: entry {test_id!r} has no reason')
    return SkipParityException(test_id=test_id, owner=owner, reason=reason.strip())


@dataclass(frozen=True)
class ModuleOutcome:
    """One dispatched module's failing junit ids; None when unattributed: the verdict has no map, or this module collected no junit."""

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

    kind: RunnerUnavailableKind
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
    flake_suppression: FlakeSuppression | None

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
            'flake_suppression': _suppression_dict(self.flake_suppression),
        }

    @property
    def comparable_skips(self) -> frozenset[str] | None:
        """The skipped ids a parity check may compare; None when a no-verdict or no junit licenses no comparison."""
        if self.no_verdict is not None or self.skipped_test_ids is None:
            return None
        return frozenset(self.skipped_test_ids)


@dataclass(frozen=True)
class SkipParity:
    """How *runner*'s skipped junit ids differ from *reference*'s on the same commit."""

    runner: str
    reference: str
    skipped_only_on_runner: tuple[str, ...]
    skipped_only_on_reference: tuple[str, ...]
    excepted: tuple[SkipParityException, ...]
    undetermined: str | None

    @property
    def unexplained(self) -> tuple[str, ...]:
        """The differing ids no owned exception covers."""
        covered = {exception.test_id for exception in self.excepted}
        differing = {*self.skipped_only_on_runner, *self.skipped_only_on_reference}
        return tuple(sorted(differing - covered))

    @property
    def holds(self) -> bool:
        return self.undetermined is None and not self.unexplained

    def to_dict(self) -> dict[str, object]:
        return {
            'runner': self.runner,
            'reference': self.reference,
            'skipped_only_on_runner': list(self.skipped_only_on_runner),
            'skipped_only_on_reference': list(self.skipped_only_on_reference),
            'excepted': [
                {'test_id': e.test_id, 'owner': owner_record(e.owner), 'reason': e.reason}
                for e in self.excepted
            ],
            'unexplained': list(self.unexplained),
            'undetermined': self.undetermined,
            'holds': self.holds,
        }


def compare_skips(
    runner: RunnerVerdict, reference: RunnerVerdict, exceptions: Sequence[SkipParityException],
) -> SkipParity:
    """The skip parity of *runner* against *reference*, failing closed when either vector is missing."""
    ours, theirs = runner.comparable_skips, reference.comparable_skips
    if ours is None or theirs is None:
        return SkipParity(
            runner=runner.runner, reference=reference.runner,
            skipped_only_on_runner=(), skipped_only_on_reference=(),
            excepted=(), undetermined=_incomparable(runner, reference),
        )
    differing = ours ^ theirs
    return SkipParity(
        runner=runner.runner,
        reference=reference.runner,
        skipped_only_on_runner=tuple(sorted(ours - theirs)),
        skipped_only_on_reference=tuple(sorted(theirs - ours)),
        excepted=tuple(e for e in exceptions if e.test_id in differing),
        undetermined=None,
    )


def _incomparable(*verdicts: RunnerVerdict) -> str:
    return '; '.join(
        f'{v.runner} produced no verdict ({v.no_verdict.kind})' if v.no_verdict is not None
        else f'{v.runner} collected no junit skip vector'
        for v in verdicts
        if v.comparable_skips is None
    )


@dataclass(frozen=True)
class PreflightOutcome:
    """Every verdict one preflight produced, its skip parity, and the record files it wrote."""

    primary: RunnerVerdict
    reference: RunnerVerdict | None
    parity: SkipParity | None
    record_paths: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        """The runner passed and, against a reference, skip parity holds."""
        return self.primary.passed and (self.parity is None or self.parity.holds)


def _listed(ids: tuple[str, ...] | None) -> list[str] | None:
    return None if ids is None else list(ids)


def _tupled(ids: list[str] | None) -> tuple[str, ...] | None:
    return None if ids is None else tuple(ids)


def _suppression_dict(suppression: FlakeSuppression | None) -> dict[str, object] | None:
    if suppression is None:
        return None
    return {
        'verdict': str(suppression.verdict),
        'test_ids': list(suppression.test_ids),
        'psi_cpu_some10': suppression.psi_cpu_some10,
    }


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
            flake_suppression=result.flake_suppression,
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
            flake_suppression=None,
        )

    def _modules(self, by_module: dict[str, list[str]] | None) -> dict[str, ModuleOutcome]:
        return {
            vc.prefix: ModuleOutcome(
                None if by_module is None or vc.prefix not in by_module else tuple(by_module[vc.prefix]),
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

    With *reference*, the same commit is verified on that runner concurrently
    and the two skip vectors are compared under the project's owned
    exceptions.  Every PreflightRefused, and a malformed exceptions file's
    SkipParityExceptionsError, is raised before any runner is built or awaited.
    """
    refuse_persistent_merge_worktree(config)
    primary_config, reference_config = preflight_targets(config, runner, reference)
    commit = resolve_commit(config.project_root, sha or config.git.main_branch)
    exceptions = (
        () if reference_config is None
        else load_skip_parity_exceptions(config.project_root / SKIP_PARITY_EXCEPTIONS_RELPATH)
    )
    dispatch = _Dispatch(
        sha=commit, spec=full_gate_spec(config), archive_root=_archive_root(config),
        gate=deployed_sha, clock=clock,
    )
    targets = [t for t in (primary_config, reference_config) if t is not None]
    primary, *others = await _verify_all(dispatch, targets, build_runner)
    reference_verdict = others[0] if others else None
    parity = None if reference_verdict is None else compare_skips(primary, reference_verdict, exceptions)
    return PreflightOutcome(
        primary=primary, reference=reference_verdict, parity=parity,
        record_paths=tuple(_write_record(config, v, parity) for v in (primary, *others)),
    )


async def _verify_all(
    dispatch: _Dispatch, targets: Sequence[VerifyRunnerConfig], build_runner: BuildRunner,
) -> list[RunnerVerdict]:
    runners = [(target.name, build_runner(target)) for target in targets]
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(dispatch.on(name, runner)) for name, runner in runners]
    return [task.result() for task in tasks]


def production_runner_builder(config: OrchestratorConfig, deployed_sha: DeployedSha) -> BuildRunner:
    """Build each preflight runner exactly as the harness builds a remote verify runner."""
    def build(runner_config: VerifyRunnerConfig) -> RemoteRunner:
        return remote_runner_for(config, runner_config, config.project_root, deployed_sha=deployed_sha)
    return build


def render_summary(outcome: PreflightOutcome) -> str:
    """A few human lines for *outcome*; the record files are the machine-readable form."""
    lines = [*_verdict_lines(outcome.primary)]
    if outcome.reference is not None:
        lines.extend(_verdict_lines(outcome.reference))
    if outcome.parity is not None:
        lines.append(_parity_line(outcome.parity))
    lines.extend(f'record: {path}' for path in outcome.record_paths)
    return '\n'.join(lines)


def _verdict_lines(verdict: RunnerVerdict) -> list[str]:
    status = 'passed' if verdict.passed else 'NOT passed'
    ensure = 'n/a' if verdict.ensure_gate_secs is None else f'{verdict.ensure_gate_secs:.1f}s'
    lines = [f'{verdict.runner} @ {verdict.sha}: {status} (wall {verdict.wall_secs:.1f}s, ensure_gate {ensure})']
    if verdict.no_verdict is not None:
        lines.append(f'  no verdict ({verdict.no_verdict.kind}): {verdict.no_verdict.message}')
    if verdict.trivial:
        lines.append('  trivial pass: the remote ran nothing, so it is not a pass')
    lines.extend(
        f'  {prefix}: ' + (
            'failing ids unattributed' if outcome.failing_count is None
            else f'{outcome.failing_count} failing'
        )
        for prefix, outcome in verdict.modules.items()
    )
    skipped = 'unknown' if verdict.skipped_test_ids is None else len(verdict.skipped_test_ids)
    lines.append(f'  skipped: {skipped}')
    return lines


def _parity_line(parity: SkipParity) -> str:
    if parity.undetermined is not None:
        return f'skip parity {parity.runner} vs {parity.reference}: undetermined ({parity.undetermined})'
    if parity.holds:
        return f'skip parity {parity.runner} vs {parity.reference}: holds ({len(parity.excepted)} owned exceptions applied)'
    return f'skip parity {parity.runner} vs {parity.reference}: broken, unexplained {", ".join(parity.unexplained)}'


def _archive_root(config: OrchestratorConfig) -> Path:
    return verify_logs_dir(config.project_root)


def record_path(config: OrchestratorConfig, runner: str, sha: str) -> Path:
    """Where the preflight record of *runner* verifying *sha* is written."""
    return _archive_root(config) / PREFLIGHT_ARCHIVE_KEY / f'{runner}-{sha}.json'


def record_json(verdict: RunnerVerdict, parity: SkipParity | None) -> str:
    """The JSON record of *verdict*, carrying the preflight's skip parity (null without a reference)."""
    record = {**verdict.to_dict(), 'skip_parity': None if parity is None else parity.to_dict()}
    return json.dumps(record, indent=2, sort_keys=True)


def _write_record(config: OrchestratorConfig, verdict: RunnerVerdict, parity: SkipParity | None) -> Path:
    path = record_path(config, verdict.runner, verdict.sha)
    atomic_write_text(path, record_json(verdict, parity) + '\n', mkdir=True)
    return path
