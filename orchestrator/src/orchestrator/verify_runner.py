"""verify_runner — host-independent MergeVerifySpec + VerifyResult JSON codec.

This module defines the data-contract types for multi-host merge verification
(PRD: plans/merge-throughput-multihost-verify-prd.md §A) and the wire codec
that serialises/deserialises them as byte-identical JSON documents.

Public API
----------
Dataclasses (frozen):
  VerifyCommand          — one module's command-bearing fields (mirrors ModuleConfig)
  UnscopedTypecheckSpec  — _run_unscoped_typechecks gate inputs
  MergeVerifySpec        — full pre-advance merge-verify bundle (PRD §A contract)

Wire codec (module-level functions):
  spec_to_json(spec)     -> str   — canonical JSON for the spec document
  spec_from_json(s)      -> MergeVerifySpec
  result_to_json(vr)     -> str   — canonical JSON for a VerifyResult document
  result_from_json(s)    -> VerifyResult
  result_to_dict(vr)     -> dict  — helper used by result_to_json
  result_from_dict(d)    -> VerifyResult

Byte-identity guarantee (PRD Invariant 1)
-----------------------------------------
All JSON is emitted with ``json.dumps(sort_keys=True, ensure_ascii=False)``.
This canonicalises dict key order so that:
  - ``to_json(from_json(s)) == s``  (re-serialisation is identical)
  - The same object always serialises to the same bytes regardless of dict
    construction order (proved by sort_keys).
Achievability is by construction: every field maps to native JSON scalar /
container types only (str, bool, float, list, dict, None); no float-precision
rounding or numeric tolerance is involved.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import shlex
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from orchestrator.config import ModuleConfig
from orchestrator.verify import VerifyResult

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

__all__ = [
    "VerifyCommand",
    "UnscopedTypecheckSpec",
    "MergeVerifySpec",
    "spec_to_json",
    "spec_from_json",
    "result_to_json",
    "result_from_json",
    "result_to_dict",
    "result_from_dict",
    "VerifyRunner",
    "LocalRunner",
    "RemoteRunner",
    "RunnerUnavailable",
    "VerifyRunnerPool",
    "build_merge_verify_spec",
    "_module_config_from_command",
    "run_merge_verify_on_worktree",
    "UNSCOPED_TYPECHECK_FAILED_CATEGORY",
    "UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY",
    "is_unscoped_gate_failure",
    "unscoped_gate_failing_subprojects",
]

# Sentinel category constants — encode an unscoped-gate failure inside a
# VerifyResult so _run_post_merge_verify can branch byte-identically.
UNSCOPED_TYPECHECK_FAILED_CATEGORY = 'unscoped_typecheck_failed'
UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY = 'unscoped_typecheck_timeout'

_UNSCOPED_SENTINEL_CATEGORIES = frozenset({
    UNSCOPED_TYPECHECK_FAILED_CATEGORY,
    UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
})


# ---------------------------------------------------------------------------
# VerifyCommand
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyCommand:
    """Frozen projection of a (scoped) ModuleConfig's command-bearing fields.

    Mirrors ``ModuleConfig.prefix`` + the three command fields.  Timeouts and
    environment variables live at ``MergeVerifySpec`` level (shared across the
    bundle), so they are NOT duplicated here.
    """

    prefix: str
    test_command: str | None = None
    lint_command: str | None = None
    type_check_command: str | None = None

    def to_dict(self) -> dict:
        return {
            "prefix": self.prefix,
            "test_command": self.test_command,
            "lint_command": self.lint_command,
            "type_check_command": self.type_check_command,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerifyCommand:
        return cls(
            prefix=d["prefix"],
            test_command=d.get("test_command"),
            lint_command=d.get("lint_command"),
            type_check_command=d.get("type_check_command"),
        )


# ---------------------------------------------------------------------------
# UnscopedTypecheckSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnscopedTypecheckSpec:
    """Inputs for the ``_run_unscoped_typechecks`` pre-advance gate.

    ``commands`` holds one ``VerifyCommand`` per module; only
    ``type_check_command`` is meaningful here (the other fields are carried
    along for consistent structure but are not used by the gate).

    ``block_on_timeout=True`` (the default) mirrors the fail-closed policy
    used at ``merge_queue.py:1565``; set to ``False`` only in tests or when
    the caller explicitly opts out of fail-closed behaviour.
    """

    commands: tuple[VerifyCommand, ...]
    block_on_timeout: bool = True

    def to_dict(self) -> dict:
        return {
            "commands": [vc.to_dict() for vc in self.commands],
            "block_on_timeout": self.block_on_timeout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UnscopedTypecheckSpec:
        return cls(
            commands=tuple(VerifyCommand.from_dict(vc) for vc in d["commands"]),
            block_on_timeout=d.get("block_on_timeout", True),
        )


# ---------------------------------------------------------------------------
# MergeVerifySpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeVerifySpec:
    """Full pre-advance merge-verify bundle — PRD §A contract.

    Fields
    ------
    verify_commands     : one VerifyCommand per module, scoped to task_files
    unscoped_typecheck  : the _run_unscoped_typechecks gate spec
    task_files          : files in the merge commit (None → full verify)
    verify_env          : environment overrides (RUSTC_WRAPPER, CARGO_INCREMENTAL, …)
    cold_timeout_secs   : merge_verify_cold cascade timeout
    is_merge_verify     : always True for merge-path specs (default)

    Note
    ----
    ``verify_env`` is stored as a mutable ``dict`` at runtime.  Although
    ``frozen=True`` prevents attribute reassignment, calling ``hash()`` on a
    ``MergeVerifySpec`` instance raises ``TypeError`` because dicts are
    unhashable.  Specs are serialization value-objects and are not intended to
    be used as dict keys or set members.
    """

    verify_commands: tuple[VerifyCommand, ...]
    unscoped_typecheck: UnscopedTypecheckSpec
    task_files: tuple[str, ...] | None
    verify_env: Mapping[str, str]
    cold_timeout_secs: float
    is_merge_verify: bool = True

    def to_dict(self) -> dict:
        return {
            "verify_commands": [vc.to_dict() for vc in self.verify_commands],
            "unscoped_typecheck": self.unscoped_typecheck.to_dict(),
            "task_files": list(self.task_files) if self.task_files is not None else None,
            "verify_env": dict(self.verify_env),
            "cold_timeout_secs": float(self.cold_timeout_secs),
            "is_merge_verify": self.is_merge_verify,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MergeVerifySpec:
        task_files_raw = d.get("task_files")
        return cls(
            verify_commands=tuple(VerifyCommand.from_dict(vc) for vc in d["verify_commands"]),
            unscoped_typecheck=UnscopedTypecheckSpec.from_dict(d["unscoped_typecheck"]),
            task_files=tuple(task_files_raw) if task_files_raw is not None else None,
            verify_env=dict(d.get("verify_env", {})),
            cold_timeout_secs=float(d["cold_timeout_secs"]),
            is_merge_verify=d.get("is_merge_verify", True),
        )


# ---------------------------------------------------------------------------
# VerifyResult codec
# ---------------------------------------------------------------------------


def result_to_dict(vr: VerifyResult) -> dict:
    """Serialise a VerifyResult to a plain dict of JSON-native types.

    Uses ``dataclasses.asdict`` which recursively converts nested dataclasses
    and preserves all field types (all VerifyResult fields are JSON-native).
    """
    return dataclasses.asdict(vr)


def result_from_dict(d: dict) -> VerifyResult:
    """Reconstruct a VerifyResult from a dict (as produced by result_to_dict)."""
    return VerifyResult(**d)


# ---------------------------------------------------------------------------
# build_merge_verify_spec factory
# ---------------------------------------------------------------------------


def build_merge_verify_spec(
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    task_files: tuple[str, ...] | None,
) -> MergeVerifySpec:
    """Build a MergeVerifySpec from live config + module_configs.

    The spec is a host-independent projection carried through dispatch for
    forward-compat with γ/δ (the remote runner consumes it over the wire).
    LocalRunner does not use it to drive execution — it uses the live objects.
    """
    verify_commands = tuple(
        VerifyCommand(
            prefix=mc.prefix,
            test_command=mc.test_command,
            lint_command=mc.lint_command,
            type_check_command=mc.type_check_command,
        )
        for mc in module_configs
    )
    unscoped_commands = tuple(
        VerifyCommand(prefix=mc.prefix, type_check_command=mc.type_check_command)
        for mc in module_configs
        if mc.type_check_command is not None
    )
    cold_timeout: float = (
        config.merge_verify_cold_command_timeout_secs
        if config.merge_verify_cold_command_timeout_secs is not None
        else (
            config.verify_cold_command_timeout_secs
            if config.verify_cold_command_timeout_secs is not None
            else 0.0
        )
    )
    return MergeVerifySpec(
        verify_commands=verify_commands,
        unscoped_typecheck=UnscopedTypecheckSpec(commands=unscoped_commands, block_on_timeout=True),
        task_files=task_files,
        verify_env=dict(config.verify_env) if config.verify_env else {},
        cold_timeout_secs=float(cold_timeout),
        is_merge_verify=True,
    )


# ---------------------------------------------------------------------------
# _module_config_from_command — inverse of build_merge_verify_spec's projection
# ---------------------------------------------------------------------------


def _module_config_from_command(vc: VerifyCommand, spec: MergeVerifySpec) -> ModuleConfig:
    """Reconstruct a ModuleConfig from a VerifyCommand + shared MergeVerifySpec fields.

    This is the exact inverse of build_merge_verify_spec's projection:
    - prefix + three command fields come from the VerifyCommand
    - verify_env and cold_timeout_secs are shared spec-level fields threaded into
      each ModuleConfig's verify_env and verify_cold_command_timeout_secs

    ModuleConfig fields the wire spec never carried (lock_depth, warm timeout, etc.)
    stay at their dataclass defaults.  Reconstruction is information-preserving for
    all verify-relevant behaviour *when build_merge_verify_spec is the sole producer*:
    it serialises a single spec-level cold_timeout_secs for all modules, so if modules
    originally had distinct per-module cold timeouts the reconstruction is not lossless
    (the spec collapses them).  In practice the merge path uses one config-level value.

    cold_timeout_secs: the 0.0 wire sentinel (emitted by build_merge_verify_spec when
    neither merge_verify_cold_command_timeout_secs nor verify_cold_command_timeout_secs
    is set) maps back to None so that _resolve_verify_timeout falls through the cold
    cascade (module→top-level→warm) exactly as a real local merge run does, instead of
    returning 0.0 and triggering an immediate asyncio.wait_for(..., timeout=0.0)
    TimeoutError.  Positive cold timeouts map verbatim.
    """
    return ModuleConfig(
        prefix=vc.prefix,
        test_command=vc.test_command,
        lint_command=vc.lint_command,
        type_check_command=vc.type_check_command,
        verify_env=dict(spec.verify_env),
        verify_cold_command_timeout_secs=(
            spec.cold_timeout_secs if spec.cold_timeout_secs > 0 else None
        ),
    )


# ---------------------------------------------------------------------------
# run_merge_verify_on_worktree — host-entry for the CLI verify-merge subcommand
# ---------------------------------------------------------------------------


async def run_merge_verify_on_worktree(
    merge_wt: Path,
    config: OrchestratorConfig,
    spec: MergeVerifySpec,
    *,
    merge_sha: str = '',
    task_id: str | None = None,
    run_scoped: Callable[..., Awaitable[VerifyResult]] | None = None,
    run_unscoped: Callable[..., Awaitable[Any]] | None = None,
) -> VerifyResult:
    """Run the combined merge-verify bundle at a materialized worktree.

    Reconstructs per-module ModuleConfig objects from the wire spec, then
    delegates to LocalRunner.run_merge_verify (the same bundle the merge queue
    runs), providing fidelity by construction (PRD §A Invariant 1 / D2).

    Args:
        merge_wt: Path to the detached worktree at the merge SHA.
        config:   OrchestratorConfig for the host project.
        spec:     Deserialized MergeVerifySpec from the --spec CLI flag.
        merge_sha: The commit SHA being verified (threaded into telemetry).
        task_id:  Optional task ID for logging/telemetry.
        run_scoped:   Injected callable for scoped verification (default: real global).
        run_unscoped: Injected callable for unscoped typecheck gate (default: real global).
    """
    # Deferred imports break the merge_queue↔verify_runner module-level cycle
    # (merge_queue imports verify_runner at module level; a module-level import of
    # merge_queue here would re-introduce the cycle). Defaults are wired in step-6.
    if run_scoped is None:
        from orchestrator.verify import run_scoped_verification  # type: ignore[attr-defined]
        run_scoped = run_scoped_verification
    if run_unscoped is None:
        from orchestrator.merge_queue import _run_unscoped_typechecks  # type: ignore[attr-defined]
        run_unscoped = _run_unscoped_typechecks

    # module_configs is reconstructed from spec.verify_commands (which carries
    # type_check_command per module).  build_merge_verify_spec — the sole spec producer
    # — copies mc.type_check_command into *both* verify_commands and
    # unscoped_typecheck.commands, so the typecheck gate's module_configs are correct.
    # If any future producer emits a spec where those two lists diverge (verify_commands
    # lacks type_check_command while unscoped_typecheck.commands carries it), the unscoped
    # gate would silently become a no-op.  Adding a new spec producer must maintain that
    # invariant or reconstruct module_configs from spec.unscoped_typecheck instead.
    module_configs = [_module_config_from_command(vc, spec) for vc in spec.verify_commands]
    task_files = tuple(spec.task_files) if spec.task_files is not None else None

    runner = LocalRunner(
        merge_wt,
        config,
        module_configs,
        task_files,
        run_scoped=run_scoped,
        run_unscoped=run_unscoped,
        task_id=task_id,
    )
    return await runner.run_merge_verify(merge_sha, spec)


# ---------------------------------------------------------------------------
# RunnerUnavailable — transport failure exception
# ---------------------------------------------------------------------------


class RunnerUnavailable(Exception):
    """Raised by RemoteRunner on any transport failure.

    Transport failures include: host down/closed, ssh failure, git push failure,
    or absent/unparseable verdict on stdout.  A parseable VerifyResult is always
    returned as a result — even passed=False or timed_out=True — and never causes
    this exception to be raised (PRD §A Invariant 5).
    """


# ---------------------------------------------------------------------------
# VerifyRunner protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VerifyRunner(Protocol):
    """Host-agnostic runner that executes a combined merge-verify bundle.

    Implementations: LocalRunner (this module), RemoteRunner (γ/δ).
    """

    name: str

    async def health(self) -> bool:
        """Return True when this runner is reachable and healthy."""
        ...

    async def run_merge_verify(
        self,
        merge_sha: str,
        spec: MergeVerifySpec,
    ) -> VerifyResult:
        """Run the full combined merge-verify bundle and return a VerifyResult."""
        ...


# ---------------------------------------------------------------------------
# LocalRunner
# ---------------------------------------------------------------------------


class LocalRunner:
    """Wraps the current local verify path (run_scoped_verification + _run_unscoped_typechecks).

    The verify callables are injected at construction time so that:
    1. Existing test patches on 'orchestrator.merge_queue.run_scoped_verification'
       keep intercepting (call-time resolution, not import-time binding).
    2. There is no verify_runner → merge_queue module-level import cycle.

    ``run_merge_verify`` runs the combined scoped + unscoped bundle: scoped first,
    unscoped only if scoped passed (short-circuit), unscoped-gate-broken outcomes
    encoded as a sentinel-category VerifyResult so callers can branch byte-identically.
    """

    name: str = 'local'

    def __init__(
        self,
        merge_wt: Path,
        config: OrchestratorConfig,
        module_configs: list[ModuleConfig],
        task_files: tuple[str, ...] | None,
        *,
        run_scoped: Callable[..., Awaitable[VerifyResult]],
        run_unscoped: Callable[..., Awaitable[Any]],
        task_id: str | None = None,
    ) -> None:
        self._merge_wt = merge_wt
        self._config = config
        self._module_configs = module_configs
        self._task_files = task_files
        self._run_scoped = run_scoped
        self._run_unscoped = run_unscoped
        self._task_id = task_id

    async def health(self) -> bool:
        return True

    async def run_merge_verify(
        self,
        merge_sha: str,
        spec: MergeVerifySpec,
    ) -> VerifyResult:
        """Run the combined scoped + unscoped bundle.

        Scoped phase runs first; unscoped gate only runs if scoped passed
        (preserving today's short-circuit). An unscoped-gate-broken outcome is
        encoded into a VerifyResult via a sentinel category so callers can branch
        byte-identically.

        NOTE: ``spec`` is accepted for VerifyRunner protocol conformance and
        forward-compat with γ/δ remote runners.  LocalRunner drives execution
        from its injected callables + live config, not from the spec.
        TODO(γ): when a RemoteRunner is added, spec replaces the per-call
        config/module_configs projection for off-host dispatch.
        """
        scoped = await self._run_scoped(
            self._merge_wt,
            self._config,
            self._module_configs,
            task_files=self._task_files,
            max_retries=0,
            is_merge_verify=True,
            force_workspace=self._config.merge_verify_workspace,
            role='merge',
        )
        if not scoped.passed:
            return scoped

        gate = await self._run_unscoped(
            self._merge_wt,
            self._config,
            self._module_configs,
            block_on_timeout=True,
            task_id=self._task_id,
        )
        if gate.broken:
            failing = gate.failing_subprojects
            timed_out = bool(gate.timed_out_subprojects)
            category = (
                UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY
                if timed_out
                else UNSCOPED_TYPECHECK_FAILED_CATEGORY
            )
            summary = ', '.join(failing)
            return VerifyResult(
                passed=False,
                test_output='',
                lint_output='',
                type_output=gate.detail if hasattr(gate, 'detail') else '',
                summary=summary,
                timed_out=timed_out,
                category=category,
            )

        return scoped


# ---------------------------------------------------------------------------
# RemoteRunner — off-host verify via git push + ssh
# ---------------------------------------------------------------------------


async def _default_subprocess_run(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
) -> tuple[int, str, str]:
    """Default subprocess helper — mirrors git_ops._run.

    Returns (returncode, stdout_str, stderr_str).
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout_b.decode().strip(),
        stderr_b.decode().strip(),
    )


class RemoteRunner:
    """Runs a merge-verify bundle on a remote host via git push + ssh.

    The remote host must have ``orchestrator verify-merge`` available (γ CLI
    contract).  The spec is shipped over ssh as a shlex-quoted JSON argument;
    stdout is parsed as a VerifyResult via result_from_json.

    Transport failures (push fail, ssh connect failure, non-zero exit,
    unparseable stdout) raise RunnerUnavailable.  A parseable VerifyResult on
    stdout is always returned unchanged — even passed=False or timed_out=True
    (PRD §A Invariant 5).

    The pushed ref ``refs/merge-verify/<request_id>`` is pruned best-effort on
    return (step-10).
    """

    def __init__(
        self,
        name: str,
        ssh_host: str,
        git_remote: str,
        cwd: str | Path,
        *,
        config_path: str | None = None,
        run: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.name = name
        self._ssh_host = ssh_host
        self._git_remote = git_remote
        self._cwd = cwd
        self._config_path = config_path
        self._run = run if run is not None else _default_subprocess_run
        self._id_factory = id_factory if id_factory is not None else (lambda: uuid.uuid4().hex)
        # Test-only injection point: tests may assign a list here to capture subprocess calls.
        self._calls: list = []

    async def health(self) -> bool:
        """Best-effort health probe: ``ssh <host> true``.

        Returns True when rc == 0, False otherwise.  Never raises.
        """
        try:
            rc, _, _ = await self._run(['ssh', self._ssh_host, 'true'])
            return rc == 0
        except Exception:
            return False

    async def run_merge_verify(
        self,
        merge_sha: str,
        spec: MergeVerifySpec,
    ) -> VerifyResult:
        """Run the combined merge-verify bundle on the remote host.

        (a) git push <git_remote> <merge_sha>:refs/merge-verify/<request_id>
        (b) ssh <ssh_host> <shlex-quoted remote argv>
        (c) parse stdout via result_from_json

        Raises RunnerUnavailable on any transport failure (step-8).
        Returns a VerifyResult unchanged — even passed=False or timed_out=True
        (PRD §A Invariant 5).
        """
        request_id = self._id_factory()
        ref = f'refs/merge-verify/{request_id}'

        # Step 1: push the merge sha to the remote
        try:
            push_rc, _push_out, push_stderr = await self._run(
                ['git', 'push', self._git_remote, f'{merge_sha}:{ref}'],
                cwd=self._cwd,
            )
        except OSError as exc:
            raise RunnerUnavailable(f'git push spawn failed: {exc}') from exc

        if push_rc != 0:
            raise RunnerUnavailable(
                f'git push {self._git_remote} {merge_sha}:{ref} failed'
                f' (rc={push_rc}): {push_stderr}'
            )

        # Push succeeded — clean up the ref on return (best-effort, in finally)
        try:
            # Step 2: build and issue the ssh command
            argv = [
                'orchestrator', 'verify-merge',
                '--sha', merge_sha,
                '--spec', spec_to_json(spec),
            ]
            if self._config_path:
                argv += ['--config', self._config_path]
            remote_cmd = ' '.join(shlex.quote(a) for a in argv)

            try:
                ssh_rc, ssh_stdout, ssh_stderr = await self._run(
                    ['ssh', self._ssh_host, remote_cmd],
                )
            except OSError as exc:
                raise RunnerUnavailable(f'ssh spawn failed: {exc}') from exc

            if ssh_rc != 0:
                raise RunnerUnavailable(
                    f'ssh {self._ssh_host} exited {ssh_rc}: {ssh_stderr}'
                )

            # Step 3: parse the host's stdout
            # Any parseable VerifyResult is returned unchanged (PRD §A Invariant 5).
            # Non-zero exit or unparseable stdout → RunnerUnavailable (transport failure).
            try:
                return result_from_json(ssh_stdout)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise RunnerUnavailable(
                    f'unparseable VerifyResult from {self._ssh_host!r}: {exc!r}'
                ) from exc

        finally:
            # Best-effort ref cleanup — never alters the returned result nor masks exceptions
            try:
                await self._run(
                    ['git', 'push', self._git_remote, '--delete', ref],
                    cwd=self._cwd,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers for merge_queue to branch on unscoped-gate verdicts
# ---------------------------------------------------------------------------


def is_unscoped_gate_failure(vr: VerifyResult) -> bool:
    """True when vr carries a sentinel category from LocalRunner's unscoped gate."""
    return vr.category in _UNSCOPED_SENTINEL_CATEGORIES


def unscoped_gate_failing_subprojects(vr: VerifyResult) -> list[str]:
    """Extract the failing subproject prefixes from a sentinel VerifyResult.

    Returns the list encoded in vr.summary (comma-joined prefixes).
    """
    if not vr.summary:
        return []
    return [p.strip() for p in vr.summary.split(',') if p.strip()]


# ---------------------------------------------------------------------------
# VerifyRunnerPool
# ---------------------------------------------------------------------------


class VerifyRunnerPool:
    """Dispatches a merge-verify bundle to a VerifyRunner and emits a telemetry event.

    Selection policy (δ): prefers the first non-local (remote) runner.
    The K-permit free/busy refinement (semaphore-based concurrency control) is
    deferred to ζ.

    Fail-safe (δ): if the selected runner raises RunnerUnavailable, dispatch
    falls back to the local runner (if distinct), logging one warning.  The
    merge_verify event reflects the runner that actually produced the result.
    dispatch() never propagates RunnerUnavailable to its caller when a local
    fallback exists (PRD §A Invariant 2).
    """

    def __init__(
        self,
        runners: Sequence[VerifyRunner],
        *,
        event_store: Any = None,
        task_id: str | None = None,
    ) -> None:
        if not runners:
            raise ValueError('VerifyRunnerPool requires at least one runner')
        self._runners = list(runners)
        # Pre-compute the local runner for fast fail-safe lookup
        self._local: VerifyRunner | None = next(
            (r for r in self._runners if r.name == 'local'), None
        )
        self._event_store = event_store
        self._task_id = task_id

    def _select_runner(self) -> VerifyRunner:
        """Prefer-remote: return the first non-local runner; fall back to runners[0].

        The K-permit free/busy refinement (load-based selection) is ζ.
        """
        for runner in self._runners:
            if runner.name != 'local':
                return runner
        return self._runners[0]

    async def dispatch(
        self,
        merge_sha: str,
        spec: MergeVerifySpec,
        *,
        attempt: int = 0,
    ) -> VerifyResult:
        """Run the verify bundle and emit a merge_verify event.

        ``attempt`` is 0 for the first dispatch and incremented for each
        ENOSPC-retry re-dispatch.  Included in the event data so consumers
        can deduplicate multiple events for the same logical merge-verify.

        Fail-safe (PRD §A Invariant 2 / D5): if the selected runner raises
        RunnerUnavailable, dispatch falls back to the local runner (if
        distinct), logging exactly one WARNING.  dispatch() never propagates
        RunnerUnavailable to its caller when a local fallback exists.  A
        returned VerifyResult (any passed/timed_out value) is passed through
        without fallback (PRD §A Invariant 5).
        """
        import logging

        from orchestrator.event_store import EventType

        _log = logging.getLogger(__name__)

        selected = self._select_runner()
        t0 = time.monotonic()
        try:
            result = await selected.run_merge_verify(merge_sha, spec)
            actual_runner = selected
        except RunnerUnavailable:
            # Fall back to the local runner if one exists and is not the
            # runner that just failed.  Log exactly one WARNING; no escalation.
            if self._local is not None and self._local is not selected:
                _log.warning(
                    'runner %r unavailable for %s — falling back to local',
                    selected.name,
                    merge_sha,
                )
                result = await self._local.run_merge_verify(merge_sha, spec)
                actual_runner = self._local
            else:
                raise
        duration_ms = round((time.monotonic() - t0) * 1000)

        if self._event_store is not None:
            self._event_store.emit(
                EventType.merge_verify,
                task_id=self._task_id,
                data={
                    'runner': actual_runner.name,
                    'merge_sha': merge_sha,
                    'passed': result.passed,
                    'duration_ms': duration_ms,
                    'attempt': attempt,
                },
            )

        return result


# ---------------------------------------------------------------------------
# Module-level wire codec
# ---------------------------------------------------------------------------


def spec_to_json(spec: MergeVerifySpec) -> str:
    """Serialise a MergeVerifySpec to canonical JSON (sort_keys=True)."""
    return json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False)


def spec_from_json(s: str) -> MergeVerifySpec:
    """Deserialise a MergeVerifySpec from a JSON string."""
    return MergeVerifySpec.from_dict(json.loads(s))


def result_to_json(vr: VerifyResult) -> str:
    """Serialise a VerifyResult to canonical JSON (sort_keys=True)."""
    return json.dumps(result_to_dict(vr), sort_keys=True, ensure_ascii=False)


def result_from_json(s: str) -> VerifyResult:
    """Deserialise a VerifyResult from a JSON string."""
    return result_from_dict(json.loads(s))
