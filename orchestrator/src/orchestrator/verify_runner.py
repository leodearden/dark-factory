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
import contextlib
import dataclasses
import json
import shlex
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

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
    # ε additions
    "EnvFingerprint",
    "fingerprint_to_json",
    "fingerprint_from_json",
    "EnvParityVerdict",
    "compare_env_fingerprints",
    "capture_env_fingerprint",
    "ParityRow",
    "VerdictParityReport",
    "parity_report_to_json",
    "parity_report_from_json",
    "run_verdict_parity",
    "render_parity_report",
    # ι additions
    "DriftVerdict",
    "DriftCheckResult",
    "DriftDetector",
    # κ additions
    "SccacheStats",
    "parse_sccache_stats",
    "capture_sccache_stats",
    "ColdWarmVerifyDelta",
    "delta_to_json",
    "delta_from_json",
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
        # κ: read effective_verify_env (the single merge rule) so the spec shipped
        # to the laptop carries the shared sccache backend even for direct-constructed
        # (eval/test) configs that never call load_config.
        verify_env=dict(config.effective_verify_env),
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
    is_local: ClassVar[bool]

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
    is_local: ClassVar[bool] = True

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
    """Default subprocess helper — similar to git_ops._run but without the WorktreeMissing pre-flight.

    Returns (returncode, stdout_str, stderr_str).  A missing ``cwd`` surfaces as
    a raw ``FileNotFoundError`` (caught by callers as OSError → RunnerUnavailable)
    rather than the ``WorktreeMissing`` sentinel that git_ops._run would raise.
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

    is_local: ClassVar[bool] = False

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
        # Optional test-instrumentation hook: tests may assign a list to this
        # attribute so they can inspect all subprocess argv lists after the fact.
        self._calls: list[list[str]] = []

    async def health(self) -> bool:
        """Best-effort health probe: ``ssh <host> true``.

        Returns True when rc == 0, False otherwise.  Never raises.
        BatchMode=yes prevents interactive password prompts; ConnectTimeout=10
        bounds the TCP-connect wait so a down host is detected quickly.
        """
        try:
            rc, _, _ = await self._run(
                ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self._ssh_host, 'true']
            )
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
                    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                     self._ssh_host, remote_cmd],
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
            with contextlib.suppress(Exception):
                await self._run(
                    ['git', 'push', self._git_remote, '--delete', ref],
                    cwd=self._cwd,
                )


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
        # Pre-compute the local runner for fast fail-safe lookup.
        # Use the is_local flag (LocalRunner.is_local = True) rather than
        # string equality so a RemoteRunner named 'local' isn't mistaken for
        # the fallback target.
        self._local: VerifyRunner | None = next(
            (r for r in self._runners if r.is_local), None
        )
        self._event_store = event_store
        self._task_id = task_id
        # ι: quarantine set — names of runners dropped from eligible dispatch.
        # Local (is_local) is the trust anchor and is never quarantined.
        self._quarantined: set[str] = set()

    # ------------------------------------------------------------------
    # ι: quarantine API
    # ------------------------------------------------------------------

    def quarantine(self, name: str) -> None:
        """Mark runner *name* as quarantined; idempotent."""
        self._quarantined.add(name)

    def clear_quarantine(self, name: str) -> None:
        """Remove runner *name* from quarantine; idempotent."""
        self._quarantined.discard(name)

    def is_quarantined(self, name: str) -> bool:
        """Return True if runner *name* is currently quarantined."""
        return name in self._quarantined

    @property
    def local_runner(self) -> VerifyRunner | None:
        """The is_local runner, or None if not present."""
        return self._local

    def eligible_remote(self) -> VerifyRunner | None:
        """First non-local runner that is not quarantined, or None."""
        for runner in self._runners:
            if not runner.is_local and runner.name not in self._quarantined:
                return runner
        return None

    def _select_runner(self) -> VerifyRunner:
        """Prefer-remote: return the first non-quarantined non-local runner; fall back to local or runners[0].

        The K-permit free/busy refinement (load-based selection) is ζ.
        RemoteRunner.is_local is False, so it is selected over LocalRunner.
        Quarantined non-local runners are skipped; local is the trust anchor.
        Delegates to eligible_remote() to avoid duplicating the quarantine predicate,
        then falls back to self._local rather than self._runners[0] so the quarantine
        invariant holds regardless of pool construction order.
        """
        eligible = self.eligible_remote()
        if eligible is not None:
            return eligible
        return self._local or self._runners[0]

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


# ---------------------------------------------------------------------------
# ε: EnvFingerprint — env-fidelity fingerprint + codec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvFingerprint:
    """Frozen snapshot of a host's verify environment.

    Fields
    ------
    toolchain:        Trimmed stdout of rustc/cargo --version probes joined by newline.
    verify_env:       A copy of the verify_env mapping (str→str).
    sccache_reachable: True when the sccache probe exits 0.
    extra_probes:     Operator-supplied key→trimmed-stdout probe results.

    The canonical-JSON codec (fingerprint_to_json / fingerprint_from_json) is
    byte-identical: sort_keys=True canonicalises dict key order so that the same
    fingerprint always serialises to the same bytes regardless of insertion order.
    """

    toolchain: str
    verify_env: Mapping[str, str]
    sccache_reachable: bool
    extra_probes: Mapping[str, str]

    def to_dict(self) -> dict:
        return {
            "toolchain": self.toolchain,
            "verify_env": dict(self.verify_env),
            "sccache_reachable": self.sccache_reachable,
            "extra_probes": dict(self.extra_probes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EnvFingerprint:
        return cls(
            toolchain=d["toolchain"],
            verify_env=dict(d["verify_env"]),
            sccache_reachable=bool(d["sccache_reachable"]),
            extra_probes=dict(d["extra_probes"]),
        )


def fingerprint_to_json(fp: EnvFingerprint) -> str:
    """Serialise an EnvFingerprint to canonical JSON (sort_keys=True, ensure_ascii=False)."""
    return json.dumps(fp.to_dict(), sort_keys=True, ensure_ascii=False)


def fingerprint_from_json(s: str) -> EnvFingerprint:
    """Deserialise an EnvFingerprint from a JSON string."""
    return EnvFingerprint.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# ε: EnvParityVerdict — compare two EnvFingerprints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvParityVerdict:
    """Result of comparing two EnvFingerprints (local vs remote).

    is_faithful:      True when every dimension matches; False if any drift.
    drift_dimensions: Names of fields that differ (empty tuple when faithful).
    """

    is_faithful: bool
    drift_dimensions: tuple[str, ...]


def compare_env_fingerprints(
    local: EnvFingerprint,
    remote: EnvFingerprint,
) -> EnvParityVerdict:
    """Compare local and remote fingerprints field by field.

    Returns an EnvParityVerdict with is_faithful=True when all fields match,
    and drift_dimensions listing the names of any fields that differ.
    """
    drifts: list[str] = []
    if local.toolchain != remote.toolchain:
        drifts.append("toolchain")
    if dict(local.verify_env) != dict(remote.verify_env):
        drifts.append("verify_env")
    if local.sccache_reachable != remote.sccache_reachable:
        drifts.append("sccache_reachable")
    if dict(local.extra_probes) != dict(remote.extra_probes):
        drifts.append("extra_probes")
    return EnvParityVerdict(
        is_faithful=not bool(drifts),
        drift_dimensions=tuple(drifts),
    )


# ---------------------------------------------------------------------------
# ε: capture_env_fingerprint — probe a host's verify env
# ---------------------------------------------------------------------------


async def capture_env_fingerprint(
    run: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
    *,
    verify_env: Mapping[str, str] | None = None,
    extra_probe_specs: Sequence[tuple[str, list[str]]] = (),
) -> EnvFingerprint:
    """Probe a host's verify environment and return a frozen EnvFingerprint.

    Parameters
    ----------
    run:
        Injected async callable ``(argv, *, cwd=None) -> (rc, stdout, stderr)``.
        Defaults to ``_default_subprocess_run`` for local capture; pass an
        ssh-wrapping adapter for remote capture.
    verify_env:
        The verify_env mapping to embed verbatim (not probed — it is operator-
        supplied configuration, not a discovered fact).
    extra_probe_specs:
        Sequence of ``(key, argv)`` pairs for operator-supplied OS-level probes.
        Each argv is issued through ``run``; trimmed stdout is stored under
        ``key``.  When a probe exits non-zero the value is
        ``'<unavailable rc=N>'``.
    """
    _run = run if run is not None else _default_subprocess_run

    # Toolchain: rustc --version + cargo --version
    _, rustc_out, _ = await _run(['rustc', '--version'])
    _, cargo_out, _ = await _run(['cargo', '--version'])
    toolchain = (rustc_out.strip() + '\n' + cargo_out.strip()).strip()

    # sccache reachability
    sccache_rc, _, _ = await _run(['sccache', '--show-stats'])
    sccache_reachable = (sccache_rc == 0)

    # Extra probes
    extra_probes: dict[str, str] = {}
    for key, argv in extra_probe_specs:
        probe_rc, probe_out, _ = await _run(argv)
        if probe_rc == 0:
            extra_probes[key] = probe_out.strip()
        else:
            extra_probes[key] = f'<unavailable rc={probe_rc}>'

    return EnvFingerprint(
        toolchain=toolchain,
        verify_env=dict(verify_env) if verify_env is not None else {},
        sccache_reachable=sccache_reachable,
        extra_probes=extra_probes,
    )


# ---------------------------------------------------------------------------
# ε: ParityRow + VerdictParityReport + run_verdict_parity + render_parity_report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityRow:
    """One corpus entry in a verdict-parity run.

    sha:             The merge SHA tested.
    expected_pass:   Operator-supplied expected verdict (None if not specified).
    local_passed:    LocalRunner verdict.
    remote_passed:   RemoteRunner verdict.
    local_category:  LocalRunner result category (for divergence detail).
    remote_category: RemoteRunner result category.
    agree:           True when local_passed == remote_passed.
    matches_expected: True/False when expected_pass is not None; None otherwise.
    """

    sha: str
    expected_pass: bool | None
    local_passed: bool
    remote_passed: bool
    local_category: str
    remote_category: str
    agree: bool
    matches_expected: bool | None

    def to_dict(self) -> dict:
        return {
            "sha": self.sha,
            "expected_pass": self.expected_pass,
            "local_passed": self.local_passed,
            "remote_passed": self.remote_passed,
            "local_category": self.local_category,
            "remote_category": self.remote_category,
            "agree": self.agree,
            "matches_expected": self.matches_expected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ParityRow:
        return cls(
            sha=d["sha"],
            expected_pass=d["expected_pass"],
            local_passed=d["local_passed"],
            remote_passed=d["remote_passed"],
            local_category=d["local_category"],
            remote_category=d["remote_category"],
            agree=d["agree"],
            matches_expected=d["matches_expected"],
        )


@dataclass(frozen=True)
class VerdictParityReport:
    """Result of a verdict-parity run over a corpus of merge SHAs.

    rows:           One ParityRow per corpus SHA.
    all_agree:      True when every row has agree==True.
    divergent_shas: SHAs where local and remote disagreed.
    """

    rows: tuple[ParityRow, ...]
    all_agree: bool
    divergent_shas: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "all_agree": self.all_agree,
            "divergent_shas": list(self.divergent_shas),
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerdictParityReport:
        return cls(
            rows=tuple(ParityRow.from_dict(r) for r in d["rows"]),
            all_agree=d["all_agree"],
            divergent_shas=tuple(d["divergent_shas"]),
        )


def parity_report_to_json(report: VerdictParityReport) -> str:
    """Serialise a VerdictParityReport to canonical JSON."""
    return json.dumps(report.to_dict(), sort_keys=True, ensure_ascii=False)


def parity_report_from_json(s: str) -> VerdictParityReport:
    """Deserialise a VerdictParityReport from a JSON string."""
    return VerdictParityReport.from_dict(json.loads(s))


async def run_verdict_parity(
    corpus: Sequence[tuple[str, bool | None]],
    local_runner: Any,
    remote_runner: Any,
    spec: MergeVerifySpec,
) -> VerdictParityReport:
    """Run the same merge SHA through BOTH runners and compare verdicts.

    Parameters
    ----------
    corpus:        Sequence of (merge_sha, expected_pass_or_None).
    local_runner:  A VerifyRunner (LocalRunner or equivalent) — called directly.
    remote_runner: A VerifyRunner (RemoteRunner or equivalent) — called directly.
    spec:          The MergeVerifySpec to pass to both runners.

    Each SHA is run through local_runner.run_merge_verify AND
    remote_runner.run_merge_verify independently (NOT via VerifyRunnerPool).
    Agreement is decided on VerifyResult.passed (boolean verdict), per
    PRD §A Invariant 1.
    """
    rows: list[ParityRow] = []
    for sha, expected_pass in corpus:
        try:
            local_result = await local_runner.run_merge_verify(sha, spec)
            remote_result = await remote_runner.run_merge_verify(sha, spec)
        except Exception as exc:
            # A runner failure for one SHA must not discard the whole corpus run.
            # Record an errored row (agree=False, matches_expected=None) and continue.
            error_msg = f"runner_error: {exc}"
            rows.append(ParityRow(
                sha=sha,
                expected_pass=expected_pass,
                local_passed=False,
                remote_passed=False,
                local_category=error_msg,
                remote_category=error_msg,
                agree=False,
                matches_expected=None,
            ))
            continue
        agree = local_result.passed == remote_result.passed
        if expected_pass is None or not agree:
            # matches_expected is undefined when there is no agreed verdict to compare.
            matches_expected = None
        else:
            matches_expected = (local_result.passed == expected_pass)
        rows.append(ParityRow(
            sha=sha,
            expected_pass=expected_pass,
            local_passed=local_result.passed,
            remote_passed=remote_result.passed,
            local_category=getattr(local_result, 'category', ''),
            remote_category=getattr(remote_result, 'category', ''),
            agree=agree,
            matches_expected=matches_expected,
        ))

    divergent_shas = tuple(r.sha for r in rows if not r.agree)
    return VerdictParityReport(
        rows=tuple(rows),
        all_agree=not bool(divergent_shas),
        divergent_shas=divergent_shas,
    )


# ---------------------------------------------------------------------------
# ι: DriftDetector — dual-host same-SHA verdict parity + divergence escalation
# ---------------------------------------------------------------------------

# Sentinel task_id used for dedup'd drift escalation (mirrors harness '__scheduler__').
_DRIFT_SENTINEL = '__drift__'


class DriftVerdict(StrEnum):
    """Outcome of a single DriftDetector.check() call."""
    AGREE = 'agree'
    DIVERGE = 'diverge'
    INCONCLUSIVE = 'inconclusive'


@dataclass(frozen=True)
class DriftCheckResult:
    """Result of one DriftDetector.check() call.

    Fields
    ------
    merge_sha:     The SHA that was checked.
    verdict:       AGREE / DIVERGE / INCONCLUSIVE.
    local_passed:  bool verdict from the local runner (None when INCONCLUSIVE).
    remote_passed: bool verdict from the remote runner (None when INCONCLUSIVE).
    escalated:     True when a new divergence escalation was submitted.
    quarantined:   True when the remote runner was quarantined.
    """
    merge_sha: str
    verdict: DriftVerdict
    local_passed: bool | None = None
    remote_passed: bool | None = None
    escalated: bool = False
    quarantined: bool = False


class DriftDetector:
    """Periodically checks that local + remote return the same verdict for a merge SHA.

    PRD §8 / §B B4/B5 (task ι).

    On agree  → emit EventType.verdict_parity_ok (no escalation).
    On diverge → dedup'd L1 blocking escalation + quarantine the remote runner.
    On inconclusive (transport failure or no eligible remote) → no side-effects.

    The detector runs the two runners DIRECTLY (not via run_verdict_parity) so
    it can distinguish RunnerUnavailable (transport ≠ divergence, Invariant 5)
    from a genuine two-verdict disagreement.
    """

    def __init__(
        self,
        pool: VerifyRunnerPool,
        *,
        event_store: Any = None,
        escalation_queue: Any = None,
        task_id: str | None = None,
        every_n_lands: int = 20,
    ) -> None:
        self._pool = pool
        self._event_store = event_store
        self._escalation_queue = escalation_queue
        self._task_id = task_id
        if every_n_lands <= 0:
            raise ValueError(
                f'every_n_lands must be a positive integer, got {every_n_lands!r}'
            )
        self._every_n_lands = every_n_lands

    def should_sample(self, land_count: int) -> bool:
        """Return True when land_count is a positive multiple of every_n_lands.

        Realises PRD §10 Open Q2 as a pure in-code cadence predicate.
        """
        return land_count > 0 and land_count % self._every_n_lands == 0

    async def check(self, merge_sha: str, spec: MergeVerifySpec) -> DriftCheckResult:
        """Run *merge_sha* on both runners and compare verdicts.

        Returns DriftCheckResult.  Side-effects:
        - AGREE   → emit verdict_parity_ok event (None-safe).
        - DIVERGE → dedup'd L1 escalation (None-safe) + quarantine remote.
        - INCONCLUSIVE → no side-effects.
        """
        local = self._pool.local_runner
        remote = self._pool.eligible_remote()
        if local is None or remote is None:
            return DriftCheckResult(merge_sha=merge_sha, verdict=DriftVerdict.INCONCLUSIVE)

        try:
            local_result = await local.run_merge_verify(merge_sha, spec)
        except RunnerUnavailable:
            # Local transport failure also yields INCONCLUSIVE — symmetric with remote.
            return DriftCheckResult(merge_sha=merge_sha, verdict=DriftVerdict.INCONCLUSIVE)
        try:
            remote_result = await remote.run_merge_verify(merge_sha, spec)
        except RunnerUnavailable:
            # Transport failure ≠ divergence (Invariant 5).
            # A closed/flaky laptop must never raise a false drift alarm.
            return DriftCheckResult(merge_sha=merge_sha, verdict=DriftVerdict.INCONCLUSIVE)

        local_passed = local_result.passed
        remote_passed = remote_result.passed

        if local_passed == remote_passed:
            # Agree — emit verdict_parity_ok event.
            if self._event_store is not None:
                from orchestrator.event_store import EventType
                self._event_store.emit(
                    EventType.verdict_parity_ok,
                    task_id=self._task_id,
                    data={
                        'merge_sha': merge_sha,
                        'local_runner': local.name,
                        'remote_runner': remote.name,
                        'passed': local_passed,
                    },
                )
            return DriftCheckResult(
                merge_sha=merge_sha,
                verdict=DriftVerdict.AGREE,
                local_passed=local_passed,
                remote_passed=remote_passed,
            )

        # Diverge — dedup'd escalation + quarantine.
        escalated = False
        if self._escalation_queue is not None and not self._escalation_queue.has_open_l1(_DRIFT_SENTINEL):
            from escalation.models import Escalation
            esc = Escalation(
                id=self._escalation_queue.make_id(_DRIFT_SENTINEL),
                task_id=_DRIFT_SENTINEL,
                agent_role='orchestrator-drift-detector',
                severity='blocking',
                level=1,
                category='verify_drift_divergence',
                summary=(
                    f'Drift detected for {merge_sha}: '
                    f'local={local_passed} remote={remote_passed}'
                ),
                detail=(
                    f'merge_sha={merge_sha!r} local_runner={local.name!r} '
                    f'({local_passed}) remote_runner={remote.name!r} ({remote_passed}). '
                    f'A remote PASS / local FAIL split can land unverified code on main.'
                ),
                suggested_action='Re-prove laptop env via run_verdict_parity; call pool.clear_quarantine after parity is restored.',
            )
            self._escalation_queue.submit(esc)
            escalated = True

        # Quarantine unconditionally (even if the escalation was deduped).
        self._pool.quarantine(remote.name)

        return DriftCheckResult(
            merge_sha=merge_sha,
            verdict=DriftVerdict.DIVERGE,
            local_passed=local_passed,
            remote_passed=remote_passed,
            escalated=escalated,
            quarantined=True,
        )


def render_parity_report(report: VerdictParityReport) -> str:
    """Render a VerdictParityReport as a Markdown string.

    Structure
    ---------
    - Headline verdict (PASS / DIVERGENCE DETECTED)
    - Results table: sha | expected | local | remote | agree
    - Divergence callout listing divergent SHAs (when non-empty)
    """
    lines: list[str] = []

    # Headline verdict
    lines.append("# Verdict Parity Report\n")
    if report.all_agree:
        lines.append("**Overall verdict: ✅ PASS — parity holds across all corpus SHAs.**\n")
    else:
        lines.append(
            f"**Overall verdict: ❌ DIVERGENCE DETECTED — "
            f"{len(report.divergent_shas)} SHA(s) disagree.**\n"
        )

    # Results table
    lines.append("## Results\n")
    lines.append("| sha | expected | local | remote | agree |")
    lines.append("|-----|----------|-------|--------|-------|")
    for row in report.rows:
        expected_str = "pass" if row.expected_pass else ("fail" if row.expected_pass is False else "—")
        local_str = "✅" if row.local_passed else "❌"
        remote_str = "✅" if row.remote_passed else "❌"
        agree_str = "✅" if row.agree else "❌"
        lines.append(
            f"| `{row.sha}` | {expected_str} | {local_str} | {remote_str} | {agree_str} |"
        )

    # Divergence callout
    if report.divergent_shas:
        lines.append("\n## Divergent SHAs\n")
        lines.append("The following SHAs produced different verdicts on local vs remote:\n")
        for sha in report.divergent_shas:
            lines.append(f"- `{sha}`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# κ: SccacheStats + parse_sccache_stats + capture_sccache_stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SccacheStats:
    """Parsed output of ``sccache --show-stats`` (κ signal: shared-backend hit rate).

    Fields
    ------
    compile_requests:  Total compilation requests processed.
    cache_hits:        Aggregate cache hits (NOT the per-language breakdown).
    cache_misses:      Total cache misses.
    cache_location:    Raw ``Cache location`` value from the stats output, e.g.
                       ``"Redis: redis://orch:6379"`` or ``"Local disk: /path"``.

    Properties
    ----------
    hit_rate:           cache_hits / (cache_hits + cache_misses); 0.0 when denominator 0.
    is_shared_backend:  True when cache_location is non-empty and does NOT start with
                        'local disk' (case-insensitive).  Makes remote_hit_rate > 0 a
                        faithful proxy for "served by the shared backend."
    remote_hit_rate:    hit_rate when is_shared_backend, else 0.0.
    remote_hits:        cache_hits when is_shared_backend, else 0.
    """

    compile_requests: int
    cache_hits: int
    cache_misses: int
    cache_location: str

    @property
    def hit_rate(self) -> float:
        denom = self.cache_hits + self.cache_misses
        return self.cache_hits / denom if denom > 0 else 0.0

    @property
    def is_shared_backend(self) -> bool:
        loc = self.cache_location.strip().lower()
        return bool(loc) and not loc.startswith('local disk')

    @property
    def remote_hit_rate(self) -> float:
        return self.hit_rate if self.is_shared_backend else 0.0

    @property
    def remote_hits(self) -> int:
        return self.cache_hits if self.is_shared_backend else 0


def parse_sccache_stats(output: str) -> SccacheStats:
    """Parse the text output of ``sccache --show-stats`` into a SccacheStats.

    Uses EXACT label matching so the ``Cache hits`` aggregate is not confused
    with ``Cache hits (Rust)`` or other per-language breakdown lines.
    """
    compile_requests = 0
    cache_hits = 0
    cache_misses = 0
    cache_location = ''

    # Numeric labels: extract the LAST whitespace-delimited token as an integer.
    # We require EXACT label == expected text (before the numeric column) so
    # "Cache hits (Rust)" is never mistaken for "Cache hits".
    def _extract_int(label_expected: str, s: str) -> int | None:
        if not s.startswith(label_expected):
            return None
        # The remainder after the label must be purely numeric (possibly with
        # leading spaces).  If the label is followed by a '(' it is a variant
        # (e.g. "Cache hits (Rust)") and must be rejected.
        remainder = s[len(label_expected):]
        if not remainder:
            return None
        # reject parenthesised variants
        if remainder.lstrip().startswith('('):
            return None
        tokens = remainder.split()
        if not tokens:
            return None
        try:
            return int(tokens[-1])
        except ValueError:
            return None

    for line in output.splitlines():
        # Strip leading whitespace; skip blank lines.
        stripped = line.strip()
        if not stripped:
            continue

        # ``Cache location`` is a string value — extract the remainder after label.
        if stripped.startswith('Cache location'):
            rest = stripped[len('Cache location'):].strip()
            # Remove a leading colon if present (e.g. "Cache location      Redis: …")
            if rest.startswith(':'):
                rest = rest[1:].strip()
            cache_location = rest
            continue

        v = _extract_int('Compile requests', stripped)
        if v is not None:
            compile_requests = v
            continue

        v = _extract_int('Cache hits', stripped)
        if v is not None:
            cache_hits = v
            continue

        v = _extract_int('Cache misses', stripped)
        if v is not None:
            cache_misses = v

    return SccacheStats(
        compile_requests=compile_requests,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_location=cache_location,
    )


async def capture_sccache_stats(
    run: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
) -> SccacheStats:
    """Probe the local sccache daemon and return a SccacheStats.

    This is the operational hook for the κ signal ("sccache --show-stats after
    a warm run").  Callers compute ``remote_hit_rate`` on the result to verify
    that the shared backend is being used.

    Parameters
    ----------
    run:
        Injected async callable ``(argv, *, cwd=None) -> (rc, stdout, stderr)``.
        Defaults to ``_default_subprocess_run`` for local capture; pass an
        ssh-wrapping adapter for remote capture.
    """
    _run = run if run is not None else _default_subprocess_run
    _, stdout, _ = await _run(['sccache', '--show-stats'])
    return parse_sccache_stats(stdout)


# ---------------------------------------------------------------------------
# κ: ColdWarmVerifyDelta + delta_to_json / delta_from_json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColdWarmVerifyDelta:
    """Cold-vs-warm laptop-verify wall-time delta (κ signal, PRD G6).

    Records the measured cold/warm times and the derived speedup ratio.
    Per PRD G6 the ~1× warm multiplier is an EXPECTATION, not a gate:
    this type carries no threshold assertion.

    Fields
    ------
    cold_secs:  Wall-clock time for a cold (no shared-cache) verify run.
    warm_secs:  Wall-clock time for a warm (shared-cache) verify run.

    Properties
    ----------
    speedup:    cold_secs / warm_secs; 0.0 when warm_secs == 0 (documented guard).
    """

    cold_secs: float
    warm_secs: float

    @property
    def speedup(self) -> float:
        """cold_secs / warm_secs; 0.0 when warm_secs == 0 (zero-guard, not inf)."""
        if self.warm_secs == 0.0:
            return 0.0
        return self.cold_secs / self.warm_secs

    def to_dict(self) -> dict:
        return {'cold_secs': self.cold_secs, 'warm_secs': self.warm_secs}

    @classmethod
    def from_dict(cls, d: dict) -> ColdWarmVerifyDelta:
        return cls(cold_secs=float(d['cold_secs']), warm_secs=float(d['warm_secs']))


def delta_to_json(d: ColdWarmVerifyDelta) -> str:
    """Serialize a ColdWarmVerifyDelta to a byte-canonical JSON string (sort_keys=True)."""
    import json as _json
    return _json.dumps(d.to_dict(), sort_keys=True, ensure_ascii=False)


def delta_from_json(s: str) -> ColdWarmVerifyDelta:
    """Deserialize a ColdWarmVerifyDelta from a JSON string produced by delta_to_json."""
    import json as _json
    return ColdWarmVerifyDelta.from_dict(_json.loads(s))
