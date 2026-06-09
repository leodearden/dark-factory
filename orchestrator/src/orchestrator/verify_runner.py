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

import dataclasses
import json
import time
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
    "VerifyRunnerPool",
    "build_merge_verify_spec",
    "_module_config_from_command",
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
    stay at their dataclass defaults so reconstruction is information-preserving for
    all verify-relevant behaviour.
    """
    return ModuleConfig(
        prefix=vc.prefix,
        test_command=vc.test_command,
        lint_command=vc.lint_command,
        type_check_command=vc.type_check_command,
        verify_env=dict(spec.verify_env),
        verify_cold_command_timeout_secs=spec.cold_timeout_secs,
    )


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
    """Dispatches a merge-verify bundle to a single VerifyRunner and emits a telemetry event.

    In β there is exactly one runner (LocalRunner). The long-lived K-permit
    pool with multiple remote runners is deferred to ζ.
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
        self._runner = runners[0]
        self._event_store = event_store
        self._task_id = task_id

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
        """
        from orchestrator.event_store import EventType

        t0 = time.monotonic()
        result = await self._runner.run_merge_verify(merge_sha, spec)
        duration_ms = round((time.monotonic() - t0) * 1000)

        if self._event_store is not None:
            self._event_store.emit(
                EventType.merge_verify,
                task_id=self._task_id,
                data={
                    'runner': self._runner.name,
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
