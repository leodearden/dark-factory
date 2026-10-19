"""verify_runner — host-independent MergeVerifySpec + VerifyResult JSON codec.

This module defines the data-contract types for multi-host merge verification
(PRD: plans/merge-throughput-multihost-verify-prd.md §A) and the wire codec
that serialises/deserialises them as byte-identical JSON documents.

Public API
----------
Dataclasses (frozen):
  UnscopedTypecheckSpec  — _run_unscoped_typechecks gate inputs
  MergeVerifySpec        — full pre-advance merge-verify bundle (PRD §A contract)
  (both carry orchestrator.verify_spec_shared.VerifyCommand, one per module)

Wire codec (module-level functions):
  spec_to_json(spec)     -> str   — canonical JSON for the spec document
  spec_from_json(s)      -> MergeVerifySpec
  result_to_json(vr)     -> str   — canonical JSON for a VerifyResult document
  result_from_json(s)    -> VerifyResult
  result_to_dict(vr)     -> dict  — helper used by result_to_json
  result_from_dict(d)    -> VerifyResult

Byte-identity guarantee (PRD Invariant 1)
-----------------------------------------
All JSON is emitted with ``verify_spec_shared.canonical_json``
(``json.dumps(sort_keys=True, ensure_ascii=False)``).
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
import os
import shlex
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    assert_never,
    get_args,
    runtime_checkable,
)

from orchestrator import flake_ledger, verify
from orchestrator.config import EnvProbeConfig, ModuleConfig, VerifyHostPolicy
from orchestrator.task_verify_spec import TaskVerifySpec, task_spec_to_json
from orchestrator.verify import (
    VerifyResult,
    _archive_merge_verify_logs,
    _target_subprocess_env,
)
from orchestrator.verify_cancel import (
    HEARTBEAT_INTERVAL_SECS,
    SSH_SERVER_ALIVE_COUNT_MAX,
    SSH_SERVER_ALIVE_INTERVAL,
    start_stdin_heartbeat,
)
from orchestrator.verify_categories import FailureCategory, _assert_sentinels_disjoint
from orchestrator.verify_spec_shared import (
    VerifyCommand,
    apply_spec_to_host_config,
    canonical_json,
    global_verify_command_for,
    module_config_from_command,
)

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.event_store import EventStore

__all__ = [
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
    "RemoteVerifySubcommand",
    "REMOTE_VERIFY_PROCESS_PATTERN",
    "RunnerUnavailable",
    "RunnerUnavailableKind",
    "ConcurrentDispatchError",
    "VerifyRunnerPool",
    # INV-2 (task 2884) — contract-currency auto-sync at dispatch
    "SyncOutcome",
    "REMOTE_LIVENESS_CMD",
    "REMOTE_TOOL_PATH_PRELUDE",
    "resolve_local_df_checkout",
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
    "_ENV_FINGERPRINT_DIMENSIONS",
    "capture_env_fingerprint",
    "env_probe_local_run",
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
    # task 4537 additions — host-environment divergence detector
    "EnvDriftCheckResult",
    # κ additions
    "SccacheStats",
    "parse_sccache_stats",
    "capture_sccache_stats",
    "ColdWarmVerifyDelta",
    "delta_to_json",
    "delta_from_json",
    # β additions
    "HostLease",
    "HostAllocator",
    # α additions (task 2306 — laptop flock-guard contention outcome)
    "FLOCK_CONTENTION_CATEGORY",
    "make_flock_contention_result",
    # task 2307 additions — workstation consumer discriminant for the α outcome
    "is_flock_contention_failure",
]

# Sentinel category constants — encode an unscoped-gate failure inside a
# VerifyResult so _run_post_merge_verify can branch byte-identically.
UNSCOPED_TYPECHECK_FAILED_CATEGORY = 'unscoped_typecheck_failed'
UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY = 'unscoped_typecheck_timeout'

_UNSCOPED_SENTINEL_CATEGORIES = frozenset({
    UNSCOPED_TYPECHECK_FAILED_CATEGORY,
    UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
})

# Fail-loud import-time guard (mirrors verify_categories._validate_exhaustive):
# these sentinels are a deliberately separate, out-of-band gate namespace —
# never produced by verify_classify.classify_failure — so they must never
# collide with a real FailureCategory value.  See plans/verify-plan-prd.md task α.
_assert_sentinels_disjoint(_UNSCOPED_SENTINEL_CATEGORIES, FailureCategory)

# Sentinel category for a laptop-side flock-contention outcome (task 2306 α,
# PRD plans/laptop-warm-verify-flock-orphan-prd.md Change A).  Mirrors the
# UNSCOPED_* precedent above: a stable constant on VerifyResult.category that
# consumers (task beta, the workstation side) key on.
FLOCK_CONTENTION_CATEGORY = 'flock_contention'


def make_flock_contention_result(
    host: str,
    holder_pgid: int | None,
    waiter_pgid: int,
) -> VerifyResult:
    """Build the distinguished VerifyResult emitted on bounded-wait flock contention.

    Emitted by ``orchestrator verify-merge`` (cli.py) when
    ``config.git.persistent_merge_worktree`` is on and the laptop-side
    ``.merge_verify.lock`` could not be acquired within the bounded wait —
    i.e. another verify invocation already holds the persistent warm
    merge-verify worktree.  Returned instead of ever falling back to an
    ephemeral worktree (PRD Invariant 5).

    This is the HOST cli.py verify-merge span's OWN lock
    (``verify_cancel.merge_verify_lock_path``), intentionally distinct
    from the local in-process consumer's shared ``<lane_dir>.lock``
    (``verify_cancel.lane_lock_path``, taken by ``GitOps.merge_verify_lease``
    and ``GitOps.reset_persistent_merge_worktree`` — task 2685): the laptop
    persistent-worktree deployment this span guards has no reify CoW
    warm-lane pool to converge with, so ``.merge_verify.lock`` stays as is.

    The ``contention`` field is a plain JSON-native dict (not a nested
    dataclass) so it round-trips losslessly through the generic
    ``result_to_dict``/``result_from_dict`` codec — task beta parses it back
    out of stdout unchanged.  ``holder_pgid`` is ``None`` when the holder's
    pgid file is absent or corrupt (fail-safe; see
    ``verify_cancel.read_lock_holder_pgid``).
    """
    return VerifyResult(
        passed=False,
        test_output='',
        lint_output='',
        type_output='',
        summary=(
            'flock contention on the host .merge_verify.lock verify span; '
            'another verify holds the persistent worktree lock'
        ),
        timed_out=False,
        category=FLOCK_CONTENTION_CATEGORY,
        contention={'host': host, 'holder_pgid': holder_pgid, 'waiter_pgid': waiter_pgid},
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
    verify_env          : environment overrides (RUSTC_WRAPPER, CARGO_INCREMENTAL, …).
                          APPLIED ONTO the consuming host's config in
                          run_merge_verify_on_worktree (task 5496): the spec wins
                          on conflict, host keys absent from the spec are
                          preserved. Neither 'replace' nor 'ignore' — the spec is
                          dispatcher-shaped, while a remote runner's own --config
                          carries per-host local necessities no dispatcher-built
                          spec can know (a narrower verify-only host widening its
                          own per-host-measured timeout budgets is the live case).
                          MERGE-DECIDING, not cosmetic: these keys SELECT TESTS,
                          so a remote green reached under a different env is the
                          task-2822 false-green class, not a performance detail.
                          Path-valued keys ship VERBATIM and resolve against the
                          REMOTE filesystem — no rewrite, no denylist, on purpose:
                          the per-module path already ships them verbatim, and a
                          name-based filter would put one project's vocabulary
                          inside generic transport code and re-diverge the two
                          paths task 5496 unified. Which keys a project sets, and
                          why any of them is deliberately absolute, stays in that
                          project's own dark-factory-orchestrator.yaml — its single
                          home (INV-9), and the file to read before assuming what a
                          key here means.
                          The failed-only retry env from
                          orchestrator/src/orchestrator/merge_lane/worker.py::_build_retry_verify_env
                          has path-valued keys that cannot resolve remotely:
                          its nextest filter files are UNTRACKED files in the
                          dispatching host's merge_wt, which the git-pushed merge
                          SHA does not carry. That env is therefore never shipped:
                          merge_lane/worker.py::_run_post_merge_verify builds it
                          only for a LOCAL dispatch (runner is None), so a REMOTE
                          retry runs FULL and is charged to the legacy (full)
                          retry budget, not the narrowed one. Pinned by
                          orchestrator/tests/test_merge_queue_verify_retry.py::test_remote_dispatch_retry_runs_full_and_writes_nothing_on_the_dispatcher
                          (and ..._is_charged_to_the_legacy_budget).
    cold_timeout_secs   : merge_verify_cold cascade timeout
    is_merge_verify     : always True for merge-path specs (default)
    merge_verify_workspace : force-workspace profile of the merge gate (fix a,
                          task 2822) — carried so the remote host runs the
                          SAME profile as the dispatching merge gate, not the
                          laptop config's (possibly narrower) default.
    merge_verify_breadth   : 'scoped' | 'full' breadth of the merge gate (fix a,
                          task 2822) — same rationale.
    global_verify_command  : the project's global full-gate commands, sourced by
                          build_merge_verify_spec ONLY when module_configs is
                          empty (a zero-module-config project, e.g. reify). It is
                          applied onto the reconstructed remote config in
                          run_merge_verify_on_worktree so the remote runs the
                          SAME gate as local instead of its own (possibly stale)
                          config globals — the fidelity fix for INV-1 (task 2883,
                          incident 966f23a6). None when module_configs is
                          non-empty (per-module verify_commands drive the gate).
    verify_cold_preprovision_command : the project's cold-worktree pre-provision
                          command, which verify.py::_preprovision_shared_venv
                          runs before the gather. Applied by
                          verify_spec_shared.py::apply_spec_to_host_config under
                          TaskVerifySpec's rule: non-empty replaces the host's
                          value, '' keeps the host's own. '' is also what a
                          pre-6127 dispatcher's spec decodes to.

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
    # Fix (a), task 2822 — the merge-gate PROFILE rides the spec so the remote
    # host runs the identical scope/profile as the dispatching merge gate.
    # Narrow defaults match the pre-fix behaviour and the OrchestratorConfig
    # field defaults (merge_verify_workspace=False, merge_verify_breadth='scoped').
    merge_verify_workspace: bool = False
    merge_verify_breadth: str = "scoped"
    # INV-1, task 2883 — the global full-gate commands for a zero-module-config
    # project, shipped so the remote runs the SAME gate as local. Narrow default
    # None matches every existing spec (module_configs non-empty → not sourced).
    global_verify_command: VerifyCommand | None = None
    # Task 6127 — the dispatcher's pre-provision command; '' keeps the host's.
    verify_cold_preprovision_command: str = ''

    def to_dict(self) -> dict:
        return {
            "verify_commands": [vc.to_dict() for vc in self.verify_commands],
            "unscoped_typecheck": self.unscoped_typecheck.to_dict(),
            "task_files": list(self.task_files) if self.task_files is not None else None,
            "verify_env": dict(self.verify_env),
            "cold_timeout_secs": float(self.cold_timeout_secs),
            "is_merge_verify": self.is_merge_verify,
            "merge_verify_workspace": self.merge_verify_workspace,
            "merge_verify_breadth": self.merge_verify_breadth,
            "global_verify_command": (
                self.global_verify_command.to_dict()
                if self.global_verify_command is not None
                else None
            ),
            "verify_cold_preprovision_command": self.verify_cold_preprovision_command,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MergeVerifySpec:
        task_files_raw = d.get("task_files")
        # Back-compat: a legacy spec dict (pre-task-2883) lacks this key, so
        # default to None — same d.get idiom as merge_verify_workspace/breadth.
        _gvc = d.get("global_verify_command")
        return cls(
            verify_commands=tuple(VerifyCommand.from_dict(vc) for vc in d["verify_commands"]),
            unscoped_typecheck=UnscopedTypecheckSpec.from_dict(d["unscoped_typecheck"]),
            task_files=tuple(task_files_raw) if task_files_raw is not None else None,
            verify_env=dict(d.get("verify_env", {})),
            cold_timeout_secs=float(d["cold_timeout_secs"]),
            is_merge_verify=d.get("is_merge_verify", True),
            # Back-compat: a legacy spec dict (pre-task-2822) lacks these keys,
            # so default to the narrow merge gate — same d.get idiom as verify_env.
            merge_verify_workspace=d.get("merge_verify_workspace", False),
            merge_verify_breadth=d.get("merge_verify_breadth", "scoped"),
            global_verify_command=(
                VerifyCommand.from_dict(_gvc) if _gvc is not None else None
            ),
            verify_cold_preprovision_command=d.get("verify_cold_preprovision_command", ""),
        )


# ---------------------------------------------------------------------------
# VerifyResult codec
# ---------------------------------------------------------------------------


def result_to_dict(vr: VerifyResult) -> dict:
    """Serialise a VerifyResult to a plain dict of JSON-native types.

    Uses ``dataclasses.asdict``, which recursively converts nested dataclasses and
    preserves all field types.  Every field is JSON-native EXCEPT ``flake_suppression``
    (task 3789 ε), which is a nested ``FlakeSuppression``: ``asdict`` flattens it to a
    dict here, and ``json.dumps`` flattens its ``StrEnum`` members to their values and
    its tuple to an array — so this direction still needs no special handling.  The
    asymmetry is entirely on the READ side; see :func:`result_from_dict`.
    """
    return dataclasses.asdict(vr)


def result_from_dict(d: dict) -> VerifyResult:
    """Reconstruct a VerifyResult from a dict (as produced by result_to_dict).

    Generic ``VerifyResult(**d)`` for every field but one.  ``flake_suppression`` (task
    3789 ε) is a nested dataclass, so ``asdict``/JSON hands it back as a plain dict with
    ``verdict``/``call_site`` as strings and ``test_ids`` as a list — passing that
    straight through would leave the field's TYPED annotation a lie on exactly the
    deserialized path it exists to serve.  Rebuild it first, in the same
    ``d.get(key)`` / ``X(v) if v is not None else None`` shape
    ``orchestrator/src/orchestrator/verify_runner.py::MergeVerifySpec.from_dict`` uses for
    its optional nested ``global_verify_command`` — this file's established idiom for a
    newly-added optional field.

    ``flake_suppression_from_wire`` NEVER raises: a malformed sub-payload degrades to
    ``None`` with a loud warning, because anything raising out of here becomes a
    ``RunnerUnavailable`` in
    ``orchestrator/src/orchestrator/verify_runner.py::RemoteRunner.run_merge_verify`` and
    costs a whole local re-verify.

    The codec's strictness is otherwise UNCHANGED — an unknown top-level key is still a
    ``TypeError`` (pinned by test_verify_runner's characterization tests), the
    pre-existing behaviour shared by every optional field added before this one.
    """
    # `isinstance` guard, not a bare `d.get`: a buggy remote can send a valid JSON
    # LIST, and that must keep failing exactly as it does today — `VerifyResult(**d)`
    # raising TypeError, which
    # `orchestrator/src/orchestrator/verify_runner.py::RemoteRunner.run_merge_verify`
    # catches — instead of a fresh
    # AttributeError that no handler on the merge path expects.
    if isinstance(d, dict):
        raw = d.get('flake_suppression')
        if raw is not None:
            # Shallow COPY — never mutate the caller's dict, which it may still inspect.
            d = {**d, 'flake_suppression': flake_ledger.flake_suppression_from_wire(raw)}
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
    verify_commands = tuple(VerifyCommand.from_module_config(mc) for mc in module_configs)
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
        # Fix (a), task 2822 — project the merge-gate profile from LIVE config so
        # the spec is the single source of truth for the profile at the transport
        # boundary; the remote host applies these over its own (laptop) config.
        merge_verify_workspace=config.merge_verify_workspace,
        merge_verify_breadth=config.merge_verify_breadth,
        global_verify_command=global_verify_command_for(config, verify_commands),
        verify_cold_preprovision_command=config.verify_cold_preprovision_command,
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
    return module_config_from_command(
        vc,
        verify_env=spec.verify_env,
        cold_timeout_secs=spec.cold_timeout_secs if spec.cold_timeout_secs > 0 else None,
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

    The reconstructed set is ALSO installed as the config's module registry
    (task 4536), so ``config.module_configs_or_empty`` and the positionally
    passed *module_configs* cannot disagree on this leg — whichever of the two
    a downstream consumer happens to read (``effective_merge_module_configs``
    reads the registry; ``run_scoped_verification`` is handed the list), it
    sees the DISPATCHER's set rather than the remote host's own discovery walk.

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
        from orchestrator.merge_lane.worker import _run_unscoped_typechecks
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

    # Fix (a), task 2822 — the merge-deciding profile rides the spec, so the
    # host's own --config can no longer narrow it; task 6127 — nor skip the
    # dispatcher's cold-venv pre-provision.
    config = apply_spec_to_host_config(
        config,
        verify_env=spec.verify_env,
        global_verify_command=spec.global_verify_command,
        module_configs=module_configs,
        field_overrides={
            'merge_verify_workspace': spec.merge_verify_workspace,
            'merge_verify_breadth': spec.merge_verify_breadth,
        },
        verify_cold_preprovision_command=spec.verify_cold_preprovision_command,
    )

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
# RunnerUnavailable — the remote host produced no usable verdict
# ---------------------------------------------------------------------------


RunnerUnavailableKind = Literal['transport', 'unparseable', 'no_verdict', 'wire_shape']


class RunnerUnavailable(Exception):
    """Raised by RemoteRunner when the remote host produced no usable verdict.

    ``kind`` says why: ``transport`` (host down/closed, git push or ssh
    failure, non-zero ssh exit), ``unparseable`` (stdout is not JSON),
    ``wire_shape`` (JSON of the wrong shape — a contract skew between the two
    hosts), or ``no_verdict`` — the ONE parseable outcome that raises: a result
    whose summary records no verdict (``verify.is_no_verdict_summary``).  Every
    other parseable VerifyResult is returned — even passed=False or
    timed_out=True (PRD §A Invariant 5).
    """

    def __init__(self, *args: object, kind: RunnerUnavailableKind = 'transport') -> None:
        super().__init__(*args)
        if kind not in get_args(RunnerUnavailableKind):
            raise ValueError(
                f'RunnerUnavailable kind must be one of {get_args(RunnerUnavailableKind)}, '
                f'got {kind!r}'
            )
        self.kind: RunnerUnavailableKind = kind


class ConcurrentDispatchError(RuntimeError):
    """A second dispatch reached a RemoteRunner that already has one in flight.

    A HostLease grants exclusive use of its RemoteRunner (PRD §4.3), so this
    is a broken invariant — a bug, never a fallback signal, and deliberately
    not a RunnerUnavailable, which callers read as a host outage.
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
    1. Existing test patches on 'orchestrator.merge_lane.worker.run_scoped_verification'
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
        archive_root: Path | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        """Initialise LocalRunner.

        *archive_root* threads merge-verify logs to ``data/verify-logs/<task_id>/``
        when set.  Default ``None`` preserves byte-identical behaviour for all
        existing constructions and the CLI ``run_merge_verify_on_worktree`` path.
        Policy lives in the caller (merge_queue.py wires the concrete path);
        cold-shadow / drift intentionally leave this ``None`` so they are
        auto-excluded from archival without any extra deny-list logic.

        *event_store* threads the dispatching store into ``run_scoped``'s merge
        gate, so a trivial pass emits ``trivial_pass_escalated`` (INV-1, task 2883).
        It defaults to ``None`` — byte-identical for the CLI
        ``run_merge_verify_on_worktree`` / remote-runner paths, which cannot reach
        the dispatching host's store — mirroring the optional ``archive_root``
        threading above.

        There is deliberately NO *escalation_queue* (task ε).  It fed only the
        merge-flake storm-streak bump, and that side-effect now happens on the
        DISPATCHER, from the ``FlakeSuppression`` the returned ``VerifyResult``
        carries — see ``verify.apply_merge_flake_suppression``.  A LocalRunner
        runs where the WORKTREE is and cannot reach the dispatching host's queue,
        so accepting one here only ever invited re-wiring a side-effect onto the
        host that cannot perform it.
        """
        self._merge_wt = merge_wt
        self._config = config
        self._module_configs = module_configs
        self._task_files = task_files
        self._run_scoped = run_scoped
        self._run_unscoped = run_unscoped
        self._task_id = task_id
        self._archive_root = archive_root
        self._event_store = event_store

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

        NOTE: ``spec`` is accepted for VerifyRunner protocol conformance.
        LocalRunner drives execution from its injected callables + live config,
        not from the spec (by design).  RemoteRunner (defined below in this
        module) is now the spec consumer for off-host dispatch — it serialises
        the spec via ``spec_to_json(spec)`` and ships it as the ``--spec``
        argument to ``orchestrator verify-merge`` over ssh.
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
            task_id=self._task_id,
            archive_root=self._archive_root,
            # INV-1, task 2883 — thread the dispatching store so run_scoped's
            # merge gate (role='merge' AND is_merge_verify) emits
            # trivial_pass_escalated. None for the CLI/remote in-worktree path
            # (run_merge_verify_on_worktree constructs LocalRunner without an
            # event_store), which cannot reach the dispatching host's store —
            # None-safe by construction, mirroring merge-flake suppression.
            event_store=self._event_store,
        )
        if not scoped.passed:
            # PRD task α: single flake-retry gate. Re-run the named failing tests
            # isolated + serial in THIS merge worktree; if they all pass, the red
            # was a CPU-starvation flake — suppress it (returns a PASSED result)
            # so the merge proceeds INTO the unscoped gate below, rather than
            # short-circuiting here.  On a non-confirmation the failing result is
            # returned (merge stays red).  Never raises (fail-closed) —
            # merge_queue.py has no VerifyInfraError handler.  Resolved via the
            # verify module so it stays monkeypatchable.
            #
            # Task ε: the hook takes only what the OBSERVATION needs.  Its two
            # side-effects — the merge_flake_suppressed emit and the INV-4 storm
            # streak — now happen on the DISPATCHER, driven off the
            # FlakeSuppression the returned result carries, because THIS code runs
            # wherever the worktree is and on the remote path that host has no
            # event store and a private copy of the streak counter.
            scoped = await verify.apply_merge_flake_suppression(
                scoped,
                worktree=self._merge_wt,
                config=self._config,
                module_configs=self._module_configs,
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
                # Task ε: carry the merge-flake observation through this FRESH
                # result.  The scoped red WAS observed (and possibly suppressed),
                # so the observation must still reach the dispatcher's recorder
                # even though the unscoped gate independently failed the merge.
                # Dropping it here would under-count the ledger and silently
                # disarm the INV-4 streak for exactly the compound failure most
                # likely to occur under load — and would REGRESS an emission that
                # happened inline before the recorder was split out.
                flake_suppression=getattr(scoped, 'flake_suppression', None),
            )

        return scoped


# ---------------------------------------------------------------------------
# RemoteRunner — off-host verify via git push + ssh
# ---------------------------------------------------------------------------


# Archive timestamp format — mirrors verify.py::_archive_stamp.
# Microsecond precision ensures uniqueness across back-to-back ENOSPC retries for
# the same task; the format is still lexicographically sortable.
_STDERR_ARCHIVE_TS_FMT = '%Y%m%dT%H%M%S_%fZ'

# A merge verify carries no attempt id; its archives stem with the same
# fallback the local merge path uses (verify.py::_archive_attempt_id).
_MERGE_ARCHIVE_ATTEMPT_ID = 1

# Shared base ssh options for all four RemoteRunner ssh argv sites (health,
# run_merge_verify dispatch, cancel_verify, probe_clean) — kept in one place
# so the keepalive flags can't drift apart across sites. The two keepalive
# values themselves are imported from verify_cancel, which derives the remote
# watchdog's deadline from them (task 4195), so the transport's dead-peer
# verdict and the watchdog's cannot drift apart either.
_SSH_BASE_OPTS = [
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=10',
    '-o', f'ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL}',
    '-o', f'ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}',
]

# Post-sync entry-point liveness probe (task 4539), issued by
# RemoteRunner.sync_if_stale after a mutating sync of the remote Dark-Factory
# CODE checkout.
#
# Deliberately a BARE command with NO `cd` and no absolute venv path: that is
# exactly how run_merge_verify builds its dispatch argv (`orchestrator
# verify-merge --sha ... --spec ...` handed straight to ssh), so the probe
# exercises the SAME PATH resolution the real dispatch will.  A probe that cd'd
# into the checkout, or invoked `<df_remote>/.venv/bin/orchestrator` directly,
# could answer rc=0 while the dispatch still hit rc=127.
#
# `verify-merge --help` is a pure click help print — it never touches config,
# git, or a worktree — so the probe is side-effect-free and cheap.
REMOTE_LIVENESS_CMD = 'orchestrator verify-merge --help'

# Exit status ssh reserves for its OWN errors rather than the remote command's:
# "ssh exits with the exit status of the remote command or with 255 if an error
# occurred" (ssh(1), EXIT STATUS).  A sleeping host, a refused connection, a
# ConnectTimeout and a BatchMode auth failure all land here.
#
# Load-bearing for env probing (task 4537): the probe adapter must tell "the
# host said 127, that tool is not installed there" — a real, reportable
# difference between the hosts — from "we never reached the host at all", which
# is Invariant 5's transport failure and no finding about anything.  Reading
# only rc!=0 conflates them and turns a closed laptop into a full env-skew
# alarm.  See RemoteRunner.env_probe_run.
_SSH_TRANSPORT_ERROR_RC = 255

# A plain (non-login) ssh shell gets sshd's default PATH, which omits the uv
# installer's ~/.local/bin (and its older ~/.cargo/bin).  $HOME expands on the
# REMOTE side, so this is never passed through shlex.quote.
REMOTE_TOOL_PATH_PRELUDE = 'PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'


def _sanitize_runner_name(name: str) -> str:
    """Sanitize a runner name for filesystem use in archive filenames.

    Applies the same ``/`` and `` `` → ``_`` replacement as ``_make_infix``
    in verify.py (without the leading dot, which is local-log-specific).
    Runner names originate from trusted operator config, so other path-hostile
    characters are not expected; this single policy ensures remote and local
    archive filename conventions stay aligned.
    """
    return name.replace('/', '_').replace(' ', '_')


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


def _env_probe_local_cwd() -> str | None:
    """Working directory for a LOCAL env probe: ``$HOME``, or None to inherit.

    An ``ssh <host> '<cmd>'`` probe lands in the remote account's home
    directory, so pinning the local side there keeps a cwd-SENSITIVE resolver
    walking the same shape of tree on both hosts — ``npx`` in particular walks
    up from cwd looking for ``node_modules/.bin``, and the daemon's cwd is the
    project root.  Defence in depth rather than a live fix: measured
    2026-08-29, ``npx pyright --version`` returns 1.1.413 from both the main
    checkout and ``$HOME`` (the checkout's ``node_modules/.bin`` holds only
    ``codex``).  It closes the gap against a future repo-local
    ``node_modules/.bin/pyright``.

    Returns None — meaning "inherit the parent's cwd" — when ``$HOME`` is not
    an existing directory, so a HOME-less daemon degrades to today's behaviour
    instead of raising FileNotFoundError out of every probe (which
    ``_probe_run_safe`` would then normalise to a bogus rc=127 on the LOCAL
    side only, i.e. manufacture exactly the divergence this seam exists to
    prevent).
    """
    home = os.path.expanduser('~')
    return home if os.path.isdir(home) else None


async def env_probe_local_run(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
) -> tuple[int, str, str]:
    """LOCAL env-probe seam — :func:`_default_subprocess_run` under the VERIFY env.

    Same contract as ``_default_subprocess_run`` — ``(argv, *, cwd=None) ->
    (rc, stdout, stderr)`` with both streams decoded and stripped, and a raw
    OSError for an absent binary (``_probe_run_safe`` normalises that to
    rc=127, unchanged) — differing only in WHERE it runs: the env comes from
    ``verify._target_subprocess_env``, and cwd defaults to
    :func:`_env_probe_local_cwd`.

    WHY, measured rather than asserted.  The orchestrator daemon is started by
    systemd as ``uv run --frozen --project orchestrator ...``
    (``scripts/orchestrator-dark-factory.service``), and ``/proc/<pid>/environ``
    of the live unit carries ``VIRTUAL_ENV=<root>/.venv`` with PATH fronted by
    ``<root>/.venv/bin``.  ``_default_subprocess_run`` passes no ``env=`` at
    all, so a probe issued through it inherits that verbatim and a bare
    ``python3`` resolves ``<root>/.venv/bin/python3`` — Python 3.13.9 on this
    workstation — while ``/usr/bin/python3`` is 3.12.3.  Every REAL local
    verify command is spawned through ``_target_subprocess_env`` (see
    ``verify.py``, which strips ``VIRTUAL_ENV`` and drops the venv ``bin`` from
    PATH), and the remote half of the fingerprint runs in a non-login ssh
    shell; both resolve 3.12.3.  Probing under the daemon's own environ
    therefore measures the DAEMON's activation, not the verify environment the
    detector exists to compare.

    Why the seam is fixed rather than the offending probe dropped: the
    resulting 3.13.9-vs-3.12.3 report is a blocking L1, and ``check_env``
    dedups on ``has_open_l1(_ENV_DRIFT_SENTINEL,
    category=_ENV_DIVERGENCE_CATEGORY)``.  A false alarm left open therefore
    SUPPRESSES the genuine env-divergence alarm until a human resolves it —
    the same masking failure the RunnerUnavailable/INCONCLUSIVE handling
    exists to prevent, and one that dropping ``python3`` would leave latent
    for every future probe an operator declares.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd if cwd is not None else _env_probe_local_cwd(),
        env=_target_subprocess_env(None),
    )
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout_b.decode().strip(),
        stderr_b.decode().strip(),
    )


#: Ceiling on the teardown join of the stdin-heartbeat writer thread in
#: :func:`_default_ssh_heartbeat_run`.  A WEDGE DETECTOR, not a cadence: the
#: writer's only blocking operation is a ``threading.Event.wait`` that ``stop()``
#: ends within one GIL switch, and its write end is non-blocking so ``os.write``
#: can never park on a full pipe — so 5.0s is ~6x the ~0.85s worst single-gap
#: additive scheduling delay measured at loadavg 113-178 (recorded in
#: ``test_laptop_warm_verify_boundary.py``'s ``HeartbeatWriter`` comment) and is
#: never reached in practice.
#:
#: WHO PAYS THE WAIT: not the event loop.  ``Thread.join`` is synchronous, and
#: the coroutine that tears the writer down runs on the orchestrator's single
#: shared loop — the loop whose 14.8-16.1s stalls are why the beat moved to a
#: thread in the first place.  Joining inline would hand that loop a fresh
#: self-inflicted stall of up to this bound, so the join is awaited through
#: ``asyncio.to_thread``: a wedged writer costs its own dispatch's teardown and
#: nothing else.
#:
#: Bounded at all, rather than a bare ``join()``, because an unbounded one would
#: let a single pathological writer hold a merge-lane dispatch open forever —
#: the failure class this protocol exists to remove, re-introduced at the other
#: end.  Giving up is safe: the thread is ``daemon=True`` so it cannot block
#: interpreter shutdown, and it still closes its own fd in its own ``finally``
#: whenever it does finish, so that fd can never be closed out from under a
#: reused descriptor.
HEARTBEAT_STOP_JOIN_SECS: float = 5.0


async def _default_ssh_heartbeat_run(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECS,
    start_heartbeat=start_stdin_heartbeat,
) -> tuple[int, str, str]:
    """Default ssh-dispatch subprocess helper — like :func:`_default_subprocess_run`,
    plus a stdin heartbeat (connection-death protocol, PRD §8.1).

    Opens *argv* on the read end of a pipe this coroutine owns (unlike
    :func:`_default_subprocess_run`, whose stdin is unset/inherited) and beats
    one ``verify_cancel.HEARTBEAT_TOKEN`` down the write end every
    *heartbeat_interval* seconds for the full duration of the call.  This is
    the dispatcher half of the connection-death protocol: the remote
    verify-merge watchdog (``verify_cancel.run_stdin_watchdog``) fires if beats
    stop arriving, tying the remote build's lifetime to this ssh child's
    lifetime.

    The beat is written from a dedicated OS thread
    (``verify_cancel.start_stdin_heartbeat``), NOT an asyncio task on this
    coroutine's loop.  That loop is the orchestrator's single shared one —
    scheduler, agent dispatch, merge worker, uvicorn escalation server — and a
    producer scheduled behind its stalls stops beating for exactly as long as
    the stall lasts, which the remote correctly reads as a dead channel.
    Owning a raw pipe is what makes the thread possible: ``proc.stdin`` would
    be a loop-bound ``StreamWriter`` whose ``write``/``drain`` must not be
    called from another thread, and routing through ``call_soon_threadsafe``
    would put the write back on the very loop whose stalls are the defect.

    A failed beat never raises out of the writer nor alters the returned
    ``(rc, stdout, stderr)``; ``verify_cancel.run_stdin_heartbeat`` names each
    suppressed error and why it is benign.

    The writer is stopped and joined before this returns, so it can never
    outlive its dispatch; the join is bounded by
    :data:`HEARTBEAT_STOP_JOIN_SECS` and is awaited OFF this loop (see that
    constant: joining inline would re-create, in the teardown, exactly the
    loop stall the beat was moved to a thread to survive).
    *start_heartbeat* is injectable
    (default ``verify_cancel.start_stdin_heartbeat``) so tests can observe the
    writer's lifetime, mirroring this module's ``run`` / ``ssh_run`` /
    ``id_factory`` seams.

    Deliberately does NOT use ``proc.communicate()`` (task 2309 boundary gate,
    SS9 Row 6): ``communicate()`` unconditionally calls its internal
    ``_feed_stdin(input)`` helper whenever ``self.stdin is not None`` — true
    whenever this coroutine opens with ``stdin=PIPE``, as it once did — and
    ``_feed_stdin`` closes ``proc.stdin`` right after (a no-op) write/drain,
    even when ``input`` is ``None``.  That closes the connection-death channel
    within milliseconds of spawn, racing the heartbeat writer so real
    heartbeats never reach the child — confirmed via a real child that reports
    back exactly what it read from stdin (a plain ``readline()``-based check
    is fooled: EOF unblocks a blocking read the same as a real newline would,
    so it can't tell "closed" from "heartbeat delivered").  Reading the
    streams directly (mirroring ``communicate()``'s own
    ``gather(...); await self.wait()`` shape, minus the stdin feed/close) kept
    the channel open for the full duration instead.

    Task 4195 changed that paragraph's premise without changing its
    conclusion: with ``stdin=<raw fd>`` ``proc.stdin`` is ``None``, so
    ``_feed_stdin`` never fires and ``communicate()`` is technically safe here
    again.  The direct read is retained anyway, and the record above kept,
    because reverting to ``stdin=PIPE`` is the single edit that would silently
    re-arm the defect — this paragraph is the only thing standing in its way.
    """
    # Ordering is load-bearing.  The writer is started BEFORE the spawn so that
    # exactly ONE owner exists for write_fd on every path: if
    # create_subprocess_exec raises, the outer finally stops the thread, which
    # closes the fd in its own finally, and nothing leaks.  start_stdin_heartbeat
    # puts the write end in non-blocking mode itself, so a full pipe raises
    # BlockingIOError in the writer (skipping one beat) instead of parking that
    # thread forever and wedging the teardown join below.
    read_fd, write_fd = os.pipe()
    heartbeat = start_heartbeat(write_fd, interval=heartbeat_interval)
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=read_fd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        finally:
            # The child holds its own dup.  A reader left open in the parent
            # would stop a dead child's pipe from ever raising BrokenPipeError
            # — the signal the writer's suppression is built around.
            os.close(read_fd)
        # stdout=PIPE/stderr=PIPE above guarantees these are populated; an
        # explicit check (rather than a bare assert) keeps the guard live
        # under `python -O`, which strips asserts.
        stdout_r, stderr_r = proc.stdout, proc.stderr
        if stdout_r is None or stderr_r is None:
            raise RuntimeError(
                'asyncio subprocess missing stdout/stderr pipes despite stdout=PIPE, '
                'stderr=PIPE'
            )
        stdout_b, stderr_b = await asyncio.gather(stdout_r.read(), stderr_r.read())
        await proc.wait()
    finally:
        # The writer owns write_fd and closes it as it exits; that close is what
        # delivers EOF to the remote's watchdog.  stop() is synchronous and
        # instant; the join that follows is not, so it is paid on a worker
        # thread — this coroutine's loop is the shared one (see
        # HEARTBEAT_STOP_JOIN_SECS).  A CancelledError raised at that await
        # during shutdown is harmless: stop() has already been signalled, the
        # thread is daemon=True, and it closes its own fd in its own finally.
        heartbeat.stop()
        await asyncio.to_thread(heartbeat.thread.join, HEARTBEAT_STOP_JOIN_SECS)

    return (
        proc.returncode or 0,
        stdout_b.decode().strip(),
        stderr_b.decode().strip(),
    )


@dataclass(frozen=True)
class SyncOutcome:
    """Result of a RemoteRunner contract-currency sync attempt (INV-2).

    Returned by ``RemoteRunner.sync_if_stale``.  The pool reads ``configured``
    and ``ok`` to decide fail-closed benching:

      * ``configured=False``  — the runner opted out of INV-2 auto-sync (no
        ``df_checkout_path`` / no resolvable local DF root).  Always paired with
        ``ok=True`` so the pool NEVER benches an un-migrated runner; this is the
        byte-identical-to-today pass-through shape (the default constructed
        value).
      * ``configured=True, ok=True``   — the remote DF checkout is current, or
        was successfully brought current (``synced=True``); adopt its verdict.
      * ``configured=True, ok=False``  — staleness was detected but the sync
        FAILED (or was skipped mid-dispatch); the pool benches the runner
        fail-closed (PRD §3.1).

    ``stale`` records whether a HEAD mismatch was detected; ``synced`` whether a
    pull+uv-sync actually completed ok AND left the remote with a working
    ``orchestrator`` CLI (task 4539's post-sync liveness assertion — a sync can
    exit 0 and still have destroyed the entry point).
    ``local_head``/``remote_head`` carry the
    compared shas (for the ``runner_stale`` payload / operator triage);
    ``detail`` is a short human string.  Frozen so an outcome cannot be mutated
    after the fact.
    """

    configured: bool = False
    stale: bool = False
    synced: bool = False
    ok: bool = True
    local_head: str | None = None
    remote_head: str | None = None
    detail: str | None = None


def resolve_local_df_checkout(start: Path | None = None) -> Path | None:
    """Walk up from *start* to the Dark-Factory repo root (the ``.git`` marker).

    *start* defaults to this module's own file (``Path(__file__)``), so the
    running orchestrator resolves its OWN source checkout — the trust anchor
    whose HEAD a remote runner's DF checkout is compared against (INV-2).

    Returns the first ancestor directory containing a ``.git`` entry — a
    directory in the main checkout, a FILE in a linked git worktree — or
    ``None`` when no such ancestor exists.  ``None`` is the fail-safe: callers
    treat an unresolvable local checkout as "auto-sync disabled" (opt-in),
    never as an error.
    """
    here = Path(__file__).resolve() if start is None else start
    # A file start begins the walk at its parent; a directory start is itself a
    # candidate.  Either way we ascend to the filesystem root.
    candidates = [here, *here.parents] if here.is_dir() else list(here.parents)
    for d in candidates:
        if (d / '.git').exists():
            return d
    return None


RemoteVerifySubcommand = Literal['verify-merge', 'verify-task']
# The single home of the remote verify process pattern; β's verify_cancel reuses it.
REMOTE_VERIFY_PROCESS_PATTERN: str = '|'.join(get_args(RemoteVerifySubcommand))


@dataclass(frozen=True)
class _DispatchOutcome:
    result: VerifyResult
    ssh_stderr: str
    duration_ms: int


def _merge_pass_summary_payload(
    runner_name: str,
    spec: MergeVerifySpec,
    result: VerifyResult,
    merge_sha: str,
    duration_ms: int,
) -> dict[str, Any]:
    """A remote merge GREEN's audit record: SCOPE (read from the spec, the single
    source of truth after task 2822 fix a — not the host config), RESULT and TIMING."""
    return {
        'merge_sha': merge_sha,
        'runner': runner_name,
        'passed': result.passed,
        'category': result.category,
        'duration_ms': duration_ms,
        'scope': {
            'task_files': list(spec.task_files) if spec.task_files is not None else None,
            'merge_verify_workspace': spec.merge_verify_workspace,
            'merge_verify_breadth': spec.merge_verify_breadth,
        },
    }


def _task_pass_summary_payload(
    runner_name: str,
    spec: TaskVerifySpec,
    result: VerifyResult,
    duration_ms: int,
) -> dict[str, Any]:
    """A remote task GREEN's audit record: identity, RESULT, TIMING and the spec's SCOPE."""
    return {
        'tip_sha': spec.tip_sha,
        'task_id': spec.task_id,
        'attempt_id': spec.attempt_id,
        'runner': runner_name,
        'passed': result.passed,
        'category': result.category,
        'duration_ms': duration_ms,
        'scope': {
            'task_files': list(spec.task_files),
            'force_workspace': spec.force_workspace,
            'scope_cargo': spec.scope_cargo,
            'modules': [vc.prefix for vc in spec.verify_commands],
        },
    }


class RemoteRunner:
    """Runs a merge-verify bundle on a remote host via git push + ssh.

    The remote host must have ``orchestrator verify-merge`` available (γ CLI
    contract).  The spec is shipped over ssh as a shlex-quoted JSON argument;
    stdout is parsed as a VerifyResult via result_from_json.

    Every dispatch goes through ``_dispatch`` (PRD §4.3).  A host that
    produces no usable verdict — push fail, ssh connect failure, non-zero
    exit, unparseable or wrong-shaped stdout, a no-verdict summary — raises
    RunnerUnavailable; any other parseable VerifyResult is returned unchanged,
    even passed=False or timed_out=True (PRD §A Invariant 5).

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
        main_branch: str | None = None,
        df_remote_checkout: str | None = None,
        df_local_checkout: str | Path | None = None,
        run: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
        ssh_run: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
        heartbeat_interval: float | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.name = name
        self._ssh_host = ssh_host
        self._git_remote = git_remote
        self._cwd = cwd
        self._config_path = config_path
        self._main_branch = main_branch
        # INV-2 (task 2884): the remote DF *code* checkout (where `orchestrator
        # verify-merge` resolves) and the dispatcher's own local DF root — the
        # HEAD-compare pair for contract-currency auto-sync.  Both None keeps
        # sync_if_stale a byte-identical no-op (opt-in).  Distinct from
        # _git_remote/_cwd, which are the PROJECT checkout.
        self._df_remote_checkout = df_remote_checkout
        self._df_local_checkout = df_local_checkout
        # Per-runner lock serialising the mutating sync (pull + `uv sync
        # --all-packages` + the post-sync liveness probe) so two concurrent
        # dispatches never `git pull` the same checkout at once (PRD §3.1).
        # Read-only HEAD probes run outside it.
        self._sync_lock = asyncio.Lock()
        self._run = run if run is not None else _default_subprocess_run
        # γ: connection-death heartbeat-watchdog (PRD §8.1). The ssh dispatch is
        # load-bearing (it's the call that blocks for the whole remote build), so
        # it is routed through a PARALLEL self._ssh_run seam rather than overloading
        # self._run (shared by git push/rev-parse/delete, which return only after
        # completion with no stdin hook). Resolution order: injected ssh_run ->
        # injected run (back-compat: existing tests inject only `run` and expect
        # it to serve ssh too) -> heartbeat default (production).
        self._heartbeat_interval = heartbeat_interval or HEARTBEAT_INTERVAL_SECS
        if ssh_run is not None:
            self._ssh_run = ssh_run
        elif run is not None:
            self._ssh_run = run
        else:
            async def _heartbeat_ssh_run(argv, *, cwd=None):
                return await _default_ssh_heartbeat_run(
                    argv, cwd=cwd, heartbeat_interval=self._heartbeat_interval,
                )
            self._ssh_run = _heartbeat_ssh_run
        self._id_factory = id_factory if id_factory is not None else (lambda: uuid.uuid4().hex)
        # Optional test-instrumentation hook: tests may assign a list to this
        # attribute so they can inspect all subprocess argv lists after the fact.
        self._calls: list[list[str]] = []
        # Reserved for future deduplication of the best-effort main-branch push.
        # When runner instances are long-lived (cached across calls), this
        # attribute can be used to skip the push when main has not advanced
        # since the last successful push to this remote.
        #
        # NOTE on current production behaviour: _build_remote_runners creates
        # fresh RemoteRunner instances on each _run_post_merge_verify /
        # _run_drift_check call, so this attribute is always None at dispatch
        # time and deduplication never fires.  The main push is already cheap
        # (git sends only a thin packfile when remote objects are present), so
        # the per-call round-trip is acceptable at current merge cadences.
        # Filed as a follow-up: cache runners and wire deduplication.
        self._last_pushed_main_sha: str | None = None
        # β: the in-flight request-id, so cancel_verify() can issue a targeted cancel.
        # One slot: claimed by _dispatch before its push, cleared when it unwinds.
        self._inflight_request_id: str | None = None

    async def health(self) -> bool:
        """Best-effort health probe: ``ssh <host> true``.

        Returns True when rc == 0, False otherwise.  Never raises.
        BatchMode=yes prevents interactive password prompts; ConnectTimeout=10
        bounds the TCP-connect wait so a down host is detected quickly; the
        ServerAlive keepalive options (task 2362) bound a mid-session stall too.
        """
        try:
            rc, _, _ = await self._run(
                ['ssh', *_SSH_BASE_OPTS, self._ssh_host, 'true']
            )
            return rc == 0
        except Exception:
            return False

    async def sync_if_stale(
        self,
        *,
        event_store: Any = None,
        task_id: str | None = None,
    ) -> SyncOutcome:
        """INV-2 contract-currency gate — bring the remote DF *code* checkout
        current with the dispatcher before its verdict is adopted.

        Cadence = HEAD-compare PER DISPATCH (PRD §9, architect's call): resolve
        the local DF HEAD (network-free ``git rev-parse``) and the remote DF
        checkout HEAD (over ssh).  Equal ⇒ the remote already runs the
        dispatcher's gate code — no pull.  Different-but-at-origin ⇒ ALSO current:
        the remote pulls ``--ff-only`` from ORIGIN, so when it already matches the
        dispatcher's last-fetched upstream ref it is NOT stale — the dispatcher's
        local DF HEAD may merely lead origin by unpushed commits (design_decisions
        [3]), and a pull would be a no-op.  Suppressing this case avoids re-firing
        a false ``runner_stale`` + a futile pull/uv-sync on EVERY dispatch of a
        healthy dispatcher-leads-origin runner, which would dilute the
        genuinely-frozen-checkout alert this event exists to raise (incident
        bb834dd42a).  Different AND the remote does not match origin ⇒ emit
        ``runner_stale`` and, serialised on the per-runner lock and only when NO
        verify is in flight (never ``git pull`` under a live verify), run ``git
        pull --ff-only`` + ``uv sync --all-packages`` on the remote DF checkout
        (prefixed with ``REMOTE_TOOL_PATH_PRELUDE`` so the non-login ssh shell
        finds ``uv``),
        then ASSERT the checkout is still runnable via ``REMOTE_LIVENESS_CMD``
        over ssh, emitting ``runner_synced`` (kind='df_checkout') on success.

        The sync command is ``--all-packages`` and the liveness assertion exists
        because of the same defect (task 4539): a DF checkout's root
        pyproject.toml declares a uv WORKSPACE, in which a bare ``uv sync``
        prunes the workspace members' console scripts — deleting the very
        ``orchestrator`` entry point this runner then invokes over ssh — while
        still EXITING 0.  So success is NOT keyed on the return codes alone; it
        is keyed on the codes plus a probe that the CLI still answers.  It is
        still not keyed on a post-sync HEAD-equality check — the dispatcher's
        local DF HEAD may legitimately lead origin (unpushed commits), and the
        remote pulls from origin (design_decisions[3]).

        Fail-closed and never-raises: staleness with a failed pull/uv-sync, a
        post-sync liveness probe that does not return 0, or any
        subprocess/transport error, returns ``ok=False`` so the pool benches the
        runner (PRD §3.1).  Not configured (no ``df_remote_checkout`` / no
        resolvable ``df_local_checkout``) returns ``configured=False, ok=True``
        and issues ZERO ssh/git calls — byte-identical to the pre-INV-2 path.
        ``event_store`` is None-safe (emit only when set), mirroring LocalRunner.
        """
        df_remote = self._df_remote_checkout
        df_local = self._df_local_checkout
        # Opt-in gate: either path unset ⇒ inert pass-through, no bench, no I/O.
        if not df_remote or df_local is None:
            return SyncOutcome(configured=False, ok=True)

        df_local_str = str(df_local)

        def _emit(event_type: Any, data: dict[str, Any]) -> None:
            if event_store is not None:
                event_store.emit(event_type, task_id=task_id, data=data)

        try:
            from orchestrator.event_store import EventType

            # 1) Resolve local DF HEAD (network-free) + remote DF HEAD (ssh).
            local_rc, local_out, _ = await self._run(
                ['git', 'rev-parse', 'HEAD'], cwd=df_local_str,
            )
            remote_rc, remote_out, _ = await self._run(
                ['ssh', *_SSH_BASE_OPTS, self._ssh_host,
                 f'git -C {shlex.quote(df_remote)} rev-parse HEAD'],
            )
            if local_rc != 0 or remote_rc != 0:
                return SyncOutcome(
                    configured=True, ok=False,
                    detail=f'HEAD resolve failed (local rc={local_rc}, remote rc={remote_rc})',
                )
            local_head = local_out.strip()
            remote_head = remote_out.strip()

            # 2) Current — remote already runs the dispatcher's gate code.
            if local_head == remote_head:
                return SyncOutcome(
                    configured=True, ok=True, stale=False,
                    local_head=local_head, remote_head=remote_head,
                    detail='current',
                )

            # 2b) Heads differ — but the remote pulls ``--ff-only`` from ORIGIN,
            #     so the currency contract is the dispatcher's SHARED origin ref,
            #     not its possibly-ahead local HEAD (unpushed commits,
            #     design_decisions[3]).  When the remote already matches the
            #     dispatcher's last-fetched upstream, a pull would be a no-op and
            #     the runner is NOT stale: suppress the false-positive
            #     ``runner_stale`` + the futile pull/uv-sync churn that would
            #     otherwise fire on every dispatch of a healthy
            #     dispatcher-leads-origin runner (diluting the genuinely-frozen
            #     alert — incident bb834dd42a).  Best-effort + network-free: an
            #     unresolvable upstream (rc!=0 / empty, e.g. detached HEAD or no
            #     tracking branch) falls through to the raw HEAD-mismatch stale
            #     path below — byte-identical to the pre-amendment behaviour.
            reference_head: str | None = None
            try:
                ref_rc, ref_out, _ = await self._run(
                    ['git', 'rev-parse', '@{upstream}'], cwd=df_local_str,
                )
                if ref_rc == 0:
                    reference_head = ref_out.strip() or None
            except Exception:
                reference_head = None
            if reference_head is not None and remote_head == reference_head:
                return SyncOutcome(
                    configured=True, ok=True, stale=False,
                    local_head=local_head, remote_head=remote_head,
                    detail='current (remote at origin; dispatcher leads origin)',
                )

            # 3) Stale — announce BEFORE any mutation so the bench/sync is
            #    auditable even if the sync is skipped (inflight) or fails.
            _emit(EventType.runner_stale, {
                'runner': self.name,
                'local_head': local_head,
                'remote_head': remote_head,
            })

            async with self._sync_lock:
                # Never git-pull under a live verify — it would disturb the
                # checkout the running verify is executing against.  Skip (do
                # NOT bench): the next dispatch re-checks with no verify in
                # flight; in the production pre-dispatch flow no verify is ever
                # in flight here, so INV-2's fail-closed guarantee is unaffected.
                if self.dispatch_in_flight:
                    return SyncOutcome(
                        configured=True, ok=True, stale=True, synced=False,
                        local_head=local_head, remote_head=remote_head,
                        detail='skipped: verify in flight',
                    )

                pull_rc, _, pull_err = await self._run(
                    ['ssh', *_SSH_BASE_OPTS, self._ssh_host,
                     f'git -C {shlex.quote(df_remote)} pull --ff-only'],
                )
                if pull_rc != 0:
                    return SyncOutcome(
                        configured=True, ok=False, stale=True, synced=False,
                        local_head=local_head, remote_head=remote_head,
                        detail=f'git pull --ff-only failed (rc={pull_rc}): {pull_err}',
                    )

                # `--all-packages`, NOT a bare `uv sync` (task 4539).  A
                # Dark-Factory checkout's root pyproject.toml declares a uv
                # WORKSPACE ([tool.uv.workspace].members), and in a workspace a
                # bare `uv sync` syncs only the ROOT project's environment and
                # PRUNES what the root does not declare — including the
                # workspace MEMBERS' console-script entry points, among them the
                # `orchestrator` script that run_merge_verify invokes over ssh a
                # few lines below.  Measured on the second host: before,
                # `.venv/bin/orchestrator` existed and the entry point answered
                # rc=0; after a bare `uv sync` it was GONE and the ssh dispatch
                # failed rc=127; `uv sync --all-packages` restored it.  So the
                # sync that exists to make the remote CURRENT was instead
                # DELETING the very CLI whose verdict it was preparing to adopt.
                #
                # `--all-packages` is correct for a workspace and harmless for a
                # single-package repo (uv treats such a project as a one-member
                # workspace), so it is issued unconditionally rather than being
                # made a per-runner knob nobody would ever set differently.
                # CONTRIBUTING.md already names it as the whole-repo sync, and
                # it is the same spelling `verify_cold_preprovision_command`
                # uses in dark-factory-orchestrator.yaml.
                uv_rc, _, uv_err = await self._run(
                    ['ssh', *_SSH_BASE_OPTS, self._ssh_host,
                     f'cd {shlex.quote(df_remote)} && '
                     f'{REMOTE_TOOL_PATH_PRELUDE} uv sync --all-packages'],
                )
                if uv_rc != 0:
                    return SyncOutcome(
                        configured=True, ok=False, stale=True, synced=False,
                        local_head=local_head, remote_head=remote_head,
                        detail=f'uv sync --all-packages failed (rc={uv_rc}): {uv_err}',
                    )

                # Post-sync LIVENESS assertion (task 4539).  The two return codes
                # above are NOT sufficient evidence that the host still works: the
                # destructive bare `uv sync` this call site used to issue exited 0
                # while deleting the entry point.  A sync whose rc says "fine" but
                # which left no runnable `orchestrator` must bench the runner HERE,
                # loudly and attributably — otherwise the breakage is rediscovered
                # one dispatch later as an rc=127 RunnerUnavailable, which is
                # fail-safe (the pool falls back to local) but indistinguishable
                # from ssh flakiness, so the remote host stays silently disabled.
                #
                # Scoped to the post-sync path: a current checkout and an
                # in-flight skip mutate nothing, so they issue no probe and the
                # common per-dispatch case pays zero extra ssh round-trips.
                #
                # Fail-closed like every other leg here, and the probe raising is
                # caught by the outer handler as a transport error.
                live_rc, _, live_err = await self._run(
                    ['ssh', *_SSH_BASE_OPTS, self._ssh_host, REMOTE_LIVENESS_CMD],
                )
                if live_rc != 0:
                    return SyncOutcome(
                        configured=True, ok=False, stale=True, synced=False,
                        local_head=local_head, remote_head=remote_head,
                        detail=(
                            f'post-sync liveness probe failed: '
                            f'`{REMOTE_LIVENESS_CMD}` on {self._ssh_host!r} '
                            f'returned rc={live_rc} — the sync left '
                            f'{df_remote!r} without a working orchestrator CLI: '
                            f'{live_err}'
                        ),
                    )

                # Best-effort post-sync HEAD for the runner_synced.to_head
                # payload — informational only, never gates ok.
                to_head: str | None = None
                try:
                    ph_rc, ph_out, _ = await self._run(
                        ['ssh', *_SSH_BASE_OPTS, self._ssh_host,
                         f'git -C {shlex.quote(df_remote)} rev-parse HEAD'],
                    )
                    if ph_rc == 0:
                        to_head = ph_out.strip() or None
                except Exception:
                    to_head = None

                _emit(EventType.runner_synced, {
                    'runner': self.name,
                    'kind': 'df_checkout',
                    'from_head': remote_head,
                    'to_head': to_head,
                    'forced': False,
                })
                return SyncOutcome(
                    configured=True, ok=True, stale=True, synced=True,
                    local_head=local_head, remote_head=to_head or remote_head,
                    detail='synced (df_checkout)',
                )
        except Exception as exc:
            # Any subprocess/transport error anywhere above ⇒ fail-closed.
            # Exception (not BaseException) so asyncio.CancelledError still
            # propagates.  Never raises to the caller.
            return SyncOutcome(
                configured=True, ok=False,
                detail=f'sync transport error: {exc!r}',
            )

    async def run_merge_verify(
        self,
        merge_sha: str,
        spec: MergeVerifySpec,
        *,
        task_id: str | None = None,
        archive_root: Path | None = None,
        event_store: Any = None,
    ) -> VerifyResult:
        """Run the combined merge-verify bundle on the remote host.

        The best-effort main-mirror push (``_push_main_mirror``) runs first,
        BEFORE the in-flight window opens; then ``_dispatch('verify-merge')``.

        *event_store* (INV-2, task 2884; None-safe, mirrors LocalRunner) receives
        a ``runner_synced`` (kind='project_main_mirror') event when the best-effort
        Step-0 main push had to force-mirror a diverged remote main ref.

        When the result has passed=False and both *task_id* and *archive_root*
        are provided, the remote ssh stderr is archived best-effort to
        ``<archive_root>/<task_id>/attempt-1.remote-<name>-<utc_ts>.stderr.log``
        (task 1920).  Symmetrically, when the result has passed=True and both are
        provided, a JSON pass-summary (scope+result+timing) is archived best-effort
        to ``<archive_root>/<task_id>/attempt-1.remote-<name>.pass-summary-<utc_ts>.json``
        (task 2822 fix c) so a remote GREEN is auditable.  Any archival error is
        swallowed so the VerifyResult is always returned unchanged (PRD §A Invariant 5).

        Raises RunnerUnavailable when the host produced no usable verdict.
        Returns any other VerifyResult unchanged — even passed=False or
        timed_out=True (PRD §A Invariant 5).
        """
        request_id = self._id_factory()
        await self._push_main_mirror(task_id=task_id, event_store=event_store)
        outcome = await self._dispatch(
            merge_sha, 'verify-merge', spec_to_json(spec), request_id=request_id,
        )
        result = outcome.result
        if archive_root is not None and task_id is not None:
            if not result.passed:
                self._archive_remote_red(
                    archive_root, task_id, _MERGE_ARCHIVE_ATTEMPT_ID, outcome, role='merge',
                )
            else:
                # task-2822 fix (c): a remote GREEN is auditable beside the red archives.
                self._archive_pass_summary(
                    archive_root, task_id, _MERGE_ARCHIVE_ATTEMPT_ID,
                    _merge_pass_summary_payload(
                        self.name, spec, result, merge_sha, outcome.duration_ms,
                    ),
                )
        return result

    async def run_task_verify(
        self,
        spec: TaskVerifySpec,
        *,
        task_id: str,
        archive_root: Path | None,
    ) -> VerifyResult:
        """Run a task-role verify of ``spec.tip_sha`` on the remote host (PRD §5.2).

        ``_dispatch('verify-task')``, with no main-mirror push.  With
        *archive_root*, a red archives the remote stderr, the
        ``attempt-<spec.attempt_id>.remote-<name>.{test,lint,type}-<stamp>.log``
        streams and their ``.summary-<stamp>.json`` (role 'task'); a green archives
        one ``.pass-summary-<stamp>.json``.  The result's archive_log_paths name
        exactly those workstation files and worktree_log_paths is emptied, since
        laptop paths mean nothing here.  *task_id* must equal ``spec.task_id``.
        """
        if task_id != spec.task_id:
            raise ValueError(
                f'run_task_verify task_id {task_id!r} disagrees with spec.task_id '
                f'{spec.task_id!r}'
            )
        outcome = await self._dispatch(
            spec.tip_sha, 'verify-task', task_spec_to_json(spec),
            request_id=self._id_factory(),
        )
        archived: list[Path] = []
        if archive_root is not None:
            if not outcome.result.passed:
                archived = self._archive_remote_red(
                    archive_root, task_id, spec.attempt_id, outcome, role='task',
                )
            else:
                pass_summary = self._archive_pass_summary(
                    archive_root, task_id, spec.attempt_id,
                    _task_pass_summary_payload(
                        self.name, spec, outcome.result, outcome.duration_ms,
                    ),
                )
                archived = [pass_summary] if pass_summary is not None else []
        return dataclasses.replace(
            outcome.result,
            worktree_log_paths=[],
            archive_log_paths=[str(path) for path in archived],
        )

    async def _push_main_mirror(self, *, task_id: str | None, event_store: Any) -> None:
        """Step 0 (best-effort): push main FIRST so the remote has a fresh view of it.

        A non-zero rc or OSError is logged at WARNING and swallowed — a
        non-fast-forward must not abort the verify.

        β dedup: resolve the local main tip via `git rev-parse` (network-free).
        When the resolved sha matches _last_pushed_main_sha the push is skipped —
        the remote already has this main sha, so the push is redundant.
        """
        if self._main_branch:
            # Resolve local main tip (network-free)
            resolved_main_sha: str | None = None
            try:
                rev_rc, rev_stdout, _ = await self._run(
                    ['git', 'rev-parse', self._main_branch],
                    cwd=self._cwd,
                )
                if rev_rc == 0:
                    resolved_main_sha = rev_stdout.strip() or None
            except OSError:
                pass  # rev-parse failed; push unconditionally below

            # Skip push when sha is unchanged (dedup)
            if resolved_main_sha is None or resolved_main_sha != self._last_pushed_main_sha:
                try:
                    main_rc, _, main_stderr = await self._run(
                        ['git', 'push', self._git_remote,
                         f'{self._main_branch}:refs/heads/{self._main_branch}'],
                        cwd=self._cwd,
                    )
                    if main_rc == 0:
                        if resolved_main_sha is not None:
                            self._last_pushed_main_sha = resolved_main_sha
                    else:
                        # INV-2 mirror-semantics (task 2884, PRD §3.1): the FF
                        # main push was rejected (typically non-fast-forward —
                        # the remote PROJECT main diverged, e.g. the laptop
                        # reify main since 07-20).  Retry ONCE with a force
                        # refspec to restore mirror semantics.  The FF fast
                        # path above is preserved — a force is attempted ONLY
                        # on FF failure, so a healthy remote never sees one and
                        # the divergence signal is not silently dropped.
                        force_rc, _, force_stderr = await self._run(
                            ['git', 'push', self._git_remote,
                             f'+{self._main_branch}:refs/heads/{self._main_branch}'],
                            cwd=self._cwd,
                        )
                        if force_rc == 0:
                            prior_main_sha = self._last_pushed_main_sha
                            if resolved_main_sha is not None:
                                self._last_pushed_main_sha = resolved_main_sha
                            if event_store is not None:
                                from orchestrator.event_store import EventType
                                event_store.emit(
                                    EventType.runner_synced,
                                    task_id=task_id,
                                    data={
                                        'runner': self.name,
                                        'kind': 'project_main_mirror',
                                        'from_head': prior_main_sha,
                                        'to_head': resolved_main_sha,
                                        'forced': True,
                                    },
                                )
                        else:
                            # Force ALSO failed — keep today's best-effort
                            # WARNING+swallow (the merge-sha push stays the
                            # load-bearing transport; a non-fatal main push
                            # must never abort the verify).
                            import logging as _logging
                            _logging.getLogger(__name__).warning(
                                'RemoteRunner %r: best-effort main push of %r to %r failed '
                                '(FF rc=%d: %s; force rc=%d: %s) — continuing with merge-sha push',
                                self.name, self._main_branch, self._git_remote,
                                main_rc, main_stderr, force_rc, force_stderr,
                            )
                except OSError as exc:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        'RemoteRunner %r: best-effort main push failed with OSError: %s'
                        ' — continuing',
                        self.name, exc,
                    )

    async def _dispatch(
        self,
        sha: str,
        subcommand: RemoteVerifySubcommand,
        spec_json: str,
        *,
        request_id: str,
    ) -> _DispatchOutcome:
        """Push *sha* to ``refs/merge-verify/<request_id>``, run *subcommand* over ssh, parse the verdict.

        The one transport every remote verify shares (PRD §4.3).  It claims the
        runner's single in-flight slot synchronously, before its first await,
        and clears it on unwind; a claim while another is live breaks the
        exclusive-use invariant and raises ConcurrentDispatchError, leaving
        the live claim untouched.  After a successful push the ref is deleted
        best-effort on every exit.

        Raises RunnerUnavailable, whose ``kind`` says why, whenever the host
        produced no usable verdict — a parseable no-verdict result included.
        """
        if self._inflight_request_id is not None:
            raise ConcurrentDispatchError(
                f'RemoteRunner {self.name!r} already has request '
                f'{self._inflight_request_id!r} in flight; refused request {request_id!r}'
            )
        self._inflight_request_id = request_id
        ref = f'refs/merge-verify/{request_id}'
        try:
            try:
                push_rc, _push_out, push_stderr = await self._run(
                    ['git', 'push', self._git_remote, f'{sha}:{ref}'],
                    cwd=self._cwd,
                )
            except OSError as exc:
                raise RunnerUnavailable(f'git push spawn failed: {exc}') from exc

            if push_rc != 0:
                raise RunnerUnavailable(
                    f'git push {self._git_remote} {sha}:{ref} failed'
                    f' (rc={push_rc}): {push_stderr}'
                )

            try:
                # --request-id is APPENDED so parsed[:4] stays back-compat.
                argv = ['orchestrator', subcommand, '--sha', sha, '--spec', spec_json]
                if self._config_path:
                    argv += ['--config', self._config_path]
                argv += ['--request-id', request_id]
                remote_cmd = ' '.join(shlex.quote(a) for a in argv)

                ssh_t0 = time.monotonic()
                try:
                    ssh_rc, ssh_stdout, ssh_stderr = await self._ssh_run(
                        ['ssh', *_SSH_BASE_OPTS, self._ssh_host, remote_cmd],
                    )
                except OSError as exc:
                    raise RunnerUnavailable(f'ssh spawn failed: {exc}') from exc
                duration_ms = round((time.monotonic() - ssh_t0) * 1000)

                if ssh_rc != 0:
                    raise RunnerUnavailable(
                        f'ssh {self._ssh_host} exited {ssh_rc}: {ssh_stderr}'
                    )
                return _DispatchOutcome(
                    result=self._parse_verdict(ssh_stdout),
                    ssh_stderr=ssh_stderr,
                    duration_ms=duration_ms,
                )
            finally:
                # Best-effort ref cleanup — never alters returned result nor masks exceptions
                with contextlib.suppress(Exception):
                    await self._run(
                        ['git', 'push', self._git_remote, '--delete', ref],
                        cwd=self._cwd,
                    )
        finally:
            self._inflight_request_id = None

    def _parse_verdict(self, stdout: str) -> VerifyResult:
        try:
            result = result_from_json(stdout)
        except ValueError as exc:  # json.JSONDecodeError included
            raise RunnerUnavailable(
                f'unparseable VerifyResult from {self._ssh_host!r}: {exc!r}',
                kind='unparseable',
            ) from exc
        except (TypeError, KeyError) as exc:
            raise RunnerUnavailable(
                f'VerifyResult of the wrong shape from {self._ssh_host!r}: {exc!r}',
                kind='wire_shape',
            ) from exc
        if not isinstance(result.summary, str):
            raise RunnerUnavailable(
                f'VerifyResult of the wrong shape from {self._ssh_host!r}: '
                f'summary {result.summary!r} is not a string',
                kind='wire_shape',
            )
        if verify.is_no_verdict_summary(result.summary):
            raise RunnerUnavailable(
                f'{self.name!r} returned no verdict: {result.summary}',
                kind='no_verdict',
            )
        return result

    def _archive_remote_red(
        self,
        archive_root: Path,
        task_id: str,
        attempt_id: int,
        outcome: _DispatchOutcome,
        *,
        role: Literal['merge', 'task'],
    ) -> list[Path]:
        """Archive a remote red: its ssh stderr (task 1920), then its captured streams (task 1921)."""
        archived: list[Path] = []
        if outcome.ssh_stderr.strip():
            stderr_path = self._archive_failure_stderr(
                archive_root, task_id, attempt_id, outcome.ssh_stderr,
            )
            if stderr_path is not None:
                archived.append(stderr_path)
        archived += self._archive_failure_streams(
            archive_root, task_id, attempt_id, outcome.result, role=role,
        )
        return archived

    def _archive_failure_stderr(
        self,
        archive_root: Path,
        task_id: str,
        attempt_id: int,
        stderr_text: str,
    ) -> Path | None:
        """Write *stderr_text* to a timestamped .stderr.log file under archive_root/task_id/.

        Filename: ``attempt-<attempt_id>.remote-<safe_name>-<utc_ts>.stderr.log``

        Co-located with local verify archives for side-by-side operator triage (task 1920,
        sibling of 1768).  Timestamp format and name sanitization mirror _archive_merge_verify_logs
        via _STDERR_ARCHIVE_TS_FMT and _sanitize_runner_name.

        Filesystem I/O is synchronous (mkdir + write_text), matching the local-archive convention.
        Remote stderr payloads are bounded by the ssh process output buffer; the event-loop
        blocking window is negligible at current sizes.

        Best-effort: any exception is swallowed with a WARNING (returning None) so the
        caller's VerifyResult is always returned unchanged (PRD §A Invariant 5).
        """
        import logging as _logging
        try:
            target_dir = Path(archive_root) / task_id
            target_dir.mkdir(parents=True, exist_ok=True)
            utc_ts = datetime.now(UTC).strftime(_STDERR_ARCHIVE_TS_FMT)
            safe = _sanitize_runner_name(self.name)
            path = target_dir / f'attempt-{attempt_id}.remote-{safe}-{utc_ts}.stderr.log'
            path.write_text(stderr_text, encoding='utf-8')
            return path
        except Exception as exc:
            _logging.getLogger(__name__).warning(
                'RemoteRunner %r: best-effort stderr archival failed: %s', self.name, exc,
            )
            return None

    def _archive_pass_summary(
        self,
        archive_root: Path,
        task_id: str,
        attempt_id: int,
        payload: dict[str, Any],
    ) -> Path | None:
        """Archive a PASSING remote verify's *payload* as a best-effort JSON pass-summary.

        Filename: ``attempt-<attempt_id>.remote-<safe_name>.pass-summary-<utc_ts>.json``

        Mirrors _archive_failure_stderr's best-effort shape (mkdir(parents,exist_ok),
        microsecond-UTC timestamp via _STDERR_ARCHIVE_TS_FMT, _sanitize_runner_name,
        swallow-with-WARNING returning None) so a remote GREEN is auditable
        side-by-side with the failure archives (task 2822 fix c).
        """
        import logging as _logging
        try:
            target_dir = Path(archive_root) / task_id
            target_dir.mkdir(parents=True, exist_ok=True)
            utc_ts = datetime.now(UTC).strftime(_STDERR_ARCHIVE_TS_FMT)
            safe = _sanitize_runner_name(self.name)
            path = target_dir / f'attempt-{attempt_id}.remote-{safe}.pass-summary-{utc_ts}.json'
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
            return path
        except Exception as exc:
            _logging.getLogger(__name__).warning(
                'RemoteRunner %r: best-effort pass-summary archival failed: %s', self.name, exc,
            )
            return None

    def _archive_failure_streams(
        self,
        archive_root: Path,
        task_id: str,
        attempt_id: int,
        result: VerifyResult,
        *,
        role: Literal['merge', 'task'],
    ) -> list[Path]:
        """Archive captured test/lint/type output streams to archive_root/task_id/ on failure.

        Mirrors verify._archive_merge_verify_logs by projecting the three VerifyResult
        output fields into synthetic per-stream run dicts, one per NON-EMPTY stream.
        Delegates to _archive_merge_verify_logs so filenames, the microsecond UTC timestamp,
        and the summary.json shape are byte-identical to local merge-verify archives.

        Filename convention (task 1921):
            ``attempt-<attempt_id>.remote-<safe_name>.{label}-<utc_ts>.log``
            ``attempt-<attempt_id>.remote-<safe_name>.summary-<utc_ts>.json``

        The module_prefix ``remote-<name>`` causes _make_infix to emit the
        ``.remote-<name>`` infix, co-located with and identically shaped to local
        archives (remote-origin marker, analogous to task 1920 stderr file naming).

        Synthetic run dicts use placeholder ``cmd=f'<remote {label}>'`` (non-None so
        the log file is emitted), ``rc=1``, ``started_at=''``, ``duration_secs=0.0``.
        The load-bearing distinguishability fields — ``timed_out``, ``category``,
        ``cause_hint`` — are threaded from the real VerifyResult so summary.json can
        distinguish infra_timeout from test_failure without re-running (task 1921 goal).

        Returns the written paths; ``[]`` when all three streams are empty (no
        orphan summary.json emitted).

        Best-effort: any exception is swallowed with a WARNING (returning ``[]``) so
        the caller's VerifyResult is always returned unchanged (PRD §A Invariant 5).
        _archive_merge_verify_logs already swallows OSError internally; this outer
        guard additionally swallows non-OSError failures as a backstop (task 1921 step-4).
        """
        import logging as _logging
        try:
            runs = [
                {
                    'label': label,
                    'cmd': f'<remote {label}>',
                    'rc': 1,
                    'output': output,
                    'timed_out': result.timed_out,
                    'started_at': '',
                    'duration_secs': 0.0,
                }
                for label, output in (
                    ('test', result.test_output),
                    ('lint', result.lint_output),
                    ('type', result.type_output),
                )
                if output
            ]
            if not runs:
                return []

            # Producer-side contract: the remote verify process is responsible for
            # populating result.category (e.g. 'infra_timeout' vs 'test_failure') and
            # result.timed_out=True when a timeout occurred.  Distinguishability in the
            # emitted summary.json depends on the remote host honouring this contract;
            # this method threads the fields through verbatim but cannot enforce them.
            return _archive_merge_verify_logs(
                runs,
                Path(archive_root),
                task_id,
                attempt_id,
                result.category,
                result.cause_hint,
                module_prefix=f'remote-{self.name}',
                role=role,
            )
        except Exception as exc:
            _logging.getLogger(__name__).warning(
                'RemoteRunner %r: best-effort stream archival failed: %s', self.name, exc,
            )
            return []

    async def cancel_verify(self) -> int:
        """Cancel the in-flight remote verify on the remote host.

        Returns 0 immediately (idempotent) when no ``_dispatch`` is in flight —
        including during run_merge_verify's main-mirror push, which precedes
        the claim — matching α's contract: cancel an unknown/finished id exits 0.

        Otherwise issues:
            ssh <_SSH_BASE_OPTS> <host>
                orchestrator cancel-verify --request-id <id> [--config <path>]

        Returns the ssh return code.  An OSError (host unreachable) returns a
        non-zero sentinel so the caller treats it as a cancel failure.
        """
        if self._inflight_request_id is None:
            return 0
        cmd_parts = [
            'orchestrator', 'cancel-verify',
            '--request-id', self._inflight_request_id,
        ]
        if self._config_path:
            cmd_parts += ['--config', self._config_path]
        remote_cmd = ' '.join(shlex.quote(a) for a in cmd_parts)
        try:
            rc, _, _ = await self._run(
                ['ssh', *_SSH_BASE_OPTS, self._ssh_host, remote_cmd]
            )
            return rc
        except OSError:
            return 1  # non-zero → caller treats as cancel failure

    async def probe_clean(self) -> bool:
        """Probe whether any remote verify (any RemoteVerifySubcommand) is still running.

        Issues: ssh <_SSH_BASE_OPTS> <host> pgrep -f 'verify-merge|verify-task'
        (REMOTE_VERIFY_PROCESS_PATTERN, quoted: '|' is a remote-shell metacharacter)

        Returns:
            True  — rc == 1 (pgrep found no match; host is clean)
            False — rc == 0 (process still running) or rc >= 2 (error / ssh failure)
                    Conservative: any non-1 rc keeps the slot PARKED.
        """
        try:
            rc, _, _ = await self._run(
                [
                    'ssh', *_SSH_BASE_OPTS, self._ssh_host,
                    f'pgrep -f {shlex.quote(REMOTE_VERIFY_PROCESS_PATTERN)}',
                ]
            )
            return rc == 1
        except Exception:
            return False  # fail-safe: stay PARKED on any transport error

    def env_probe_run(self) -> Callable[..., Awaitable[tuple[int, str, str]]]:
        """Return a :func:`capture_env_fingerprint`-compatible run adapter.

        The returned async callable has capture's ``(argv, *, cwd=None) ->
        (rc, stdout, stderr)`` shape, and dispatches each probe as::

            ssh <_SSH_BASE_OPTS> <host> <REMOTE_TOOL_PATH_PRELUDE> <shlex-joined argv>

        so a remote fingerprint can be captured through exactly the same
        function that captures the local one.  The rc/stdout/stderr triple is
        returned unchanged.  The prelude (task 7097) puts ``~/.local/bin`` on
        the non-login shell's PATH, as for RemoteRunner's own remote ``uv
        sync``; without it a host with uv installed reports rc=127.

        Routed through ``self._run`` (the plain seam shared with health /
        cancel_verify / probe_clean) rather than ``self._ssh_run``: probes are
        short read-only commands, not the load-bearing long build that needs
        the stdin heartbeat.  Using ``self._run`` also means every existing
        test that injects only ``run`` gets the probe path for free.

        A binary absent on the REMOTE host makes the remote shell exit 127,
        which capture records as ``'<unavailable rc=127>'`` — the same value
        the local ENOENT normalisation in :func:`_probe_run_safe` produces, so
        a tool missing on both hosts compares equal rather than as drift.

        TRANSPORT FAILURE IS RAISED, NOT RECORDED
        -----------------------------------------
        ``self._run`` defaults to :func:`_default_subprocess_run`, which
        RETURNS ``(rc, stdout, stderr)`` for every outcome and never raises —
        so an unreachable host arrives here as ssh's own rc=255, not as an
        exception.  Recorded verbatim that would render EVERY probe
        ``'<unavailable rc=255>'`` against real local versions, i.e. a
        sleeping laptop would read as total env skew.  This adapter therefore
        converts ssh-level failure into RunnerUnavailable, which is what
        ``DriftDetector.check_env`` already catches to return INCONCLUSIVE
        (Invariant 5: transport failure is not divergence) — the same
        conversion ``run_merge_verify`` performs for the dispatch itself.

        Each probe is additionally bounded by ``_SSH_PROBE_TIMEOUT_SECS``.
        ``ConnectTimeout=10`` in ``_SSH_BASE_OPTS`` bounds only the HANDSHAKE;
        a connection that establishes and then hangs would otherwise hold the
        merge-verify lease and both host slots indefinitely.  A hang converts
        to RunnerUnavailable for the same reason rc=255 does — transport
        failure is not divergence.

        Scoped to ``_SSH_TRANSPORT_ERROR_RC`` and OSError specifically, NOT to
        rc != 0: a reached host that simply lacks the tool exits 127, and that
        is a genuine reportable difference between the two hosts — the most
        likely real finding this detector exists to produce.  Broadening the
        guard would silently disarm it.
        """
        async def _run_probe(
            argv: list[str],
            *,
            cwd: str | Path | None = None,
        ) -> tuple[int, str, str]:
            remote_cmd = ' '.join(shlex.quote(a) for a in argv)
            try:
                async with asyncio.timeout(_SSH_PROBE_TIMEOUT_SECS):
                    rc, stdout, stderr = await self._run([
                        'ssh', *_SSH_BASE_OPTS, self._ssh_host,
                        f'{REMOTE_TOOL_PATH_PRELUDE} {remote_cmd}',
                    ])
            except TimeoutError as exc:
                # BEFORE the OSError clause: TimeoutError subclasses OSError.
                # A reached-then-wedged connection is a transport failure, not
                # a finding about the remote's tooling — recording it as
                # rc=124 would compare a sentinel against real local versions
                # and file a false divergence L1 over a slow link.
                raise RunnerUnavailable(
                    f'ssh {self._ssh_host} probe {remote_cmd!r} exceeded '
                    f'{_SSH_PROBE_TIMEOUT_SECS}s'
                ) from exc
            except OSError as exc:
                # Spawning the LOCAL ssh binary failed.  Must not fall through
                # to _probe_run_safe's OSError→rc=127 normalisation: that rule
                # exists to make an absent binary compare EQUAL across hosts,
                # and here it would mark every remote probe unavailable while
                # the local side reports real versions.
                raise RunnerUnavailable(f'ssh spawn failed: {exc}') from exc
            if rc == _SSH_TRANSPORT_ERROR_RC:
                raise RunnerUnavailable(
                    f'ssh {self._ssh_host} exited {rc} probing '
                    f'{remote_cmd!r}: {stderr}'
                )
            return (rc, stdout, stderr)

        return _run_probe

    @property
    def dispatch_in_flight(self) -> bool:
        """True while a ``_dispatch`` (ref push + ssh verify) is live.

        Derived from ``_inflight_request_id`` (claimed at the top of
        ``_dispatch``, cleared as it unwinds): True == the dispatch's death is
        owned by task 2362's ssh keepalive; False == either no dispatch has
        started (run_merge_verify's main-mirror push precedes the claim), or
        it has returned and the lease may still be held for post-dispatch
        work such as archiving.  Consumed by merge_queue's abort-poll
        loop (task 2566) to bound that post-dispatch, remote-lease-held
        window, which nothing else watches.
        """
        return self._inflight_request_id is not None


# ---------------------------------------------------------------------------
# Helpers for merge_queue to branch on unscoped-gate verdicts
# ---------------------------------------------------------------------------


def is_unscoped_gate_failure(vr: VerifyResult) -> bool:
    """True when vr carries a sentinel category from LocalRunner's unscoped gate."""
    return vr.category in _UNSCOPED_SENTINEL_CATEGORIES


def is_flock_contention_failure(vr: VerifyResult) -> bool:
    """True when vr is the distinguished flock-contention outcome (task 2306 α).

    Keyed on the stable FLOCK_CONTENTION_CATEGORY constant (not on
    ``contention is not None``) so the check stays robust even if the payload
    shape changes.  Consumed by merge_queue (task 2307 β) to recognize a
    laptop-side flock-guard contention VerifyResult and file a born-at-L2
    escalation instead of treating it as an ordinary verify failure.
    """
    return vr.category == FLOCK_CONTENTION_CATEGORY


def unscoped_gate_failing_subprojects(vr: VerifyResult) -> list[str]:
    """Extract the failing subproject prefixes from a sentinel VerifyResult.

    Returns the list encoded in vr.summary (comma-joined prefixes).
    """
    if not vr.summary:
        return []
    return [p.strip() for p in vr.summary.split(',') if p.strip()]


# ---------------------------------------------------------------------------
# merge_verify event honesty — retry-scope fields (task 2837, PRD D5)
# ---------------------------------------------------------------------------


def retry_scope_event_fields(verify_env: Mapping[str, str]) -> dict[str, Any]:
    """Derive the merge_verify event's retry-scope fields from ``verify_env``.

    Source-of-truth rationale (task 2837, PRD verify-retry-failed-only §D5):
    the authoritative record of whether a post-merge verify ran under the D2
    failed-only NARROWED retry contract is ``spec.verify_env`` — the env the
    orchestrator ITSELF built and shipped to reify (merge_queue.
    _build_retry_verify_env) — NOT a reify stdout marker (no such marker exists
    or is parsed anywhere in the tree, and VerifyResult carries no retry-scope
    field).  Recording the orchestrator-selected scope keeps the event honest
    (INV-2 structured-facts-at-failure) and deterministically GREENable without
    a cross-project reify capability.

    Returns a dict with two ALWAYS-present keys (mirroring the depth/speculative
    always-present convention, task 2340) so the survey's runtime mining
    (milestone 5254) reads a uniform ``data.get('retry_scope')`` with no
    per-event schema branching:
      - ``retry_scope``: ``None`` for a full/legacy verify (no
        REIFY_VERIFY_RETRY_SCOPE='failed_only'); ``'failed_only'`` for a
        narrowed retry.  ``None`` is unambiguously "full green gate", so a
        narrowed retry is never miscounted as one.
      - ``retry_subset_sizes``: ``None`` unless narrowed; otherwise a nested
        dict of the per-suite subset sizes.
    """
    if verify_env.get('REIFY_VERIFY_RETRY_SCOPE') != 'failed_only':
        return {'retry_scope': None, 'retry_subset_sizes': None}
    # Narrowed failed-only retry — per-suite subset sizes.  run_all/gui come
    # from the SPACE-delimited env values.  `merge_queue._build_retry_verify_env`
    # is the SINGLE SOURCE OF TRUTH for that format (reify word-splits both, and
    # the gui shell-safety allowlist excludes ','); counting the same way here is
    # what keeps producer and reader from drifting apart again.  Argless
    # `str.split()` already drops empties, so '' counts 0 with no extra filter.
    run_all = len(verify_env.get('REIFY_RUN_ALL_MEMBER_SUBSET', '').split())
    gui = len(verify_env.get('REIFY_GUI_RETRY_SPECS', '').split())

    def _nextest_subset_size(path_str: str | None) -> int | None:
        """Line-count a nextest filter file, or None on any read failure.

        The filter files reliably exist at emit time (verify just ran against
        them), but the read is GUARDED so an absent key, a missing/unreadable
        path, or a non-UTF-8/undecodable file degrades to None (INV-2
        structured-facts-at-failure, no silent crash) rather than aborting the
        whole merge_verify event.  The except spans (OSError, ValueError)
        because Path.read_text() decodes as UTF-8 and raises UnicodeDecodeError
        — a ValueError subclass, NOT an OSError — on non-UTF-8 bytes.
        """
        if not path_str:
            return None
        try:
            return len(Path(path_str).read_text().splitlines())
        except (OSError, ValueError):
            return None

    nextest_debug = _nextest_subset_size(verify_env.get('REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_DEBUG'))
    nextest_release = _nextest_subset_size(verify_env.get('REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_RELEASE'))
    return {
        'retry_scope': 'failed_only',
        'retry_subset_sizes': {
            'run_all': run_all,
            'gui': gui,
            'nextest_debug': nextest_debug,
            'nextest_release': nextest_release,
        },
    }


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
        archive_root: Path | None = None,
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
        # task-1920: thread archive_root into RemoteRunner.run_merge_verify so failed
        # remote-verify stderr is archived beside local merge-verify logs for triage.
        self._archive_root = archive_root
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
        depth: int | None = None,
        speculative: bool | None = None,
        chain_items: int = 1,
        chain_build_ms: int | None = None,
    ) -> VerifyResult:
        """Run the verify bundle and emit a merge_verify event.

        ``attempt`` is 0 for the first dispatch and incremented for each
        ENOSPC-retry re-dispatch.  Included in the event data so consumers
        can deduplicate multiple events for the same logical merge-verify.

        ``depth``/``speculative`` (task 2340) default to ``None`` and are
        always included in the emitted event data (never omitted), so the
        offline calibration reader can treat ``depth is None`` as "no depth
        signal" via a plain ``data.get('depth')`` — no KeyError, no
        per-event schema branching.  Bare/legacy callers that omit both
        kwargs get ``None`` for each, which is byte-identical to pre-2340
        behaviour aside from the two extra always-present keys.

        ``chain_items`` (task 3185, PRD γ decision 8) is the **count, in
        CHAIN-ITEM units, of the items contained in the tree this verify
        actually exercised** — the dispatching item is chain item #1 and each
        chained successor actually built adds one.  It is deliberately
        frontier-INDEPENDENT: a verify that chained nothing is 1 however many
        other verifies are in flight, which is what makes ``chain_items >= 2``
        a sound deep-verify discriminator.  It defaults to ``1``, deliberately
        NOT to ``None`` the way ``depth``/``speculative`` do, for two
        reasons.  (1) Semantics: a
        count of items in a verified tree has a smallest TRUTHFUL value of
        1 — every merge verify exercises at least the one item it was created
        for — so there is no "absent" state to represent, and ``None`` would
        be a lie rather than a missing signal.  (2) Contract: the PRD requires
        the field on EVERY merge verify, and several callers thread no
        telemetry kwargs at all (``merge_shadow.py:1254``/``:1368``); a
        ``None`` default would emit ``chain_items: null`` for exactly those
        and break both the contract and η1's reader
        (``scripts/merge-deep-canary-predicate.sh``).

        ``chain_build_ms`` (task 3185 amend) is the wall-clock cost of the
        deep chain build that produced this verify's tree, in milliseconds.
        Unlike ``chain_items`` it DOES default to ``None``, and correctly so:
        a verify that chained nothing paid no build, so there is a genuine
        "absent" state to represent and 0 would be a lie.  It is emitted
        always-present-but-nullable for the same reason ``depth`` is — a
        reader does a plain ``data.get('chain_build_ms')`` with no
        per-event schema branching.  It exists because the build is awaited
        INLINE on the merge worker's dispatch path, so it is a per-round
        DISPATCH STALL (no head can be finalized or landed while it runs) that
        must never be mistaken for verify time; η1 reads it alongside
        drain-time.  Non-``None`` implies ``chain_items >= 2``.

        ``chain_items`` SUPERSEDES ``depth`` as the honest depth signal: a
        firing speculation probe (task 2359) relabels ``depth`` into an
        attribution fact about a stack that was never verified, whereas
        ``chain_items`` is derived from the tree that actually ran.  See the
        ``merge_verify`` doc-comment in ``event_store.py`` for the full field
        note.

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

        # INV-2 (task 2884): pre-dispatch contract-currency gate.  A RemoteRunner's
        # verdict is adoptable only if it executed CURRENT gate logic, so bring its
        # DF *code* checkout current before dispatch (HEAD-compare + git pull/uv
        # sync inside sync_if_stale).  On a fail-closed sync (configured & not ok)
        # bench that remote (quarantine) and RE-SELECT: prefer-remote surfaces the
        # NEXT healthy remote in a multi-remote pool, then the local trust anchor
        # (2-runner pools).  Only when no distinct healthy runner remains do we
        # raise RunnerUnavailable, so the single-runner production pool's caller
        # benches + re-dispatches on a free host (merge_queue._run_inflight_verify
        # → _finalize_inflight quarantine_and_release).  A not-configured runner
        # (default None df paths) returns configured=False/ok=True → byte-identical
        # to the pre-INV-2 path.  The loop is bounded by the runner count (each
        # failed sync quarantines one distinct runner, so it always converges);
        # the for-else is a fail-closed backstop for the unreachable exhaustion.
        for _ in range(len(self._runners)):
            if not isinstance(selected, RemoteRunner):
                break
            outcome = await selected.sync_if_stale(
                event_store=self._event_store, task_id=self._task_id,
            )
            if not (outcome.configured and not outcome.ok):
                break  # sync ok, or not configured → dispatch `selected`
            # Fail-closed: bench this remote and try the next eligible runner.
            self.quarantine(selected.name)
            nxt = self._select_runner()
            if nxt is selected or (
                isinstance(nxt, RemoteRunner) and self.is_quarantined(nxt.name)
            ):
                # No distinct healthy runner remains (all remotes benched, no
                # local anchor) → single-runner production fail-closed bench.
                raise RunnerUnavailable(
                    f'runner {selected.name!r} benched: INV-2 contract-currency '
                    f'sync failed ({outcome.detail})'
                )
            selected = nxt
        else:
            # Defensive (unreachable): the bounded loop settled on a still-benched
            # remote instead of breaking/raising.  Fail-closed rather than dispatch
            # a quarantined runner.
            if isinstance(selected, RemoteRunner) and self.is_quarantined(selected.name):
                raise RunnerUnavailable(
                    f'runner {selected.name!r} benched: INV-2 contract-currency '
                    f'sync gate exhausted'
                )

        t0 = time.monotonic()
        try:
            # task-1920: thread archive_root + task_id into RemoteRunner only.
            # Existing 2-arg test doubles and LocalRunner (which archives via its own
            # constructor) are left untouched — the isinstance branch confines the
            # change to the one runner that needs it.
            # INV-2: event_store is threaded too so run_merge_verify can emit the
            # project-main mirror-push runner_synced event.
            if isinstance(selected, RemoteRunner):
                result = await selected.run_merge_verify(
                    merge_sha, spec,
                    task_id=self._task_id,
                    archive_root=self._archive_root,
                    event_store=self._event_store,
                )
            else:
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
                # RunnerUnavailable→local fallback: always 2-arg (local never archives remote stderr)
                result = await self._local.run_merge_verify(merge_sha, spec)
                actual_runner = self._local
            else:
                raise
        duration_ms = round((time.monotonic() - t0) * 1000)

        # Task 3789 (ε): re-stamp the carried observation's `runner`.
        #
        # `FlakeSuppression.runner` means WHERE the isolated re-run executed, and the
        # discriminator can only stamp 'local' — a host-RELATIVE truth that reads as a
        # lie once the observation crosses the wire.  THIS is the only scope that knows
        # which runner really ran, and it knows it only HERE, after the
        # RunnerUnavailable->local fallback above: `merge_queue` passes a `runner`
        # argument reflecting the runner it INTENDED, so recording the correction at
        # the recorder's call site would file a fallback verify's flakes against an
        # innocent remote.  θ's class-3 systemic check reads this column to tell a bad
        # HOST from a bad SUITE, so a fleet-wide 'local' would make that undecidable.
        #
        # A pure `dataclasses.replace` and nothing else: `dispatch` is a TRANSPORT
        # concern, and the ledger write / event / streak bump belong to
        # `flake_recorder` on the merge path (recording here too would double-count).
        #
        # BOTH objects `replace` touches are guarded, not just the inner one: the
        # observation must be a real `FlakeSuppression` (a payload that somehow arrived
        # as a bare dict degrades to an un-stamped observation) AND `result` must be a
        # dataclass INSTANCE, since a runner returning a Protocol-conformant fake or a
        # test double would otherwise raise `TypeError` out of the OUTER `replace` and
        # into the merge path — which has no VerifyInfraError handler.  Mirrors
        # `verify._is_attachable`: an observation is evidence ABOUT a verdict and must
        # never be able to destroy the verdict it describes.
        carried = getattr(result, 'flake_suppression', None)
        if (
            isinstance(carried, flake_ledger.FlakeSuppression)
            and dataclasses.is_dataclass(result)
            and not isinstance(result, type)
        ):
            result = dataclasses.replace(
                result,
                flake_suppression=dataclasses.replace(
                    carried, runner=actual_runner.name,
                ),
            )

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
                    'depth': depth,
                    'speculative': speculative,
                    # task 3185 (PRD γ): 1-indexed count of items in the tree
                    # actually verified -- always >= 1, never None. See
                    # dispatch()'s docstring for why the default is 1.
                    'chain_items': chain_items,
                    'chain_build_ms': chain_build_ms,
                    # task 2837 (PRD verify-retry-failed-only D5): retry_scope +
                    # retry_subset_sizes — always-present (None for a full/legacy
                    # verify), derived from the D2 failed-only contract carried in
                    # spec.verify_env, so a narrowed retry is never miscounted as
                    # a full green gate.
                    **retry_scope_event_fields(spec.verify_env),
                },
            )

        return result


# ---------------------------------------------------------------------------
# Module-level wire codec
# ---------------------------------------------------------------------------


def spec_to_json(spec: MergeVerifySpec) -> str:
    """Serialise a MergeVerifySpec to canonical JSON (sort_keys=True)."""
    return canonical_json(spec.to_dict())


def spec_from_json(s: str) -> MergeVerifySpec:
    """Deserialise a MergeVerifySpec from a JSON string."""
    return MergeVerifySpec.from_dict(json.loads(s))


def result_to_json(vr: VerifyResult) -> str:
    """Serialise a VerifyResult to canonical JSON (sort_keys=True)."""
    return canonical_json(result_to_dict(vr))


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
    return canonical_json(fp.to_dict())


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


# Single source of truth for the comparable EnvFingerprint fields, in the
# canonical order drift_dimensions is reported in.  `dimensions=` selects a
# SUBSET of these; the order of the reported tuple is always this one, never
# the caller's argument order.
_ENV_FINGERPRINT_DIMENSIONS = (
    "toolchain",
    "verify_env",
    "sccache_reachable",
    "extra_probes",
)


def _resolve_env_dimensions(dimensions: Sequence[str] | None) -> set[str]:
    """Validate a ``dimensions`` selection and resolve it to a set of fields.

    ``None`` means "all four".  Anything else must name at least one REAL
    comparable field.  Both rejections exist for the same reason: a detective
    control that compares nothing is permanently green, which is strictly worse
    than not having it — it reports parity it never measured.

    Two distinct config errors, two distinct messages:

    * an unknown NAME (``('extra_probez',)``) — a typo that would otherwise
      select nothing;
    * an EMPTY selection (``()``) — every name valid, nothing selected, which
      the unknown-name check alone cannot catch because there is no name to
      object to.

    Shared by :func:`compare_env_fingerprints` and
    :meth:`DriftDetector.check_env` so the latter can reject a bad selection
    BEFORE it spends an ssh round trip per probe on a comparison it could never
    make.
    """
    if dimensions is None:
        return set(_ENV_FINGERPRINT_DIMENSIONS)
    selected = set(dimensions)
    if not selected:
        raise ValueError(
            'dimensions must name at least one comparable field; an empty '
            'selection compares nothing and would report every pair of hosts '
            f'as faithful.  Legal dimensions are '
            f'{list(_ENV_FINGERPRINT_DIMENSIONS)!r}; pass None for all of them.'
        )
    unknown = sorted(selected - set(_ENV_FINGERPRINT_DIMENSIONS))
    if unknown:
        raise ValueError(
            f"unknown env fingerprint dimension(s): {unknown!r}; "
            f"legal dimensions are {list(_ENV_FINGERPRINT_DIMENSIONS)!r}"
        )
    return selected


def compare_env_fingerprints(
    local: EnvFingerprint,
    remote: EnvFingerprint,
    *,
    dimensions: Sequence[str] | None = None,
) -> EnvParityVerdict:
    """Compare local and remote fingerprints field by field.

    Returns an EnvParityVerdict with is_faithful=True when all compared fields
    match, and drift_dimensions listing the names of any that differ.

    Parameters
    ----------
    dimensions:
        Optional subset of :data:`_ENV_FINGERPRINT_DIMENSIONS` to compare.
        ``None`` (the default) compares all four — byte-identical to the
        pre-scoping behaviour, so existing callers are unaffected.  The drift
        detector passes ``('extra_probes',)``: ``toolchain`` and
        ``sccache_reachable`` are hardcoded Rust probes that would manufacture
        false drift on a non-Rust project, and ``verify_env`` is host-specific
        by design (jobserver FIFO path, RAM-tuned concurrency), so legitimate
        differences there would false-alarm.  A project that DOES care about
        the Rust toolchain declares rustc/cargo as explicit operator probes and
        gets the same signal through ``extra_probes``.

    Raises
    ------
    ValueError
        If ``dimensions`` names a field that is not comparable, OR is an empty
        sequence.  Silently comparing nothing would turn a config typo into a
        permanently-green detector, which is the worst possible failure for a
        detective control — and ``()`` reaches that state without a single
        misspelled name, so it is rejected as the same class of error rather
        than treated as "compare nothing, report faithful".  ``None`` (not
        ``()``) is how a caller asks for every dimension.
    """
    selected = _resolve_env_dimensions(dimensions)

    drifts: list[str] = []
    if "toolchain" in selected and local.toolchain != remote.toolchain:
        drifts.append("toolchain")
    if "verify_env" in selected and dict(local.verify_env) != dict(remote.verify_env):
        drifts.append("verify_env")
    if (
        "sccache_reachable" in selected
        and local.sccache_reachable != remote.sccache_reachable
    ):
        drifts.append("sccache_reachable")
    if "extra_probes" in selected and dict(local.extra_probes) != dict(remote.extra_probes):
        drifts.append("extra_probes")
    return EnvParityVerdict(
        is_faithful=not bool(drifts),
        drift_dimensions=tuple(drifts),
    )


# ---------------------------------------------------------------------------
# ε: capture_env_fingerprint — probe a host's verify env
# ---------------------------------------------------------------------------


# POSIX "command not found" return code.  A remote shell reached over ssh
# already returns 127 when the requested binary does not exist, whereas
# ``asyncio.create_subprocess_exec`` (the LOCAL path) raises FileNotFoundError
# for the same condition.  Normalising the local OSError to this rc is what
# makes a tool that is absent on BOTH hosts produce an IDENTICAL fingerprint
# value on each side, so "uniformly unavailable" reads as parity rather than
# as drift (or, worse, as a crashed local capture).
_PROBE_NOT_FOUND_RC = 127

# Return code ``timeout(1)`` reports for a command it had to kill.  Same
# cross-host-agreement logic as _PROBE_NOT_FOUND_RC: a probe that wedges on
# BOTH hosts records the identical value on each side, so "uniformly slow"
# reads as parity rather than as drift.
_PROBE_TIMEOUT_RC = 124

# Wall-clock ceiling on ONE probe, LOCAL side.  check_env runs inside
# _run_drift_check's `async with git_ops.merge_verify_lease(lane_dir=wt)` block
# and AFTER allocator.acquire_local()/acquire_remote() have taken both verify
# host slots, and _maybe_run_drift_check fires the coroutine with
# asyncio.create_task and applies no timeout of its own — so nothing above this
# would ever reclaim those resources from a hung probe.  The local seam
# (env_probe_local_run, and capture's own _default_subprocess_run fallback) is
# a bare create_subprocess_exec + communicate()
# with no bound at all, and the committed probe set includes
# `npx pyright --version`, which on a cold npm cache reaches the npm registry.
# 30s is generous for a --version query and short against the lease.
_PROBE_TIMEOUT_SECS = 30.0

# Same ceiling for a probe dispatched over ssh, deliberately SHORTER than
# _PROBE_TIMEOUT_SECS so the two bounds cannot race: a wedged REMOTE probe must
# deterministically surface as RunnerUnavailable (-> INCONCLUSIVE) rather than
# as a recorded '<unavailable rc=124>' compared against a real local version,
# which would read as env divergence and file a false L1 against a slow link.
# ssh's own ConnectTimeout=10 in _SSH_BASE_OPTS bounds only the HANDSHAKE; a
# connection that establishes and then hangs is bounded by this and nothing
# else.
_SSH_PROBE_TIMEOUT_SECS = 20.0


async def _probe_run_safe(
    run: Callable[..., Awaitable[tuple[int, str, str]]],
    argv: list[str],
    *,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """Await ``run(argv)`` bounded by *timeout*, normalising failures to an rc.

    An absent binary must degrade to a recorded rc, never propagate: a single
    missing probe would otherwise abort the whole fingerprint capture.  See
    ``_PROBE_NOT_FOUND_RC`` for why 127 specifically.

    A HUNG probe is the same hazard with a worse blast radius — it holds the
    merge-verify lease and both verify host slots for as long as it runs — so
    it is bounded here and normalised the same way, to ``_PROBE_TIMEOUT_RC``.

    ``timeout=None`` reads the module-level ``_PROBE_TIMEOUT_SECS`` at CALL
    time rather than binding it as a default-argument value, so the ceiling
    stays monkeypatchable (tests cannot afford to wait 30s to observe it).

    EXCEPT-CLAUSE ORDER IS LOAD-BEARING: ``TimeoutError`` is a subclass of
    ``OSError`` in Python 3, so an ``except OSError`` placed first would
    swallow the timeout and misreport a wedged probe as a missing binary.

    Cancelling the inner await can orphan an already-spawned child (the seam is
    injected, so there is no handle to kill here).  Accepted deliberately: the
    point is to stop HOLDING the lease, and the probes are short read-only
    ``--version`` queries, not builds.
    """
    bound = _PROBE_TIMEOUT_SECS if timeout is None else timeout
    try:
        async with asyncio.timeout(bound):
            return await run(argv)
    except TimeoutError:
        return (
            _PROBE_TIMEOUT_RC, '',
            f'probe {argv!r} exceeded {bound}s and was abandoned',
        )
    except OSError as exc:
        return (_PROBE_NOT_FOUND_RC, '', str(exc))


async def capture_env_fingerprint(
    run: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
    *,
    verify_env: Mapping[str, str] | None = None,
    extra_probe_specs: Sequence[tuple[str, list[str]]] = (),
    include_toolchain: bool = True,
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
    include_toolchain:
        Whether to issue the three HARDCODED Rust probes (``rustc --version``,
        ``cargo --version``, ``sccache --show-stats``).  ``True`` (the default)
        preserves every existing caller byte for byte.  ``False`` skips them
        and yields ``toolchain=''`` / ``sccache_reachable=False``, which is
        correct only when the caller will not COMPARE those fields — see
        :meth:`DriftDetector.check_env`, which derives this from its
        ``dimensions``.  The saving is not cosmetic: on the remote side each
        probe is a separate ssh invocation (``_SSH_BASE_OPTS`` sets up no
        ControlMaster), so skipping three of them turns a 4-probe check from
        seven full TCP+auth handshakes into four, serially, over a possibly-WAN
        link — plus three fewer local subprocess spawns.

    Absent binaries
    ---------------
    Every probe site is routed through :func:`_probe_run_safe`, which catches
    OSError (the FileNotFoundError ``create_subprocess_exec`` raises for a
    binary that does not exist) and normalises it to rc=127 — the same rc an
    ssh remote shell reports for "command not found".  Consequences: a
    rust-less host yields ``toolchain == ''`` and still returns a fingerprint
    rather than raising; an absent sccache yields ``sccache_reachable=False``;
    a genuinely-absent extra probe renders ``'<unavailable rc=127>'`` on BOTH
    the local and the remote side, so it compares equal instead of
    manufacturing drift.
    """
    _run = run if run is not None else _default_subprocess_run

    toolchain = ''
    sccache_reachable = False
    if include_toolchain:
        # Toolchain: rustc --version + cargo --version
        _, rustc_out, _ = await _probe_run_safe(_run, ['rustc', '--version'])
        _, cargo_out, _ = await _probe_run_safe(_run, ['cargo', '--version'])
        toolchain = (rustc_out.strip() + '\n' + cargo_out.strip()).strip()

        # sccache reachability
        sccache_rc, _, _ = await _probe_run_safe(_run, ['sccache', '--show-stats'])
        sccache_reachable = (sccache_rc == 0)

    # Extra probes
    extra_probes: dict[str, str] = {}
    for key, argv in extra_probe_specs:
        probe_rc, probe_out, _ = await _probe_run_safe(_run, list(argv))
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
    return canonical_json(report.to_dict())


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

# Sentinel task_id for the dedup'd ENV-drift escalation (task 4537).
# DELIBERATELY DISTINCT from _DRIFT_SENTINEL above, and not a category
# refinement of it.  The verdict path's dedup calls
# ``has_open_l1(_DRIFT_SENTINEL)`` with NO category argument, and
# EscalationQueue.has_open_l1 counts ANY pending L1 on that task_id when
# category is None (escalation/src/escalation/queue.py::EscalationQueue.
# has_open_l1).  Sharing one sentinel would therefore make an open env-drift
# escalation silently suppress a genuine VERDICT-divergence escalation — the
# alarm that stops unverified code landing on main — and vice versa.  Two
# sentinels make the two dedup namespaces disjoint by construction rather
# than by both filers remembering to pass a category.
_ENV_DRIFT_SENTINEL = '__env_drift__'

# Escalation category for env divergence.  A named constant rather than two
# string literals because the dedup pre-check and the filed Escalation MUST
# spell it identically.  It must also be registered in
# escalation/src/escalation/categories.py::EscalationCategory: has_open_l1 and
# submit both refuse an unregistered category, and check_env_safely would
# swallow that refusal, so the detector would never file.
_ENV_DIVERGENCE_CATEGORY = 'verify_env_divergence'


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
    local_category:  VerifyResult.category from the local runner, defaulting to ''.
    remote_category: VerifyResult.category from the remote runner, defaulting to ''.
                   Both are always populated when a comparison actually happened,
                   and both stay '' when INCONCLUSIVE (nothing was compared --
                   `verdict` is the disambiguator, exactly as it is for
                   local_passed/remote_passed, which stay None there even when
                   the local arm genuinely produced a result).  '' is ALSO the
                   ordinary value for a clean verify result carrying no sentinel
                   category, so '' never means "missing".  A non-empty value such
                   as 'merge_flake_suppressed' marks an arm whose verdict came
                   from a sentinel path rather than a clean first-pass run.
    escalated:     True when a new divergence escalation was submitted.
    quarantined:   True when the remote runner was quarantined.
    """
    merge_sha: str
    verdict: DriftVerdict
    local_passed: bool | None = None
    remote_passed: bool | None = None
    # Field order mirrors ParityRow (local_passed, remote_passed, local_category,
    # remote_category) -- the sibling record of the same two-arm comparison.
    local_category: str = ''
    remote_category: str = ''
    escalated: bool = False
    quarantined: bool = False


@dataclass(frozen=True)
class EnvDriftCheckResult:
    """Result of one DriftDetector.check_env() call (task 4537, Lever C).

    Sibling of DriftCheckResult, and deliberately a SEPARATE type: the two
    checks answer different questions (do the hosts AGREE? vs are the hosts
    the SAME?) and, crucially, have different remedies.  DriftCheckResult
    carries a `quarantined` flag; this one does not, because env divergence
    never benches a host — see check_env's fail-safe note.

    Fields
    ------
    verdict:          AGREE / DIVERGE / INCONCLUSIVE.
    is_faithful:      True when every compared dimension matched.  ALSO True
                      when the verdict is INCONCLUSIVE, i.e. when the check
                      could not be taken: a caller gating an alarm on
                      ``not is_faithful`` must not fire because a host was
                      unreachable.  Read ``verdict`` to tell "the hosts match"
                      from "we do not know".
    drift_dimensions: EnvFingerprint field names that differed (empty when
                      faithful or inconclusive).
    local_probes:     The local host's extra_probes mapping ({} when unprobed).
    remote_probes:    The remote host's extra_probes mapping ({} when unprobed).
    escalated:        True when a NEW env-divergence escalation was submitted
                      (False when deduped against an already-open L1).
    """
    verdict: DriftVerdict
    is_faithful: bool
    drift_dimensions: tuple[str, ...] = ()
    local_probes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    remote_probes: Mapping[str, str] = dataclasses.field(default_factory=dict)
    escalated: bool = False


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
        - AGREE   → emit verdict_parity_ok event (None-safe), whose data carries
          merge_sha, local_runner, remote_runner, passed, and both arms'
          local_category / remote_category ('' for a clean, sentinel-free arm).
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
        # Same defensive getattr form run_verdict_parity uses: merge_drift wraps
        # this whole check in a broad `except Exception`, so a bare .category
        # AttributeError against an odd result object would silently kill the
        # detective control rather than degrade one telemetry field.
        local_category = getattr(local_result, 'category', '')
        remote_category = getattr(remote_result, 'category', '')

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
                        # Emitted UNCONDITIONALLY ('' for a clean arm) so the
                        # payload shape is uniform across every drift parity
                        # event -- a consumer never has to tell an absent key
                        # apart from a clean result.
                        'local_category': local_category,
                        'remote_category': remote_category,
                    },
                )
            return DriftCheckResult(
                merge_sha=merge_sha,
                verdict=DriftVerdict.AGREE,
                local_passed=local_passed,
                remote_passed=remote_passed,
                local_category=local_category,
                remote_category=remote_category,
            )

        # Diverge — dedup'd escalation + quarantine.
        escalated = False
        if self._escalation_queue is not None and not self._escalation_queue.has_open_l1(_DRIFT_SENTINEL):
            from escalation.models import Escalation
            # Explain the suppression sentinel ONLY when an arm actually carries it.
            # The structured local_category=/remote_category= echo below stays
            # unconditional (that is the task's always-populated contract); it is
            # just this ~40-word operator footnote that would otherwise dilute
            # every divergence escalation with hypothetical guidance.
            # The literal is matched, not imported: the only cycle-safe home for a
            # shared constant is verify.py, which this task does not own.  A rename
            # there degrades to "footnote not shown" -- never to a lost signal, since
            # the raw category is still echoed verbatim in the structured fields.
            suppression_note = ''
            if 'merge_flake_suppressed' in (local_category, remote_category):
                suppression_note = (
                    ' A category of "merge_flake_suppressed" on either arm means that '
                    "arm's green came from an isolated flake-suppression rerun "
                    '(verify.apply_merge_flake_suppression), not a clean first-pass '
                    'run -- weigh that when deciding which host is wrong.'
                )
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
                    f'({local_passed}, local_category={local_category!r}) '
                    f'remote_runner={remote.name!r} '
                    f'({remote_passed}, remote_category={remote_category!r}). '
                    f'A remote PASS / local FAIL split can land unverified code on main.'
                    f'{suppression_note}'
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
            local_category=local_category,
            remote_category=remote_category,
            escalated=escalated,
            quarantined=True,
        )

    async def check_env(
        self,
        probe_specs: Sequence[tuple[str, list[str]]],
        *,
        local_run: Callable[..., Awaitable[tuple[int, str, str]]] | None = None,
        dimensions: Sequence[str] = ('extra_probes',),
    ) -> EnvDriftCheckResult:
        """Compare the local and remote hosts' ENVIRONMENTS, not their verdicts.

        ``check()`` above asks whether the two hosts reached the same verdict
        for one SHA.  Two hosts running different tool versions answer that
        question identically on almost every commit — and diverge exactly on
        the one that matters.  This asks the prior question: are they running
        the same tools at all?

        Captures an EnvFingerprint on each host through the SAME
        :func:`capture_env_fingerprint` (local via *local_run* /
        :func:`env_probe_local_run`, remote via ``remote.env_probe_run()``),
        then compares them under :func:`compare_env_fingerprints` scoped to
        *dimensions*.

        THE PROBE SEAM IS THE VERIFY SEAM, on both sides: the remote probes go
        through the same non-login ssh shell a remote verify runs in, and the
        local ones through the same ``verify._target_subprocess_env`` scrub a
        local verify command is spawned under.  A fingerprint captured under
        anything else measures the wrong environment and reports divergence
        that no verify run would ever experience.

        Parameters
        ----------
        probe_specs:
            ``(key, argv)`` pairs from ``config.verify_env_probes``.  Empty
            (the shipped default) means the detector is off.
        local_run:
            Injected local run seam.  None uses :func:`env_probe_local_run` —
            the VERIFY seam (``verify._target_subprocess_env``, cwd ``$HOME``),
            deliberately not ``capture_env_fingerprint``'s own
            ``_default_subprocess_run`` default, which would inherit the
            daemon's ``uv run`` venv activation and report the daemon's
            interpreter instead of the host's.
        dimensions:
            Compared EnvFingerprint fields.  Defaults to ``('extra_probes',)``
            — only what the operator explicitly declared.  toolchain and
            sccache_reachable are hardcoded RUST probes that would manufacture
            drift on a Python project, and verify_env is host-specific by
            design (jobserver FIFO path, RAM-tuned concurrency).

        Side-effects
        ------------
        AGREE → emit ``verify_env_parity_ok`` (None-safe).
        DIVERGE → emit ``verify_env_divergence`` + one deduped L1 escalation.
        INCONCLUSIVE → nothing at all: no event, no escalation, no probe.
        INCONCLUSIVE covers every reason the check could not be TAKEN — no
        probes declared, no local runner, no eligible remote (including one
        the verdict path just quarantined), a remote that is not a
        RemoteRunner, or a transport failure (an UNREACHABLE host, which ssh
        reports as rc=255 and ``RemoteRunner.env_probe_run`` converts into
        RunnerUnavailable) — never a finding about the hosts.  Alarming on any
        of those would train operators to ignore the alarm, and here it would
        do worse than that: the L1 dedups on
        ``(_ENV_DRIFT_SENTINEL, _ENV_DIVERGENCE_CATEGORY)``, so one false
        alarm filed against a sleeping laptop suppresses the genuine one until
        a human resolves it.

        Raises
        ------
        ValueError
            If *dimensions* names an unknown field or is empty.  Deliberately
            NOT folded into INCONCLUSIVE: those are host conditions, this is a
            caller/config error, and an empty selection would otherwise report
            every pair of hosts as faithful.  :meth:`check_env_safely` catches
            it and logs at WARNING, so the mistake is visible rather than a
            permanently-green detector.

        FAIL-SAFE, NOT FAIL-CLOSED
        --------------------------
        This method NEVER calls ``self._pool.quarantine``, unlike ``check()``.
        A verdict split can land unverified code on main, so benching the
        remote is proportionate there.  A probe-version skew is a loud
        operator signal, and auto-benching a verify host over a cosmetic
        difference would cost more than the divergence it reports.  Whether
        env drift should ever quarantine is an operator policy decision, not
        one for this detector to make silently.
        """
        # Validate the SELECTION before anything else, including before the
        # no-probes guard.  An unknown or empty `dimensions` is a caller error,
        # not a host condition, and the two must not be conflated: returning
        # INCONCLUSIVE for it would hide the mistake behind the same quiet
        # degradation that a sleeping laptop produces, and comparing an empty
        # selection would emit a verify_env_parity_ok affirming a parity that
        # was never measured.  Resolved here rather than left to
        # compare_env_fingerprints so the rejection costs no ssh round trips.
        selected_dimensions = _resolve_env_dimensions(dimensions)

        inconclusive = EnvDriftCheckResult(
            verdict=DriftVerdict.INCONCLUSIVE, is_faithful=True
        )

        # Guard order is load-bearing: the no-probes case is checked FIRST so
        # the shipped default (verify_env_probes: []) costs nothing and touches
        # neither host — "off" must mean off, not "probed and compared nothing".
        if not probe_specs:
            return inconclusive

        local = self._pool.local_runner
        remote = self._pool.eligible_remote()
        if local is None or remote is None:
            # Lever C not configured, or the verdict path already quarantined
            # the remote.  Mirrors check()'s same-shaped guard.
            return inconclusive

        # Duck-typed rather than a Protocol method: env_probe_run is a
        # RemoteRunner affordance, and widening the VerifyRunner Protocol for
        # it would force every stand-in and test double to grow an ssh probe
        # seam it has no use for.
        probe_run = getattr(remote, 'env_probe_run', None)
        if probe_run is None:
            return inconclusive

        # REMOTE FIRST, deliberately.  The remote capture is the one that can
        # fail — it is an ssh round trip to a host that may be asleep — while
        # the local capture spawns a real subprocess per probe (plus the three
        # hardcoded Rust ones) through capture's default seam.  Probing the
        # local host first spends all of that building a fingerprint that is
        # then discarded the moment the remote turns out to be unreachable,
        # which on the production path is EVERY drift check taken while the
        # laptop is closed.  Ordering it this way also keeps a caller that
        # supplies an unusable remote off the local machine entirely.
        # Do not PROBE what will not be COMPARED.  The three hardcoded Rust
        # probes are skipped unless `dimensions` actually reads them, which on
        # the production path (('extra_probes',)) it never does — the same
        # reasoning that scopes the comparison in the first place.  Derived
        # from the selection rather than hardcoded off so a caller that DOES
        # ask for toolchain/sccache_reachable still gets a real answer instead
        # of a silent ''/False that would compare equal on every pair of hosts.
        include_toolchain = bool(
            {'toolchain', 'sccache_reachable'} & selected_dimensions
        )
        try:
            remote_fp = await capture_env_fingerprint(
                probe_run(),
                extra_probe_specs=probe_specs,
                include_toolchain=include_toolchain,
            )
            local_fp = await capture_env_fingerprint(
                # NOT capture's own default (_default_subprocess_run): that is
                # a bare create_subprocess_exec with no env=, so it would probe
                # under the orchestrator daemon's venv activation rather than
                # under the verify environment.  See env_probe_local_run.
                local_run if local_run is not None else env_probe_local_run,
                extra_probe_specs=probe_specs,
                include_toolchain=include_toolchain,
            )
        except RunnerUnavailable:
            # Invariant 5: transport failure is not divergence.  A closed or
            # flaky laptop must never raise a false env-skew alarm — the same
            # reasoning check() applies to a failed verify dispatch.
            #
            # This arm is REACHED on the production seam, not decorative:
            # RemoteRunner.env_probe_run converts ssh's rc=255 (and an ssh
            # spawn failure) into RunnerUnavailable precisely so an unreachable
            # host lands here instead of comparing '<unavailable rc=255>'
            # against real local versions and alarming on every declared tool.
            return inconclusive

        parity = compare_env_fingerprints(local_fp, remote_fp, dimensions=dimensions)

        if parity.is_faithful:
            if self._event_store is not None:
                from orchestrator.event_store import EventType
                self._event_store.emit(
                    EventType.verify_env_parity_ok,
                    task_id=self._task_id,
                    data={
                        'local_runner': local.name,
                        'remote_runner': remote.name,
                        'probe_keys': sorted(local_fp.extra_probes),
                        'dimensions': list(dimensions),
                    },
                )
            return EnvDriftCheckResult(
                verdict=DriftVerdict.AGREE,
                is_faithful=True,
                local_probes=dict(local_fp.extra_probes),
                remote_probes=dict(remote_fp.extra_probes),
            )

        # Diverge — name the offending probes, emit, and file ONE deduped L1.
        # Only the keys that actually differ are reported: a run where pyright
        # agrees and uv skews must point at uv, not at "extra_probes changed".
        diverging_probes = {
            key: {
                'local': local_fp.extra_probes.get(key, ''),
                'remote': remote_fp.extra_probes.get(key, ''),
            }
            for key in sorted(set(local_fp.extra_probes) | set(remote_fp.extra_probes))
            if local_fp.extra_probes.get(key) != remote_fp.extra_probes.get(key)
        }

        if self._event_store is not None:
            from orchestrator.event_store import EventType
            self._event_store.emit(
                EventType.verify_env_divergence,
                task_id=self._task_id,
                data={
                    'local_runner': local.name,
                    'remote_runner': remote.name,
                    'drift_dimensions': list(parity.drift_dimensions),
                    'diverging_probes': diverging_probes,
                },
            )

        escalated = False
        if self._escalation_queue is not None and not self._escalation_queue.has_open_l1(
            _ENV_DRIFT_SENTINEL, category=_ENV_DIVERGENCE_CATEGORY
        ):
            from escalation.models import Escalation
            # diverging_probes is empty whenever a caller scopes `dimensions`
            # at something with no per-probe detail (toolchain, verify_env,
            # sccache_reachable).  The production path never does, but this is
            # a public parameter, and an L1 whose one-line summary trails off
            # after the colon tells an operator nothing.  Degrade to naming
            # the dimensions and pointing at the event payload instead.
            offenders = ', '.join(diverging_probes) or (
                'no per-probe detail for this dimension — read the '
                'verify_env_divergence event'
            )
            skew = ', '.join(
                f'{key}: local={vals["local"]!r} remote={vals["remote"]!r}'
                for key, vals in diverging_probes.items()
            ) or 'none (the divergence is outside extra_probes)'
            esc = Escalation(
                id=self._escalation_queue.make_id(_ENV_DRIFT_SENTINEL),
                task_id=_ENV_DRIFT_SENTINEL,
                agent_role='orchestrator-env-drift-detector',
                severity='blocking',
                level=1,
                category=_ENV_DIVERGENCE_CATEGORY,
                summary=(
                    f'Host env divergence in {"/".join(parity.drift_dimensions)}: '
                    f'{offenders}'
                ),
                detail=(
                    f'local_runner={local.name!r} remote_runner={remote.name!r} '
                    f'drift_dimensions={list(parity.drift_dimensions)!r}. '
                    f'Diverging probes — {skew}. '
                    f'The two hosts are not running the same tools, so their verify '
                    f'verdicts are not interchangeable: they will agree on almost '
                    f'every commit and diverge on the one where the version matters. '
                    f'The remote has NOT been quarantined (env drift is a detective '
                    f'control, not a gate).'
                ),
                suggested_action=(
                    'Pin the diverging tool to one version across both hosts (or '
                    'install the matching version on the lagging host), then re-run '
                    'the drift check to confirm parity is restored.'
                ),
            )
            self._escalation_queue.submit(esc)
            escalated = True

        # NO self._pool.quarantine call here — see this method's docstring.
        # check() quarantines because a verdict split can land unverified code
        # on main; a probe-version skew does not, and benching a verify host
        # over a cosmetic difference would cost more than the finding.
        return EnvDriftCheckResult(
            verdict=DriftVerdict.DIVERGE,
            is_faithful=False,
            drift_dimensions=parity.drift_dimensions,
            local_probes=dict(local_fp.extra_probes),
            remote_probes=dict(remote_fp.extra_probes),
            escalated=escalated,
        )

    async def check_env_safely(self, probes: Sequence[EnvProbeConfig]) -> None:
        """:meth:`check_env` over ``config.verify_env_probes``, for the land-hook drift path.

        Call it BEFORE :meth:`check`: ``check`` quarantines the remote on
        verdict divergence, after which there is no eligible remote and the
        env check could only ever report INCONCLUSIVE.

        Never raises.  The verdict check that follows is the load-bearing
        alarm, and this is a newer detective in front of it, so any failure
        here — an ssh probe hanging up, a malformed argv, a bug in
        ``check_env``, its ValueError for a bad selection — is logged at
        WARNING with the traceback and swallowed rather than allowed to skip
        ``check``.
        """
        import logging

        try:
            await self.check_env([(probe.key, list(probe.argv)) for probe in probes])
        except Exception:
            logging.getLogger(__name__).warning(
                'Env drift check failed for task %s; ignoring (detective control)',
                self._task_id,
                exc_info=True,
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
    probe_ok:          True when the ``sccache --show-stats`` subprocess exited 0;
                       False when the daemon was unreachable or returned non-zero.
                       Defaults to True (parse_sccache_stats never sets it to False;
                       capture_sccache_stats sets it based on the subprocess rc).

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
    probe_ok: bool = True

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
        # The remainder after the label must consist of EXACTLY ONE token and
        # that token must be an integer.  This rejects both parenthesised
        # variants ("Cache hits (Rust)") and suffix-word variants like
        # "Compile requests executed   N" that also start with the expected
        # prefix but carry extra words before the numeric column.
        remainder = s[len(label_expected):]
        if not remainder:
            return None
        toks = remainder.split()
        if len(toks) != 1:
            return None
        try:
            return int(toks[0])
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
    rc, stdout, _ = await _run(['sccache', '--show-stats'])
    stats = parse_sccache_stats(stdout)
    if rc != 0:
        # Daemon absent or returned non-zero — surface the failure so callers
        # can distinguish a probe error from a legitimately cold shared cache
        # (which would also yield all-zero stats but with probe_ok=True).
        return dataclasses.replace(stats, probe_ok=False)
    return stats


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
    return canonical_json(d.to_dict())


def delta_from_json(s: str) -> ColdWarmVerifyDelta:
    """Deserialize a ColdWarmVerifyDelta from a JSON string produced by delta_to_json."""
    return ColdWarmVerifyDelta.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# β: HostLease + HostAllocator — per-host slots, config-selected order (verify_host_policy), cancel-aware release
# ---------------------------------------------------------------------------

# Slot state constants
_SLOT_FREE = 'FREE'
_SLOT_BUSY = 'BUSY'
_SLOT_PARKED = 'PARKED'   # cancel-fail path: held + non-acquirable, pending pgrep probe

# Wire vocabulary for the slot constants above (task 3275).  Derived from the
# constants rather than re-spelled so the two can never drift apart.  Consumers
# of HostAllocator.host_states() see ONLY these lowercase strings, never the
# internal _SLOT_* spelling — see host_states() for the rationale.
_SLOT_WIRE = {
    _SLOT_FREE: 'free',
    _SLOT_BUSY: 'busy',
    _SLOT_PARKED: 'parked',
}


@dataclass(frozen=True)
class HostLease:
    """A held slot for a verify-host.  Returned by HostAllocator.acquire().

    Never held by the main-health probe (task 2565) — that classifier is
    leaseless by design; see :class:`HostAllocator`'s cross-ref note.

    Fields
    ------
    name      : host name (matches RemoteRunner.name, or 'local' for the local slot)
    runner    : the VerifyRunner instance assigned to this slot
    is_local  : True when name == 'local' (LocalRunner)
    """

    name: str
    runner: Any
    is_local: bool


class HostAllocator:
    """Worker-lifetime host allocator: one slot per host, config-selected order.

    Cross-ref (task 2565): the main-health probe
    (:func:`~orchestrator.verify.verify_failure_is_preexisting_on_main`) is
    intentionally leaseless and is NEVER routed through this allocator — it
    never calls ``acquire_local``/``acquire_remote``, so its classification
    work never occupies a slot here and a host's occupancy in this allocator
    never reflects a main-health probe running.

    Selection policy: chosen by ``OrchestratorConfig.verify_host_policy`` and
    supplied to :meth:`acquire` per call (task 5097 / PRD task C′).
    ``'prefer_local'`` — the default, and β decision 1's original order — takes
    local as the trust anchor and lets remotes take overflow only when local is
    busy.  ``'prefer_remote'`` inverts it, so a free remote absorbs verify load
    instead of idling while the lane serialises on local.

    The policy is NEVER captured on the allocator.  ``_ensure_host_allocator``
    caches one allocator for the worker's whole lifetime, so a captured policy
    could not change without a process restart; reading it per call is what
    makes the green-tier ``RELOADABLE_FIELDS`` registration real — a flip lands
    on the NEXT dispatch and cannot split an in-flight merge.

    The trust-anchor caveat that goes with ``'prefer_remote'`` is stated once,
    on ``config.py::OrchestratorConfig.verify_host_policy``.  This class sets
    no cadence.

    Unaffected by the policy: the leaseless main-health probe (above),
    quarantine, PARKED, cancel-aware release, and ``free_host_count``.

    Slot states
    -----------
    FREE   : slot is available for acquisition
    BUSY   : slot is held by an in-flight verify
    PARKED : cancel-fail path — held + non-acquirable, pending pgrep probe

    PARKED is distinct from the shared _runner_quarantine set (which is permanent
    until worker restart).  PARKED is transient — cleared when probe_clean() →
    True or when max_attempts is exhausted (in which case the slot stays PARKED).

    Shared quarantine
    -----------------
    The quarantine set is passed by reference so HostAllocator-driven
    (RunnerUnavailable) and DriftDetector-driven quarantines share one
    source of truth with the worker's _runner_quarantine set.
    ``clear_quarantine(name)`` discards from this same set; because it is
    shared by reference the host immediately becomes acquire_remote()-eligible
    again without an orchestrator restart.  This is the re-engagement
    mechanism used by the auto-reprobe path (task 1795).
    """

    def __init__(
        self,
        remote_runners: list,
        *,
        quarantine: set[str] | None = None,
        local_name: str = 'local',
    ) -> None:
        self._local_name = local_name
        # Preserve insertion order: remote runners keyed by name
        self._remote_runners: dict[str, Any] = {r.name: r for r in remote_runners}
        # Slot state per host name (local first, then remotes in declaration order)
        self._slots: dict[str, str] = {local_name: _SLOT_FREE}
        for r in remote_runners:
            self._slots[r.name] = _SLOT_FREE
        # Shared quarantine — by-reference so mutations are visible to the caller
        self._quarantine: set[str] = quarantine if quarantine is not None else set()

    @property
    def host_names(self) -> list[str]:
        """All managed host names in declaration order (local first, then remotes)."""
        return list(self._slots.keys())

    def _is_acquirable(self, name: str) -> bool:
        """FREE, and for a remote also not quarantined (local is the trust anchor)."""
        if self._slots[name] != _SLOT_FREE:
            return False
        return name == self._local_name or name not in self._quarantine

    def free_host_count(self) -> int:
        """Number of slots :meth:`acquire` could hand out right now."""
        return sum(1 for name in self._slots if self._is_acquirable(name))

    def is_busy(self, name: str) -> bool:
        """True when the slot for *name* is BUSY or PARKED (not FREE)."""
        return self._slots.get(name, _SLOT_FREE) != _SLOT_FREE

    def acquire_local(self, factory: Any) -> HostLease | None:
        """Try to acquire the local slot.  Returns None if the slot is not FREE."""
        if self._is_acquirable(self._local_name):
            runner = factory()
            self._slots[self._local_name] = _SLOT_BUSY
            return HostLease(name=self._local_name, runner=runner, is_local=True)
        return None

    def acquire_remote(self) -> HostLease | None:
        """Acquire the first FREE, non-quarantined, non-PARKED remote slot."""
        for name, runner in self._remote_runners.items():
            if self._is_acquirable(name):
                self._slots[name] = _SLOT_BUSY
                return HostLease(name=name, runner=runner, is_local=False)
        return None

    async def acquire(
        self, local_factory: Any, *, policy: VerifyHostPolicy,
    ) -> HostLease | None:
        """Acquire a host slot in the order *policy* asks for.

        'prefer_local' takes local when free and overflows to the first
        eligible remote; 'prefer_remote' takes the first eligible remote and
        falls back to local.  Either way, None means nothing was eligible.

        Eligibility is NOT re-checked here: acquire_remote() already demands a
        FREE, non-quarantined, non-PARKED slot, so a busy, quarantined or
        PARKED remote falls through to local on its own.
        """
        match policy:
            case 'prefer_local':
                local = self.acquire_local(local_factory)
                return local if local is not None else self.acquire_remote()
            case 'prefer_remote':
                remote = self.acquire_remote()
                if remote is not None:
                    return remote
                return self.acquire_local(local_factory)
            case _:
                assert_never(policy)

    async def release(self, lease: HostLease) -> None:
        """Release a held slot back to FREE.  Idempotent."""
        current = self._slots.get(lease.name)
        if current in (_SLOT_BUSY, _SLOT_PARKED):
            self._slots[lease.name] = _SLOT_FREE

    async def quarantine_and_release(self, lease: HostLease) -> None:
        """Add a remote host to the shared quarantine set and free its slot.

        For a local lease only the slot is freed — local is the trust anchor
        and is never added to the shared quarantine set.

        Note: wired into production by task 1762.  Called in
        ``_finalize_inflight()`` on the RUNNER_UNAVAILABLE path — quarantine
        the unhealthy remote host and free its lease before re-merge.  It is
        exercised by :class:`TestHostAllocatorQuarantine` unit tests.
        """
        if not lease.is_local:
            self._quarantine.add(lease.name)
        await self.release(lease)

    def clear_quarantine(self, name: str) -> None:
        """Remove a host from the shared quarantine set (idempotent).

        Mirrors :meth:`VerifyRunnerPool.clear_quarantine`.  Because the
        quarantine set is shared by reference with the worker's
        ``_runner_quarantine``, discarding *name* here immediately makes the
        host eligible for :meth:`acquire_remote` again — no restart required.

        Clearing a name that is not in the quarantine (including names that
        were never quarantined) is a safe no-op.
        """
        self._quarantine.discard(name)

    def readmit(self, name: str) -> None:
        """Fully re-engage host *name* in the pool: un-quarantine AND un-PARK.

        This is the auto-reprobe re-engagement primitive (task 1795 recovery,
        task 3043 strand fix).  :meth:`clear_quarantine` alone cannot recover a
        host whose slot the cancel-fail path PARKed, because
        :meth:`acquire_remote` additionally requires ``_SLOT_FREE`` — so a
        recovery that only discards the quarantine can resolve the host's L1 and
        pop its unavailability-tracker entry while leaving it non-acquirable:
        unquarantined, untracked AND unusable, invisible to every recovery
        mechanism until an orchestrator restart.  That is how a host that was
        down at orchestrator start stays out of the pool indefinitely.

        BUSY carve-out: only a PARKED slot is reset.  A BUSY slot is left BUSY,
        so re-admission can never steal a verify that is genuinely in flight
        (cf. the "Known limitation (ABA)" note on :meth:`cancel_and_release`).
        An unknown name never fabricates a slot entry — the lookup uses
        ``.get``, never ``[]``/``setdefault`` — and the local host is unaffected
        because local is never PARKED by the remote cancel-fail path.

        Caller obligation: PARK means "the cancel RPC failed, so a stale verify
        process may still be running on that host".  Callers must therefore have
        probed the host clean first — the reprobe sweep gates re-admission on
        ``health()`` AND ``probe_clean()`` — so that freeing the slot cannot
        double-dispatch onto a host still churning on a previous merge.

        Idempotent: a repeat call on an already-FREE, unquarantined host is a
        safe no-op.
        """
        self._quarantine.discard(name)
        if self._slots.get(name) == _SLOT_PARKED:
            self._slots[name] = _SLOT_FREE

    def is_quarantined(self, name: str) -> bool:
        """Return True if host *name* is currently quarantined.

        Mirrors :meth:`VerifyRunnerPool.is_quarantined`.  Because the
        quarantine set is shared **by reference** with the worker's
        ``_runner_quarantine``, this reflects every writer of that set — not
        just :meth:`quarantine_and_release`, but also the DriftDetector-driven
        verdict-divergence quarantines and the land-time per-land cross-check,
        which add names to the worker's set directly without going through the
        allocator.

        A name this allocator does not manage returns its membership in the
        shared set rather than raising — the read is pure set membership, not
        a slot lookup, so it answers for unmanaged names too (False for one
        that was never quarantined).  That is precisely why it exists
        alongside :meth:`host_states`: its production caller is
        ``SpeculativeMergeWorker._host_states_block``'s **orphan** arm (task
        3275), which classifies RU-tracked hosts that have no slot in this
        allocator.  :meth:`host_states` structurally cannot answer for those —
        it enumerates ``_slots`` — so the two are complementary, not redundant.
        """
        return name in self._quarantine

    def host_states(self) -> list[dict]:
        """Return per-host slot state + quarantine membership, one dict per host.

        Ordering matches the :attr:`host_names` property: ``_slots`` insertion
        order, i.e. local first, then remotes in declaration order.

        Each dict is ``{'name', 'is_local', 'slot_state', 'quarantined'}`` —
        a uniform schema, every key always present.

        Notes
        -----
        - The returned dicts are **plain data, not live views**: mutating them
          cannot affect allocator state, and they do not update as slots change.
        - ``slot_state`` is the lowercase wire vocabulary (``free`` / ``busy`` /
          ``parked``) from :data:`_SLOT_WIRE`, deliberately distinct from the
          internal ``_SLOT_*`` constants so a consumer can never come to depend
          on the internal spelling.
        - This is the sanctioned read path for
          :meth:`~orchestrator.merge_lane.worker.SpeculativeMergeWorker.snapshot`
          (task 3275), which must NOT reach into ``_slots`` / ``_quarantine``
          directly.  Pure read — no mutation, no I/O, no await — so it is safe
          to call from that synchronous method.
        """
        return [
            {
                'name': name,
                'is_local': name == self._local_name,
                'slot_state': _SLOT_WIRE[state],
                'quarantined': name in self._quarantine,
            }
            for name, state in self._slots.items()
        ]

    @property
    def local_name(self) -> str:
        """Name of the local (trust-anchor) host this allocator was built with.

        The O(1) read for "is this host the local anchor?".  Callers that need
        only the name must use this rather than scanning :meth:`host_states`
        for its ``is_local`` flag, which materialises one dict per host on
        every call (task 3043 amend).  :meth:`host_states` remains the
        sanctioned read when the *full* per-host block is wanted.

        Pure read of the constructor argument — never None, never mutated
        after construction.
        """
        return self._local_name

    def is_parked(self, name: str) -> bool:
        """Return True if host *name*'s slot is PARKED (held + non-acquirable).

        Exists because PARKED is a *strand* state no other accessor can cheaply
        answer for (task 3043).  When :meth:`cancel_and_release` runs against an
        unreachable host the cancel RPC returns rc != 0 and every
        ``probe_clean()`` poll fails, so on exhaustion the slot is deliberately
        left PARKED — the correct fail-closed state, since a stale verify
        process may still be running there.  But that path writes **only**
        ``_slots``: the host is NOT added to ``_quarantine``, so
        :meth:`quarantined_remote_runners` never yields it and the auto-reprobe
        path cannot even consider it.  ``SpeculativeMergeWorker`` uses this
        predicate on its release paths to detect exactly that strand and record
        the host in its unavailability tracker.

        Deliberately consistent with :meth:`host_states`'s
        ``slot_state == 'parked'`` — both read ``_slots``, and ``_SLOT_WIRE`` is
        just the wire spelling of the same state.  This is the cheap per-host
        boolean for the hot path; ``host_states()`` remains the sanctioned
        list-shaped read for snapshot consumers.

        A name this allocator does not manage returns False rather than raising.
        """
        return self._slots.get(name) == _SLOT_PARKED

    def remote_runner(self, name: str) -> Any | None:
        """Return the runner object declared for remote host *name*, else None.

        Exists because reprobe candidacy is TRACKER-driven (task 3043): the
        sweep iterates the worker's ``_runner_unavailable`` tracker and must
        resolve a runner for hosts that are unavailable but **not** in the
        quarantine set — the escaping-exception and PARKED-strand shapes.
        :meth:`quarantined_remote_runners` structurally cannot supply those; it
        filters on quarantine membership by construction.

        Resolution is independent of slot state and of quarantine membership —
        it is a pure declaration lookup.  Returns None for the local host name
        (``_remote_runners`` holds only remotes; local is the trust anchor and
        is never probed for re-admission) and None for an unknown name.
        """
        return self._remote_runners.get(name)

    def quarantined_remote_runners(self) -> list[tuple[str, Any]]:
        """Return (name, runner) pairs for remote runners currently in quarantine.

        Only remotes present in ``_remote_runners`` are checked — the local
        host is never included.  The result is in the same declaration order
        as the original ``remote_runners`` list.  Returns ``[]`` when nothing
        is quarantined.

        Used by the auto-reprobe path (task 1795) to identify candidates for
        health probing: only RU-quarantined hosts (those also in the worker's
        ``_runner_unavailable`` tracker) are ultimately probed and cleared.
        """
        return [
            (name, runner)
            for name, runner in self._remote_runners.items()
            if name in self._quarantine
        ]

    async def cancel_and_release(
        self,
        lease: HostLease | None,
        *,
        sleep: Any | None = None,
        max_attempts: int = 10,
    ) -> bool:
        """Cancel an in-flight verify and release the slot.

        Local lease:
            Release the slot immediately and return True.

        Remote lease:
            await lease.runner.cancel_verify()
            rc == 0  → release the slot, return True   (clean cancel)
            rc != 0  → PARK the slot (held + non-acquirable) and poll
                       lease.runner.probe_clean() until clean or max_attempts
                       exhausted.  Un-park + free the slot when probe returns
                       True.  Return False on the cancel-fail path regardless
                       of probe outcome.

        Parameters
        ----------
        sleep       : injected async sleep callable (defaults to asyncio.sleep).
                      Tests pass a no-op to drive the probe loop synchronously.
        max_attempts: maximum number of probe polls before giving up (slot stays
                      PARKED on exhaustion).

        Idempotency: a repeat call on a lease whose slot is already FREE
        (i.e. a prior cancel_and_release/release already ran) is a no-op
        that returns True WITHOUT re-issuing the remote cancel RPC. This
        matters because several call sites in SpeculativeMergeWorker can
        invoke cancel_and_release on the same lease (shutdown drain,
        cascade abandon, resolve/release, finalize-inflight); without this
        guard a second call would re-issue cancel_verify() on a host with
        nothing left to cancel, and a non-zero rc from that redundant call
        would PARK an already-healthy, released slot. A PARKED slot is NOT
        short-circuited — re-invoking on a PARKED slot is a legitimate
        recovery retry (re-cancel + re-probe). A ``None`` lease is also a
        no-op returning True (defensive guard for already-cleared callers).

        Known limitation (ABA): the guard keys on current slot STATE
        (FREE/BUSY/PARKED), not on lease identity — ``HostLease`` is a frozen
        dataclass with no generation/epoch token. If a lease's slot is
        released (FREE) and then re-acquired (BUSY) by a *different*
        in-flight verify before a stale ``cancel_and_release(old_lease)``
        call arrives, the guard sees BUSY, falls through, and would cancel
        the NEW occupant's verify instead of no-op'ing. This is pre-existing
        (not introduced by this guard) and out of scope here: callers must
        never invoke ``cancel_and_release`` with a lease after that lease's
        slot may have already been released and re-acquired elsewhere.
        Closing it fully would require tagging ``HostLease`` and the slot map
        with a monotonically increasing generation counter and refusing to
        cancel on a generation mismatch.

        Note: wired into production by tasks 1757 & 1762.  Called in
        ``stop()`` (shutdown drain of ``_inflight`` in-flight entries), in
        ``_verifier_loop()`` (head-failure cascade / operator-halt REQUEUED
        abandon path), and in ``_finalize_inflight()`` (finalize-head
        ``finally`` cancel-release path).  It is exercised by
        :class:`TestHostAllocatorCancelRelease`,
        :class:`TestHostAllocatorCancelFail`, and
        :class:`TestHostAllocatorCancelReleaseIdempotent` unit tests.
        """
        if lease is None:
            return True

        if sleep is None:
            import asyncio as _asyncio
            sleep = _asyncio.sleep

        if self._slots.get(lease.name) == _SLOT_FREE:
            # Already released — nothing to cancel. Short-circuit before any
            # RPC so a double-release cannot re-issue cancel_verify() and
            # PARK a healthy slot. This checks slot STATE only, not lease
            # identity, so it does not cover the ABA case (slot re-acquired
            # by a different lease between two calls) — see the docstring's
            # "Known limitation (ABA)" note.
            return True

        if lease.is_local:
            await self.release(lease)
            return True

        rc = await lease.runner.cancel_verify()
        if rc == 0:
            await self.release(lease)
            return True

        # Cancel failed: PARK the slot and poll until the host is clean
        self._slots[lease.name] = _SLOT_PARKED
        for attempt in range(max_attempts):
            clean = await lease.runner.probe_clean()
            if clean:
                self._slots[lease.name] = _SLOT_FREE
                return False
            if attempt < max_attempts - 1:
                await sleep(1.0)

        # max_attempts exhausted — slot remains PARKED (non-acquirable)
        return False
