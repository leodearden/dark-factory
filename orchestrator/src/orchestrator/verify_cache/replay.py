"""Which planned verify units replay a recorded green, decided once per verify from the dispatched tree."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from shared.git_async import run_git

from orchestrator import deployed_sha, verify_evidence
from orchestrator.config import ModuleConfig, OrchestratorConfig, VerifyCacheConfig
from orchestrator.verify_cache import keys, policy, toolchain, tree_objects
from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.modes import CacheMode, VerifyRole
from orchestrator.verify_cache.policy import CacheDecision, Decision, LegResult, Unavailable
from orchestrator.verify_cache.recorder import CacheRecord, LegRun, RecordedUnit
from orchestrator.verify_cache.registry import (
    Candidate,
    Entry,
    Registry,
    RegistryView,
    registry_db_path,
)
from orchestrator.verify_cache.remote_view import ShippedView
from orchestrator.verify_evidence import (
    COMMAND_FIELD_BY_LEG,
    GREEN_OUTCOMES,
    Leg,
    UnitOutcome,
    UnitTier,
    VerifyUnit,
    module_leg_name,
    module_unit_ids,
)
from orchestrator.verify_plan import VerifyPlan
from orchestrator.verify_spec_shared import canonical_json

logger = logging.getLogger(__name__)

#: The effective env a module's legs run under (``verify.py::_resolve_verify_env``).
LegEnv = Callable[[ModuleConfig | None], Mapping[str, str]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _deployed_code(deployed: deployed_sha.DeployedSha | None) -> str | Unavailable:
    """The code identity a dispatcher's keys carry: its deployed sha, unless that cannot name the running code."""
    if deployed is None:
        return Unavailable('no deployed sha was captured for this orchestrator process')
    if deployed.indeterminate:
        return Unavailable(
            f'deployed sha {deployed.sha} is indeterminate: HEAD moved while the verify code was imported',
        )
    if deployed.dirty_wire_path:
        return Unavailable(f'deployed sha {deployed.sha} had uncommitted wire-path edits at import')
    return deployed.sha


def _gate_checkout_code() -> str | Unavailable:
    """The sha the gate checkout holding a runner's code is pinned at, which
    ``cli.py::gate_identity`` reports to the dispatcher's handshake."""
    checkout = deployed_sha.resolve_code_checkout()
    if checkout is None:
        return Unavailable('the running orchestrator code is in no git checkout')
    try:
        return deployed_sha.checkout_head(checkout)
    except Exception as exc:
        return Unavailable(f'cannot read the gate checkout HEAD of {checkout}: {type(exc).__name__}: {exc}')


@dataclass(frozen=True)
class CacheAccess:
    """How a verify reaches recorded greens, and the code identity its keys carry: the
    dispatcher reads its *registry*; a runner holds none and decides against the view
    its dispatcher *shipped* (A5), or against nothing when none was shipped."""

    registry: Registry | None
    orchestrator_code: str | Unavailable
    clock: Callable[[], datetime] = _utc_now
    shipped: RegistryView | None = None

    @classmethod
    def on_dispatcher(cls, project_root: Path, *, deployed: deployed_sha.DeployedSha | None) -> CacheAccess:
        return cls(registry=Registry(registry_db_path(project_root)), orchestrator_code=_deployed_code(deployed))

    @classmethod
    async def on_runner(cls, view: ShippedView | None = None) -> CacheAccess:
        return cls(
            registry=None, orchestrator_code=await asyncio.to_thread(_gate_checkout_code),
            shipped=None if view is None else view.registry_view(),
        )


@dataclass(frozen=True)
class ExecutedModule:
    """What one executed module runs: its commands and its own verify_env. A None prefix is the global set."""

    prefix: str | None
    test_command: str | None
    lint_command: str | None
    type_check_command: str | None
    verify_env: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, mc: ModuleConfig | None, config: OrchestratorConfig) -> ExecutedModule:
        if mc is None:
            return cls(None, config.test_command, config.lint_command, config.type_check_command, ())
        env = tuple(sorted((mc.verify_env or {}).items()))
        return cls(mc.prefix, mc.test_command, mc.lint_command, mc.type_check_command, env)

    def unit_ids(self) -> tuple[str, ...]:
        """The unit of every leg this module runs."""
        return module_unit_ids(self.prefix, self)


def executed_spec_hash(plan: Mapping | None, executed: Sequence[ExecutedModule]) -> str:
    """The tree unit's spec: the plan's identity and every command the verify executes."""
    spec = {
        'plan': verify_evidence.plan_hash(plan),
        'executed': [dataclasses.asdict(module) for module in executed],
    }
    return hashlib.sha256(canonical_json(spec).encode()).hexdigest()


@dataclass(frozen=True)
class UnitDecision:
    """One unit's key (None when it has none) and its decision."""

    key: UnitKey | None
    decision: CacheDecision


@dataclass(frozen=True)
class ReplayPlan:
    """The cache decision for each unit a verify will run, by unit id, and for its whole tree."""

    role: VerifyRole
    modes: VerifyCacheConfig
    tree: UnitDecision
    modules: Mapping[str, UnitDecision]

    @property
    def tree_hit(self) -> bool:
        return self.tree.decision.decision is Decision.HIT

    def replayed_legs(self, prefix: str | None) -> frozenset[Leg]:
        """The legs of module *prefix* that replay a recorded green instead of running."""
        return frozenset(self._hits(prefix))

    def replayed(self, prefix: str | None) -> tuple[tuple[str, Entry | Candidate], ...]:
        """Each unit of module *prefix* that replays a recorded green, with the entry it replays."""
        return tuple(self._hits(prefix).values())

    @property
    def tree_replayed(self) -> tuple[tuple[str, Entry | Candidate], ...]:
        """The tree unit and the entry it replays, when the whole tree replays."""
        entry = self.tree.decision.entry
        return ((keys.TREE_UNIT_ID, entry),) if self.tree_hit and entry is not None else ()

    def _hits(self, prefix: str | None) -> dict[Leg, tuple[str, Entry | Candidate]]:
        hits: dict[Leg, tuple[str, Entry | Candidate]] = {}
        for leg in Leg:
            unit_id = module_leg_name(prefix, leg)
            decided = self.modules.get(unit_id)
            if decided is not None and decided.decision.decision is Decision.HIT and decided.decision.entry is not None:
                hits[leg] = (unit_id, decided.decision.entry)
        return hits

    def without_replayed(self, mc: ModuleConfig) -> ModuleConfig | None:
        """*mc* minus its replayed legs; None when nothing is left to run."""
        replayed = self.replayed_legs(mc.prefix)
        if not replayed:
            return mc
        remaining = dataclasses.replace(mc, **{COMMAND_FIELD_BY_LEG[leg]: None for leg in replayed})
        if not any(getattr(remaining, attr) for attr in COMMAND_FIELD_BY_LEG.values()):
            return None
        return remaining

    def settle(
        self, ran_units: Sequence[VerifyUnit], legs: Sequence[LegRun],
    ) -> tuple[tuple[VerifyUnit, ...], CacheRecord]:
        """The verify's units settled against these decisions, the tree unit last, and what it leaves
        for recording. A replayed unit did not run, so it is settled from its decision alone."""
        ran = {unit.unit: unit for unit in ran_units}
        leg_of = {leg_run.unit_id: leg_run.leg for leg_run in legs}
        settled = [
            _settled(unit_id, UnitTier.MODULE, decided, _outcome_of(ran.get(unit_id)), leg_of.get(unit_id))
            for unit_id, decided in self.modules.items()
            if unit_id in ran or decided.decision.decision is Decision.HIT
        ]
        settled += [
            RecordedUnit(dataclasses.replace(unit, reason='not keyed: not in the replay plan'), None, leg_of.get(unit_id), None)
            for unit_id, unit in ran.items() if unit_id not in self.modules
        ]
        tree_green = all(recorded.unit.outcome in GREEN_OUTCOMES for recorded in settled)
        tree_ran = None if self.tree_hit else UnitOutcome.RAN_GREEN if tree_green else UnitOutcome.RAN_RED
        tree = _settled(keys.TREE_UNIT_ID, UnitTier.TREE, self.tree, tree_ran, None)
        record = CacheRecord(role=self.role, modes=self.modes, legs=tuple(legs), units=tuple(settled), tree=tree)
        return (*(r.unit for r in settled), tree.unit), record


def _settled(
    unit_id: str, tier: UnitTier, decided: UnitDecision, ran: UnitOutcome | None, leg: LegResult | None,
) -> RecordedUnit:
    outcome, evidence = _settled_outcome(decided.decision, ran)
    unit = VerifyUnit(
        unit=unit_id, tier=tier, key=None if decided.key is None else decided.key.key,
        outcome=outcome, evidence_event_id=evidence, reason=decided.decision.reason,
    )
    return RecordedUnit(unit, decided.key, leg, decided.decision.entry)


def _settled_outcome(decision: CacheDecision, ran: UnitOutcome | None) -> tuple[UnitOutcome, int | None]:
    """A unit's outcome from its decision and how it ran (None: it did not run, so it replayed)."""
    evidence = None if decision.entry is None else decision.entry.evidence_event_id
    if ran is None:
        return UnitOutcome.CACHE_HIT, evidence
    green = ran in GREEN_OUTCOMES
    if decision.decision is Decision.WOULD_HIT:
        return (UnitOutcome.WOULD_HIT if green else UnitOutcome.RAN_RED), evidence
    if decision.decision in (Decision.UNAVAILABLE, Decision.DISABLED):
        return (UnitOutcome.UNAVAILABLE if green else UnitOutcome.RAN_RED), None
    return (UnitOutcome.RAN_GREEN if green else UnitOutcome.RAN_RED), None


def _outcome_of(unit: VerifyUnit | None) -> UnitOutcome | None:
    return None if unit is None else unit.outcome


async def plan_replay(
    worktree: Path,
    plan: VerifyPlan | None,
    executed: Sequence[ModuleConfig | None],
    *,
    config: OrchestratorConfig,
    role: VerifyRole,
    access: CacheAccess,
    leg_env: LegEnv,
    verify_outputs: Sequence[str] = (),
) -> ReplayPlan | None:
    """Decide every unit of a verify that runs *executed* under *plan* in *worktree*; None when
    the cache is off for *role*, before any I/O. Never raises (I2): what cannot be read is UNAVAILABLE.

    *verify_outputs* are the root-relative directories the verify itself writes into *worktree*;
    what is in them does not make the working copy differ from HEAD."""
    modes = config.verify_cache
    if modes.mode_for(role) is CacheMode.OFF:
        return None
    modules = tuple(ExecutedModule.of(mc, config) for mc in executed)
    code = access.orchestrator_code
    if isinstance(code, Unavailable):
        return _unavailable(role, modes, modules, code.reason)
    try:
        return await _decide_all(
            worktree, plan, executed, modules,
            config=config, role=role, access=access, orchestrator_code=code, leg_env=leg_env,
            verify_outputs=verify_outputs,
        )
    except Exception as exc:
        logger.warning('verify_cache.plan_replay unavailable: %s: %s', type(exc).__name__, exc)
        return _unavailable(role, modes, modules, f'{type(exc).__name__}: {exc}')


async def _decide_all(
    worktree: Path,
    plan: VerifyPlan | None,
    executed: Sequence[ModuleConfig | None],
    modules: tuple[ExecutedModule, ...],
    *,
    config: OrchestratorConfig,
    role: VerifyRole,
    access: CacheAccess,
    orchestrator_code: str,
    leg_env: LegEnv,
    verify_outputs: Sequence[str],
) -> ReplayPlan:
    modes = config.verify_cache
    changed = await _working_copy_changes(worktree, verify_outputs)
    if changed:
        noun = 'path' if changed == 1 else 'paths'
        return _unavailable(role, modes, modules, f'working copy differs from HEAD in {changed} {noun}')
    tree = await tree_objects.tree_of(worktree, 'HEAD')
    base = await toolchain.read_toolchain(
        worktree, worktree, tree, role=role, verify_env=leg_env(None),
        preprovision_command=config.verify_cold_preprovision_command,
        orchestrator_code=orchestrator_code,
    )
    plan_dict = None if plan is None else plan.to_dict()
    tree_key = keys.tree_unit_key(tree, executed_spec_hash(plan_dict, modules), base)
    module_keys = await _module_keys(worktree, tree, plan, executed, base, leg_env)
    source = access.registry if access.registry is not None else access.shipped
    view = await policy.consult(source, [tree_key, *module_keys.values()])
    never = {mc.prefix for mc in executed if mc is not None and mc.verify_cache == 'never'}
    decide = functools.partial(policy.decide, modes, role, view=view, now=access.clock())
    tree_decision = UnitDecision(tree_key, decide(tree_key, never_cache=bool(never)))
    return ReplayPlan(
        role=role, modes=modes, tree=tree_decision,
        modules=MappingProxyType(_module_decisions(modules, module_keys, tree_decision, never, decide)),
    )


async def _working_copy_changes(worktree: Path, ignored_dirs: Sequence[str]) -> int:
    """How many paths outside *ignored_dirs* the working copy differs from HEAD in, untracked ones included."""
    argv = [
        'git', '--no-optional-locks', 'status', '--porcelain=v1', '-z', '--untracked-files=normal',
        '--', ':(top)', *(f':(top,exclude){directory}' for directory in ignored_dirs),
    ]
    result = await run_git(argv, cwd=worktree, timeout=tree_objects.GIT_READ_TIMEOUT_SECS)
    if not result.ok:
        raise tree_objects.TreeReadError(
            f'git status failed in {worktree} (rc={result.returncode}, timed_out={result.timed_out}): '
            f'{result.stderr[-500:]}',
        )
    entries = 0
    origin_follows = False
    for token in (token for token in result.stdout.split('\0') if token):
        if origin_follows:
            origin_follows = False
            continue
        entries += 1
        origin_follows = token[:1] in ('R', 'C')
    return entries


async def _module_keys(
    worktree: Path,
    tree: str,
    plan: VerifyPlan | None,
    executed: Sequence[ModuleConfig | None],
    base: toolchain.ToolchainFingerprint,
    leg_env: LegEnv,
) -> dict[str, UnitKey]:
    """Each executed module's planned unit keys, under that module's own leg env."""
    if plan is None:
        return {}
    inputs = await keys.read_tree_inputs(worktree, tree, plan)
    found: dict[str, UnitKey] = {}
    for mc in executed:
        runs = () if mc is None else tuple(run for run in plan.runs if run.module_prefix == mc.prefix)
        if mc is None or not runs:
            continue
        fingerprint = dataclasses.replace(base, verify_env=tuple(leg_env(mc).items()))
        found.update((key.unit_id, key) for key in keys.module_unit_keys(inputs, VerifyPlan(runs=runs), fingerprint))
    return found


def _module_decisions(
    modules: tuple[ExecutedModule, ...],
    module_keys: Mapping[str, UnitKey],
    tree: UnitDecision,
    never: set[str],
    decide: Callable[..., CacheDecision],
) -> dict[str, UnitDecision]:
    if tree.decision.decision is Decision.HIT:
        replayed = CacheDecision(Decision.HIT, tree.decision.entry, f'the tree green replays every unit: {tree.decision.reason}')
        return {unit_id: UnitDecision(module_keys.get(unit_id), replayed) for m in modules for unit_id in m.unit_ids()}
    decided: dict[str, UnitDecision] = {}
    for module in modules:
        for unit_id in module.unit_ids():
            key = module_keys.get(unit_id)
            decision = (
                decide(key, never_cache=module.prefix in never) if key is not None
                else CacheDecision(Decision.MISS, None, 'not keyed: no planned run for this unit')
            )
            decided[unit_id] = UnitDecision(key, decision)
    return decided


def _unavailable(
    role: VerifyRole, modes: VerifyCacheConfig, modules: tuple[ExecutedModule, ...], reason: str,
) -> ReplayPlan:
    unavailable = UnitDecision(None, CacheDecision(Decision.UNAVAILABLE, None, reason))
    return ReplayPlan(
        role=role, modes=modes, tree=unavailable,
        modules=MappingProxyType({unit_id: unavailable for m in modules for unit_id in m.unit_ids()}),
    )

