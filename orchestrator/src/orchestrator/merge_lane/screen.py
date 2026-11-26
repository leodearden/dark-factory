"""The merge pre-verify screen: the manifest's checks on a merged tree, attributed base-first.

plans/merge-pre-verify-screen-prd.md decisions 4-8 are the contract:
:func:`run_screen` runs the manifest (``merge_screen.manifest``, read in the
screened tree) through ONE node-id runner call and maps each id back to its
check; :class:`MergeScreen` runs it in the warm ``_screen-merger`` lane, re-runs
a red check on the base, bounces only a defect the item introduced, and fails
open, loudly, on every instrument fault. Imports nothing from ``worker``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from escalation.categories import EscalationCategory
from escalation.models import Escalation

from orchestrator.event_store import EventType
from orchestrator.git_ops import (
    EphemeralWorktreeError,
    GitOps,
    ScreenLane,
    ScreenLaneUnavailable,
    WorktreeKind,
)
from orchestrator.merge_lane.outcome_facts import MergeBlockCategory
from orchestrator.merge_lane.types import MergeOutcome
from orchestrator.screen_manifest import (
    MANIFEST_SENTINEL,
    ManifestCheck,
    ScreenManifest,
    load_screen_manifest,
)
from orchestrator.verify import NodeIdResult, run_node_id_results

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger('orchestrator.merge_lane')

MERGE_SCREEN_REASON_PREFIX = 'Merge screen found'
MERGE_SCREEN_FAILURE_CATEGORY = MergeBlockCategory.MERGE_SCREEN.value

#: Sentinel task id of the ``merge_screen_unavailable`` alarm, its dedup key.
MERGE_SCREEN_ALARM_TASK_ID = '__merge_screen_unavailable__'
_ALARM_ROLE = 'orchestrator-merge-screen'

#: Base shas whose per-check verdicts are remembered; the oldest is evicted first.
BASE_VERDICT_CACHE_CAP = 64

#: Bound on one failing id's detail inside a check's findings.
_FINDING_MAX_CHARS = 1500

ScreenCheckStatus = Literal['pass', 'finding', 'error']
ScreenStatus = Literal['pass', 'bounce', 'base_red', 'error']
ScreenLaneKind = Literal['warm', 'fallback']


class NodeIdRunner(Protocol):
    """The node-id runner's shape: :func:`orchestrator.verify.run_node_id_results`."""

    def __call__(
        self, tree: Path, config: OrchestratorConfig, module_configs: Sequence[ModuleConfig],
        ids: Collection[str], *, concurrent_groups: bool = ...,
    ) -> Awaitable[Mapping[str, NodeIdResult]]: ...


@dataclass(frozen=True)
class ScreenCheck:
    """One manifest check's verdict on one tree.

    ``findings`` carries the failing ids' detail for a ``finding``; ``causes``
    says why an ``error`` check has no verdict. ``duration_s`` is the wall of
    the one runner call every check of the screen shares.
    """

    id: str
    status: ScreenCheckStatus
    findings: tuple[str, ...] = ()
    causes: tuple[str, ...] = ()
    duration_s: float = 0.0


@dataclass(frozen=True)
class ScreenVerdict:
    tree_sha: str
    checks: tuple[ScreenCheck, ...]

    @property
    def findings(self) -> tuple[ScreenCheck, ...]:
        return tuple(check for check in self.checks if check.status == 'finding')

    @property
    def errors(self) -> tuple[ScreenCheck, ...]:
        return tuple(check for check in self.checks if check.status == 'error')


async def run_screen(
    tree: Path,
    config: OrchestratorConfig,
    *,
    tree_sha: str,
    only: frozenset[str] | None = None,
    run_ids: NodeIdRunner = run_node_id_results,
) -> ScreenVerdict:
    """Run the manifest checks (or *only* those named) in *tree*, checked out at *tree_sha*.

    A ``fail`` id makes its check a finding; an ``error``, ``skipped`` or
    unreported id makes it an error (SKIP is not ALLOW); a manifest entry
    error is an error check of its own. Every selected id goes to ONE
    *run_ids* call with ``concurrent_groups=True`` and the configs of the
    manifest-named modules only.
    """
    registry = config.module_configs_or_empty
    manifest = _load_manifest(tree, config, registry)
    selected = tuple(check for check in manifest.checks if only is None or check.id in only)
    entry_errors = tuple(
        ScreenCheck(error.check, 'error', causes=(error.message,))
        for error in manifest.errors
        if only is None or error.check in only or error.check == MANIFEST_SENTINEL
    )
    ids = tuple(dict.fromkeys(node for check in selected for node in check.node_ids))
    if not ids:
        return ScreenVerdict(tree_sha, entry_errors)
    module_configs = [registry[module] for module in sorted({check.module for check in selected})]
    started = time.monotonic()
    results = await run_ids(tree, config, module_configs, ids, concurrent_groups=True)
    duration_s = time.monotonic() - started
    checks = tuple(_check_verdict(check, results, duration_s) for check in selected)
    return ScreenVerdict(tree_sha, checks + entry_errors)


def _load_manifest(
    tree: Path, config: OrchestratorConfig, registry: Mapping[str, ModuleConfig],
) -> ScreenManifest:
    relpath = config.merge_screen.manifest
    if relpath is None:
        raise ValueError('run_screen needs merge_screen.manifest; null is the kill switch')
    return load_screen_manifest(tree / relpath, registry)


def _check_verdict(
    check: ManifestCheck, results: Mapping[str, NodeIdResult], duration_s: float,
) -> ScreenCheck:
    outcomes = [(node, results.get(node)) for node in check.node_ids]
    failed = tuple(
        f'{node}: {result.detail[:_FINDING_MAX_CHARS]}'
        for node, result in outcomes if result is not None and result.status == 'fail'
    )
    if failed:
        return ScreenCheck(check.id, 'finding', findings=failed, duration_s=duration_s)
    no_verdict = tuple(
        _no_verdict_cause(node, result)
        for node, result in outcomes if result is None or result.status != 'pass'
    )
    if no_verdict:
        return ScreenCheck(check.id, 'error', causes=no_verdict, duration_s=duration_s)
    return ScreenCheck(check.id, 'pass', duration_s=duration_s)


def _no_verdict_cause(node: str, result: NodeIdResult | None) -> str:
    if result is None:
        return f'{node}: unreported'
    detail = f' ({result.detail[:_FINDING_MAX_CHARS]})' if result.detail else ''
    return f'{node}: {result.status}{detail}'


class ScreenSite(StrEnum):
    """Where a screen runs: the per-item merger, or a declared train's merged tip."""

    MERGER = 'merger'
    TRAIN = 'train'


@dataclass(frozen=True)
class Proceed:
    """Go on to the verify as before the screen existed."""


@dataclass(frozen=True)
class Bounce:
    """The item's own defect: resolve it with *outcome* and run no verify."""

    outcome: MergeOutcome
    check_ids: tuple[str, ...]


ScreenDecision = Proceed | Bounce

PROCEED = Proceed()


class ScreenPort(Protocol):
    async def screen_item(
        self, *, task_id: str, merged_sha: str, base_sha: str | None,
        config: OrchestratorConfig, site: ScreenSite,
    ) -> ScreenDecision: ...


async def screen_or_proceed(
    port: ScreenPort, *, task_id: str, merged_sha: str, base_sha: str | None,
    config: OrchestratorConfig, site: ScreenSite,
) -> ScreenDecision:
    """Screen through *port*; any fault in the port is a Proceed (PRD decision 8).

    The use-site guard every call site goes through, whatever the port.
    """
    try:
        return await port.screen_item(
            task_id=task_id, merged_sha=merged_sha, base_sha=base_sha, config=config, site=site,
        )
    except Exception:
        logger.warning(
            'merge screen for task %s at %s raised; the item proceeds', task_id, merged_sha,
            exc_info=True,
        )
        return PROCEED


class ScreenEventSink(Protocol):
    """Where the per-screen ``merge_screen`` event goes: :class:`orchestrator.event_store.EventStore`."""

    def emit(
        self, event_type: EventType, *, task_id: str | None = None, data: dict | None = None,
    ) -> int | None: ...


class ScreenAlarmQueue(Protocol):
    """The calls the ``merge_screen_unavailable`` alarm makes on an escalation queue."""

    def make_id(self, key: str) -> str: ...

    def submit(self, escalation: Escalation) -> object: ...

    def has_open_l1(self, task_id: str) -> bool: ...

    def get_by_task(self, task_id: str, status: str | None = None) -> Sequence[Escalation]: ...

    def resolve(
        self, escalation_id: str, resolution: str, *,
        resolved_by: str | None = None, resolution_class: str | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class _ScreenOutcome:
    """What one screen concluded; ``own`` holds the checks red on the item only."""

    status: ScreenStatus
    checks: tuple[ScreenCheck, ...] = ()
    causes: tuple[str, ...] = ()
    own: tuple[ScreenCheck, ...] = ()


def _failed(*causes: str, checks: tuple[ScreenCheck, ...] = ()) -> _ScreenOutcome:
    return _ScreenOutcome('error', checks=checks, causes=causes)


class _LaneChoice:
    """Which tree kind one screen runs in; a fallback sticks for its base re-run."""

    def __init__(self) -> None:
        self.kind: ScreenLaneKind = 'warm'
        self.fallback_reason = ''

    def fall_back(self, reason: str) -> None:
        self.kind = 'fallback'
        self.fallback_reason = reason


class MergeScreen:
    """The production :class:`ScreenPort`.

    It keeps only what must outlive one screen: a bounded per-base-sha cache
    of pass/finding verdicts, and the consecutive-error streak behind the
    ``merge_screen_unavailable`` alarm. The event sink and the alarm queue are
    optional, and a fault in either is logged and swallowed: reporting never
    costs an item its decision.
    """

    def __init__(
        self,
        git_ops: GitOps,
        event_store: ScreenEventSink | None,
        escalation_queue: ScreenAlarmQueue | None,
        *,
        run_ids: NodeIdRunner = run_node_id_results,
    ) -> None:
        self._git_ops = git_ops
        self._event_store = event_store
        self._escalation_queue = escalation_queue
        self._run_ids = run_ids
        self._base_verdicts: OrderedDict[str, dict[str, ScreenCheckStatus]] = OrderedDict()
        self._error_streak = 0

    async def screen_item(
        self, *, task_id: str, merged_sha: str, base_sha: str | None,
        config: OrchestratorConfig, site: ScreenSite,
    ) -> ScreenDecision:
        if config.merge_screen.manifest is None:
            return PROCEED
        started = time.monotonic()
        lane = _LaneChoice()
        outcome = await self._bounded_screen(merged_sha, base_sha, config, lane)
        self._track_streak(outcome)
        self._sync_alarm(outcome, config)
        self._emit(task_id, site, merged_sha, base_sha, outcome, lane, time.monotonic() - started)
        if outcome.status != 'bounce':
            return PROCEED
        return Bounce(_bounce_outcome(outcome.own), tuple(check.id for check in outcome.own))

    async def _bounded_screen(
        self, merged_sha: str, base_sha: str | None, config: OrchestratorConfig, lane: _LaneChoice,
    ) -> _ScreenOutcome:
        timeout_secs = config.merge_screen.timeout_secs
        try:
            async with asyncio.timeout(timeout_secs):
                return await self._screen(merged_sha, base_sha, config, lane)
        except TimeoutError:
            return _failed(f'screen timed out after {timeout_secs}s')
        except EphemeralWorktreeError as exc:
            return _failed(f'no screen tree: lane: {lane.fallback_reason}; fallback: {exc}')
        except Exception as exc:
            logger.warning('merge screen of %s raised; the item proceeds', merged_sha, exc_info=True)
            return _failed(f'screen raised {type(exc).__name__}: {exc}')

    async def _screen(
        self, merged_sha: str, base_sha: str | None, config: OrchestratorConfig, lane: _LaneChoice,
    ) -> _ScreenOutcome:
        item = await self._verdict_at(merged_sha, config, lane, only=None)
        if item.errors:
            return _failed(*_error_causes(item.errors), checks=item.checks)
        if not item.findings:
            return _ScreenOutcome('pass', checks=item.checks)
        if base_sha is None:
            return _failed('no base sha to attribute the findings to', checks=item.checks)
        red = frozenset(check.id for check in item.findings)
        base = await self._base_statuses(base_sha, red, config, lane)
        base_errors = sorted(check_id for check_id, status in base.items() if status == 'error')
        if base_errors:
            return _failed(
                *(f'{check_id}: no verdict on base {base_sha}' for check_id in base_errors),
                checks=item.checks,
            )
        own = tuple(check for check in item.findings if base[check.id] == 'pass')
        return _ScreenOutcome('bounce' if own else 'base_red', checks=item.checks, own=own)

    async def _verdict_at(
        self, sha: str, config: OrchestratorConfig, lane: _LaneChoice, *, only: frozenset[str] | None,
    ) -> ScreenVerdict:
        async with contextlib.AsyncExitStack() as stack:
            tree = await self._enter_tree(stack, sha, lane)
            return await run_screen(tree, config, tree_sha=sha, only=only, run_ids=self._run_ids)

    async def _enter_tree(self, stack: contextlib.AsyncExitStack, sha: str, lane: _LaneChoice) -> Path:
        if lane.kind == 'warm':
            try:
                return await stack.enter_async_context(self._git_ops.screen_lane(ScreenLane.MERGER, sha))
            except (ScreenLaneUnavailable, RuntimeError) as exc:
                lane.fall_back(str(exc))
        return await stack.enter_async_context(
            self._git_ops.ephemeral_worktree(WorktreeKind.MERGE_SCREEN, sha),
        )

    async def _base_statuses(
        self, base_sha: str, check_ids: frozenset[str], config: OrchestratorConfig, lane: _LaneChoice,
    ) -> dict[str, ScreenCheckStatus]:
        cached = self._base_verdicts.get(base_sha, {})
        statuses: dict[str, ScreenCheckStatus] = {
            check_id: cached[check_id] for check_id in check_ids & cached.keys()
        }
        missing = check_ids - cached.keys()
        if missing:
            verdict = await self._verdict_at(base_sha, config, lane, only=missing)
            fresh: dict[str, ScreenCheckStatus] = {
                check.id: check.status for check in verdict.checks if check.id in missing
            }
            for check_id in missing:
                statuses[check_id] = fresh.get(check_id, 'error')
            self._remember(base_sha, {k: v for k, v in fresh.items() if v != 'error'})
        return statuses

    def _remember(self, base_sha: str, statuses: dict[str, ScreenCheckStatus]) -> None:
        if not statuses:
            return
        self._base_verdicts.setdefault(base_sha, {}).update(statuses)
        self._base_verdicts.move_to_end(base_sha)
        while len(self._base_verdicts) > BASE_VERDICT_CACHE_CAP:
            self._base_verdicts.popitem(last=False)

    def _track_streak(self, outcome: _ScreenOutcome) -> None:
        self._error_streak = self._error_streak + 1 if outcome.status == 'error' else 0

    def _sync_alarm(self, outcome: _ScreenOutcome, config: OrchestratorConfig) -> None:
        queue = self._escalation_queue
        if queue is None:
            return
        try:
            if outcome.status != 'error':
                _close_alarm(queue)
            elif self._error_streak >= config.merge_screen.error_streak_alarm:
                self._raise_alarm(queue, outcome.causes)
        except Exception:
            logger.warning('merge screen alarm sync failed; the screen goes on', exc_info=True)

    def _raise_alarm(self, queue: ScreenAlarmQueue, causes: tuple[str, ...]) -> None:
        if queue.has_open_l1(MERGE_SCREEN_ALARM_TASK_ID):
            return
        last_cause = causes[0] if causes else 'unknown'
        queue.submit(Escalation(
            id=queue.make_id(MERGE_SCREEN_ALARM_TASK_ID),
            task_id=MERGE_SCREEN_ALARM_TASK_ID,
            agent_role=_ALARM_ROLE,
            severity='info',
            level=1,
            category=EscalationCategory.MERGE_SCREEN_UNAVAILABLE.value,
            summary=(
                f'Merge screen errored on {self._error_streak} consecutive screens; '
                f'last cause: {last_cause}'
            ),
            detail=(
                'Every errored screen let its item proceed to verify (fail open), so '
                'nothing is blocked, but the screen is catching nothing.\n'
                'Causes of the last screen:\n' + '\n'.join(causes)
            ),
            suggested_action=(
                'Fix the cause named above, or switch the screen off with a config '
                'reload setting merge_screen.manifest: null. The next clean screen '
                'resolves this alarm.'
            ),
        ))

    def _emit(
        self, task_id: str, site: ScreenSite, merged_sha: str, base_sha: str | None,
        outcome: _ScreenOutcome, lane: _LaneChoice, wall_s: float,
    ) -> None:
        sink = self._event_store
        if sink is None:
            return
        try:
            sink.emit(EventType.merge_screen, task_id=task_id, data={
                'site': site.value,
                'status': outcome.status,
                'merged_sha': merged_sha,
                'base_sha': base_sha,
                'lane': lane.kind,
                'fallback_reason': lane.fallback_reason,
                'checks': [
                    {'id': check.id, 'status': check.status, 'duration_s': check.duration_s}
                    for check in outcome.checks
                ],
                'errors': list(outcome.causes),
                'wall_s': wall_s,
                'error_streak': self._error_streak,
            })
        except Exception:
            logger.warning('merge screen event emit failed; the screen goes on', exc_info=True)


def _close_alarm(queue: ScreenAlarmQueue) -> None:
    for esc in queue.get_by_task(MERGE_SCREEN_ALARM_TASK_ID, status='pending'):
        queue.resolve(
            esc.id, 'merge screen ran clean again', resolved_by=_ALARM_ROLE,
            resolution_class='benign',
        )


def _error_causes(checks: tuple[ScreenCheck, ...]) -> tuple[str, ...]:
    return tuple(f'{check.id}: {cause}' for check in checks for cause in check.causes)


def _bounce_outcome(own: tuple[ScreenCheck, ...]) -> MergeOutcome:
    check_ids = sorted(check.id for check in own)
    findings = '\n'.join(f'[{check.id}] {text}' for check in own for text in check.findings)
    return MergeOutcome(
        'blocked',
        reason=f'{MERGE_SCREEN_REASON_PREFIX} the item\'s own defect in {", ".join(check_ids)}:\n{findings}',
        failure_category=MERGE_SCREEN_FAILURE_CATEGORY,
        failure_cause_hint=','.join(check_ids),
        block_category=MergeBlockCategory.MERGE_SCREEN,
    )
