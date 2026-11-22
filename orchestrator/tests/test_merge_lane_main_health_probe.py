"""The contract of ``merge_lane/main_health_probe.py`` (PRD κ part 2, task 5040).

The module answers one question for a failed post-merge verify -- does bare
main fail the same way? It is driven here through its public names alone:

- ``classify_main_health_red``, the synchronous classifier;
- ``spawn_main_health_probe``, which puts the off-critical-path classifier in
  flight, or declines to;
- the deferred probe's verdict: a confirmed, still-fresh break goes to the
  handles' ``auto_heal`` callback, else to
  ``file_main_health_escalation``, the dedup'd L1 filer.

What the lane does about a confirmed break (the auto-heal policy) is the
worker's, not this module's: it reaches the probe only as that callback.

Bare main's verdict is scripted by seeding the process-wide probe cache that
``verify.py::verify_failure_is_preexisting_on_main`` answers from before it
pays for a worktree, so the real classifier runs and nothing under test is
patched. Every expected value is today's ``SpeculativeMergeWorker``
behaviour; none is new.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import make_placeholder_future
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.main_health_actions import MainHealthNextAction
from orchestrator.merge_lane.disposition import MergeFailureDisposition
from orchestrator.merge_lane.main_health_probe import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MainHealthProbeHandles,
    classify_main_health_red,
    file_main_health_escalation,
    spawn_main_health_probe,
)
from orchestrator.merge_lane.types import (
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
)
from orchestrator.verify import _PROBE_CACHE, VerifyResult, clear_main_probe_caches
from orchestrator.verify_categories import PREEXISTING_BREAK_SKIP_CATEGORIES
from orchestrator.workflow import (
    _normalize_cause_hint,
    compute_preexisting_main_break_fingerprint,
)

MAIN_SHA = 'cafecafe1234567890deadbeef'

COMPILE_ERROR = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2322: StatusBar.tsx:12',
    summary='tsc failed',
    cause_hint='error TS2322: StatusBar.tsx',
    category='compile_error',
)


@pytest.fixture(autouse=True)
def _fresh_probe_caches() -> Iterator[None]:
    clear_main_probe_caches()
    yield
    clear_main_probe_caches()


def _config(tmp_path: Path, *, escalate_preexisting: bool = True) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        escalate_preexisting_main_break=escalate_preexisting,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _git_ops(tmp_path: Path, main_sha: str = MAIN_SHA) -> GitOps:
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)
    return git_ops


def _request(tmp_path: Path, config: OrchestratorConfig, task_id: str = '99') -> MergeRequest:
    worktree = tmp_path / 'task-wt'
    worktree.mkdir(exist_ok=True)
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=make_placeholder_future(),
        lane='normal',
    )


def _script_bare_main(verify: VerifyResult, *, fails: bool, main_sha: str = MAIN_SHA) -> None:
    """Bare main at *main_sha* fails *verify*'s signature, or passes."""
    key = (main_sha, verify.category or '', _normalize_cause_hint(verify.cause_hint))
    _PROBE_CACHE[key] = (time.monotonic(), fails)


def _confirmed_break(verify: VerifyResult, *, fingerprint: str) -> MergeOutcome:
    """The outcome a probe builds for a confirmed break."""
    return MergeOutcome(
        'blocked',
        reason=f'{MAIN_HEALTH_RED_REASON_PREFIX} (category={verify.category!r}): {verify.cause_hint}',
        failure_category=verify.category,
        failure_cause_hint=verify.cause_hint,
        dedupe_fingerprint=fingerprint,
        disposition=MergeFailureDisposition.MAIN_RED,
    )


# ---------------------------------------------------------------------------
# (a) classify_main_health_red


class TestClassifyMainHealthRed:
    def test_a_failure_bare_main_shares_is_a_main_health_red_block(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        _script_bare_main(COMPILE_ERROR, fails=True)

        outcome = asyncio.run(classify_main_health_red(
            _git_ops(tmp_path), _request(tmp_path, config), COMPILE_ERROR,
        ))

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX)
        assert outcome.failure_category == 'compile_error'
        assert outcome.failure_cause_hint == COMPILE_ERROR.cause_hint
        assert outcome.dedupe_fingerprint == compute_preexisting_main_break_fingerprint(
            'compile_error', COMPILE_ERROR.cause_hint, MAIN_SHA,
        )
        assert outcome.disposition is MergeFailureDisposition.MAIN_RED

    def test_a_failure_bare_main_passes_is_left_to_the_task(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        _script_bare_main(COMPILE_ERROR, fails=False)

        outcome = asyncio.run(classify_main_health_red(
            _git_ops(tmp_path), _request(tmp_path, config), COMPILE_ERROR,
        ))

        assert outcome is None


# ---------------------------------------------------------------------------
# (b) spawn_main_health_probe


def _skipped_cases() -> list[tuple[str, bool, VerifyResult]]:
    return [
        ('feature off', False, COMPILE_ERROR),
        ('timed out', True, dataclasses.replace(COMPILE_ERROR, timed_out=True)),
        *(
            (f'{category} is never probed', True,
             dataclasses.replace(COMPILE_ERROR, category=str(category)))
            for category in sorted(PREEXISTING_BREAK_SKIP_CATEGORIES)
        ),
    ]


class TestSpawnMainHealthProbe:
    @pytest.mark.parametrize(
        ('escalate_preexisting', 'verify'),
        [case[1:] for case in _skipped_cases()],
        ids=[case[0] for case in _skipped_cases()],
    )
    def test_a_skipped_classification_spawns_nothing(
        self, tmp_path: Path, escalate_preexisting: bool, verify: VerifyResult,
    ) -> None:
        config = _config(tmp_path, escalate_preexisting=escalate_preexisting)
        handles = MainHealthProbeHandles(background_tasks=set())

        async def spawn() -> bool:
            return spawn_main_health_probe(
                handles, _git_ops(tmp_path), _request(tmp_path, config), verify,
            )

        assert asyncio.run(spawn()) is False
        assert handles.background_tasks == set()

    def test_a_classifiable_failure_puts_exactly_one_probe_in_flight(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        _script_bare_main(COMPILE_ERROR, fails=False)
        handles = MainHealthProbeHandles(background_tasks=set())

        async def spawn_and_settle() -> tuple[bool, int]:
            spawned = spawn_main_health_probe(
                handles, _git_ops(tmp_path), _request(tmp_path, config), COMPILE_ERROR,
            )
            in_flight = set(handles.background_tasks)
            await asyncio.gather(*in_flight)
            return spawned, len(in_flight)

        assert asyncio.run(spawn_and_settle()) == (True, 1)
        assert handles.background_tasks == set()


# ---------------------------------------------------------------------------
# (c) The deferred probe's verdict: the auto-heal callback, or the escalation


def _git_ops_answering(tmp_path: Path, *main_shas: str) -> GitOps:
    """Main HEAD as the probe sees it, then as the staleness re-check sees it."""
    git_ops = _git_ops(tmp_path)
    git_ops.get_main_sha = AsyncMock(side_effect=list(main_shas))
    return git_ops


def _settle_deferred_probe(
    tmp_path: Path, git_ops: GitOps, handles: MainHealthProbeHandles,
    queue: EscalationQueue | None = None,
) -> None:
    config = _config(tmp_path)

    async def spawn_and_settle() -> None:
        assert spawn_main_health_probe(
            handles, git_ops, _request(tmp_path, config), COMPILE_ERROR,
            escalation_queue=queue,
        )
        await asyncio.gather(*handles.background_tasks)

    asyncio.run(spawn_and_settle())


class TestDeferredVerdict:
    def test_a_fresh_break_goes_to_the_auto_heal_callback_alone(self, tmp_path: Path) -> None:
        _script_bare_main(COMPILE_ERROR, fails=True)
        healed: list[tuple[MergeOutcome, MergeRequest]] = []

        async def auto_heal(outcome: MergeOutcome, req: MergeRequest) -> None:
            healed.append((outcome, req))

        queue = EscalationQueue(tmp_path / 'escalations')
        handles = MainHealthProbeHandles(background_tasks=set(), auto_heal=auto_heal)
        _settle_deferred_probe(
            tmp_path, _git_ops_answering(tmp_path, MAIN_SHA, MAIN_SHA), handles, queue,
        )

        [(outcome, req)] = healed
        assert req.task_id == '99'
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX)
        assert outcome.dedupe_fingerprint == compute_preexisting_main_break_fingerprint(
            'compile_error', COMPILE_ERROR.cause_hint, MAIN_SHA,
        )
        assert queue.get_pending() == [], 'the callback owns filing'

    def test_without_a_callback_a_fresh_break_files_one_l1(self, tmp_path: Path) -> None:
        _script_bare_main(COMPILE_ERROR, fails=True)
        queue = EscalationQueue(tmp_path / 'escalations')
        handles = MainHealthProbeHandles(background_tasks=set())

        _settle_deferred_probe(
            tmp_path, _git_ops_answering(tmp_path, MAIN_SHA, MAIN_SHA), handles, queue,
        )

        [escalation] = queue.get_pending()
        assert escalation.category == 'preexisting_main_break'
        assert escalation.level == 1
        assert escalation.suggested_action == MainHealthNextAction.NO_FIX_OWNER
        assert escalation.summary.startswith(MAIN_HEALTH_RED_REASON_PREFIX)

    def test_a_break_main_has_moved_past_is_dropped(self, tmp_path: Path) -> None:
        _script_bare_main(COMPILE_ERROR, fails=True)
        healed: list[MergeOutcome] = []

        async def auto_heal(outcome: MergeOutcome, req: MergeRequest) -> None:
            healed.append(outcome)

        queue = EscalationQueue(tmp_path / 'escalations')
        handles = MainHealthProbeHandles(background_tasks=set(), auto_heal=auto_heal)
        _settle_deferred_probe(
            tmp_path, _git_ops_answering(tmp_path, MAIN_SHA, 'beefbeef'), handles, queue,
        )

        assert healed == []
        assert queue.get_pending() == []


# ---------------------------------------------------------------------------
# (d) file_main_health_escalation


class TestFileMainHealthEscalation:
    def test_breaks_of_one_signature_fold_into_one_parent(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        queue = EscalationQueue(tmp_path / 'escalations')
        brk = _confirmed_break(COMPILE_ERROR, fingerprint='fp-main')

        first = file_main_health_escalation(
            queue, _request(tmp_path, config, '1'), brk,
            suggested_action=MainHealthNextAction.NO_FIX_OWNER,
        )
        second = file_main_health_escalation(
            queue, _request(tmp_path, config, '2'), brk,
            suggested_action=MainHealthNextAction.NO_FIX_OWNER,
        )

        assert first is not None
        assert second == first
        assert [esc.id for esc in queue.get_pending()] == [first]

    def test_no_queue_files_nothing(self, tmp_path: Path) -> None:
        assert file_main_health_escalation(
            None, _request(tmp_path, _config(tmp_path)),
            _confirmed_break(COMPILE_ERROR, fingerprint='fp-main'),
            suggested_action=MainHealthNextAction.NO_FIX_OWNER,
        ) is None
