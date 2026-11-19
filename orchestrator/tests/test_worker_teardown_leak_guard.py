"""No pytest item under orchestrator/tests stops a worker outside a ``finally:``
whose ``try:`` encloses its every responsive wait: the esc-3980-4 leak
invariant recorded at _merge_worker_drive.py::stop_merge_worker, as
_wait_hygiene_scan.py::scan_worker_teardown reads it.

The sweep is OPT-OUT over every ``orchestrator/tests/**/*.py``, so a new module
is covered the moment it exists.  ``_UNGUARDED_TEARDOWN_DEBT`` names the items
that still offend; it only shrinks, because an entry whose item no longer
offends fails too.  Task 7676 burns it down.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from _tree_scan import (
    ORCHESTRATOR_TESTS,
    TESTS_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
)
from _wait_budget_audit import RESPONSIVE_WAIT_HELPERS
from _wait_hygiene_scan import WorkerTeardownScan, scan_worker_teardown
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

# Shrink-only debt (task 7676 burns it down): delete an entry when its item stops offending; never add one for new code.
_UNGUARDED_TEARDOWN_DEBT: frozenset[str] = frozenset({
    'test_merge_queue.py::TestEnospcTransientInfraRetry::test_enospc_then_pass_on_retry_completes_done',
    'test_merge_queue.py::TestEnospcTransientInfraRetry::test_persistent_enospc_prunes_retries_once_then_blocks',
    'test_merge_queue.py::TestGroupMergeRequestSpeculativeWorker::test_regular_then_train_both_land',
    'test_merge_queue.py::TestGroupMergeRequestSpeculativeWorker::test_speculative_worker_train_pushes_on_success',
    'test_merge_queue.py::TestGroupMergeRequestSpeculativeWorker::test_train_lands_via_speculative_worker',
    'test_merge_queue.py::TestHeartbeatTaskLifecycle::test_heartbeat_task_started_by_run_and_cancelled_by_stop',
    'test_merge_queue.py::TestLanePickIntegration::test_high_lane_flows_while_normal_halted',
    'test_merge_queue.py::TestLanePickIntegration::test_high_lane_picked_after_current_verify',
    'test_merge_queue.py::TestMergeFailureDiagnostic::test_merger_loop_speculative_ghost_returns_unknown_branch',
    'test_merge_queue.py::TestMergeFailureDiagnostic::test_merger_loop_unknown_branch_intercepted_before_diagnostic',
    'test_merge_queue.py::TestMergeVerifyColdTimeout::test_speculative_worker_passes_is_merge_verify_true',
    'test_merge_queue.py::TestPreVerifyDiskGuardWiring::test_speculative_worker_short_circuits_on_persistent_low_disk',
    'test_merge_queue.py::TestPushHook::test_speculative_worker_invokes_push_main_on_success',
    'test_merge_queue.py::TestSoftCancelMidVerify::test_normal_sole_waiter_no_cancel_advances_done',
    'test_merge_queue.py::TestSpeculationPermitLeakOnMergerError::test_abandoned_speculative_releases_speculation_permit',
    'test_merge_queue.py::TestSpeculationPermitLeakOnMergerError::test_merger_exception_releases_speculation_permit',
    'test_merge_queue.py::TestSpeculationPermitLeakOnMergerError::test_worktree_missing_releases_speculation_permit',
    'test_merge_queue.py::TestSpeculationSlotSemaphoreDepth::test_k2_builds_two_speculative_ahead',
    'test_merge_queue.py::TestSpeculativeBackwardCompat::test_already_merged',
    'test_merge_queue.py::TestSpeculativeBackwardCompat::test_basic_merge',
    'test_merge_queue.py::TestSpeculativeBackwardCompat::test_verify_failure',
    'test_merge_queue.py::TestSpeculativeGateReverifyConsumesAdvanceOutcome::test_gate_reverify_consumes_advance_outcome_sha_fields',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_abandoned_counted_item_releases_cap',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_chain_invalidated_pre_rebased_n2_red_tree_blocked',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_chain_invalidated_pre_rebased_n2_verify_runs',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_chain_invalidation_with_cap_no_leak',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_merger_ahead_cap_bounds_blocking_path',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_merger_exception_resolves_inflight_future',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_merger_post_merge_exception_cleans_worktree',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_oob_conflict_emits_terminal_event',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_oob_delivery_unblocks_waiter_while_verifier_busy',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_pickup_rebase_pre_rebased_forces_verify',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_pre_rebased_main_unchanged_runs_merge_gate_speculative',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_retention_ring_populated_on_oob_conflict',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_revparse_failure_produces_blocked',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_already_merged_n_plus_1',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_basic_throughput',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_cas_failure_retries_until_advanced',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_cas_retry_limit_exhausted',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_chain_invalidation_propagates',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_conflict_n_plus_1',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_depth_cap',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_discard_on_failure',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_events_emitted',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_follower_chain_invalidated_after_pickup_rebase',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_merge_worker_emits_duration_ms_on_done',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_merge_worker_success_returns_merge_sha',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_merger_phase_emits_duration_ms',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_permanent_failure_returns_blocked',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_remerge_preserves_duration_ms',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_single_item_degenerates',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_stash_failed_abandoned_takes_halt_early_out',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_stash_failed_halts_worker',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_verify_called_with_max_retries_zero',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_speculative_verify_called_with_role_merge',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_stop_race_resolves_inflight_merger_future',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_unknown_branch_emits_terminal_merge_attempt',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_verifier_exception_releases_speculation_slot',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_verifier_remerge_exception_releases_slot',
    'test_merge_queue.py::TestSpeculativeMergeWorker::test_verify_pickup_rebases_when_main_advanced',
    'test_merge_queue.py::TestSpeculativeMergeWorkerDequeueEvent::test_speculative_worker_emits_merge_dequeued_after_dequeue',
    'test_merge_queue.py::TestSpeculativeMergeWorkerLedgerAwarePrune::test_disk_pressure_prune_removes_only_orphan_through_worker',
    'test_merge_queue.py::TestSpeculativeMergeWorkerLedgerAwarePrune::test_worker_passes_ledger_snapshot_to_post_merge_verify',
    'test_merge_queue.py::TestSpeculativeWorkerDequeueDepth::test_speculative_worker_merge_dequeued_carries_queue_depth',
    'test_merge_queue.py::TestWipHaltSpeculativeMergeWorker::test_done_wip_recovery_outcome',
    'test_merge_queue.py::TestWipHaltSpeculativeMergeWorker::test_speculative_done_wip_recovery_propagates_advanced_sha',
    'test_merge_queue.py::TestWipHaltSpeculativeMergeWorker::test_speculative_pop_conflict_no_advance_outcome',
    'test_merge_queue.py::TestWipHaltSpeculativeMergeWorker::test_speculative_unmerged_state_returns_unmerged_state_and_halts',
    'test_merge_queue.py::TestWipHaltSpeculativeMergeWorker::test_wip_halted_blocks_subsequent_tasks',
})

# Non-vacuity floors, ~75% of the census measured when this guard landed.
_MIN_WAITING_MODULES = 40
_MIN_WAITING_ITEMS = 230
_MIN_STOPPING_ITEMS = 89
_MIN_STOPPING_MODULES = 9

_HANDOVER_MODULE = 'test_merge_queue_concurrent_verify.py'

_NOTHING_TO_SCAN = WorkerTeardownScan((), (), ())


def _module_scan(module: ParsedModule) -> WorkerTeardownScan:
    if not module.may_name(*RESPONSIVE_WAIT_HELPERS):
        return _NOTHING_TO_SCAN
    return scan_worker_teardown(module.source)


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=ORCHESTRATOR_TESTS, extract=_module_scan)
TREE_ANALYSES = (ANALYSIS,)


@pytest.fixture(scope='module')
def scans(tree_pass: TreePass) -> Mapping[str, WorkerTeardownScan]:
    return {
        path.relative_to(TESTS_DIR).as_posix(): scan
        for path, scan in tree_pass.facts(ANALYSIS).items()
    }


def _unguarded_items(scans: Mapping[str, WorkerTeardownScan]) -> frozenset[str]:
    return frozenset(
        f'{module}::{item}' for module, scan in scans.items() for item in scan.unguarded_items
    )


def test_no_unguarded_teardown_outside_the_ledger(
    scans: Mapping[str, WorkerTeardownScan],
) -> None:
    unledgered = sorted(_unguarded_items(scans) - _UNGUARDED_TEARDOWN_DEBT)
    assert unledgered == [], (
        'These items stop a worker where a responsive-wait give-up skips the '
        'stop and leaks the worker into the next test:\n'
        + '\n'.join(f'  - {item}' for item in unledgered)
        + '\nMove the stop into `finally: await stop_merge_worker(worker, worker_task)` '
        'on a try that encloses every wait_responsive/poll_responsive call, and put '
        'any post-stop assertions after the try/finally.  A renamed ledgered item is '
        'fixed, not re-keyed: never add an entry to _UNGUARDED_TEARDOWN_DEBT.'
    )


def test_every_ledger_entry_still_offends(scans: Mapping[str, WorkerTeardownScan]) -> None:
    stale = sorted(_UNGUARDED_TEARDOWN_DEBT - _unguarded_items(scans))
    assert stale == [], '\n'.join(
        f'{item}: no longer an offender -- delete the entry.' for item in stale
    )


def test_the_scan_reaches_the_tree(scans: Mapping[str, WorkerTeardownScan]) -> None:
    totals = {
        'modules with a waiting item': (
            sum(bool(scan.waiting_items) for scan in scans.values()),
            _MIN_WAITING_MODULES,
        ),
        'waiting items': (
            sum(len(scan.waiting_items) for scan in scans.values()),
            _MIN_WAITING_ITEMS,
        ),
        'stopping items': (
            sum(len(scan.stopping_items) for scan in scans.values()),
            _MIN_STOPPING_ITEMS,
        ),
        'modules with a stopping item': (
            sum(bool(scan.stopping_items) for scan in scans.values()),
            _MIN_STOPPING_MODULES,
        ),
    }
    below_floor = [
        f'  - {name}: {count} < {floor}'
        for name, (count, floor) in totals.items()
        if count < floor
    ]

    assert not below_floor, (
        'The scan reached less of the tree than it did when this guard '
        'landed, so it may have gone vacuous:\n' + '\n'.join(below_floor)
    )
    assert scans.get(_HANDOVER_MODULE, _NOTHING_TO_SCAN).stopping_items, (
        f'{_HANDOVER_MODULE} contributes no stopping item, so this guard no '
        'longer holds what its retired self-scan '
        '(TestLoudWaitMigrationRatchet::test_migrated_methods_stop_worker_in_finally) held.'
    )


_PROBE_OFFENDER = (
    'class TestProbe:\n'
    '    async def test_it(self):\n'
    "        await wait_responsive(fut, timeout=5.0, label='x')\n"
    '        await worker.stop()\n'
)


def test_the_extract_reports_a_real_offending_file(tmp_path: Path) -> None:
    probe_dir = tmp_path.resolve()
    offender = probe_dir / 'test_offender.py'
    offender.write_text(_PROBE_OFFENDER, encoding='utf-8')
    unrelated = probe_dir / 'test_unrelated.py'
    unrelated.write_text('def test_it():\n    assert True\n', encoding='utf-8')
    probe = TreeAnalysis('teardown_probe', files=lambda: [offender, unrelated], extract=_module_scan)

    facts = TreePass.run([probe]).facts(probe)

    assert facts[offender].unguarded_items == ('TestProbe::test_it',)
    assert facts[unrelated] == _NOTHING_TO_SCAN
