"""Tests for orchestrator.merge_types: extracted request/outcome/item/entry

types + registries (MQ-refactor task α).

These tests encode the two behavior-preserving contracts of the module
split:

1. Module-existence — ``orchestrator.merge_types`` exists and exports the
   full closure of moved public (and internal-but-referenced) symbols.
2. Shim identity — ``orchestrator.merge_queue`` re-exports the *same*
   objects (not copies) so every existing importer keeps working
   unchanged.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import OrchestratorConfig


def test_merge_types_exports_moved_public_symbols() -> None:
    from orchestrator.merge_types import (
        GroupMergeRequest,
        InflightEntry,
        InFlightMergeRegistry,
        InflightVerifyResult,
        MainHealthAutoHealRegistry,
        MergeBounceRegistry,
        MergeDispatchResult,
        MergeOutcome,
        MergeRequest,
        SoloVerifyResult,
        SpeculativeItem,
        TerminalOutcomeRecord,
        TerminalOutcomeRetention,
        TrainCallbackFactory,
        TrainCallbacks,
        WaiterRecord,
    )

    for name, obj in {
        "GroupMergeRequest": GroupMergeRequest,
        "InflightEntry": InflightEntry,
        "InflightVerifyResult": InflightVerifyResult,
        "InFlightMergeRegistry": InFlightMergeRegistry,
        "MainHealthAutoHealRegistry": MainHealthAutoHealRegistry,
        "MergeBounceRegistry": MergeBounceRegistry,
        "MergeDispatchResult": MergeDispatchResult,
        "MergeOutcome": MergeOutcome,
        "MergeRequest": MergeRequest,
        "SoloVerifyResult": SoloVerifyResult,
        "SpeculativeItem": SpeculativeItem,
        "TerminalOutcomeRecord": TerminalOutcomeRecord,
        "TerminalOutcomeRetention": TerminalOutcomeRetention,
        "TrainCallbackFactory": TrainCallbackFactory,
        "TrainCallbacks": TrainCallbacks,
        "WaiterRecord": WaiterRecord,
    }.items():
        assert obj is not None, f"{name} must not be None"


def test_merge_queue_reexports_identical_objects() -> None:
    """merge_queue re-exports the SAME objects from merge_types (shim identity).

    Covers every moved name, including the private/alias ones that staying
    worker code in merge_queue.py still references by bare name
    (``_InFlightEntry``, ``_HostUnavailability``, ``MergeReadyPredicate``,
    ``_INFLIGHT_MERGE_ETA_ESTIMATE_SECS``).

    RED (pre-shim): merge_queue.py still defines its own independent copies
    of these types (the duplicate definitions left in place by the EXPAND
    step), so ``getattr(merge_queue, name) is getattr(merge_types, name)``
    fails for every name — two distinct objects that merely share a name.
    """
    import orchestrator.merge_queue as merge_queue
    import orchestrator.merge_types as merge_types

    moved_names = [
        "MainHealthAutoHealRegistry",
        "MergeBounceRegistry",
        "TerminalOutcomeRecord",
        "TerminalOutcomeRetention",
        "_InFlightEntry",
        "InFlightMergeRegistry",
        "MergeDispatchResult",
        "WaiterRecord",
        "MergeRequest",
        "GroupMergeRequest",
        "MergeOutcome",
        "SoloVerifyResult",
        "SpeculativeItem",
        "InflightEntry",
        "_HostUnavailability",
        "InflightVerifyResult",
        "TrainCallbacks",
        "TrainCallbackFactory",
        "MergeReadyPredicate",
        "_INFLIGHT_MERGE_ETA_ESTIMATE_SECS",
    ]

    for name in moved_names:
        mq_obj = getattr(merge_queue, name)
        mt_obj = getattr(merge_types, name)
        assert mq_obj is mt_obj, (
            f"{name}: orchestrator.merge_queue.{name} and "
            f"orchestrator.merge_types.{name} must be the identical object"
        )

    assert issubclass(merge_queue.GroupMergeRequest, merge_queue.MergeRequest)

    outcome = merge_queue.MergeOutcome(status='done')
    assert outcome.status == 'done'

    request = merge_queue.MergeRequest(
        task_id='t1',
        branch=merge_queue.QueuedBranch.parse('591', 'task/'),
        worktree=Path('/tmp/wt'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=OrchestratorConfig(),
        result=make_placeholder_future(),
    )
    assert request.task_id == 't1'
    assert request.branch.bare_id == '591'


class TestQueuedBranch:
    """Unit tests for QueuedBranch.parse (PRD merge-queue-reliability-prd.md
    task mu, scope 5, DD7 — B9 boundary test, producer half).

    parse(raw, branch_prefix) is the ONLY place branch-prefix logic lives
    (parse-don't-validate): a bare id ('4778') and an already-prefixed name
    ('task/4778') both collapse to the same canonical QueuedBranch value.
    """

    def test_parse_bare_raw(self) -> None:
        """parse('4778', 'task/') -> bare_id='4778', full_name='task/4778'."""
        from orchestrator.merge_types import QueuedBranch

        qb = QueuedBranch.parse('4778', 'task/')
        assert qb.bare_id == '4778'
        assert qb.full_name == 'task/4778'

    def test_parse_already_prefixed_raw(self) -> None:
        """parse('task/4778', 'task/') yields the SAME shape as the bare form."""
        from orchestrator.merge_types import QueuedBranch

        qb = QueuedBranch.parse('task/4778', 'task/')
        assert qb.bare_id == '4778'
        assert qb.full_name == 'task/4778'

    def test_bare_and_prefixed_raw_compare_equal(self) -> None:
        """Mixed input shape collapses to one canonical value (value equality)."""
        from orchestrator.merge_types import QueuedBranch

        assert QueuedBranch.parse('4778', 'task/') == QueuedBranch.parse('task/4778', 'task/')

    def test_parse_is_idempotent_round_trip(self) -> None:
        """Re-parsing either derived field reproduces the same QueuedBranch."""
        from orchestrator.merge_types import QueuedBranch

        qb = QueuedBranch.parse('4778', 'task/')
        assert QueuedBranch.parse(qb.full_name, 'task/') == qb
        assert QueuedBranch.parse(qb.bare_id, 'task/') == qb

    def test_parse_empty_prefix_is_noop(self) -> None:
        """An empty branch_prefix leaves bare_id and full_name identical."""
        from orchestrator.merge_types import QueuedBranch

        qb = QueuedBranch.parse('4778', '')
        assert qb.bare_id == '4778'
        assert qb.full_name == '4778'

    def test_incoherent_pair_raises_value_error(self) -> None:
        """A hand-built (bare_id, full_name) pair that doesn't cohere is rejected.

        parse() never produces this shape — full_name always ends with
        bare_id — so this only trips on a direct, bypassing-parse
        construction. Realizes "mixed shape unrepresentable" (PRD DD7).
        """
        from orchestrator.merge_types import QueuedBranch

        with pytest.raises(ValueError):
            QueuedBranch(bare_id='4778', full_name='wrong/9999')

    def test_parse_result_is_frozen(self) -> None:
        """A QueuedBranch produced by parse is immutable."""
        from orchestrator.merge_types import QueuedBranch

        qb = QueuedBranch.parse('4778', 'task/')
        with pytest.raises(dataclasses.FrozenInstanceError):
            qb.bare_id = 'x'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestTerminalOutcomeForget — task ε (2928) step-1 RED / step-2 GREEN
# ---------------------------------------------------------------------------


class TestTerminalOutcomeForget:
    """Unit tests for ``TerminalOutcomeRetention.forget(request_id)``.

    ``forget`` is the "sticky per-task result cleared" primitive used by
    ``merge_cancel`` retirement (task ε).  It removes a record from
    ``_index`` and, under an object-identity guard mirroring ``record()``'s
    eviction guard, from the ``_by_branch`` / ``_by_task`` secondary indexes;
    it also drops any alias whose key OR value is the forgotten request_id.
    The deque slot is intentionally left in place (lossy-eviction contract)
    but becomes unreachable through every lookup.  ``forget`` returns whether
    a record was removed from ``_index``.
    """

    @staticmethod
    def _rec(request_id: str, branch: str, task_id: str, state: str = 'abandoned'):
        from orchestrator.merge_types import TerminalOutcomeRecord

        return TerminalOutcomeRecord(
            request_id=request_id,
            task_id=task_id,
            branch=branch,
            state=state,
        )

    def test_forget_removes_from_all_indexes_and_returns_true(self) -> None:
        """(a) forget clears _index / _by_branch / _by_task and returns True."""
        from orchestrator.merge_types import TerminalOutcomeRetention

        ring = TerminalOutcomeRetention(maxlen=10)
        rec = self._rec('mr-a', branch='B', task_id='T')
        ring.record(rec)
        # Pre-condition: all three lookups resolve to rec.
        assert ring.get('mr-a') is rec
        assert ring.get_by_branch('B') is rec
        assert ring.get_by_task('T') is rec

        assert ring.forget('mr-a') is True

        # Post-condition: every lookup is now a miss.
        assert ring.get('mr-a') is None
        assert ring.get_by_branch('B') is None
        assert ring.get_by_task('T') is None

    def test_forget_identity_guard_preserves_newer_secondary_entry(self) -> None:
        """(b) forget(old) must NOT drop a secondary entry owned by a newer record."""
        from orchestrator.merge_types import TerminalOutcomeRetention

        ring = TerminalOutcomeRetention(maxlen=10)
        rec1 = self._rec('mr-old', branch='B', task_id='T', state='blocked')
        rec2 = self._rec('mr-new', branch='B', task_id='T', state='abandoned')
        ring.record(rec1)
        ring.record(rec2)  # newest-wins: _by_branch['B'] and _by_task['T'] are rec2
        assert ring.get_by_branch('B') is rec2
        assert ring.get_by_task('T') is rec2

        # Forgetting the OLD record must leave the newer record's secondary
        # ownership intact (identity guard).
        assert ring.forget('mr-old') is True
        assert ring.get('mr-old') is None
        assert ring.get('mr-new') is rec2
        assert ring.get_by_branch('B') is rec2
        assert ring.get_by_task('T') is rec2

    def test_forget_drops_alias_whose_target_is_request_id(self) -> None:
        """(c) forget(primary) drops an alias whose value points to it."""
        from orchestrator.merge_types import TerminalOutcomeRetention

        ring = TerminalOutcomeRetention(maxlen=10)
        rec = self._rec('mr-a', branch='B', task_id='T')
        ring.record(rec)
        ring.record_alias('mr-coalesced', 'mr-a')
        assert ring.get('mr-coalesced') is rec  # alias resolves pre-forget

        assert ring.forget('mr-a') is True
        assert ring.get('mr-a') is None
        assert ring.get('mr-coalesced') is None  # alias dropped (target forgotten)

    def test_forget_drops_alias_whose_key_is_request_id(self) -> None:
        """(c2) forget(alias_key) drops the alias registered under that key.

        The key has no direct ``_index`` record, so forget returns False, but
        the alias itself must still be removed.
        """
        from orchestrator.merge_types import TerminalOutcomeRetention

        ring = TerminalOutcomeRetention(maxlen=10)
        primary = self._rec('mr-primary', branch='P', task_id='TP', state='done')
        ring.record(primary)
        ring.record_alias('mr-x', 'mr-primary')
        assert ring.get('mr-x') is primary  # resolves via alias pre-forget

        assert ring.forget('mr-x') is False  # no _index record for mr-x
        assert ring.get('mr-x') is None  # alias keyed by 'mr-x' is gone
        assert ring.get('mr-primary') is primary  # primary untouched

    def test_forget_unknown_returns_false_noop(self) -> None:
        """(d) forget on an unknown request_id returns False and changes nothing."""
        from orchestrator.merge_types import TerminalOutcomeRetention

        ring = TerminalOutcomeRetention(maxlen=10)
        rec = self._rec('mr-keep', branch='K', task_id='TK', state='done')
        ring.record(rec)

        assert ring.forget('mr-unknown') is False
        # Existing record untouched.
        assert ring.get('mr-keep') is rec
        assert ring.get_by_branch('K') is rec
        assert ring.get_by_task('TK') is rec

    def test_forget_then_fresh_record_same_branch_task_resolves_fresh(self) -> None:
        """(e) After forget, a fresh record for the same branch/task resolves cleanly."""
        from orchestrator.merge_types import TerminalOutcomeRetention

        ring = TerminalOutcomeRetention(maxlen=10)
        rec_a = self._rec('mr-a', branch='B', task_id='T', state='abandoned')
        ring.record(rec_a)
        assert ring.forget('mr-a') is True

        rec_fresh = self._rec('mr-fresh', branch='B', task_id='T', state='done')
        ring.record(rec_fresh)
        assert ring.get('mr-fresh') is rec_fresh
        assert ring.get_by_branch('B') is rec_fresh
        assert ring.get_by_task('T') is rec_fresh
        assert ring.get('mr-a') is None
