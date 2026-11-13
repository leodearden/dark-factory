"""The §5.3 verify-base detector's policy and alarm (task 5302).

A verify should start on the frozen-prefix tip.  ``detect_verify_base_divergence``
decides, at dispatch, whether a real-verify item broke that rule and returns
the record of it; ``alarm_verify_base_divergence`` files the record once as an
L1.  Neither ever changes what the lane does (esc-3206-3, PRD §5.3).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from _merge_lane_fakes import RecordingEscalations, lane_scene_config
from _reviewer_trial_mining_fixtures import build_synthetic_runs_db
from escalation.categories import EscalationCategory

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.evals.reviewer_trial.mining import mine_fn_candidates
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import MergeResult
from orchestrator.merge_lane.types import (
    DecidedItem,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    RealMergeItem,
    RemergeBase,
    RemergeReason,
    VerifyBaseDivergence,
    VerifyBaseDivergenceKind,
)
from orchestrator.merge_lane.verify_base_detector import (
    VERIFY_BASE_DIVERGENCE_SENTINEL,
    alarm_verify_base_divergence,
    detect_verify_base_divergence,
)

_WIRE_KEYS = {
    'request_id', 'task_id', 'base_sha', 'expected_tip', 'verify_depth', 'observed_at', 'kind',
}


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return lane_scene_config(tmp_path, GitConfig())


def _request(config: OrchestratorConfig, task_id: str = 't-1') -> MergeRequest:
    """A request for *task_id*; needs a running loop for its result future."""
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=config.project_root,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


def _real_item(
    config: OrchestratorConfig, *, base_sha: str, remerge_base: RemergeBase | None = None,
) -> RealMergeItem:
    return RealMergeItem(
        request=_request(config),
        merge_result=MergeResult(success=True, merge_commit='sha-own'),
        merge_wt=Path('/fake/merge-wt'),
        base_sha=base_sha,
        speculative=False,
        remerge_base=remerge_base,
    )


def _detect(item: RealMergeItem | DecidedItem, expected_tip: str) -> VerifyBaseDivergence | None:
    return detect_verify_base_divergence(item, expected_tip, verify_depth=2, observed_at=123.5)


def _divergence(config: OrchestratorConfig) -> VerifyBaseDivergence:
    divergence = _detect(_real_item(config, base_sha='sha-spec'), 'sha-tip')
    assert divergence is not None
    return divergence


class _RecordingEvents:
    """An event store that records each ``emit`` call instead of writing it."""

    def __init__(self) -> None:
        self.emitted: list[tuple[EventType, dict[str, Any]]] = []

    def emit(self, event_type: EventType, **fields: Any) -> None:
        self.emitted.append((event_type, fields))


@pytest.mark.asyncio
class TestDetectVerifyBaseDivergence:
    """What counts as a dispatch-time divergence, and what it records."""

    async def test_decided_item_is_not_a_candidate(self, config: OrchestratorConfig) -> None:
        item = DecidedItem(
            request=_request(config),
            immediate_outcome=MergeOutcome('conflict'),
            base_sha='sha-spec',
            speculative=False,
        )
        assert _detect(item, 'sha-tip') is None

    async def test_item_without_a_base_is_not_a_candidate(
        self, config: OrchestratorConfig,
    ) -> None:
        assert _detect(_real_item(config, base_sha=''), 'sha-tip') is None

    async def test_recovery_remerge_on_main_is_carved_out(self, config: OrchestratorConfig) -> None:
        item = _real_item(
            config, base_sha='real-main', remerge_base=RemergeBase(None, RemergeReason.NO_TIP),
        )
        assert _detect(item, 'sha-tip') is None

    async def test_a_recovery_remerge_that_fell_to_main_is_exempt_whatever_its_reason(
        self, config: OrchestratorConfig,
    ) -> None:
        for reason in RemergeReason:
            if reason is RemergeReason.LIVE_TIP:
                continue
            item = _real_item(config, base_sha='real-main', remerge_base=RemergeBase(None, reason))
            assert _detect(item, 'sha-tip') is None, reason

    async def test_an_ordinary_stacked_item_off_the_tip_is_off_frozen_tip(
        self, config: OrchestratorConfig,
    ) -> None:
        divergence = _detect(_real_item(config, base_sha='sha-spec'), 'sha-tip')
        assert divergence is not None
        assert divergence.kind is VerifyBaseDivergenceKind.OFF_FROZEN_TIP

    async def test_a_live_tip_reanchor_off_the_tip_is_reanchor_onto_dead_tip(
        self, config: OrchestratorConfig,
    ) -> None:
        """Its tip died or was vacated between re-merge and dispatch
        (plans/merge-lane-policy-prd.md D12)."""
        item = _real_item(
            config, base_sha='sha-old-tip',
            remerge_base=RemergeBase('sha-old-tip', RemergeReason.LIVE_TIP),
        )
        divergence = _detect(item, 'sha-tip')
        assert divergence is not None
        assert divergence.kind is VerifyBaseDivergenceKind.REANCHOR_ONTO_DEAD_TIP

    async def test_base_on_the_tip_is_no_divergence(self, config: OrchestratorConfig) -> None:
        assert _detect(_real_item(config, base_sha=' sha-tip\n'), 'sha-tip') is None

    async def test_divergence_records_the_dispatch_facts(
        self, config: OrchestratorConfig,
    ) -> None:
        item = _real_item(config, base_sha='sha-spec')

        divergence = _detect(item, 'sha-tip')

        assert divergence == VerifyBaseDivergence(
            request_id=item.request.request_id,
            task_id=item.request.task_id,
            base_sha='sha-spec',
            expected_tip='sha-tip',
            verify_depth=2,
            observed_at=123.5,
        )

    async def test_divergence_is_immutable(self, config: OrchestratorConfig) -> None:
        divergence = _divergence(config)
        pytest.raises(dataclasses.FrozenInstanceError, setattr, divergence, 'base_sha', 'other')

    async def test_wire_form_is_json_with_exactly_the_record_fields(
        self, config: OrchestratorConfig,
    ) -> None:
        divergence = _divergence(config)

        wire = divergence.to_wire()

        assert set(wire) == _WIRE_KEYS
        assert json.loads(json.dumps(wire)) == wire
        assert wire['request_id'] == divergence.request_id
        assert wire['verify_depth'] == 2
        assert wire['kind'] == 'off_frozen_tip'


@pytest.mark.asyncio
class TestAlarmVerifyBaseDivergence:
    """A divergence is filed once as a dedup'd L1; nothing else happens."""

    async def test_no_queue_is_a_no_op(self, config: OrchestratorConfig) -> None:
        alarm_verify_base_divergence(None, _divergence(config))

    async def test_files_one_l1_naming_the_request_and_both_shas(
        self, config: OrchestratorConfig,
    ) -> None:
        escalations = RecordingEscalations()
        divergence = _divergence(config)

        alarm_verify_base_divergence(escalations, divergence)

        assert len(escalations.filed) == 1
        filed = escalations.filed[0]
        assert filed.task_id == VERIFY_BASE_DIVERGENCE_SENTINEL
        assert filed.level == 1
        assert filed.category == 'merge_verify_base_divergence'
        for needle in (divergence.request_id, 'sha-spec', 'sha-tip', 'off_frozen_tip'):
            assert needle in filed.summary

    async def test_the_l1_names_a_reanchor_hit_by_its_kind(
        self, config: OrchestratorConfig,
    ) -> None:
        escalations = RecordingEscalations()
        item = _real_item(
            config, base_sha='sha-old-tip',
            remerge_base=RemergeBase('sha-old-tip', RemergeReason.LIVE_TIP),
        )
        divergence = _detect(item, 'sha-tip')
        assert divergence is not None

        alarm_verify_base_divergence(escalations, divergence)

        [filed] = escalations.filed
        assert 'reanchor_onto_dead_tip' in filed.summary

    async def test_filing_emits_one_self_describing_event(
        self, config: OrchestratorConfig,
    ) -> None:
        """The event carries the record plus what an escalation reader keys on."""
        escalations = RecordingEscalations()
        events = _RecordingEvents()
        divergence = _divergence(config)

        alarm_verify_base_divergence(escalations, divergence, event_store=events)

        [filed] = escalations.filed
        assert filed.task_id == VERIFY_BASE_DIVERGENCE_SENTINEL
        assert events.emitted == [(
            EventType.escalation_created,
            {
                'task_id': filed.task_id,
                'data': {
                    **divergence.to_wire(),
                    'escalation_id': filed.id,
                    'category': 'merge_verify_base_divergence',
                    'summary': filed.summary,
                },
            },
        )]

    async def test_a_hit_is_not_charged_to_the_dispatched_task_as_a_reviewer_miss(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        divergence = _divergence(config)
        db = build_synthetic_runs_db(tmp_path, [{'task_id': divergence.task_id, 'review_cycles': 1}])
        events = EventStore(db, run_id='run-test')

        alarm_verify_base_divergence(RecordingEscalations(), divergence, event_store=events)

        assert [
            (row['task_id'], row['data']['task_id'])
            for row in events.fetch_events_by_type(EventType.escalation_created)
        ] == [(VERIFY_BASE_DIVERGENCE_SENTINEL, divergence.task_id)]
        assert mine_fn_candidates(db) == []

    async def test_open_l1_suppresses_a_second_filing(
        self, config: OrchestratorConfig,
    ) -> None:
        escalations = RecordingEscalations(open_l1=True)
        events = _RecordingEvents()

        alarm_verify_base_divergence(escalations, _divergence(config), event_store=events)

        assert escalations.filed == []
        assert events.emitted == []


def test_category_is_registered() -> None:
    assert 'merge_verify_base_divergence' in {c.value for c in EscalationCategory}
