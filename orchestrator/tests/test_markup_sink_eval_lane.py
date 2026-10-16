"""An eval cell's markup records never reach the production escalation queue (task 6479).

plan-tools is injected into every eval cell, and an eval worktree is a LINKED
worktree of the main checkout, so the markup sink would otherwise file pending
records into production ``data/escalations``. These tests drive the public
sink against a real ``EscalationQueue`` under ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.mcp import markup_sink

FIXTURE_SUBJECT = 'df_task_2430_adv_plan'
PRODUCTION_SUBJECT = '6479'

SPEC = markup_sink.MarkupSinkSpec(
    server_label='plan-tools',
    agent_role='plan-tools-markup-guard',
    residue_anchor_task_id='plan-tools-markup-residue',
    storm_anchor_task_id='plan-tools-markup-storm',
    refusal_consequence='Nothing was written to the plan.',
    storm_consequence='every later leaked call is written through.',
    attribution_source='read the test journal',
)


def _residue_record() -> dict[str, Any]:
    return {
        'error_type': markup_sink.MARKUP_RESIDUE_ERROR_TYPE,
        'category': 'mcp_markup_residue',
        'owner': 'l2-escalation-watcher',
        'level': 2,
        'tool': 'add_design_decision',
        'field': 'decision',
        'matched_pattern': 'x',
        'agent_id': None,
        'project': None,
        'raw_value': 'the caller payload',
        'summary': 'Unrepairable MCP envelope markup in add_design_decision.decision',
        'suggested_action': 'Recover the raw_value.',
    }


def _storm_record() -> dict[str, Any]:
    return {
        'error_type': markup_sink.MARKUP_STORM_ERROR_TYPE,
        'count': 7,
        'threshold': 5,
        'window_seconds': 3600,
        'outcome': 'repaired',
        'project': None,
        'callers': [],
    }


RECORD_KINDS = pytest.mark.parametrize(
    'record', [_residue_record, _storm_record], ids=['residue', 'storm'],
)


def _sibling_eval_worktree(tmp_path: Path) -> Path:
    return tmp_path / 'proj-eval-worktrees' / 'df_task_2430_adv_plan' / 'run-c2fbbe84'


def _legacy_eval_worktree(tmp_path: Path) -> Path:
    return tmp_path / 'proj' / '.eval-worktrees' / 'df_task_2339' / 'run-ac3ab562'


def _task_worktree(tmp_path: Path) -> Path:
    return tmp_path / 'proj' / '.worktrees' / '6479'


#: Each eval-lane signal the escalation server's own gate recognises
#: (``shared.eval_lane.eval_lane_provenance``): an eval worktree in either
#: layout, or an eval fixture subject in an ordinary-looking worktree.
EVAL_LANE_PROVENANCE = pytest.mark.parametrize(
    ('worktree', 'subject'),
    [
        (_sibling_eval_worktree, PRODUCTION_SUBJECT),
        (_legacy_eval_worktree, PRODUCTION_SUBJECT),
        (_task_worktree, FIXTURE_SUBJECT),
    ],
    ids=['sibling-eval-worktree', 'legacy-eval-worktree', 'fixture-subject'],
)


def _queue_dir(root: Path) -> Path:
    return root / markup_sink.MARKUP_QUEUE_DIRNAME


def _pending(root: Path) -> list[Escalation]:
    return EscalationQueue(_queue_dir(root)).get_pending()


def _sink(tmp_path: Path, worktree: Path, subject: str):
    return markup_sink.make_escalation_sink(
        worktree=worktree,
        spec=SPEC,
        subject_task_id=lambda: subject,
        resolve_root=lambda _wt: tmp_path,
    )


class TestAnEvalLaneRecordFilesNothing:
    @EVAL_LANE_PROVENANCE
    @RECORD_KINDS
    @pytest.mark.asyncio
    async def test_the_sink_answers_none_and_opens_no_queue(
        self,
        tmp_path: Path,
        worktree: Callable[[Path], Path],
        subject: str,
        record: Callable[[], dict[str, Any]],
    ):
        filed = await _sink(tmp_path, worktree(tmp_path), subject)(record())

        assert filed is None
        assert not _queue_dir(tmp_path).exists()


class TestAProductionRecordStillFiles:
    @pytest.mark.asyncio
    async def test_a_residue_files_one_pending_level_2_record_under_its_anchor(
        self, tmp_path: Path,
    ):
        worktree = _task_worktree(tmp_path)

        filed = await _sink(tmp_path, worktree, PRODUCTION_SUBJECT)(_residue_record())

        assert isinstance(filed, str)
        [escalation] = _pending(tmp_path)
        assert escalation.id == filed
        assert escalation.level == 2
        assert escalation.task_id == SPEC.residue_anchor_task_id
        assert escalation.worktree == str(worktree)

    @pytest.mark.asyncio
    async def test_a_burst_alarm_files_one_pending_record_under_its_anchor(
        self, tmp_path: Path,
    ):
        worktree = _task_worktree(tmp_path)

        filed = await _sink(tmp_path, worktree, PRODUCTION_SUBJECT)(_storm_record())

        assert isinstance(filed, str)
        [escalation] = _pending(tmp_path)
        assert escalation.id == filed
        assert escalation.level == markup_sink.MARKUP_STORM_LEVEL
        assert escalation.task_id == SPEC.storm_anchor_task_id
