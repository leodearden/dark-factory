"""An eval cell's markup residue never reaches the production escalation queue (task 6479).

plan-tools is injected into every eval cell, and an eval worktree is a LINKED
worktree of the main checkout, so ``markup_sink.file_residue`` would otherwise
file a pending level-2 record into production ``data/escalations``. These tests
drive the public sink against a real ``EscalationQueue`` under ``tmp_path``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.mcp import markup_sink

SUBJECT = 'df_task_2430_adv_plan'

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


def _sibling_eval_worktree(tmp_path: Path) -> Path:
    return tmp_path / 'proj-eval-worktrees' / 'df_task_2430_adv_plan' / 'run-c2fbbe84'


def _legacy_eval_worktree(tmp_path: Path) -> Path:
    return tmp_path / 'proj' / '.eval-worktrees' / 'df_task_2339' / 'run-ac3ab562'


def _task_worktree(tmp_path: Path) -> Path:
    return tmp_path / 'proj' / '.worktrees' / '6479'


EVAL_LAYOUTS = pytest.mark.parametrize(
    'eval_worktree', [_sibling_eval_worktree, _legacy_eval_worktree], ids=['sibling', 'legacy'],
)


def _pending(root: Path) -> list[Escalation]:
    return EscalationQueue(root / markup_sink.MARKUP_QUEUE_DIRNAME).get_pending()


def _sink(tmp_path: Path, worktree: Path):
    return markup_sink.make_escalation_sink(
        worktree=worktree,
        spec=SPEC,
        subject_task_id=lambda: SUBJECT,
        resolve_root=lambda _wt: tmp_path,
    )


def _file_residue_directly(tmp_path: Path, worktree: Path) -> str | None:
    queue = EscalationQueue(tmp_path / markup_sink.MARKUP_QUEUE_DIRNAME)
    return markup_sink.file_residue(
        Escalation, queue, worktree, SUBJECT, _residue_record(), SPEC,
    )


class TestAnEvalCellResidueFilesNoEscalation:
    @EVAL_LAYOUTS
    @pytest.mark.asyncio
    async def test_the_sink_files_nothing_and_answers_none(self, tmp_path, eval_worktree):
        filed = await _sink(tmp_path, eval_worktree(tmp_path))(_residue_record())

        assert filed is None
        assert _pending(tmp_path) == []

    @EVAL_LAYOUTS
    def test_file_residue_itself_files_nothing(self, tmp_path, eval_worktree):
        assert _file_residue_directly(tmp_path, eval_worktree(tmp_path)) is None
        assert _pending(tmp_path) == []

    @EVAL_LAYOUTS
    def test_the_drop_is_logged_naming_the_worktree(self, tmp_path, eval_worktree, caplog):
        worktree = eval_worktree(tmp_path)
        with caplog.at_level(logging.INFO, logger='orchestrator.mcp.markup_sink'):
            _file_residue_directly(tmp_path, worktree)

        assert any(
            record.name == 'orchestrator.mcp.markup_sink'
            and str(worktree) in record.getMessage()
            for record in caplog.records
        )


class TestARealTaskWorktreeResidueStillFiles:
    @pytest.mark.asyncio
    async def test_one_pending_level_2_record_under_the_residue_anchor(self, tmp_path):
        worktree = _task_worktree(tmp_path)

        filed = await _sink(tmp_path, worktree)(_residue_record())

        assert isinstance(filed, str)
        [escalation] = _pending(tmp_path)
        assert escalation.id == filed
        assert escalation.level == 2
        assert escalation.task_id == SPEC.residue_anchor_task_id
        assert escalation.worktree == str(worktree)
