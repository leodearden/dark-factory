"""Tests for shared.eval_lane — the eval-lane provenance recognizer.

The load-bearing half is the regression guard: every non-numeric PRODUCTION
sentinel task id must stay unrecognised, or containment would silence real
human L2s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.eval_lane import (
    eval_lane_provenance,
    is_eval_fixture_task_id,
    is_eval_worktree_path,
)

SIBLING_EVAL_WORKTREE = (
    '/home/leo/src/dark-factory-eval-worktrees/df_task_2430_adv_plan/run-c2fbbe84'
)
LEGACY_EVAL_WORKTREE = '/home/leo/src/dark-factory/.eval-worktrees/df_task_2339/run-ac3ab562'
TASK_WORKTREE = '/home/leo/src/dark-factory/.worktrees/3096'
MARKER_NAMED_PROJECT = '/home/leo/src/eval-worktree-tools'


class TestIsEvalFixtureTaskId:
    @pytest.mark.parametrize(
        'task_id',
        [
            'df_task_12',
            'df_task_1993',
            'reify_task_27',
            'reify_task_5221',
            'kl_task_543',
            'df_task_2430_adv_plan',
            'df_task_2339_adv_verify',
            'df_task_2284_adv_regression',
        ],
    )
    def test_corpus_fixture_ids_are_recognised(self, task_id: str) -> None:
        assert is_eval_fixture_task_id(task_id) is True

    @pytest.mark.parametrize(
        'task_id',
        [
            'shadow_5383_01JCELL',
            'shadow_5383_01J9ZK3QWERTY0123456789ABCD',
        ],
    )
    def test_live_shadow_fixture_ids_are_recognised(self, task_id: str) -> None:
        assert is_eval_fixture_task_id(task_id) is True

    @pytest.mark.parametrize(
        'task_id',
        [
            'task-path-guard',
            'task-curator',
            'task-999',
            '__recovery_veto_streak__5381',
            '__merge_request_stuck__mr-4a2b869f',
            '__merge_resource_leak__',
            '__scheduler__',
            '__session_resume_storm__',
            '__watcher_supervisor__',
            'fused-memory',
            'infra',
            'infra-openai-quota-exhausted',
            'dirty-project-root-startup',
            'orchestrator-self-redeploy',
            'candidate-key-migration',
            'markup-tripwire',
            'fleet-weekly-limit',
            'main-sweep-3d8e73053011',
            'None',
        ],
    )
    def test_production_sentinel_ids_are_not_recognised(self, task_id: str) -> None:
        assert is_eval_fixture_task_id(task_id) is False

    @pytest.mark.parametrize('task_id', ['3096', '2430', '12'])
    def test_numeric_production_ids_are_not_recognised(self, task_id: str) -> None:
        assert is_eval_fixture_task_id(task_id) is False

    @pytest.mark.parametrize(
        'task_id',
        [
            'shadow',
            'shadow_',
            'shadow_5383',
            'my_shadow_5383_X',
            'df-task-12',
            'DF_TASK_12',
            'df_task_',
        ],
    )
    def test_near_misses_are_not_recognised(self, task_id: str) -> None:
        assert is_eval_fixture_task_id(task_id) is False

    @pytest.mark.parametrize('task_id', [None, '', '   '])
    def test_degenerate_inputs_are_not_recognised(self, task_id: str | None) -> None:
        assert is_eval_fixture_task_id(task_id) is False


class TestIsEvalWorktreePath:
    @pytest.mark.parametrize(
        'path',
        [
            SIBLING_EVAL_WORKTREE,
            LEGACY_EVAL_WORKTREE,
            '/home/leo/src/eval-worktree-tools-eval-worktrees/kl_task_543/run-1',
        ],
    )
    @pytest.mark.parametrize('as_type', [str, Path])
    def test_eval_worktree_layouts_are_recognised(self, path: str, as_type: type) -> None:
        assert is_eval_worktree_path(as_type(path)) is True

    @pytest.mark.parametrize(
        'path',
        [
            '/home/leo/src/dark-factory',
            TASK_WORKTREE,
            MARKER_NAMED_PROJECT,
            f'{MARKER_NAMED_PROJECT}/.worktrees/12',
            '/home/leo/src/my-eval-worktree/.worktrees/5',
            '/home/leo/src/eval-worktrees-archive/.worktrees/5',
        ],
    )
    @pytest.mark.parametrize('as_type', [str, Path])
    def test_production_paths_are_not_recognised(self, path: str, as_type: type) -> None:
        assert is_eval_worktree_path(as_type(path)) is False

    @pytest.mark.parametrize('path', [None, ''])
    def test_degenerate_inputs_are_not_recognised(self, path: str | None) -> None:
        assert is_eval_worktree_path(path) is False


class TestEvalLaneProvenance:
    def test_id_signal_alone_names_the_id(self) -> None:
        assert (
            eval_lane_provenance('df_task_2430_adv_plan', None)
            == 'fixture-task-id:df_task_2430_adv_plan'
        )

    @pytest.mark.parametrize('worktree', [SIBLING_EVAL_WORKTREE, LEGACY_EVAL_WORKTREE])
    def test_worktree_signal_alone_names_the_path(self, worktree: str) -> None:
        assert eval_lane_provenance('2339', worktree) == f'eval-worktree:{worktree}'

    def test_both_signals_report_the_id(self) -> None:
        assert (
            eval_lane_provenance('df_task_2430_adv_plan', SIBLING_EVAL_WORKTREE)
            == 'fixture-task-id:df_task_2430_adv_plan'
        )

    def test_worktree_defaults_to_absent(self) -> None:
        assert eval_lane_provenance('shadow_5383_01JCELL') == 'fixture-task-id:shadow_5383_01JCELL'

    @pytest.mark.parametrize(
        'task_id, worktree',
        [
            ('3096', TASK_WORKTREE),
            ('task-path-guard', None),
            (None, None),
        ],
    )
    def test_production_filings_have_no_provenance(
        self, task_id: str | None, worktree: str | None
    ) -> None:
        assert eval_lane_provenance(task_id, worktree) is None
