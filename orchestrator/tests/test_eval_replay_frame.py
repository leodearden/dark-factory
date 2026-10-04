"""Tests for the eval REPLAY FRAME — evals/replay_frame.py (task 4844).

Covered here: the replay-frame primitive (block text, persisted frame id, base
commit refusal) and the ``EvalMetrics.replay_frame`` field. The runner wiring
lives in ``test_eval_architect.py`` and ``test_eval_driver.py``.
"""

from __future__ import annotations

import pytest

_BASE = '20c934ca597c7c9e3e2eb79f572970a8099f9243'


class TestBuildReplayFrameBlock:
    def test_block_names_the_base_commit_verbatim(self):
        from orchestrator.evals.replay_frame import build_replay_frame_block

        assert _BASE in build_replay_frame_block(_BASE)

    def test_block_is_a_deterministic_function_of_the_base(self):
        from orchestrator.evals.replay_frame import build_replay_frame_block

        assert build_replay_frame_block(_BASE) == build_replay_frame_block(_BASE)
        assert build_replay_frame_block(_BASE) != build_replay_frame_block('abc123')

    def test_block_opens_its_own_markdown_section(self):
        from orchestrator.evals.replay_frame import build_replay_frame_block

        block = build_replay_frame_block(_BASE)
        assert block.startswith('\n')
        assert block.lstrip('\n').startswith('# ')

    @pytest.mark.parametrize('base_commit', ['', '   ', None])
    def test_missing_base_commit_is_refused_by_name(self, base_commit):
        from orchestrator.evals.replay_frame import build_replay_frame_block

        with pytest.raises(ValueError) as excinfo:
            build_replay_frame_block(base_commit)  # type: ignore[arg-type]
        assert repr(base_commit) in str(excinfo.value)


def test_replay_frame_id_is_the_persisted_vocabulary():
    from orchestrator.evals.replay_frame import REPLAY_FRAME_ID

    assert REPLAY_FRAME_ID == 'honest-frame-v1'


class TestEvalMetricsReplayFrameField:
    def test_default_is_none_the_unframed_sentinel(self):
        from orchestrator.evals.metrics import EvalMetrics

        assert EvalMetrics().replay_frame is None

    def test_to_dict_carries_the_key_unconditionally(self):
        from orchestrator.evals.metrics import EvalMetrics

        assert 'replay_frame' in EvalMetrics().to_dict()
        assert EvalMetrics().to_dict()['replay_frame'] is None

    def test_stamped_frame_round_trips_through_to_dict(self):
        from orchestrator.evals.metrics import EvalMetrics
        from orchestrator.evals.replay_frame import REPLAY_FRAME_ID

        assert EvalMetrics(replay_frame=REPLAY_FRAME_ID).to_dict()[
            'replay_frame'] == 'honest-frame-v1'


class TestPlanOnlyCliAppendsReplayFrame:
    def test_plan_only_architect_is_briefed_in_the_replay_frame(
        self, tmp_path, monkeypatch,
    ):
        import json
        from unittest.mock import AsyncMock, MagicMock

        from click.testing import CliRunner

        import orchestrator.cli
        from orchestrator.cli import main
        from orchestrator.config import OrchestratorConfig
        from orchestrator.evals.replay_frame import build_replay_frame_block

        monkeypatch.setattr(
            orchestrator.cli, 'load_config',
            lambda _p: OrchestratorConfig(project_root=tmp_path),
        )
        fixture = tmp_path / 'df_task_9001.json'
        fixture.write_text(json.dumps({
            'id': 'df_task_9001',
            'project_root': str(tmp_path),
            'pre_task_commit': 'feedface1234',
            'task_definition': {'title': 'T', 'description': 'D'},
            'plan': None,
        }))
        dummy_yaml = tmp_path / 'dummy.yaml'
        dummy_yaml.write_text('')

        briefing = MagicMock()
        briefing.build_architect_prompt = AsyncMock(return_value='ARCH PROMPT')
        invoke_agent = AsyncMock(return_value=MagicMock(success=False, output='refused'))
        monkeypatch.setattr(
            'orchestrator.evals.snapshots.create_eval_worktree',
            AsyncMock(return_value=(tmp_path / 'wt', 'run-x')),
        )
        monkeypatch.setattr(
            'orchestrator.evals.snapshots.cleanup_eval_worktree', AsyncMock(),
        )
        monkeypatch.setattr(
            'orchestrator.agents.briefing.BriefingAssembler',
            MagicMock(return_value=briefing),
        )
        monkeypatch.setattr('orchestrator.artifacts.TaskArtifacts', MagicMock())
        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', invoke_agent)

        r = CliRunner().invoke(main, [
            'eval', '--plan-only', '--task', str(fixture), '--config', str(dummy_yaml),
        ])

        assert r.exit_code == 0, r.output
        invoke_agent.assert_awaited_once()
        assert invoke_agent.await_args.kwargs['prompt'] == (
            'ARCH PROMPT' + build_replay_frame_block('feedface1234')
        )
