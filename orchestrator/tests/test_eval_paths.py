"""The λ contract: every eval default resolves under one checkout root.

plans/review-all-s18-evals-measurement-prd.md D11.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import shared.project_layout
from _eval_refusal import answered
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV, CheckoutRootError

from orchestrator.evals import paths
from orchestrator.evals.prompt_opt import curator_corpus, optimizer
from orchestrator.evals.reviewer_trial import mining, scorer
from orchestrator.evals.reviewer_trial.corpus import GroundTruthIssue


@pytest.fixture
def main_checkout_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Relocate shared.project_layout into a tmp main checkout named dark-factory."""
    root = tmp_path / 'dark-factory'
    (root / '.git').mkdir(parents=True)
    monkeypatch.setattr(
        shared.project_layout, '__file__', str(root / 'shared' / 'src' / 'shared' / 'project_layout.py')
    )
    monkeypatch.delenv(KNOWN_PROJECT_ROOTS_ENV, raising=False)
    return root.resolve()


class TestEvalPaths:
    def test_main_checkout_is_the_derived_root(self, main_checkout_at: Path) -> None:
        assert paths.main_checkout() == main_checkout_at

    @pytest.mark.parametrize('project_id', ['dark-factory', 'dark_factory'])
    def test_this_repository_is_its_own_project_checkout(
        self, main_checkout_at: Path, project_id: str
    ) -> None:
        assert paths.project_checkout(project_id) == main_checkout_at

    def test_another_project_comes_from_the_registry(
        self, main_checkout_at: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, f'{tmp_path / "other"},{tmp_path / "reify"}')
        assert paths.project_checkout('reify') == tmp_path / 'reify'

    def test_this_repository_wins_over_a_registered_namesake(
        self, main_checkout_at: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(tmp_path / 'elsewhere' / 'dark-factory'))
        assert paths.project_checkout('dark-factory') == main_checkout_at

    def test_an_explicit_environ_is_honoured(self, main_checkout_at: Path, tmp_path: Path) -> None:
        environ = {KNOWN_PROJECT_ROOTS_ENV: str(tmp_path / 'reify')}
        assert paths.project_checkout('reify', environ=environ) == tmp_path / 'reify'

    def test_an_unregistered_project_reports_an_empty_registry(self, main_checkout_at: Path) -> None:
        with pytest.raises(paths.ProjectCheckoutNotFound) as caught:
            paths.project_checkout('reify')
        assert isinstance(caught.value, CheckoutRootError)
        message = str(caught.value)
        for fragment in ('reify', 'dark_factory', KNOWN_PROJECT_ROOTS_ENV, 'unset'):
            assert fragment in message

    def test_an_unregistered_project_names_the_registered_entries(
        self, main_checkout_at: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(tmp_path / 'other'))
        with pytest.raises(paths.ProjectCheckoutNotFound) as caught:
            paths.project_checkout('reify')
        message = str(caught.value)
        for fragment in ('reify', 'dark_factory', KNOWN_PROJECT_ROOTS_ENV, str(tmp_path / 'other')):
            assert fragment in message


class TestHarnessCallsRunInTheMainCheckout:
    @pytest.mark.asyncio
    async def test_the_heuristics_optimizer(self, main_checkout_at: Path) -> None:
        calls: list[dict] = []

        async def fake(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(structured_output=None, output='x')

        await optimizer.propose_heuristics_edit(
            'h', [], [], max_edits=1, optimizer_model='m', invoke_fn=fake
        )
        assert [call['cwd'] for call in calls] == [main_checkout_at]

    @pytest.mark.asyncio
    async def test_the_issue_matcher(self, main_checkout_at: Path) -> None:
        ground_truth = GroundTruthIssue(
            id='gt1', location='a.py:1', category='bug', severity='blocking',
            description='A bug', mutation_type='logic',
        )
        with patch('orchestrator.evals.reviewer_trial.scorer.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = answered({'matches': []}, cost_usd=0.1)
            await scorer.match_issues(
                reviewer_issues=[{'location': 'a.py:1', 'category': 'bug', 'description': 'x'}],
                ground_truth=[ground_truth],
                diff_text='diff',
            )
        assert mock_invoke.call_args.kwargs['cwd'] == main_checkout_at

    @pytest.mark.asyncio
    async def test_the_frontier_label_proposer(self, main_checkout_at: Path) -> None:
        with patch('orchestrator.evals.reviewer_trial.mining.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = answered({'issues': []}, cost_usd=0.15)
            await mining.propose_labels_frontier('diff text', model='opus', description='d')
        assert mock_invoke.call_args.kwargs['cwd'] == main_checkout_at

    @pytest.mark.asyncio
    async def test_the_curator_label_proposer_ignores_the_old_env_var(
        self, main_checkout_at: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('CURATOR_REPO_ROOT', str(tmp_path / 'decoy'))
        payload = {'action': 'create', 'justification': 'new work'}
        with patch('orchestrator.agents.invoke.invoke_agent', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = answered(payload, cost_usd=0.2)
            await curator_corpus.propose_curator_label_frontier({'title': 't'})
        assert mock_invoke.call_args.kwargs['cwd'] == main_checkout_at
