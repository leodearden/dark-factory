"""The λ contract: every eval default resolves under one checkout root.

plans/review-all-s18-evals-measurement-prd.md D11.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import shared.project_layout
from _eval_refusal import answered
from click.testing import CliRunner
from shared.project_layout import (
    KNOWN_PROJECT_ROOTS_ENV,
    CheckoutRootError,
    escalation_queue_dir,
    reconciliation_tickets_db,
    runs_db,
)

from orchestrator.evals import paths
from orchestrator.evals.prompt_opt import curator_corpus, optimizer
from orchestrator.evals.prompt_opt.__main__ import cli as prompt_opt_cli
from orchestrator.evals.reviewer_trial import mining, scorer
from orchestrator.evals.reviewer_trial.__main__ import cli as reviewer_trial_cli
from orchestrator.evals.reviewer_trial.checkout import FixtureNotMaterialisable, fixture_checkout
from orchestrator.evals.reviewer_trial.corpus import (
    CorpusDiff,
    CorpusManifest,
    FixtureContext,
    FixtureTaskRecord,
    GroundTruthIssue,
)


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


def _corpus_diff(project: str | None, *, cwd: Path | None = None) -> CorpusDiff:
    return CorpusDiff(
        diff_id='fx_1', language='python', source='synthetic', diff_text='',
        description='a fixture', ground_truth=[], project=project, cwd=cwd,
        context=FixtureContext(
            task=FixtureTaskRecord(title='T', description='d', test_strategy='ts'),
            changed_files=[],
            base_sha='0' * 40,
        ),
    )


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    entries = [
        {'diff_id': f'fx_{index}', 'language': 'python', 'source': 'synthetic', 'description': 'd'}
        | ({'project': project} if project else {})
        for index, project in enumerate(['dark-factory', 'reify', 'reify-gui', None])
    ]
    path = corpus / 'manifest.json'
    path.write_text(json.dumps({'version': '1.0', 'diffs': entries}))
    return path


class TestCorpusFixtureSources:
    def test_each_project_resolves_to_its_checkout(
        self, main_checkout_at: Path, manifest_path: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(tmp_path / 'reify'))
        sources = [diff.source_dir() for diff in CorpusManifest.load(manifest_path).diffs]
        assert sources == [
            main_checkout_at, tmp_path / 'reify', tmp_path / 'reify' / 'gui', main_checkout_at,
        ]

    @pytest.mark.parametrize('project', [None, 'dark-factory', 'reify'])
    def test_an_explicit_cwd_wins(
        self, main_checkout_at: Path, tmp_path: Path, project: str | None
    ) -> None:
        assert _corpus_diff(project, cwd=tmp_path / 'x').source_dir() == tmp_path / 'x'

    def test_an_unregistered_project_loads_but_has_no_source(
        self, main_checkout_at: Path, manifest_path: Path
    ) -> None:
        diffs = CorpusManifest.load(manifest_path).diffs
        assert [diff.cwd for diff in diffs] == [None, None, None, None]
        reify = next(diff for diff in diffs if diff.project == 'reify')
        with pytest.raises(paths.ProjectCheckoutNotFound):
            reify.source_dir()

    @pytest.mark.asyncio
    async def test_an_unregistered_project_is_not_materialised_in_this_repository(
        self, main_checkout_at: Path, tmp_path: Path
    ) -> None:
        scratch = tmp_path / 'scratch'
        scratch.mkdir()
        with pytest.raises(FixtureNotMaterialisable) as caught:
            async with fixture_checkout(_corpus_diff('reify'), scratch_parent=scratch):
                pass
        assert KNOWN_PROJECT_ROOTS_ENV in caught.value.reason
        assert 'reify' in caught.value.reason
        assert list(scratch.iterdir()) == []


class TestCliDefaultsResolveUnderTheMainCheckout:
    def test_the_miner_reads_and_echoes_main_checkout_sources(
        self, main_checkout_at: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('REVIEWER_TRIAL_RUNS_DB', str(tmp_path / 'decoy' / 'runs.db'))
        monkeypatch.setenv('REVIEWER_TRIAL_ESCALATIONS_DIR', str(tmp_path / 'decoy' / 'escalations'))
        monkeypatch.setenv('REVIEWER_TRIAL_REPO_ROOT', str(tmp_path / 'decoy'))

        result = CliRunner().invoke(reviewer_trial_cli, ['mine'])

        assert result.exit_code == 1, result.output
        root = main_checkout_at
        for fragment in (
            f'runs_db={runs_db(root)}',
            f'escalations_dir={escalation_queue_dir(root)}',
            f'repo_path={root}',
            f'runs.db not found at {runs_db(root)}',
        ):
            assert fragment in result.output
        assert 'decoy' not in result.output

    def test_the_canary_reads_the_main_checkout_runs_db(
        self, main_checkout_at: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PROMPT_OPT_RUNS_DB', str(tmp_path / 'decoy' / 'runs.db'))

        result = CliRunner().invoke(prompt_opt_cli, [
            'canary', '--deploy-at', '2026-06-15T00:00:00+00:00', '--project-id', 'dark_factory',
        ])

        assert result.exit_code == 2, result.output
        assert f'runs.db not found at {runs_db(main_checkout_at)}' in result.output
        assert 'decoy' not in result.output

    def test_the_curator_corpus_builder_reads_the_main_checkout_tickets_db(
        self, main_checkout_at: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('CURATOR_TICKETS_DB', str(tmp_path / 'decoy' / 'tickets.db'))

        result = CliRunner().invoke(prompt_opt_cli, ['build-curator-corpus'])

        assert result.exit_code != 0, result.output
        assert f'tickets.db not found at {reconciliation_tickets_db(main_checkout_at)}' in result.output
        assert 'decoy' not in result.output
