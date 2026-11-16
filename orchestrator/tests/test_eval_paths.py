"""The λ contract: every eval default resolves under one checkout root.

plans/review-all-s18-evals-measurement-prd.md D11.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import shared.project_layout
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV, CheckoutRootError

from orchestrator.evals import paths


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
