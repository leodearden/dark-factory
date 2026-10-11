"""Tests for shared.project_layout: where a project's stores live under its root.

The literal path values are the contract (plans/review-all-s14-shared-leaves-prd.md):
every locator must return exactly the spelling its current call sites build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.project_layout import (
    ESCALATION_QUEUE_RELPATH,
    KNOWN_PROJECT_ROOTS_ENV,
    escalation_queue_dir,
    known_project_roots,
    orchestrator_data_dir,
    park_eviction_requests_db,
    runs_db,
    scheduler_overrides_db,
    scheduler_state_json,
    tasks_db,
)

_ORCHESTRATOR_STORES = (
    (runs_db, 'runs.db'),
    (scheduler_state_json, 'scheduler_state.json'),
    (scheduler_overrides_db, 'scheduler_overrides.db'),
    (park_eviction_requests_db, 'park_eviction_requests.db'),
)


@pytest.fixture(params=['str', 'path'])
def root(request: pytest.FixtureRequest, tmp_path: Path) -> str | Path:
    project = tmp_path / 'project'
    return str(project) if request.param == 'str' else project


class TestStoreLocators:
    def test_escalation_queue_dir(self, root: str | Path) -> None:
        assert escalation_queue_dir(root) == Path(root) / 'data' / 'escalations'
        assert escalation_queue_dir(root) == Path(root) / ESCALATION_QUEUE_RELPATH
        assert ESCALATION_QUEUE_RELPATH == 'data/escalations'

    def test_orchestrator_data_dir(self, root: str | Path) -> None:
        assert orchestrator_data_dir(root) == Path(root) / 'data' / 'orchestrator'

    @pytest.mark.parametrize(('locator', 'name'), _ORCHESTRATOR_STORES)
    def test_orchestrator_store(self, root: str | Path, locator, name: str) -> None:
        assert locator(root) == Path(root) / 'data' / 'orchestrator' / name

    def test_tasks_db_is_the_real_store_not_the_decoy(self, root: str | Path) -> None:
        assert tasks_db(root) == Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'

    @pytest.mark.parametrize(
        'locator',
        [
            escalation_queue_dir,
            orchestrator_data_dir,
            *(locator for locator, _ in _ORCHESTRATOR_STORES),
            tasks_db,
        ],
    )
    def test_returns_a_path(self, root: str | Path, locator) -> None:
        assert isinstance(locator(root), Path)

    @pytest.mark.parametrize(('locator', 'name'), _ORCHESTRATOR_STORES)
    def test_orchestrator_stores_share_one_parent(
        self, root: str | Path, locator, name: str
    ) -> None:
        assert locator(root).parent == orchestrator_data_dir(root)


class TestLocatorsArePure:
    @pytest.mark.parametrize(
        'locator',
        [
            escalation_queue_dir,
            orchestrator_data_dir,
            *(locator for locator, _ in _ORCHESTRATOR_STORES),
            tasks_db,
        ],
    )
    def test_no_io_on_a_missing_root(self, tmp_path: Path, locator) -> None:
        missing = tmp_path / 'absent'
        locator(missing)
        assert not missing.exists()
        assert list(tmp_path.iterdir()) == []

    def test_relative_root_stays_relative(self) -> None:
        assert runs_db('proj') == Path('proj/data/orchestrator/runs.db')
        assert tasks_db('proj') == Path('proj/.taskmaster/tasks/tasks.db')
        assert escalation_queue_dir('proj') == Path('proj/data/escalations')


class TestKnownProjectRoots:
    def test_env_var_name(self) -> None:
        assert KNOWN_PROJECT_ROOTS_ENV == 'DASHBOARD_KNOWN_PROJECT_ROOTS'

    def test_strips_entries_and_drops_empties(self) -> None:
        environ = {KNOWN_PROJECT_ROOTS_ENV: ' /a , ,/b,, '}
        assert known_project_roots(environ) == (Path('/a'), Path('/b'))

    @pytest.mark.parametrize('environ', [{}, {KNOWN_PROJECT_ROOTS_ENV: ''}, {KNOWN_PROJECT_ROOTS_ENV: ' , '}])
    def test_unset_or_blank_is_empty(self, environ: dict[str, str]) -> None:
        assert known_project_roots(environ) == ()

    def test_order_and_duplicates_are_preserved(self) -> None:
        environ = {KNOWN_PROJECT_ROOTS_ENV: '/c,/a,/c,/b'}
        assert known_project_roots(environ) == (Path('/c'), Path('/a'), Path('/c'), Path('/b'))

    def test_returns_a_tuple(self) -> None:
        assert isinstance(known_project_roots({KNOWN_PROJECT_ROOTS_ENV: '/a'}), tuple)

    def test_default_reads_the_live_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, '/x,/y')
        assert known_project_roots() == (Path('/x'), Path('/y'))
