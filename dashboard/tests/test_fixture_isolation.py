"""The TestClient lifespan must never touch the operator's live project root.

WHY this module exists (task 3503, incident behind task 3466):
``lifespan()`` (dashboard/src/dashboard/app.py) calls ``DashboardConfig.from_env()``
and then opens **writable WAL** stores at ``config.burndown_db`` and
``config.metrics_db``, while ``_metrics_loop`` read-only-opens
``config.reconciliation_db`` and ``config.tickets_db`` through ``DbPool``.
With ``DASHBOARD_PROJECT_ROOT`` unset, ``project_root`` resolved to the live
checkout — so merely *running the dashboard test suite* created and wrote
``<live checkout>/data/burndown/{burndown.db,metrics.db}`` and read-only-opened
the live ``reconciliation.db``.  That read-only open of a live WAL database is
exactly the path that produced the ``SQLITE_READONLY_RECOVERY`` incident behind
task 3466.

The contract asserted here: every ``TestClient(app)`` in this suite, at **any**
fixture scope, runs its lifespan against a pytest-owned temp root.  The two
scopes are tested separately on purpose — they fail for different reasons, and
only a *session*-scoped fix satisfies both.  A module-scoped ``TestClient``
fixture cannot request the function-scoped ``monkeypatch`` fixture, so a
function-scoped isolation fixture would be instantiated only *after* such a
lifespan had already opened the live databases.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

# tests/ -> dashboard/ -> repo root
_DASHBOARD_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The literal that used to be DashboardConfig.project_root's default.
_LIVE_CHECKOUT = Path('/home/leo/src/dark-factory')


@pytest.fixture(scope='module')
def _client():
    """Module-scoped TestClient, mirroring the idiom in test_tab_merge_queue.py:29.

    Deliberately module-scoped: this is the scope that a function-scoped
    isolation fixture provably CANNOT protect, so it is the load-bearing half
    of this module's contract.
    """
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


def _assert_lifespan_config_is_isolated(basetemp: Path) -> None:
    """Assert the live ``app.state.config`` points at pytest's temp tree.

    Reads ``app.state.config`` — the object the lifespan actually built and
    handed to the burndown/metrics stores — rather than re-deriving a config,
    so the assertion covers the real resource-opening path.
    """
    from dashboard.app import app

    cfg = app.state.config
    root = cfg.project_root

    # Not the operator's live checkout, under any spelling.
    assert root != _LIVE_CHECKOUT, (
        f'lifespan opened WAL databases under the live checkout {root}'
    )
    assert root != _REPO_ROOT, f'lifespan opened WAL databases under the repo root {root}'
    assert root != _DASHBOARD_DIR, (
        f'lifespan opened WAL databases under the dashboard subproject dir {root}'
    )

    # Positively: pytest owns the directory.
    assert root.is_relative_to(basetemp.resolve()), (
        f'project_root {root} is not under pytest basetemp {basetemp}'
    )

    # Every database the lifespan touches must follow project_root there.
    for label, db_path in (
        ('burndown_db', cfg.burndown_db),  # writable WAL, opened by lifespan
        ('metrics_db', cfg.metrics_db),  # writable WAL, opened by lifespan
        ('reconciliation_db', cfg.reconciliation_db),  # read-only DbPool open (task 3466)
        ('tickets_db', cfg.tickets_db),  # read-only DbPool open (task 3466)
    ):
        assert db_path.is_relative_to(root), f'{label} {db_path} escapes project_root {root}'


class TestLifespanProjectRootIsolation:
    def test_function_scoped_client_is_isolated(self, client, tmp_path_factory):
        """The shared conftest ``client`` fixture (function-scoped)."""
        _assert_lifespan_config_is_isolated(tmp_path_factory.getbasetemp())

    def test_module_scoped_client_is_isolated(self, _client, tmp_path_factory):
        """A module-scoped ``TestClient`` fixture — the ~15-site idiom in this suite."""
        _assert_lifespan_config_is_isolated(tmp_path_factory.getbasetemp())
