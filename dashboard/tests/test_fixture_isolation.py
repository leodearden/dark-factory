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
from _dashboard_helpers import apply_isolated_env
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

    # project_root is not the only root the app opens databases under:
    # _project_scoped_dbs / _cost_dbs / _performance_resources (app.py) and
    # data/burndown.py fan out DbPool.get(root / 'data/orchestrator/runs.db')
    # over every known_project_roots entry too.  Any entry outside pytest's
    # basetemp is another live WAL database opened by the suite.
    assert all(r.is_relative_to(basetemp.resolve()) for r in cfg.known_project_roots), (
        f'known_project_roots escapes pytest basetemp {basetemp}: '
        f'{cfg.known_project_roots}'
    )


class TestLifespanProjectRootIsolation:
    def test_function_scoped_client_is_isolated(self, client, tmp_path_factory):
        """The shared conftest ``client`` fixture (function-scoped)."""
        _assert_lifespan_config_is_isolated(tmp_path_factory.getbasetemp())

    def test_module_scoped_client_is_isolated(self, _client, tmp_path_factory):
        """A module-scoped ``TestClient`` fixture — the ~15-site idiom in this suite."""
        _assert_lifespan_config_is_isolated(tmp_path_factory.getbasetemp())


class TestApplyIsolatedEnvNeutralizesAmbientRuntimeDirs:
    """``DASHBOARD_PROJECT_ROOT`` alone is not sufficient isolation.

    ``DashboardConfig._runtime_data_dir`` (config.py) reads
    ``RECONCILIATION_DATA_DIR`` / ``QUEUE_DATA_DIR`` straight from
    ``os.environ`` and they WIN over ``project_root`` for ``reconciliation_db``,
    ``tickets_db``, ``write_queue_db``, ``write_journal_db`` and
    ``reconciliation_escalations_dir``.  ``reconciliation_db`` and
    ``tickets_db`` are exactly the two read-only ``DbPool.get()`` opens in
    ``_metrics_loop`` that produced the task-3466 ``SQLITE_READONLY_RECOVERY``,
    so leaving them ambient would leave that incident's own trigger path
    un-isolated.

    Ambient presence is live, not hypothetical: the orchestrator's managed
    fused-memory spawn (``orchestrator/src/orchestrator/mcp_lifecycle.py``)
    injects both into managed subprocess environments.  This test simulates
    that environment with decoy values pointing OUTSIDE the isolated root.
    """

    def test_decoy_runtime_dirs_do_not_survive(self, tmp_path):
        from dashboard.config import DashboardConfig

        decoy = tmp_path / 'decoy'
        isolated_root = tmp_path / 'isolated'

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('RECONCILIATION_DATA_DIR', str(decoy / 'recon'))
            mp.setenv('QUEUE_DATA_DIR', str(decoy / 'q'))

            apply_isolated_env(mp, isolated_root)
            cfg = DashboardConfig.from_env()

            # These are LAZY properties: _runtime_data_dir re-reads os.environ on
            # every access, so the config object captures nothing.  Reading them
            # after the MonkeyPatch context exits would observe the restored
            # (decoy-free) environment and pass vacuously — the assertions MUST
            # stay inside this block.
            derived = {
                'reconciliation_db': cfg.reconciliation_db,
                'tickets_db': cfg.tickets_db,
                'write_journal_db': cfg.write_journal_db,
                'reconciliation_escalations_dir': cfg.reconciliation_escalations_dir,
                'write_queue_db': cfg.write_queue_db,
            }

            for label, path in derived.items():
                assert not path.is_relative_to(decoy), (
                    f'{label} {path} followed the ambient decoy env var '
                    f'out of the isolated root'
                )
                assert path.is_relative_to(cfg.project_root), (
                    f'{label} {path} is not under the isolated project_root '
                    f'{cfg.project_root}'
                )


class TestApplyIsolatedEnvNeutralizesAmbientKnownRoots:
    """Redirecting ``project_root`` is not sufficient isolation either.

    ``from_env()`` reads ``DASHBOARD_KNOWN_PROJECT_ROOTS`` into
    ``known_project_roots``, and ``_project_scoped_dbs`` / ``_cost_dbs`` /
    ``_performance_resources`` (dashboard/src/dashboard/app.py) plus
    ``data/burndown.py`` fan out ``DbPool.get(root / 'data/orchestrator/runs.db')``
    over EVERY entry — so an ambient value read-only-opens live WAL databases in
    whatever checkouts the operator registered, the same task-3466
    ``SQLITE_READONLY_RECOVERY`` class as the ``project_root`` path.

    The lifespan assertions above pass vacuously when this var happens to be
    unset in the running environment (it is, here), so the contract is pinned
    non-vacuously here with a decoy — mirroring the installed dashboard systemd
    unit, which does set it.
    """

    def test_decoy_known_project_roots_do_not_survive(self, tmp_path):
        from dashboard.config import DashboardConfig

        decoy_a = tmp_path / 'decoy-checkout-a'
        decoy_b = tmp_path / 'decoy-checkout-b'
        isolated_root = tmp_path / 'isolated'

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv('DASHBOARD_KNOWN_PROJECT_ROOTS', f'{decoy_a},{decoy_b}')

            apply_isolated_env(mp, isolated_root)
            cfg = DashboardConfig.from_env()

            assert cfg.known_project_roots == [], (
                f'known_project_roots followed the ambient decoy env var: '
                f'{cfg.known_project_roots} — every entry gets a '
                f"DbPool.get(root / 'data/orchestrator/runs.db')"
            )


class TestHermeticFusedMemoryUrls:
    """The endpoint the suite fans out at must be MEASURED dead, not assumed dead.

    ``DASHBOARD_FUSED_MEMORY_URLS`` is the network axis of the same isolation
    contract the classes above pin on the filesystem axis.  Left unset it falls
    back to ``DEFAULT_FUSED_MEMORY_URLS = ('http://localhost:8002',)`` — the
    operator's live shared fused-memory instance — so every app lifespan in
    this suite fans ``_burndown_loop`` and ``_metrics_loop`` out at production.

    Measurement, not a comment, because this suite has already been burned by
    exactly that substitution: ``test_api_curator_cancel.py`` documented 8002
    as "the same unreachable URL" while 8002 answered a 404 in 1.4ms, and
    believing that comment is most likely why this gap survived as long as it
    did.  A comment asserting a port is dead is the one form of evidence
    already disproven here.

    So the deadness is re-measured on every run.  If a port here becomes live,
    the suite is NOT hermetic and every fan-out test is quietly talking to a
    real service — going red is the correct outcome, and the fix is to pick
    another dead port, never to relax the check.  Same stance as
    ``_dashboard_helpers.build_dual_escalation_tree``'s containment assertion.
    """

    @staticmethod
    def _endpoints():
        """Return ``[(url, host, port), ...]`` for the hermetic constant."""
        from urllib.parse import urlsplit

        from _dashboard_helpers import HERMETIC_FUSED_MEMORY_URLS

        return [
            (url, urlsplit(url).hostname, urlsplit(url).port)
            for url in HERMETIC_FUSED_MEMORY_URLS
        ]

    def test_is_an_immutable_tuple_of_loopback_urls(self):
        from _dashboard_helpers import HERMETIC_FUSED_MEMORY_URLS

        assert isinstance(HERMETIC_FUSED_MEMORY_URLS, tuple), (
            'a tuple, like DEFAULT_FUSED_MEMORY_URLS it stands in for — a '
            'mutable default is one test away from being edited for everyone'
        )
        assert HERMETIC_FUSED_MEMORY_URLS, 'the list must not be empty'
        for url, host, port in self._endpoints():
            assert host in ('127.0.0.1', '::1'), (
                f'{url} must name a loopback literal: a hostname can resolve '
                f'off-box, and one that resolves to ::1 first costs a second '
                f'connect attempt before refusing — latency back in the very '
                f'path this constant exists to make instant. Got host {host!r}'
            )
            assert port is not None, f'{url} must name an explicit port'

    def test_shares_nothing_with_the_production_default(self):
        from _dashboard_helpers import HERMETIC_FUSED_MEMORY_URLS

        from dashboard.config import DEFAULT_FUSED_MEMORY_URLS

        assert not set(HERMETIC_FUSED_MEMORY_URLS) & set(DEFAULT_FUSED_MEMORY_URLS), (
            'the hermetic endpoint must not BE the production default — that '
            'is the traffic it exists to stop'
        )
        for url, _host, _port in self._endpoints():
            assert '8002' not in url, (
                f'{url} names the operator\'s live fused-memory port; the '
                f'whole point is to dial somewhere that answers nothing'
            )

    def test_every_port_genuinely_refuses_a_connection(self):
        """The load-bearing one: re-measured every run, never assumed."""
        import socket

        for url, host, port in self._endpoints():
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    pass
            except ConnectionRefusedError:
                continue
            except OSError as exc:
                raise AssertionError(
                    f'{url} neither answered nor refused ({exc!r}). The suite '
                    f'needs an INSTANT refusal; anything else puts a stall back '
                    f'into every app lifespan. Pick another dead port in '
                    f'_dashboard_helpers.HERMETIC_FUSED_MEMORY_URLS rather than '
                    f'relaxing this check.'
                ) from exc
            raise AssertionError(
                f'{url} ACCEPTED a connection. Something is listening on port '
                f'{port}, so this suite is not hermetic: every TestClient '
                f'lifespan is fanning _burndown_loop and _metrics_loop out at '
                f'a real service. Pick another dead port in '
                f'_dashboard_helpers.HERMETIC_FUSED_MEMORY_URLS rather than '
                f'relaxing this check.'
            )


class TestApplyIsolatedEnvNeutralizesAmbientFusedMemoryUrls:
    """The network axis of the same contract, pinned in two halves.

    ``from_env()`` reads ``DASHBOARD_FUSED_MEMORY_URLS``; unset, it falls back
    to the operator's live fused-memory instance, which every lifespan then
    dials through ``_burndown_loop`` -> ``collect_snapshot`` -> ``fetch_tasks``
    and through ``_metrics_loop``.

    TWO halves, because either alone passes while the suite is still
    un-hermetic.  The helper contract alone would pass if nothing ever called
    the helper; the end-to-end alone would pass vacuously on a box where the
    var happened to be set correctly by hand.

    The DELETED case in the first half is the one that matters most, and is
    why this var is SET rather than deleted like its three siblings: unset is
    exactly the state the whole suite runs in today, and for this variable
    unset means production.  The decoy case follows the precedent of
    ``TestApplyIsolatedEnvNeutralizesAmbientKnownRoots`` above — an operator
    shell or systemd unit may well have it set to something live.
    """

    def test_neither_a_decoy_nor_an_absent_value_survives(self, tmp_path):
        from _dashboard_helpers import HERMETIC_FUSED_MEMORY_URLS

        from dashboard.config import DEFAULT_FUSED_MEMORY_URLS, DashboardConfig

        isolated_root = tmp_path / 'isolated'
        ambient_states = {
            'decoy': lambda mp: mp.setenv(
                'DASHBOARD_FUSED_MEMORY_URLS', 'http://localhost:8002'
            ),
            'absent': lambda mp: mp.delenv(
                'DASHBOARD_FUSED_MEMORY_URLS', raising=False
            ),
        }

        for label, make_ambient in ambient_states.items():
            with pytest.MonkeyPatch.context() as mp:
                make_ambient(mp)

                apply_isolated_env(mp, isolated_root)
                cfg = DashboardConfig.from_env()

                assert cfg.fused_memory_urls == list(HERMETIC_FUSED_MEMORY_URLS), (
                    f'[{label}] apply_isolated_env must SET '
                    f'DASHBOARD_FUSED_MEMORY_URLS at the hermetic endpoint, not '
                    f'leave from_env() to resolve {cfg.fused_memory_urls}'
                )
                assert cfg.fused_memory_urls != list(DEFAULT_FUSED_MEMORY_URLS), (
                    f'[{label}] the resolved list is the production default — '
                    f'every app lifespan in this suite would fan out at the '
                    f"operator's live fused-memory instance"
                )

    def test_the_real_client_fixture_resolves_the_hermetic_endpoint(self, client):
        """Non-vacuous end-to-end: no MonkeyPatch context, the real session fixture.

        Pins the whole chain — session-autouse fixture -> env -> ``from_env()``
        -> ``lifespan`` -> ``app.state.config`` — so an edit that drops the
        ``setenv`` reds HERE rather than silently re-aiming the suite at
        production.
        """
        from _dashboard_helpers import HERMETIC_FUSED_MEMORY_URLS

        assert client.app.state.config.fused_memory_urls == list(
            HERMETIC_FUSED_MEMORY_URLS
        ), (
            f'the live lifespan resolved '
            f'{client.app.state.config.fused_memory_urls} — the suite is fanning '
            f'out somewhere other than the measured-dead endpoint'
        )
