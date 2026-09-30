"""Unit tests for γ's FalkorDB index-provisioning wiring (task 3708).

What this module pins
---------------------
The WIRING of β's ``GraphitiBackend.ensure_indices`` into the two places PRD D6
(docs/prds/falkordb-index-provisioning.md) names — the startup sweep and the
first write to a graph — and the D5 registry that scopes both: which graphs are
provisioned, when, under which lock, and what a provisioning failure does (and
does not) do to the write that triggered it.

HAZARD compliance: every test here is mock-driven.  No live FalkorDB, and no
``FalkorDriver`` / ``_MultiTenantFalkorDriver`` construction or
``GraphitiBackend.initialize()`` call anywhere — ``FalkorDriver.__init__``
fire-and-forgets index creation when an event loop is running, and
``initialize()``'s identity scan REPAIR-writes every graph on the server.  The
live lane lives in ``tests/test_index_provisioning_wiring_integration.py``.
"""

from __future__ import annotations

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.models.scope import KNOWN_PROJECT_ROOTS_ENV


class TestRegisteredGraphIds:
    """PRD D5: the provisioning scope is the project registry, not a graph name."""

    @pytest.fixture
    def roots(self, tmp_path):
        """Three project roots with no manifest, so their ids derive from basenames."""
        made = {name: tmp_path / name for name in ('Primary-Root', 'Alpha-Proj', 'beta')}
        for path in made.values():
            path.mkdir()
        return made

    def test_default_registry_is_taskmaster_root_plus_known_project_roots_env(
        self, mock_config, roots, monkeypatch,
    ):
        monkeypatch.setenv(
            KNOWN_PROJECT_ROOTS_ENV, f"{roots['Alpha-Proj']},{roots['beta']}",
        )
        config = mock_config.model_copy(deep=True)
        config.taskmaster = TaskmasterConfig(project_root=str(roots['Primary-Root']))

        registered = GraphitiBackend(config).registered_graph_ids

        assert registered == frozenset({'primary_root', 'alpha_proj', 'beta'})
        assert isinstance(registered, frozenset)

    def test_without_a_taskmaster_section_the_registry_is_the_env_roots_only(
        self, mock_config, roots, monkeypatch,
    ):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(roots['Alpha-Proj']))
        config = mock_config.model_copy(deep=True)
        config.taskmaster = None

        assert GraphitiBackend(config).registered_graph_ids == frozenset({'alpha_proj'})

    def test_an_injected_registry_wins_over_the_environment_and_is_canonicalized(
        self, mock_config, roots, monkeypatch,
    ):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(roots['Alpha-Proj']))

        backend = GraphitiBackend(mock_config, registered_graph_ids=['My-Project'])

        assert backend.registered_graph_ids == frozenset({'my_project'})
