"""Tests asserting that dead code has been removed (task #157).

Verifies the following cleanup:
  - queue_service.py deleted from fused_memory.services
  - fused_memory.services.__init__ no longer exports QueueService
  - orchestrator.protocols module deleted
  - fused_memory.reconciliation.stages.integrity_check shim deleted
  - fused_memory.backends.__init__ no longer re-exports GraphitiBackend / Mem0Backend

Each test group is expected to FAIL before the corresponding deletion and PASS after.
"""

import importlib
import importlib.util
from pathlib import Path

# Root of the repository (two levels up from fused-memory/tests/)
_REPO_ROOT = Path(__file__).parent.parent.parent


class TestQueueServiceRemoved:
    """Assert queue_service.py has been deleted and its symbol removed from services."""

    def test_queue_service_module_not_importable(self):
        """fused_memory.services.queue_service should not be importable."""
        spec = importlib.util.find_spec("fused_memory.services.queue_service")
        assert spec is None, (
            "fused_memory.services.queue_service still exists — delete queue_service.py"
        )

    def test_queue_service_not_in_services_all(self):
        """QueueService should not appear in fused_memory.services.__all__."""
        import fused_memory.services as svc

        all_exports = getattr(svc, "__all__", [])
        assert "QueueService" not in all_exports, (
            "'QueueService' is still listed in fused_memory.services.__all__ — "
            "remove the import and __all__ entry from services/__init__.py"
        )

    def test_queue_service_not_attribute_of_services(self):
        """fused_memory.services.QueueService attribute should not exist."""
        import fused_memory.services as svc

        assert not hasattr(svc, "QueueService"), (
            "fused_memory.services still has a QueueService attribute — "
            "remove the import from services/__init__.py"
        )


class TestOrchestratorProtocolsRemoved:
    """Assert orchestrator.protocols module has been deleted.

    Note: orchestrator is a separate package not installed in the fused-memory
    venv, so we use a direct file-system check rather than importlib.util.find_spec.
    """

    def test_protocols_file_does_not_exist(self):
        """orchestrator/src/orchestrator/protocols.py should not exist on disk."""
        protocols_path = (
            _REPO_ROOT / "orchestrator" / "src" / "orchestrator" / "protocols.py"
        )
        assert not protocols_path.exists(), (
            f"orchestrator/protocols.py still exists at {protocols_path} — delete it"
        )


class TestIntegrityCheckShimRemoved:
    """Assert the integrity_check.py re-export shim has been deleted."""

    def test_integrity_check_shim_not_importable(self):
        """fused_memory.reconciliation.stages.integrity_check should not exist."""
        spec = importlib.util.find_spec(
            "fused_memory.reconciliation.stages.integrity_check"
        )
        assert spec is None, (
            "fused_memory.reconciliation.stages.integrity_check still exists — "
            "delete the shim file"
        )


class TestBackendsInitCleaned:
    """Assert backends/__init__.py no longer re-exports GraphitiBackend / Mem0Backend."""

    def test_graphiti_backend_not_in_backends_all(self):
        """GraphitiBackend should not be in fused_memory.backends.__all__."""
        import fused_memory.backends as backends

        all_exports = getattr(backends, "__all__", [])
        assert "GraphitiBackend" not in all_exports, (
            "'GraphitiBackend' is still in fused_memory.backends.__all__ — "
            "remove the re-export from backends/__init__.py"
        )

    def test_mem0_backend_not_in_backends_all(self):
        """Mem0Backend should not be in fused_memory.backends.__all__."""
        import fused_memory.backends as backends

        all_exports = getattr(backends, "__all__", [])
        assert "Mem0Backend" not in all_exports, (
            "'Mem0Backend' is still in fused_memory.backends.__all__ — "
            "remove the re-export from backends/__init__.py"
        )

    def test_graphiti_backend_not_direct_attribute_of_backends(self):
        """fused_memory.backends.GraphitiBackend attribute should not exist."""
        import fused_memory.backends as backends

        assert not hasattr(backends, "GraphitiBackend"), (
            "fused_memory.backends still exposes GraphitiBackend directly — "
            "remove the import from backends/__init__.py"
        )

    def test_mem0_backend_not_direct_attribute_of_backends(self):
        """fused_memory.backends.Mem0Backend attribute should not exist."""
        import fused_memory.backends as backends

        assert not hasattr(backends, "Mem0Backend"), (
            "fused_memory.backends still exposes Mem0Backend directly — "
            "remove the import from backends/__init__.py"
        )
