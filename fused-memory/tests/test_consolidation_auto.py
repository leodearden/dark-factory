"""Tests for the deterministic auto-consolidation predicate + canonical builder.

Task 5237, PRD `plans/memory-auto-consolidation-prd.md` contracts C2 (the pure
predicate) and C3 (the canonical text builder).

Every assertion here RUNS the code under test against fixture inputs and
asserts on the returned verdict or string. None inspects source text, greps for
a symbol, or pins English phrasing — with the single deliberate exception of
``TestBuildAutoCanonical.test_renders_the_prd_template_verbatim``, where the
template IS the contract and duplicating it once in the test is the only way to
notice drift in its wording, backticks or punctuation.
"""

from __future__ import annotations

import subprocess
import sys


class TestImportLeafAndSingleHomes:
    """``reconciliation/consolidation_auto.py`` is a STDLIB-ONLY import leaf.

    Its stated precedent is ``reconciliation/consolidation_gate.py``, but this
    module holds a STRICTLY STRONGER property, which is the point of the test
    below. MEASURED on this branch: ``import
    fused_memory.reconciliation.consolidation_gate`` pulls 1030 modules,
    including pydantic, yaml, mem0 and ``fused_memory.config.schema`` — the
    last three arriving via ``memory_metadata`` -> ``backends.mem0_client`` ->
    ``config.schema``. ``import fused_memory.reconciliation`` alone pulls 38
    and none of them. So this module can hold ``topic_slug``'s
    ``test_module_is_import_light`` property that the gate cannot, and that is
    exactly what makes its ``ConsolidationAutoConfig`` reference a
    ``TYPE_CHECKING``-only import rather than a runtime one.

    PRD D4 records a MEASURED hard import cycle from a careless import of
    exactly this kind (``config/schema.py`` -> ``memory_metadata`` ->
    ``backends.mem0_client`` -> ``config.schema``, raising ``ImportError:
    cannot import name 'FusedMemoryConfig'``). A cycle is DIRECTIONAL — the
    measured failure only surfaced from one side — so the two orders below are
    two different probes, not one probe run twice.

    Every probe runs in a FRESH interpreter, because this test process has
    already imported everything at collection and an in-process ``sys.modules``
    check would pass vacuously.
    """

    #: Heavy modules the leaf must never pull in — the precedent's four.
    #: ``targeted``/``harness`` are the reconciliation runtime;
    #: ``services.memory_service`` and ``server.tools`` are the store and MCP
    #: layers.
    FORBIDDEN = (
        'fused_memory.reconciliation.targeted',
        'fused_memory.reconciliation.harness',
        'fused_memory.services.memory_service',
        'fused_memory.server.tools',
    )

    #: The strictly stronger set: nothing that would make this module's import
    #: cost a config load, a pydantic build or the mem0 SDK.
    HEAVY = (
        'pydantic',
        'yaml',
        'mem0',
        'fused_memory.config.schema',
        'fused_memory.memory_metadata',
    )

    @staticmethod
    def _probe(body: str):
        return subprocess.run(
            [sys.executable, '-c', body], capture_output=True, text=True, timeout=300,
        )

    def test_module_imports_stay_leaf(self):
        """Importing the leaf alone must not drag in the reconciliation
        runtime, the memory service or the MCP tool layer."""
        forbidden = ', '.join(repr(m) for m in self.FORBIDDEN)
        result = self._probe(
            'import sys\n'
            'import fused_memory.reconciliation.consolidation_auto  # noqa: F401\n'
            f'forbidden = [{forbidden}]\n'
            'present = [m for m in forbidden if m in sys.modules]\n'
            'assert not present, present\n',
        )
        assert result.returncode == 0, result.stderr

    def test_module_is_import_light(self):
        """Stdlib only: no pydantic, no yaml, no mem0, no config load.

        If this fails, someone made the ``ConsolidationAutoConfig`` reference a
        runtime import instead of a ``TYPE_CHECKING`` one, and the predicate's
        import cost is now a whole config build.
        """
        heavy = ', '.join(repr(m) for m in self.HEAVY)
        result = self._probe(
            'import sys\n'
            'import fused_memory.reconciliation.consolidation_auto  # noqa: F401\n'
            f'heavy = [{heavy}]\n'
            'present = [m for m in heavy if m in sys.modules]\n'
            'assert not present, present\n',
        )
        assert result.returncode == 0, result.stderr

    def test_imports_with_the_gate_first(self):
        """The gate leaf first, then this one."""
        result = self._probe(
            'import fused_memory.reconciliation.consolidation_gate as cg\n'
            'import fused_memory.reconciliation.consolidation_auto as ca\n'
            'assert cg is not None and ca is not None\n',
        )
        assert result.returncode == 0, result.stderr

    def test_imports_with_the_gate_second(self):
        """The reversed order. These two leaves are the pair task delta's
        executor will hold together, so both orders are production orders."""
        result = self._probe(
            'import fused_memory.reconciliation.consolidation_auto as ca\n'
            'import fused_memory.reconciliation.consolidation_gate as cg\n'
            'assert cg is not None and ca is not None\n',
        )
        assert result.returncode == 0, result.stderr
