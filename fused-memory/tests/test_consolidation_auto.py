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
import uuid

from fused_memory.reconciliation.consolidation_auto import build_auto_canonical
from fused_memory.topic_slug import TOPIC_SLUG_MAX_LEN


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


class TestBuildAutoCanonical:
    """PRD C3: the ONE home of auto-consolidated canonical text.

    The template is duplicated between the test below and the implementation
    DELIBERATELY, and only here: it is the one thing this class exists to pin,
    and a test that derived the expected string from the implementation would
    assert nothing about drift in its wording, backticks or punctuation.
    """

    CLAIM = (
        'Stage 1 keeps re-proposing the same cluster because the ledger row is '
        'never addressed.'
    )
    TOPIC = 'memory-consolidation'
    RUN_ID = 'ce8590f1-cc05-48da-9428-1cf1f54f3fff'

    def test_renders_the_prd_template_verbatim(self):
        expected = (
            f'{self.CLAIM}\n\n'
            f'Index canonical for topic `{self.TOPIC}` over 5 short peers; the '
            f'live metadata.topic scroll is the member list (auto-consolidated, '
            f'run {self.RUN_ID}).'
        )

        assert build_auto_canonical(
            self.CLAIM, self.TOPIC, 5, self.RUN_ID,
        ) == expected

    def test_the_claim_is_the_first_paragraph_verbatim(self):
        """PRD §2 measured that a template canonical retrieves within 0.015
        cosine of a hand-written one. That property rests on the claim LEADING
        the body, so the claim must open the string and be followed by exactly
        one blank line."""
        out = build_auto_canonical(self.CLAIM, self.TOPIC, 5, self.RUN_ID)

        assert out.startswith(self.CLAIM)
        assert out[len(self.CLAIM):len(self.CLAIM) + 3] == '\n\nI', (
            'exactly one blank line must separate the claim from the index line'
        )

    def test_bound_is_under_500_chars_at_the_extremes(self):
        """PRD D5's evidence, and why there is no `canonical_max_chars` leaf.

        Every input at its own cap: a claim at the `claim_max_chars` default
        (200), a slug at TOPIC_SLUG_MAX_LEN (100), N at the `member_max`
        default (20), and a real recon run id — `str(uuid4())`, 36 chars, the
        shape reconciliation/harness.py and targeted.py both generate.

        The lower bound is asserted too: a silent shrink to a truncating
        implementation would otherwise sail past a bare `< 500`.
        """
        out = build_auto_canonical(
            'x' * 200, 't' * TOPIC_SLUG_MAX_LEN, 20, str(uuid.uuid4()),
        )

        assert len(out) < 500, len(out)
        assert len(out) >= 460, (
            f'expected the measured 464-char maximum, got {len(out)} — a '
            'shorter maximum means the builder is clipping something'
        )

    def test_extreme_inputs_are_not_truncated(self):
        """The builder never clips: a 200-char claim appears in full."""
        claim = 'y' * 200
        out = build_auto_canonical(claim, 't' * TOPIC_SLUG_MAX_LEN, 20, self.RUN_ID)

        assert claim in out
