"""Tests for the consolidation-gate brief + closure predicate (task 3112).

Covers the two defects the task owns: the gate-filing instruction that
prescribed no end-state shape (Defect 1 — :func:`render_end_state_brief` /
:func:`render_consolidation_gate_section`), and the absence of any mechanical
refusal to close a gate task over a malformed cluster (Defect 2 —
:func:`evaluate_closure`).

Assertions are pinned to runtime return values (the verdict dataclass, the
builder's returned dict) and to stable load-bearing substrings within the
rendered prose — NOT verbatim prompt-text equality — mirroring the
``test_recon_self_model.py`` / ``test_predicate_contradiction.py`` convention.
"""

from __future__ import annotations

from fused_memory import memory_metadata
from fused_memory.reconciliation import consolidation_gate
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    render_end_state_brief,
)


class TestRenderEndStateBrief:
    """Defect 1's payload: the end-state shape a filed gate must carry.

    Load-bearing-token assertions only — the prose is free to change, the
    shape it prescribes is not.
    """

    def test_returns_non_empty_str(self):
        brief = render_end_state_brief()
        assert isinstance(brief, str)
        assert brief.strip()

    def test_mandates_the_option_c_shape(self):
        """N short single-claim peers sharing one topic, exactly one canonical."""
        brief = render_end_state_brief()
        assert 'metadata.topic' in brief
        assert 'metadata.canonical' in brief
        # The canonical must itself be SHORT — an index/summary claim, not a
        # concatenation of the cluster.  This is the property PRD §3 measured
        # the inversion on, so the brief has to say it in so many words.
        assert 'short' in brief.lower()
        assert 'index' in brief.lower()

    def test_retires_the_appendix_end_state_by_name(self):
        """The '1 canonical + 1 appendix' absorbing target is explicitly dropped.

        Naming ``appendix`` is what stops a reader mistaking WHICH target is
        being retired — an unnamed retirement is indistinguishable from prose.
        """
        brief = render_end_state_brief()
        assert 'appendix' in brief.lower()
        assert 'retired' in brief.lower()

    def test_points_at_the_landed_op_and_its_ratified_retain_arm(self):
        """The brief and the executable op cannot prescribe different end states."""
        brief = render_end_state_brief()
        assert 'consolidate_memories' in brief
        assert 'retain' in brief

    def test_public_surface_is_exported(self):
        assert 'render_end_state_brief' in consolidation_gate.__all__
        assert 'GATE_METADATA_KEY' in consolidation_gate.__all__

    def test_gate_key_is_composed_from_the_shared_experimental_prefix(self):
        """Composed, never re-spelled — the CONTESTED_METADATA_KEY precedent."""
        assert GATE_METADATA_KEY.startswith(memory_metadata.EXPERIMENTAL_KEY_PREFIX)
        assert GATE_METADATA_KEY != memory_metadata.EXPERIMENTAL_KEY_PREFIX
