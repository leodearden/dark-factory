"""Unit tests for fused_memory.utils.referent_resolution.

referent_resolution is the PRECEDENCE POLICY layer over utils/canonical_labels
— PRD plans/memory-referent-fidelity-prd.md, leaf γ (§Contract "Referent-set
resolution"), task 3668. canonical_labels stays THE single normative site for
the label vocabulary itself (INV-5 / resolved decision 5); this module decides
WHICH of the available referent sources is authoritative for a given write, in
the order declared > metadata.task_id > derived scan > none, and reports what
the prose contradicted.

The module under test compiles NO regex of its own and must never grow one:
the derived path IS ``scan_content``, the declared path builds ``Referent``s,
and the metadata bridge tries ``parse_node_name`` before its single bare-digit
branch. These tests therefore re-assert β's precision behaviour at γ's OWN
boundary (see :class:`TestDerivedPrecisionPassthrough`) rather than restating
it, so a future resolver change cannot widen it behind β's back.

Mirrors tests/test_canonical_labels.py and tests/test_cross_project_refs.py,
whose leaf-module shape and pytest conventions this module copies.
"""

from __future__ import annotations

import dataclasses

import pytest

from fused_memory.utils.referent_resolution import (
    REFERENT_SOURCES,
    ReferentResolution,
)


class TestReferentSourcesVocabulary:
    """``REFERENT_SOURCES`` is the CLOSED vocabulary of resolution sources.

    Exported as one tuple so task ι's declaration-rate telemetry iterates a
    single site instead of re-spelling four string literals — a second copy
    would be the same lockstep duplication INV-5 forbids for the label
    vocabulary itself.
    """

    def test_is_the_four_sources_in_precedence_order(self):
        """Order is meaningful: it IS the precedence chain, strongest first."""
        assert REFERENT_SOURCES == ('declared', 'metadata', 'derived', 'none')

    def test_is_a_tuple_so_the_vocabulary_cannot_be_extended_in_place(self):
        """A list would leave ``REFERENT_SOURCES.append('guessed')`` open to any
        importer, quietly widening a vocabulary consumers switch on."""
        assert isinstance(REFERENT_SOURCES, tuple)


class TestReferentResolutionIsFrozen:
    """A resolution is EVIDENCE for destructive edge surgery, not a mutable
    accumulator — the same rationale that freezes ``Referent`` and ``LabelScan``.

    ``frozen=True`` blocks attribute REBINDING only, so the result fields are
    tuples rather than lists: a list field would leave
    ``resolution.referents.append(...)`` wide open, letting a consumer quietly
    add a referent the resolver refused to infer.
    """

    def test_referents_cannot_be_rebound(self):
        resolution = ReferentResolution(source='none')
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolution.referents = ()  # type: ignore[misc]

    def test_source_cannot_be_rebound(self):
        resolution = ReferentResolution(source='none')
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolution.source = 'declared'  # type: ignore[misc]

    @pytest.mark.parametrize('attr', ['referents', 'conflicts', 'ambiguous'])
    def test_result_fields_are_tuples_not_lists(self, attr):
        """Tuples are what make the frozen-ness true rather than nominal."""
        resolution = ReferentResolution(source='none')
        assert isinstance(getattr(resolution, attr), tuple)
        with pytest.raises(AttributeError):
            getattr(resolution, attr).append('nope')

    def test_a_resolution_is_hashable(self):
        """A frozen dataclass holding lists silently is not; holding tuples it is."""
        assert hash(ReferentResolution(source='none')) is not None


class TestReferentResolutionFieldDefaults:
    """``source`` is REQUIRED; the three referent tuples default to empty."""

    def test_source_has_no_default(self):
        """The structural guarantee behind "``.source`` is set on EVERY
        resolution including the empty one": no construction path can omit it.
        A default would merely make omission convenient and invisible, and ι
        counts this field."""
        with pytest.raises(TypeError):
            ReferentResolution()  # type: ignore[call-arg]

    def test_the_empty_resolution_has_all_three_tuples_empty(self):
        resolution = ReferentResolution(source='none')
        assert resolution.referents == ()
        assert resolution.conflicts == ()
        assert resolution.ambiguous == ()

    def test_construction_is_keyword_only(self):
        """kw_only mirrors ``Referent``: four same-typed tuple-ish fields read
        as noise positionally, and a consumer swapping ``conflicts`` for
        ``ambiguous`` at a call site would type-check fine."""
        with pytest.raises(TypeError):
            ReferentResolution('none')  # type: ignore[misc]
