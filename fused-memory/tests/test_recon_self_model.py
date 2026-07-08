"""Tests for fused_memory.reconciliation.recon_self_model — the single-source
self-model of recon's control-plane mechanisms (task 2220, W5-β, PRD
plans/recon-reliability-prd.md §8.4, stream W5 foundations phase).

FOUNDATIONS-FIRST: this task builds ONLY this module + these tests. The
prompt cutover (stage1.py/stage2.py importing the rendered sections) and the
premise-lint wiring at the recon submit path are task ξ.

Assertions are pinned to runtime return values (constants, rendered strings,
predicate bools, Violation lists) and stable load-bearing substrings within
rendered sections — NOT verbatim prompt-text equality, which is ξ's exact
drift invariant to own.
"""

from __future__ import annotations

from fused_memory.reconciliation import recon_self_model as m

# --------------------------------------------------------------------------- #
# Static vocabulary constants (step-1/2)
# --------------------------------------------------------------------------- #


class TestVocabularyConstants:
    """MARKER_KINDS / EXECUTION_CLASSES / MCP_CALL_SIGNATURES are the
    single-sourced static vocabulary (PRD §8.1, §8.5)."""

    def test_marker_kinds_is_full_record_kind_vocabulary(self):
        """MARKER_KINDS is the full 5-value §8.1 record_kind vocabulary, as a tuple."""
        assert m.MARKER_KINDS == (
            'stage1_flag_marker',
            'stage1_flag_suppression',
            'stage2_persistence_marker',
            'flag_for_stage2',
            'cycle_summary',
        )

    def test_execution_classes(self):
        """EXECUTION_CLASSES names the three PRD §8.5 execution classes, in order."""
        assert m.EXECUTION_CLASSES == ('code_tdd', 'operational', 'decision')

    def test_mcp_call_signatures_covers_recon_tool_surface(self):
        """MCP_CALL_SIGNATURES is a non-empty mapping covering the recon tool surface."""
        assert isinstance(m.MCP_CALL_SIGNATURES, dict)
        assert m.MCP_CALL_SIGNATURES
        required_keys = {
            'submit_task',
            'resolve_ticket',
            'add_finding',
            'cite_task',
            'add_memory',
            'search',
        }
        assert required_keys <= m.MCP_CALL_SIGNATURES.keys(), (
            f'Missing MCP_CALL_SIGNATURES keys: '
            f'{required_keys - m.MCP_CALL_SIGNATURES.keys()}'
        )
        for key in required_keys:
            sig = m.MCP_CALL_SIGNATURES[key]
            assert isinstance(sig, str) and sig, (
                f'MCP_CALL_SIGNATURES[{key!r}] must be a non-empty str, got {sig!r}'
            )


# --------------------------------------------------------------------------- #
# MARKER_LIFECYCLE + consistency with recon_ledger.MARKER_KINDS (step-3/4)
# --------------------------------------------------------------------------- #


class TestMarkerLifecycle:
    """MARKER_LIFECYCLE documents writer/deleter per record_kind, and its
    GC-on-terminal subset must equal recon_ledger.MARKER_KINDS exactly (the
    two constants are deliberately different in scope — see the module
    docstring — but must not silently drift apart)."""

    def test_every_marker_kind_has_a_lifecycle_entry(self):
        """Every MARKER_KINDS entry has a MARKER_LIFECYCLE entry with a writer and deleter."""
        for kind in m.MARKER_KINDS:
            assert kind in m.MARKER_LIFECYCLE, f'{kind!r} missing from MARKER_LIFECYCLE'
            entry = m.MARKER_LIFECYCLE[kind]
            assert isinstance(entry.writer, str) and entry.writer, (
                f'MARKER_LIFECYCLE[{kind!r}].writer must be a non-empty str'
            )
            assert isinstance(entry.deleter, str) and entry.deleter, (
                f'MARKER_LIFECYCLE[{kind!r}].deleter must be a non-empty str'
            )

    def test_ledger_gc_kinds_is_subset_of_marker_kinds(self):
        """recon_ledger.MARKER_KINDS (the GC-on-terminal marker subset) is a
        subset of the full record_kind vocabulary."""
        from fused_memory.reconciliation.recon_ledger import MARKER_KINDS as LEDGER_GC_KINDS

        assert set(LEDGER_GC_KINDS) <= set(m.MARKER_KINDS)

    def test_gc_on_terminal_subset_equals_ledger_marker_kinds(self):
        """The MARKER_LIFECYCLE kinds whose deleter is DELETER_GC equal
        recon_ledger.MARKER_KINDS exactly — i.e. exactly stage1_flag_marker,
        stage2_persistence_marker, and flag_for_stage2 are GC'd on terminal
        task; stage1_flag_suppression and cycle_summary are NOT."""
        from fused_memory.reconciliation.recon_ledger import MARKER_KINDS as LEDGER_GC_KINDS

        gc_kinds = {
            kind
            for kind, lifecycle in m.MARKER_LIFECYCLE.items()
            if lifecycle.deleter == m.DELETER_GC
        }
        assert gc_kinds == set(LEDGER_GC_KINDS), (
            f'GC-on-terminal subset {gc_kinds} must equal recon_ledger.MARKER_KINDS '
            f'{set(LEDGER_GC_KINDS)}'
        )
        assert 'stage1_flag_suppression' not in gc_kinds
        assert 'cycle_summary' not in gc_kinds


# --------------------------------------------------------------------------- #
# FINGERPRINT_IDENTITY_FIELDS + harness._derive_affected_ids cross-check (step-5/6)
# --------------------------------------------------------------------------- #


class TestFingerprintIdentityFields:
    """FINGERPRINT_IDENTITY_FIELDS single-sources the fingerprint identity
    against the live harness._derive_affected_ids logic and flag_dedup's
    content-fingerprint fallback."""

    def test_fingerprint_identity_fields_names_expected_containers(self):
        """Names the four typed citation containers, the legacy affected_ids
        field, and the content-fp fallback inputs (description, flag_type)."""
        assert set(m.FINGERPRINT_IDENTITY_FIELDS) == {
            'affected_ids',
            'cited_tasks',
            'cited_entities',
            'cited_edges',
            'cited_memories',
            'flag_type',
            'description',
        }

    def test_derive_affected_ids_reads_exactly_the_named_citation_containers(self):
        """harness._derive_affected_ids flattens exactly the four typed
        citation containers named in FINGERPRINT_IDENTITY_FIELDS."""
        from fused_memory.reconciliation.harness import _derive_affected_ids

        assert {'cited_tasks', 'cited_entities', 'cited_edges', 'cited_memories'} <= set(
            m.FINGERPRINT_IDENTITY_FIELDS
        )
        finding = {
            'cited_tasks': [{'task_id': '7'}],
            'cited_entities': [{'canonical_name': 'Foo'}],
            'cited_edges': [{'edge_uuid': 'e1'}],
            'cited_memories': [{'memory_id': 'm1'}],
        }
        result = _derive_affected_ids(finding)
        assert result == ['7', 'Foo', 'e1', 'm1'], (
            f'_derive_affected_ids must flatten the four typed citation containers '
            f'named in FINGERPRINT_IDENTITY_FIELDS; got {result!r}'
        )

    def test_content_fingerprint_fallback_fields_present(self):
        """description/flag_type are named — the content-fingerprint fallback
        inputs read by flag_dedup.compute_content_fingerprint_signature when
        no task anchor exists."""
        assert 'description' in m.FINGERPRINT_IDENTITY_FIELDS
        assert 'flag_type' in m.FINGERPRINT_IDENTITY_FIELDS


# --------------------------------------------------------------------------- #
# render_marker_lifecycle_section (step-7/8)
# --------------------------------------------------------------------------- #


class TestRenderMarkerLifecycleSection:
    """render_marker_lifecycle_section() renders the marker-lifecycle /
    run_id-fresh-per-cycle prose faithful to stage1.py:562-592."""

    def test_returns_non_empty_str(self):
        assert isinstance(m.render_marker_lifecycle_section(), str)
        assert m.render_marker_lifecycle_section()

    def test_contains_load_bearing_invariant_tokens(self):
        text = m.render_marker_lifecycle_section()
        assert 'run_id' in text
        assert 'single-cycle' in text or 'single cycle' in text
        assert 'swept' in text
        assert 'stage1_flag_marker' in text
        assert 'flag_for_stage2' in text


# --------------------------------------------------------------------------- #
# render_suppression_schema_section (step-9/10)
# --------------------------------------------------------------------------- #


class TestRenderSuppressionSchemaSection:
    """render_suppression_schema_section() renders the canonical suppression
    record schema faithful to stage1.py:498-560."""

    def test_returns_non_empty_str(self):
        assert isinstance(m.render_suppression_schema_section(), str)
        assert m.render_suppression_schema_section()

    def test_contains_canonical_schema_tokens(self):
        text = m.render_suppression_schema_section()
        assert 'stage1_flag_suppression' in text
        assert 'metadata.task_id' in text
        assert 'metadata.flag_types' in text

    def test_contains_scoped_vs_blanket_semantics(self):
        text = m.render_suppression_schema_section()
        assert 'blanket' in text
        assert 'scoped' in text
        # Blanket wins on conflict with a scoped record for the same task_id.
        assert 'wins' in text


# --------------------------------------------------------------------------- #
# render_cycle_summary_section (step-11/12)
# --------------------------------------------------------------------------- #


class TestRenderCycleSummarySection:
    """render_cycle_summary_section() renders the per-cycle summary metadata
    convention faithful to stage2.py:236-302, single-sourced from
    recon_pool_map's stage->recon_pool tags."""

    def test_returns_non_empty_str(self):
        assert isinstance(m.render_cycle_summary_section(), str)
        assert m.render_cycle_summary_section()

    def test_contains_cycle_summary_and_run_id(self):
        text = m.render_cycle_summary_section()
        assert 'cycle_summary' in text
        assert 'run_id' in text
        assert 'metadata' in text

    def test_recon_pool_tag_is_single_sourced_from_recon_pool_map(self):
        """The rendered text must contain the actual recon_pool_map constant
        value, not a re-hardcoded literal, proving it's single-sourced."""
        text = m.render_cycle_summary_section()
        assert m.STAGE2_CYCLE_SUMMARY_RECON_POOL in text
        assert m.STAGE2_CYCLE_SUMMARY_RECON_POOL == 'stage2_cycle_summary'
