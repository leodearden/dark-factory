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

import pytest

from fused_memory.reconciliation import recon_self_model as m
from fused_memory.reconciliation import standing_decision_constants as sdc
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    render_consolidation_gate_section,
    render_end_state_brief,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

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

    def test_mem0_tombstone_lifecycle_is_registered_with_ttl_deleter(self):
        """mem0_tombstone (task 3041) is documented in MARKER_LIFECYCLE and
        expires purely by TTL.

        NOT DELETER_GC: that would break the subset==recon_ledger.MARKER_KINDS
        invariant below. NOT DELETER_POOL_TRIM either — a tombstone is never
        evicted by a pool cap, only by its own expires_at.
        """
        entry = m.MARKER_LIFECYCLE['mem0_tombstone']
        assert isinstance(entry.writer, str) and entry.writer
        assert entry.deleter == m.DELETER_TTL, (
            f'a tombstone expires only via expires_at, got {entry.deleter!r}'
        )

    def test_mem0_tombstone_is_in_neither_marker_kinds_constant(self):
        """A Mem0 memory uuid must never reach a task-id-keyed path.

        mem0_tombstone's task_id column holds a memory uuid, not a Taskmaster
        task id, so it is registered in MARKER_LIFECYCLE ONLY — MARKER_LIFECYCLE
        is a SUPERSET of the record_kind vocabulary, not a mirror of it.
        """
        from fused_memory.reconciliation.recon_ledger import MARKER_KINDS as LEDGER_GC_KINDS

        assert 'mem0_tombstone' not in LEDGER_GC_KINDS
        assert 'mem0_tombstone' not in m.MARKER_KINDS

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

    def test_mem0_tombstone_writer_names_the_batched_writer(self):
        """Regression pin (task 4421): MARKER_LIFECYCLE['mem0_tombstone'].writer
        must name the PLURAL batched writer mem0_tombstone.
        record_mem0_deletion_tombstones — the form all three production call
        sites (summary_pool.enforce_summary_pool_cap,
        task_knowledge_sync._sweep_stale_mem0_pool,
        server.tools.consolidate_memories) actually use.

        Ground truth verified this run: the SINGULAR
        record_mem0_deletion_tombstone has ZERO production callers — its only
        non-test occurrences repo-wide are its own `def`, a docstring
        cross-reference in memory_service.py/recon_ledger.py, its own
        docstring, and the __all__ export at mem0_tombstone.py:108. The
        batched record_mem0_deletion_tombstones has three production callers.
        """
        writer = m.MARKER_LIFECYCLE['mem0_tombstone'].writer
        # Names the plural writer that actually has production callers.
        assert 'record_mem0_deletion_tombstones' in writer
        # Does NOT name the singular form as the writer. 'record_mem0_deletion_tombstone'
        # is a proper substring of 'record_mem0_deletion_tombstones', so a naive
        # `not in` can never pass here — counting occurrences is what actually
        # distinguishes "names only the plural" from "names the singular too".
        assert writer.count('record_mem0_deletion_tombstone') == writer.count(
            'record_mem0_deletion_tombstones'
        )
        # Scope is the named deleters, not "every recon-initiated delete".
        assert 'CONFIRMED recon-initiated Mem0 delete' not in writer


# --------------------------------------------------------------------------- #
# MEM0_TOMBSTONE_DELETERS — single-sourced against the live deleter tags
# --------------------------------------------------------------------------- #


class TestMem0TombstoneDeleters:
    """MEM0_TOMBSTONE_DELETERS names the deleter audit tags of the ONLY Mem0
    delete paths that write a mem0_tombstone ledger row, via
    mem0_tombstone.record_mem0_deletion_tombstones. Hand-declared in
    recon_self_model (import-light contract — see module docstring) rather
    than imported, and cross-checked here against the live constants so
    drift between the two fails a test instead of silently diverging."""

    def test_mem0_tombstone_deleters_shape(self):
        """MEM0_TOMBSTONE_DELETERS is a non-empty tuple of non-empty str."""
        assert isinstance(m.MEM0_TOMBSTONE_DELETERS, tuple)
        assert m.MEM0_TOMBSTONE_DELETERS
        for deleter in m.MEM0_TOMBSTONE_DELETERS:
            assert isinstance(deleter, str) and deleter

    def test_mem0_tombstone_deleters_match_live_delete_sites(self):
        """Drift ratchet: MEM0_TOMBSTONE_DELETERS must equal the six live
        deleter tags used by the three production call sites of
        mem0_tombstone.record_mem0_deletion_tombstones (summary_pool.
        enforce_summary_pool_cap, task_knowledge_sync._sweep_stale_mem0_pool,
        server.tools.consolidate_memories).

        KNOWN LIMIT of this ratchet: it catches a RENAMED or REMOVED deleter
        tag (the live import breaks or the set comparison fails), but a
        brand-new call site introducing a brand-new constant is only caught
        once someone adds it to `expected` below — it cannot discover an
        unknown-unknown deleter on its own. That is the honest scope of this
        guard, matching MARKER_KINDS' cross-check against recon_ledger.

        Imports the stage modules lazily, inside the test body (costs ~20s
        cold) rather than at module scope, matching this file's existing
        `from fused_memory.reconciliation.recon_ledger import MARKER_KINDS`
        idiom (see test_mem0_tombstone_is_in_neither_marker_kinds_constant
        above) — recon_self_model itself must stay import-light.
        """
        from fused_memory.reconciliation.stages import memory_consolidator as mc
        from fused_memory.reconciliation.stages import task_knowledge_sync as tks
        from fused_memory.server import tools as srv_tools

        expected = {
            tks._STAGE2_PERSISTENCE_MARKER_GC_SWEEP_SOURCE,
            tks._STAGE1_FLAG_MARKER_GC_SWEEP_SOURCE,
            tks._FLAG_FOR_STAGE2_GC_SWEEP_SOURCE,
            tks._STAGE2_CYCLE_SUMMARY_TRIM_SOURCE,
            mc._STAGE1_CYCLE_SUMMARY_TRIM_SOURCE,
            srv_tools._CONSOLIDATE_SOURCE,
        }
        assert set(m.MEM0_TOMBSTONE_DELETERS) == expected


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

    def test_tombstone_claim_is_not_universal(self):
        """Regression pin (task 4421): the false universal quantifier must be
        gone. 'Any recon-initiated Mem0 deletion now leaves a TOMBSTONE'
        licensed the invalid contrapositive 'no tombstone => not deliberately
        deleted => data loss', which manufactures phantom data-loss reports
        (recon gate 165 / esc-165-1, task 3041) — six of the nine production
        Mem0 delete call sites write no tombstone at all."""
        text = m.render_cycle_summary_section()
        assert 'Any recon-initiated Mem0 deletion' not in text

    def test_tombstone_claim_names_its_deleters(self):
        """The scoped claim is single-sourced from MEM0_TOMBSTONE_DELETERS —
        every entry must appear in the rendered text, so the prose cannot
        silently disagree with the step-1 drift ratchet."""
        text = m.render_cycle_summary_section()
        for deleter in m.MEM0_TOMBSTONE_DELETERS:
            assert deleter in text, f'{deleter!r} missing from render_cycle_summary_section()'

    def test_tombstone_absence_is_explicitly_not_evidence(self):
        """The contrapositive retraction must be present: absence of a
        tombstone is NOT evidence a record wasn't deliberately deleted.
        Pinned via load-bearing tokens (this file's stated convention, see
        module docstring), not a verbatim sentence."""
        text = m.render_cycle_summary_section()
        assert 'absence' in text.lower()
        # The untombstoned MCP delete_memory tool is the load-bearing gap —
        # it's what the recon stages' own flag-reaping instructions call.
        assert 'delete_memory' in text

    def test_tombstone_true_half_is_retained(self):
        """The surviving true half must not be lost in the rewrite: a
        tombstone IS surfaced by get_memory_by_id on its not-found branch."""
        text = m.render_cycle_summary_section()
        assert 'tombstone' in text
        assert 'get_memory_by_id' in text


# --------------------------------------------------------------------------- #
# render_execution_class_section (step-13/14)
# --------------------------------------------------------------------------- #


class TestRenderExecutionClassSection:
    """render_execution_class_section() renders the NEW execution_class
    contract text (PRD §8.5) — there is no current prompt precedent for this
    text (execution_class is net-new); this section is consumed by η."""

    def test_returns_non_empty_str(self):
        assert isinstance(m.render_execution_class_section(), str)
        assert m.render_execution_class_section()

    def test_names_every_execution_class(self):
        """Iterates EXECUTION_CLASSES so the section stays single-sourced
        from the constant rather than re-hardcoding the class names."""
        text = m.render_execution_class_section()
        for execution_class in m.EXECUTION_CLASSES:
            assert execution_class in text, (
                f'render_execution_class_section() must name {execution_class!r}'
            )

    def test_states_declaration_is_required_and_rejected_otherwise(self):
        """A recon-stage submit_task must set metadata.execution_class;
        omitting it (or an unknown value) is rejected."""
        text = m.render_execution_class_section()
        assert 'metadata.execution_class' in text
        assert 'submit_task' in text
        assert 'rejected' in text

    def test_states_operational_and_decision_route_off_the_tdd_pipeline(self):
        """operational/decision asks are routed off the architect+TDD
        pipeline; code_tdd stays on it."""
        text = m.render_execution_class_section()
        assert 'architect' in text
        assert 'TDD' in text
        assert 'routed off' in text


# --------------------------------------------------------------------------- #
# render_source_completion_section (PRD plans/operational-ask-routing-prd.md task ε)
# --------------------------------------------------------------------------- #


class TestRenderSourceCompletionSection:
    """render_source_completion_section(*, can_file_tasks) renders the NEW
    source-completion brief (PRD task ε): the tool-holding recon stage COMPLETES
    safe memory merges inline (it already holds add_memory/delete_memory/
    merge_entities — DISALLOW_MEMORY_WRITES scopes only Stage 3) and files ONLY
    the residual irreversible judgment call as an `operational` +
    `operational_mode='gate'` task.

    The residual-handling clause differs per stage — Stage 2 holds submit_task
    and files the residual itself; Stage 1 lacks it (DISALLOW_TASK_WRITES) and
    must relay it to Stage 2 — so the function is parameterized on
    can_file_tasks rather than being one shared verbatim block.
    """

    def test_returns_non_empty_str_both_modes(self):
        for can_file in (True, False):
            text = m.render_source_completion_section(can_file_tasks=can_file)
            assert isinstance(text, str), f'can_file_tasks={can_file} must return str'
            assert text, f'can_file_tasks={can_file} must return a non-empty str'

    def test_both_modes_state_safe_merge_inline_and_conservative_predicate(self):
        """Both stages hold the memory-mutation tools, so both are told to
        complete safe (exact/near-exact duplicate) merges INLINE and to escalate
        only the content-losing / irreversible judgment call."""
        for can_file in (True, False):
            text = m.render_source_completion_section(can_file_tasks=can_file)
            assert 'inline' in text, f'can_file_tasks={can_file} must direct inline merge'
            assert 'duplicate' in text, f'can_file_tasks={can_file} must name the duplicate case'
            assert 'near-exact' in text, (
                f'can_file_tasks={can_file} must state the conservative near-exact predicate'
            )
            assert 'irreversible' in text, (
                f'can_file_tasks={can_file} must reserve escalation for the irreversible call'
            )

    def test_both_modes_state_operational_gate_filing_vocabulary(self):
        """Both stages route the residual to the operational + operational_mode='gate'
        human gate (the metadata field delivered by prereq task α/2801). Assert the
        full literals the function emits: bare 'operational' is a trivial substring
        of 'operational_mode', and neither bare token distinguishes the 'gate' mode
        from the 'llm' mode the function's own text explicitly contrasts."""
        for can_file in (True, False):
            text = m.render_source_completion_section(can_file_tasks=can_file)
            assert "metadata.operational_mode='gate'" in text, (
                f"can_file_tasks={can_file} must emit metadata.operational_mode='gate'"
            )
            assert "execution_class='operational'" in text, (
                f"can_file_tasks={can_file} must emit execution_class='operational'"
            )

    def test_can_file_tasks_true_files_residual_via_submit_task(self):
        """Stage 2 holds submit_task, so it files the residual itself."""
        text = m.render_source_completion_section(can_file_tasks=True)
        assert 'submit_task' in text, (
            'the can_file_tasks=True (Stage 2) variant must file the residual via submit_task'
        )

    def test_can_file_tasks_false_relays_to_stage2_without_self_filing(self):
        """Stage 1 lacks submit_task (DISALLOW_TASK_WRITES), so it must relay the
        residual to Stage 2 rather than be told to file it itself — honoring the
        loud-over-silent norm of never instructing a stage to call a tool it does
        not hold."""
        text = m.render_source_completion_section(can_file_tasks=False)
        # Relays the residual to Stage 2.
        assert 'Stage 2' in text, (
            'the can_file_tasks=False (Stage 1) variant must relay the residual to Stage 2'
        )
        # Positively told it does NOT hold submit_task — never instructed to call it itself.
        assert 'do NOT hold submit_task' in text, (
            'the Stage 1 variant must state it does NOT hold submit_task'
        )
        # The Stage-2-only affirmative self-file directive must be absent here.
        assert 'file it yourself' not in text, (
            'the Stage 1 variant must NOT tell Stage 1 to file the residual itself'
        )


# --------------------------------------------------------------------------- #
# Invariant predicates (step-15/16)
# --------------------------------------------------------------------------- #


class TestInvariantPredicates:
    """run_id_is_fresh_per_run() / markers_deleted_only_by_gc() are the
    assertable predicates premise_lint (and downstream consumers) reference."""

    def test_run_id_is_fresh_per_run(self):
        result = m.run_id_is_fresh_per_run()
        assert isinstance(result, bool)
        assert result is True

    def test_markers_deleted_only_by_gc(self):
        result = m.markers_deleted_only_by_gc()
        assert isinstance(result, bool)
        assert result is True


# --------------------------------------------------------------------------- #
# premise_lint + Violation (step-17/18)
# --------------------------------------------------------------------------- #


class TestPremiseLint:
    """premise_lint(task_description) -> list[Violation] flags task
    descriptions containing known-false premises about recon's control-plane
    mechanisms (the 2083/2092/2093 false-premise batch)."""

    def test_flags_run_id_persistence_premise(self):
        """A description asserting run_id persists across cycles is flagged,
        referencing the run_id-fresh invariant."""
        violations = m.premise_lint('run_id persists across cycles')
        assert violations
        for v in violations:
            assert v.invariant == 'run_id_is_fresh_per_run'

    def test_flags_stage3_deletes_marker_premise(self):
        """A description asserting Stage 3 remediation deletes the flag
        marker is flagged, referencing the markers_deleted_only_by_gc
        invariant."""
        violations = m.premise_lint('Stage 3 remediation deletes the flag marker')
        assert violations
        for v in violations:
            assert v.invariant == 'markers_deleted_only_by_gc'

    def test_benign_description_returns_empty_list(self):
        """A benign, true description matches no known false premise."""
        violations = m.premise_lint('Reconcile task 7 status against the knowledge graph')
        assert violations == []

    def test_violation_is_a_frozen_dataclass(self):
        import dataclasses

        v = m.Violation(premise='x', invariant='y', detail='z')
        assert dataclasses.is_dataclass(v)
        with pytest.raises(dataclasses.FrozenInstanceError):
            # setattr, not a direct attribute assignment, so this stays pyright-clean
            # (a direct assignment on a frozen dataclass is reportAttributeAccessIssue).
            setattr(v, 'premise', 'mutated')  # noqa: B010


# --------------------------------------------------------------------------- #
# render_entity_standing_decision_schema_section (task 2898 ε, step-1/2)
# --------------------------------------------------------------------------- #


class TestRenderEntityStandingDecisionSchemaSection:
    """render_entity_standing_decision_schema_section() renders the SHARED
    entity_standing_decision schema section (record kind + grounds enum), the
    ad-hoc-kind demotion prose, and the pre-emission advisory-check prose —
    rendered byte-identically into BOTH the Stage-1 and Stage-2 prompts."""

    def test_returns_non_empty_str(self):
        assert isinstance(m.render_entity_standing_decision_schema_section(), str)
        assert m.render_entity_standing_decision_schema_section()

    def test_record_kind_is_single_sourced_from_constants(self):
        """The rendered text contains the α record-kind constant VALUE (not a
        re-hardcoded literal), proving it is single-sourced from
        standing_decision_constants (INV-5)."""
        text = m.render_entity_standing_decision_schema_section()
        assert sdc.RECORD_KIND_ENTITY_STANDING_DECISION in text
        assert sdc.RECORD_KIND_ENTITY_STANDING_DECISION == 'entity_standing_decision'

    def test_grounds_enum_value_is_single_sourced(self):
        """The rendered text contains the α grounds enum constant VALUE."""
        text = m.render_entity_standing_decision_schema_section()
        assert sdc.GROUNDS_STRUCTURAL_SIZE_CONFLATION in text
        assert sdc.GROUNDS_STRUCTURAL_SIZE_CONFLATION == 'structural_size_conflation'

    def test_demotion_names_both_ad_hoc_kind_identifiers(self):
        """Item 4: the demotion prose names the two exact ad-hoc mem0 kind
        identifiers it demotes to evidence-only. These are specific identifier
        tokens (not free-form prose), so pinning them guards that the renderer
        keeps naming both demoted kinds. The weaker 'evidence-only'/'sole'
        wording pin is intentionally dropped — it can pass on unrelated prose,
        and the demotion prose is already pinned byte-identically into both
        assembled prompts by test_standing_decision_prompt_drift.py."""
        text = m.render_entity_standing_decision_schema_section()
        assert 'recurring_flag_standing_decision' in text
        assert 'stage1_finding_correction' in text

    def test_advisory_check_names_the_mem0_read_tool(self):
        """Item 2: the pre-emission advisory check names the specific mem0-read
        tool (`get_memories_by_metadata`) — the one real-invariant instruction
        token worth pinning intentionally. The surrounding advisory wording
        ('advisory', 'authoritative', 'never consult mem0') is left to the
        byte-identical drift guard rather than pinned here as churny prose."""
        text = m.render_entity_standing_decision_schema_section()
        assert 'get_memories_by_metadata' in text


# --------------------------------------------------------------------------- #
# render_investigation_outcome_section (task 2898 ε, step-3/4)
# --------------------------------------------------------------------------- #


class TestRenderInvestigationOutcomeSection:
    """render_investigation_outcome_section() renders the STAGE-2-ONLY
    investigation_outcome mem0-kind schema plus the Stage-2 write instruction
    (write one record on every not-actionable investigation conclusion; the
    pool feeds β's authorization arm-2 evidence)."""

    def test_returns_non_empty_str(self):
        assert isinstance(m.render_investigation_outcome_section(), str)
        assert m.render_investigation_outcome_section()

    def test_kind_is_single_sourced_from_constants(self):
        """The rendered text contains the α mem0-kind constant VALUE (not a
        re-hardcoded literal), proving it is single-sourced (INV-5)."""
        text = m.render_investigation_outcome_section()
        assert sdc.MEM0_KIND_INVESTIGATION_OUTCOME in text
        assert sdc.MEM0_KIND_INVESTIGATION_OUTCOME == 'investigation_outcome'

    def test_names_record_schema_field_identifiers(self):
        """The record schema is defined by three exact metadata field
        identifiers (`entity_uuid`, `actionable`, `run_id`) — structural schema
        tokens, not free-form prose — so pinning them guards the record
        convention this section introduces. The weaker prose pins the original
        draft carried ('false', 'not-actionable', 'arm'/'authorization'/
        'evidence') are dropped: they can pass on unrelated prose, and the full
        section text is already pinned byte-identically into the Stage-2 prompt
        by test_standing_decision_prompt_drift.py."""
        text = m.render_investigation_outcome_section()
        assert 'entity_uuid' in text
        assert 'actionable' in text
        assert 'run_id' in text


class TestRenderConsolidationGateSection:
    """render_consolidation_gate_section() is Defect 1's prompt-side payload
    (task 3112).

    Until it landed, render_source_completion_section was the WHOLE gate-filing
    instruction and it named no end state, so each filed gate invented its own
    (DF gates 2969/2973/3011/3016/3036/3063/3092). This section supplies the
    Option-C shape and points the closure predicate and the prompt at one text.

    Every end-state assertion below is parameterized over BOTH
    ``can_file_tasks`` variants: the target end state is identical for the two
    stages, and only WHO files the gate differs.

    Load-bearing-token assertions only, following TestRenderSourceCompletionSection.
    """

    @pytest.mark.parametrize('can_file', [True, False])
    def test_returns_non_empty_str(self, can_file):
        text = render_consolidation_gate_section(can_file_tasks=can_file)
        assert isinstance(text, str)
        assert text.strip()

    @pytest.mark.parametrize('can_file', [True, False])
    def test_names_the_option_c_end_state(self, can_file):
        text = render_consolidation_gate_section(can_file_tasks=can_file)
        assert 'metadata.topic' in text
        assert 'metadata.canonical' in text

    @pytest.mark.parametrize('can_file', [True, False])
    def test_instructs_the_filer_to_emit_a_topic_not_a_member_list(self, can_file):
        """A hand-written enumeration is what DF gate 3036 did, and a later
        cycle extended it 7->8 while it still defined 'done'."""
        text = render_consolidation_gate_section(can_file_tasks=can_file)
        assert GATE_METADATA_KEY in text
        assert 'build_consolidation_gate_task' in text

    @pytest.mark.parametrize('can_file', [True, False])
    def test_names_the_seam_that_enforces_closure(self, can_file):
        """The refusal is the user-observable signal, so the filer must know
        WHERE it fires before they file, not discover it at close time.

        ``set_task_status`` is the tool identifier for the one seam step-14
        gates.  The refusal BEHAVIOUR is covered by
        ``test_consolidation_closure_seam.py``'s structured assertions on
        ``result['error'] == 'consolidation_not_closed'`` — not here, by prose.
        """
        text = render_consolidation_gate_section(can_file_tasks=can_file)
        assert 'set_task_status' in text

    @pytest.mark.parametrize('can_file', [True, False])
    def test_reuses_the_end_state_brief_verbatim(self, can_file):
        """One text for the prompt, the filed gate description and the
        predicate's target — so they cannot drift into disagreeing."""
        assert render_end_state_brief() in render_consolidation_gate_section(
            can_file_tasks=can_file
        )

    def test_stage1_variant_relays_instead_of_filing(self):
        """Stage 1 does NOT hold `submit_task`, so its variant must not tell it
        to build a submission — it must relay the gate to Stage 2.

        The relay uses the SAME channel vocabulary
        ``render_source_completion_section``'s `can_file_tasks=False` branch
        already uses, so the two clauses in one prompt reinforce rather than
        compete, and it must carry the payload Stage 2 needs to build the gate
        itself: the cluster's topic slug and the rationale.
        """
        text = render_consolidation_gate_section(can_file_tasks=False)
        assert 'as_submit_task_kwargs' not in text
        assert 'submit_task' in text  # named, in its DENIAL sense
        assert 'flag_for_stage2' in text
        assert 'flagged_items' in text
        assert 'topic' in text
        assert 'rationale' in text

    def test_stage2_variant_files_it_directly(self):
        """Stage 2 holds `submit_task`, so it files the gate itself."""
        text = render_consolidation_gate_section(can_file_tasks=True)
        assert 'as_submit_task_kwargs' in text

    def test_both_stage_prompts_embed_the_correct_variant(self):
        """Both stages hold the memory-mutation tools, so both can REACH a
        gate-worthy judgment call — but only Stage 2 holds `submit_task`, so
        each prompt must carry the variant matching what that stage can do."""
        assert (
            render_consolidation_gate_section(can_file_tasks=False)
            in STAGE1_SYSTEM_PROMPT
        )
        assert (
            render_consolidation_gate_section(can_file_tasks=True)
            in STAGE2_SYSTEM_PROMPT
        )
        # Anti-regression direction: the filing variant must never reach Stage 1.
        assert (
            render_consolidation_gate_section(can_file_tasks=True)
            not in STAGE1_SYSTEM_PROMPT
        )

    def test_stage1_prompt_never_instructs_a_denied_task_write(self):
        """Asserted against the WHOLE assembled Stage 1 prompt, not one
        section, so a future section added elsewhere cannot reintroduce this.

        ``mcp__fused-memory__submit_task`` is a member of
        ``cli_stage_runner::DISALLOW_TASK_WRITES``, which
        ``cli_stage_runner::STAGE1_DISALLOWED`` folds in — so a Stage 1 prompt
        that says to submit a gate task is instructing a tool the stage does
        not hold.  The rule, quoting ``render_source_completion_section``'s own
        docstring: "Never instruct Stage 1 to call a tool it does not hold
        (loud-over-silent)."
        """
        assert 'as_submit_task_kwargs' not in STAGE1_SYSTEM_PROMPT

    def test_it_sits_alongside_the_source_completion_section(self):
        assert m.render_source_completion_section(can_file_tasks=False) in (
            STAGE1_SYSTEM_PROMPT
        )
        assert m.render_source_completion_section(can_file_tasks=True) in (
            STAGE2_SYSTEM_PROMPT
        )

    def test_source_completion_defers_to_it_for_gate_shape(self):
        """The two sections must not give the stages conflicting consolidation
        instructions. render_source_completion_section's silence on shape IS
        Defect 1's root cause, so it now names this section as the authority."""
        for can_file in (True, False):
            text = m.render_source_completion_section(can_file_tasks=can_file)
            # Resolvable identifiers, not a heading phrase: both live in the
            # shared trailing clause OUTSIDE the can_file_tasks if/else, so a
            # rename of either symbol breaks this instead of silently
            # orphaning the cross-reference.
            assert 'consolidation_gate' in text, (
                f'can_file_tasks={can_file} must point at the gate module'
            )
            assert 'render_consolidation_gate_section' in text, (
                f'can_file_tasks={can_file} must point at the gate section'
            )
