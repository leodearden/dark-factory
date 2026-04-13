"""Documentation-accuracy guards for task 773.

Guard 1 (TestPass2GuardDocstringWording):
    Asserts that the docstring of
    TestContextAssemblerCancellation.test_pass2_guard_narrowed_to_exception
    does NOT contain the stale phrasing
    'current ``isinstance(ctx_result, BaseException)`` guard'.
    After task 573 merged, the 'current' guard is the NARROWED
    isinstance(ctx_result, Exception) one; the BaseException guard is the 'old'
    one.  The later sentences in the same docstring already use 'old' / 'narrowed'
    correctly; only the opening sentence retained the misleading 'current' token.

Guard 2 (TestAssembleCrossReferenceComment):
    Asserts that the comment block inside ContextAssembler.assemble that introduces
    the two-tier gather(return_exceptions=True) check lists propagate_cancellations
    as the PRIMARY cross-reference ('See propagate_cancellations ...') and demotes
    graphiti_client.rebuild_entity_summaries to a SUPPLEMENTARY example, rather than
    naming graphiti_client.rebuild_entity_summaries as the 'canonical' reference.
    Rationale: propagate_cancellations is the shared contract for all
    gather(return_exceptions=True) callsites; a cross-reference to a concrete
    implementation (graphiti_client) goes stale if that module is renamed or moved.
"""

import inspect
import pathlib

from fused_memory.reconciliation.context_assembler import ContextAssembler


class TestPass2GuardDocstringWording:
    """Guard: the stale 'current BaseException guard' phrasing is absent from
    test_context_assembler.py.

    The sentence originally read:
        'With the current ``isinstance(ctx_result, BaseException)`` guard, ...'
    which is misleading after task 573 landed — the *current* guard is the
    narrowed Exception one.  The fix is a one-word swap: 'current' -> 'old'.
    """

    def test_docstring_does_not_say_current_baseexception_guard(self) -> None:
        """'current ``isinstance(ctx_result, BaseException)`` guard' must not appear."""
        tests_dir = pathlib.Path(__file__).parent
        source = (tests_dir / "test_context_assembler.py").read_text()
        stale = "current ``isinstance(ctx_result, BaseException)`` guard"
        assert stale not in source, (
            f"Stale phrasing {stale!r} found in test_context_assembler.py. "
            "After task 573, the *current* guard is the narrowed "
            "isinstance(ctx_result, Exception) one; the BaseException guard "
            "is the 'old' guard.  Replace 'current' with 'old' in the opening "
            "sentence of test_pass2_guard_narrowed_to_exception's docstring "
            "(see task 773 for the 'old vs narrowed' rationale)."
        )


class TestAssembleCrossReferenceComment:
    """Guard: the comment inside ContextAssembler.assemble lists
    propagate_cancellations as the PRIMARY cross-reference and demotes
    graphiti_client.rebuild_entity_summaries to a SUPPLEMENTARY example.

    Before task 773's fix the comment ended with:
        'See graphiti_client.rebuild_entity_summaries for the canonical
        two-pass reference (Pass 1 via propagate_cancellations + Pass 2
        with isinstance(r, Exception)).'

    After the fix the wording is (approximately):
        'See propagate_cancellations for the Pass 1 contract;
        graphiti_client.rebuild_entity_summaries for a complete two-pass
        example.'

    Rationale: propagate_cancellations is the authoritative shared contract
    for all gather(return_exceptions=True) callsites; naming a concrete module
    (graphiti_client) as 'canonical' creates documentation coupling that goes
    stale if that module is renamed or moved.
    """

    def _assemble_source(self) -> str:
        return inspect.getsource(ContextAssembler.assemble)

    def test_primary_reference_is_propagate_cancellations(self) -> None:
        """Comment must say 'See propagate_cancellations' and NOT 'See graphiti_client.rebuild_entity_summaries'."""
        source = self._assemble_source()
        assert "See propagate_cancellations" in source, (
            "ContextAssembler.assemble comment must include 'See propagate_cancellations' "
            "as the primary cross-reference for the two-tier gather pattern "
            "(task 773: propagate_cancellations is the shared Pass 1 contract)."
        )
        stale = "See graphiti_client.rebuild_entity_summaries"
        assert stale not in source, (
            f"Stale primary reference {stale!r} found in ContextAssembler.assemble. "
            "graphiti_client.rebuild_entity_summaries should be the SUPPLEMENTARY "
            "example, not the canonical reference.  See task 773 for rationale."
        )

    def test_graphiti_client_reference_is_supplementary(self) -> None:
        """graphiti_client.rebuild_entity_summaries must still appear but AFTER propagate_cancellations."""
        source = self._assemble_source()
        assert "graphiti_client.rebuild_entity_summaries" in source, (
            "ContextAssembler.assemble must still mention "
            "'graphiti_client.rebuild_entity_summaries' as a supplementary "
            "two-pass example (task 773: demote to supplementary, do not remove)."
        )
        pc_pos = source.index("propagate_cancellations")
        gc_comment_phrase = "graphiti_client.rebuild_entity_summaries"
        gc_pos = source.index(gc_comment_phrase)
        assert pc_pos < gc_pos, (
            f"Expected 'propagate_cancellations' (pos {pc_pos}) to appear before "
            f"'graphiti_client.rebuild_entity_summaries' (pos {gc_pos}) in "
            "ContextAssembler.assemble source — primary reference must precede "
            "the supplementary example (task 773)."
        )
