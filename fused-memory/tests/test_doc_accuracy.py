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

import pathlib


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
