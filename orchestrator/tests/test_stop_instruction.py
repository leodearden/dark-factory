"""Tests for orchestrator.stop_instruction — shared stop-instruction phrase detector.

Task 2509 (governance gap, reconciliation finding 0aac21b4): a single,
pure, stdlib-only detector is the one source of truth reused by b3_gate's
mechanical hard-abort and DeterministicRunner's before_done guard (and
referenced by the /unblock + unblock-low-risk runbook prose), so the
curated phrase set cannot drift between the three enforcement surfaces.

step-1: STOP_INSTRUCTION_PHRASES shape (non-empty, multi-word-only,
        lowercased/stripped) + detect_stop_instruction matching contract
        (case/hyphen/whitespace-insensitive, None-safe, varargs).
"""

from __future__ import annotations


class TestStopInstructionPhrasesShape:
    """STOP_INSTRUCTION_PHRASES is a curated, multi-word-only phrase set."""

    def test_is_non_empty_frozenset(self):
        from orchestrator.stop_instruction import STOP_INSTRUCTION_PHRASES
        assert isinstance(STOP_INSTRUCTION_PHRASES, frozenset)
        assert len(STOP_INSTRUCTION_PHRASES) > 0

    def test_contains_do_not_apply_and_human_rehearsal(self):
        from orchestrator.stop_instruction import STOP_INSTRUCTION_PHRASES
        assert 'do not apply' in STOP_INSTRUCTION_PHRASES
        assert 'human rehearsal' in STOP_INSTRUCTION_PHRASES

    def test_all_entries_lowercased_and_stripped(self):
        from orchestrator.stop_instruction import STOP_INSTRUCTION_PHRASES
        for phrase in STOP_INSTRUCTION_PHRASES:
            assert phrase == phrase.casefold(), f'{phrase!r} is not lowercased/casefolded'
            assert phrase == phrase.strip(), f'{phrase!r} has leading/trailing whitespace'

    def test_no_bare_single_word_entries(self):
        """No entry may be a single word (e.g. 'stop'/'halt') — bounds false positives."""
        from orchestrator.stop_instruction import STOP_INSTRUCTION_PHRASES
        for phrase in STOP_INSTRUCTION_PHRASES:
            assert ' ' in phrase, f'{phrase!r} is a bare single-word entry — not allowed'
        assert 'stop' not in STOP_INSTRUCTION_PHRASES
        assert 'halt' not in STOP_INSTRUCTION_PHRASES


class TestDetectStopInstructionMatching:
    """detect_stop_instruction matches case/hyphen/whitespace variants."""

    def test_uppercase_variant_matches(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        result = detect_stop_instruction('Please note: DO NOT APPLY this patch yet.')
        assert result == 'do not apply', f'expected "do not apply", got {result!r}'

    def test_hyphenated_variant_matches(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        result = detect_stop_instruction('Investigated the fix — do not auto-apply until reviewed.')
        assert result is not None, 'expected a match for the hyphenated variant'

    def test_spaced_variant_of_same_phrase_matches(self):
        """The non-hyphenated ('do not auto apply') spelling must match the SAME
        canonical phrase as the hyphenated ('do not auto-apply') spelling above —
        normalization must be applied uniformly to both text and phrase."""
        from orchestrator.stop_instruction import detect_stop_instruction
        hyphenated = detect_stop_instruction('do not auto-apply until reviewed')
        spaced = detect_stop_instruction('do not auto apply until reviewed')
        assert hyphenated is not None
        assert spaced is not None
        assert hyphenated == spaced, (
            f'hyphenated and spaced variants must resolve to the same canonical '
            f'phrase: {hyphenated!r} != {spaced!r}'
        )

    def test_hyphenated_title_case_variant_matches(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        result = detect_stop_instruction('Human-Rehearsal required before this lands.')
        assert result == 'human rehearsal', f'expected "human rehearsal", got {result!r}'

    def test_benign_text_returns_none(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        result = detect_stop_instruction('apply the fix and merge the branch')
        assert result is None

    def test_empty_string_returns_none(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        assert detect_stop_instruction('') is None

    def test_none_arg_is_safe_and_returns_none(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        assert detect_stop_instruction(None) is None

    def test_no_args_returns_none(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        assert detect_stop_instruction() is None


class TestDetectStopInstructionDeterminism:
    """When a text matches more than one phrase, the reported phrase must be
    stable/reproducible (task 2509 review amendment) rather than depend on
    frozenset iteration order, which is not guaranteed stable across
    processes."""

    def test_multiple_matches_return_the_same_phrase_every_call(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        text = 'Please do not run this step, and also do not execute that one.'
        # Both 'do not run' and 'do not execute' are present. Sorted order
        # ('do not execute' < 'do not run') must pick the same phrase on
        # every call, in-process and across processes.
        results = {detect_stop_instruction(text) for _ in range(10)}
        assert results == {'do not execute'}, (
            f'expected a single stable phrase across repeated calls, got {results!r}'
        )


class TestDetectStopInstructionVarargs:
    """detect_stop_instruction(*texts) matches if ANY positional text matches."""

    def test_match_in_any_position_is_found(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        result = detect_stop_instruction(
            'benign summary', 'do not apply this change', 'another benign string',
        )
        assert result == 'do not apply', f'expected "do not apply", got {result!r}'

    def test_none_and_benign_mixed_with_match(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        result = detect_stop_instruction(None, 'nothing to see here', None, 'hold for human review')
        assert result == 'hold for human', f'expected "hold for human", got {result!r}'

    def test_all_benign_or_none_returns_none(self):
        from orchestrator.stop_instruction import detect_stop_instruction
        result = detect_stop_instruction(None, 'benign one', '', 'benign two')
        assert result is None
