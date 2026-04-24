"""Unit tests for the pre-triage module (orchestrator.agents.triage)."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

from orchestrator.agents.triage import (
    _combine_suggestion_hashes,
    build_triage_prompt,
    format_pretriaged_detail,
    parse_triage_result,
    sha256_16,
    suggestion_hash,
)

# ---------------------------------------------------------------------------
# build_triage_prompt
# ---------------------------------------------------------------------------

class TestBuildTriagePrompt:
    def test_includes_all_suggestions_numbered(self):
        suggestions = [
            {'reviewer': 'test_analyst', 'location': 'a.py:1',
             'category': 'coverage', 'description': 'Missing test',
             'suggested_fix': 'Add test'},
            {'reviewer': 'style_cop', 'location': 'b.py:10',
             'category': 'style', 'description': 'Bad name',
             'suggested_fix': 'Rename'},
            {'reviewer': 'arch_auditor', 'location': 'c.py:5',
             'category': 'architecture', 'description': 'Duplication',
             'suggested_fix': 'Extract'},
        ]
        task = {'id': '42', 'title': 'Test Task', 'description': 'A test'}
        prompt = build_triage_prompt(suggestions, task)

        assert '[0]' in prompt
        assert '[1]' in prompt
        assert '[2]' in prompt
        assert 'Missing test' in prompt
        assert 'Bad name' in prompt
        assert 'Duplication' in prompt
        assert '3 items' in prompt

    def test_includes_task_context(self):
        task = {'id': '99', 'title': 'My Feature', 'description': 'Add foo'}
        prompt = build_triage_prompt([], task)
        assert 'Task 99' in prompt
        assert 'My Feature' in prompt

    def test_handles_missing_fields_gracefully(self):
        suggestions = [{'description': 'something'}]
        task = {}
        prompt = build_triage_prompt(suggestions, task)
        assert '[0]' in prompt
        assert 'something' in prompt


# ---------------------------------------------------------------------------
# parse_triage_result
# ---------------------------------------------------------------------------

class TestParseTriageResult:
    def test_valid_structured_output(self):
        result = MagicMock()
        result.structured_output = {
            'accepted': [{'index': 0, 'suggestion': 'x', 'reason': 'y',
                          'files': ['a.py'], 'proposed_task_title': 'Fix x'}],
            'skipped': [{'index': 1, 'suggestion': 'z', 'reason': 'n/a'}],
            'proposed_task_groups': [{'title': 'Fix x', 'description': 'do it',
                                      'accepted_indices': [0]}],
        }
        result.success = True
        parsed = parse_triage_result(result)
        assert parsed is not None
        assert len(parsed['accepted']) == 1
        assert len(parsed['skipped']) == 1

    def test_returns_none_on_missing_keys(self):
        result = MagicMock()
        result.structured_output = {'accepted': []}
        result.success = True
        assert parse_triage_result(result) is None

    def test_returns_none_on_no_structured_output(self):
        result = MagicMock()
        result.structured_output = None
        result.success = False
        assert parse_triage_result(result) is None

    def test_returns_none_on_non_dict(self):
        result = MagicMock()
        result.structured_output = 'not a dict'
        result.success = True
        assert parse_triage_result(result) is None


# ---------------------------------------------------------------------------
# format_pretriaged_detail
# ---------------------------------------------------------------------------

class TestFormatPretriagedDetail:
    def test_contains_header(self):
        triage_result = {
            'accepted': [{'index': 0, 'suggestion': 'x', 'reason': 'y',
                          'files': ['a.py'], 'proposed_task_title': 'Fix x'}],
            'skipped': [],
            'proposed_task_groups': [{'title': 'Fix x', 'description': 'do it',
                                      'accepted_indices': [0]}],
        }
        detail = format_pretriaged_detail(triage_result, [{'desc': 'original'}])
        assert detail.startswith('## Pre-Triaged Results')

    def test_includes_task_groups(self):
        triage_result = {
            'accepted': [
                {'index': 0, 'suggestion': 'a', 'reason': 'r',
                 'files': ['x.py'], 'proposed_task_title': 'Fix a'},
                {'index': 1, 'suggestion': 'b', 'reason': 'r',
                 'files': ['y.py'], 'proposed_task_title': 'Fix b'},
            ],
            'skipped': [],
            'proposed_task_groups': [
                {'title': 'Combined Fix', 'description': 'Fix a and b',
                 'accepted_indices': [0, 1]},
            ],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert 'Combined Fix' in detail
        assert 'x.py' in detail
        assert 'y.py' in detail

    def test_includes_skipped_items(self):
        triage_result = {
            'accepted': [],
            'skipped': [{'index': 0, 'suggestion': 'noise', 'reason': 'meritless'}],
            'proposed_task_groups': [],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert 'Skipped' in detail
        assert 'noise' in detail
        assert 'meritless' in detail

    def test_includes_original_suggestions_as_reference(self):
        originals = [{'description': 'test', 'location': 'foo.py:1'}]
        triage_result = {
            'accepted': [], 'skipped': [], 'proposed_task_groups': [],
        }
        detail = format_pretriaged_detail(triage_result, originals)
        assert 'Original Suggestions' in detail
        assert 'foo.py:1' in detail

    # ── R4: idempotency stamping ─────────────────────────────────────

    def test_escalation_id_embeds_stamps_and_instructions(self):
        triage_result = {
            'accepted': [
                {'index': 0, 'suggestion': 's0', 'reason': 'r',
                 'files': ['x.py'], 'proposed_task_title': 't0'},
            ],
            'skipped': [],
            'proposed_task_groups': [
                {'title': 'Fix 0', 'description': 'd',
                 'accepted_indices': [0]},
            ],
        }
        originals = [{
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix the thing',
        }]
        detail = format_pretriaged_detail(
            triage_result, originals, escalation_id='esc-1912-179',
        )
        assert 'Task Idempotency Stamps' in detail
        assert 'esc-1912-179' in detail
        # Per-group suggestion_hash rendered deterministically
        expected_hash = suggestion_hash(originals[0])
        assert expected_hash in detail
        # Steward-facing instruction present
        assert 'escalation_id' in detail
        assert 'suggestion_hash' in detail
        assert 'interceptor will' in detail
        # submit_task call must show the metadata= kwarg form
        assert 'submit_task' in detail
        assert 'metadata=' in detail
        # Two-step resolution: resolve_ticket and combined status must be named
        assert 'resolve_ticket' in detail, (
            'Pre-triaged block must name resolve_ticket so the steward '
            'knows to call it after submit_task'
        )
        assert 'combined' in detail, (
            "Pre-triaged block must name the 'combined' status returned by "
            'resolve_ticket on an R4 idempotency hit'
        )

    def test_escalation_id_absent_keeps_legacy_format(self):
        triage_result = {
            'accepted': [], 'skipped': [], 'proposed_task_groups': [],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert 'Task Idempotency Stamps' not in detail
        assert 'suggestion_hash' not in detail

    def test_summary_counts(self):
        triage_result = {
            'accepted': [
                {'index': i, 'suggestion': f's{i}', 'reason': 'r',
                 'files': [], 'proposed_task_title': f't{i}'}
                for i in range(3)
            ],
            'skipped': [
                {'index': i, 'suggestion': f'k{i}', 'reason': 'r'}
                for i in range(2)
            ],
            'proposed_task_groups': [
                {'title': 'g', 'description': 'd', 'accepted_indices': [0, 1, 2]},
            ],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert '3 accepted' in detail
        assert '2 skipped' in detail
        assert '1 task group' in detail


# ---------------------------------------------------------------------------
# R4: suggestion_hash determinism
# ---------------------------------------------------------------------------

class TestSuggestionHash:
    """R4 requires deterministic hashes so steward re-queues produce the
    same ``(escalation_id, suggestion_hash)`` tuple across retries.
    """

    def test_same_suggestion_same_hash(self):
        s = {
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix it',
        }
        assert suggestion_hash(s) == suggestion_hash(s)

    def test_differs_on_description_change(self):
        base = {
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix it',
        }
        variant = {**base, 'description': 'Something else'}
        assert suggestion_hash(base) != suggestion_hash(variant)

    def test_ignores_unrelated_fields(self):
        base = {
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix it',
            'suggested_fix': 'v1',
        }
        variant = {**base, 'suggested_fix': 'rephrased v2'}
        # suggested_fix is not part of the identity tuple.
        assert suggestion_hash(base) == suggestion_hash(variant)

    def test_hash_length_is_16(self):
        s = {'reviewer': 'r', 'location': 'l', 'category': 'c', 'description': 'd'}
        assert len(suggestion_hash(s)) == 16

    def test_combine_sorted_deterministically(self):
        a = _combine_suggestion_hashes(['bbb', 'aaa', 'ccc'])
        b = _combine_suggestion_hashes(['ccc', 'aaa', 'bbb'])
        assert a == b

    def test_combine_single_returns_self(self):
        assert _combine_suggestion_hashes(['abcd1234abcd1234']) == 'abcd1234abcd1234'


# ---------------------------------------------------------------------------
# sha256_16 — canonical 16-char sha256-hex helper
# ---------------------------------------------------------------------------

class TestSha256_16:
    """sha256_16 is the shared 16-char sha256 helper that both suggestion_hash
    and the escalation-watcher skill's cleanup_needed snippet depend on.
    """

    def test_length_is_16(self):
        assert len(sha256_16('anything')) == 16

    def test_deterministic(self):
        assert sha256_16('hello') == sha256_16('hello')

    def test_differs_across_inputs(self):
        assert sha256_16('hello') != sha256_16('world')

    def test_matches_raw_hashlib_reference(self):
        """Pins the exact construction so the skill's legacy snippet is byte-compatible."""
        assert sha256_16('test') == hashlib.sha256(b'test').hexdigest()[:16]

    def test_empty_string_is_fixed_prefix(self):
        """Documents the collision callers must avoid — blank detail must not be passed raw."""
        assert sha256_16('') == 'e3b0c44298fc1c14'
